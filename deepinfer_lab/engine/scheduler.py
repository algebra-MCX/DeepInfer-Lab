"""Request scheduler, mirroring the NORMAL engine of DeepInfer's csrc/engine/scheduler.hpp.

Each step is either a pure prefill step or a pure decode step; the two are never mixed.

- Prefill: admit requests from the head of the waiting queue in FCFS order, subject to
  max_num_seqs, max_num_batched_tokens, KV block and state slot availability. A request that does
  not fit the token budget is chunked; at most one chunked request is scheduled per step.
- Decode: every running request advances by one token. When KV blocks run out, requests are
  preempted following DeepInfer's policy: requests with shorter outputs are served first and
  victims are taken from the longest outputs. Preempted requests release all their blocks and
  return to the head of the waiting queue to be recomputed.
- Interleaving: a prefill is attempted only every `prefill_attempt_interval` decode steps, and
  no new requests are admitted while KV usage is above the watermark.

Zero-overhead scheduling: decisions depend only on CPU-side reservations (num_cached_tokens and
num_inflight_tokens), never on the previous step's sampled tokens, so scheduling step N+1 can
overlap with step N running on the GPU.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from deepinfer_lab.engine.block_manager import AllocStatus, BlockManager
from deepinfer_lab.sequence import Sequence, SequenceStatus

logger = logging.getLogger(__name__)


@dataclass
class SchedulerOutput:
    is_prompt: bool
    seqs: list[Sequence] = field(default_factory=list)
    seqlens_q: list[int] = field(default_factory=list)
    # Sequences that sample a token this step, with their epoch at schedule time. In a prefill
    # step only sequences whose final prompt chunk is scheduled sample.
    sample_seqs: list[tuple[Sequence, int]] = field(default_factory=list)
    ignored_seqs: list[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.seqs

    @property
    def num_batched_tokens(self) -> int:
        return sum(self.seqlens_q)


class Scheduler:
    GPU_CACHE_PREFILL_WATERMARK = 0.9

    def __init__(
        self,
        block_manager: BlockManager,
        state_manager: BlockManager,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        chunked_prefill: bool = True,
        prefill_attempt_interval: int = 1,
        eos_token_id: int = -1,
    ) -> None:
        self.block_manager = block_manager
        self.state_manager = state_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.chunked_prefill = chunked_prefill
        self.prefill_attempt_interval = prefill_attempt_interval
        self.eos_token_id = eos_token_id
        self.prompt_limit = max_model_len if chunked_prefill else min(max_model_len, max_num_batched_tokens)

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.decode_step = 0
        self.num_preempted = 0

    # Request lifecycle.

    def add_sequence(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:
        return bool(self.waiting or self.running)

    def snapshot(self) -> dict[str, int | float]:
        """Return CPU-side scheduler state without touching device results."""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "unfinished": len(self.waiting) + len(self.running),
            "preempted": self.num_preempted,
            "kv_usage": self.block_manager.usage(),
            "state_usage": self.state_manager.usage(),
        }

    def _finish(self, seq: Sequence, status: SequenceStatus) -> None:
        seq.status = status
        self._release(seq)
        self.running.remove(seq)

    def _preempt(self, seq: Sequence) -> None:
        """Preempt by recomputation: release all KV blocks and requeue at the head of `waiting`.

        Any step already in flight for this sequence becomes stale (epoch + 1). Thanks to
        deterministic sampling, recomputation reproduces the discarded token exactly.
        A partially prefilled request at the head keeps its place: it holds blocks that the
        preempted sequence may need, so queueing ahead of it could deadlock.
        """
        self._release(seq)
        seq.num_cached_tokens = 0
        seq.num_inflight_tokens = 0
        seq.epoch += 1
        seq.status = SequenceStatus.WAITING
        self.waiting.insert(1 if self.waiting and self.waiting[0].block_table else 0, seq)
        self.num_preempted += 1

    def _reached_limit(self, seq: Sequence) -> bool:
        """True if, counting in-flight tokens, the sequence already hit max_tokens or max_model_len.

        Such a sequence needs no further steps; it finishes once its last result comes back.
        """
        return seq.get_output_len() >= seq.sampling_params.max_tokens or seq.get_len() >= self.max_model_len

    def _release(self, seq: Sequence) -> None:
        """Returns a sequence's KV blocks and state slot (DeepInfer frees both under one lock)."""
        self.block_manager.free(seq.block_table)
        seq.block_table = []
        if seq.state_slot:
            self.state_manager.free([seq.state_slot])
            seq.state_slot = 0

    def abort_request(self, request_id: str | int) -> Sequence | None:
        """Cancel one request after its outstanding device step has been drained.

        ``LLMEngine.abort_request`` drains the in-flight ``StepHandle`` before calling this
        method. Keeping the scheduler operation CPU-only prevents a freed block from being
        reused while a previous GPU step can still read it.
        """
        request_id = str(request_id)
        seq = next((item for item in self.waiting if item.request_id == request_id), None)
        if seq is not None:
            self.waiting.remove(seq)
        else:
            seq = next((item for item in self.running if item.request_id == request_id), None)
            if seq is not None:
                self.running.remove(seq)
        if seq is None:
            return None
        seq.epoch += 1
        seq.status = SequenceStatus.FINISHED_ABORTED
        seq.num_inflight_tokens = 0
        self._release(seq)
        return seq

    # Scheduling.

    def schedule(self) -> SchedulerOutput:
        has_decode_work = any(not self._reached_limit(seq) for seq in self.running)
        try_prefill = bool(self.waiting) and (
            not has_decode_work
            or (
                self.decode_step % self.prefill_attempt_interval == 0
                and self.block_manager.usage() <= self.GPU_CACHE_PREFILL_WATERMARK
            )
        )
        if try_prefill:
            output = self._schedule_prefill()
            if not output.is_empty or output.ignored_seqs:
                self.decode_step = 0
                return output
        output = self._schedule_decode()
        if not output.is_empty:
            self.decode_step += 1
        return output

    def _schedule_prefill(self) -> SchedulerOutput:
        output = SchedulerOutput(is_prompt=True)
        num_seqs = len(self.running)
        num_batched_tokens = 0
        bm = self.block_manager

        while self.waiting:
            seq = self.waiting[0]
            if not seq.block_table:
                # First chunk: validate the length and make sure blocks for the whole prompt
                # can be allocated up front.
                if seq.get_len() > self.prompt_limit:
                    logger.warning("prompt (%d tokens) exceeds limit %d, ignored", seq.get_len(), self.prompt_limit)
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    output.ignored_seqs.append(self.waiting.popleft())
                    continue
                status = bm.can_allocate(seq.num_blocks_for(seq.get_len()))
                if status == AllocStatus.LATER or self.state_manager.num_free_blocks == 0:
                    break
                if status == AllocStatus.NEVER:
                    logger.warning("prompt (%d tokens) exceeds KV cache capacity, ignored", seq.get_len())
                    seq.status = SequenceStatus.FINISHED_IGNORED
                    output.ignored_seqs.append(self.waiting.popleft())
                    continue
            if num_seqs + 1 > self.max_num_seqs:
                break

            num_remaining = seq.get_len() - seq.num_cached_tokens
            seqlen_q = min(num_remaining, self.max_num_batched_tokens - num_batched_tokens)
            if seqlen_q <= 0:
                break
            is_chunked = seqlen_q < num_remaining
            if is_chunked and not self.chunked_prefill:
                break

            # The sequence is scheduled; commit all side effects below.
            if not seq.block_table:
                seq.block_table = bm.allocate(seq.num_blocks_for(seq.get_len()))
                seq.state_slot = self.state_manager.allocate(1)[0]
            seq.num_cached_tokens += seqlen_q
            num_batched_tokens += seqlen_q
            output.seqs.append(seq)
            output.seqlens_q.append(seqlen_q)
            if is_chunked:
                break  # Stays at the head of `waiting`; the next prefill step continues it.
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            seq.num_inflight_tokens += 1
            output.sample_seqs.append((seq, seq.epoch))
            self.running.append(seq)
            num_seqs += 1
        return output

    def _schedule_decode(self) -> SchedulerOutput:
        output = SchedulerOutput(is_prompt=False)
        bm = self.block_manager
        idle = [seq for seq in self.running if self._reached_limit(seq)]
        candidates = [seq for seq in self.running if not self._reached_limit(seq)]
        # Longest outputs first. Schedule from the tail (shortest outputs); when blocks run out,
        # preempt from the head (longest outputs).
        candidates.sort(key=lambda s: s.get_output_len(), reverse=True)

        scheduled: list[Sequence] = []
        l, r = 0, len(candidates)
        while l < r and len(scheduled) < self.max_num_seqs:
            seq = candidates[r - 1]
            need = seq.num_blocks_for(seq.get_len()) - len(seq.block_table)
            while bm.num_free_blocks < need and r - l > 1:
                self._preempt(candidates[l])
                l += 1
            if bm.num_free_blocks >= need:
                seq.block_table += bm.allocate(need)
                scheduled.append(seq)
            else:
                self._preempt(seq)
            r -= 1
        self.running = scheduled + candidates[l:r] + idle

        scheduled.sort(key=lambda s: s.get_len(), reverse=True)
        for seq in scheduled:
            # Feed the token at position get_len() - 1, write its KV, and reserve one new token.
            assert seq.num_cached_tokens == seq.get_len() - 1, seq
            seq.num_cached_tokens += 1
            seq.num_inflight_tokens += 1
            output.seqs.append(seq)
            output.seqlens_q.append(1)
            output.sample_seqs.append((seq, seq.epoch))
        return output

    # Output processing.

    def process_outputs(self, sample_seqs: list[tuple[Sequence, int]], token_ids: list[int]) -> list[Sequence]:
        """Commit one step's sampled tokens and return the sequences that finished.

        Under zero-overhead scheduling the results may arrive after the next step has already
        been scheduled; results for sequences that have finished or been preempted since are dropped.
        """
        finished = []
        for (seq, epoch), token_id in zip(sample_seqs, token_ids, strict=True):
            if seq.is_finished or seq.epoch != epoch:
                continue
            seq.append_token(token_id)
            sp = seq.sampling_params
            if (not sp.ignore_eos and token_id == self.eos_token_id) or token_id in sp.stop_token_ids:
                self._finish(seq, SequenceStatus.FINISHED_STOPPED)
            elif len(seq.token_ids) - seq.prompt_len >= sp.max_tokens or len(seq.token_ids) >= self.max_model_len:
                self._finish(seq, SequenceStatus.FINISHED_LENGTH)
            else:
                continue
            finished.append(seq)
        return finished
