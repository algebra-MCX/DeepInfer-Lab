"""LLM engine: drives the schedule -> execute -> post-process loop.

With zero-overhead scheduling (ZOS) enabled, one call to `step()` does:

    1. schedule step N             (CPU; step N-1 may still be running on the GPU)
    2. enqueue step N on the GPU   (asynchronous; may read step N-1's samples on the GPU)
    3. post-process step N-1       (CPU; overlaps with step N on the GPU)

so the GPU never idles waiting for Python-side bookkeeping. The price is that a sequence that
emitted EOS in step N-1 has already been scheduled for step N; that extra token is discarded.
With ZOS disabled, step N is post-processed synchronously at the end of its own `step()`.
"""

from __future__ import annotations

import copy
import logging
import random
import time

from minideepinfer.config import Config
from minideepinfer.engine.block_manager import BlockManager
from minideepinfer.engine.scheduler import Scheduler
from minideepinfer.sampling_params import SamplingParams
from minideepinfer.sequence import Sequence
from minideepinfer.tokenizer import load_tokenizer
from minideepinfer.worker.model_runner import ModelRunner, StepHandle

logger = logging.getLogger(__name__)


class StatsLogger:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.last_log = time.perf_counter()
        self.prefill_tokens = 0
        self.decode_tokens = 0

    def record(self, is_prompt: bool, num_tokens: int, engine: LLMEngine) -> None:
        if is_prompt:
            self.prefill_tokens += num_tokens
        else:
            self.decode_tokens += num_tokens
        now = time.perf_counter()
        if self.interval <= 0 or now - self.last_log < self.interval:
            return
        elapsed = now - self.last_log
        sched = engine.scheduler
        logger.info(
            "prefill %.0f tok/s, decode %.0f tok/s, running %d, waiting %d, KV usage %.1f%%, preempted %d",
            self.prefill_tokens / elapsed,
            self.decode_tokens / elapsed,
            len(sched.running),
            len(sched.waiting),
            100 * engine.block_manager.usage(),
            sched.num_preempted,
        )
        self.last_log, self.prefill_tokens, self.decode_tokens = now, 0, 0


class LLMEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.tokenizer = load_tokenizer(config.model_path)
        self.model_runner = ModelRunner(config)
        self.block_manager = BlockManager(self.model_runner.num_blocks)
        self.state_manager = BlockManager(self.model_runner.num_slots)
        self.scheduler = Scheduler(
            self.block_manager,
            self.state_manager,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_model_len=config.max_model_len,
            chunked_prefill=config.chunked_prefill,
            prefill_attempt_interval=config.prefill_attempt_interval,
            eos_token_id=config.eos_token_id,
        )
        self.rng = random.Random(config.seed)
        self.inflight: StepHandle | None = None
        # Sequences that finished while an abort drained the in-flight step; returned by the next step().
        self.pending_finished: list[Sequence] = []
        self.stats = StatsLogger(config.log_stats_interval)
        self._requests: dict[str, Sequence] = {}

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        request_id: str | int | None = None,
    ) -> Sequence:
        sampling_params = copy.copy(sampling_params)
        sampling_params.assign_seed(self.rng)
        seq = Sequence(prompt_token_ids, sampling_params, self.config.block_size, request_id=request_id)
        previous = self._requests.get(seq.request_id)
        if previous is not None and not previous.is_finished:
            raise ValueError(f"request_id {seq.request_id!r} is already active")
        self._requests[seq.request_id] = seq
        self.scheduler.add_sequence(seq)
        return seq

    def abort_request(self, request_id: str | int) -> Sequence | None:
        """Cancel a request after draining GPU work that can reference its cache."""
        if self.inflight is not None:
            self.pending_finished += self._process(self.inflight)
            self.inflight = None
        seq = self.scheduler.abort_request(request_id)
        if seq is not None:
            return seq
        previous = self._requests.get(str(request_id))
        return previous if previous is not None and previous.is_finished else None

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_seqs() or self.inflight is not None or bool(self.pending_finished)

    def step(self) -> list[Sequence]:
        """Run one engine iteration and return the sequences that finished during it."""
        finished, self.pending_finished = self.pending_finished, []
        sched = self.scheduler.schedule()
        finished += sched.ignored_seqs
        handle = None if sched.is_empty else self.model_runner.execute(sched, self.inflight)

        # Post-process the previous step while the current one runs on the GPU.
        if self.inflight is not None:
            finished += self._process(self.inflight)
            self.inflight = None
        elif handle is None and not finished and self.scheduler.has_unfinished_seqs():
            raise RuntimeError("scheduler made no progress: no step scheduled and nothing in flight")

        if handle is not None:
            if self.config.enable_zos:
                self.inflight = handle
            else:
                finished += self._process(handle)
        return finished

    def _process(self, handle: StepHandle) -> list[Sequence]:
        token_ids = handle.wait()
        self.stats.record(handle.sched.is_prompt, handle.sched.num_batched_tokens, self)
        return self.scheduler.process_outputs(handle.sched.sample_seqs, token_ids)
