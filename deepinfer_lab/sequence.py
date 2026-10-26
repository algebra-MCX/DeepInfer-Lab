from __future__ import annotations

import enum
import itertools
import time

from deepinfer_lab.sampling_params import SamplingParams


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED_STOPPED = enum.auto()  # Generated EOS or a configured stop token.
    FINISHED_LENGTH = enum.auto()  # Hit max_tokens or max_model_len.
    FINISHED_IGNORED = enum.auto()  # Prompt can never be scheduled (too long).
    FINISHED_ABORTED = enum.auto()  # Explicitly cancelled by the caller.

    @property
    def is_finished(self) -> bool:
        return self not in (SequenceStatus.WAITING, SequenceStatus.RUNNING)


class Sequence:
    """A single generation request.

    Length bookkeeping. Under zero-overhead scheduling the CPU view lags the GPU by one step:

        token_ids            Tokens known on the CPU (prompt + outputs already returned).
        num_inflight_tokens  Tokens scheduled and sampled on the GPU but not yet returned (0 or 1).
        get_len()            len(token_ids) + num_inflight_tokens, the true length as seen by the GPU.
        num_cached_tokens    Tokens whose KV has been (or is being) written to the cache. Advanced
                             at schedule time, without waiting for results.

    A decode step feeds the token at position get_len() - 1. If that token is in flight it only
    exists on the GPU (the previous step's sample); otherwise it is token_ids[-1].
    """

    _counter = itertools.count()

    def __init__(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        block_size: int,
        request_id: str | None = None,
    ) -> None:
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        self.seq_id = next(Sequence._counter)
        self.request_id = str(self.seq_id) if request_id is None else str(request_id)
        self.token_ids = list(prompt_token_ids)
        self.prompt_len = len(prompt_token_ids)
        self.sampling_params = sampling_params
        self.block_size = block_size

        self._status = SequenceStatus.WAITING
        self.block_table: list[int] = []
        self.state_slot = 0  # Per-sequence recurrent state (sliding window, compressor tail); 0 = none.
        self.num_cached_tokens = 0
        self.num_inflight_tokens = 0
        # Incremented on every preemption. Results of steps launched before the preemption
        # carry a stale epoch and are discarded when they come back.
        self.epoch = 0

        # Wall-clock timestamps (time.perf_counter) for latency metrics, see `deepinfer_lab.metrics`.
        self.arrival_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.finish_time: float | None = None

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.request_id}, status={self.status.name}, len={self.get_len()}, "
            f"cached={self.num_cached_tokens}, inflight={self.num_inflight_tokens}, blocks={len(self.block_table)}, "
            f"slot={self.state_slot})"
        )

    @property
    def status(self) -> SequenceStatus:
        return self._status

    @status.setter
    def status(self, status: SequenceStatus) -> None:
        if status.is_finished and self.finish_time is None:
            self.finish_time = time.perf_counter()
        self._status = status

    def get_len(self) -> int:
        return len(self.token_ids) + self.num_inflight_tokens

    def get_output_len(self) -> int:
        return self.get_len() - self.prompt_len

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished

    def num_blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def append_token(self, token_id: int) -> None:
        assert self.num_inflight_tokens > 0
        self.num_inflight_tokens -= 1
        self.token_ids.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
