from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Sampling parameters for text generation.

    Args:
        temperature: Softmax temperature. Zero means greedy decoding.
        top_p: Nucleus sampling threshold in (0, 1]. 1 disables it.
        top_k: Number of top tokens to keep. -1 disables it.
        max_tokens: Maximum number of tokens to generate.
        ignore_eos: Keep generating after an EOS token.
        stop_token_ids: Additional token IDs that finish the request with ``stop``.
        seq_seed: Per-sequence random seed. The sampling noise is a pure function of
            (seq_seed, token position), independent of batch composition and scheduling order,
            so identical inputs with an identical seed always yield identical outputs
            (DeepInfer's deterministic sampling). If None, the engine assigns one from its
            global seed.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    ignore_eos: bool = False
    seq_seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k must be -1 (disable) or >= 1, got {self.top_k}.")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}.")
        self.stop_token_ids = tuple(dict.fromkeys(int(token_id) for token_id in self.stop_token_ids))
        if self.seq_seed is not None:
            self.seq_seed &= (1 << 63) - 1

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5

    def assign_seed(self, rng: random.Random) -> None:
        if self.seq_seed is None:
            self.seq_seed = rng.getrandbits(63)
