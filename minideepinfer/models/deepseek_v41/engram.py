"""Engram n-gram hashing, bit-exact with the reference `inference/engram.py`.

Each position is hashed together with the `max_ngram_size - 1` tokens before it. Token ids are first
mapped into a compressed vocabulary (tokens that normalize alike collapse together); every
(n-gram size, head) pair then owns a disjoint prime-sized bucket range of the layer's table.

The engine supplies the raw n-gram window of every token (`[T, max_ngram_size]`, -1 where the
window reaches before the sequence start), so hashing needs no per-sequence history on the GPU.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
from torch import nn

from minideepinfer.models.deepseek_v41.config import V41Config


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@functools.lru_cache(maxsize=4)
def build_compressed_token_map(tokenizer_path: str) -> tuple[tuple[int, ...], int]:
    """Maps every token id onto a compressed id space; returns (lookup, compressed vocab size)."""
    from tokenizers import Regex, Tokenizer, normalizers

    sentinel = ""  # keeps a single-space token from collapsing to the empty string
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = Tokenizer.from_file(f"{tokenizer_path}/tokenizer.json")
    key_to_new: dict[str, int] = {}
    vocab_size = backend.get_vocab_size(with_added_tokens=True)
    lookup = [0] * vocab_size
    for token_id in range(vocab_size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:  # partial UTF-8 byte token: key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return tuple(lookup), len(key_to_new)


class EngramHasher(nn.Module):
    def __init__(self, cfg: V41Config, tokenizer_path: str) -> None:
        super().__init__()
        self.max_ngram_size = cfg.engram_max_ngram_size
        token_map, vocab_size = build_compressed_token_map(tokenizer_path)
        if vocab_size != cfg.engram_compressed_vocab_size:
            raise ValueError(f"compressed vocab size {vocab_size} != config {cfg.engram_compressed_vocab_size}")

        # Prime bucket sizes, drawn in order and never reused: [layer][n-gram size][head].
        primes, seen = [], set()
        for _ in cfg.engram_layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], cfg.engram_vocab_size - 1
                for _ in range(cfg.engram_n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(sizes)
            primes.append(per_ngram)
        flat = [[p for sizes in layer for p in sizes] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]

        # One odd multiplier per (layer, lookback), bounded so token * multiplier fits in int64.
        bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        multipliers = []
        for layer_id in cfg.engram_layer_ids:
            values = np.random.default_rng(10007 * layer_id).integers(0, bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers.append(torch.tensor(values * 2 + 1))

        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)
        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", torch.stack(multipliers), persistent=False)
        self.pad_id = token_map[cfg.engram_pad_id]

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        """ngram_ids: [T, max_ngram_size] raw ids (column s = token s positions back, -1 = none).

        Returns hash ids [T, n_engram_layers, (max_ngram_size - 1) * n_heads].
        """
        blocked = ngram_ids < 0
        tokens = torch.where(blocked, self.pad_id, self.token_map[ngram_ids.clamp_min(0)])
        products = tokens.unsqueeze(1) * self.multipliers  # [T, n_layers, max_ngram_size]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets
