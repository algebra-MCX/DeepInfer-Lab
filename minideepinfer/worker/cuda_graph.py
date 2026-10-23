"""CUDA graphs for decode steps, after DeepInfer's `worker/cuda_graph_runner.py`.

A decode step launches thousands of small kernels; replaying them as one captured graph removes the
per-kernel launch cost. A graph fixes every tensor shape, so graphs are captured for a grid of
(batch size, context bucket) and a step runs the smallest graph that covers it:

    batch size      the step's sequences, padded with rows that point at the reserved state slot 0
                    and block 0 (their writes land in storage no sequence reads, their outputs are
                    dropped)
    context bucket  an upper bound of the step's longest context; it sizes the indexer's scan over
                    compressed entries, whose entries past each sequence's length are masked anyway

The graphs read their inputs from static buffers that every replay overwrites, padding rows included:
a row left over from a previous, larger step would otherwise write into a live sequence's state.
"""

from __future__ import annotations

import logging
from bisect import bisect_left

import torch

from minideepinfer.models.deepseek_v41 import DeepseekV41
from minideepinfer.models.deepseek_v41.model import PAGE_TOKENS, V41Batch

logger = logging.getLogger(__name__)


def build_graph_batch_sizes(max_size: int, interval: int = 8) -> list[int]:
    """Powers of two below `interval`, then multiples of `interval`, then `max_size` (as DeepInfer)."""
    interval = min(interval, max_size)
    sizes, size = [], 1
    while size < interval:
        sizes.append(size)
        size *= 2
    sizes.extend(range(interval, max_size, interval))
    sizes.append(max_size)
    return sizes


def build_context_buckets(max_model_len: int) -> list[int]:
    """Powers of two from two blocks up, capped at max_model_len."""
    buckets, size = [], 2 * PAGE_TOKENS
    while size < max_model_len:
        buckets.append(size)
        size *= 2
    return [*buckets, max_model_len]


class DecodeGraphRunner:
    def __init__(self, model: DeepseekV41, max_batch_size: int, max_blocks: int, max_model_len: int) -> None:
        self.model = model
        self.batch_sizes = build_graph_batch_sizes(max_batch_size)
        self.context_buckets = build_context_buckets(max_model_len)
        n, dev = max_batch_size, "cuda"
        max_ngram = model.cfg.engram_max_ngram_size if model.hasher is not None else 0
        # Inputs of padding rows; also the static buffers' initial contents.
        self.padding = {
            "input_ids": torch.zeros(n, dtype=torch.int64, device=dev),
            "positions": torch.zeros(n, dtype=torch.int64, device=dev),
            "token_seq": torch.arange(n, device=dev),
            "slots": torch.zeros(n, dtype=torch.int64, device=dev),
            "chunk_start": torch.zeros(n, dtype=torch.int64, device=dev),
            "chunk_end": torch.ones(n, dtype=torch.int64, device=dev),
            "block_tables": torch.zeros(n, max_blocks, dtype=torch.int64, device=dev),
        }
        if max_ngram:
            ngram = torch.full((n, max_ngram), -1, dtype=torch.int64, device=dev)
            ngram[:, 0] = 0
            self.padding["ngram_ids"] = ngram
        self.static = {name: t.clone() for name, t in self.padding.items()}
        self.graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def _batch(self, batch_size: int, context: int) -> V41Batch:
        views = {"ngram_ids": None} | {name: t[:batch_size] for name, t in self.static.items()}
        return V41Batch(**views, cu_seqlens=list(range(batch_size + 1)), max_end=context)

    @torch.inference_mode()
    def capture(self) -> None:
        """Captures every (batch size, context) graph, largest first, sharing one memory pool."""
        pool, stream = None, torch.cuda.Stream()
        for context in reversed(self.context_buckets):
            for batch_size in reversed(self.batch_sizes):
                batch = self._batch(batch_size, context)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    self.model(batch)  # warm-up: one-time allocations and lazy initialization
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    hidden = self.model(batch)
                torch.cuda.current_stream().wait_stream(stream)
                pool = graph.pool()
                self.graphs[batch_size, context] = graph, hidden
        logger.info(
            "captured %d decode CUDA graphs: batch sizes %s, contexts %s",
            len(self.graphs),
            self.batch_sizes,
            self.context_buckets,
        )

    def can_run(self, batch: V41Batch) -> bool:
        return batch.input_ids.numel() == len(batch.cu_seqlens) - 1 <= self.batch_sizes[-1]

    def run(self, batch: V41Batch) -> torch.Tensor:
        """Replays the smallest graph covering a decode batch; returns hidden states [B, dim]."""
        n = batch.input_ids.numel()
        batch_size = self.batch_sizes[bisect_left(self.batch_sizes, n)]
        context = self.context_buckets[bisect_left(self.context_buckets, batch.max_end)]
        for name, static in self.static.items():
            value = getattr(batch, name)
            static[n:batch_size].copy_(self.padding[name][n:batch_size])
            if name == "block_tables":
                static[:n, value.size(1) :].zero_()
                static[:n, : value.size(1)].copy_(value)
            else:
                static[:n].copy_(value)
        graph, hidden = self.graphs[batch_size, context]
        graph.replay()
        return hidden[:n]
