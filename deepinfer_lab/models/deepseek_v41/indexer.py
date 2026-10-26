"""Fused indexer scoring for prefill: sum_h weights[t, h] * relu(q[t, h] . keys[seq(t), n]) without [T, H, n] scratch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _index_scores(Q, K, W, Seq, Out, rows, n, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """A [BM rows, BN entries] tile. Each sequence's key tile is loaded once and reused by every head;
    a block whose rows span several sequences scores them one sequence at a time."""
    start = tl.program_id(0) * BM
    m = start + tl.arange(0, BM)
    j = tl.program_id(1) * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    row_ok, col_ok = m < rows, j < n
    seq = tl.load(Seq + m, row_ok, other=-1)
    first, last = tl.load(Seq + start), tl.load(Seq + tl.minimum(start + BM, rows) - 1)
    acc = tl.zeros((BM, BN), tl.float32)
    if first == last:  # the common prefill case: the whole block belongs to one sequence
        keys = tl.load(K + (first * n + j[None, :]) * D + d[:, None], col_ok[None, :], other=0.0)  # [D, BN]
        for h in range(H):
            q = tl.load(Q + (m[:, None] * H + h) * D + d[None, :], row_ok[:, None], other=0.0)
            w = tl.load(W + m * H + h, row_ok, other=0.0).to(tl.float32)
            acc += tl.maximum(tl.dot(q, keys), 0.0) * w[:, None]
    else:
        for s in range(first, last + 1):
            mine = seq == s  # other rows load zero queries and weights, so they add nothing
            keys = tl.load(K + (s * n + j[None, :]) * D + d[:, None], col_ok[None, :], other=0.0)
            for h in range(H):
                q = tl.load(Q + (m[:, None] * H + h) * D + d[None, :], mine[:, None], other=0.0)
                w = tl.load(W + m * H + h, mine, other=0.0).to(tl.float32)
                acc += tl.maximum(tl.dot(q, keys), 0.0) * w[:, None]
    tl.store(Out + m[:, None] * n + j[None, :], acc.to(Out.dtype.element_ty), row_ok[:, None] & col_ok[None, :])


def index_scores_cuda(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, token_seq: torch.Tensor) -> torch.Tensor:
    """q [T, H, D], keys [B, n, D], weights [T, H], token_seq [T] (rows of a sequence contiguous) -> [T, n]."""
    rows, heads, dim = q.shape
    n = keys.shape[1]
    out = torch.empty(rows, n, dtype=q.dtype, device=q.device)
    if rows and n:
        bm, bn = 64, 128
        _index_scores[(triton.cdiv(rows, bm), triton.cdiv(n, bn))](
            q, keys, weights, token_seq, out, rows, n, H=heads, D=dim, BM=bm, BN=bn, num_warps=4, num_stages=2,
        )
    return out
