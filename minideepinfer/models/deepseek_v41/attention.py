"""Attention over gathered keys, preserving the reference's BF16 probability rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attention(Q, KV, Valid, Sink, Scores, Out, heads, keys, dim,
                      SCALE: tl.constexpr, WIDTH: tl.constexpr, HEADS: tl.constexpr, TILE: tl.constexpr):
    """One query row and a block of HEADS heads, which share the row's keys (multi-query attention).

    The heads form the M dimension of tensor-core products, so each key tile is loaded once per
    block rather than once per head. A first pass writes the scores and finds the global key maximum;
    the second rounds the unnormalized probabilities to BF16 against that maximum, as the reference does.
    """
    row, block = tl.program_id(0), tl.program_id(1)
    h = block * HEADS + tl.arange(0, HEADS)
    d, k = tl.arange(0, WIDTH), tl.arange(0, TILE)
    head_ok, cols = h < heads, d < dim
    q = tl.load(Q + (row * heads + h[:, None]) * dim + d[None, :], head_ok[:, None] & cols[None, :], other=0.0)
    scores = Scores + (row * heads + h[:, None]) * keys
    maximum = tl.full((HEADS,), -1e30, tl.float32)
    for start in range(0, keys, TILE):
        ks = start + k
        inside = ks < keys
        valid = tl.load(Valid + row * keys + ks, inside, other=False)
        kv = tl.load(KV + (row * keys + ks[:, None]) * dim + d[None, :], inside[:, None] & cols[None, :], other=0.0)
        s = tl.where(valid[None, :], tl.dot(q, tl.trans(kv)) * SCALE, -float("inf"))
        maximum = tl.maximum(maximum, tl.max(s, 1))
        tl.store(scores + ks[None, :], s, head_ok[:, None] & inside[None, :])
    tl.debug_barrier()  # the second pass reads scores written by other threads of this program

    denominator = tl.exp(tl.load(Sink + h, head_ok, other=0.0) - maximum)
    acc = tl.zeros((HEADS, WIDTH), tl.float32)
    for start in range(0, keys, TILE):
        ks = start + k
        inside = ks < keys
        s = tl.load(scores + ks[None, :], head_ok[:, None] & inside[None, :], other=-float("inf"))
        p = tl.exp(s - maximum[:, None])
        denominator += tl.sum(p, 1)
        kv = tl.load(KV + (row * keys + ks[:, None]) * dim + d[None, :], inside[:, None] & cols[None, :], other=0.0)
        acc = tl.dot(p.to(tl.bfloat16), kv, acc)
    out = Out + (row * heads + h[:, None]) * dim + d[None, :]
    tl.store(out, tl.div_rn(acc, denominator[:, None]), head_ok[:, None] & cols[None, :])


def sparse_attention_cuda(q: torch.Tensor, kv: torch.Tensor, valid: torch.Tensor,
                          sink: torch.Tensor, scale: float) -> torch.Tensor:
    rows, heads, dim = q.shape
    keys = kv.shape[1]
    out = torch.empty_like(q)
    if rows:
        block = 16 if heads <= 16 else 32
        scores = torch.empty(rows, heads, keys, dtype=torch.float32, device=q.device)
        _sparse_attention[(rows, triton.cdiv(heads, block))](
            q, kv, valid, sink, scores, out, heads, keys, dim,
            SCALE=scale, WIDTH=max(16, triton.next_power_of_2(dim)), HEADS=block, TILE=32,
            num_warps=4, num_stages=3, enable_fp_fusion=False,
        )
    return out
