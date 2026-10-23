"""Combining routed-expert outputs back into token order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _combine(Y, Pos, Local, Out, dim, K: tl.constexpr, BD: tl.constexpr):
    """Out[t] = sum over slots s, in slot order, of Y[Pos[t, s]] where the slot's expert is local."""
    t = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    ok = d < dim
    acc = tl.zeros((BD,), tl.float32)
    for s in tl.static_range(K):
        local = tl.load(Local + t * K + s)
        row = tl.load(Pos + t * K + s)
        acc += tl.load(Y + row * dim + d, ok & local, other=0.0).to(tl.float32)
    tl.store(Out + t * dim + d, acc, ok)


def combine_cuda(y: torch.Tensor, order: torch.Tensor, is_local: torch.Tensor, k: int) -> torch.Tensor:
    """y [T * k, dim] in expert-sorted order (`order[i]` = flat choice of row i) -> [T, dim] fp32.

    Only rows of local experts are read, so the GEMMs need not zero the others.
    """
    pos = torch.empty_like(order)
    pos[order] = torch.arange(order.numel(), device=order.device)
    rows, dim = order.numel() // k, y.size(1)
    out = torch.empty(rows, dim, dtype=torch.float32, device=y.device)
    if rows:
        bd = 1024
        _combine[(rows, triton.cdiv(dim, bd))](y, pos, is_local, out, dim, K=k, BD=bd, num_warps=4)
    return out
