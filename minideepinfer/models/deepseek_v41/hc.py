"""Fused hyper-connection mix splitting and Sinkhorn normalization, after DeepInfer's mHC kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_sinkhorn(Mixes, Scale, Base, Pre, Post, Comb,
                    HC: tl.constexpr, WIDTH: tl.constexpr, ITERS: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    channels = tl.arange(0, WIDTH)
    start = row * HC * (HC + 2)
    pre = (tl.load(Mixes + start + channels, channels < HC, other=0.0) * tl.load(Scale)
           + tl.load(Base + channels, channels < HC, other=0.0))
    post = (tl.load(Mixes + start + HC + channels, channels < HC, other=0.0) * tl.load(Scale + 1)
            + tl.load(Base + HC + channels, channels < HC, other=0.0))
    tl.store(Pre + row * HC + channels, tl.sigmoid(pre) + EPS, channels < HC)
    tl.store(Post + row * HC + channels, 2 * tl.sigmoid(post), channels < HC)

    index = channels[:, None] * HC + channels[None, :]
    valid = (channels[:, None] < HC) & (channels[None, :] < HC)
    logits = (tl.load(Mixes + start + 2 * HC + index, valid, other=0.0) * tl.load(Scale + 2)
              + tl.load(Base + 2 * HC + index, valid, other=0.0))
    logits = tl.where(valid, logits, -float("inf"))
    exp = tl.where(valid, tl.exp(logits - tl.max(logits, 1)[:, None]), 0.0)
    comb = tl.where(valid, tl.div_rn(exp, tl.sum(exp, 1)[:, None]) + EPS, 0.0)
    comb = tl.div_rn(comb, tl.sum(comb, 0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        comb = tl.div_rn(comb, tl.sum(comb, 1)[:, None] + EPS)
        comb = tl.div_rn(comb, tl.sum(comb, 0)[None, :] + EPS)
    tl.store(Comb + row * HC * HC + index, comb, valid)


def split_sinkhorn_cuda(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One program per token keeps the tiny [hc, hc] matrix in registers for every iteration."""
    rows = mixes.shape[0]
    pre = mixes.new_empty(rows, hc)
    post = torch.empty_like(pre)
    comb = mixes.new_empty(rows, hc, hc)
    if rows:
        _split_sinkhorn[(rows,)](
            mixes, scale, base, pre, post, comb,
            HC=hc, WIDTH=triton.next_power_of_2(hc), ITERS=iters, EPS=eps,
            num_warps=1, enable_fp_fusion=False,
        )
    return pre, post, comb


@triton.jit
def _collapse(X, Pre, Out, dim, HC: tl.constexpr, CHANNELS: tl.constexpr, WIDTH: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    d = block * WIDTH + tl.arange(0, WIDTH)
    c = tl.arange(0, CHANNELS)
    x = tl.load(X + (row * HC + c[:, None]) * dim + d[None, :],
                (c[:, None] < HC) & (d[None, :] < dim), other=0.0).to(tl.float32)
    pre = tl.load(Pre + row * HC + c, c < HC, other=0.0)
    tl.store(Out + row * dim + d, tl.sum(x * pre[:, None], 0), d < dim)


@triton.jit
def _expand(X, Residual, Post, Comb, Out, dim,
            HC: tl.constexpr, CHANNELS: tl.constexpr, WIDTH: tl.constexpr):
    row, dst, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = block * WIDTH + tl.arange(0, WIDTH)
    src = tl.arange(0, CHANNELS)
    x = tl.load(X + row * dim + d, d < dim, other=0.0).to(tl.float32)
    residual = tl.load(Residual + (row * HC + src[:, None]) * dim + d[None, :],
                       (src[:, None] < HC) & (d[None, :] < dim), other=0.0).to(tl.float32)
    # Comb is [source, destination]; reduce the source channels, not the destination channels.
    comb = tl.load(Comb + (row * HC + src) * HC + dst, src < HC, other=0.0)
    post = tl.load(Post + row * HC + dst)
    y = post * x + tl.sum(residual * comb[:, None], 0)
    tl.store(Out + (row * HC + dst) * dim + d, y, d < dim)


def collapse_cuda(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    rows, hc, dim = x.shape
    out = x.new_empty(rows, dim)
    if rows:
        _collapse[(rows, triton.cdiv(dim, 128))](
            x, pre, out, dim, HC=hc, CHANNELS=triton.next_power_of_2(hc), WIDTH=128,
            num_warps=4, enable_fp_fusion=False,
        )
    return out


def expand_cuda(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    rows, hc, dim = residual.shape
    out = torch.empty_like(residual)
    if rows:
        _expand[(rows, hc, triton.cdiv(dim, 128))](
            x, residual, post, comb, out, dim, HC=hc, CHANNELS=triton.next_power_of_2(hc), WIDTH=128,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
