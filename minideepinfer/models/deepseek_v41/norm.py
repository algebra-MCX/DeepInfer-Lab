"""RMS normalization in one device pass, retaining fp32 normalization and the input's output dtype."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(X, Weight, Out, dim, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    channels = tl.arange(0, BLOCK)
    values = tl.load(X + row * dim + channels, channels < dim, other=0.0).to(tl.float32)
    weight = tl.load(Weight + channels, channels < dim, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, 0) / dim
    normalized = values * tl.rsqrt(variance + EPS)
    tl.store(Out + row * dim + channels, weight * normalized, channels < dim)


def rms_norm_cuda(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = torch.empty_like(x)
    dim = x.shape[-1]
    rows = x.numel() // dim
    if rows:
        block = triton.next_power_of_2(dim)
        _rms_norm[(rows,)](
            x, weight, out, dim, EPS=eps, BLOCK=block,
            num_warps=4 if block < 2048 else 8, enable_fp_fusion=False,
        )
    return out
