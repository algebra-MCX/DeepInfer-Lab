"""Fake quantization (quantize then dequantize in place), matching the reference TileLang kernels.

V4.1 stores parts of its attention state in low precision, so the model output depends on these
round trips: the sliding-window KV and the attention output go through FP8, the indexer through
FP4 with power-of-two scales, and the compressed KV through FP4 with E4M3 scales.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
FP4_MAX = 6.0


@functools.cache
def _fp4_grid(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """E2M1 magnitudes, the midpoints between them and their mantissa bits, cached on `device`.

    Building them per call would copy from pageable host memory: a host sync, illegal in a CUDA graph.
    """
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    odd = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device, dtype=torch.bool)
    return grid, (grid[1:] + grid[:-1]) / 2, odd


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive fp32 x, exact (bit manipulation as in the reference)."""
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    exp = exp - 127 + ((bits & 0x7FFFFF) != 0).int()
    return ((exp + 127) << 23).view(torch.float32)


@triton.jit
def _pow2_ceil_device(x):
    bits = x.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fp8_fake_quant_kernel(X, Out, size, block, BLOCK: tl.constexpr, GROUPS: tl.constexpr):
    groups = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    channels = tl.arange(0, BLOCK)
    offsets = groups[:, None] * block + channels[None, :]
    valid = (offsets < size) & (channels[None, :] < block)
    values = tl.load(X + offsets, valid, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), 1), 1e-4)
    ratio = tl.div_rn(amax, 448.0)
    # Round the scale up to a power of two using exactly the reference's fp32 exponent rule.
    scale = _pow2_ceil_device(ratio)
    scaled = tl.div_rn(values, scale[:, None])
    quantized = tl.minimum(tl.maximum(scaled, -448.0), 448.0).to(tl.float8e4nv).to(tl.float32)
    tl.store(Out + offsets, quantized * scale[:, None], valid)


def fp8_fake_quant(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    """E4M3 with one power-of-two (UE8M0) scale per `block` values along the last dim."""
    if x.is_cuda and x.is_contiguous() and x.shape[-1] % block == 0:
        out = torch.empty_like(x)
        _fp8_fake_quant_kernel[(triton.cdiv(x.numel() // block, 16),)](
            x, out, x.numel(), block, BLOCK=triton.next_power_of_2(block), GROUPS=16,
        )
        return out
    shape = x.shape
    v = x.float().unflatten(-1, (-1, block))
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil(amax / FP8_MAX)
    q = (v / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
    return q.reshape(shape).to(x.dtype)


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value, ties to the even mantissa, keeping the sign."""
    grid, mid, odd = _fp4_grid(v.device)
    a = v.abs()
    i = torch.bucketize(a, mid)  # nearest grid index; an exact midpoint maps to the lower value
    tie = (i < len(mid)) & (a == mid[i.clamp(max=len(mid) - 1)])
    i = torch.where(tie & odd[i], i + 1, i)
    return torch.copysign(grid[i], v)


@triton.jit
def _round_fp4_device(x):
    a = tl.abs(x)
    # Alternating > / >= thresholds implement ties to the even E2M1 mantissa.
    index = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32)
             + (a >= 1.75).to(tl.int32) + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
             + (a > 5.0).to(tl.int32))
    power = (((index >> 1) + 127 - 2) << 23).to(tl.float32, bitcast=True)
    magnitude = tl.where(index < 2, index * 0.5, (2 + (index & 1)) * power)
    # Copy the sign bit, including negative zero.
    bits = magnitude.to(tl.int32, bitcast=True) | (x.to(tl.int32, bitcast=True) & -2147483648)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _fp4_fake_quant_kernel(X, Out, size, block, E4M3: tl.constexpr, BLOCK: tl.constexpr, GROUPS: tl.constexpr):
    groups = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    channels = tl.arange(0, BLOCK)
    offsets = groups[:, None] * block + channels[None, :]
    valid = (offsets < size) & (channels[None, :] < block)
    values = tl.load(X + offsets, valid, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(values), 1)
    if E4M3:
        scale = tl.div_rn(tl.maximum(amax, 6 * 2**-9), 6.0).to(tl.float8e4nv).to(tl.float32)
    else:
        scale = _pow2_ceil_device(tl.div_rn(tl.maximum(amax, 6 * 2**-126), 6.0))
    scaled = tl.minimum(tl.maximum(tl.div_rn(values, scale[:, None]), -6.0), 6.0)
    tl.store(Out + offsets, _round_fp4_device(scaled) * scale[:, None], valid)


def fp4_fake_quant(x: torch.Tensor, block: int, e4m3_scale: bool) -> torch.Tensor:
    """E2M1 with one scale per `block` values: E4M3 scales (compressed KV) or power-of-two (indexer)."""
    if x.is_cuda and x.is_contiguous() and x.shape[-1] % block == 0:
        out = torch.empty_like(x)
        _fp4_fake_quant_kernel[(triton.cdiv(x.numel() // block, 16),)](
            x, out, x.numel(), block, E4M3=e4m3_scale, BLOCK=triton.next_power_of_2(block), GROUPS=16,
        )
        return out
    shape = x.shape
    v = x.float().unflatten(-1, (-1, block))
    amax = v.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        s = (amax.clamp_min(6 * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2**-126) / FP4_MAX)
    q = _round_fp4((v / s).clamp(-FP4_MAX, FP4_MAX)) * s
    return q.reshape(shape).to(x.dtype)
