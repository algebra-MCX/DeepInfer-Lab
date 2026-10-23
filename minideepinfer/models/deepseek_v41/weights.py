"""Checkpoint reading and FP8 / FP4 dequantization for DeepSeek-V4.1.

FP8 linear weights carry one E8M0 scale per 32x32 block; FP4 expert weights pack two E2M1 values
per byte with one E8M0 scale per 32 inputs; the Engram tables are FP8 with one scale per 32 values
of a row. The engine keeps tensors as stored (`iterate_raw`) and dequantizes on use; the reference
model is loaded in bf16 (`iterate_dequantized`). Dequantization is exact up to the bf16 rounding.
"""

from __future__ import annotations

import functools
import glob
import os
from collections.abc import Iterator

import torch
import triton
import triton.language as tl
from safetensors import safe_open

BLOCK = 32
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
# Vision tower and VL routing bias are not used by the text model; the DSpark head is loaded separately.
SKIPPED_PREFIXES = ("vision.", "aligner.", "image_", "mtp.")


@functools.cache
def _fp4_table(device: torch.device) -> torch.Tensor:
    return FP4_TABLE.to(device)  # cached: a per-call copy from pageable memory would sync the host


@triton.jit
def _dequant_fp8_kernel(
    W, S, Out, rows, cols, w_row, w_col, s_row, s_col,
    ROW_SCALES: tl.constexpr, QBLOCK: tl.constexpr, TILE: tl.constexpr,
):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row, col = offsets // cols, offsets % cols
    scale_row = row if ROW_SCALES else row // QBLOCK
    w = tl.load(W + row * w_row + col * w_col, row < rows, other=0.0).to(tl.float32)
    s = tl.load(S + scale_row * s_row + (col // QBLOCK) * s_col, row < rows, other=0)
    tl.store(Out + offsets, w * s, row < rows)


def dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.is_cuda:
        # Convert only the small scale table; index it directly instead of expanding it to [out, in].
        scales = scale.float()
        out = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
        _dequant_fp8_kernel[(triton.cdiv(weight.numel(), 1024),)](
            weight, scales, out, *weight.shape, *weight.stride(), *scales.stride(),
            ROW_SCALES=scale.size(0) == weight.size(0), QBLOCK=BLOCK, TILE=1024,
        )
        return out
    w = weight.float()
    s = scale.float()
    if s.size(0) == w.size(0):  # one scale per 32 values of a row (Engram tables)
        s = s.repeat_interleave(BLOCK, dim=1)[:, : w.size(1)]
    else:  # one scale per 32x32 block
        s = s.repeat_interleave(BLOCK, dim=0)[: w.size(0)].repeat_interleave(BLOCK, dim=1)[:, : w.size(1)]
    return (w * s).to(torch.bfloat16)


@triton.jit
def _dequant_fp4_kernel(Packed, Scales, Out, size, QBLOCK: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    byte = tl.load(Packed + offsets // 2, offsets < size, other=0).to(tl.uint8)
    code = (byte >> ((offsets % 2) * 4)) & 15
    magnitude = code & 7
    # E2M1: subnormals are 0 and 0.5; normal magnitudes are (2 + mantissa) * 2**(exponent - 2).
    exponent = (magnitude >> 1).to(tl.int32)
    power = ((exponent + 127 - 2) << 23).to(tl.float32, bitcast=True)
    value = tl.where(magnitude < 2, magnitude * 0.5, (2 + (magnitude & 1)) * power)
    value = tl.where((code & 8) != 0, -value, value)
    # E8M0 scale 2**(e - 127): e = 0 is the fp32 subnormal 2**-127 and e = 255 is NaN.
    e = tl.load(Scales + offsets // QBLOCK, offsets < size, other=127).to(tl.int32)
    bits = tl.where(e == 0, 0x00400000, tl.where(e == 255, 0x7FC00000, e << 23))
    tl.store(Out + offsets, value * bits.to(tl.float32, bitcast=True), offsets < size)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """bf16 weights from packed E2M1 pairs and one E8M0 scale per 32 values; written into `out` if given."""
    if packed.is_cuda and packed.is_contiguous() and scale.is_contiguous() and scale.dtype == torch.float8_e8m0fnu:
        if out is None:
            out = torch.empty(*packed.shape[:-1], packed.size(-1) * 2, dtype=torch.bfloat16, device=packed.device)
        assert out.is_contiguous() and out.numel() == 2 * packed.numel()
        _dequant_fp4_kernel[(triton.cdiv(out.numel(), 1024),)](
            packed, scale.view(torch.uint8), out, out.numel(), QBLOCK=BLOCK, TILE=1024,
        )
        return out
    x = packed.view(torch.uint8)
    table = _fp4_table(x.device)
    values = torch.stack([table[(x & 0x0F).long()], table[(x >> 4).long()]], dim=-1).flatten(-2)
    values = (values * scale.float().repeat_interleave(BLOCK, dim=-1)).to(torch.bfloat16)
    return values if out is None else out.copy_(values)


def _shards(model_path: str) -> Iterator:
    """Opens each safetensors shard in turn, yielding (handle, sorted text-model tensor names)."""
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {model_path}")
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            yield f, [n for n in sorted(f.keys()) if not n.startswith(SKIPPED_PREFIXES) and not n.endswith("bias_vl")]


def iterate_raw(model_path: str, select=lambda name: True) -> Iterator[tuple[str, torch.Tensor]]:
    """Yields (name, tensor) as stored, FP8 / FP4 tensors and their scales included.

    `select(name)` returns True to read a tensor, False to skip it (another rank's shard), or a
    `slice` to read only those rows (a row-sharded table).
    """
    for f, names in _shards(model_path):
        for name in names:
            part = select(name)
            if part is True:
                yield name, f.get_tensor(name)
            elif part is not False:
                yield name, f.get_slice(name)[part]


def iterate_dequantized(model_path: str, keep_engram_fp8: bool = False) -> Iterator[tuple[str, torch.Tensor]]:
    """Yields (reference parameter name, tensor) for every text-model tensor of the checkpoint."""
    for f, names in _shards(model_path):
        for name in names:
            if name.endswith(".scale"):
                continue
            tensor = f.get_tensor(name)
            scale_name = name.removesuffix(".weight") + ".scale"
            if name.endswith(".weight") and scale_name in names:
                scale = f.get_tensor(scale_name)
                if ".engram.embed." in name and keep_engram_fp8:
                    yield name, tensor
                    yield scale_name, scale
                    continue
                tensor = dequant_fp4(tensor, scale) if tensor.dtype == torch.int8 else dequant_fp8(tensor, scale)
            yield name, tensor
