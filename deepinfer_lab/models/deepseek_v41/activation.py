"""SwiGLU clipping, SiLU, routing weights and the final BF16 cast in one device pass."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _swiglu(Gate, Up, Weight, Out, size, width, LIMIT: tl.constexpr,
            WEIGHTED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(Gate + i, i < size, other=0.0).to(tl.float32)
    up = tl.load(Up + i, i < size, other=0.0).to(tl.float32)
    if LIMIT > 0:
        # Gate has only an upper bound; up is clipped symmetrically. Preserve NaNs and signed zero.
        gate = tl.where(gate > LIMIT, LIMIT, gate)
        up = tl.where(up > LIMIT, LIMIT, tl.where(up < -LIMIT, -LIMIT, up))
    value = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up
    if WEIGHTED:
        value *= tl.load(Weight + i // width, i < size, other=0.0)
    tl.store(Out + i, value, i < size)


def swiglu_cuda(gate: torch.Tensor, up: torch.Tensor, limit: float,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    out = torch.empty_like(gate)
    if gate.numel():
        _swiglu[(triton.cdiv(gate.numel(), 256),)](
            gate, up, gate if weights is None else weights, out, gate.numel(), gate.shape[-1],
            LIMIT=limit, WEIGHTED=weights is not None, BLOCK=256, enable_fp_fusion=False,
        )
    return out
