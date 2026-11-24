"""mHC inference backends with the same flat-token, source-to-destination mix convention.

The native backend is the existing PyTorch/Triton reference. The optional DeepSeek backend
composes DeepGEMM's prenorm GEMM and TileKernels' individual inference operators. Keeping
these stages separate makes this a correctness baseline for later fusion experiments.
"""

from functools import cache
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from deepinfer_lab.models.deepseek_v41.hc import collapse_cuda, expand_cuda, split_sinkhorn_cuda

Mixes = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def split_sinkhorn(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, eps: float,
) -> Mixes:
    """Return pre/post [T, HC] and comb [T, source HC, destination HC]."""
    if mixes.is_cuda and iters >= 1 and all(
        t.is_contiguous() and t.dtype == torch.float32 for t in (mixes, scale, base)
    ):
        return split_sinkhorn_cuda(mixes, scale, base, hc, iters, eps)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class NativeMHC:
    """Current implementation, including its CPU and noncontiguous reference paths."""

    @staticmethod
    def expand(x: torch.Tensor, hc: int) -> torch.Tensor:
        return x.unsqueeze(1).repeat(1, hc, 1)

    @staticmethod
    def mixes(
        x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
        norm_eps: float, iters: int, eps: float,
    ) -> Mixes:
        xf = x.flatten(1).float()
        mixes = F.linear(xf, fn) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
        return split_sinkhorn(mixes, scale, base, x.shape[1], iters, eps)

    @staticmethod
    def pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        if (x.is_cuda and x.is_contiguous() and pre.is_contiguous()
                and x.dtype == torch.bfloat16 and pre.dtype == torch.float32 and 0 < x.shape[1] <= 8):
            return collapse_cuda(x, pre)
        return (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)

    @staticmethod
    def post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        if (x.is_cuda and all(t.is_contiguous() for t in (x, residual, post, comb))
                and x.dtype == residual.dtype == torch.bfloat16 and post.dtype == comb.dtype == torch.float32
                and 0 < residual.shape[1] <= 8):
            return expand_cuda(x, residual, post, comb)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(1)
        return y.type_as(x)


@cache
def _deepseek_ops() -> SimpleNamespace:
    """Import once on first use, outside graph capture during the runner's warmup."""
    try:
        from deep_gemm import tf32_hc_prenorm_gemm
        from tile_kernels.mhc.expand_kernel import expand_to_mhc_fwd
        from tile_kernels.mhc.norm_fn_kernel import mhc_reduce_partials_and_rmsnorm_fwd
        from tile_kernels.mhc.post_kernel import mhc_post_fwd
        from tile_kernels.mhc.pre_apply_mix_kernel import mhc_pre_apply_mix_fwd
        from tile_kernels.mhc.pre_split_mixes_kernel import mhc_pre_split_mixes_fwd
        from tile_kernels.mhc.sinkhorn_kernel import mhc_sinkhorn_fwd
    except ImportError as exc:
        raise RuntimeError(
            "mhc_backend='deepseek' requires DeepGEMM and TileKernels; "
            "install the 'mhc' extra (see docs/MHC.md) with a compatible CUDA toolkit."
        ) from exc
    return SimpleNamespace(
        prenorm=tf32_hc_prenorm_gemm, reduce=mhc_reduce_partials_and_rmsnorm_fwd,
        split=mhc_pre_split_mixes_fwd, sinkhorn=mhc_sinkhorn_fwd,
        pre=mhc_pre_apply_mix_fwd, post=mhc_post_fwd, expand=expand_to_mhc_fwd,
    )


class DeepSeekMHC:
    """Official, unfused inference chain for HC=4 and CUDA BF16 hidden sizes divisible by 128.

    An explicitly selected backend never silently falls back to the native implementation.
    Output allocations belong to each call, including pre-mixes retained for the next layer.
    """

    @staticmethod
    def _check(x: torch.Tensor) -> None:
        if (not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 3
                or x.shape[1] != 4 or x.shape[2] == 0 or x.shape[2] % 128):
            raise ValueError("DeepSeek mHC requires CUDA BF16 [T, 4, D] with positive D divisible by 128")

    @staticmethod
    def _check_mix(tensor: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> None:
        if tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != device:
            raise ValueError(f"DeepSeek mHC expects FP32 mix/weight shape {shape} on {device}")

    def expand(self, x: torch.Tensor, hc: int) -> torch.Tensor:
        # Check the residual contract without materializing the expanded tensor.
        self._check(x.unsqueeze(1).expand(-1, hc, -1))
        out = x.new_empty(x.shape[0], hc, x.shape[1])
        if x.shape[0]:
            _deepseek_ops().expand(x.contiguous(), out)
        return out

    def mixes(
        self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
        norm_eps: float, iters: int, eps: float,
    ) -> Mixes:
        self._check(x)
        rows, hc, dim = x.shape
        width = hc * (hc + 2)
        for tensor, shape in ((fn, (width, hc * dim)), (scale, (3,)), (base, (width,))):
            self._check_mix(tensor, shape, x.device)
        if iters < 1:
            raise ValueError("DeepSeek mHC needs at least one Sinkhorn iteration")
        pre = torch.empty(rows, hc, dtype=torch.float32, device=x.device)
        post = torch.empty_like(pre)
        comb = torch.empty(rows, hc, hc, dtype=torch.float32, device=x.device)
        if not rows:
            return pre, post, comb
        ops = _deepseek_ops()
        # One unsplit GEMM is the initial baseline: BF16 residual x FP32 fn -> FP32 partials.
        # The squared sum is over all HC * D features, matching the native RMS denominator.
        partial = torch.empty(1, rows, 1, width, dtype=torch.float32, device=x.device)
        sqr_partial = torch.empty(1, rows, 1, dtype=torch.float32, device=x.device)
        ops.prenorm(x.flatten(1).contiguous(), fn.contiguous(), partial.view(rows, width), sqr_partial.view(rows))
        mixes = torch.empty(rows, width, dtype=torch.float32, device=x.device)
        ops.reduce(
            partial, sqr_partial, torch.empty_like(partial[0]), torch.empty_like(sqr_partial[0]), mixes,
            rms_group_size=hc * dim, rms_eps=norm_eps, n_splits=1,
        )
        comb_logits = torch.empty_like(comb)
        ops.split(mixes, scale.contiguous(), base.contiguous(), pre, post, comb_logits.view(rows, hc * hc), 2.0, eps)
        ops.sinkhorn(comb_logits, comb, iters, eps)
        return pre, post, comb

    def pre(self, x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        self._check(x)
        self._check_mix(pre, x.shape[:2], x.device)
        out = x.new_empty(x.shape[0], x.shape[2])
        if x.shape[0]:
            _deepseek_ops().pre(x.contiguous(), pre.contiguous(), out)
        return out

    def post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        self._check(residual)
        rows, hc, dim = residual.shape
        if x.shape != (rows, dim) or x.dtype != residual.dtype or x.device != residual.device:
            raise ValueError("DeepSeek mHC post requires a CUDA BF16 [T, D] sublayer output")
        self._check_mix(post, (rows, hc), residual.device)
        self._check_mix(comb, (rows, hc, hc), residual.device)
        if not rows:
            return torch.empty_like(residual)
        # TileKernels post uses [batch, tokens, HC, D]; keep source/destination axes unchanged.
        return _deepseek_ops().post(
            x.contiguous().unsqueeze(0), residual.contiguous().unsqueeze(0),
            post.contiguous().view(1, rows, hc, 1), comb.contiguous().unsqueeze(0),
        ).squeeze(0)


@cache
def get_mhc_backend(name: str) -> NativeMHC | DeepSeekMHC:
    if name == "native":
        return NativeMHC()
    if name == "deepseek":
        return DeepSeekMHC()
    raise ValueError(f"unknown mhc_backend {name!r}; choose 'native' or 'deepseek'")
