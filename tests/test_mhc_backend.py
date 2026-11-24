"""mHC backend contracts, delayed pre-mix ownership, and the official operator chain."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from deepinfer_lab.config import Config
from deepinfer_lab.models.deepseek_v41.mhc import DeepSeekMHC, NativeMHC, get_mhc_backend
from deepinfer_lab.models.deepseek_v41.model import Block


def test_importing_the_runtime_does_not_load_optional_mhc_dependencies():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import deepinfer_lab; "
        "assert 'deep_gemm' not in sys.modules; assert 'tile_kernels' not in sys.modules",
    ], check=True)


def test_invalid_backend_fails_before_checkpoint_resolution():
    with pytest.raises(ValueError, match="mhc_backend"):
        Config("unused-model", mhc_backend="unknown")


@pytest.mark.parametrize("hc,dim", [(3, 128), (4, 33), (4, 0)])
def test_official_backend_checks_model_shape_before_loading_weights(monkeypatch, hc, dim):
    monkeypatch.setattr("deepinfer_lab.config.resolve_model_path", lambda _: "unused")
    monkeypatch.setattr(
        "deepinfer_lab.config.V41Config.from_pretrained",
        lambda _: SimpleNamespace(hc_mult=hc, dim=dim),
    )
    with pytest.raises(ValueError, match="hc_mult=4"):
        Config("unused", mhc_backend="deepseek")


def test_native_mixes_normalize_over_the_entire_residual():
    x = torch.tensor([[[1., 2.], [3., 4.]]]).bfloat16()
    fn = torch.arange(32).reshape(8, 4).float() / 32
    scale = torch.tensor([0.2, 0.3, 0.4])
    base = torch.linspace(-0.5, 0.5, 8)
    pre, post, comb = NativeMHC.mixes(x, fn, scale, base, 0.01, 20, 1e-6)
    z = (x.flatten(1).float() @ fn.T) / (x.float().square().mean() + 0.01).sqrt()
    torch.testing.assert_close(pre, torch.sigmoid(z[:, :2] * scale[0] + base[:2]) + 1e-6)
    torch.testing.assert_close(post, 2 * torch.sigmoid(z[:, 2:4] * scale[1] + base[2:4]))
    torch.testing.assert_close(comb.sum(-2), torch.ones_like(pre), atol=1e-5, rtol=0)


def test_native_mixing_keeps_source_destination_direction_and_fp32_products():
    residual = torch.tensor([[[1., 2.], [3., 4.]]]).bfloat16()
    x = torch.tensor([[5., 6.]]).bfloat16()
    pre = torch.tensor([[2., 3.]])
    comb = torch.tensor([[[1., 2.], [4., 8.]]])
    assert NativeMHC.pre(residual, pre).tolist() == [[11, 16]]
    assert NativeMHC.post(x, residual, pre, comb).tolist() == [[[23, 30], [41, 54]]]
    residual = torch.tensor([[[256.], [1.], [-256.]]]).bfloat16()
    assert NativeMHC.pre(residual, torch.tensor([[1.001, 1., 1.]])).item() == 1.2578125


def test_block_passes_pre_mix_to_the_next_sublayer():
    """A backend port must not apply a sublayer's newly generated pre-mix to that sublayer."""
    seen = []

    class Ops:
        def mixes(self, x, fn, scale, base, *args):
            value = 2 if not seen else 3
            return torch.full((1, 4), value), None, None

        def pre(self, x, mix):
            seen.append(mix.clone())
            return x[:, 0]

        def post(self, x, residual, post, comb):
            return residual

    class Attention(nn.Module):
        def forward(self, x, *args):
            return x

    block = Block.__new__(Block)
    nn.Module.__init__(block)
    block.cfg = SimpleNamespace(norm_eps=1e-6, hc_sinkhorn_iters=20, hc_eps=1e-6)
    block.mhc = Ops()
    block.attn, block.ffn = Attention(), nn.Identity()
    block.attn_norm, block.ffn_norm = nn.Identity(), nn.Identity()
    for part in ("attn", "ffn"):
        for name in ("fn", "scale", "base"):
            setattr(block, f"hc_{part}_{name}", torch.empty(0))
    incoming = torch.tensor([[1., 0., 0., 0.]])
    residual = torch.zeros(1, 4, 128)
    out, next_mix = block(residual, incoming, None, None, None)
    assert out is residual
    torch.testing.assert_close(seen[0], incoming)
    assert seen[1].tolist() == [[2, 2, 2, 2]]
    assert next_mix.tolist() == [[3, 3, 3, 3]]


def test_deepseek_backend_rejects_cpu_instead_of_silently_falling_back():
    with pytest.raises(ValueError, match="CUDA BF16"):
        DeepSeekMHC().pre(torch.zeros(1, 4, 128).bfloat16(), torch.ones(1, 4))


@pytest.fixture
def official_mhc():
    pytest.importorskip("deep_gemm")
    pytest.importorskip("tile_kernels")
    return get_mhc_backend("deepseek")


@pytest.mark.gpu
@pytest.mark.parametrize("rows,dim,iters", [(0, 128, 20), (1, 128, 1), (7, 512, 20), (35, 5120, 20)])
@torch.inference_mode()
def test_official_mhc_chain_matches_reference(official_mhc, rows, dim, iters):
    gen = torch.Generator().manual_seed(173)
    residual = torch.randn(rows, 4, dim, generator=gen).bfloat16()
    fn = torch.randn(24, 4 * dim, generator=gen) * 0.01
    scale, base = torch.randn(3, generator=gen) * 0.2, torch.randn(24, generator=gen)
    expected = NativeMHC.mixes(residual, fn, scale, base, 1e-6, iters, 1e-6)
    gpu = residual.cuda()
    actual = official_mhc.mixes(gpu, fn.cuda(), scale.cuda(), base.cuda(), 1e-6, iters, 1e-6)
    for a, e in zip(actual, expected, strict=True):
        assert a.dtype == torch.float32 and a.is_contiguous()
        torch.testing.assert_close(a.cpu(), e, rtol=1e-4, atol=1e-5)
    pre, post, comb = actual
    x = torch.randn(rows, dim, generator=gen).bfloat16()
    torch.testing.assert_close(
        official_mhc.expand(x.cuda(), 4).cpu(), NativeMHC.expand(x, 4), rtol=0, atol=0,
    )
    for a, e in (
        (official_mhc.pre(gpu, pre), NativeMHC.pre(residual, expected[0])),
        (official_mhc.post(x.cuda(), gpu, post, comb), NativeMHC.post(x, residual, expected[1], expected[2])),
    ):
        assert a.dtype == torch.bfloat16
        torch.testing.assert_close(a.cpu().float(), e.float(), rtol=1e-2, atol=2e-5)


@pytest.mark.gpu
@torch.inference_mode()
def test_official_post_direction_and_strided_inputs(official_mhc):
    residual = torch.zeros(1, 4, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
    residual[:, :, :2] = torch.tensor([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]], device="cuda")
    x = torch.ones(1, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
    post = torch.tensor([[1., 2., 3., 4.]], device="cuda")
    comb = torch.arange(16, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    expected = NativeMHC.post(x.cpu(), residual.cpu(), post.cpu(), comb.cpu())
    torch.testing.assert_close(official_mhc.post(x, residual, post, comb).cpu(), expected, rtol=0, atol=0)


@pytest.mark.gpu
@torch.inference_mode()
def test_official_graph_replays_updated_inputs_and_keeps_previous_results(official_mhc):
    residual = torch.randn(7, 4, 128, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 512, device="cuda") * 0.01
    scale, base = torch.rand(3, device="cuda"), torch.randn(24, device="cuda")

    def chain():
        mixes = official_mhc.mixes(residual, fn, scale, base, 1e-6, 20, 1e-6)
        x = official_mhc.pre(residual, mixes[0])
        return official_mhc.post(x, residual, mixes[1], mixes[2]), mixes[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        previous = chain()
        saved = tuple(t.clone() for t in previous)
        chain()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = chain()
    torch.cuda.current_stream().wait_stream(stream)
    residual.mul_(0.5)
    fn.neg_()
    scale.add_(0.1)
    base.neg_()
    graph.replay()
    for a, e in zip(actual, chain(), strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    for a, e in zip(previous, saved, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
