"""Fused activation quantization keeps the reference's exact rounding and scale rules."""

import pytest
import torch

from minideepinfer.models.deepseek_v41.quant import fp4_fake_quant, fp8_fake_quant


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block", [16, 32, 48])
def test_fp8_fake_quant_matches_cpu(dtype, block):
    """Wide dynamic range, partial groups of blocks, zeros and non-power-of-two block lengths."""
    gen = torch.Generator().manual_seed(29)
    x = (torch.randn(35, 96, generator=gen) * torch.logspace(-8, 8, 96)).to(dtype)
    x[0].zero_()
    x[1].fill_(-0.0)
    expected = fp8_fake_quant(x, block)
    actual = fp8_fake_quant(x.cuda(), block).cpu()
    assert torch.equal(actual, expected)
    zeros = expected == 0
    assert torch.equal(actual[zeros].signbit(), expected[zeros].signbit())


@pytest.mark.gpu
def test_fp8_fake_quant_strided_fallback():
    x = torch.randn(96, 35).T.bfloat16()
    gpu = x.cuda()
    assert not gpu.is_contiguous()
    assert torch.equal(fp8_fake_quant(gpu).cpu(), fp8_fake_quant(x))


@pytest.mark.gpu
def test_fp8_rounding_midpoints_and_scale_boundaries():
    grid = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (grid[:-1] + grid[1:]) / 2
    midpoints = torch.nn.functional.pad(midpoints, (0, -midpoints.numel() % 31)).view(-1, 31)
    # Fix scale=1, so every adjacent FP8 midpoint exercises round-to-nearest-even directly.
    x = torch.cat([midpoints, torch.full((len(midpoints), 1), 448.0)], dim=1)
    assert torch.equal(fp8_fake_quant(x.cuda()).cpu(), fp8_fake_quant(x))
    boundary = torch.tensor([448.0])
    neighbors = torch.cat([boundary.nextafter(torch.zeros(1)), boundary, boundary.nextafter(torch.full((1,), float("inf")))])
    x = torch.ldexp(neighbors[:, None].expand(-1, 32), torch.tensor([-30, 0, 30])[:, None])
    assert torch.equal(fp8_fake_quant(x.cuda()).cpu(), fp8_fake_quant(x))


@pytest.mark.gpu
@pytest.mark.parametrize("block,e4m3", [(16, True), (32, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp4_fake_quant_matches_cpu(block, e4m3, dtype):
    gen = torch.Generator().manual_seed(31)
    x = (torch.randn(35, 96, generator=gen) * torch.logspace(-6, 2, 96)).to(dtype)
    x[0].zero_()
    x[1].fill_(-0.0)
    expected = fp4_fake_quant(x, block, e4m3)
    actual = fp4_fake_quant(x.cuda(), block, e4m3).cpu()
    assert torch.equal(actual, expected)
    zeros = expected == 0
    assert torch.equal(actual[zeros].signbit(), expected[zeros].signbit())


@pytest.mark.gpu
@pytest.mark.parametrize("block,e4m3", [(16, True), (32, False)])
def test_fp4_rounding_midpoints_and_strided_fallback(block, e4m3):
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    row = torch.cat([mids, -mids, torch.tensor([-0.0, 6.0])])
    x = row.repeat(3, block // 16)
    expected = fp4_fake_quant(x, block, e4m3)
    assert torch.equal(fp4_fake_quant(x.cuda(), block, e4m3).cpu(), expected)
    gpu = x.cuda().T.contiguous().T
    assert not gpu.is_contiguous()
    assert torch.equal(fp4_fake_quant(gpu, block, e4m3).cpu(), expected)
