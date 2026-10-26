"""Weight dequantization must preserve the checkpoint's bf16 values exactly."""

import pytest
import torch
import triton

from deepinfer_lab.models.deepseek_v41.weights import dequant_fp4, dequant_fp8


@pytest.mark.gpu
@pytest.mark.parametrize("rows,cols,row_scales", [(1, 33, True), (35, 67, False), (35, 67, True), (64, 96, False)])
@pytest.mark.parametrize("strided", [False, True])
def test_fp8_dequant_matches_cpu(rows, cols, row_scales, strided):
    """Partial blocks/tiles, row scales, block scales, and noncontiguous views all use the same values."""
    gen = torch.Generator().manual_seed(17)
    w = torch.randn(rows, cols, generator=gen).mul(50).clamp(-448, 448).to(torch.float8_e4m3fn)
    shape = (rows if row_scales else triton.cdiv(rows, 32), triton.cdiv(cols, 32))
    s = torch.randint(120, 136, shape, generator=gen, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    expected = dequant_fp8(w, s)
    w, s = w.cuda(), s.cuda()
    if strided:
        # Transpose twice through contiguous storage to produce column-major input views.
        w, s = w.T.contiguous().T, s.T.contiguous().T
    actual = dequant_fp8(w, s)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.gpu
def test_fp8_dequant_special_values_and_graph_replay():
    """Every FP8 bit pattern and E8M0 exponent; a replay must read updated weights and scales."""
    w = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)[None, :].repeat(256, 1)
    s = torch.arange(256, dtype=torch.uint8)[:, None].expand(-1, 8).contiguous().view(torch.float8_e8m0fnu)
    expected = dequant_fp8(w, s)
    w, s = w.cuda(), s.cuda()
    torch.testing.assert_close(dequant_fp8(w, s).cpu(), expected, rtol=0, atol=0, equal_nan=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dequant_fp8(w, s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = dequant_fp8(w, s)
    torch.cuda.current_stream().wait_stream(stream)
    w.fill_(2)
    s.fill_(4)
    graph.replay()
    assert torch.equal(out, torch.full_like(out, 8))


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 16), (35, 48), (3, 35, 48)])
def test_fp4_dequant_matches_cpu(shape):
    """Unpack adjacent low/high nibbles in order, across rows, expert stacks and partial tiles."""
    gen = torch.Generator().manual_seed(19)
    packed = torch.randint(-128, 128, shape, generator=gen, dtype=torch.int8)
    scales = torch.randint(120, 136, (*shape[:-1], shape[-1] // 16), generator=gen, dtype=torch.uint8)
    scales = scales.view(torch.float8_e8m0fnu)
    assert torch.equal(dequant_fp4(packed.cuda(), scales.cuda()).cpu(), dequant_fp4(packed, scales))


@pytest.mark.gpu
def test_fp4_all_codes_and_scales():
    """Every packed byte and E8M0 exponent, including signed zeros, overflow, underflow and NaNs."""
    packed = torch.arange(256, dtype=torch.uint8)[None, :].repeat(256, 1).view(torch.int8)
    scales = torch.arange(256, dtype=torch.uint8)[:, None].expand(-1, 16).contiguous().view(torch.float8_e8m0fnu)
    expected = dequant_fp4(packed, scales)
    actual = dequant_fp4(packed.cuda(), scales.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = (expected == 0)
    assert torch.equal(actual[zeros].signbit(), expected[zeros].signbit())


@pytest.mark.gpu
def test_fp4_strided_fallback():
    """A noncontiguous view still follows the reference path correctly."""
    packed = torch.arange(64, dtype=torch.int8).reshape(2, 32)[:, ::2]
    scales = torch.ones(2, 1).to(torch.float8_e8m0fnu)
    expected = dequant_fp4(packed, scales)
    gpu = packed.cuda().T.contiguous().T
    assert not gpu.is_contiguous()
    assert torch.equal(dequant_fp4(gpu, scales.cuda()).cpu(), expected)


@pytest.mark.gpu
def test_fp4_graph_reads_updated_inputs():
    packed = torch.full((1, 16), 0x21, dtype=torch.int8, device="cuda")
    scales = torch.ones(1, 1, device="cuda").to(torch.float8_e8m0fnu)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dequant_fp4(packed, scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = dequant_fp4(packed, scales)
    torch.cuda.current_stream().wait_stream(stream)
    packed.fill_(0x76)
    scales.fill_(2)
    graph.replay()
    assert torch.equal(out, torch.tensor([[8, 12] * 16], device="cuda", dtype=torch.bfloat16))

