"""SwiGLU keeps clipping direction and FP32 routing multiplication before the final BF16 cast."""

import pytest
import torch
import torch.nn.functional as F

from deepinfer_lab.models.deepseek_v41.model import swiglu_bf16


def reference(gate, up, limit, weights=None):
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    out = F.silu(gate) * up
    return (out if weights is None else weights * out).bfloat16()


def check_close(actual, expected):
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual.isinf(), expected.isinf())
    finite = actual.isfinite() & expected.isfinite()
    ulps = (actual.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
    assert not finite.any() or ulps[finite].max() <= 1
    zeros = (actual == 0) & (expected == 0)
    assert torch.equal(actual.signbit()[zeros], expected.signbit()[zeros])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_swiglu_empty_and_strided_fallback(device):
    gate, up = [torch.randn(7, 34, device=device).bfloat16()[:, ::2] for _ in range(2)]
    weights = torch.randn(7, 1, device=device)
    broadcast = torch.ones(7, 1, device=device, dtype=torch.bfloat16)
    for g, u, w in [(gate, up, weights), (gate[:0], up[:0], weights[:0]),
                    (gate.contiguous(), broadcast, weights)]:
        assert torch.equal(swiglu_bf16(g, u, 10., w), reference(g, u, 10., w))


@pytest.mark.gpu
@pytest.mark.parametrize("rows,width,limit,weighted", [(1, 67, 0., False), (7, 67, 10., True),
                                                      (64, 256, 10., True), (4096, 256, 0., True)])
def test_swiglu_matches_unfused(rows, width, limit, weighted):
    torch.manual_seed(91)
    gate, up = [torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) * 12 for _ in range(2)]
    weights = torch.randn(rows, 1, device="cuda") if weighted else None
    check_close(swiglu_bf16(gate, up, limit, weights), reference(gate, up, limit, weights))


@pytest.mark.gpu
@pytest.mark.parametrize("limit", [0., 10.])
def test_swiglu_every_bf16_gate_encoding(limit):
    gate = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).view(256, 256).cuda()
    up = torch.full_like(gate, .25)
    weights = torch.full((256, 1), .7, device="cuda")
    check_close(swiglu_bf16(gate, up, limit, weights), reference(gate, up, limit, weights))


@pytest.mark.gpu
def test_swiglu_weights_precede_bf16_rounding_and_gate_has_no_lower_clip():
    gate = torch.tensor([[.25, -20.]], device="cuda", dtype=torch.bfloat16)
    up = torch.tensor([[.25, -20.]], device="cuda", dtype=torch.bfloat16)
    weights = torch.tensor([[.7]], device="cuda")
    actual = swiglu_bf16(gate, up, 10., weights)
    assert actual[0, 0].item() == 0.0245361328125  # Casting before weights gives 0.024658203125.
    assert torch.equal(actual, reference(gate, up, 10., weights))


@pytest.mark.gpu
def test_swiglu_graph_reads_updated_inputs():
    gate, up = [torch.randn(7, 67, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    weights = torch.rand(7, 1, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        swiglu_bf16(gate, up, 10., weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = swiglu_bf16(gate, up, 10., weights)
    torch.cuda.current_stream().wait_stream(stream)
    gate.mul_(20)
    up.neg_()
    weights[0], weights[1], weights[2] = 0., -.5, float("nan")
    graph.replay()
    check_close(actual, reference(gate, up, 10., weights))
