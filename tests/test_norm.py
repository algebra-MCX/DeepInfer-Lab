"""RMSNorm fusion bounds reduction error without relaxing the model's reference tolerances."""

import pytest
import torch

from deepinfer_lab.models.deepseek_v41.model import RMSNorm

pytestmark = pytest.mark.gpu


def unfused(x, weight, eps):
    values = x.float()
    return (weight * (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


def check_close(actual, expected):
    if actual.dtype == torch.bfloat16:
        # A change in fp32 reduction order can move a value across one final bf16 rounding boundary.
        ulps = (actual.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
        assert ulps.numel() == 0 or ulps.max() <= 1
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("shape", [(1, 1), (35, 67), (7, 3, 512), (4096, 512), (3, 8192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_matches_unfused(shape, dtype):
    torch.manual_seed(43)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    norm = RMSNorm(shape[-1], 1e-6).to(device="cuda", dtype=dtype)
    norm.weight.data.normal_()
    check_close(norm(x), unfused(x, norm.weight, norm.eps))


def test_rmsnorm_zero_empty_and_strided_inputs():
    norm = RMSNorm(67, 1e-6).cuda()
    x = torch.zeros(2, 67, device="cuda")
    x[1].fill_(-0.0)
    actual, expected = norm(x), unfused(x, norm.weight, norm.eps)
    assert torch.equal(actual, expected) and torch.equal(actual.signbit(), expected.signbit())
    assert norm(torch.empty(0, 67, device="cuda")).shape == (0, 67)
    x = torch.randn(67, 35, device="cuda").T
    assert not x.is_contiguous()
    assert torch.equal(norm(x), unfused(x, norm.weight, norm.eps))


def test_rmsnorm_graph_reads_updated_inputs():
    x = torch.randn(7, 67, device="cuda")
    norm, stream = RMSNorm(67, 1e-6).cuda(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        norm(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = norm(x)
    torch.cuda.current_stream().wait_stream(stream)
    x.mul_(3)
    norm.weight.data.mul_(0.5)
    graph.replay()
    check_close(actual, unfused(x, norm.weight, norm.eps))
