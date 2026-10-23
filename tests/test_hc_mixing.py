"""Hyper-connection mixing preserves channel direction, FP32 accumulation and graph input ownership."""

import pytest
import torch

from minideepinfer.models.deepseek_v41.model import Block

pytestmark = pytest.mark.gpu


def check_mixing(x, residual, pre, post, comb):
    expected = (
        Block.hc_pre(residual.cpu(), pre.cpu()),
        Block.hc_post(x.cpu(), residual.cpu(), post.cpu(), comb.cpu()),
    )
    actual = Block.hc_pre(residual, pre), Block.hc_post(x, residual, post, comb)
    for a, e in zip(actual, expected):
        # Changing FP32 reduction order can cross final BF16 rounding boundaries.
        torch.testing.assert_close(a.cpu().float(), e.float(), rtol=1e-2, atol=2e-5)
    return actual


@pytest.mark.parametrize("hc,dim", [(1, 1), (3, 33), (4, 512), (8, 4096)])
def test_hc_mixing_matches_cpu(hc, dim):
    torch.manual_seed(61)
    residual = torch.randn(7, hc, dim, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(7, dim, device="cuda", dtype=residual.dtype)
    pre, post = torch.randn(2, 7, hc, device="cuda")
    comb = torch.randn(7, hc, hc, device="cuda")
    check_mixing(x, residual, pre, post, comb)


def test_hc_mixing_channel_direction():
    residual = torch.tensor([[[1, 2], [3, 4]]], device="cuda", dtype=torch.bfloat16)
    x = torch.tensor([[5, 6]], device="cuda", dtype=residual.dtype)
    pre = torch.tensor([[2., 3.]], device="cuda")
    comb = torch.tensor([[[1., 2.], [4., 8.]]], device="cuda")
    assert Block.hc_pre(residual, pre).tolist() == [[11, 16]]
    assert Block.hc_post(x, residual, pre, comb).tolist() == [[[23, 30], [41, 54]]]


def test_hc_mixing_accumulates_products_in_fp32():
    residual = torch.tensor([[[256.], [1.], [-256.]]], device="cuda", dtype=torch.bfloat16)
    pre = torch.tensor([[1.001, 1., 1.]], device="cuda")
    comb = torch.zeros(1, 3, 3, device="cuda")
    comb[0, :, 0] = pre[0]
    x, post = residual.new_zeros(1, 1), torch.zeros_like(pre)
    assert Block.hc_pre(residual, pre).item() == 1.2578125
    assert Block.hc_post(x, residual, post, comb).tolist() == [[[1.2578125], [0.], [0.]]]


def test_hc_mixing_empty_and_strided():
    residual = torch.randn(7, 3, 66, device="cuda", dtype=torch.bfloat16)[..., ::2]
    x = torch.randn(7, 66, device="cuda", dtype=residual.dtype)[:, ::2]
    pre, post = torch.randn(2, 7, 3, device="cuda")
    comb = torch.randn(7, 3, 3, device="cuda")
    check_mixing(x, residual, pre, post, comb)
    collapsed, expanded = check_mixing(x[:0].contiguous(), residual[:0].contiguous(), pre[:0], post[:0], comb[:0])
    assert collapsed.shape == (0, 33) and expanded.shape == (0, 3, 33)


def test_hc_mixing_graph_reads_updated_inputs():
    residual = torch.randn(7, 4, 33, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(7, 33, device="cuda", dtype=residual.dtype)
    pre, post = torch.randn(2, 7, 4, device="cuda")
    comb = torch.randn(7, 4, 4, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        Block.hc_pre(residual, pre)
        Block.hc_post(x, residual, post, comb)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = Block.hc_pre(residual, pre), Block.hc_post(x, residual, post, comb)
    torch.cuda.current_stream().wait_stream(stream)
    x.neg_()
    residual.mul_(2)
    pre.add_(0.3)
    post.mul_(0.5)
    comb.neg_()
    graph.replay()
    expected = Block.hc_pre(residual.cpu(), pre.cpu()), Block.hc_post(x.cpu(), residual.cpu(), post.cpu(), comb.cpu())
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu().float(), e.float(), rtol=1e-2, atol=2e-5)
