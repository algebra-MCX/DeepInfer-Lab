"""Hyper-connection fusion preserves mix weights, normalization and the CPU reference path."""

import pytest
import torch

from deepinfer_lab.models.deepseek_v41.model import hc_split_sinkhorn

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("hc", [1, 3, 4, 8])
@pytest.mark.parametrize("iters", [1, 20])
def test_sinkhorn_matches_cpu(hc, iters):
    """Include non-power-of-two widths and the initial column normalization without further iterations."""
    gen = torch.Generator().manual_seed(41)
    mixes = torch.randn(35, hc * (hc + 2), generator=gen)
    scale = torch.rand(3, generator=gen)
    base = torch.randn(hc * (hc + 2), generator=gen)
    expected = hc_split_sinkhorn(mixes, scale, base, hc, iters, 1e-6)
    pre, post, comb = actual = hc_split_sinkhorn(mixes.cuda(), scale.cuda(), base.cuda(), hc, iters, 1e-6)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, rtol=1e-5, atol=1e-5)
    assert (pre >= 1e-6).all() and (post >= 0).all() and (post <= 2).all()
    assert (comb >= 0).all()
    torch.testing.assert_close(comb.sum(-2), torch.ones_like(pre), rtol=0, atol=1e-5)


def test_sinkhorn_empty_and_strided_inputs():
    hc = 4
    mixes = torch.randn(24, 7).T  # exercise the existing strided fallback
    scale, base = torch.rand(3), torch.randn(24)
    expected = hc_split_sinkhorn(mixes, scale, base, hc, 20, 1e-6)
    for a, e in zip(hc_split_sinkhorn(mixes.cuda(), scale.cuda(), base.cuda(), hc, 20, 1e-6), expected):
        torch.testing.assert_close(a.cpu(), e, rtol=1e-5, atol=1e-5)
    empty = torch.empty(0, 24, device="cuda")
    pre, post, comb = hc_split_sinkhorn(empty, scale.cuda(), base.cuda(), hc, 20, 1e-6)
    assert pre.shape == post.shape == (0, 4) and comb.shape == (0, 4, 4)


def test_sinkhorn_graph_reads_updated_inputs():
    mixes = torch.randn(7, 24, device="cuda")
    scale, base = torch.rand(3, device="cuda"), torch.randn(24, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    torch.cuda.current_stream().wait_stream(stream)
    mixes.mul_(0.5)
    scale.add_(0.3)
    base.neg_()
    graph.replay()
    expected = hc_split_sinkhorn(mixes.cpu(), scale.cpu(), base.cpu(), 4, 20, 1e-6)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, rtol=1e-5, atol=1e-5)
