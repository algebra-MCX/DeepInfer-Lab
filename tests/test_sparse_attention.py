"""Gathered attention keeps masked rows, BF16 rounding, and graph replay semantics."""

import pytest
import torch

from minideepinfer.models.deepseek_v41.model import sparse_attention

pytestmark = pytest.mark.gpu


# Head blocks of 16 or 32 and key tiles of 32: partial head blocks (33, 40), partial and single
# key tiles (1, 31, 33), and the real V4.1 shape (64 heads over 128 window + 512 compressed keys).
@pytest.mark.parametrize("heads,keys,dim", [(1, 1, 33), (3, 17, 33), (4, 129, 512), (33, 31, 64),
                                            (40, 33, 128), (64, 640, 512)])
def test_sparse_attention_matches_cpu(heads, keys, dim):
    gen = torch.Generator().manual_seed(47)
    q = torch.randn(7, heads, dim, generator=gen).bfloat16()
    kv = torch.randn(7, keys, dim, generator=gen).bfloat16()
    valid = torch.rand(7, keys, generator=gen) > 0.2
    valid[0] = False
    sink = torch.randn(heads, generator=gen)
    expected = sparse_attention(q, kv, valid, sink, dim**-0.5)
    actual = sparse_attention(q.cuda(), kv.cuda(), valid.cuda(), sink.cuda(), dim**-0.5)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert actual[0].count_nonzero() == 0


def test_sparse_attention_rounds_unnormalized_probabilities():
    """Removing BF16 probability rounding or including the sink in max changes this result."""
    q = torch.tensor([[[-0.005279541015625]]], device="cuda", dtype=torch.bfloat16)
    kv = torch.tensor([[[1.5625], [0.11767578125], [0.5234375]]], device="cuda", dtype=q.dtype)
    valid = torch.ones(1, 3, device="cuda", dtype=torch.bool)
    sink = torch.tensor([2.7068405151367188], device="cuda")
    actual = sparse_attention(q, kv, valid, sink, 1.0)
    assert actual.item() == 0.12158203125


def test_sparse_attention_empty_and_strided():
    q = torch.randn(3, 2, 66, device="cuda", dtype=torch.bfloat16)[..., ::2]
    kv = torch.randn(3, 17, 66, device="cuda", dtype=q.dtype)[..., ::2]
    valid = torch.ones(3, 17, device="cuda", dtype=torch.bool)
    sink = torch.randn(2, device="cuda")
    expected = sparse_attention(q.cpu(), kv.cpu(), valid.cpu(), sink.cpu(), 0.2)
    actual = sparse_attention(q, kv, valid, sink, 0.2)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert sparse_attention(q[:0].contiguous(), kv[:0].contiguous(), valid[:0], sink, 0.2).shape == (0, 2, 33)


def test_sparse_attention_graph_reads_updated_inputs():
    q = torch.randn(7, 4, 33, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(7, 17, 33, device="cuda", dtype=q.dtype)
    valid = torch.ones(7, 17, device="cuda", dtype=torch.bool)
    sink = torch.randn(4, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sparse_attention(q, kv, valid, sink, 0.2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = sparse_attention(q, kv, valid, sink, 0.2)
    torch.cuda.current_stream().wait_stream(stream)
    q.mul_(0.5)
    kv.neg_()
    valid[0] = False
    valid[1, 1:] = False
    sink.add_(2)
    graph.replay()
    expected = sparse_attention(q.cpu(), kv.cpu(), valid.cpu(), sink.cpu(), 0.2)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert actual[0].count_nonzero() == 0
