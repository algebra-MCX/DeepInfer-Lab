"""Indexer scoring paths against a per-query reference."""

import pytest
import torch

from deepinfer_lab.models.deepseek_v41.model import index_scores

pytestmark = pytest.mark.gpu
H, D = 4, 128


def reference(q, keys, weights, cu_seqlens):
    out = []
    for b in range(len(cu_seqlens) - 1):
        for t in range(cu_seqlens[b], cu_seqlens[b + 1]):
            s = (q[t].float() @ keys[b].float().T).relu()  # [H, n]
            out.append((s * weights[t, :, None].float()).sum(0))
    return torch.stack(out)


def rows_to_seq(seqlens):
    return torch.repeat_interleave(torch.arange(len(seqlens)), torch.tensor(seqlens)).cuda()


# "short-40": 64-row blocks that span up to 33 sequences of the fused kernel.
@pytest.mark.parametrize("fused", [False, True], ids=["sliced", "fused"])
@pytest.mark.parametrize("seqlens", [[1], [1] * 37, [3, 1, 600, 2], [2] * 40],
                         ids=["decode-1", "decode-37", "prefill", "short-40"])
def test_index_scores_match_reference(seqlens, fused):
    torch.manual_seed(0)
    T, B, n = sum(seqlens), len(seqlens), 300
    q = torch.randn(T, H, D, device="cuda").bfloat16()
    keys = torch.randn(B, n, D, device="cuda").bfloat16()
    weights = torch.randn(T, H, device="cuda").bfloat16() * H**-0.5
    cu_seqlens = torch.tensor([0, *seqlens]).cumsum(0).tolist()
    expected = reference(q, keys, weights, cu_seqlens)
    actual = index_scores(q, keys, weights, cu_seqlens, rows_to_seq(seqlens) if fused else None)
    assert actual.shape == (T, n) and actual.dtype == torch.bfloat16
    # bf16 products and output; the reference accumulates in fp32
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=0.25)


def test_decode_rows_independent_of_batch():
    torch.manual_seed(1)
    q = torch.randn(8, H, D, device="cuda").bfloat16()
    keys = torch.randn(8, 64, D, device="cuda").bfloat16()
    weights = torch.randn(8, H, device="cuda").bfloat16()
    full = index_scores(q, keys, weights, list(range(9)))
    single = index_scores(q[3:4], keys[3:4], weights[3:4], [0, 1])
    torch.testing.assert_close(full[3:4], single, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("n", [1, 129, 1000])
def test_fused_scores_at_real_head_shape(n):
    """V4.1's 32 x 128 indexer heads; query blocks of 64 and entry blocks of 128 left partial."""
    torch.manual_seed(2)
    seqlens = [65, 130]
    T, H32 = sum(seqlens), 32
    q = torch.randn(T, H32, D, device="cuda").bfloat16()
    keys = torch.randn(len(seqlens), n, D, device="cuda").bfloat16()
    weights = (torch.randn(T, H32, device="cuda") * (H32 * D) ** -0.5).bfloat16()
    cu_seqlens = [0, 65, 195]
    actual = index_scores(q, keys, weights, cu_seqlens, rows_to_seq(seqlens))
    torch.testing.assert_close(actual.float(), reference(q, keys, weights, cu_seqlens), rtol=2e-2, atol=2e-2)
