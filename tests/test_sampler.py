import pytest
import torch

from minideepinfer.layers.sampler import Sampler, SamplingTensors, apply_top_k_top_p, gumbel_noise, mix_seed

pytestmark = pytest.mark.gpu
V = 1000


def tensors(n, seeds, temperature=1.0, top_k=V, top_p=1.0, greedy=False):
    return SamplingTensors(
        temperatures=torch.full((n,), temperature, device="cuda"),
        greedy=torch.full((n,), greedy, device="cuda"),
        top_k=torch.full((n,), top_k, device="cuda"),
        top_p=torch.full((n,), top_p, device="cuda"),
        seeds=torch.tensor(seeds, device="cuda"),
        all_greedy=greedy,
        need_top_k_top_p=top_k < V or top_p < 1.0,
    )


def test_greedy_is_argmax():
    logits = torch.randn(4, V, device="cuda")
    out = Sampler()(logits, tensors(4, [0] * 4, greedy=True))
    assert torch.equal(out, logits.argmax(-1))


def test_batch_invariant_and_deterministic():
    torch.manual_seed(0)
    logits = torch.randn(8, V, device="cuda")
    seeds = [mix_seed(s, p) for s, p in zip(range(8), range(100, 108))]
    sampler = Sampler()
    full = sampler(logits, tensors(8, seeds))
    assert torch.equal(full, sampler(logits, tensors(8, seeds)))
    # Each row sampled alone, or in reversed order, gives the same token.
    for i in range(8):
        assert sampler(logits[i : i + 1], tensors(1, [seeds[i]])).item() == full[i].item()
    rev = sampler(logits.flip(0), tensors(8, seeds[::-1]))
    assert torch.equal(rev.flip(0), full)


def test_distribution_matches_softmax():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], device="cuda").repeat(20000, 1)
    out = Sampler()(logits, SamplingTensors(
        temperatures=torch.ones(20000, device="cuda"),
        greedy=torch.zeros(20000, dtype=torch.bool, device="cuda"),
        top_k=torch.full((20000,), 4, device="cuda"),
        top_p=torch.ones(20000, device="cuda"),
        seeds=torch.tensor([mix_seed(7, p) for p in range(20000)], device="cuda"),
        all_greedy=False,
        need_top_k_top_p=False,
    ))
    freq = torch.bincount(out, minlength=4).float() / 20000
    torch.testing.assert_close(freq, logits[0].softmax(-1), atol=0.015, rtol=0)


def test_top_k_top_p_mask():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], device="cuda")
    masked = apply_top_k_top_p(logits, torch.tensor([2], device="cuda"), torch.tensor([1.0], device="cuda"))
    assert torch.isfinite(masked).tolist() == [[True, True, False, False, False]]
    probs = logits.softmax(-1)  # ~[0.64, 0.24, 0.09, ...]
    masked = apply_top_k_top_p(logits, torch.tensor([5], device="cuda"), torch.tensor([0.8], device="cuda"))
    assert torch.isfinite(masked).tolist() == [[True, True, False, False, False]], probs


@pytest.mark.parametrize("vocab", [1, 1000, 4097, 129280])
def test_fused_sampling_matches_materialized_scores(vocab):
    """Mixed greedy/stochastic rows and a final partial tile must use the same Philox draws and argmax."""
    torch.manual_seed(23)
    logits = torch.randn(vocab, 3, device="cuda").T  # strided rows
    t = tensors(3, [mix_seed(i, 11) for i in range(3)])
    t.greedy[1] = True
    t.temperatures.copy_(torch.tensor([0.3, 1.0, 1.7], device="cuda"))
    expected = torch.where(
        t.greedy, logits.argmax(-1),
        (logits / t.temperatures[:, None] + gumbel_noise(t.seeds, vocab)).argmax(-1),
    )
    assert torch.equal(Sampler()(logits, t), expected)


def test_fused_sampling_ties_and_nan_match_argmax():
    logits = torch.full((3, 8193), -float("inf"), device="cuda")
    logits[1, 4096] = logits[1, 8192] = float("inf")
    logits[2, 4097] = logits[2, 8192] = float("nan")
    t = tensors(3, [0, 1, 2])
    expected = (logits + gumbel_noise(t.seeds, logits.size(1))).argmax(-1)
    assert torch.equal(Sampler()(logits, t), expected)


def test_fused_sampler_graph_uses_current_parameters():
    logits = torch.randn(3, 4097, device="cuda")
    t = tensors(3, [1, 2, 3])
    sampler, stream = Sampler(), torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sampler(logits, t)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = sampler(logits, t)
    torch.cuda.current_stream().wait_stream(stream)
    logits.mul_(2)
    t.seeds.add_(17)
    t.temperatures.fill_(0.7)
    t.greedy[1] = True
    graph.replay()
    expected = torch.where(
        t.greedy, logits.argmax(-1),
        (logits / t.temperatures[:, None] + gumbel_noise(t.seeds, logits.size(1))).argmax(-1),
    )
    assert torch.equal(out, expected)
