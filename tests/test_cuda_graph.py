"""Decode CUDA graphs against the eager forward."""

import pytest
import torch

from minideepinfer.worker.cuda_graph import build_context_buckets, build_graph_batch_sizes
from tests.conftest import MODEL

LOGIT_ATOL = 0.05  # as tests/test_v41.py: bf16 accumulation-order noise


def test_graph_sizes():
    assert build_graph_batch_sizes(64) == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    assert build_graph_batch_sizes(5) == [1, 2, 4, 5]
    assert build_graph_batch_sizes(1) == [1]
    assert build_context_buckets(4096) == [256, 512, 1024, 2048, 4096]
    assert build_context_buckets(300) == [256, 300]
    assert build_context_buckets(100) == [100]


def run(enable_cuda_graph, prompts, max_tokens):
    from minideepinfer import LLM, SamplingParams

    llm = LLM(MODEL, max_num_seqs=8, max_model_len=640, kv_cache_memory_gb=0.5, enable_cuda_graph=enable_cuda_graph)
    runner = llm.llm_engine.model_runner
    logits, graph_steps, execute = {}, [], runner.execute

    def capture(sched, prev):
        handle = execute(sched, prev)
        if not sched.is_prompt:
            graph_steps.append(len(sched.seqs))
        for row, (seq, _) in enumerate(sched.sample_seqs):
            logits[tuple(seq.token_ids[: seq.prompt_len]), seq.num_cached_tokens] = handle.logits[row].float().cpu()
        return handle

    runner.execute = capture
    sps = [SamplingParams(temperature=0, max_tokens=n, ignore_eos=True) for n in max_tokens]
    outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sps, use_tqdm=False)
    used_graphs = runner.graphs is not None
    del llm
    torch.cuda.empty_cache()
    return outputs, logits, graph_steps, used_graphs


@pytest.mark.gpu
def test_graph_decode_matches_eager():
    """Requests finish at different steps, so the batch shrinks 5 -> 1 through padded graph sizes
    (8, 4, 2, 1) and rows that held a live sequence become padding: a stale padding row would
    corrupt another sequence's state. Contexts cross the 256-token bucket boundary."""
    prompts = [list(range(100 + i, 100 + i + n)) for i, n in enumerate([5, 130, 250, 300, 1])]
    max_tokens = [3, 40, 20, 12, 33]
    eager, eager_logits, _, used = run(False, prompts, max_tokens)
    assert not used
    graph, graph_logits, decode_sizes, used = run(True, prompts, max_tokens)
    assert used and {5, 3, 2, 1} <= set(decode_sizes)
    for e, g in zip(eager, graph):
        for step, (te, tg) in enumerate(zip(e.token_ids, g.token_ids)):
            key = (tuple(e.prompt_token_ids), len(e.prompt_token_ids) + step)
            diff = (eager_logits[key] - graph_logits[key]).abs().max().item()
            assert diff < LOGIT_ATOL, (len(e.prompt_token_ids), step, diff)
            top2 = eager_logits[key].topk(2).values
            if top2[0] - top2[1] < 2 * LOGIT_ATOL:
                break  # a near tie may flip; trajectories can diverge from here
            assert te == tg, (len(e.prompt_token_ids), step)


@pytest.mark.gpu
def test_rows_past_the_batch_are_reset_to_padding():
    """A row a previous, larger step used must not keep its sequence's slot and blocks: replaying it
    would write into storage that may by then belong to (or be shared with) another sequence."""
    from minideepinfer import LLM
    from minideepinfer.models.deepseek_v41.model import V41Batch

    llm = LLM(MODEL, max_num_seqs=8, max_model_len=640, kv_cache_memory_gb=0.5)
    graphs = llm.llm_engine.model_runner.graphs

    def decode_batch(n):
        dev = "cuda"
        ngram = torch.full((n, graphs.static["ngram_ids"].size(1)), -1, device=dev)
        ngram[:, 0] = 7
        return V41Batch(
            input_ids=torch.full((n,), 7, device=dev),
            positions=torch.full((n,), 9, device=dev),
            token_seq=torch.arange(n, device=dev),
            slots=torch.arange(1, n + 1, device=dev),
            chunk_start=torch.full((n,), 9, device=dev),
            chunk_end=torch.full((n,), 10, device=dev),
            block_tables=torch.arange(1, n + 1, device=dev)[:, None],
            ngram_ids=ngram,
            cu_seqlens=list(range(n + 1)),
            max_end=10,
        )

    assert graphs.run(decode_batch(4)).shape[0] == 4
    assert graphs.static["slots"][3] == 4 and graphs.static["block_tables"][3, 0] == 4
    assert graphs.run(decode_batch(3)).shape[0] == 3  # replays the batch-4 graph with one padding row
    for name, static in graphs.static.items():
        assert torch.equal(static[3], graphs.padding[name][3]), name
