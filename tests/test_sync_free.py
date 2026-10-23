"""The model forward must not synchronize with the host: ZOS overlap and CUDA graphs depend on it."""

import pytest
import torch

from tests.conftest import MODEL

pytestmark = pytest.mark.gpu


def test_forward_has_no_host_sync(monkeypatch):
    from minideepinfer import LLM, SamplingParams

    # Eager decode is checked here; captured graphs could not contain a sync in the first place.
    llm = LLM(
        MODEL,
        max_num_seqs=4,
        max_model_len=512,
        max_num_batched_tokens=200,
        kv_cache_memory_gb=0.5,
        enable_cuda_graph=False,
    )
    model = llm.llm_engine.model_runner.model
    forward, calls = model.forward, []

    def strict_forward(batch):
        calls.append(batch.positions.numel())
        torch.cuda.set_sync_debug_mode("error")
        try:
            return forward(batch)
        finally:
            torch.cuda.set_sync_debug_mode(0)

    sp = SamplingParams(max_tokens=9, ignore_eos=True)
    llm.generate(prompt_token_ids=[[1, 2, 3]], sampling_params=sp, use_tqdm=False)  # one-time device constants
    monkeypatch.setattr(model, "forward", strict_forward)
    # 333 tokens with a 200-token budget: a chunked prefill whose chunks end mid compression group;
    # odd and even prompt lengths put decode positions on both sides of a group boundary.
    prompts = [list(range(100, 433)), [7], list(range(500, 629))]
    outputs = llm.generate(prompt_token_ids=prompts, sampling_params=sp, use_tqdm=False)
    assert all(len(out.token_ids) == 9 for out in outputs)
    assert max(calls) == 200 and min(calls) < 4  # both chunked prefill and decode steps ran strictly
