"""Bounded caches preserve BF16 values, checkpoint state and captured weight addresses on reload."""

import pytest
import torch

from deepinfer_lab.config import Config
from deepinfer_lab.models.deepseek_v41.model import DeepseekV41, QuantLinear


def model_weights():
    model = torch.nn.Module()
    model.large = QuantLinear(32, 64)  # 4096 cached bytes
    model.small = QuantLinear(32, 1)  # 64 cached bytes
    model.expert = QuantLinear(32, 2, fp4=True)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, QuantLinear):
                module.weight.copy_(torch.full(module.weight.shape, 2.0).to(module.weight.dtype))
                module.scale.copy_(torch.ones(module.scale.shape).to(module.scale.dtype))
    return model


def test_weight_cache_budget_and_expert_exclusion():
    model = model_weights()
    for budget, expected in [(0, 0), (63, 0), (64, 64), (4095, 64), (4160, 4160)]:
        used = DeepseekV41.cache_non_expert_weights(model, budget)
        assert used == expected <= budget
        assert model.expert._cached_weight is None
        for module in (model.large, model.small):
            if module._cached_weight is not None:
                assert torch.equal(module._cached_weight.view(torch.uint8), module._dequant().view(torch.uint8))
        assert not any("_cached_weight" in name for name in model.state_dict())
    assert DeepseekV41.cache_non_expert_weights(model, 0) == 0
    assert all(module._cached_weight is None for module in (model.large, model.small))


def test_loading_weights_refreshes_cache_without_changing_addresses():
    model = model_weights()
    DeepseekV41.cache_non_expert_weights(model, 4160)
    pointers = [module._cached_weight.data_ptr() for module in (model.large, model.small)]
    state = {name: torch.full(p.shape, 3.0 if name.endswith("weight") else 2.0).to(p.dtype)
             for name, p in model.named_parameters()}
    DeepseekV41.load_weights(model, state.items())
    for module, pointer in zip((model.large, model.small), pointers):
        assert module._cached_weight.data_ptr() == pointer
        assert torch.equal(module.dequant(), torch.full_like(module.dequant(), 6))


@pytest.mark.parametrize("budget", [-0.1, float("inf"), float("nan")])
def test_invalid_weight_cache_budget_fails_before_model_resolution(budget):
    with pytest.raises(ValueError, match="weight_cache_memory_gb"):
        Config("unused-invalid-budget-model", weight_cache_memory_gb=budget)


@pytest.mark.gpu
def test_graph_reads_refreshed_cached_weights():
    model = model_weights().cuda()
    DeepseekV41.cache_non_expert_weights(model, 4160)
    x = torch.ones(2, 32, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model.small(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = model.small(x)
    torch.cuda.current_stream().wait_stream(stream)
    state = {name: torch.full(p.shape, 3.0 if name.endswith("weight") else 2.0, device="cuda").to(p.dtype)
             for name, p in model.named_parameters()}
    DeepseekV41.load_weights(model, state.items())
    graph.replay()
    assert torch.equal(actual, torch.full_like(actual, 192))
