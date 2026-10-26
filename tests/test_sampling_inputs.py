"""Packed sampling inputs preserve typed values, alignment, and the full integer seed range."""

import pytest
import torch

from deepinfer_lab.layers.sampler import Sampler, SamplingTensors
from deepinfer_lab.worker.model_runner import _sampling_to_gpu

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("n", [0, 1, 3, 64, 4096])
def test_sampling_input_bytes_and_alignment(n):
    temps = [float("nan"), -0.0, 0.3] * n
    temps = temps[:n]
    flags = [i % 2 == 0 for i in range(n)]
    top_k, top_p = [129280 - i for i in range(n)], [0.75] * n
    seeds = [(1 << 63) - 1 - i for i in range(n)]
    actual = _sampling_to_gpu(temps, flags, top_k, top_p, seeds)
    expected = dict(temperatures=(temps, torch.float32), greedy=(flags, torch.bool),
                    top_k=(top_k, torch.int64), top_p=(top_p, torch.float32), seeds=(seeds, torch.int64))
    for name, (values, dtype) in expected.items():
        reference = torch.tensor(values, dtype=dtype, device="cuda")
        assert actual[name].dtype == dtype and actual[name].shape == (n,) and actual[name].is_contiguous()
        if n:
            # Bit equality also checks NaN payloads and signed zero rather than numerical equality.
            assert torch.equal(actual[name].view(torch.uint8), reference.view(torch.uint8))
    if n:
        assert len({v.untyped_storage().data_ptr() for v in actual.values()}) == 1
        assert actual["seeds"].tolist() == seeds


def test_packed_inputs_preserve_mixed_filtered_sampling():
    logits = torch.randn(3, 4097, device="cuda")
    fields = dict(temperatures=[0.3, 1.0, 1.7], greedy=[False, True, False], top_k=[17, 4097, 64],
                  top_p=[0.8, 1.0, 0.9], seeds=[0, (1 << 63) - 1, 54321])
    packed = _sampling_to_gpu(fields["temperatures"], fields["greedy"], fields["top_k"], fields["top_p"], fields["seeds"])
    separate = {name: torch.tensor(values, dtype=packed[name].dtype, device="cuda") for name, values in fields.items()}
    flags = dict(all_greedy=False, need_top_k_top_p=True)
    sampler = Sampler()
    assert torch.equal(sampler(logits, SamplingTensors(**packed, **flags)),
                       sampler(logits, SamplingTensors(**separate, **flags)))
