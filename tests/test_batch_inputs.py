"""Batch metadata transfer must preserve chunk boundaries and tokens still in flight."""

from types import SimpleNamespace

import pytest
import torch

from deepinfer_lab import SamplingParams
from deepinfer_lab.engine.scheduler import SchedulerOutput
from deepinfer_lab.sequence import Sequence
from deepinfer_lab.worker.model_runner import ModelRunner


@pytest.mark.gpu
@pytest.mark.parametrize("max_ngram", [0, 3])
@pytest.mark.parametrize("is_prompt", [False, True])
def test_batch_inputs_preserve_chunks_and_zos_remap(max_ngram, is_prompt):
    runner = ModelRunner.__new__(ModelRunner)
    runner.max_ngram = max_ngram
    a = Sequence([7, 8], SamplingParams(), 128)
    b = Sequence([11, 12, 13, 14] if is_prompt else [11, 12, 13], SamplingParams(), 128)
    a.num_cached_tokens, b.num_cached_tokens = 2, 4
    a.state_slot, b.state_slot = 1, 2
    a.block_table, b.block_table = [1], [2, 3]
    prev = SimpleNamespace(row_of={b.seq_id: 0}, sampled_gpu=torch.tensor([14], device="cuda"))
    sched = SchedulerOutput(is_prompt, [a, b], [2 if is_prompt else 1, 1], [(a, 0), (b, 0)])
    batch, sample_rows = runner._prepare(sched, None if is_prompt else prev)
    if is_prompt:
        expected = dict(
            input_ids=[7, 8, 14], positions=[0, 1, 3], token_seq=[0, 0, 1], slots=[1, 1, 2],
            chunk_start=[0, 0, 3], chunk_end=[2, 2, 4], block_tables=[[1, 0], [2, 3]],
        )
        assert batch.cu_seqlens == [0, 2, 3]
        assert sample_rows.tolist() == [1, 2]
        ngram_ids = [[7, -1, -1], [8, 7, -1], [14, 13, 12]]
    else:
        expected = dict(
            input_ids=[8, 14], positions=[1, 3], token_seq=[0, 1], slots=[1, 2],
            chunk_start=[1, 3], chunk_end=[2, 4], block_tables=[[1, 0], [2, 3]],
        )
        assert batch.cu_seqlens == [0, 1, 2]
        assert sample_rows.tolist() == [0, 1]
        ngram_ids = [[8, 7, -1], [14, 13, 12]]
    for name, values in expected.items():
        assert getattr(batch, name).tolist() == values, name
    assert batch.max_end == 4
    if max_ngram:
        assert batch.ngram_ids.tolist() == ngram_ids
    else:
        assert batch.ngram_ids is None


@pytest.mark.gpu
def test_unfinished_prompt_chunk_has_no_sample_rows():
    runner = ModelRunner.__new__(ModelRunner)
    runner.max_ngram = 0
    seq = Sequence([7, 8, 9], SamplingParams(), 128)
    seq.num_cached_tokens, seq.state_slot, seq.block_table = 2, 1, [1]
    batch, sample_rows = runner._prepare(SchedulerOutput(True, [seq], [2]), None)
    assert batch.input_ids.tolist() == [7, 8]
    assert sample_rows.dtype == torch.int64 and sample_rows.shape == (0,)
