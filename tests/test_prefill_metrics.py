import pytest

from tools.benchmark_metrics import summarize_prefill_window


def test_prefill_window_counts_only_timed_tokens():
    result = summarize_prefill_window([(999, 10.0), (4096, 10.25), (1024, 10.75)])
    assert result["prefill_window_steps"] == 2
    assert result["prefill_window_tokens"] == 5120
    assert result["prefill_window_seconds"] == 0.75
    assert result["prefill_tokens_per_s_per_gpu"] == pytest.approx(5120 / 0.75)
    assert result["prefill_step_tokens"] == [4096, 1024]
    assert result["prefill_step_seconds"] == [0.25, 0.5]


@pytest.mark.parametrize("samples", [[], [(1, 0.0)], [(1, 0.0), (1, 0.0)]])
def test_incomplete_prefill_window_is_rejected(samples):
    with pytest.raises(ValueError, match="increasing"):
        summarize_prefill_window(samples)
