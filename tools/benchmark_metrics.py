"""Steady prefill metrics from completed-step boundaries."""

from collections.abc import Iterable
from itertools import pairwise


def summarize_prefill_window(samples: Iterable[tuple[int, float]]) -> dict:
    """The first timestamp starts the window; only subsequent steps have timed tokens."""
    intervals = [(tokens, end - start) for (_, start), (tokens, end) in pairwise(samples)]
    if not intervals or any(tokens <= 0 or seconds <= 0 for tokens, seconds in intervals):
        raise ValueError("prefill needs positive token counts and at least two increasing step timestamps")
    tokens = sum(n for n, _ in intervals)
    seconds = sum(dt for _, dt in intervals)
    return {
        "prefill_window_steps": len(intervals),
        "prefill_window_tokens": tokens,
        "prefill_window_seconds": seconds,
        "prefill_step_tokens": [n for n, _ in intervals],
        "prefill_step_seconds": [dt for _, dt in intervals],
        "prefill_tokens_per_s_per_gpu": tokens / seconds,
    }
