"""Per-request latency metrics and goodput.

TTFT (time to first token) runs from arrival to the first output token reaching the CPU; TPOT (time
per output token) averages the gaps between the remaining tokens. Goodput counts only the requests
that meet both latency SLOs, as in DistServe: a faster engine that blows the SLO for most requests
has high throughput but low goodput.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepinfer_lab.sequence import Sequence


@dataclass
class RequestMetrics:
    ttft: float  # Seconds from arrival to the first output token.
    tpot: float  # Mean seconds per output token after the first; 0 for single-token outputs.
    latency: float  # Seconds from arrival to finish.
    num_output_tokens: int

    @classmethod
    def from_sequence(cls, seq: Sequence) -> RequestMetrics | None:
        """None for a request that never produced a token (ignored or aborted early)."""
        if seq.first_token_time is None or seq.finish_time is None:
            return None
        num_output = len(seq.output_token_ids)
        decode_time = seq.finish_time - seq.first_token_time
        return cls(
            ttft=seq.first_token_time - seq.arrival_time,
            tpot=decode_time / (num_output - 1) if num_output > 1 else 0.0,
            latency=seq.finish_time - seq.arrival_time,
            num_output_tokens=num_output,
        )


def summarize(
    metrics: list[RequestMetrics | None], elapsed: float, ttft_slo: float, tpot_slo: float
) -> dict[str, float]:
    """Aggregates request metrics over a run of `elapsed` seconds. Latencies are reported in ms."""
    done = [m for m in metrics if m is not None]
    if not done or elapsed <= 0:
        raise ValueError("no finished requests to summarize")
    ttft = np.array([m.ttft for m in done]) * 1e3
    tpot = np.array([m.tpot for m in done if m.num_output_tokens > 1] or [0.0]) * 1e3
    good = sum(m.ttft <= ttft_slo and m.tpot <= tpot_slo for m in done)
    return {
        "requests": len(done),
        "output_throughput": sum(m.num_output_tokens for m in done) / elapsed,
        "request_throughput": len(done) / elapsed,
        "goodput": good / elapsed,
        "slo_attainment": good / len(done),
        "ttft_p50_ms": float(np.percentile(ttft, 50)),
        "ttft_p99_ms": float(np.percentile(ttft, 99)),
        "tpot_p50_ms": float(np.percentile(tpot, 50)),
        "tpot_p99_ms": float(np.percentile(tpot, 99)),
    }
