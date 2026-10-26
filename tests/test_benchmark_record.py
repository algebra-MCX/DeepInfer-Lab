"""Saved requests must be sufficient to recompute the benchmark's reported metrics."""

import json

from deepinfer_lab.metrics import RequestMetrics, summarize
from tools.benchmark_record import build_record
from tests.test_metrics import timed_seq


def test_record_round_trip_recomputes_summary_and_preserves_missing_requests():
    seqs = [timed_seq(10, 10.5, 11.5, 11), timed_seq(10, 10, 10, 1), timed_seq(10, None, 11, 0)]
    metrics = [RequestMetrics.from_sequence(seq) for seq in seqs]
    summary = summarize(metrics, 2, 2, 0.1)
    summary["request_rate"] = float("inf")
    metadata = {"configuration": {"request_rate": float("inf")}, "commit": "example"}
    record = json.loads(json.dumps(build_record(summary, seqs, 10, metadata), allow_nan=False))
    assert record["schema_version"] == 1 and record["request_rate"] == "inf"
    assert record["metadata"]["configuration"]["request_rate"] == "inf"
    assert summary["request_rate"] == float("inf") and metadata["configuration"]["request_rate"] == float("inf")
    saved = record["request_metrics"]
    assert len(saved) == 3 and saved[2]["metrics"] is None and saved[2]["first_token_s"] is None
    assert saved[1]["first_token_s"] == saved[1]["finish_s"] == 0
    assert saved[0]["prompt_token_ids"] == [1, 2, 3] and saved[0]["output_token_ids"] == [7] * 11
    assert saved[0]["request_id"] == seqs[0].request_id and saved[0]["arrival_s"] == 0
    assert saved[0]["sampling"]["seq_seed"] == 0 and saved[0]["sampling"]["max_tokens"] == 11
    restored = [None if r["metrics"] is None else RequestMetrics(**r["metrics"]) for r in saved]
    assert summarize(restored, 2, 2, 0.1) == {k: v for k, v in summary.items() if k != "request_rate"}


def test_record_copies_tokens_and_encodes_nonfinite_configuration():
    seq = timed_seq(10, 10, 11, 2)
    record = build_record({}, [seq], 10, {"values": [float("inf"), -float("inf"), float("nan")]})
    seq.token_ids.append(99)
    assert record["request_metrics"][0]["output_token_ids"] == [7, 7]
    assert record["metadata"]["values"] == ["inf", "-inf", "nan"]
    json.dumps(record, allow_nan=False)
