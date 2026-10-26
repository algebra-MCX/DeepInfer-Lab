"""Lossless benchmark evidence: request timelines, tokens, metrics and run provenance."""

import math
from dataclasses import asdict

from deepinfer_lab.metrics import RequestMetrics


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def build_record(summary, seqs, start, metadata):
    """Copy a run into standard JSON values; keep requests excluded from aggregate metrics too."""
    requests = []
    for seq in seqs:
        metrics = RequestMetrics.from_sequence(seq)
        times = {
            name: None if timestamp is None else timestamp - start
            for name, timestamp in (
                ("arrival_s", seq.arrival_time), ("first_token_s", seq.first_token_time), ("finish_s", seq.finish_time)
            )
        }
        requests.append(dict(
            request_id=seq.request_id, status=seq.status.name,
            prompt_token_ids=list(seq.token_ids[:seq.prompt_len]), output_token_ids=list(seq.output_token_ids),
            sampling=asdict(seq.sampling_params), metrics=None if metrics is None else asdict(metrics), **times,
        ))
    return _json_safe(dict(summary, schema_version=1, metadata=metadata, request_metrics=requests))
