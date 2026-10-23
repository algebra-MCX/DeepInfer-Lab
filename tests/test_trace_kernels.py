"""CPU-only test of the kernel-time summary used for prefill profiles."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "trace_kernels.py"


def kernel(name, ts, dur):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur}


def summarize(tmp_path, events, steps):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return subprocess.run([sys.executable, str(SCRIPT), str(path), str(steps)], capture_output=True, text=True, check=True).stdout


def test_times_are_per_step_and_overlap_counts_once_as_busy(tmp_path):
    events = [kernel("_sparse_attention", 0, 3000), kernel("ncclDevKernel_AllGather", 1000, 1000)]
    out = summarize(tmp_path, events, steps=2)
    assert "sum 2.0 ms/step  busy 1.5 ms/step  span 1.5 ms/step" in out
    assert "1.50   75.0%  attention" in out and "0.50   25.0%  comm" in out


def test_merged_trace_keeps_only_the_first_steps(tmp_path):
    """The official trace appends decode steps after the profiled prefill steps."""
    schedules = [{"ph": "X", "cat": "user_annotation", "name": "deepinfer:schedule", "ts": t, "dur": 1} for t in (0, 100, 200)]
    events = schedules + [kernel("mega_moe", 10, 50), kernel("mega_moe", 110, 50), kernel("decode_only", 210, 50)]
    out = summarize(tmp_path, events, steps=2)
    assert "decode_only" not in out
    assert "kernels 2  sum 0.1 ms/step" in out
