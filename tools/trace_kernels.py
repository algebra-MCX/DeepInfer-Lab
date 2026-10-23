"""Summarize GPU kernel time in a Chrome trace: per kernel name and per coarse category.

usage: trace_kernels.py TRACE.json[.gz] STEPS [TOP]
Times are divided by STEPS, giving ms per prefill step. Overlapping kernels on different streams
are summed, so the total can exceed the wall time; the busy time (union of intervals) is also printed.
"""

import gzip
import json
import re
import sys
from collections import defaultdict

path, steps = sys.argv[1], int(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
    trace = json.load(f)
events = trace["traceEvents"] if isinstance(trace, dict) else trace
kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
for e in kernels:
    e["ts"], e["dur"] = float(e["ts"]), float(e["dur"])
schedules = sorted(float(e["ts"]) for e in events if e.get("name") == "deepinfer:schedule")
if len(schedules) > steps:  # official merged trace: keep only the first STEPS (prefill) steps
    kernels = [e for e in kernels if e["ts"] < schedules[steps]]

CATEGORIES = [
    ("comm", r"nccl|all_?gather|reduce_?scatter|allreduce|deep_ep|dispatch|combine|nvshmem|Userbuffer"),
    ("moe_gemm", r"mega_moe|grouped|m_grouped|fp8_fp4_gemm|_grouped_mm|cutlass.*group|sm100.*group"),
    ("indexer", r"mqa_logits|topk|deep_select|sort|radix|bitonic|gather_topk|candidate"),
    ("attention", r"flash_mla|sparse_attention|_indexed_sparse|attn|mla"),
    ("dense_gemm", r"gemm|cutlass|cublas|nvjet|sm\d+_xmma|fp8_einsum|bf16_gemm|matmul"),
    ("dequant", r"dequant|_fp4_kernel|fp8_fake|(?<!no)cast|quant"),
    ("hc_norm", r"mhc|sinkhorn|rms|norm|hc_"),
    ("elementwise_copy", r"Memcpy|Memset|index|gather|scatter|copy|cat|elementwise|vectorized|reduce|where|fill"),
]


def category(name: str) -> str:
    for cat, pattern in CATEGORIES:
        if re.search(pattern, name, re.IGNORECASE):
            return cat
    return "other"


by_name, by_cat, counts = defaultdict(float), defaultdict(float), defaultdict(int)
intervals = []
for e in kernels:
    by_name[e["name"]] += e["dur"]
    counts[e["name"]] += 1
    by_cat[category(e["name"])] += e["dur"]
    intervals.append((e["ts"], e["ts"] + e["dur"]))

busy, end = 0.0, float("-inf")
for s, t in sorted(intervals):
    if t > end:
        busy += t - max(s, end)
        end = t
span = (max(t for _, t in intervals) - min(s for s, _ in intervals)) if intervals else 0
total = sum(by_name.values())
print(f"kernels {len(kernels)}  sum {total / 1e3 / steps:.1f} ms/step  busy {busy / 1e3 / steps:.1f} ms/step  span {span / 1e3 / steps:.1f} ms/step")
print("\n== by category (ms/step, % of summed kernel time)")
for cat, us in sorted(by_cat.items(), key=lambda kv: -kv[1]):
    print(f"{us / 1e3 / steps:10.2f}  {100 * us / total:5.1f}%  {cat}")
print(f"\n== top {top} kernels (ms/step, calls/step)")
for name, us in sorted(by_name.items(), key=lambda kv: -kv[1])[:top]:
    print(f"{us / 1e3 / steps:10.2f}  {counts[name] / steps:7.1f}  [{category(name)}] {name[:150]}")
