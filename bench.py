"""Serving benchmark with random prompts: throughput, TTFT / TPOT percentiles and goodput.

Requests arrive all at once (`--request-rate inf`, offline throughput) or as a Poisson process.
Goodput counts the requests per second that meet both `--ttft-slo` and `--tpot-slo`.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from deepinfer_lab import LLM, SamplingParams
from deepinfer_lab.metrics import RequestMetrics, summarize
from tools.benchmark_record import build_record

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny-random/deepseek-v4.1-flash-1017")
parser.add_argument("--num-seqs", type=int, default=64)
parser.add_argument("--max-input-len", type=int, default=1024)
parser.add_argument("--max-output-len", type=int, default=256)
parser.add_argument("--kv-cache-gb", type=float, default=4.0)
parser.add_argument("--weight-cache-gb", type=float, default=0.0, help="bf16 cache budget for non-expert FP8 weights")
parser.add_argument("--request-rate", type=float, default=float("inf"), help="Poisson arrival rate in requests/s")
parser.add_argument("--ttft-slo", type=float, default=2.0, help="seconds")
parser.add_argument("--tpot-slo", type=float, default=0.1, help="seconds")
parser.add_argument("--no-zos", action="store_true", help="disable zero-overhead scheduling")
parser.add_argument("--result-json", help="append summary, request timelines/tokens and run metadata as one JSON line")
args = parser.parse_args()

rng = random.Random(0)
prompt_token_ids = [[rng.randint(100, 120000) for _ in range(rng.randint(100, args.max_input_len))] for _ in range(args.num_seqs)]
sampling_params = [
    SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=rng.randint(16, args.max_output_len)) for _ in range(args.num_seqs)
]
gaps = np.random.default_rng(0).exponential(1 / args.request_rate, args.num_seqs) if args.request_rate < float("inf") else np.zeros(args.num_seqs)
arrivals = np.cumsum(gaps)

startup_start = time.perf_counter()
llm = LLM(args.model, max_num_seqs=args.num_seqs, max_model_len=4096, enable_zos=not args.no_zos,
          kv_cache_memory_gb=args.kv_cache_gb, weight_cache_memory_gb=args.weight_cache_gb)
llm.generate(prompt_token_ids=[[100] * 16], sampling_params=SamplingParams(max_tokens=8), use_tqdm=False)
startup_s = time.perf_counter() - startup_start
startup_memory = dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
torch.cuda.reset_peak_memory_stats()

engine = llm.llm_engine
seqs = []
start = time.perf_counter()
while len(seqs) < args.num_seqs or engine.has_unfinished_requests():
    now = time.perf_counter() - start
    while len(seqs) < args.num_seqs and arrivals[len(seqs)] <= now:
        i = len(seqs)
        seq = engine.add_request(prompt_token_ids[i], sampling_params[i])
        seq.arrival_time = start + arrivals[i]  # Count the delay until the engine got to admit it.
        seqs.append(seq)
    if engine.has_unfinished_requests():
        engine.step()
    else:
        time.sleep(max(0.0, arrivals[len(seqs)] - now))
elapsed = time.perf_counter() - start

summary = summarize([RequestMetrics.from_sequence(seq) for seq in seqs], elapsed, args.ttft_slo, args.tpot_slo)
summary.update(
    zos=not args.no_zos,
    request_rate=args.request_rate,
    input_tokens=sum(map(len, prompt_token_ids)),
    preempted=engine.scheduler.num_preempted,
    elapsed_s=elapsed,
)
print(
    f"zos={summary['zos']} rate={args.request_rate}: {summary['requests']} requests in {elapsed:.2f}s; "
    f"output {summary['output_throughput']:.0f} tok/s, "
    f"TTFT p50/p99 {summary['ttft_p50_ms']:.0f}/{summary['ttft_p99_ms']:.0f} ms, "
    f"TPOT p50/p99 {summary['tpot_p50_ms']:.1f}/{summary['tpot_p99_ms']:.1f} ms, "
    f"goodput {summary['goodput']:.2f} req/s ({100 * summary['slo_attainment']:.0f}% within SLO), "
    f"preempted {summary['preempted']}"
)
if args.result_json:
    repo = str(Path(__file__).resolve().parent)
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", repo, "diff-index", "--quiet", "HEAD", "--"], capture_output=True)
    model_path = Path(engine.config.model_path)
    metadata = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
        configuration={key: value for key, value in vars(args).items() if key != "result_json"},
        engine={key: getattr(engine.config, key) for key in (
            "max_model_len", "max_num_seqs", "max_num_batched_tokens", "seed", "enable_zos", "enable_cuda_graph"
        )},
        random_seed=0, python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in (
            "torch", "triton", "numpy", "transformers", "safetensors", "tokenizers", "huggingface_hub"
        )},
        commit=commit.stdout.strip() if commit.returncode == 0 else None,
        dirty=bool(dirty.returncode) if dirty.returncode in (0, 1) else None,
        model_revision=model_path.name if model_path.parent.name == "snapshots" else None,
        model_dtype=str(engine.config.dtype), cuda_runtime=torch.version.cuda,
        weight_cache_bytes=engine.model_runner.weight_cache_bytes,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        gpu_name=torch.cuda.get_device_name(), gpu_total_memory=torch.cuda.get_device_properties(0).total_memory,
        startup_peak_memory_bytes=startup_memory,
        serving_peak_memory_bytes=dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()),
        memory_accounting="PyTorch CUDA allocator peaks, including resident weights/cache; driver-managed memory excluded",
        startup_s=startup_s, startup_includes="model loading, graph capture and warmup",
        decode_graph_count=len(engine.model_runner.graphs.graphs) if engine.model_runner.graphs is not None else 0,
        warmup=dict(prompt_tokens=16, output_tokens=8, excluded_from_elapsed=True),
    )
    with open(args.result_json, "a") as f:
        f.write(json.dumps(build_record(summary, seqs, start, metadata), allow_nan=False) + "\n")
