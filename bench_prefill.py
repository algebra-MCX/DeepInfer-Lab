"""Prefill throughput sweep, mirroring DeepInfer's `examples/sweep/run_sweep.sh prefill`.

Same experiment as the official prefill Pareto plot: EP x DP over all launched ranks, every request has
`context_length` tokens of which all but the last `non_cached_tokens` are already cached (prefilled once,
then forked by sharing its KV blocks), and each sweep point fixes the prefill tokens per step per rank.
Request counts, warm-up / measurement windows and metrics follow `examples/benchmark.py` with PROFILE=0,
and results are written in its jsonl format, so DeepInfer's `plot_pareto.py` can plot them.

Not reproduced: EPLB redundant experts, forced load balance, decoder bounded replay, and the decode
phase (requests stop after their first token).

    torchrun --nproc-per-node 4 bench_prefill.py --model /path/to/DeepSeek-V4.1-Flash --result_jsonl out.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import time
from collections import deque

import torch
import torch.distributed as dist
from torch import nn

from minideepinfer import LLM, SamplingParams
from minideepinfer.models.deepseek_v41 import model as v41
from minideepinfer.sequence import Sequence
from tools.benchmark_metrics import summarize_prefill_window

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--context_length", type=int, default=131072)
parser.add_argument("--non_cached_tokens", type=int, default=4096)
parser.add_argument("--prefill_batch_size", type=int, nargs="+", default=[4096, 8192, 16384])
parser.add_argument("--decode_num_requests", type=int, default=64, help="only enters the request count, as upstream")
parser.add_argument("--setup_chunk", type=int, default=16384, help="prefill tokens per step for the cached prefix")
parser.add_argument("--kv_cache_gb", type=float, default=12.0)
parser.add_argument("--result_jsonl", default=None)
parser.add_argument("--workload_json", default=None, help="shared JSON[.gz] prompts, one full prompt per DP rank")
parser.add_argument("--no_cuda_graph", action="store_true", help="skip decode graph capture for prefill-only comparisons")
parser.add_argument("--max_model_len", type=int, default=None)
parser.add_argument("--profile_dir", default=None, help="trace PROFILE_STEPS steps after warm-up into this directory")
args = parser.parse_args()

WARMUP_STEPS, MEASURE_STEPS, PROFILE_STEPS = 5, 10, 2
TOTAL_STEPS = WARMUP_STEPS + 2 * (MEASURE_STEPS + 1) + 2 * PROFILE_STEPS  # upstream's prefill_total_steps


def num_requests(point: int) -> int:
    prefill = min(math.ceil(TOTAL_STEPS * point / args.non_cached_tokens), 512)
    return max(prefill, args.decode_num_requests)


local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
world_size, rank = dist.get_world_size(), dist.get_rank()


def log(msg: str) -> None:
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


cached_tokens = args.context_length - args.non_cached_tokens
start = time.perf_counter()
llm = LLM(
    args.model,
    max_num_seqs=max(map(num_requests, args.prefill_batch_size)),
    max_num_batched_tokens=max(args.prefill_batch_size + [args.setup_chunk]),
    max_model_len=args.max_model_len or args.context_length + 8,
    kv_cache_memory_gb=args.kv_cache_gb,
    enable_cuda_graph=not args.no_cuda_graph,
)
engine = llm.llm_engine
sched, bm, sm = engine.scheduler, engine.block_manager, engine.state_manager
cache, block_size = engine.model_runner.model.cache, engine.config.block_size
assert cached_tokens % block_size == 0, "the shared prefix must end on a block boundary"
log(f"loaded in {time.perf_counter() - start:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")

rng = random.Random(rank)
special = set(llm.tokenizer.added_tokens_decoder)
vocab = [t for t in range(engine.config.model_config.vocab_size) if t not in special]


def random_tokens(n: int) -> list[int]:
    return rng.choices(vocab, k=n)


# Prefill the shared prefix once, synchronously, and keep its blocks and recurrent state as a template.
start = time.perf_counter()
engine.config.enable_zos = False
sched.max_num_batched_tokens = args.setup_chunk
if args.workload_json:
    opener = gzip.open if args.workload_json.endswith(".gz") else open
    with opener(args.workload_json, "rt") as f:
        workload = json.load(f)
    assert workload["context_length"] == args.context_length
    assert workload["non_cached_tokens"] == args.non_cached_tokens
    shared_prompt = workload["prompt_token_ids_by_rank"][rank]
    assert len(shared_prompt) == args.context_length
else:
    shared_prompt = None
prefix = shared_prompt[:cached_tokens] if shared_prompt is not None else random_tokens(cached_tokens)
source = engine.add_request(prefix, SamplingParams(temperature=0, max_tokens=2, ignore_eos=True))
while not source.output_token_ids:
    engine.step()
prefix_blocks = list(source.block_table)
bm.pin(prefix_blocks)
template_slot = sm.allocate(1)[0]
cache.copy_slot(source.state_slot, template_slot)
engine.abort_request(source.request_id)
engine.config.enable_zos = True
dist.barrier()
log(f"prefilled the {cached_tokens}-token prefix in {time.perf_counter() - start:.0f}s")


def remaining_prefill_tokens() -> int:
    return sum(seq.get_len() - seq.num_cached_tokens for seq in sched.waiting)


def add_forks(count: int) -> None:
    for _ in range(count):
        sp = SamplingParams(temperature=0.3, top_k=1024, top_p=0.95, max_tokens=1, ignore_eos=True)
        sp.assign_seed(rng)
        prompt = shared_prompt if shared_prompt is not None else prefix + random_tokens(args.non_cached_tokens)
        seq = Sequence(list(prompt), sp, block_size)
        bm.pin(prefix_blocks)
        seq.block_table = prefix_blocks + bm.allocate(seq.num_blocks_for(seq.get_len()) - len(prefix_blocks))
        seq.state_slot = sm.allocate(1)[0]
        cache.copy_slot(template_slot, seq.state_slot)
        seq.num_cached_tokens = cached_tokens
        sched.add_sequence(seq)


PROFILED_MODULES = (v41.Attention, v41.Compressor, v41.Indexer, v41.Gate, v41.MoE, v41.GroupedExperts, v41.Expert, v41.Engram)


def profile_steps(point: int) -> None:
    """Traces PROFILE_STEPS engine steps, each module above as a labeled range.

    Writes one Chrome trace per rank and prints rank 0's per-label and per-kernel GPU time.
    """
    ranges: dict[nn.Module, torch.profiler.record_function] = {}

    def enter(module, _):
        ranges[module] = torch.profiler.record_function(type(module).__name__).__enter__()

    def leave(module, _, __):
        ranges.pop(module).__exit__(None, None, None)

    handles = []
    for module in engine.model_runner.model.modules():
        if isinstance(module, PROFILED_MODULES):
            handles += [module.register_forward_pre_hook(enter), module.register_forward_hook(leave)]
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        for _ in range(PROFILE_STEPS):
            with torch.profiler.record_function("engine.step"):
                engine.step()
        torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    os.makedirs(args.profile_dir, exist_ok=True)
    prof.export_chrome_trace(os.path.join(args.profile_dir, f"prefill{point}_rank{rank}.json"))
    if rank == 0:
        table = prof.key_averages().table(sort_by="device_time_total", row_limit=60, max_name_column_width=70)
        with open(os.path.join(args.profile_dir, f"prefill{point}_rank0.txt"), "w") as f:
            f.write(table)
        print(table, flush=True)


def run_point(point: int) -> dict:
    count = num_requests(point)
    sched.max_num_batched_tokens, sched.max_num_seqs = point, count
    add_forks(count)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    stable: deque[tuple[int, float]] = deque(maxlen=MEASURE_STEPS + 1)

    def step(measure: bool) -> None:
        before = remaining_prefill_tokens()
        engine.step()
        processed = before - remaining_prefill_tokens()
        assert processed > 0
        if measure:
            stable.append((processed, time.perf_counter()))

    for _ in range(WARMUP_STEPS):
        step(measure=False)
    if args.profile_dir:
        profile_steps(point)
    while remaining_prefill_tokens() >= (MEASURE_STEPS + 2) * point:
        step(measure=False)
    while remaining_prefill_tokens() > 0:
        step(measure=True)
    result = {
        "engine": "mini-deepinfer",
        "model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "ep_size": world_size,
        "dp_size": world_size,
        "cp": 1,
        "num_routed_experts": engine.config.model_config.n_routed_experts,
        "context_length": args.context_length,
        "non_cached_tokens": args.non_cached_tokens,
        "prefill_batch_size": point,
        "cached_tokens": cached_tokens,
        "batch_num_requests": count,
        "nextn": 0,
        "yoco_clip": 0,
        "num_extra_experts": 0,
        "mtp_varlen": False,
        "enable_cuda_graph": not args.no_cuda_graph,
        **summarize_prefill_window(stable),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    while engine.has_unfinished_requests():
        engine.step()
    dist.barrier()
    result["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return result


for point in args.prefill_batch_size:
    log(f"--- prefill_batch_size={point}: {num_requests(point)} requests")
    result = run_point(point)
    log(
        f"prefill throughput={result['prefill_tokens_per_s_per_gpu']:.1f} tokens/s/GPU, "
        f"latency={result['prefill_window_seconds'] / result['prefill_window_steps'] * 1000:.0f} ms/step, "
        f"peak memory={result['peak_memory_gib']:.1f} GiB"
    )
    if rank == 0 and args.result_jsonl:
        with open(args.result_jsonl, "a") as f:
            f.write(json.dumps(result) + "\n")

dist.destroy_process_group()
