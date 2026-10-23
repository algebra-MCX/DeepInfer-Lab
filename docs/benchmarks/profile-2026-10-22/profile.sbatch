#!/bin/bash
#SBATCH -J mdi-prefill-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH -o /workspace/mdi-experiments/feasibility-20261022/profile-%j.log
set -uo pipefail
OUT=/workspace/mdi-experiments/feasibility-20261022
RUN_DIR="$OUT/run-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"
MINI=/workspace/mini-DeepInfer-tidy
OFFICIAL=/workspace/DeepInfer
WORKLOAD=/workspace/mdi-experiments/real4-20261022/workload.json.gz
MODEL=/models/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EP_DISABLE_GIN=1 DEEPINFER_ENGRAM_NVLINK_FETCH=1 DEEPINFER_ENGRAM_FORCE_GPU_STORAGE=1
export DEEPINFER_ALLOCATED_DEVICES="$CUDA_VISIBLE_DEVICES"
export DEEPINFER_USE_NEXTN_DECODE=0 DEEPINFER_ENABLE_MTP_VARLEN=0
export DEEPINFER_ENFORCE_EAGER=1 DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS=0
export BENCH_FORCE_LOAD_BALANCE=0 BENCH_LB_STAT_INTERVAL=1000000
export MASTER_ADDR=127.0.0.1
git -C "$MINI" rev-parse HEAD > "$RUN_DIR/mini-head.txt"
git -C "$OFFICIAL" rev-parse HEAD > "$RUN_DIR/official-head.txt"

cd "$MINI"
echo "mini start $(date -Is)"
timeout --kill-after=30s 720s .venv/bin/torchrun --nnodes=1 --nproc-per-node=4 \
    --master-addr=127.0.0.1 --master-port=$((35000 + SLURM_JOB_ID % 10000)) \
    bench_prefill.py --model "$MODEL" --prefill_batch_size 4096 --setup_chunk 4096 \
    --context_length 131072 --non_cached_tokens 4096 --max_model_len 131328 --no_cuda_graph \
    --workload_json "$WORKLOAD" --result_jsonl "$RUN_DIR/mini.jsonl" --profile_dir "$RUN_DIR/mini-trace" \
    > "$RUN_DIR/mini.log" 2>&1
echo "mini exit $?"

cd "$OFFICIAL"
echo "official start $(date -Is)"
export WORLD_SIZE=1 RANK=0 MASTER_PORT=$((45000 + SLURM_JOB_ID % 10000)) PROFILE=1
timeout --kill-after=30s 900s .venv/bin/python examples/benchmark.py \
    --model_name "$MODEL" --cp 1 --prefill_batch_size 4096 --decode_num_requests 64 \
    --context_length 131072 --non_cached_tokens 4096 --max_model_len 131328 --yoco_clip 0 \
    --workload_json "$WORKLOAD" --result_jsonl "$RUN_DIR/official.jsonl" --tracing_path "$RUN_DIR/official-trace" \
    > "$RUN_DIR/official.log" 2>&1
echo "official exit $?"
