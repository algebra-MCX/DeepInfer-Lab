#!/bin/bash
#SBATCH -J mdi-bench-tiny
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o /workspace/mdi-experiments/bench-rev-tiny-2/job-%j.log
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="/workspace/mini-DeepInfer-tidy/.venv/bin:$PATH"
echo 'perf/tensor-core-sparse-attention (f759f32196ff) run 1'
cd /workspace/mdi-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/mini-deepinfer/worktrees/f759f32196ff
python -c 'import minideepinfer, sys; sys.exit(not minideepinfer.__file__.startswith("/workspace/.cache/mini-deepinfer/worktrees/f759f32196ff/"))' || { echo 'minideepinfer not imported from /workspace/.cache/mini-deepinfer/worktrees/f759f32196ff' > /workspace/mdi-experiments/bench-rev-tiny-2/f759f32196ff-run1.log; exit 1; }
timeout 1200s python bench.py  > /workspace/mdi-experiments/bench-rev-tiny-2/f759f32196ff-run1.log 2>&1 || echo 'run failed: f759f32196ff-run1'
echo 'perf/fused-index-scores (1868b01b328c) run 1'
cd /workspace/mdi-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/mini-deepinfer/worktrees/1868b01b328c
python -c 'import minideepinfer, sys; sys.exit(not minideepinfer.__file__.startswith("/workspace/.cache/mini-deepinfer/worktrees/1868b01b328c/"))' || { echo 'minideepinfer not imported from /workspace/.cache/mini-deepinfer/worktrees/1868b01b328c' > /workspace/mdi-experiments/bench-rev-tiny-2/1868b01b328c-run1.log; exit 1; }
timeout 1200s python bench.py  > /workspace/mdi-experiments/bench-rev-tiny-2/1868b01b328c-run1.log 2>&1 || echo 'run failed: 1868b01b328c-run1'
echo 'perf/tensor-core-sparse-attention (f759f32196ff) run 2'
cd /workspace/mdi-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/mini-deepinfer/worktrees/f759f32196ff
python -c 'import minideepinfer, sys; sys.exit(not minideepinfer.__file__.startswith("/workspace/.cache/mini-deepinfer/worktrees/f759f32196ff/"))' || { echo 'minideepinfer not imported from /workspace/.cache/mini-deepinfer/worktrees/f759f32196ff' > /workspace/mdi-experiments/bench-rev-tiny-2/f759f32196ff-run2.log; exit 1; }
timeout 1200s python bench.py  > /workspace/mdi-experiments/bench-rev-tiny-2/f759f32196ff-run2.log 2>&1 || echo 'run failed: f759f32196ff-run2'
echo 'perf/fused-index-scores (1868b01b328c) run 2'
cd /workspace/mdi-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/mini-deepinfer/worktrees/1868b01b328c
python -c 'import minideepinfer, sys; sys.exit(not minideepinfer.__file__.startswith("/workspace/.cache/mini-deepinfer/worktrees/1868b01b328c/"))' || { echo 'minideepinfer not imported from /workspace/.cache/mini-deepinfer/worktrees/1868b01b328c' > /workspace/mdi-experiments/bench-rev-tiny-2/1868b01b328c-run2.log; exit 1; }
timeout 1200s python bench.py  > /workspace/mdi-experiments/bench-rev-tiny-2/1868b01b328c-run2.log 2>&1 || echo 'run failed: 1868b01b328c-run2'
echo 'perf/tensor-core-sparse-attention (f759f32196ff) run 3'
cd /workspace/mdi-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/mini-deepinfer/worktrees/f759f32196ff
python -c 'import minideepinfer, sys; sys.exit(not minideepinfer.__file__.startswith("/workspace/.cache/mini-deepinfer/worktrees/f759f32196ff/"))' || { echo 'minideepinfer not imported from /workspace/.cache/mini-deepinfer/worktrees/f759f32196ff' > /workspace/mdi-experiments/bench-rev-tiny-2/f759f32196ff-run3.log; exit 1; }
timeout 1200s python bench.py  > /workspace/mdi-experiments/bench-rev-tiny-2/f759f32196ff-run3.log 2>&1 || echo 'run failed: f759f32196ff-run3'
echo 'perf/fused-index-scores (1868b01b328c) run 3'
cd /workspace/mdi-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/mini-deepinfer/worktrees/1868b01b328c
python -c 'import minideepinfer, sys; sys.exit(not minideepinfer.__file__.startswith("/workspace/.cache/mini-deepinfer/worktrees/1868b01b328c/"))' || { echo 'minideepinfer not imported from /workspace/.cache/mini-deepinfer/worktrees/1868b01b328c' > /workspace/mdi-experiments/bench-rev-tiny-2/1868b01b328c-run3.log; exit 1; }
timeout 1200s python bench.py  > /workspace/mdi-experiments/bench-rev-tiny-2/1868b01b328c-run3.log 2>&1 || echo 'run failed: 1868b01b328c-run3'
