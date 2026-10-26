#!/bin/bash
#SBATCH -J deepinfer-lab-bench-tiny
#SBATCH -p main
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/job-%j.log
set -uo pipefail
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="/workspace/DeepInfer-Lab-tidy/.venv/bin:$PATH"
echo 'perf/tensor-core-sparse-attention (f759f32196ff) run 1'
cd /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/deepinfer-lab/worktrees/f759f32196ff
python -c 'import deepinfer_lab, sys; sys.exit(not deepinfer_lab.__file__.startswith("/workspace/.cache/deepinfer-lab/worktrees/f759f32196ff/"))' || { echo 'deepinfer_lab not imported from /workspace/.cache/deepinfer-lab/worktrees/f759f32196ff' > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/f759f32196ff-run1.log; exit 1; }
timeout 1200s python bench.py  > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/f759f32196ff-run1.log 2>&1 || echo 'run failed: f759f32196ff-run1'
echo 'perf/fused-index-scores (1868b01b328c) run 1'
cd /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/deepinfer-lab/worktrees/1868b01b328c
python -c 'import deepinfer_lab, sys; sys.exit(not deepinfer_lab.__file__.startswith("/workspace/.cache/deepinfer-lab/worktrees/1868b01b328c/"))' || { echo 'deepinfer_lab not imported from /workspace/.cache/deepinfer-lab/worktrees/1868b01b328c' > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/1868b01b328c-run1.log; exit 1; }
timeout 1200s python bench.py  > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/1868b01b328c-run1.log 2>&1 || echo 'run failed: 1868b01b328c-run1'
echo 'perf/tensor-core-sparse-attention (f759f32196ff) run 2'
cd /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/deepinfer-lab/worktrees/f759f32196ff
python -c 'import deepinfer_lab, sys; sys.exit(not deepinfer_lab.__file__.startswith("/workspace/.cache/deepinfer-lab/worktrees/f759f32196ff/"))' || { echo 'deepinfer_lab not imported from /workspace/.cache/deepinfer-lab/worktrees/f759f32196ff' > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/f759f32196ff-run2.log; exit 1; }
timeout 1200s python bench.py  > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/f759f32196ff-run2.log 2>&1 || echo 'run failed: f759f32196ff-run2'
echo 'perf/fused-index-scores (1868b01b328c) run 2'
cd /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/deepinfer-lab/worktrees/1868b01b328c
python -c 'import deepinfer_lab, sys; sys.exit(not deepinfer_lab.__file__.startswith("/workspace/.cache/deepinfer-lab/worktrees/1868b01b328c/"))' || { echo 'deepinfer_lab not imported from /workspace/.cache/deepinfer-lab/worktrees/1868b01b328c' > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/1868b01b328c-run2.log; exit 1; }
timeout 1200s python bench.py  > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/1868b01b328c-run2.log 2>&1 || echo 'run failed: 1868b01b328c-run2'
echo 'perf/tensor-core-sparse-attention (f759f32196ff) run 3'
cd /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/deepinfer-lab/worktrees/f759f32196ff
python -c 'import deepinfer_lab, sys; sys.exit(not deepinfer_lab.__file__.startswith("/workspace/.cache/deepinfer-lab/worktrees/f759f32196ff/"))' || { echo 'deepinfer_lab not imported from /workspace/.cache/deepinfer-lab/worktrees/f759f32196ff' > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/f759f32196ff-run3.log; exit 1; }
timeout 1200s python bench.py  > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/f759f32196ff-run3.log 2>&1 || echo 'run failed: f759f32196ff-run3'
echo 'perf/fused-index-scores (1868b01b328c) run 3'
cd /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/harness && export PYTHONPATH=/workspace/.cache/deepinfer-lab/worktrees/1868b01b328c
python -c 'import deepinfer_lab, sys; sys.exit(not deepinfer_lab.__file__.startswith("/workspace/.cache/deepinfer-lab/worktrees/1868b01b328c/"))' || { echo 'deepinfer_lab not imported from /workspace/.cache/deepinfer-lab/worktrees/1868b01b328c' > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/1868b01b328c-run3.log; exit 1; }
timeout 1200s python bench.py  > /workspace/deepinfer-lab-experiments/bench-rev-tiny-2/1868b01b328c-run3.log 2>&1 || echo 'run failed: 1868b01b328c-run3'
