# mHC 基础模块迁移

本轮接入 DeepSeek 官方 mHC 算子，建立可对照的推理基线。默认 `native` 路径保留原有
PyTorch / Triton 实现；`deepseek` 路径使用 DeepGEMM 和 TileKernels。两条路径共享 checkpoint
参数名、FP32 混合权重以及跨子层传递 pre-mix 的语义。本轮不引入 Mega mHC 或新的融合策略，
也不以迁移本身作为性能提升的证据。

## 依赖与运行

官方后端需要 Python 3.12+、兼容的 NVIDIA CUDA 环境、BF16 hidden states、`hc_mult=4`，
且 hidden size 是 128 的正整数倍。tiny checkpoint 的 hidden size 为 128，可运行此路径。
DeepGEMM 的 prenorm GEMM 支持 SM90 / SM100；本项目主要验证目标为 B200（SM100）。

```bash
# 已按项目说明安装 PyTorch 和 CUDA toolkit 后：
python -m pip install setuptools wheel packaging
python -m pip install --no-build-isolation '.[mhc]'
```

依赖固定为 TileKernels 2.1.0 和 DeepGEMM
[`5081f851`](https://github.com/deepseek-ai/DeepGEMM/tree/5081f8513af57e61696de4f4f04b0e707ffb503e)。
DeepGEMM 从源码构建还需要 C++20 编译器、CUDA 开发文件和 elfutils 开发头文件
（Debian / Ubuntu 的 `libdw-dev`、`libelf-dev`）。缺少依赖或输入不兼容时，显式选择的
`deepseek` 后端会报错，避免把参考实现的运行误记为官方后端结果。

```python
from deepinfer_lab import LLM

llm = LLM("tiny-random/deepseek-v4.1-flash-1017", mhc_backend="deepseek")
```

benchmark 可分别指定 `bench.py --mhc-backend deepseek` 和
`bench_prefill.py --mhc_backend deepseek`，结果记录中包含后端选择。

## 算子与数据契约

运行时接口在 `deepinfer_lab/models/deepseek_v41/mhc.py`。token batch 始终保持扁平：

| 阶段 | 官方算子 | 输入 / 输出 |
| --- | --- | --- |
| 初始 residual 展开 | `expand_to_mhc_fwd` | `[T, D]` BF16 → `[T, 4, D]` BF16 |
| prenorm GEMM | `tf32_hc_prenorm_gemm` | BF16 residual × FP32 `fn`，输出 FP32 投影与平方和 |
| RMS 归一化 | `mhc_reduce_partials_and_rmsnorm_fwd` | 分母覆盖整个 `4 * D`，使用模型的 `norm_eps` |
| 拆分混合权重 | `mhc_pre_split_mixes_fwd` | pre / post `[T, 4]`，comb logits `[T, 16]`；post 系数为 2 |
| Sinkhorn | `mhc_sinkhorn_fwd` | `[T, 4, 4]` FP32，使用模型的迭代次数和 `hc_eps` |
| pre 混合 | `mhc_pre_apply_mix_fwd` | `[T, 4, D]` → `[T, D]`，FP32 累加后写 BF16 |
| post 混合 | `mhc_post_fwd` | sublayer 输出 + residual → 新 residual，写 BF16 |

GEMM 先使用一次不分割的调用，再交给官方 RMS 算子；后续可另做 split-K 实验。
post 算子要求四维输入，适配器仅添加一个大小为 1 的 batch 维，返回时移除它。
`comb[t, source, destination]` 的轴顺序不变。

V4.1 的 pre-mix 延迟一个子层使用：Attention 使用传入的 `pre_mix`，FFN 使用 Attention
计算的 pre-mix，FFN 计算的 pre-mix 返回给下一个 block；最后一个 pre-mix 用于模型输出
collapse。迁移不改成“本子层生成、本子层使用”。每次调用独立分配输出，避免后续 forward
覆盖仍被持有的混合系数。空 token batch 直接返回正确形状的空张量。

## 验证

```bash
pytest -q -m 'not gpu'
pytest -q tests/test_mhc_backend.py tests/test_hc.py tests/test_hc_mixing.py tests/test_v41.py
```

CPU 测试覆盖归一化范围、矩阵方向、FP32 累加、pre-mix 传递和非法后端。
GPU 测试比较完整官方算子链与参考实现，包含空 batch、单 token、不对齐的 token 数、
真实 hidden size 5120、非连续输入、CUDA Graph 更新输入与旧输出所有权。
系数与混合应用分别对照；整链另外要求相对 L2 误差不超过 0.5%，避免接近零的输出因
prenorm 系数的微小误差而出现不稳定的逐元素相对误差。端到端 logits 容差仍为 0.05。
`test_v41.py` 还对官方参考轨迹逐位置检查 logits，覆盖 eager、CUDA Graph、chunked prefill
和抢占。缺少可选依赖会明确 skip；skipped 不代表官方后端已经验证。

手动 GPU CI 的 `mhc_backend` 输入默认选择 `deepseek`，会安装可选依赖并先检查官方算子
能否导入，再运行全套测试。选择 `native` 可以只安装基础依赖；官方 mHC 专项用例此时可能
skip。GPU runner 需要提前具备上述构建工具与系统头文件。

上游接口依据：
[TileKernels mHC](https://github.com/deepseek-ai/TileKernels/tree/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/mhc)、
[DeepGEMM prenorm GEMM](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/apis/hyperconnection.hpp)。
