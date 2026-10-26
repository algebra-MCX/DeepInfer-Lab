<h1 align="center">mini-DeepInfer</h1>

**mini-DeepInfer 的核心目标，是探究 DeepInfer 各个组件对整个推理引擎的影响，并进行量化分析。** 对我而言，项目最珍贵的部分就是完整、公开的优化过程：在 [Issues](https://github.com/algebra-MCX/mini-DeepInfer/issues) 中，你可以看到我的每一项优化，包括遇到了什么问题、设计思路是什么、采取了哪些解决措施，以及最终在真实 **4×B200** 上的端到端（E2E）真机测试。当前仅支持 **DeepSeek-V4.1-Flash**，实验聚焦 **Prefill**，通过组件分析与整机测试理解每项优化对引擎整体表现的影响。

我也是 **vLLM 贡献者**，在 **Model Runner V2（MRV2）** 的两个 PR 已合入主线（[#63372](https://github.com/vllm-project/vllm/pull/63372)、[#63476](https://github.com/vllm-project/vllm/pull/63476)），最近发起了「**vLLM 晋升挑战**」（详情见[我的小红书主页](https://www.xiaohongshu.com/user/profile/6ad229240000000005006800)），并与 **Inferact** 团队保持着良好的关系。如果项目反响良好，我会积极寻求与 Inferact 合作，争取借用 **64 张 SM100 GPU 的集群**，进一步开展 **Decode 优化**。

如果你也想参与项目，欢迎通过[我的个人主页](https://github.com/algebra-MCX)与我联系。如果你想学习 DeepInfer 各个模块的优化核心思路，了解 **DeepGEMM、DeepEP、FlashMLA 如何在 DeepInfer 中协同设计**，欢迎 **Star、Fork 并分享本仓库**。

**当仓库突破 2,000 Stars 和 100 Forks，我会积极与 Inferact 联系，争取集群资源并完成最重要的 Decode 优化。** 你的参与和热情，将是我寻求官方合作的最大底气。

> **项目近况（2026-10-25）：** 这个项目会持续更新。近日身体有些不适，我先休息几天。为了保持本仓库提交历史的整洁，我已将大部分个人优化代码放到私有仓库中。项目仍会继续推进，短暂休息后会恢复公开仓库的更新。感谢大家的理解与支持。

<p align="center">
  <a href="https://github.com/algebra-MCX/mini-DeepInfer/actions/workflows/ci.yml"><img alt="CPU checks" src="https://github.com/algebra-MCX/mini-DeepInfer/actions/workflows/ci.yml/badge.svg"></a> <img alt="Target" src="https://img.shields.io/badge/Target-NVIDIA%20B200-blue"> <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green"><br>
  <a href="README.md">English</a> · 简体中文
</p>

<p align="center">
  <img src="docs/assets/mini-deepinfer-mark.png" width="320" alt="mini-DeepInfer 抽象推理徽标">
</p>

> 一个可以读懂的 DeepSeek-V4.1 运行时：从请求入队一路跟到下一个采样 token。

**用约 2,000 行 PyTorch 读懂 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)。** DeepInfer 在几十张 GPU 上服务
DeepSeek-V4.1-Flash，靠 C++ 调度器、EP/DP/CP 并行和定制 kernel，代码难读，也需要 SM100 才能运行。mini-DeepInfer
保留它最有特色的引擎设计，去掉规模相关的部分：单张 GPU 就能跑完整的 V4.1 架构，4 张 B200 能跑真实的 V4.1-Flash，
请求的每一步都可以跟踪和修改。

## 快速开始

```bash
git clone https://github.com/algebra-MCX/mini-DeepInfer.git
cd mini-DeepInfer
uv run examples/offline_inference.py
```

- **环境**：首次运行时 `uv` 按 `uv.lock` 建好 `.venv`（PyTorch 2.14 + CUDA 13.0，需 NVIDIA 驱动 ≥ 580），之后直接复用。
- **没有 uv**：先 `curl -LsSf https://astral.sh/uv/install.sh | sh`；或者不用 uv，`pip install -e . && python examples/offline_inference.py`。
- **模型**：默认自动下载 [`tiny-random/deepseek-v4.1-flash-1017`](https://huggingface.co/tiny-random/deepseek-v4.1-flash-1017)，
  6800 万参数。架构特性和 V4.1 完全一致，但权重是随机的，生成的文本没有意义。
- **硬件**：任意一张 8 GB 显存的 CUDA GPU；没有 GPU 只能跑 CPU 测试，见[硬件要求](#硬件要求)。

## 核心成果

- **在 4 × B200 上跑真实的 DeepSeek-V4.1-Flash，并按 DeepInfer 官方口径测出性能。** EP4 × DP4，131k 上下文，
  请求数、热身和测量窗口、指标定义都与官方 `run_sweep.sh prefill` 一致，结果直接用官方 `plot_pareto.py` 出图。
  并行层只有约 40 行（[`parallel.py`](minideepinfer/models/deepseek_v41/parallel.py)）。[→ 性能](#性能)
- **逐 token 对照 DeepSeek 官方参考实现。** 量化、候选块筛选与官方 TileLang kernel 逐位一致；整个引擎在 batch、
  ZOS、chunked prefill、抢占等所有调度模式下，logits 与官方 `inference/model.py` 的差距都小于 0.05。[→ 正确性](#正确性)
- **完整的 V4.1 架构**：滑窗 + 压缩 KV、两级稀疏 indexer、hyper-connections、Engram n-gram 记忆、MoE，
  以及注意力状态的 FP8 / FP4 量化往返；权重保持 checkpoint 的 FP8 / FP4 格式。
- **DeepInfer 的引擎设计**：分页 KV block + 每序列 state slot、prefill / decode 分步调度（chunked prefill、抢占）、
  零开销调度（ZOS）、与 batch 组成无关的确定性采样、请求取消与 stop token。

## 性能

### 真实 DeepSeek-V4.1-Flash，单机 4 × B200，prefill 实测

2026-10-22 配对结果（含报告 A / B1 / C1 的三项优化）：相同四张 GPU、相同 checkpoint 和输入，EP4 / DP4 / CP1，131,072-token
上下文，其中 126,976 token 已缓存；每步每 rank 处理 4,096 个非缓存输入 token，统计 10 个完整稳态步。

| 引擎 | 输入吞吐（tok/s/GPU） | 平均每步（ms） |
| --- | ---: | ---: |
| mini-DeepInfer（当前） | **4,315.9** | 949.0 |
| 官方 DeepInfer + 单节点 NVLink 适配 | **36,170.9** | 113.2 |

mini 当前达到本次官方配置的 **11.9%**（差距 8.4 倍）。这是单轮结果，不是三轮中位数，也不是全部优化开启后的性能上限。
mini 使用已验证的当前运行时代码，包含 grouped GEMM、量化、HC、RMSNorm、SwiGLU，以及三项新优化：
tensor core 稀疏注意力、融合 indexer 打分、MoE 只合并本地行并原地反量化；可选非专家权重缓存未开启。双方均 full replay、真实路由、无冗余专家、无
投机解码，均未启用 decode CUDA Graph；本轮只测 prefill，KV 显存预算未强制相等。

**优化与剩余差距：** 本结果包含稀疏注意力改用 tensor core
（[报告 A](docs/perf-reports/A-tensor-core-sparse-attention.md)）、融合 indexer 打分
（[报告 B1](docs/perf-reports/B1-fused-index-scores.md)）、MoE 只合并本地行
（[报告 C1](docs/perf-reports/C1-moe-combine.md)）三项优化。现在最大的两项是注意力（约 432 ms，
其中 182 ms 是拼接 key）和 MoE（410 ms）。分环节数据与运行方式见 [benchmark 文档](docs/benchmarks/README.md)。
tiny 模型的单卡生成吞吐不能与这组真实大模型的输入吞吐直接比较。

## 正确性

`tests/test_v41.py` 在相同权重上运行 DeepSeek 官方参考实现（`inference/model.py`，随 V4.1-Flash checkpoint 发布）进行对照。

| 检查项 | 结果 |
| --- | --- |
| FP8 / FP4 fake quant 对比官方 TileLang kernel | 逐位一致 |
| 两级候选块筛选 | 逐位一致 |
| Sinkhorn 混合、稀疏注意力对比官方 kernel | bf16 误差范围内一致 |
| 引擎端到端：4 个 prompt（37–520 token）同批；ZOS 开 / 关；chunked prefill 128 和 200；抢占 | 每一步 logits 与官方差距小于 0.05（实测最大约 0.02）；greedy token 一致 |

token 比较到官方自身 top-2 差距小于容差的那一步为止：随机权重的 logits 很平，近似平局之后两边比较的只是噪声。

```bash
pytest -m "not gpu"   # 仅 CPU：调度器、block manager
pytest                # 完整测试：需要 GPU + TileLang
```

CPU CI 只跑不依赖 CUDA 的检查；GPU 测试在没有 GPU 时显示为 skipped，不代表通过。

## 架构

```mermaid
flowchart TB
    A["LLM.generate()"] -->|add_request| B["LLMEngine.step()"]
    subgraph CPU["CPU: scheduling"]
        S["Scheduler<br/>waiting / running queues<br/>prefill or decode per step<br/>chunked prefill, preemption"]
        BM["BlockManager x2<br/>KV blocks, 128 tokens each<br/>state slots, one per sequence"]
        S --> BM
    end
    subgraph GPU["GPU: execution"]
        R["ModelRunner<br/>batch construction, ZOS remap"]
        M["DeepseekV41<br/>n_layers x Block"]
        SA["Sampler<br/>Gumbel-max, seed + position"]
        C[("V41Cache<br/>blocks: compressed KV, indexer keys<br/>slots: window ring, compressor tail")]
        R --> M --> SA
        M <--> C
    end
    B -->|"1. schedule step N"| S
    B -->|"2. execute step N, async"| R
    SA -.->|"3. tokens of step N, read during step N+1"| B
```

V4.1 单个注意力层内部：

```mermaid
flowchart LR
    X["hidden x"] --> QR["qr = q_norm(wq_a x)"]
    QR --> Q["q = wq_b qr<br/>n_heads query heads"]
    X --> KV["kv = kv_norm(wkv x)<br/>1 KV head, FP8 round trip"]
    KV --> W[("window ring<br/>last 128 positions")]
    X --> CP["Compressor<br/>pools every ratio tokens"]
    CP --> CKV[("compressed KV, FP4")]
    CP --> IK[("indexer keys, FP4")]
    QR --> IDX["Indexer<br/>top-k compressed entries"]
    IK --> IDX
    Q --> ATT["sparse attention<br/>+ attention sink"]
    W --> ATT
    CKV --> ATT
    IDX --> ATT
    ATT --> O["grouped low-rank output<br/>wo_a, wo_b"]
```

只有 KV 源层运行压缩器，只有 index 源层运行 indexer；中间的层读取源层的缓存，并复用它的 top-k 选择。

### 一个 step 的执行过程

```
第 N 步（enable_zos=True）
  CPU  schedule(N)        只依据预定状态决策（已缓存 / 在途 token 数）
  CPU  execute(N)         异步提交输入、forward、采样和回拷；仍在 GPU 上的在途 token
                          从第 N-1 步的输出缓冲中读取（remap）
  CPU  process(N-1)       等待第 N-1 步完成，追加 token，结束并释放序列
  GPU  ... 第 N-1 步 ... | ... 第 N 步 ...
```

V4.1 有两类注意力状态，调度器同时分配两者：

| 状态 | 存放位置 | 写入者 |
| --- | --- | --- |
| 压缩 KV、indexer 的 key | 分页 KV block，每块 128 个 token | KV 源层，每 `ratio` 个 token 一项 |
| 滑窗环形缓冲、压缩器尾部 | 每条序列一个 state slot | 每一层 / KV 源层 |

被抢占的序列释放两者，之后重新计算；确定性采样保证重算出的 token 与丢弃的完全相同。

### 多卡：EP × DP

每张卡跑自己的调度器和 batch；注意力等稠密层每卡复制一份。路由专家按卡均分，Engram 表按行切分：这些层先汇总
所有卡的 token，只用本卡的那一份计算，再把结果 reduce-scatter 回各卡。单进程运行时这些通信都是空操作。

## 用法

```python
from minideepinfer import LLM, SamplingParams

llm = LLM("tiny-random/deepseek-v4.1-flash-1017", max_model_len=4096, kv_cache_memory_gb=2.0)
outputs = llm.generate(
    prompt_token_ids=[[100, 200, 300]],
    sampling_params=SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16, seq_seed=42),
)
print(outputs[0].token_ids)
```

| 参数 | 含义 |
| --- | --- |
| `max_num_seqs`、`max_num_batched_tokens`、`max_model_len` | batch 上限 |
| `chunked_prefill`、`prefill_attempt_interval` | prefill 调度（DeepInfer 的默认间隔为 70） |
| `kv_cache_memory_gb` / `gpu_memory_utilization` | 缓存预算：绝对值，或空闲显存的比例 |
| `weight_cache_memory_gb` | 非专家 FP8 投影的 BF16 权重缓存上限，单位 GiB；默认 `0` 关闭 |
| `enable_zos` | 零开销调度 |
| `SamplingParams(seq_seed=...)` | 相同输入 + 相同 seed = 相同输出，与 batch 组成无关 |
| `SamplingParams(stop_token_ids=...)` | 额外的停止 token 集合；`LLMEngine.abort_request()` 可安全取消在途请求 |

显存充足时可设置 `LLM(..., weight_cache_memory_gb=0.02)`，避免每次投影都反量化权重。
缓存按模块顺序选取能放入预算的投影，保留原始量化权重，不包含路由专家和 Engram 表；
tiny 模型该配置实际增加 10.56 MiB。缓存先于自动 KV 预算分配，因此会占用部分原本可用于 KV 的显存。
基准对应参数为 `bench.py --weight-cache-gb 0.02`；收益与模型和负载有关，默认保持关闭。

## 读代码

| 想了解的问题 | 从哪里开始 |
| --- | --- |
| 一个请求如何穿过整个运行时？ | [`docs/TECHNICAL.md`](docs/TECHNICAL.md) 第 2–5 节 |
| V4.1 的注意力状态存在哪里？ | [`models/deepseek_v41/model.py`](minideepinfer/models/deepseek_v41/model.py) 与技术说明第 3 节 |
| prefill、decode 和抢占如何配合？ | [`engine/scheduler.py`](minideepinfer/engine/scheduler.py) |
| 缓存预算怎样计算？ | [`worker/model_runner.py`](minideepinfer/worker/model_runner.py) |
| 多卡怎样切分？ | [`models/deepseek_v41/parallel.py`](minideepinfer/models/deepseek_v41/parallel.py) 与 `MoE`、`EngramTable` |
| 正确性怎样验证？ | [`tests/test_v41.py`](tests/test_v41.py) 与 [`tools/v41_reference.py`](tools/v41_reference.py) |

与 DeepInfer 源码的对应关系：

| mini-DeepInfer | DeepInfer 对应位置 |
| --- | --- |
| `engine/block_manager.py` | `csrc/engine/block_manager.hpp` |
| `engine/scheduler.py` | `csrc/engine/scheduler.hpp`（`_schedule_prefill`、`_schedule_decode`） |
| `engine/llm_engine.py` | `deepinfer/engine/llm_engine.py`（`step`） |
| `worker/model_runner.py` | `deepinfer/worker/worker.py`（`execute_model_step`）、`csrc/engine/zos.cpp` |
| `layers/sampler.py` | `tilelang_ops/random_sampling.py` |
| `models/deepseek_v41/` | DeepSeek 的 `inference/model.py`，改写为分页、批处理执行 |
| `bench_prefill.py` | `examples/benchmark.py` + `examples/sweep/run_sweep.sh prefill` |
| `tools/v41_reference.py` | 从官方参考实现生成基准 logits |

[运行时设计](docs/DEEPINFER_DESIGN.md) 解释官方 DeepInfer 的核心边界与本项目的实现顺序，[技术说明](docs/TECHNICAL.md) 逐段对应当前代码。

## 硬件要求

| 用途 | 配置 | 说明 |
| --- | --- | --- |
| CPU 测试 | 2 vCPU、4 GB 内存、2 GB 磁盘 | 调度器与 block manager；CI 使用这个级别 |
| tiny checkpoint（快速开始） | 任意 CUDA GPU、8 GB 显存（建议 16 GB）、16 GB 内存、10 GB 磁盘 | `uv.lock` 锁定 CUDA 13.0 版 PyTorch，需要 NVIDIA 驱动 ≥ 580 |
| 真实 DeepSeek-V4.1-Flash | 已验证：4 × NVIDIA B200，checkpoint 约 510 GB 磁盘 | EP4 × DP4，每卡常驻约 137 GiB（权重 + 12 GiB KV cache）；单卡放不下 |

没有 CPU 上的模型执行路径。使用真实 checkpoint 前请先确认它的许可证。

## 路线图

- [x] DeepSeek-V4.1 模型与官方实现对照测试套件
- [x] 分页 block + state slot、chunked prefill、抢占、ZOS、确定性采样
- [x] EP × DP 多卡运行真实 V4.1-Flash，按官方口径测 prefill
- [ ] 长 chunk（≥ 32,768 token / 卡）的显存：indexer 与注意力输出按 query 分块
- [ ] Decoder bounded replay（YoCo consumer 层 prefill 只重放末尾 140 token）
- [x] 专家分组 GEMM、decode 的 CUDA graph；去掉 forward 中的 host 同步，使 ZOS 重叠
- [ ] DSpark 投机解码，按置信度调度验证
- [ ] 代价模型的 prefill chunk 切分（DeepInfer Q&A）
- [ ] 端到端 batch-invariant 推理
- [ ] 图像输入（V4.1 自带 ViT，checkpoint 已包含视觉权重）

## 许可

Apache 2.0。`models/deepseek_v41/` 的部分代码改写自 DeepSeek 的 MIT 协议参考实现，见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)。

## 致谢

设计参考 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 和 DeepSeek 的 V4.1 参考实现，项目结构受 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 启发。
