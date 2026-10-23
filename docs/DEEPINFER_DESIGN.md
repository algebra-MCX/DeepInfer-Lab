# DeepInfer 风格的运行时设计

这份文档先于代码实现，记录 mini-DeepInfer 要坚持的工程边界、状态不变量和验证方法。目标不是复制 DeepInfer 的生产代码，而是保留它最值得学习的技术品味：把调度、内存、设备执行和正确性证据拆开，让每一个性能优化都能落回一个可检查的状态转移。

官方 DeepInfer 的公开说明将优化 kernel、并行、DSpark、异步调度、batch-invariant、确定性采样和结构化输出列为核心能力；官方调度器源码则把 `SchedulerOutputs`、prefill/decode、block manager 和异步 step 作为独立边界：

- [DeepInfer README](https://github.com/deepseek-ai/DeepInfer)
- [官方 `scheduler.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/scheduler.hpp)
- [官方 `block_manager.hpp`](https://github.com/deepseek-ai/DeepInfer/blob/main/csrc/engine/block_manager.hpp)
- [官方异步调度说明](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md#asynchronous-scheduling-zero-overhead-batch-scheduler)

## 1. 先定义运行时的承诺

mini-DeepInfer 当前是以 PyTorch 为主的可读实现，支持单卡和机内 EP × DP。它需要满足四个可观察的承诺：

1. **每一个 step 都有明确输入和输出。** 调度输出描述本轮 token 数、序列顺序、block table 和需要采样的行；设备执行不偷偷改变 CPU 队列。
2. **资源生命周期可以被证明。** KV block 和 V4.1 state slot 都有唯一的所有者；抢占、完成、取消都会释放两类资源。
3. **重算是等价的。** 采样随机性只由序列 seed 和 token position 决定，不能依赖 batch 行号或全局 RNG 的历史。
4. **性能路径有可回归的正确性证据。** 先用最小测试暴露状态差异，再改实现；性能数字必须同时记录硬件、checkpoint、batch、上下文、dtype 和 warm-up。

这四条比“把所有生产特性搬过来”更重要。EP、CP、定制 kernel 和多机通信都可以以后替换进来，但不能破坏这些边界。

## 2. 官方设计与本项目的映射

| 官方 DeepInfer 概念 | mini-DeepInfer 的实现 | 当前边界 |
| --- | --- | --- |
| C++ `Scheduler` / `SchedulerOutputs` | `engine/scheduler.py` 的 `Scheduler` / `SchedulerOutput` | 单进程、无 DP 组投票 |
| `BlockManager` free list 与引用计数 | `engine/block_manager.py` | 单 GPU，无跨 rank 共享 |
| iteration-level prefill/decode | `Scheduler._schedule_prefill()` / `_schedule_decode()` | 不混合两种 q length |
| zero-overhead scheduling | `worker/model_runner.py` 的 `StepHandle` 与 `remap` | forward 已去除 host 同步；结果在 `StepHandle.wait()` 回收 |
| grouped expert GEMM | `models/deepseek_v41/model.py` 的 `GroupedExperts` | FP4 按需反量化为 bf16，三次 `torch._grouped_mm`，非原生 FP4 GEMM |
| decode CUDA Graph | `worker/cuda_graph.py` 的 `DecodeGraphRunner` | 按 batch/context 分桶；padding 使用保留 slot/block 0；prefill、logits 与采样仍 eager |
| deterministic sampling | `layers/sampler.py` 的 `mix_seed()` + Gumbel-max | 只覆盖 token 采样 |
| V4.1 attention state | `models/deepseek_v41/model.py` 的 block + state slot | 每个 rank 独立管理分页缓存 |
| structured output / DSpark | 预留在 `SamplingParams` 和执行边界 | 尚未实现 grammar 编译或 draft verify |

映射表是维护约束。新增功能必须先说明属于哪个边界，不能把调度策略塞进模型 forward，也不能让模型直接操作 waiting/running 队列。

## 3. 一个 step 的数据契约

一次引擎迭代按下面的顺序进行：

```text
schedule(N)
  -> SchedulerOutput(prompt/decode, seqs, seqlens_q, sample_seqs, reservations)
execute(N)
  -> ModelRunner 在 GPU stream 上构造 batch、运行 model、采样
process(N-1)
  -> 等待上一个 StepHandle，提交 token，处理 stop/length/abort，释放状态
```

开启 ZOS 时，`process(N-1)` 可以和 `execute(N)` 重叠。这里的“重叠”只描述依赖关系，不代表 Python 的每个算子都已经无同步；任何需要读取 GPU 结果的地方都必须集中在 `StepHandle.wait()`。

### 3.1 调度输出必须是完整快照

`SchedulerOutput` 至少要能回答：

- 本轮是 prefill 还是 decode？
- batch 中每个序列占多少 query token？
- 哪些序列要采样，采样结果对应哪一个 epoch？
- 哪些序列被忽略、完成、抢占或取消？

`ModelRunner` 只消费这份快照，不重新推断调度策略。这样可以在 CPU 上单独测试调度，也可以用假执行器重放整个生命周期。

### 3.2 CPU 视图与 GPU 视图

ZOS 下同一条序列有两个长度：

```text
CPU 已知 token:       len(token_ids)
GPU 已预留 token:     get_len() = len(token_ids) + num_inflight_tokens
已写入/预留 KV:       num_cached_tokens
```

decode 的合法条件是 `num_cached_tokens == get_len() - 1`。调度时先增加 reservation，结果返回时才追加真实 token；旧结果携带的 `epoch` 如果已经失效，就只能丢弃，不能重新写入序列。

## 4. V4.1 状态布局

V4.1 的缓存不是一张传统 KV 表，而是两种不同生命周期的状态：

1. **分页压缩状态：** 压缩 KV 和 indexer key 按压缩后的 entry 写进 128-token block。它适合用 block table 做批量寻址。
2. **序列状态：** sliding-window ring 和 compressor tail 每条序列各占一个 state slot。它必须跟着序列一起迁移、抢占和释放。

因此，一个序列的资源释放必须成对发生：

```python
block_manager.free(seq.block_table)
state_manager.free([seq.state_slot])
```

不能只释放 block；否则 scheduler 看起来有空闲 KV，但下一条请求会在 state slot 上泄漏。

## 5. 调度策略

### Prefill

- waiting 队列保持 FCFS；
- 队头 prompt 不可容纳时停止继续尝试，避免越过队头造成不可预测的饥饿；
- `max_num_batched_tokens` 只限制本轮 q token，chunked prefill 允许一个 prompt 跨多个 step；
- prompt 最后一块才产生采样 reservation；
- 首次进入 running 前，同时分配完整 prompt 所需的 block 和一个 state slot。

### Decode

- 每条 running 序列一次前进一个 token；
- 先按输出长度决定候选顺序；
- block 不足时释放最长输出的序列并把它放回 waiting 队列头；
- 抢占会递增 epoch，后续旧的 GPU 结果不会污染重算；
- 到达 EOS、`max_tokens` 或 `max_model_len` 后，在结果提交点释放资源。

### 取消请求

取消是一个显式的终态。它不能伪装成 EOS，也不能只从队列中删除对象。安全顺序是：

1. 让仍在 GPU 上的 `StepHandle` 完成；
2. 让 scheduler 使请求的 epoch 失效；
3. 从 waiting/running 中移除；
4. 同时释放 block 和 state slot；
5. 返回 `abort` finish reason。

这样取消不会在 GPU 仍读取旧 block 时提前复用同一块存储。

## 6. 采样与可重放性

官方文档把 deterministic sampling 和 batch-invariant 作为独立能力。mini-DeepInfer 使用以下约束实现教学版等价：

```text
sample_seed = mix_seed(sequence_seed, output_position)
token = GumbelMax(logits / temperature, sample_seed)
```

采样不能使用 batch row 作为随机种子，也不能让 top-k/top-p 的筛选依赖其他行。验证必须覆盖：

- 单行与多行 batch 相同；
- 正序与逆序 batch 相同；
- 抢占重算后的 token 序列相同；
- 取消一个请求不改变其他请求的结果。

## 7. 按证据推进的实现顺序

后续代码按下面的顺序推进，每一步都要有一个可以独立审阅的提交：

1. **请求生命周期完整化：** request id、取消、finish reason、资源快照和 CPU 回归测试。
2. **调度成本可观察：** 记录 prefill/decode token、抢占次数、block/slot 水位，并将指标与调度输出分离。
3. **采样边界扩展：** stop token 集合和可选的 logits mask，保持 batch-invariant。
4. **设备执行优化（已实现）：** 去除 forward 中的 host-visible 分支，使用 grouped GEMM 和 decode CUDA Graph；下一步评估权重反量化、logits 与采样开销。
5. **高级路径：** DSpark draft/verify、SWA bounded replay、结构化输出和真实 checkpoint 的端到端验证。

第 4、5 步都不能在没有基准和 reference 对照的情况下直接宣称“完成”。如果一个优化改变了 cache layout 或 scheduler contract，必须先更新本文件的映射表和不变量。

第 4 步的 tiny checkpoint 验证见 PR [#13](https://github.com/algebra-MCX/mini-DeepInfer/pull/13)、
[#14](https://github.com/algebra-MCX/mini-DeepInfer/pull/14)、[#15](https://github.com/algebra-MCX/mini-DeepInfer/pull/15)、
[#16](https://github.com/algebra-MCX/mini-DeepInfer/pull/16)：包含空专家、EP 分片、压缩组边界和 graph padding 回归测试，
以及官方 reference logits 对照（`LOGIT_ATOL=0.05`）。其中 1×B200 的 tiny 离线输出吞吐从 478 提升至
2643 tok/s（3 次取中位数）；这不代表真实 checkpoint 的多卡 prefill 数据已更新。具体 workload、
TTFT、TPOT、goodput 和 capture 开销以各 PR 的实验记录为准。

## 8. 验证门槛

每次修改至少通过：

```bash
pytest tests/test_block_manager.py tests/test_scheduler.py
pytest
git diff --check
```

有 CUDA 和官方 reference 时，再运行 `tests/test_sampler.py`、`tests/test_v41.py` 和端到端抢占矩阵。没有对应硬件时，报告为 skipped/未验证，不把它写成通过。

## 9. 当前明确不承诺的内容

- 单卡 PyTorch 实现不等于官方多卡吞吐；
- 随机权重 checkpoint 只能验证执行语义，不能证明模型能力；
- 没有 grammar compiler 时，`structured output` 不能写成已支持；
- 没有 draft model 和 acceptance trace 时，DSpark 不能写成已支持；
- 任何 benchmark 都必须注明硬件、模型、batch、长度、dtype 和 warm-up。

这份文档描述的是实现的“骨架”。代码可以逐步变快，但这些边界必须保持可读、可测、可重放。
