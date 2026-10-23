# Prefill 性能

## 当前结果

单机 4 × B200、真实 DeepSeek-V4.1-Flash，EP4 / DP4 / CP1，131,072-token 上下文（126,976 已缓存），
每步每 rank 4,096 个非缓存输入 token，统计 10 个完整稳态步。

| 引擎 | 输入 tok/s/GPU | 平均每步（ms） |
| --- | ---: | ---: |
| mini-DeepInfer | 4,315.9 | 949.0 |
| 官方 DeepInfer + NVLink 适配 | 36,170.9 | 113.2 |

mini 为官方配置的 **11.9%**（差距 8.4 倍）。双方均 full replay、真实路由，只测 prefill；mini 的可选权重缓存未开启，
两边均未启用 decode CUDA Graph。仅一组配对结果，KV 显存预算也未强制相等，不能称为全部优化启用后的上限。

## 分环节耗时（profile，ms/步）

| 环节 | mini | 官方 |
| --- | ---: | ---: |
| 稀疏注意力核心 | 99.0 | 14.9 |
| 注意力 key 拼接（gather / cat / where） | 182 | 0 |
| Indexer | 79.3 | 7.9 |
| 注意力其余（投影、压缩、伪量化） | 72 | 约 19 |
| MoE 合计 | 410 | 52.8 |
| HC、归一化及其他 | 122 | 约 10 |
| **合计（profile 下的步长）** | **965** | **110.6** |

各项优化的动机、方案、kernel 级数据和精度验证见 [`../perf-reports/`](../perf-reports/)。原始日志与 profile 摘要见
[`profile-2026-10-22/`](profile-2026-10-22/)。

## 运行入口

```bash
torchrun --nproc-per-node 4 bench_prefill.py \
  --model /path/to/DeepSeek-V4.1-Flash \
  --prefill_batch_size 4096 \
  --no_cuda_graph \
  --result_jsonl results.jsonl
```

模型加载与共享前缀 prefill 的初始化耗时不计入吞吐测量。
