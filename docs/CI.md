# CI 与硬件测试

仓库把自动门禁和 GPU 回归分开。这样每个 Pull Request 都能快速得到 CPU
结果，同时不会把一个没有注册 GPU runner 的作业伪装成“已通过”。

| 工作流 | 触发方式 | 执行内容 | 分支保护 |
| --- | --- | --- | --- |
| `CI` | `main` 的 push、指向 `main` 的 PR | CPU 测试、GPU 测试收集检查、compileall、空白检查 | required check：`test` |
| `GPU CI` | 手动 `workflow_dispatch` | CUDA 检查、完整 `pytest -q`、compileall、空白检查 | 当前不设为 required |

## CPU CI

CPU job 明确运行 `pytest -q -m "not gpu"`，所以绿色结果表示 CPU 测试真的通过，
而不是把 GPU 测试的 skipped 当成通过。随后用 `pytest --collect-only -m gpu`
确认 GPU 测试仍然被收集，防止测试文件意外失去标记或被打包遗漏。

## GPU CI

GPU workflow 需要一个带有以下标签的 GitHub Actions self-hosted runner：

```text
self-hosted, linux, x64, gpu
```

它先检查 `torch.cuda.is_available()` 和每张卡的名称，再运行完整测试套件。仓库
当前没有注册 self-hosted runner，因此这个 workflow 保持手动触发，也没有加入
`main` 的 required checks；注册 runner 后再把它提升为必需检查。

在 Slurm 环境中，等价的本地回归是：

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
pytest -q -rs
python -m compileall -q minideepinfer tests tools examples bench.py
git diff --check
```

GPU 结果必须记录 GPU 型号、驱动、CUDA/PyTorch 版本、checkpoint、batch、上下文
长度、dtype、耗时和 skipped/failed 数量。没有这些数据时，不把 GPU 结果写成已验证。
