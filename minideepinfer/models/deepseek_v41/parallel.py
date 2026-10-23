"""Expert parallelism across data-parallel ranks (DeepInfer's EP x DP layout).

Every rank runs its own scheduler and batch. Attention and dense layers are replicated; routed
experts and the Engram tables are sharded. A sharded layer gathers the tokens of all ranks, applies
its local shard, and reduce-scatters the partial results back, so each rank receives the sum over
all shards for its own tokens. With a single process every helper is a no-op.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def world() -> tuple[int, int]:
    """(world_size, rank), or (1, 0) when torch.distributed is not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def all_gather_rows(x: torch.Tensor) -> torch.Tensor:
    """Concatenates every rank's rows, each rank padded to the largest row count: [W * T_max, ...]."""
    n = torch.tensor([x.size(0)], device=x.device)
    dist.all_reduce(n, op=dist.ReduceOp.MAX)
    pad = int(n.item()) - x.size(0)
    if pad:
        x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])])
    out = x.new_empty(world()[0] * x.size(0), *x.shape[1:])
    dist.all_gather_into_tensor(out, x.contiguous())
    return out


def reduce_scatter_rows(y: torch.Tensor, num_rows: int) -> torch.Tensor:
    """Inverse of `all_gather_rows`: sums `y` over ranks and returns this rank's first `num_rows` rows."""
    out = y.new_empty(y.size(0) // world()[0], *y.shape[1:])
    dist.reduce_scatter_tensor(out, y.contiguous())
    return out[:num_rows]
