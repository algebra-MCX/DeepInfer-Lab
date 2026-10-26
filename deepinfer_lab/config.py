from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

from deepinfer_lab.models.deepseek_v41.config import V41Config
from deepinfer_lab.models.deepseek_v41.model import PAGE_TOKENS


def resolve_model_path(model: str) -> str:
    """Return a local checkpoint directory, resolving HF model ids through the HF cache."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


@dataclass
class Config:
    """Engine configuration: a minimal union of DeepInfer's model, cache and scheduler configs."""

    model: str
    # Scheduler.
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    chunked_prefill: bool = True
    # Number of consecutive decode steps before a prefill is attempted again. DeepInfer defaults
    # to 70; a prefill is always attempted when there is no decode work.
    prefill_attempt_interval: int = 8
    # KV cache. If kv_cache_memory_gb is set it is the cache budget; otherwise the budget is
    # gpu_memory_utilization times the memory free after model loading and weight caching.
    kv_cache_memory_gb: float | None = None
    gpu_memory_utilization: float = 0.5
    num_gpu_blocks: int = -1  # If > 0, overrides the block count (useful to force preemption).
    weight_cache_memory_gb: float = 0.0  # Optional bf16 cache for non-expert FP8 projections; 0 disables it.
    # Runtime.
    enable_zos: bool = True  # Zero-overhead scheduling: overlap CPU scheduling/post-processing with GPU work.
    enable_cuda_graph: bool = True  # Replay decode steps as captured CUDA graphs.
    seed: int = 0
    log_stats_interval: float = 0.0  # If > 0, log throughput every this many seconds.

    model_path: str = field(init=False)
    model_config: V41Config = field(init=False)
    block_size: int = field(init=False, default=PAGE_TOKENS)
    dtype: torch.dtype = field(init=False, default=torch.bfloat16)

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight_cache_memory_gb) or self.weight_cache_memory_gb < 0:
            raise ValueError("weight_cache_memory_gb must be finite and nonnegative")
        self.model_path = resolve_model_path(self.model)
        self.model_config = V41Config.from_pretrained(self.model_path)
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError("max_num_batched_tokens must be >= block_size")
        if not self.chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("without chunked prefill, max_num_batched_tokens must be >= max_model_len")
        if self.prefill_attempt_interval < 1:
            raise ValueError("prefill_attempt_interval should be >= 1")

    @property
    def eos_token_id(self) -> int:
        return self.model_config.eos_token_id

    @property
    def max_blocks_per_seq(self) -> int:
        return (self.max_model_len + self.block_size - 1) // self.block_size
