"""DeepSeek-V4.1 model configuration, parsed from a Hugging Face `config.json`."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class V41Config:
    """Text-model hyperparameters. Field names follow the official reference `ModelArgs`."""

    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    score_func: str
    route_scale: float
    swiglu_limit: float
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    compress_rope_theta: float
    original_seq_len: int
    rope_theta: float
    rope_factor: float
    beta_fast: int
    beta_slow: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_id: int
    engram_compressed_vocab_size: int
    n_mtp_layers: int
    dspark_block_size: int
    dspark_noise_token_id: int
    dspark_target_layer_ids: tuple[int, ...]
    dspark_markov_rank: int
    dspark_n_routed_experts: int
    dspark_n_activated_experts: int
    eos_token_id: int

    @classmethod
    def from_pretrained(cls, model_path: str) -> V41Config:
        with open(os.path.join(model_path, "config.json")) as f:
            raw = json.load(f)
        if raw.get("model_type") != "deepseek_v41":
            raise ValueError(f"unsupported model_type {raw.get('model_type')!r}; only deepseek_v41 is supported")
        t = raw["text_config"]
        rope = t.get("rope_scaling") or {}
        return cls(
            vocab_size=t["vocab_size"],
            dim=t["hidden_size"],
            moe_inter_dim=t["moe_intermediate_size"],
            n_layers=t["num_hidden_layers"],
            n_heads=t["num_attention_heads"],
            n_routed_experts=t["n_routed_experts"],
            n_shared_experts=t["n_shared_experts"],
            n_activated_experts=t["num_experts_per_tok"],
            score_func=t["scoring_func"],
            route_scale=t["routed_scaling_factor"],
            swiglu_limit=t["swiglu_limit"],
            q_lora_rank=t["q_lora_rank"],
            head_dim=t["head_dim"],
            rope_head_dim=t["qk_rope_head_dim"],
            norm_eps=t["rms_norm_eps"],
            o_groups=t["o_groups"],
            o_lora_rank=t["o_lora_rank"],
            window_size=t["sliding_window"],
            compress_ratios=tuple(t["compress_ratios"]),
            kv_source_layers=tuple(t["kv_source_layer_ids"]),
            index_source_layers=tuple(t["index_source_layer_ids"]),
            compress_rope_theta=t["compress_rope_theta"],
            original_seq_len=rope.get("original_max_position_embeddings", 0),
            rope_theta=t["rope_theta"],
            rope_factor=rope.get("factor", 1.0),
            beta_fast=rope.get("beta_fast", 32),
            beta_slow=rope.get("beta_slow", 1),
            index_n_heads=t["index_n_heads"],
            index_head_dim=t["index_head_dim"],
            index_topk=t["index_topk"],
            candidate_source_layer=t.get("candidate_source_layer_id", -1),
            candidate_topk_blocks=t.get("candidate_topk_blocks", 0),
            candidate_block_size=t.get("candidate_block_size", 0),
            hc_mult=t["hc_mult"],
            hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
            hc_eps=t["hc_eps"],
            engram_layer_ids=tuple(t.get("engram_layer_ids", ())),
            engram_num_embeddings=tuple(t.get("engram_num_embeddings", ())),
            engram_max_ngram_size=t.get("engram_max_ngram_size", 1),
            engram_vocab_size=t.get("engram_vocab_size", 0),
            engram_n_heads=t.get("engram_n_heads", 0),
            engram_head_dim=t.get("engram_head_dim", 0),
            engram_pad_id=t.get("engram_pad_token_id", 2),
            engram_compressed_vocab_size=t.get("engram_compressed_vocab_size", 0),
            n_mtp_layers=t.get("num_nextn_predict_layers", 0),
            dspark_block_size=t.get("dspark_block_size", 0),
            dspark_noise_token_id=t.get("dspark_noise_token_id", 0),
            dspark_target_layer_ids=tuple(t.get("dspark_target_layer_ids", ())),
            dspark_markov_rank=t.get("dspark_markov_rank", 0),
            dspark_n_routed_experts=t.get("dspark_n_routed_experts", 0),
            dspark_n_activated_experts=t.get("dspark_num_experts_per_tok", 0),
            eos_token_id=raw.get("eos_token_id", t.get("eos_token_id", 1)),
        )

    def reference_args(self) -> dict:
        """Keyword arguments for the official reference `ModelArgs` (text only, no DSpark head)."""
        skip = {"eos_token_id", "dspark_block_size"}
        args = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        args["vision_n_layers"] = 0
        args["dspark_block_size"] = 0
        return args

    # Derived layout helpers.

    def kv_source_of(self, layer_id: int) -> int:
        """The layer whose compressed KV cache this layer reads (the latest source at or before it)."""
        return max(s for s in self.kv_source_layers if s <= layer_id)

    def index_source_of(self, layer_id: int) -> int:
        """The layer whose top-k indices this layer reuses."""
        return max(s for s in self.index_source_layers if s <= layer_id)
