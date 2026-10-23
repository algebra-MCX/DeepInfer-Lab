"""Model runner: owns the model and its caches, and turns scheduler output into GPU work.

`execute()` only enqueues GPU work and returns a `StepHandle`; results are read back later by
`StepHandle.wait()`. This split is what enables zero-overhead scheduling (ZOS): the engine
post-processes step N-1 and schedules step N+1 on the CPU while step N runs on the GPU.

With ZOS, a decode step may need an input token that the CPU has not seen yet (sampled by the
previous, still unprocessed step). It is gathered on the GPU from the previous step's sampled-token
buffer through a `remap` index, mirroring DeepInfer's `prepare_zos_inputs`. The same token also
heads that position's Engram n-gram window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from minideepinfer.config import Config
from minideepinfer.engine.scheduler import SchedulerOutput
from minideepinfer.layers.sampler import Sampler, SamplingTensors, mix_seed
from minideepinfer.models.deepseek_v41 import DeepseekV41
from minideepinfer.models.deepseek_v41.model import V41Batch, V41Cache
from minideepinfer.models.deepseek_v41.weights import iterate_raw
from minideepinfer.sequence import Sequence
from minideepinfer.worker.cuda_graph import DecodeGraphRunner

logger = logging.getLogger(__name__)


@dataclass
class StepHandle:
    """A step that has been enqueued on the GPU but whose results are not yet consumed."""

    sched: SchedulerOutput
    sampled_gpu: torch.Tensor  # [num_sample] int64, kept alive for the next step's remap.
    sampled_cpu: torch.Tensor  # Pinned host copy, valid after `event` completes.
    event: torch.cuda.Event
    row_of: dict[int, int] = field(default_factory=dict)  # seq_id -> row in sampled_gpu.
    logits: torch.Tensor | None = None  # Kept for tests that compare against the reference.

    def wait(self) -> list[int]:
        self.event.synchronize()
        return self.sampled_cpu.tolist()


def _to_gpu(data: list | np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(data), dtype=dtype).pin_memory().cuda(non_blocking=True)


def _batch_to_gpu(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Transfer integer batch metadata together; recover named views without device kernels."""
    flat = _to_gpu(np.concatenate([a.ravel() for a in arrays.values()]), torch.int64)
    parts = flat.split([a.size for a in arrays.values()])
    return {name: part.view(a.shape) for (name, a), part in zip(arrays.items(), parts)}


def _sampling_to_gpu(temps, greedy, top_k, top_p, seeds) -> dict[str, torch.Tensor]:
    """Transfer one byte buffer; recover contiguous typed views without GPU conversions."""
    # Eight-byte fields precede four-byte fields and bools, keeping every view aligned for odd batches too.
    arrays = dict(
        seeds=np.asarray(seeds, dtype=np.int64), top_k=np.asarray(top_k, dtype=np.int64),
        temperatures=np.asarray(temps, dtype=np.float32), top_p=np.asarray(top_p, dtype=np.float32),
        greedy=np.asarray(greedy, dtype=np.bool_),
    )
    if not seeds:
        return {name: torch.empty(0, device="cuda", dtype=torch.from_numpy(a).dtype) for name, a in arrays.items()}
    # Fill pinned storage directly; avoid a second CPU copy and its large-buffer thread-pool overhead.
    pinned = torch.empty(sum(a.nbytes for a in arrays.values()), dtype=torch.uint8, pin_memory=True)
    host, offset = pinned.numpy(), 0
    for a in arrays.values():
        host[offset:offset + a.nbytes] = a.view(np.uint8)
        offset += a.nbytes
    flat = pinned.cuda(non_blocking=True)
    parts = flat.split([a.nbytes for a in arrays.values()])
    return {name: part.view(torch.from_numpy(a).dtype) for (name, a), part in zip(arrays.items(), parts)}


class ModelRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        cfg = config.model_config
        self.vocab_size = cfg.vocab_size
        self.max_ngram = cfg.engram_max_ngram_size if cfg.engram_layer_ids else 0

        torch.manual_seed(config.seed)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.dtype)
        with torch.device("cuda"):
            self.model = DeepseekV41(cfg, config.max_model_len + 1, config.model_path)
        torch.set_default_dtype(default_dtype)
        self.model.load_weights(iterate_raw(config.model_path, self.model.checkpoint_select))
        self.model.eval()
        # Reserve cached weights before deriving the KV budget from the remaining free memory.
        self.weight_cache_bytes = self.model.cache_non_expert_weights(int(config.weight_cache_memory_gb * 2**30))
        logger.info("cached %.2f MiB of non-expert FP8 weights in bf16", self.weight_cache_bytes / 2**20)
        self.sampler = Sampler()

        # Slot 0 is reserved; one extra slot covers a chunked prefill alongside a full running set.
        self.num_slots = config.max_num_seqs + 2
        self.num_blocks = self._num_blocks()
        self.model.cache = V41Cache.allocate(cfg, self.num_blocks, self.num_slots)
        self.graphs = None
        if config.enable_cuda_graph:
            self.graphs = DecodeGraphRunner(
                self.model, config.max_num_seqs, config.max_blocks_per_seq, config.max_model_len
            )
            self.graphs.capture()

    def _num_blocks(self) -> int:
        cfg = self.config
        if cfg.num_gpu_blocks > 0:
            return cfg.num_gpu_blocks
        torch.cuda.empty_cache()
        if cfg.kv_cache_memory_gb is not None:
            budget = cfg.kv_cache_memory_gb * 2**30
        else:
            budget = torch.cuda.mem_get_info()[0] * cfg.gpu_memory_utilization
        budget -= self.num_slots * V41Cache.bytes_per_slot(cfg.model_config)
        num_blocks = int(budget) // V41Cache.bytes_per_block(cfg.model_config)
        # Never allocate more than every sequence could use at max_model_len.
        num_blocks = min(num_blocks, cfg.max_num_seqs * cfg.max_blocks_per_seq + 1)
        if num_blocks <= 1:
            raise RuntimeError(f"not enough GPU memory for the KV cache (budget {budget / 2**30:.2f} GiB)")
        logger.info("KV cache: %d blocks x %d tokens, %d state slots", num_blocks, cfg.block_size, self.num_slots)
        return num_blocks

    # Input preparation.

    def _seq_inputs(
        self, seq: Sequence, start: int, end: int, prev: StepHandle | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Host-side inputs of positions [start, end) of one sequence.

        Returns (positions, input_ids, remap, ngram_ids). A token still in flight on the GPU appears
        as a placeholder 0 in `input_ids`, and `remap` holds its row in the previous step's samples
        (-1 elsewhere). `ngram_ids` [seqlen_q, max_ngram] looks back from each position, -1 before
        the sequence start.
        """
        ngram_shifts = np.arange(max(self.max_ngram, 1))
        # Tokens of [lookback, end) are needed: the n-gram window of `start` reaches back that far.
        lookback = max(0, start - len(ngram_shifts) + 1)
        known = np.zeros(end - lookback, dtype=np.int64)
        available = seq.token_ids[lookback:end]
        known[: len(available)] = available

        positions = np.arange(start, end)
        input_ids = known[positions - lookback]
        remap = np.full(end - start, -1, dtype=np.int64)
        if end > len(seq.token_ids):  # The decode input was sampled by the previous step, only on the GPU.
            assert prev is not None and seq.seq_id in prev.row_of, f"input token of {seq} is not in flight"
            remap[-1] = prev.row_of[seq.seq_id]
        ngram_pos = positions[:, None] - ngram_shifts[None, :]
        ngram_ids = np.where(ngram_pos >= 0, known[np.maximum(ngram_pos, lookback) - lookback], -1)
        return positions, input_ids, remap, ngram_ids

    def _prepare(self, sched: SchedulerOutput, prev: StepHandle | None) -> tuple[V41Batch, torch.Tensor]:
        seqs, seqlens = sched.seqs, sched.seqlens_q
        ends = [seq.num_cached_tokens for seq in seqs]
        starts = [end - seqlen for end, seqlen in zip(ends, seqlens)]
        per_seq = [self._seq_inputs(seq, start, end, prev) for seq, start, end in zip(seqs, starts, ends)]
        positions, input_ids, remap, ngram_ids = (np.concatenate(parts) for parts in zip(*per_seq))

        cu_seqlens = np.cumsum([0, *seqlens]).tolist()
        sample_ids = {seq.seq_id for seq, _ in sched.sample_seqs}
        sample_rows = [cu_seqlens[b + 1] - 1 for b, seq in enumerate(seqs) if seq.seq_id in sample_ids]
        block_tables = np.zeros((len(seqs), max(len(seq.block_table) for seq in seqs)), dtype=np.int64)
        for b, seq in enumerate(seqs):
            block_tables[b, : len(seq.block_table)] = seq.block_table

        def per_token(values: list[int] | np.ndarray) -> np.ndarray:
            """Broadcasts one value per sequence to each of its tokens."""
            return np.repeat(values, seqlens)

        arrays = dict(
            input_ids=input_ids,
            positions=positions,
            token_seq=per_token(np.arange(len(seqs))),
            slots=per_token([seq.state_slot for seq in seqs]),
            chunk_start=per_token(starts),
            chunk_end=per_token(ends),
            block_tables=block_tables,
            sample_rows=np.asarray(sample_rows, dtype=np.int64),
        )
        needs_remap = (remap >= 0).any()
        if needs_remap:
            arrays["remap"] = remap
        if self.max_ngram:
            arrays["ngram_ids"] = ngram_ids
        tensors = _batch_to_gpu(arrays)
        sample_rows_gpu = tensors.pop("sample_rows")
        input_ids_gpu = tensors["input_ids"]
        if needs_remap:
            remap_gpu = tensors.pop("remap")
            input_ids_gpu = torch.where(remap_gpu >= 0, prev.sampled_gpu[remap_gpu.clamp(min=0)], input_ids_gpu)
            tensors["input_ids"] = input_ids_gpu
        if self.max_ngram:
            tensors["ngram_ids"][:, 0] = input_ids_gpu  # Include the token just remapped from the GPU.
        else:
            tensors["ngram_ids"] = None
        return V41Batch(**tensors, cu_seqlens=cu_seqlens, max_end=max(ends)), sample_rows_gpu

    def _prepare_sampling(self, sched: SchedulerOutput) -> SamplingTensors:
        temps, greedy, top_k, top_p, seeds = [], [], [], [], []
        for seq, _ in sched.sample_seqs:
            sp = seq.sampling_params
            greedy.append(sp.greedy)
            temps.append(1.0 if sp.greedy else sp.temperature)
            top_k.append(self.vocab_size if sp.top_k < 0 else sp.top_k)
            top_p.append(sp.top_p)
            # The sampled token lands at position num_cached_tokens.
            seeds.append(mix_seed(sp.seq_seed, seq.num_cached_tokens))
        return SamplingTensors(
            **_sampling_to_gpu(temps, greedy, top_k, top_p, seeds),
            all_greedy=all(greedy),
            need_top_k_top_p=any(k < self.vocab_size or p < 1.0 for k, p in zip(top_k, top_p)),
        )

    # Execution.

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput, prev: StepHandle | None) -> StepHandle:
        batch, sample_rows = self._prepare(sched, prev)
        if self.graphs is not None and not sched.is_prompt and self.graphs.can_run(batch):
            hidden = self.graphs.run(batch)
        else:
            hidden = self.model(batch)
        logits = self.model.compute_logits(hidden[sample_rows])
        sampled = self.sampler(logits, self._prepare_sampling(sched))
        sampled_cpu = torch.empty(sampled.shape, dtype=sampled.dtype, pin_memory=True)
        sampled_cpu.copy_(sampled, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        row_of = {seq.seq_id: i for i, (seq, _) in enumerate(sched.sample_seqs)}
        return StepHandle(sched, sampled, sampled_cpu, event, row_of, logits)
