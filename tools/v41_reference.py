"""Golden logits from DeepSeek's official V4.1 reference implementation (`inference/model.py`).

The official code ships inside the DeepSeek-V4.1-Flash checkpoint repo; only its `inference/`
folder is downloaded. It runs here in bf16 mode (`dtype="bf16"`, `expert_dtype=None`) on weights
dequantized from the FP8/FP4 checkpoint, which is the numerics DeepInfer-Lab reproduces.

Usage:
    python tools/v41_reference.py --model tiny-random/deepseek-v4.1-flash-1017 --out golden.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from huggingface_hub import snapshot_download
from deepinfer_lab.tokenizer import load_tokenizer

from deepinfer_lab.config import resolve_model_path
from deepinfer_lab.models.deepseek_v41.config import V41Config
from deepinfer_lab.models.deepseek_v41.weights import iterate_dequantized

REFERENCE_REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def load_reference_module():
    ref_dir = os.path.join(snapshot_download(REFERENCE_REPO, allow_patterns=["inference/*"]), "inference")
    sys.path.insert(0, ref_dir)
    import model as ref  # noqa: PLC0415

    return ref


def build_reference(model_path: str, max_seq_len: int):
    ref = load_reference_module()
    torch.set_default_device("cuda")  # the reference creates some index tensors without a device
    cfg = V41Config.from_pretrained(model_path)
    args = ref.ModelArgs(**cfg.reference_args(), max_batch_size=1, max_seq_len=max_seq_len, dtype="bf16", expert_dtype=None)
    tokenizer = load_tokenizer(model_path)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = ref.Transformer(args, tokenizer)
    torch.set_default_dtype(prev)
    params = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for name, tensor in iterate_dequantized(model_path, keep_engram_fp8=True):
            if name in params:
                params[name].copy_(tensor)
                loaded.add(name)
    missing = set(params) - loaded
    assert not missing, f"reference parameters not loaded: {sorted(missing)[:8]}"
    return model, tokenizer


@torch.inference_mode()
def golden_logits(model, prompt_ids: list[int], num_decode: int) -> tuple[list[int], torch.Tensor]:
    """Greedy-decodes `num_decode` tokens; returns the generated ids and the logits of every step."""
    x = torch.tensor([prompt_ids], device="cuda")
    _, logits, _ = model(x, 0)
    all_logits, out = [logits[0].float().cpu()], []
    pos = len(prompt_ids)
    for _ in range(num_decode):
        token = int(logits[0].argmax())
        out.append(token)
        _, logits, _ = model(torch.tensor([[token]], device="cuda"), pos)
        all_logits.append(logits[0].float().cpu())
        pos += 1
    return out, torch.stack(all_logits)


def make_prompts(lengths: list[int], seed: int = 0) -> list[list[int]]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randint(100, 120000, (n,), generator=gen, device="cpu").tolist() for n in lengths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny-random/deepseek-v4.1-flash-1017")
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[300])
    parser.add_argument("--num-decode", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model, _ = build_reference(model_path, max_seq_len=1024)
    cases = []
    for prompt in make_prompts(args.prompt_lens):
        tokens, logits = golden_logits(model, prompt, args.num_decode)
        cases.append({"prompt": prompt, "tokens": tokens, "logits": logits})
        print(f"prompt_len={len(prompt)} greedy tokens: {tokens}")
    torch.save(cases, args.out)


if __name__ == "__main__":
    main()
