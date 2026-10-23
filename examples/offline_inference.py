"""Offline inference with a DeepSeek-V4.1 checkpoint.

The default checkpoint is a tiny randomly initialized V4.1 (`tiny-random/deepseek-v4.1-flash-1017`):
it exercises the full architecture but its text is meaningless. Pass `--model` to use another one.
"""

import argparse

from minideepinfer import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny-random/deepseek-v4.1-flash-1017", help="HF model id or local checkpoint directory")
args = parser.parse_args()

llm = LLM(args.model, max_model_len=4096, kv_cache_memory_gb=2.0)
prompts = [
    "Introduce yourself in one sentence.",
    "List all prime numbers below 30.",
    "Explain what a KV cache is in LLM inference.",
]
params = [
    SamplingParams(temperature=0, max_tokens=32),  # Greedy.
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=32, seq_seed=42),  # Reproducible sampling.
    SamplingParams(temperature=0.7, top_k=50, max_tokens=64, seq_seed=42),
]
for prompt, out in zip(prompts, llm.generate(prompts, params)):
    print(f"Q: {prompt}\nA: {out.text!r} [{out.finish_reason}, {len(out.token_ids)} tokens]\n")
