from __future__ import annotations

import json
import os

from transformers import PreTrainedTokenizerFast


def load_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """Loads `tokenizer.json` directly, so checkpoints written for newer transformers releases still load."""
    kwargs = {}
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        kwargs = {k: config[k] for k in ("bos_token", "eos_token", "pad_token", "chat_template") if isinstance(config.get(k), str)}
    return PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_path, "tokenizer.json"), **kwargs)
