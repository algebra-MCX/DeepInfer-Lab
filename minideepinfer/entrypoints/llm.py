from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from minideepinfer.config import Config
from minideepinfer.engine.llm_engine import LLMEngine
from minideepinfer.metrics import RequestMetrics
from minideepinfer.sampling_params import SamplingParams


@dataclass
class RequestOutput:
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    finish_reason: str  # "stop", "length", "ignored" or "abort".
    request_id: str = ""
    metrics: RequestMetrics | None = None


_FINISH_REASONS = {
    "FINISHED_STOPPED": "stop",
    "FINISHED_LENGTH": "length",
    "FINISHED_IGNORED": "ignored",
    "FINISHED_ABORTED": "abort",
}


class LLM:
    """Offline batched inference.

    Example:
        llm = LLM("tiny-random/deepseek-v4.1-flash-1017")
        outputs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=32))

    See `Config` for all keyword arguments.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.llm_engine = LLMEngine(Config(model, **kwargs))

    @property
    def tokenizer(self):  # noqa: ANN201
        return self.llm_engine.tokenizer

    def generate(
        self,
        prompts: str | list[str] | None = None,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        prompt_token_ids: list[list[int]] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        """Generate completions; outputs are returned in the order of the inputs."""
        if prompt_token_ids is None:
            if prompts is None:
                raise ValueError("either prompts or prompt_token_ids must be provided")
            if isinstance(prompts, str):
                prompts = [prompts]
            prompt_token_ids = [self.tokenizer.encode(p) for p in prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompt_token_ids)
        if len(sampling_params) != len(prompt_token_ids):
            raise ValueError("sampling_params and prompts must have the same length")

        engine = self.llm_engine
        seqs = [engine.add_request(ids, sp) for ids, sp in zip(prompt_token_ids, sampling_params)]
        pbar = tqdm(total=len(seqs), desc="Generating", disable=not use_tqdm)
        while engine.has_unfinished_requests():
            pbar.update(len(engine.step()))
        pbar.close()

        return [
            RequestOutput(
                request_id=seq.request_id,
                prompt_token_ids=seq.token_ids[: seq.prompt_len],
                token_ids=seq.output_token_ids,
                text=self.tokenizer.decode(seq.output_token_ids, skip_special_tokens=True),
                finish_reason=_FINISH_REASONS[seq.status.name],
                metrics=RequestMetrics.from_sequence(seq),
            )
            for seq in seqs
        ]
