from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict
import tinker

from ._runtime import AsyncRuntime, ConcurrentAPIFuture
from .data import model_input_tokens
from .errors import UnsupportedCapabilityError


class SamplingTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str
    model_path: str | None = None
    lora: str | None = None


class SamplingProvider(Protocol):
    async def sample(
        self,
        *,
        target: SamplingTarget,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool,
        topk_prompt_logprobs: int,
    ) -> tinker.SampleResponse: ...

    async def compute_logprobs(
        self, *, target: SamplingTarget, prompt: tinker.ModelInput
    ) -> list[float | None]: ...


class SamplingClient:
    def __init__(
        self,
        runtime: AsyncRuntime,
        provider: SamplingProvider,
        target: SamplingTarget,
        tokenizer_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._runtime = runtime
        self._provider = provider
        self._target = target
        self._tokenizer_factory = tokenizer_factory

    def sample(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> Future[tinker.SampleResponse]:
        model_input_tokens(prompt)
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if topk_prompt_logprobs < 0:
            raise ValueError("topk_prompt_logprobs must be nonnegative")
        return self._runtime.submit_future(
            self._provider.sample(
                target=self._target,
                prompt=prompt,
                num_samples=num_samples,
                sampling_params=sampling_params,
                include_prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
            )
        )

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> tinker.SampleResponse:
        return await ConcurrentAPIFuture(
            self.sample(
                prompt,
                num_samples,
                sampling_params,
                include_prompt_logprobs,
                topk_prompt_logprobs,
            )
        )

    def compute_logprobs(self, prompt: tinker.ModelInput) -> Future[list[float | None]]:
        model_input_tokens(prompt)
        return self._runtime.submit_future(
            self._provider.compute_logprobs(target=self._target, prompt=prompt)
        )

    async def compute_logprobs_async(
        self, prompt: tinker.ModelInput
    ) -> list[float | None]:
        return await ConcurrentAPIFuture(self.compute_logprobs(prompt))

    def get_tokenizer(self) -> Any:
        if self._tokenizer_factory is None:
            raise UnsupportedCapabilityError(
                "Remote Training does not expose a pinned tokenizer revision; "
                "configure tokenizer_factory"
            )
        return self._tokenizer_factory(self._target.base_model)

    def get_base_model(self) -> str:
        return self._target.base_model

    async def get_base_model_async(self) -> str:
        return self.get_base_model()

    def get_telemetry(self) -> None:
        return None
