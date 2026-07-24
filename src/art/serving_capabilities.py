from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

ServingFeature = Literal[
    "binary_routed_experts",
    "fast_metrics",
    "inplace_lora_load",
    "in_flight_lora_updates",
    "policy_token_spans",
    "prompt_token_distributions",
]


class ServingCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Literal["openai_compatible", "art_vllm"]
    protocol_version: int
    binary_routed_experts: bool = False
    fast_metrics: bool = False
    inplace_lora_load: bool = False
    in_flight_lora_updates: bool = False
    policy_token_spans: bool = False
    prompt_token_distributions: bool = False
    prompt_token_distribution_version: int | None = None
    max_prompt_logprobs: int | None = None
    full_prompt_distribution: bool = False
    prompt_distribution_temperature: Literal["unit_only"] | None = None
    token_space_fingerprint: str | None = None
    logical_vocab_size: int | None = None

    @classmethod
    def _prompt_distribution_fields(cls) -> tuple[str, ...]:
        return (
            "prompt_token_distribution_version",
            "max_prompt_logprobs",
            "prompt_distribution_temperature",
            "token_space_fingerprint",
            "logical_vocab_size",
        )

    def model_post_init(self, __context: object) -> None:
        values = tuple(
            getattr(self, field) for field in self._prompt_distribution_fields()
        )
        if self.prompt_token_distributions:
            if any(value is None for value in values):
                raise ValueError(
                    "prompt-token distributions require a version, capacity, "
                    "and temperature contract"
                )
            if self.prompt_token_distribution_version != 1:
                raise ValueError(
                    "unsupported prompt-token distribution protocol version"
                )
            if self.max_prompt_logprobs is None or self.max_prompt_logprobs <= 0:
                raise ValueError("max_prompt_logprobs must be positive")
            if not self.token_space_fingerprint:
                raise ValueError("token_space_fingerprint must not be empty")
            if self.logical_vocab_size is None or self.logical_vocab_size <= 0:
                raise ValueError("logical_vocab_size must be positive")
        elif (
            any(value is not None for value in values) or self.full_prompt_distribution
        ):
            raise ValueError("prompt-token distribution details require the capability")

    @classmethod
    def openai_compatible(cls) -> "ServingCapabilities":
        return cls(runtime="openai_compatible", protocol_version=0)

    def require(self, feature: ServingFeature, *, operation: str) -> None:
        if not getattr(self, feature):
            raise RuntimeError(
                f"{operation} requires serving capability {feature!r}; "
                f"connected runtime is {self.runtime!r}."
            )


async def discover_serving_capabilities(
    *,
    base_url: str,
    headers: dict[str, str] | None,
    allow_openai_compatible: bool,
) -> ServingCapabilities:
    url = f"{base_url.rstrip('/')}/art/capabilities"
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url, headers=headers)
    if response.status_code == 404 and allow_openai_compatible:
        return ServingCapabilities.openai_compatible()
    try:
        response.raise_for_status()
        return ServingCapabilities.model_validate(response.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(
            f"Serving runtime returned invalid ART capabilities from {url}."
        ) from exc
