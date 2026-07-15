from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict

ServingFeature = Literal[
    "binary_routed_experts",
    "fast_metrics",
    "inplace_lora_load",
    "in_flight_lora_updates",
    "policy_token_spans",
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
