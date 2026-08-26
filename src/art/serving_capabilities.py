from ipaddress import ip_address
from typing import Literal

import httpx
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    model_validator,
)

ART_SERVING_PROTOCOL_VERSION = 4

ServingFeature = Literal[
    "binary_routed_experts",
    "fast_metrics",
    "inplace_lora_load",
    "in_flight_lora_updates",
    "policy_token_spans",
]


class FastMetricsEndpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: AnyHttpUrl

    @model_validator(mode="after")
    def _validate_url(self) -> "FastMetricsEndpoint":
        host = self.url.host
        if host is None:
            raise ValueError("fast metrics URL must include a host")
        try:
            unspecified = ip_address(host.strip("[]")).is_unspecified
        except ValueError:
            unspecified = False
        if unspecified:
            raise ValueError("fast metrics URL must not use an unspecified host")
        return self


class FastMetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    source: Literal["art_vllm_runtime"]
    last_update_unix_s: FiniteFloat = Field(ge=0)
    record_count: int = Field(ge=0)
    engine_count: int = Field(ge=0)
    metrics: dict[str, FiniteFloat]
    process_uuid: str = Field(min_length=1)
    generation: int = Field(ge=0)


class ServingCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Literal["openai_compatible", "art_vllm"]
    protocol_version: int
    binary_routed_experts: bool = False
    fast_metrics: FastMetricsEndpoint | None = None
    inplace_lora_load: bool = False
    in_flight_lora_updates: bool = False
    policy_token_spans: bool = False

    @model_validator(mode="after")
    def _validate_protocol(self) -> "ServingCapabilities":
        expected = ART_SERVING_PROTOCOL_VERSION if self.runtime == "art_vllm" else 0
        if self.protocol_version != expected:
            raise ValueError(
                f"{self.runtime} serving protocol must be version {expected}"
            )
        return self

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
