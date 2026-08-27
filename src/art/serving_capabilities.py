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

ART_SERVING_PROTOCOL_VERSION = 8

ServingFeature = Literal[
    "binary_routed_experts",
    "fast_metrics",
    "inplace_lora_load",
    "in_flight_lora_updates",
    "policy_token_spans",
    "presigned_route_uploads",
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


class ModelBackendCapabilities(BaseModel):
    """Conformance evidence for the one model/backend pair served here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    base_model: str = Field(min_length=1)
    architectures: tuple[str, ...] = Field(min_length=1)
    backend: Literal["vllm"]
    backend_version: str = Field(min_length=1)
    validation_status: Literal["validated", "unvalidated", "unsupported"]
    lora_implementation: Literal[
        "native", "art_runtime_patch", "unvalidated", "unavailable"
    ]
    exact_token_ids: bool
    exact_token_logprobs: bool
    prompt_policy_spans: bool
    decode_policy_spans: bool
    in_flight_lora_updates: bool
    active_request_kv_continuation: bool
    new_request_policy_cache_isolation: bool
    binary_route_capture: bool = Field(
        description=(
            "Request-scoped route capture is supported by both OpenAI chat and "
            "completion generation endpoints under the advertised constraints."
        )
    )
    route_capture_dcp: int = Field(ge=1)
    route_capture_pcp: int = Field(ge=1)

    def require_trainable_generation(
        self,
        *,
        expected_base_model: str | None = None,
        require_binary_routes: bool = False,
    ) -> None:
        if expected_base_model is not None and self.base_model != expected_base_model:
            raise RuntimeError(
                f"Remote RL expected model {expected_base_model!r}, but the serving "
                f"capability document describes {self.base_model!r}."
            )
        if self.validation_status != "validated":
            raise RuntimeError(
                f"Remote RL is not validated for {self.base_model!r} on "
                f"{self.backend} {self.backend_version}; capability status is "
                f"{self.validation_status!r}."
            )
        required = {
            "exact_token_ids": self.exact_token_ids,
            "exact_token_logprobs": self.exact_token_logprobs,
            "prompt_policy_spans": self.prompt_policy_spans,
            "decode_policy_spans": self.decode_policy_spans,
            "in_flight_lora_updates": self.in_flight_lora_updates,
            "active_request_kv_continuation": self.active_request_kv_continuation,
            "new_request_policy_cache_isolation": (
                self.new_request_policy_cache_isolation
            ),
        }
        if self.lora_implementation not in {"native", "art_runtime_patch"}:
            required["model_lora_support"] = False
        if require_binary_routes:
            required["binary_route_capture"] = self.binary_route_capture
        missing = sorted(name for name, available in required.items() if not available)
        if missing:
            raise RuntimeError(
                f"Remote RL capability contract is incomplete for "
                f"{self.base_model!r}: {missing}."
            )


class ServingCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Literal["openai_compatible", "art_vllm"]
    protocol_version: int
    binary_routed_experts: bool = False
    fast_metrics: FastMetricsEndpoint | None = None
    inplace_lora_load: bool = False
    in_flight_lora_updates: bool = False
    policy_token_spans: bool = False
    presigned_route_uploads: bool = False
    model_backend: ModelBackendCapabilities | None = None

    @model_validator(mode="after")
    def _validate_protocol(self) -> "ServingCapabilities":
        expected = ART_SERVING_PROTOCOL_VERSION if self.runtime == "art_vllm" else 0
        if self.protocol_version != expected:
            raise ValueError(
                f"{self.runtime} serving protocol must be version {expected}"
            )
        if self.runtime == "openai_compatible" and self.model_backend is not None:
            raise ValueError(
                "OpenAI-compatible capabilities cannot claim ART conformance"
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

    def require_trainable_generation(
        self,
        *,
        expected_base_model: str | None = None,
        require_binary_routes: bool = False,
    ) -> None:
        if self.runtime != "art_vllm":
            return
        if self.model_backend is None:
            raise RuntimeError(
                "ART vLLM did not advertise a model/backend conformance document."
            )
        self.model_backend.require_trainable_generation(
            expected_base_model=expected_base_model,
            require_binary_routes=require_binary_routes,
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
        capabilities = ServingCapabilities.model_validate(response.json())
        if capabilities.runtime == "art_vllm" and capabilities.model_backend is None:
            raise ValueError("ART vLLM capabilities omitted model/backend conformance")
        return capabilities
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(
            f"Serving runtime returned invalid ART capabilities from {url}."
        ) from exc
