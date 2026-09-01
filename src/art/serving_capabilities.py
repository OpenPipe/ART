from ipaddress import ip_address
import re
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

from art.runtime_attestation import RuntimeArchitectureAttestation

ART_SERVING_PROTOCOL_VERSION = 11

ART_PRIVATE_CACHE_IDENTITY_HEADER = "x-art-cache-identity"
ART_PRIVATE_REQUEST_IDENTITY_HEADER = "x-art-request-identity"
ART_PRIVATE_ROUTE_CAPTURE_HEADER = "x-art-route-capture"
ART_PRIVATE_ROUTE_MAX_BYTES_HEADER = "x-art-route-max-bytes"
ART_PRIVATE_RUN_ID_HEADER = "x-art-run-id"
ART_PRIVATE_SERVICE_TIER_HEADER = "x-art-service-tier"
ART_PRIVATE_TENANT_ID_HEADER = "x-art-tenant-id"
ART_RUNTIME_TARGET_HEADER = "x-art-runtime-target"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

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


class ServingProfileIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str = Field(min_length=1)
    model_identifier: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    model_support_key: str = Field(min_length=1)
    handler_name: str = Field(min_length=1)
    lora_rank: int = Field(ge=1)
    lora_alpha: FiniteFloat = Field(gt=0)
    lora_target_modules: tuple[str, ...] = Field(min_length=1)
    trainer_dtype: Literal["bfloat16", "float16", "float32"]
    route_replay: bool
    lora_transport: Literal["local", "nixl"]
    retained_route_transport: Literal["none", "holder_local", "caios_lota"]
    retained_route_max_bytes: int = Field(ge=0)
    retained_route_max_bundles: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ServingProfileIdentity":
        bounds = (self.retained_route_max_bytes, self.retained_route_max_bundles)
        if self.retained_route_transport != "none":
            valid_bounds = all(value > 0 for value in bounds)
        else:
            valid_bounds = bounds == (0, 0)
        if not valid_bounds:
            raise ValueError(
                "retained route transport and bounds must be present together"
            )
        if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
            raise ValueError("LoRA target modules must be unique")
        return self


class ServingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[3] = 3
    identity: ServingProfileIdentity
    architecture: RuntimeArchitectureAttestation
    runtime_model: str = Field(min_length=1)
    runtime_revision: str | None = None
    tokenizer: str = Field(min_length=1)
    tokenizer_revision: str | None = None
    model_dtype: str = Field(min_length=1)
    quantization: str | None = None
    tensor_parallel_size: int = Field(ge=1)
    pipeline_parallel_size: int = Field(ge=1)
    data_parallel_size: int = Field(ge=1)
    prefill_context_parallel_size: int = Field(ge=1)
    enable_expert_parallel: bool
    max_model_len: int = Field(ge=1)
    max_num_batched_tokens: int = Field(ge=1)
    max_num_seqs: int = Field(ge=1)
    max_num_partial_prefills: int = Field(ge=1)
    kv_cache_dtype: str = Field(min_length=1)
    kv_block_size: int = Field(ge=1)
    kv_block_bytes_per_rank: int = Field(ge=1)
    kv_capacity_blocks_per_rank: int = Field(ge=1)
    kv_capacity_bytes_per_rank: int = Field(ge=1)
    prefix_caching: bool
    prefix_hash_algorithm: str = Field(min_length=1)
    max_loras: int = Field(ge=1)
    max_lora_rank: int = Field(ge=1)
    lora_dtype: str = Field(min_length=1)
    speculative_method: str | None = None
    multi_token_prediction: bool
    exact_token_ids: Literal[True] = True
    selected_token_logprobs: Literal[True] = True
    policy_span_schema: Literal["prompt_completion_v1"] = "prompt_completion_v1"
    cache_transition: Literal["policy_history_route_salt_v1"] = (
        "policy_history_route_salt_v1"
    )
    lora_update_semantics: Literal["holder_local_in_flight_v1"] = (
        "holder_local_in_flight_v1"
    )
    route_capture_format: Literal["art_inference_route_bundle_v1"] | None = None

    @model_validator(mode="after")
    def _validate_profile(self) -> "ServingProfile":
        if self.architecture.runtime_kind != "inference":
            raise ValueError("serving architecture has another runtime kind")
        if (
            self.architecture.base_model != self.identity.base_model
            or self.architecture.model_support_key != self.identity.model_support_key
            or self.architecture.handler_name != self.identity.handler_name
        ):
            raise ValueError("serving architecture differs from profile identity")
        if (
            self.runtime_model != self.identity.model_identifier
            or (self.runtime_revision or "default") != self.identity.model_revision
        ):
            raise ValueError("runtime model identity disagrees with the launch profile")
        if self.max_lora_rank < self.identity.lora_rank:
            raise ValueError("runtime max LoRA rank is below the training layout")
        if self.identity.route_replay != (self.route_capture_format is not None):
            raise ValueError("route replay identity and runtime capture disagree")
        if self.multi_token_prediction != (self.speculative_method == "mtp"):
            raise ValueError("MTP flag and speculative method disagree")
        if self.kv_capacity_bytes_per_rank != (
            self.kv_capacity_blocks_per_rank * self.kv_block_bytes_per_rank
        ):
            raise ValueError("KV cache capacity disagrees with its block geometry")
        return self


class PairedInferenceEndpoint(BaseModel):
    """Private, incarnation-fenced service endpoint for one paired runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    url: AnyHttpUrl
    target_id: str = Field(pattern=_SHA256_PATTERN)
    runtime_generation: int = Field(ge=0)
    runtime_source_id: str = Field(min_length=1, max_length=512)
    runtime_source_epoch: int = Field(ge=0)
    authorization_token: str = Field(min_length=32, repr=False)
    profile: ServingProfile
    fast_metrics: FastMetricsEndpoint | None = None

    def request_headers(
        self,
        *,
        request_identity: str,
        cache_identity: str,
        tenant_id: str,
        run_id: str,
        service_tier: str,
        route_capture_max_bytes: int | None = None,
    ) -> dict[str, str]:
        for name, value in (
            ("request_identity", request_identity),
            ("cache_identity", cache_identity),
        ):
            if not re.fullmatch(_SHA256_PATTERN, value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name, value, maximum in (
            ("tenant_id", tenant_id, 255),
            ("run_id", run_id, 255),
            ("service_tier", service_tier, 128),
        ):
            if not value or len(value) > maximum:
                raise ValueError(f"{name} must contain 1-{maximum} characters")
        headers = {
            **self.runtime_headers(),
            ART_PRIVATE_REQUEST_IDENTITY_HEADER: request_identity,
            ART_PRIVATE_CACHE_IDENTITY_HEADER: cache_identity,
            ART_PRIVATE_TENANT_ID_HEADER: tenant_id,
            ART_PRIVATE_RUN_ID_HEADER: run_id,
            ART_PRIVATE_SERVICE_TIER_HEADER: service_tier,
            "x-request-id": request_identity,
        }
        if route_capture_max_bytes is not None:
            identity = self.profile.identity
            if (
                isinstance(route_capture_max_bytes, bool)
                or route_capture_max_bytes < 1
                or identity.retained_route_transport == "none"
                or route_capture_max_bytes > identity.retained_route_max_bytes
            ):
                raise ValueError(
                    "route capture bytes must fit the retained-route capacity"
                )
            headers[ART_PRIVATE_ROUTE_CAPTURE_HEADER] = "retained"
            headers[ART_PRIVATE_ROUTE_MAX_BYTES_HEADER] = str(route_capture_max_bytes)
        return headers

    def runtime_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.authorization_token}",
            ART_RUNTIME_TARGET_HEADER: self.target_id,
        }


class ServingCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Literal["openai_compatible", "art_vllm"]
    protocol_version: int
    binary_routed_experts: bool = False
    fast_metrics: FastMetricsEndpoint | None = None
    inplace_lora_load: bool = False
    in_flight_lora_updates: bool = False
    policy_token_spans: bool = False
    profile: ServingProfile | None = None

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
