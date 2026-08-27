import pytest

from art.serving_capabilities import (
    ART_SERVING_PROTOCOL_VERSION,
    ModelBackendCapabilities,
    ServingCapabilities,
)


def _model_backend(**updates: object) -> ModelBackendCapabilities:
    values: dict[str, object] = {
        "schema_version": 1,
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "architectures": ("Qwen3_5MoeForConditionalGeneration",),
        "backend": "vllm",
        "backend_version": "0.25.1",
        "validation_status": "validated",
        "lora_implementation": "native",
        "exact_token_ids": True,
        "exact_token_logprobs": True,
        "prompt_policy_spans": True,
        "decode_policy_spans": True,
        "in_flight_lora_updates": True,
        "active_request_kv_continuation": True,
        "new_request_policy_cache_isolation": True,
        "binary_route_capture": True,
        "route_capture_dcp": 1,
        "route_capture_pcp": 1,
    }
    values.update(updates)
    return ModelBackendCapabilities.model_validate(values)


def test_validated_model_backend_satisfies_trainable_contract() -> None:
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        model_backend=_model_backend(),
    )

    capabilities.require_trainable_generation()


@pytest.mark.parametrize("status", ["unvalidated", "unsupported"])
def test_unvalidated_model_backend_fails_closed(status: str) -> None:
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        model_backend=_model_backend(validation_status=status),
    )

    with pytest.raises(RuntimeError, match="not validated"):
        capabilities.require_trainable_generation()


def test_missing_required_generation_semantic_fails_closed() -> None:
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        model_backend=_model_backend(new_request_policy_cache_isolation=False),
    )

    with pytest.raises(RuntimeError, match="new_request_policy_cache_isolation"):
        capabilities.require_trainable_generation()


def test_model_backend_identity_must_match_requested_training_model() -> None:
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        model_backend=_model_backend(),
    )

    with pytest.raises(RuntimeError, match="expected model"):
        capabilities.require_trainable_generation(expected_base_model="other/model")


@pytest.mark.parametrize("implementation", ["unvalidated", "unavailable"])
def test_unvalidated_model_lora_support_fails_closed(implementation: str) -> None:
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        model_backend=_model_backend(lora_implementation=implementation),
    )

    with pytest.raises(RuntimeError, match="model_lora_support"):
        capabilities.require_trainable_generation()


def test_required_binary_routes_use_model_backend_evidence() -> None:
    capabilities = ServingCapabilities(
        runtime="art_vllm",
        protocol_version=ART_SERVING_PROTOCOL_VERSION,
        binary_routed_experts=True,
        model_backend=_model_backend(binary_route_capture=False),
    )

    with pytest.raises(RuntimeError, match="binary_route_capture"):
        capabilities.require_trainable_generation(require_binary_routes=True)
