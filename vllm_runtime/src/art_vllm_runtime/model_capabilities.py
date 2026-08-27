"""Fail-closed model/backend conformance declarations for ART vLLM."""

from importlib.metadata import version
from typing import Any

_QWEN_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
_GEMMA4_ARCHITECTURES = frozenset(
    {"Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"}
)
_DSV4_ARCHITECTURES = frozenset({"DeepseekV4ForCausalLM"})
_GPT_OSS_ARCHITECTURES = frozenset(
    {"GptOssForCausalLM", "GptOssForConditionalGeneration"}
)
_VALIDATED_QWEN35_MOE_MODEL = "Qwen/Qwen3.5-35B-A3B"
_VALIDATED_QWEN35_MOE_ARCHITECTURES = frozenset({"Qwen3_5MoeForConditionalGeneration"})
_VALIDATED_QWEN35_MOE_VLLM_VERSIONS = frozenset({"0.25.1", "0.25.1+cu129"})


def model_backend_capabilities(
    model_config: Any,
    *,
    capability_base_model: str | None = None,
    binary_route_capture: bool,
) -> dict[str, object]:
    """Describe implementation support without inventing conformance evidence."""

    hf_config = model_config.hf_config
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
    if not architectures:
        raise RuntimeError("loaded model config does not declare an architecture")
    architecture_set = frozenset(architectures)
    # Local conformance fixtures may load from a generated directory while
    # retaining the canonical model contract asserted by the ART launcher.
    loaded_model_source = str(model_config.model)
    base_model = capability_base_model or loaded_model_source
    backend_version = version("vllm")

    if architecture_set & _GPT_OSS_ARCHITECTURES:
        validation_status = "unsupported"
        lora_implementation = "unavailable"
    elif (
        base_model == _VALIDATED_QWEN35_MOE_MODEL
        and architecture_set == _VALIDATED_QWEN35_MOE_ARCHITECTURES
        and backend_version in _VALIDATED_QWEN35_MOE_VLLM_VERSIONS
    ):
        validation_status = "validated"
        lora_implementation = "native"
    elif architecture_set <= _QWEN_ARCHITECTURES:
        # Architecture dispatch is not evidence that this exact model revision,
        # runtime build, topology, and route layout passed deployment conformance.
        validation_status = "unvalidated"
        lora_implementation = "native"
    elif architecture_set & _GEMMA4_ARCHITECTURES:
        validation_status = "unvalidated"
        lora_implementation = "art_runtime_patch"
    elif architecture_set & _DSV4_ARCHITECTURES:
        validation_status = "unvalidated"
        lora_implementation = "art_runtime_patch"
    else:
        validation_status = "unvalidated"
        lora_implementation = "unvalidated"

    return {
        "schema_version": 1,
        "base_model": base_model,
        "architectures": architectures,
        "backend": "vllm",
        "backend_version": backend_version,
        "validation_status": validation_status,
        "lora_implementation": lora_implementation,
        "exact_token_ids": True,
        "exact_token_logprobs": True,
        "prompt_policy_spans": True,
        "decode_policy_spans": True,
        "in_flight_lora_updates": True,
        "active_request_kv_continuation": True,
        "new_request_policy_cache_isolation": True,
        "binary_route_capture": binary_route_capture,
        # Route/token ordering has not been validated under DCP or PCP > 1.
        "route_capture_dcp": 1,
        "route_capture_pcp": 1,
    }
