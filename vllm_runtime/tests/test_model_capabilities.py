from types import SimpleNamespace

from art_vllm_runtime import model_capabilities
import pytest


def _capabilities(
    monkeypatch,
    architecture: str,
    *,
    model: str = "test/model",
    capability_base_model: str | None = None,
    backend_version: str = "0.25.1",
) -> dict[str, object]:
    monkeypatch.setattr(model_capabilities, "version", lambda _package: backend_version)
    model_config = SimpleNamespace(
        model=model,
        hf_config=SimpleNamespace(architectures=[architecture]),
    )
    return model_capabilities.model_backend_capabilities(
        model_config,
        capability_base_model=capability_base_model,
        binary_route_capture=True,
    )


@pytest.mark.parametrize(
    ("architecture", "status", "lora"),
    [
        ("Qwen3_5MoeForConditionalGeneration", "unvalidated", "native"),
        ("Gemma4ForCausalLM", "unvalidated", "art_runtime_patch"),
        ("DeepseekV4ForCausalLM", "unvalidated", "art_runtime_patch"),
        ("GptOssForCausalLM", "unsupported", "unavailable"),
        ("UnknownForCausalLM", "unvalidated", "unvalidated"),
    ],
)
def test_model_backend_capabilities_fail_closed(
    monkeypatch, architecture: str, status: str, lora: str
) -> None:
    capabilities = _capabilities(monkeypatch, architecture)

    assert capabilities["validation_status"] == status
    assert capabilities["lora_implementation"] == lora
    assert capabilities["route_capture_dcp"] == 1
    assert capabilities["route_capture_pcp"] == 1


@pytest.mark.parametrize("backend_version", ["0.25.1", "0.25.1+cu129"])
def test_qwen35_moe_exact_model_backend_is_validated(
    monkeypatch, backend_version: str
) -> None:
    capabilities = _capabilities(
        monkeypatch,
        "Qwen3_5MoeForConditionalGeneration",
        model="Qwen/Qwen3.5-35B-A3B",
        backend_version=backend_version,
    )

    assert capabilities["base_model"] == "Qwen/Qwen3.5-35B-A3B"
    assert capabilities["backend_version"] == backend_version
    assert capabilities["validation_status"] == "validated"
    assert capabilities["lora_implementation"] == "native"


def test_qwen35_moe_fixture_uses_trusted_canonical_identity(monkeypatch) -> None:
    capabilities = _capabilities(
        monkeypatch,
        "Qwen3_5MoeForConditionalGeneration",
        model="/run/e2e_throughput/production_width_model",
        capability_base_model="Qwen/Qwen3.5-35B-A3B",
    )

    assert capabilities["base_model"] == "Qwen/Qwen3.5-35B-A3B"
    assert capabilities["validation_status"] == "validated"


@pytest.mark.parametrize(
    "capability_base_model",
    [None, "Qwen/Qwen3.5-27B"],
)
def test_qwen35_moe_fixture_without_exact_canonical_identity_fails_closed(
    monkeypatch, capability_base_model: str | None
) -> None:
    capabilities = _capabilities(
        monkeypatch,
        "Qwen3_5MoeForConditionalGeneration",
        model="/run/e2e_throughput/production_width_model",
        capability_base_model=capability_base_model,
    )

    assert capabilities["validation_status"] == "unvalidated"


@pytest.mark.parametrize(
    ("model", "architecture", "backend_version"),
    [
        (
            "Qwen/Qwen3.5-27B",
            "Qwen3_5MoeForConditionalGeneration",
            "0.25.1",
        ),
        (
            "Qwen/Qwen3.5-35B-A3B",
            "Qwen3_5MoeForCausalLM",
            "0.25.1",
        ),
        (
            "Qwen/Qwen3.5-35B-A3B",
            "Qwen3_5MoeForConditionalGeneration",
            "0.25.2",
        ),
    ],
)
def test_qwen_models_outside_validated_tuple_fail_closed(
    monkeypatch, model: str, architecture: str, backend_version: str
) -> None:
    capabilities = _capabilities(
        monkeypatch,
        architecture,
        model=model,
        backend_version=backend_version,
    )

    assert capabilities["validation_status"] == "unvalidated"
    assert capabilities["lora_implementation"] == "native"


def test_model_backend_capabilities_require_declared_architecture(monkeypatch) -> None:
    monkeypatch.setattr(model_capabilities, "version", lambda _package: "0.25.1")
    model_config = SimpleNamespace(
        model="test/model", hf_config=SimpleNamespace(architectures=[])
    )

    with pytest.raises(RuntimeError, match="does not declare"):
        model_capabilities.model_backend_capabilities(
            model_config, binary_route_capture=False
        )
