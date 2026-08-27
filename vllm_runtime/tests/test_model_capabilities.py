from types import SimpleNamespace

from art_vllm_runtime import model_capabilities
import pytest


def _capabilities(monkeypatch, architecture: str) -> dict[str, object]:
    monkeypatch.setattr(model_capabilities, "version", lambda _package: "0.25.1")
    model_config = SimpleNamespace(
        model="test/model",
        hf_config=SimpleNamespace(architectures=[architecture]),
    )
    return model_capabilities.model_backend_capabilities(
        model_config, binary_route_capture=True
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


def test_model_backend_capabilities_require_declared_architecture(monkeypatch) -> None:
    monkeypatch.setattr(model_capabilities, "version", lambda _package: "0.25.1")
    model_config = SimpleNamespace(
        model="test/model", hf_config=SimpleNamespace(architectures=[])
    )

    with pytest.raises(RuntimeError, match="does not declare"):
        model_capabilities.model_backend_capabilities(
            model_config, binary_route_capture=False
        )
