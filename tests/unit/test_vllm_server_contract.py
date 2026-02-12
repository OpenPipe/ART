"""Unit tests for ART's vLLM server compatibility helpers."""

import pytest

pytest.importorskip("cloudpickle")
pytest.importorskip("vllm")

from vllm.lora.request import LoRARequest

from art.vllm.server import _get_openai_serving_models_module, _normalize_lora_request


def test_get_openai_serving_models_module_exposes_expected_class() -> None:
    serving_models = _get_openai_serving_models_module()
    assert hasattr(serving_models, "OpenAIServingModels")


def test_normalize_lora_request_handles_versioned_fields() -> None:
    class DummyLoRARequest:
        lora_name = "adapter-name"
        lora_int_id = 7
        lora_path = "/tmp/adapter"
        base_model_name = "base-model"
        long_lora_max_len = 4096
        load_inplace = True

    normalized = _normalize_lora_request(DummyLoRARequest())

    assert isinstance(normalized, LoRARequest)
    assert normalized.lora_name == "adapter-name"
    assert normalized.lora_int_id == 7
    assert normalized.lora_path == "/tmp/adapter"
    if hasattr(normalized, "base_model_name"):
        assert normalized.base_model_name == "base-model"
    if hasattr(normalized, "long_lora_max_len"):
        assert normalized.long_lora_max_len == 4096
    if hasattr(normalized, "load_inplace"):
        assert normalized.load_inplace is True
