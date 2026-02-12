"""Unit tests for ART's vLLM server helpers."""

import pytest

pytest.importorskip("cloudpickle")
pytest.importorskip("vllm")

from vllm.lora.request import LoRARequest

from art.vllm.server import _normalize_lora_request


def test_normalize_lora_request_maps_lora_fields() -> None:
    class DummyLoRARequest:
        lora_name = "adapter-name"
        lora_int_id = 7
        lora_path = "/tmp/adapter"
        base_model_name = "base-model"
        load_inplace = True

    normalized = _normalize_lora_request(DummyLoRARequest())

    assert isinstance(normalized, LoRARequest)
    assert normalized.lora_name == "adapter-name"
    assert normalized.lora_int_id == 7
    assert normalized.lora_path == "/tmp/adapter"
    assert normalized.base_model_name == "base-model"
    assert normalized.load_inplace is True
