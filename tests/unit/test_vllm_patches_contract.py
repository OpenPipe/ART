"""Unit tests for ART's vLLM patch contract."""

import importlib
from typing import Any, cast

import pytest

pytest.importorskip("cloudpickle")
pytest.importorskip("vllm")

from art.vllm.patches import (
    patch_tool_parser_manager,
    patch_transformers_v5_compat,
    subclass_chat_completion_request,
)


def test_subclass_chat_completion_request_forces_logprobs() -> None:
    protocol = importlib.import_module(
        "vllm.entrypoints.openai.chat_completion.protocol"
    )
    original = getattr(protocol, "ChatCompletionRequest")

    try:
        subclass_chat_completion_request()
        request_cls = getattr(protocol, "ChatCompletionRequest")
        request = request_cls(
            messages=[{"role": "user", "content": "hello"}],
            model="dummy-model",
        )
        assert request.logprobs is True
        assert request.top_logprobs == 0
    finally:
        setattr(protocol, "ChatCompletionRequest", original)


def test_patch_tool_parser_manager_falls_back_to_empty_delta_message() -> None:
    protocol = importlib.import_module("vllm.entrypoints.openai.engine.protocol")
    DeltaMessage = protocol.DeltaMessage

    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    class DummyToolParser:
        @staticmethod
        def extract_tool_calls_streaming(*_args, **_kwargs):
            return None

    original_get_tool_parser = ToolParserManager.get_tool_parser

    try:
        setattr(
            ToolParserManager,
            "get_tool_parser",
            classmethod(lambda _cls, _name: DummyToolParser),
        )
        patch_tool_parser_manager()

        parser_cls = ToolParserManager.get_tool_parser("dummy")
        result = parser_cls.extract_tool_calls_streaming("", "", "", [], [], [], None)  # ty:ignore[missing-argument,invalid-argument-type]

        assert isinstance(result, DeltaMessage)
    finally:
        setattr(ToolParserManager, "get_tool_parser", original_get_tool_parser)


def test_patch_transformers_v5_compat_normalizes_rope_ignore_keys() -> None:
    from transformers.configuration_utils import PretrainedConfig

    patch_transformers_v5_compat()

    class DummyRopeConfig:
        default_theta = 10000.0
        rope_parameters = None

        def standardize_rope_params(self) -> None:
            pass

        def validate_rope(self, ignore_keys=None) -> None:
            self.ignore_keys = ignore_keys

    dummy = DummyRopeConfig()
    PretrainedConfig.convert_rope_params_to_dict(
        cast(Any, dummy),
        ignore_keys_at_rope_validation=cast(Any, ["mrope_section"]),
        partial_rotary_factor=0.25,
    )

    assert dummy.ignore_keys == {"mrope_section", "partial_rotary_factor"}
