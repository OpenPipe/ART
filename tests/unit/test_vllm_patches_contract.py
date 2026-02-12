"""Unit tests for ART's vLLM patch contract."""

import importlib

import pytest

pytest.importorskip("cloudpickle")
pytest.importorskip("vllm")

from art.vllm.patches import patch_tool_parser_manager, subclass_chat_completion_request


def get_chat_protocol_module():
    try:
        return importlib.import_module("vllm.entrypoints.openai.protocol")
    except ModuleNotFoundError:
        return importlib.import_module("vllm.entrypoints.openai.chat_completion.protocol")


def get_delta_message_class():
    try:
        protocol = importlib.import_module("vllm.entrypoints.openai.protocol")
    except ModuleNotFoundError:
        protocol = importlib.import_module("vllm.entrypoints.openai.engine.protocol")
    return protocol.DeltaMessage


def test_subclass_chat_completion_request_forces_logprobs() -> None:
    protocol = get_chat_protocol_module()
    original = protocol.ChatCompletionRequest

    try:
        subclass_chat_completion_request()
        request = protocol.ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            model="dummy-model",
        )
        assert request.logprobs is True
        assert request.top_logprobs == 0
    finally:
        protocol.ChatCompletionRequest = original


def test_patch_tool_parser_manager_falls_back_to_empty_delta_message() -> None:
    DeltaMessage = get_delta_message_class()

    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

    class DummyToolParser:
        @staticmethod
        def extract_tool_calls_streaming(*_args, **_kwargs):
            return None

    original_get_tool_parser = ToolParserManager.get_tool_parser

    try:
        ToolParserManager.get_tool_parser = classmethod(
            lambda _cls, _name: DummyToolParser
        )
        patch_tool_parser_manager()

        parser_cls = ToolParserManager.get_tool_parser("dummy")
        result = parser_cls.extract_tool_calls_streaming()

        assert isinstance(result, DeltaMessage)
    finally:
        ToolParserManager.get_tool_parser = original_get_tool_parser
