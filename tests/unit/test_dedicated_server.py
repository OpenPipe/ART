"""Unit tests for dedicated vLLM server entry point."""

import pytest

pytest.importorskip("cloudpickle")
pytest.importorskip("vllm")

from art.vllm.dedicated_server import parse_args


def test_parse_args_required():
    args = parse_args(
        [
            "--model",
            "Qwen/Qwen3-14B",
            "--port",
            "8000",
            "--cuda-visible-devices",
            "1",
            "--lora-path",
            "/tmp/checkpoints/0000",
            "--served-model-name",
            "my-model@0",
        ]
    )
    assert args.model == "Qwen/Qwen3-14B"
    assert args.port == 8000
    assert args.cuda_visible_devices == "1"
    assert args.lora_path == "/tmp/checkpoints/0000"
    assert args.served_model_name == "my-model@0"
    assert args.host == "127.0.0.1"
    assert args.engine_args_json == "{}"
    assert args.server_args_json == "{}"


def test_parse_args_with_engine_args():
    args = parse_args(
        [
            "--model",
            "test-model",
            "--port",
            "9000",
            "--cuda-visible-devices",
            "2",
            "--lora-path",
            "/tmp/lora",
            "--served-model-name",
            "test@1",
            "--engine-args-json",
            '{"max_model_len": 4096}',
        ]
    )
    assert args.engine_args_json == '{"max_model_len": 4096}'


def test_parse_args_custom_host():
    args = parse_args(
        [
            "--model",
            "test-model",
            "--port",
            "8000",
            "--cuda-visible-devices",
            "0",
            "--lora-path",
            "/tmp/lora",
            "--served-model-name",
            "test@0",
            "--host",
            "0.0.0.0",
        ]
    )
    assert args.host == "0.0.0.0"


def test_parse_args_with_server_args():
    args = parse_args(
        [
            "--model",
            "test-model",
            "--port",
            "8000",
            "--cuda-visible-devices",
            "1",
            "--lora-path",
            "/tmp/lora",
            "--served-model-name",
            "test@0",
            "--server-args-json",
            '{"enable_auto_tool_choice": true, "tool_call_parser": "hermes"}',
        ]
    )
    import json

    server_args = json.loads(args.server_args_json)
    assert server_args["enable_auto_tool_choice"] is True
    assert server_args["tool_call_parser"] == "hermes"
