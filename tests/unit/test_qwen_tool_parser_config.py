from art.megatron.model_support.handlers.qwen3_5 import QWEN3_5_MOE_HANDLER


def test_qwen35_uses_model_native_vllm_tool_parser() -> None:
    assert QWEN3_5_MOE_HANDLER.vllm_server_args() == {"tool_call_parser": "qwen3_xml"}
