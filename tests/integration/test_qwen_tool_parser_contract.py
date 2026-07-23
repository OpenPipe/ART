import json
from pathlib import Path

import pytest

_vllm_protocol = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
_vllm_tool_parsers = pytest.importorskip("vllm.tool_parsers")


_CONTRACT = json.loads(
    (Path(__file__).parents[1] / "support/qwen_tool_parser_contract.json").read_text()
)


class _UnusedTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}


def test_qwen3_xml_matches_tinker_qwen35_tool_call_contract() -> None:
    request = _vllm_protocol.ChatCompletionRequest(
        model="Qwen/Qwen3.6-35B-A3B",
        messages=[{"role": "user", "content": "Check the weather."}],
        tools=_CONTRACT["tools"],
        tool_choice="auto",
    )
    parser_type = _vllm_tool_parsers.ToolParserManager.get_tool_parser("qwen3_xml")
    parser = parser_type(_UnusedTokenizer(), request.tools)  # type: ignore[arg-type]

    parsed = parser.extract_tool_calls(_CONTRACT["model_output"], request)

    assert parsed.tools_called is True
    normalized_tool_calls = [
        {
            "name": tool_call.function.name,
            "arguments": json.loads(tool_call.function.arguments),
        }
        for tool_call in parsed.tool_calls
    ]
    assert normalized_tool_calls == _CONTRACT["expected_tool_calls"]
