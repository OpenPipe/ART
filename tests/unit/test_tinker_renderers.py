import json
from typing import cast

from art.tinker.cookbook_v import renderers
from art.tinker.cookbook_v.tokenizer_utils import Tokenizer
from art.tinker.renderers import get_renderer_name
from art.tinker_native.data import convert_openai_messages_to_renderer_format


class FakeTokenizer:
    name_or_path = "fake/qwen3_5"

    _SPECIAL_TOKENS = ("<|im_end|>", "</think>")

    def __init__(self) -> None:
        self._text_to_id: dict[str, int] = {}
        self._id_to_text: dict[int, str] = {}
        self._next_id = 100
        for idx, token in enumerate(self._SPECIAL_TOKENS, start=1):
            self._text_to_id[token] = idx
            self._id_to_text[idx] = token

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens: list[int] = []
        idx = 0
        while idx < len(text):
            matched = False
            for special in self._SPECIAL_TOKENS:
                if text.startswith(special, idx):
                    tokens.append(self._text_to_id[special])
                    idx += len(special)
                    matched = True
                    break
            if matched:
                continue

            char = text[idx]
            if char not in self._text_to_id:
                self._text_to_id[char] = self._next_id
                self._id_to_text[self._next_id] = char
                self._next_id += 1
            tokens.append(self._text_to_id[char])
            idx += 1
        return tokens

    def decode(self, tokens: int | list[int]) -> str:
        if isinstance(tokens, int):
            return self._id_to_text[tokens]
        return "".join(self._id_to_text[token] for token in tokens)


def _decode_model_input(tokenizer: FakeTokenizer, model_input: object) -> str:
    tokens: list[int] = []
    for chunk in model_input.chunks:  # type: ignore[attr-defined]
        assert hasattr(chunk, "tokens"), f"Unexpected non-text chunk: {chunk!r}"
        tokens.extend(list(chunk.tokens))
    return tokenizer.decode(tokens)


def _get_test_renderer(name: str, tokenizer: FakeTokenizer) -> renderers.Renderer:
    return renderers.get_renderer(name, cast(Tokenizer, tokenizer))


def test_get_renderer_name_autodetects_qwen3_5() -> None:
    assert get_renderer_name("Qwen/Qwen3.5-35B-A3B") == "qwen3_5"


def test_qwen3_5_generation_prompt_matches_hf_suffixes() -> None:
    tokenizer = FakeTokenizer()

    renderer = _get_test_renderer("qwen3_5", tokenizer)
    prompt = renderer.build_generation_prompt(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Interim answer"},
        ]
    )
    rendered = _decode_model_input(tokenizer, prompt)
    assert (
        "<|im_start|>assistant\n<think>\n\n</think>\n\nInterim answer<|im_end|>"
        in rendered
    )
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")

    disable_renderer = _get_test_renderer("qwen3_5_disable_thinking", tokenizer)
    disable_prompt = disable_renderer.build_generation_prompt([])
    disable_rendered = _decode_model_input(tokenizer, disable_prompt)
    assert disable_rendered == "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def test_qwen3_5_parse_response_handles_xml_tool_calls() -> None:
    tokenizer = FakeTokenizer()
    renderer = _get_test_renderer("qwen3_5", tokenizer)

    response = tokenizer.encode(
        "  reasoning  </think>\n\nAnswer first.\n\n"
        "<tool_call>\n"
        "<function=lookup_weather>\n"
        "<parameter=city>\n"
        "San Francisco\n"
        "</parameter>\n"
        "<parameter=days>\n"
        "3\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
        "<|im_end|>"
    )

    message, success = renderer.parse_response(response)

    assert success is True
    assert message["content"] == [
        {"type": "thinking", "thinking": "reasoning"},
        {"type": "text", "text": "Answer first.\n\n"},
    ]
    assert "unparsed_tool_calls" not in message
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0].function.name == "lookup_weather"
    assert json.loads(message["tool_calls"][0].function.arguments) == {
        "city": "San Francisco",
        "days": 3,
    }


def test_qwen3_5_to_openai_message_uses_mapping_tool_arguments() -> None:
    tokenizer = FakeTokenizer()
    renderer = _get_test_renderer("qwen3_5", tokenizer)

    message: renderers.Message = {
        "role": "assistant",
        "content": [
            renderers.ThinkingPart(type="thinking", thinking="reason"),
            renderers.TextPart(type="text", text="Answer"),
        ],
        "tool_calls": [
            renderers.ToolCall(
                function=renderers.ToolCall.FunctionBody(
                    name="lookup_weather",
                    arguments=json.dumps({"city": "San Francisco", "days": 3}),
                )
            )
        ],
    }

    openai_message = renderer.to_openai_message(message)

    assert openai_message["content"] == "Answer"
    assert openai_message["reasoning_content"] == "reason"
    assert openai_message["tool_calls"][0]["function"]["arguments"] == {
        "city": "San Francisco",
        "days": 3,
    }


def test_convert_openai_messages_to_renderer_format_stringifies_dict_arguments() -> (
    None
):
    tokenizer = FakeTokenizer()
    renderer = _get_test_renderer("qwen3_5", tokenizer)

    converted = convert_openai_messages_to_renderer_format(
        [
            {
                "role": "assistant",
                "content": "Calling a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": {"city": "San Francisco", "days": 3},
                        },
                    }
                ],
            }
        ],
        tools=None,
        renderer=renderer,
    )

    tool_call = converted[0]["tool_calls"][0]
    assert tool_call.function.name == "lookup_weather"
    assert json.loads(tool_call.function.arguments) == {
        "city": "San Francisco",
        "days": 3,
    }


def test_get_renderer_supports_kimi_k25_factory() -> None:
    tokenizer = FakeTokenizer()

    renderer = _get_test_renderer("kimi_k25", tokenizer)

    assert renderer.__class__.__name__ == "KimiK25Renderer"
