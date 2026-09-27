from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr


class _ContentOnly:
    eos_token_id: int | None = None

    def __init__(self, assistant_extent: str = "") -> None:
        self.assistant_extent = assistant_extent
        if assistant_extent:
            self.eos_token_id = ord(assistant_extent)

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"input_ids": list(map(ord, text))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids: list[int], **kwargs: Any) -> str:
        return "".join(map(chr, ids))

    def apply_chat_template(
        self, messages: list[dict[str, Any]], *, tokenize: bool = True, **kwargs: Any
    ) -> str | list[int]:
        text = "".join(
            (message.get("content") or "")
            + (self.assistant_extent if message.get("role") == "assistant" else "")
            for message in messages
        )
        return list(map(ord, text)) if tokenize else text


@pytest.mark.parametrize(
    "assistant,prompt,extent,accepted",
    [
        ("", [113], "", True),
        ("", [1, 113], "", True),
        (None, [1, 113], "", True),
        ("", [72, 113], "H", True),
        ("H", [1, 72, 113], "", False),
    ],
)
def test_initial_request_role_requires_observed_extent(
    assistant: str | None, prompt: list[int], extent: str, accepted: bool
) -> None:
    exchange = _chat_exchange(prompt, [97])
    request = [] if assistant is None else [{"role": "assistant", "content": assistant}]
    request.append({"role": "user", "content": "q"})
    exchange.request["messages"] = cast(Any, request)
    choice = exchange.response.choices[0]
    choice.message.content = "a"
    choice.finish_reason = "length"
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )
    history = trajectory.histories()[0]
    assert isinstance(history, tr.ChatCompletionsHistory)
    before = trajectory.model_dump_json()
    tokenizer = _ContentOnly(extent)
    if not accepted:
        with pytest.raises(ValueError, match="Cannot preserve assistant boundaries"):
            history.tokenize(tokenizer=tokenizer)
    else:
        result = history.tokenize(tokenizer=tokenizer)
        assert result.tokens == prompt + [97]
        assert result.logprobs[len(prompt) :] == [-9.7]
        assert [i for i, f in enumerate(result.flags) if f & tr.TokenFlag.SAMPLED] == [
            len(prompt)
        ]
        assert [i for i, f in enumerate(result.flags) if f & tr.TokenFlag.OUTPUT] == [
            len(prompt)
        ]
        assert not result.flags[-1] & tr.TokenFlag.STOP
        if extent:
            assert result.flags[0] == (
                tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT | tr.TokenFlag.STOP
            )
        else:
            assert result.flags[:-1] == [tr.TokenFlag.EXACT] * len(prompt)
    assert trajectory.model_dump_json() == before
