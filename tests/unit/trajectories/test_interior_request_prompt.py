from copy import deepcopy
import math
from typing import cast

from openai.types.chat import ChatCompletionMessageParam
import pytest
from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange

import art.trajectories as tr


@pytest.mark.parametrize("change", ["none", "after_role", "inside_role"])
def test_interior_roles_require_the_complete_final_request(change):
    tokenizer = _CharacterTemplateTokenizer()
    first_messages = [{"role": "user", "content": "q0"}]
    first_prompt = tokenizer.apply_chat_template(
        first_messages, add_generation_prompt=True
    )
    assert isinstance(first_prompt, list)
    output = tokenizer._encode("answer§")
    first = _chat_exchange(first_prompt, output)
    first.request["messages"] = cast(
        list[ChatCompletionMessageParam], deepcopy(first_messages)
    )
    messages = [
        *first_messages,
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "history"},
        {"role": "user", "content": "last"},
    ]
    recorded = deepcopy(messages)
    if change == "after_role":
        recorded[-1]["content"] = "LAST"
    elif change == "inside_role":
        recorded[-2]["content"] = "HISTORY"
    prompt = tokenizer.apply_chat_template(recorded, add_generation_prompt=True)
    assert isinstance(prompt, list)
    second = _chat_exchange(prompt, output, offset=1)
    second.request["messages"] = cast(
        list[ChatCompletionMessageParam], deepcopy(messages)
    )
    value = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )
    before = value.model_dump_json()
    if change != "none":
        with pytest.raises(ValueError, match="[Rr]equest roles|assistant boundaries"):
            value.tokenize(tokenizer=tokenizer, multi_history=True)
    else:
        histories = value.tokenize(tokenizer=tokenizer, multi_history=True).histories
        assert len(histories) == 1
        actual = histories[0]
        assert actual.tokens == prompt + output
        sampled = (
            tr.TokenFlag.EXACT
            | tr.TokenFlag.SAMPLED
            | tr.TokenFlag.OUTPUT
            | tr.TokenFlag.ASSISTANT
        )
        for start in (len(first_prompt), len(prompt)):
            assert actual.logprobs[start : start + len(output)] == [
                -token / 10 for token in output
            ]
            assert all(
                flag & sampled == sampled
                for flag in actual.flags[start : start + len(output)]
            )
        interior = range(
            len(first_prompt) + len(output) + len("u1"), len(prompt) - len("last")
        )
        assert all(actual.flags[i] & tr.TokenFlag.ASSISTANT for i in interior)
        assert all(
            not actual.flags[i] & (tr.TokenFlag.SAMPLED | tr.TokenFlag.OUTPUT)
            for i in interior
        )
        assert all(math.isnan(actual.logprobs[i]) for i in interior)
    assert value.model_dump_json() == before


@pytest.mark.parametrize("body", ["text", "tool", "reasoning"])
def test_initial_request_roles_do_not_reencode_later_native_bodies(body):
    from openai.types.chat import ChatCompletion

    tokenizer = _CharacterTemplateTokenizer()
    messages = [
        {"role": "user", "content": "intro"},
        {"role": "assistant", "content": "history"},
        {"role": "user", "content": "turn0"},
    ]
    prompt = tokenizer._encode("introhistory§turn0")
    output = tokenizer._encode("different native body§")
    first = _chat_exchange(prompt, output)
    first.request["messages"] = cast(
        list[ChatCompletionMessageParam], deepcopy(messages)
    )
    payload = first.response.model_dump(mode="python")
    message = payload["choices"][0]["message"]
    if body == "tool":
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": "public",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key": 1}'},
            }
        ]
    elif body == "reasoning":
        message["reasoning_content"] = "structured reasoning"
    first.response = ChatCompletion.model_validate(payload)
    final_prompt = [*prompt, *output, *tokenizer._encode("turn1")]
    second = _chat_exchange(final_prompt, tokenizer._encode("answer§"), offset=1)
    second.request["messages"] = cast(
        list[ChatCompletionMessageParam],
        [
            *deepcopy(messages),
            first.response.choices[0].message.model_dump(
                mode="python", exclude_none=True
            ),
            {"role": "user", "content": "turn1"},
        ],
    )
    value = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )
    before = value.model_dump_json()
    actual = value.tokenize(tokenizer=tokenizer, multi_history=True).histories
    assert len(actual) == 1
    result = actual[0]
    assert result.tokens == [*final_prompt, *tokenizer._encode("answer§")]
    for start, ids in [
        (len(prompt), output),
        (len(final_prompt), tokenizer._encode("answer§")),
    ]:
        assert result.logprobs[start : start + len(ids)] == [
            -token / 10 for token in ids
        ]
        assert all(
            flag & tr.TokenFlag.SAMPLED
            for flag in result.flags[start : start + len(ids)]
        )
    historical = slice(len("intro"), len("introhistory§"))
    assert all(flag & tr.TokenFlag.ASSISTANT for flag in result.flags[historical])
    assert result.flags[len("introhistory")] & tr.TokenFlag.STOP
    assert not any(
        flag & (tr.TokenFlag.SAMPLED | tr.TokenFlag.OUTPUT)
        for flag in result.flags[historical]
    )
    assert all(math.isnan(lp) for lp in result.logprobs[historical])
    assert value.model_dump_json() == before
