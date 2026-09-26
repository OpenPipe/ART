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
