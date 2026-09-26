"""Public three-exchange discriminator; no production monkeypatches."""

from copy import deepcopy
import math
from typing import Any, cast

from openai.types.chat import ChatCompletion
import pytest
from test_recorded_prompt_roles import _interior_historical_case
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


def case(native_later=True, mismatch=None, third=True):
    value, tokenizer, records = _interior_historical_case()
    exchanges = value.exchanges.chat_completions
    second = exchanges[1]
    prompt = list(records[1][0])
    output = list(map(ord, "native1§" if native_later else "answer1§"))
    payload = second.response.model_dump(mode="python")
    choice = payload["choices"][0]
    choice["token_ids"] = output
    choice["logprobs"]["content"] = [
        dict(token=f"token_id:{token}", logprob=-token / 10, bytes=[], top_logprobs=[])
        for token in output
    ]
    second.response = ChatCompletion.model_validate(payload)
    records[1] = (prompt, output)
    if mismatch == "query":
        second.request["messages"][-1]["content"] = "NEXT QUERY"
    elif mismatch == "role":
        second.request["messages"][-2]["content"] = "x</think>Y"
    if third:
        final_messages = [
            *deepcopy(second.request["messages"]),
            {"role": "assistant", "content": "answer1"},
            {"role": "user", "content": "final query"},
        ]
        final_prompt = [*prompt, *output, *map(ord, "final query")]
        final_output = list(map(ord, "answer2§"))
        last = _chat_exchange(final_prompt, final_output, offset=2)
        last.request["messages"] = cast(Any, final_messages)
        last.request["chat_template"] = tokenizer.chat_template
        payload = last.response.model_dump(mode="python")
        payload["choices"][0]["message"]["content"] = "answer2"
        last.response = ChatCompletion.model_validate(payload)
        exchanges.append(last)
        records.append((final_prompt, final_output))
    return value, tokenizer, records


def role_proof(exchange, tokenizer):
    prompt = exchange.response.choices[0].model_extra["prompt_token_ids"]
    return module._recorded_prompt_role_masks(
        exchange.request["messages"],
        [None] * len(exchange.request["messages"]),
        prompt,
        tokenizer=tokenizer,
        template=tokenizer.chat_template,
        tools=None,
        kwargs={},
    )


def assert_exact(value, tokenizer, records):
    before = value.model_dump_json()
    result = value.tokenize(tokenizer=tokenizer, multi_history=True)
    assert len(result.histories) == 1
    actual = result.histories[0]
    assert actual.tokens == records[-1][0] + records[-1][1]
    sampled = (
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.ASSISTANT
    )
    expected_sampled = set()
    for prompt, output in records:
        assert actual.tokens[: len(prompt)] == prompt
        span = range(len(prompt), len(prompt) + len(output))
        expected_sampled.update(span)
        assert actual.logprobs[len(prompt) : len(prompt) + len(output)] == [
            -token / 10 for token in output
        ]
        assert all(actual.flags[i] & sampled == sampled for i in span)
        assert actual.flags[len(prompt) + len(output) - 1] & tr.TokenFlag.STOP
    assert {
        i for i, flag in enumerate(actual.flags) if flag & tr.TokenFlag.SAMPLED
    } == expected_sampled
    # Original public parser produces y§ for x</think>y. This request-owned
    # interior assistant precedes the second query/output; it is never sampled.
    start = len(records[0][0]) + len(records[0][1]) + len("middle query")
    assert actual.tokens[start : start + 2] == list(map(ord, "y§"))
    assert all(
        actual.flags[i] & tr.TokenFlag.ASSISTANT for i in range(start, start + 2)
    )
    assert actual.flags[start + 1] & tr.TokenFlag.STOP
    assert all(
        not actual.flags[i] & (tr.TokenFlag.SAMPLED | tr.TokenFlag.OUTPUT)
        for i in range(start, start + 2)
    )
    assert all(math.isnan(actual.logprobs[i]) for i in range(start, start + 2))
    assert value.model_dump_json() == before


def test_first_sufficient_original_request_preserves_later_native_body():
    value, tokenizer, records = case()
    # Establish the actual gap before invoking the public route: exchange 2
    # supplies the complete historical role/query proof; exchange 3 cannot
    # re-encode the independently exact sampled body native1§ from answer1.
    assert role_proof(value.exchanges.chat_completions[1], tokenizer) is not None
    assert role_proof(value.exchanges.chat_completions[2], tokenizer) is None
    assert_exact(value, tokenizer, records)


@pytest.mark.parametrize("third,native_later", [(False, True), (True, False)])
def test_adjacent_valid_earlier_and_render_congruent_controls(third, native_later):
    value, tokenizer, records = case(native_later=native_later, third=third)
    assert_exact(value, tokenizer, records)


@pytest.mark.parametrize("mismatch", ["query", "role"])
def test_complete_request_mismatch_still_refuses(mismatch):
    value, tokenizer, _ = case(mismatch=mismatch)
    before = value.model_dump_json()
    assert role_proof(value.exchanges.chat_completions[1], tokenizer) is None
    with pytest.raises(ValueError, match="request roles|assistant boundaries"):
        value.tokenize(tokenizer=tokenizer, multi_history=True)
    assert value.model_dump_json() == before
