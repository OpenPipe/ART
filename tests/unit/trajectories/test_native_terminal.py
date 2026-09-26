from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import pytest
from test_tokenize import (
    _character_template_history,
    _CharacterTemplateTokenizer,
    _chat_exchange,
)

import art.trajectories as tr
from art.trajectories import _tokenize as module


@pytest.fixture(autouse=True)
def restore_warning_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_WARNED_PREFIX_RETOKENIZATION", False)


class ProjectedToolTokenizer(_CharacterTemplateTokenizer):
    def apply_chat_template(
        self,
        messages: Any,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any:
        text = ""
        for message in messages:
            text += message["role"] + ":" + str(message.get("content") or "")
            if message.get("tool_calls"):
                text += json.dumps(message["tool_calls"], sort_keys=True)
            if message["role"] == "assistant":
                text += "§"
        if add_generation_prompt:
            text += "assistant:"
        return self._encode(text) if tokenize else text


def projected_history(
    *, finish: str, sampled_eos: bool, earlier_length: bool
) -> tuple[
    tr.Trajectory,
    ProjectedToolTokenizer,
    list[tuple[list[int], list[int], list[float]]],
]:
    tokenizer = ProjectedToolTokenizer()
    exchanges = []
    messages: list[dict[str, Any]] = []
    records = []
    for index in range(2 if earlier_length else 1):
        messages.append({"role": "user", "content": f"query{index}"})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        terminal = not earlier_length or index == 1
        raw = ("raw tool output " * 8) if terminal else "earlier response\n\n"
        message = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "public_call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
            if terminal
            else {"role": "assistant", "content": raw}
        )
        output = tokenizer._encode(raw + ("§" if terminal and sampled_eos else ""))
        exchange = _chat_exchange(prompt, output, offset=index)
        exchange.request["messages"] = cast(
            list[ChatCompletionMessageParam], deepcopy(messages)
        )
        payload = exchange.response.model_dump(mode="python")
        payload["choices"][0]["message"] = message
        payload["choices"][0]["finish_reason"] = finish if terminal else "length"
        exchange.response = ChatCompletion.model_validate(payload)
        exchanges.append(exchange)
        logprobs = exchange.response.choices[0].logprobs
        assert logprobs is not None and logprobs.content is not None
        records.append((prompt, output, [entry.logprob for entry in logprobs.content]))
        messages.append(message)
    return (
        tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=exchanges)),
        tokenizer,
        records,
    )


@pytest.mark.parametrize("finish", ["stop", "tool_calls", "length"])
@pytest.mark.parametrize("sampled_eos", [False, True])
@pytest.mark.parametrize("earlier_length", [False, True])
def test_complete_native_terminal_does_not_reconstruct_projected_tool_body(
    finish: str,
    sampled_eos: bool,
    earlier_length: bool,
) -> None:
    trajectory, tokenizer, records = projected_history(
        finish=finish, sampled_eos=sampled_eos, earlier_length=earlier_length
    )
    original = trajectory.model_dump(mode="python")
    result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert len(result.histories) == 1
    history = result.histories[0]
    assert history.tokens == records[-1][0] + records[-1][1]
    required = (
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.ASSISTANT
    )
    for prompt, output, expected in records:
        start, end = len(prompt), len(prompt) + len(output)
        assert history.tokens[:end] == prompt + output
        assert all(flag & required == required for flag in history.flags[start:end])
        assert history.logprobs[start:end] == expected
    final_start = len(records[-1][0])
    sampled_stops = [
        i
        for i in range(final_start, len(history.tokens))
        if history.flags[i] & tr.TokenFlag.STOP
    ]
    assert sampled_stops == (
        [len(history.tokens) - 1] if sampled_eos and finish != "length" else []
    )
    if earlier_length:
        boundary = len(records[0][0]) + len(records[0][1])
        assert history.flags[boundary] == tr.TokenFlag.EXACT | tr.TokenFlag.STOP
        assert math.isnan(history.logprobs[boundary])
    assert tr.first_occurrence_masks(
        result.histories, where=tr.TokenFlag.OUTPUT
    ) == tr.first_occurrence_masks(result.histories, where=tr.TokenFlag.SAMPLED)
    assert trajectory.model_dump(mode="python") == original


def test_terminal_length_native_path_does_not_load_or_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exchange = _chat_exchange([1], [2, 3])
    exchange.response.choices[0].finish_reason = "length"
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange])
    )

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("complete terminal native data needs no renderer")

    monkeypatch.setattr(module, "_load_tokenizer", unexpected)
    monkeypatch.setattr(module, "_tokenizer_config", unexpected)
    result = trajectory.tokenize()
    assert result.tokens == [1, 2, 3]
    assert (
        result.flags[-1]
        == tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.ASSISTANT
    )


def test_explicit_template_override_retains_rendered_terminal_tail() -> None:
    history, tokenizer, _ = _character_template_history(terminal_sampled_stop=False)
    value = history.tokenize(tokenizer=tokenizer, chat_template="explicit renderer")
    assert value.tokens[-1] == 9
    assert (
        value.flags[-1]
        == tr.TokenFlag.ASSISTANT | tr.TokenFlag.OUTPUT | tr.TokenFlag.STOP
    )


@pytest.mark.parametrize("finish,sampled_eos", [("tool_calls", False), ("stop", True)])
@pytest.mark.parametrize("standalone", [False, True])
def test_request_owned_assistant_roles_survive_terminal_native_tool_output(
    finish: str, sampled_eos: bool, standalone: bool
) -> None:
    trajectory, tokenizer, records = projected_history(
        finish=finish, sampled_eos=sampled_eos, earlier_length=False
    )
    exchange = trajectory.exchanges.chat_completions[0]
    exchange.request["messages"].insert(
        0, {"role": "assistant", "content": "historical context"}
    )
    prompt = tokenizer.apply_chat_template(
        exchange.request["messages"], add_generation_prompt=True
    )
    payload = exchange.response.model_dump(mode="python")
    payload["prompt_token_ids"] = prompt
    payload["choices"][0]["prompt_token_ids"] = prompt
    exchange.response = ChatCompletion.model_validate(payload)
    original = trajectory.model_dump(mode="python")
    if standalone:
        selected = trajectory.histories()[0]
        assert isinstance(selected, tr.ChatCompletionsHistory)
    else:
        selected = trajectory
    result = selected.tokenize(tokenizer=tokenizer)
    output = records[-1][1]
    assert result.tokens == prompt + output
    prefix_flags = result.flags[: len(prompt)]
    assert any(flag & tr.TokenFlag.ASSISTANT for flag in prefix_flags)
    assert any(flag & tr.TokenFlag.STOP for flag in prefix_flags)
    assert not any(
        flag & (tr.TokenFlag.OUTPUT | tr.TokenFlag.SAMPLED) for flag in prefix_flags
    )
    assert result.logprobs[len(prompt) :] == records[-1][2]
    expected = [tr.TokenFlag.EXACT] * len(prompt)
    start = len(tokenizer._encode("assistant:"))
    end = len(tokenizer._encode("assistant:historical context§"))
    expected[start:end] = [tr.TokenFlag.EXACT | tr.TokenFlag.ASSISTANT] * (end - start)
    expected[end - 1] |= tr.TokenFlag.STOP
    assert prefix_flags == expected
    assert trajectory.model_dump(mode="python") == original


def test_unresolved_nonterminal_stop_still_loads_boundary_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory, tokenizer, records = projected_history(
        finish="length", sampled_eos=False, earlier_length=True
    )
    trajectory.exchanges.chat_completions[0].response.choices[0].finish_reason = "stop"
    loaded = []
    monkeypatch.setattr(
        module,
        "_tokenizer_config",
        lambda *args: module._TokenizerConfig(base_model="test/model"),
    )

    def load(config: Any) -> Any:
        loaded.append(config)
        return tokenizer

    monkeypatch.setattr(module, "_load_tokenizer", load)
    result = trajectory.tokenize()
    assert len(loaded) == 1
    assert result.tokens == records[-1][0] + records[-1][1]
    boundary = len(records[0][0]) + len(records[0][1])
    assert (
        result.flags[boundary]
        == tr.TokenFlag.EXACT
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT
        | tr.TokenFlag.STOP
    )
    assert math.isnan(result.logprobs[boundary])
