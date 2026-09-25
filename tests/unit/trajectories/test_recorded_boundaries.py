from __future__ import annotations

import math
from typing import Any, cast

import pytest
from test_tokenize import _character_template_history

from art.trajectories import TokenFlag, first_occurrence_masks
from art.trajectories import _tokenize as module


@pytest.fixture(autouse=True)
def restore_warning_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_WARNED_PREFIX_RETOKENIZATION", False)


def assert_same(left: Any, right: Any) -> None:
    assert left.history is right.history
    assert left.model == right.model
    assert left.tokens == right.tokens
    assert left.flags == right.flags
    assert len(left.logprobs) == len(right.logprobs)
    assert all(
        a == b or math.isnan(a) and math.isnan(b)
        for a, b in zip(left.logprobs, right.logprobs, strict=True)
    )
    for flag in (
        TokenFlag.SAMPLED,
        TokenFlag.OUTPUT,
        TokenFlag.ASSISTANT,
        TokenFlag.STOP,
    ):
        assert first_occurrence_masks([left], where=flag) == first_occurrence_masks(
            [right], where=flag
        )


@pytest.mark.parametrize("terminal_sampled_stop", [False, True])
def test_recorded_length_boundaries_do_not_reencode_sampled_content(
    monkeypatch: pytest.MonkeyPatch, terminal_sampled_stop: bool
) -> None:
    history, tokenizer, _ = _character_template_history(
        terminal_sampled_stop=terminal_sampled_stop
    )
    original = history.model_dump(mode="python")
    helper = module._tokenize_recorded_chat_boundaries
    admissions = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        value = helper(*args, **kwargs)
        admissions.append(value)
        return value

    monkeypatch.setattr(module, "_tokenize_recorded_chat_boundaries", observe)
    rendered = []
    original_render = tokenizer.apply_chat_template

    def render(*args: Any, **kwargs: Any) -> Any:
        rendered.append(kwargs.get("tokenize", True))
        return original_render(*args, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    tokenized = history.tokenize(tokenizer=tokenizer)
    assert admissions == [tokenized]
    assert rendered and not any(rendered)
    monkeypatch.setattr(
        module, "_tokenize_recorded_chat_boundaries", lambda *args, **kwargs: None
    )
    baseline = history.tokenize(tokenizer=tokenizer)
    assert_same(tokenized, baseline)
    assert history.model_dump(mode="python") == original


@pytest.mark.parametrize("change", ["missing_tail", "reasoning", "override", "edited"])
def test_unproved_boundaries_preserve_existing_path(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    history, tokenizer, _ = _character_template_history(
        omit_length_tail=change == "missing_tail",
        length_reasoning="not in native output" if change == "reasoning" else None,
    )
    kwargs = (
        {"chat_template": "explicit caller template"} if change == "override" else {}
    )
    if change == "edited":
        history.messages[3]["content"] = "edited"
    helper = module._tokenize_recorded_chat_boundaries
    admissions = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        value = helper(*args, **kwargs)
        admissions.append(value)
        return value

    def result() -> Any:
        try:
            return history.tokenize(tokenizer=tokenizer, **kwargs)
        except (ValueError, AssertionError) as error:
            return type(error), str(error)

    monkeypatch.setattr(module, "_tokenize_recorded_chat_boundaries", observe)
    candidate = result()
    if change != "reasoning":
        assert not any(admissions)
    monkeypatch.setattr(
        module, "_tokenize_recorded_chat_boundaries", lambda *args, **kwargs: None
    )
    baseline = result()
    if isinstance(candidate, tuple):
        assert candidate == baseline
    else:
        assert_same(candidate, baseline)


@pytest.mark.parametrize("tool_position", [0, 1])
def test_recorded_tool_boundaries_preserve_native_conditioning(
    monkeypatch: pytest.MonkeyPatch, tool_position: int
) -> None:
    from copy import deepcopy
    import json

    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
    from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange

    import art.trajectories as tr

    class Tokenizer(_CharacterTemplateTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tokenize: bool = True,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> str | list[int]:
            del kwargs
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

    tokenizer = Tokenizer()
    exchanges = []
    messages: list[dict[str, Any]] = []
    expected_spans = []
    for index in range(2):
        messages.append({"role": "user", "content": f"query{index}"})
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        message = {"role": "assistant", "content": "answer"}
        if index == tool_position:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "public_call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"x":1}'},
                    }
                ],
            }
        completion = tokenizer.apply_chat_template(
            [*messages, message], add_generation_prompt=False
        )
        assert isinstance(prompt, list) and isinstance(completion, list)
        output = completion[len(prompt) :]
        if index == tool_position:
            output = output[:-1]  # Server stopped on a tool call without emitting EOS.
        exchange = _chat_exchange(prompt, output, offset=index)
        exchange.request["messages"] = cast(
            list[ChatCompletionMessageParam], deepcopy(messages)
        )
        payload = exchange.response.model_dump(mode="python")
        payload["choices"][0]["message"] = message
        payload["choices"][0]["finish_reason"] = (
            "tool_calls" if index == tool_position else "stop"
        )
        exchange.response = ChatCompletion.model_validate(payload)
        exchanges.append(exchange)
        messages.append(message)
        expected_spans.append((prompt, output))
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=exchanges)
    )
    before = trajectory.model_dump(mode="python")
    result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert len(result.histories) == 1
    tokenized = result.histories[0]
    for prompt, output in expected_spans:
        assert tokenized.tokens[: len(prompt)] == prompt
        assert tokenized.tokens[len(prompt) : len(prompt) + len(output)] == output
        assert all(
            flag & TokenFlag.SAMPLED
            for flag in tokenized.flags[len(prompt) : len(prompt) + len(output)]
        )
    assert sum(bool(flag & TokenFlag.STOP) for flag in tokenized.flags) == 2
    assert trajectory.model_dump(mode="python") == before
    monkeypatch.setattr(
        module, "_tokenize_recorded_chat_boundaries", lambda *args, **kwargs: None
    )
    try:
        baseline = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    except ValueError as error:
        assert "boundary" in str(error) or "prefix" in str(error)
    else:
        old = baseline.histories[0]
        if tool_position == 0:
            # The former text replacement deleted the served, nonsampled EOS:
            # its returned second response no longer had its recorded prompt.
            prompt, _ = expected_spans[1]
            assert old.tokens[: len(prompt)] != prompt
        else:
            # The template owns this EOS; it must not become a sampled token.
            assert old.tokens == tokenized.tokens[:-1]
        tool_prompt, tool_output = expected_spans[tool_position]
        stop_position = len(tool_prompt) + len(tool_output)
        assert tokenized.flags[stop_position] & TokenFlag.STOP
        assert not tokenized.flags[stop_position] & TokenFlag.SAMPLED


@pytest.mark.parametrize("footer", ["footer§", "user-owned footer"])
def test_custom_footer_is_not_inferred_as_an_assistant_boundary(
    monkeypatch: pytest.MonkeyPatch, footer: str
) -> None:
    history, tokenizer, _ = _character_template_history()
    original = tokenizer.apply_chat_template

    def render(
        messages: Any,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any:
        text = original(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if not add_generation_prompt and messages[-1]["role"] == "assistant":
            assert isinstance(text, str)
            text += footer
        assert isinstance(text, str)
        return tokenizer._encode(text) if tokenize else text

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    helper = module._tokenize_recorded_chat_boundaries
    outcomes = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        value = helper(*args, **kwargs)
        outcomes.append(value)
        return value

    monkeypatch.setattr(module, "_tokenize_recorded_chat_boundaries", observe)
    try:
        actual = history.tokenize(tokenizer=tokenizer)
    except ValueError as error:
        actual = type(error), str(error)
    assert outcomes == [None]
    monkeypatch.setattr(
        module, "_tokenize_recorded_chat_boundaries", lambda *args, **kwargs: None
    )
    try:
        expected = history.tokenize(tokenizer=tokenizer)
    except ValueError as error:
        expected = type(error), str(error)
    if isinstance(actual, tuple):
        assert actual == expected
    else:
        assert_same(actual, expected)


@pytest.mark.parametrize("logprob", [-0.3, math.nan, 1e100])
def test_copied_suffix_is_context_not_a_new_sampled_edge(logprob: float) -> None:
    from test_tokenize import _chat_exchange

    import art.trajectories as tr

    first = _chat_exchange([1], [2, 3])
    recorded = first.response.choices[0].logprobs
    assert recorded is not None and recorded.content is not None
    recorded.content[-1].logprob = logprob
    second = _chat_exchange([1, 3, 4], [5], offset=1)
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )
    original = trajectory.model_dump_json()
    tokenized = trajectory.tokenize(multi_history=True)
    assert len(tokenized.histories) == 2
    original_result, copied_result = tokenized.histories
    assert original_result.tokens == [1, 2, 3]
    assert original_result.flags == [
        TokenFlag.EXACT,
        (TokenFlag.EXACT | TokenFlag.SAMPLED | TokenFlag.ASSISTANT | TokenFlag.OUTPUT),
        (TokenFlag.EXACT | TokenFlag.SAMPLED | TokenFlag.ASSISTANT | TokenFlag.OUTPUT),
    ]
    assert (
        original_result.logprobs[2] == logprob
        or math.isnan(original_result.logprobs[2])
        and math.isnan(logprob)
    )
    assert copied_result.tokens == [1, 3, 4, 5]
    assert copied_result.flags == [
        TokenFlag.EXACT,
        TokenFlag.EXACT | TokenFlag.ASSISTANT | TokenFlag.OUTPUT,
        TokenFlag.EXACT,
        (TokenFlag.EXACT | TokenFlag.SAMPLED | TokenFlag.ASSISTANT | TokenFlag.OUTPUT),
    ]
    assert math.isnan(copied_result.logprobs[1])
    assert first_occurrence_masks(tokenized.histories, where=TokenFlag.SAMPLED) == [
        [False, True, True],
        [False, False, False, True],
    ]
    assert trajectory.model_dump_json() == original
    standalone = trajectory.histories()[1]
    assert isinstance(standalone, tr.ChatCompletionsHistory)
    with pytest.raises(ValueError, match="complete original sampled occurrence"):
        standalone.tokenize()


def test_length_copy_keeps_proven_synthetic_boundary_flags() -> None:
    from openai.types.chat import ChatCompletion
    from test_tokenize import _CharacterTemplateTokenizer, _chat_exchange

    import art.trajectories as tr

    class Tokenizer(_CharacterTemplateTokenizer):
        def apply_chat_template(
            self,
            messages: Any,
            *,
            tokenize: bool = True,
            add_generation_prompt: bool,
            **kwargs: Any,
        ) -> Any:
            text = "".join(
                str(message.get("reasoning") or message.get("reasoning_content") or "")
                + str(message.get("content") or "")
                + ("§" if message["role"] == "assistant" else "")
                for message in messages
            )
            return self._encode(text) if tokenize else text

    tokenizer = Tokenizer()
    prompt = tokenizer._encode("turn 0")
    first = _chat_exchange(prompt, tokenizer._encode("ranswer"))
    payload = first.response.model_dump(mode="python")
    payload["choices"][0]["message"]["reasoning_content"] = "r"
    payload["choices"][0]["finish_reason"] = "length"
    first.response = ChatCompletion.model_validate(payload)
    next_prompt = tokenizer._encode("turn 0answer§turn 1")
    second = _chat_exchange(next_prompt, tokenizer._encode("answer§"), offset=1)
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )
    result = trajectory.tokenize(tokenizer=tokenizer, multi_history=True)
    assert len(result.histories) == 2
    original, copied = result.histories
    assert original.tokens == tokenizer._encode("turn 0ranswer§")
    assert copied.tokens == tokenizer._encode("turn 0answer§turn 1answer§")
    copy_start, copy_end = len(prompt), len(prompt) + len("answer")
    assert copied.flags[copy_start:copy_end] == [
        TokenFlag.EXACT | TokenFlag.ASSISTANT | TokenFlag.OUTPUT
    ] * len("answer")
    assert all(math.isnan(lp) for lp in copied.logprobs[copy_start:copy_end])
    assert copied.flags[copy_end] == TokenFlag.EXACT | TokenFlag.STOP
    assert copied.tokens[: len(next_prompt)] == next_prompt
    standalone = trajectory.histories()[1]
    assert isinstance(standalone, tr.ChatCompletionsHistory)
    with pytest.raises(ValueError, match="complete original sampled occurrence"):
        standalone.tokenize(tokenizer=tokenizer)


@pytest.mark.parametrize(
    "tamper", ["model", "owner", "ids", "logprob", "sampled", "trace"]
)
def test_copied_context_requires_actual_prior_source_ownership(tamper: str) -> None:
    from test_tokenize import _chat_exchange

    import art.trajectories as tr

    first = _chat_exchange([1], [2, 3])
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first])
    )
    result, traces = module._tokenize_trajectory_with_trace(trajectory)
    prior, trace = result.histories[0], traces[0]
    assert isinstance(prior.history, tr.ChatCompletionsHistory)
    source = prior.history.message_sources[1]
    assert source is not None
    assert module._complete_source_is_represented(
        source.model_copy(), [1], [2, 3], [-0.2, -0.3], [(prior, trace)]
    )
    key = module._sampled_source_key(source)
    if tamper == "model":
        prior.model = "other/model"
    elif tamper == "owner":
        trace.sources[key] = source.model_copy(
            update={"exchange": first.model_copy(deep=True)}
        )
    elif tamper == "ids":
        prior.tokens[0] = 999
    elif tamper == "logprob":
        prior.logprobs[-1] = -999
    elif tamper == "sampled":
        prior.flags[-1] &= ~TokenFlag.SAMPLED
    else:
        trace.source_keys[-1] = None
    assert not module._complete_source_is_represented(
        source, [1], [2, 3], [-0.2, -0.3], [(prior, trace)]
    )


def test_context_copy_survives_compact_input_and_result_roundtrip() -> None:
    from test_tokenize import _chat_exchange

    import art.trajectories as tr

    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2, 3]),
                _chat_exchange([1, 3, 4], [5], offset=1),
            ]
        )
    )
    restored = tr.compact_validate(tr.compact_dump(trajectory), type=tr.Trajectory)
    result = trajectory.tokenize(multi_history=True)
    repeated = restored.tokenize(multi_history=True)
    decoded = tr.compact_validate(
        tr.compact_dump(result), type=tr.TokenizedMultiHistoryTrajectory
    )
    assert (
        result.model_dump_json()
        == repeated.model_dump_json()
        == decoded.model_dump_json()
    )


@pytest.mark.parametrize("change", ["template", "kwargs", "edited"])
def test_copied_context_explicit_rendering_keeps_generic_route(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from test_tokenize import _chat_exchange, _FakeTokenizer

    import art.trajectories as tr

    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(
            chat_completions=[
                _chat_exchange([1], [2, 3]),
                _chat_exchange([1, 3, 4], [5], offset=1),
            ]
        )
    )
    history = trajectory.histories()[1]
    assert isinstance(history, tr.ChatCompletionsHistory)
    assert module._partial_native_context(history)
    kwargs: dict[str, Any] = {}
    if change == "template":
        kwargs["chat_template"] = "explicit public renderer"
    elif change == "kwargs":
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        history.messages[0]["content"] = "edited context"
        history.message_sources[0] = None
    tokenizer = _FakeTokenizer()

    def not_native(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("explicit/edited rendering must not require prior native ownership")

    monkeypatch.setattr(module, "_complete_source_is_represented", not_native)
    monkeypatch.setattr(module, "_certify_copied_context", not_native)

    def outcome() -> Any:
        try:
            value = history.tokenize(tokenizer=tokenizer, **kwargs)
        except (ValueError, AssertionError) as error:
            return type(error), str(error)
        return (
            value.tokens,
            value.flags,
            [None if math.isnan(x) else x for x in value.logprobs],
        )

    candidate = outcome()
    assert tokenizer.calls  # The real generic renderer was reached.
    monkeypatch.setattr(module, "_partial_native_context", lambda history: [])
    assert outcome() == candidate


def test_native_record_reuse_is_local_and_observes_later_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import Counter

    from test_tokenize import _chat_exchange

    import art.trajectories as tr

    first = _chat_exchange([1], [2, 3])
    second = _chat_exchange([1, 2, 3, 4], [5, 6], offset=1)
    first.response.choices[0].index = 7  # Choice indices are not list positions.
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )
    other = _chat_exchange([8], [9])
    other.request["model"] = "other/model"
    other.response.model = "other/model"
    nested = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[other]))
    read = module._chat_source_record
    calls: Counter[int] = Counter()
    nested_results = []

    def observe(source: object) -> Any:
        exchange = getattr(source, "exchange")
        calls[id(exchange)] += 1
        if exchange is first:
            nested_results.append(nested.tokenize())
        return read(source)

    monkeypatch.setattr(module, "_chat_source_record", observe)
    result = trajectory.tokenize()
    assert result.tokens == [1, 2, 3, 4, 5, 6]
    assert calls[id(first)] == calls[id(second)] == calls[id(other)] == 1
    assert nested_results[0].tokens == [8, 9]
    assert nested_results[0].model == "other/model"
    lp = first.response.choices[0].logprobs
    assert lp is not None and lp.content is not None
    lp.content[1].logprob = -7.5
    repeated = trajectory.tokenize()
    assert repeated.logprobs[2] == -7.5
    assert result.logprobs[2] == -0.3
    assert calls[id(first)] == calls[id(second)] == calls[id(other)] == 2

    failure = ValueError("public native record failure")

    def fail(source: object) -> Any:
        raise failure

    monkeypatch.setattr(module, "_chat_source_record", fail)
    with pytest.raises(ValueError) as caught:
        trajectory.tokenize()
    assert caught.value is failure
    monkeypatch.setattr(module, "_chat_source_record", read)
    assert trajectory.tokenize().logprobs[2] == -7.5


@pytest.mark.parametrize("carrier", ["empty", "encoded", "mixed"])
def test_copied_context_preflight_uses_authoritative_token_carriers(
    carrier: str,
) -> None:
    from test_tokenize import _chat_exchange

    import art.trajectories as tr

    first = _chat_exchange([1], [2, 3])
    second = _chat_exchange([1, 3, 4], [5], offset=1)
    first_extra = first.response.choices[0].model_extra
    second_extra = second.response.choices[0].model_extra
    assert first_extra is not None and second_extra is not None
    first_extra["token_ids"] = (
        [] if carrier == "empty" else ["token_id:2", "token_id:3"]
    )
    if carrier == "encoded":
        first_extra["prompt_token_ids"] = ["token_id:1"]
        second_extra["prompt_token_ids"] = [
            "token_id:1",
            "token_id:3",
            "token_id:4",
        ]
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[first, second])
    )
    before = trajectory.model_dump_json()
    result = trajectory.tokenize(multi_history=True)
    assert [history.tokens for history in result.histories] == [[1, 2, 3], [1, 3, 4, 5]]
    assert result.histories[0].logprobs[1:] == [-0.2, -0.3]
    assert math.isnan(result.histories[1].logprobs[1])
    assert not result.histories[1].flags[1] & TokenFlag.SAMPLED
    assert trajectory.model_dump_json() == before


def test_messages_copied_context_requires_and_preserves_original_owner() -> None:
    from test_tokenize import _message_exchange

    import art.trajectories as tr

    first = _message_exchange(
        tr.MessagesRequest(
            model="test/model",
            max_tokens=16,
            messages=[{"role": "user", "content": "one"}],
        ),
        content=[
            {"type": "thinking", "thinking": "reason", "signature": "public"},
            {"type": "text", "text": "answer"},
        ],
        prompt_token_ids=[1],
        token_ids=[2, 3],
        logprobs=[-0.2, -0.3],
    )
    second = _message_exchange(
        tr.MessagesRequest(
            model="test/model",
            max_tokens=16,
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "two"},
            ],
        ),
        identifier="message-2",
        offset=1,
        content=[{"type": "text", "text": "next"}],
        prompt_token_ids=[1, 3, 4],
        token_ids=[5],
        logprobs=[-0.5],
    )
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(messages=[first, second])
    )
    before = trajectory.model_dump_json()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: Any) -> list[int]:
            return {
                "one": [1],
                "reason": [2],
                "answer": [3],
                "two": [4],
                "next": [5],
            }.get(text, [99])

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
            return {
                1: [1],
                2: [1, 2, 3] if messages[-1].get("reasoning") else [1, 3],
                3: [1, 3, 4],
                4: [1, 3, 4, 5],
            }[len(messages)]

    tokenizer = Tokenizer()
    result = trajectory.tokenize(multi_history=True, tokenizer=tokenizer)
    assert [history.tokens for history in result.histories] == [[1, 2, 3], [1, 3, 4, 5]]
    assert result.histories[0].logprobs[1:] == [-0.2, -0.3]
    assert (
        result.histories[0].flags[1:]
        == [
            TokenFlag.EXACT | TokenFlag.SAMPLED | TokenFlag.ASSISTANT | TokenFlag.OUTPUT
        ]
        * 2
    )
    assert (
        result.histories[1].flags[1]
        == TokenFlag.EXACT | TokenFlag.ASSISTANT | TokenFlag.OUTPUT
    )
    assert math.isnan(result.histories[1].logprobs[1])
    assert result.histories[1].logprobs[-1] == -0.5
    assert trajectory.model_dump_json() == before
    standalone = trajectory.histories()[1]
    assert isinstance(standalone, tr.AnthropicMessagesHistory)
    with pytest.raises(ValueError, match="complete original sampled occurrence"):
        standalone.tokenize(tokenizer=tokenizer)


def test_unsupported_native_body_decode_preserves_generic_boundary_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history, tokenizer, _ = _character_template_history()
    decode = tokenizer.decode
    probes = []

    def limited_decode(tokens: list[int], **kwargs: Any) -> str:
        if any(token in {7001, 7002} for token in tokens):
            probes.append(tuple(tokens))
            raise ValueError("served-only public token cannot be decoded")
        return decode(tokens, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", limited_decode)
    candidate = history.tokenize(tokenizer=tokenizer)
    assert probes
    monkeypatch.setattr(
        module, "_tokenize_recorded_chat_boundaries", lambda *a, **k: None
    )
    baseline = history.tokenize(tokenizer=tokenizer)
    assert_same(candidate, baseline)


def test_rendered_responses_copy_clears_old_logprob_without_sampling_it() -> None:
    from openai.types.responses import Response
    from test_tokenize import _response_exchange

    import art.trajectories as tr

    first = _response_exchange("first", 3, prompt_token_ids=[1])
    payload = first.response.model_dump(mode="python")
    text = payload["output"][0]
    text["content"][0]["logprobs"] = [
        {
            "token": "answer",
            "bytes": list(b"answer"),
            "logprob": -0.3,
            "top_logprobs": [],
        }
    ]
    payload["output"] = [
        {
            "id": "public-reasoning",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "think"}],
        },
        text,
    ]
    payload["token_generations"] = [
        {
            "prompt_token_ids": [1],
            "output_tokens": [
                {"token_id": 2, "logprob": -0.2},
                {"token_id": 3, "logprob": -0.3},
            ],
            "output_indices": [0, 1],
        }
    ]
    first.response = Response.model_validate(payload)
    second = _response_exchange(
        "second", 5, previous_response_id="first", offset=1, prompt_token_ids=[1, 3, 4]
    )
    payload = second.response.model_dump(mode="python")
    payload["status"] = "incomplete"
    payload["incomplete_details"] = {"reason": "max_output_tokens"}
    payload["output"][0]["content"][0]["text"] = "next"
    second.response = Response.model_validate(payload)
    trajectory = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(responses=[first, second])
    )
    before = trajectory.model_dump_json()

    class Tokenizer:
        def __call__(self, text: str, **kwargs: Any) -> list[int]:
            return {
                "turn 0": [1],
                "think": [2],
                "answer": [3],
                "turn 1": [4],
                "next": [5],
            }.get(text, [99])

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
            tokens = []
            for message in messages:
                if message.get("reasoning"):
                    tokens += self(message["reasoning"])
                if message.get("content"):
                    tokens += self(message["content"])
            return tokens

    result = trajectory.tokenize(multi_history=True, tokenizer=Tokenizer())
    assert [history.tokens for history in result.histories] == [[1, 2, 3], [1, 3, 4, 5]]
    assert result.histories[0].logprobs[1:] == [-0.2, -0.3]
    copied = result.histories[1]
    assert copied.flags[1] == TokenFlag.EXACT | TokenFlag.ASSISTANT | TokenFlag.OUTPUT
    assert math.isnan(copied.logprobs[1])
    assert copied.logprobs[-1] == -0.1
    assert trajectory.model_dump_json() == before


@pytest.mark.parametrize("opaque", ["image", "redacted_thinking"])
def test_complete_messages_records_do_not_require_a_chat_projection(
    monkeypatch: pytest.MonkeyPatch, opaque: str
) -> None:
    from test_tokenize import _message_exchange

    import art.trajectories as tr

    request = tr.MessagesRequest(
        model="test/model",
        max_tokens=16,
        messages=[{"role": "user", "content": "question"}],
    )
    if opaque == "image":
        request["messages"][0]["content"] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "public",
                },
            }
        ]
    exchange = _message_exchange(
        request,
        prompt_token_ids=[1, 2],
        token_ids=[3],
        logprobs=[-0.3],
        content=[{"type": "redacted_thinking", "data": "public"}]
        if opaque == "redacted_thinking"
        else None,
    )
    trajectory = tr.Trajectory(exchanges=tr.TrajectoryExchanges(messages=[exchange]))
    before = trajectory.model_dump_json()
    monkeypatch.setattr(
        module,
        "_load_tokenizer",
        lambda *_: pytest.fail("complete native record must stay offline"),
    )
    result = trajectory.tokenize()
    assert result.tokens == [1, 2, 3]
    assert result.logprobs[-1] == -0.3
    assert result.flags == [
        TokenFlag.EXACT,
        TokenFlag.EXACT,
        TokenFlag.EXACT | TokenFlag.SAMPLED | TokenFlag.ASSISTANT | TokenFlag.OUTPUT,
    ]
    assert trajectory.model_dump_json() == before


def _boundary_render(tokenizer: Any) -> module._ChatRender:
    def render(
        selected_messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> str:
        value = tokenizer.apply_chat_template(
            selected_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(value, str)
        return value

    return render


def test_optional_trailing_decode_valueerror_declines(monkeypatch):
    history, tokenizer, _ = _character_template_history()
    decode = tokenizer.decode
    trailing = []

    def limited(tokens, **kwargs):
        if not tokens:
            trailing.append(True)
            raise ValueError("public empty suffix unsupported")
        return decode(tokens, **kwargs)

    monkeypatch.setattr(tokenizer, "decode", limited)
    result = module._tokenize_recorded_chat_boundaries(
        history,
        [dict(message) for message in history.messages],
        tokenizer=tokenizer,
        render=_boundary_render(tokenizer),
        _trace=None,
    )
    assert trailing and result is None


@pytest.mark.parametrize(
    "stage", ["native_record", "render", "final_builder", "decoder_runtime"]
)
def test_other_errors_propagate_same_exception(monkeypatch, stage):
    history, tokenizer, _ = _character_template_history()
    error = (
        RuntimeError("public decoder failure")
        if stage == "decoder_runtime"
        else ValueError("public required validation failed")
    )
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise error

    if stage == "native_record":
        monkeypatch.setattr(module, "_chat_source_record", fail)
    elif stage == "final_builder":
        monkeypatch.setattr(module, "_tokenize_exact_projected_chat_history", fail)
    elif stage == "decoder_runtime":
        monkeypatch.setattr(tokenizer, "decode", fail)
    render = fail if stage == "render" else _boundary_render(tokenizer)
    with pytest.raises(type(error)) as caught:
        module._tokenize_recorded_chat_boundaries(
            history,
            [dict(message) for message in history.messages],
            tokenizer=tokenizer,
            render=render,
            _trace=None,
        )
    assert calls == [True] and caught.value is error


def test_malformed_native_record_is_still_rejected(monkeypatch):
    history, tokenizer, _ = _character_template_history()
    source = history.message_sources[3]
    assert source is not None and isinstance(
        source.exchange, module.ChatCompletionsExchange
    )
    extra = source.exchange.response.choices[0].model_extra
    assert extra is not None
    extra["token_ids"] = ["not-an-exact-id"]
    called = []

    def render(*args, **kwargs):
        called.append(True)
        raise AssertionError("should not reach rendering")

    with pytest.raises(ValueError, match="token_ids"):
        module._tokenize_recorded_chat_boundaries(
            history,
            [dict(message) for message in history.messages],
            tokenizer=tokenizer,
            render=render,
            _trace=None,
        )
    assert not called
