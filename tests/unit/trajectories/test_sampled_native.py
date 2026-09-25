from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import math
import pickle
import struct
from typing import Any, cast

from openai.types.chat import ChatCompletion
from openai.types.responses import Response
import pytest

import art
from art import trajectories as tr
from art.preprocessing.dynamo_tokens import COMPLETION_LOGPROBS_KEY
from art.trajectories import _parallel, _sampled_native, _tokenize
from art.trajectories._serialization import _equal_with_nan


class StopOnly:
    eos_token_id = 9
    all_special_tokens: list[str] = []
    special_tokens_map: dict[str, str] = {}

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Native sampled construction must not render")

    def __call__(self, text: str, **kwargs: Any) -> list[int]:
        raise AssertionError("This numeric STOP authority must not encode content")


def recorded(
    *,
    terminal_tool: bool = False,
    nested: bool = True,
    repeated: bool = False,
    model: str = "policy",
    lp: float = -0.2,
) -> tr.Trajectory:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "public question"}]
    prompts = [[1], [1, 20, 21, 99, 2], [1, 20, 21, 99, 2, 30, 31, 9, 3]]
    if not nested:
        prompts[-1][6] = 32
    outputs = [[20, 21], [30, 31], [40, 9]]
    replies = [
        {"role": "assistant", "content": "public truncated answer"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-public",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"public"}'},
                }
            ],
        },
        {"role": "assistant", "content": "public final answer"},
    ]
    exchanges = []
    for i in range(2 if terminal_tool else 3):
        response = ChatCompletion.model_validate(
            {
                "id": f"public-{i}",
                "object": "chat.completion",
                "created": i,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if i == 0 else "stop",
                        "message": replies[i],
                        "prompt_token_ids": prompts[i],
                        "token_ids": outputs[i],
                        "logprobs": {
                            "content": [
                                {
                                    "token": f"token_id:{token}",
                                    "logprob": lp,
                                    "bytes": [],
                                    "top_logprobs": [],
                                }
                                for token in outputs[i]
                            ]
                        },
                    }
                ],
            }
        )
        now = datetime(2026, 1, 1) + timedelta(seconds=i)
        exchanges.append(
            tr.ChatCompletionsExchange(
                request=tr.ChatCompletionsRequest(
                    model=model,
                    messages=cast(Any, deepcopy(messages)),
                    chat_template_kwargs={
                        "enable_thinking": False,
                        "preserve_thinking": not (repeated and i == 1),
                    },
                ),
                response=response,
                start_time=now,
                end_time=now,
            )
        )
        messages.append(deepcopy(replies[i]))
        messages.append({"role": "user", "content": f"public follow-up {i}"})
    return tr.Trajectory(
        reward=0.75,
        metrics={"retained": 2},
        metadata={"public": True},
        exchanges=tr.TrajectoryExchanges(chat_completions=exchanges),
    )


@pytest.fixture
def authority(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def config(model: str, base: str | None) -> _tokenize._TokenizerConfig:
        assert base is None
        calls.append(model)
        return _tokenize._TokenizerConfig(model, "public-revision:" + model)

    monkeypatch.setattr(_tokenize, "_tokenizer_config", config)
    monkeypatch.setattr(_tokenize, "_load_tokenizer", lambda config: StopOnly())
    monkeypatch.setattr(_parallel, "_cpu_capacity", lambda: 2)
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: False)
    return calls


def source_terms(t: tr.Trajectory, model: str | None = None) -> list[tuple]:
    """Independent original native inventory in canonical encounter order."""
    rows = []
    for history in t.histories(model=model):
        assert isinstance(history, tr.ChatCompletionsHistory)
        seen = set()
        for message, source in zip(
            history.messages, history.message_sources, strict=True
        ):
            if (
                message.get("role") != "assistant"
                or source is None
                or source.choice_index is None
            ):
                continue
            identity = (id(source.exchange), source.choice_index)
            if identity in seen:
                continue
            seen.add(identity)
            assert isinstance(source.exchange, tr.ChatCompletionsExchange)
            choice = next(
                choice
                for choice in source.exchange.response.choices
                if choice.index == source.choice_index
            )
            extra = choice.model_extra or {}
            prompt, output = extra["prompt_token_ids"], extra["token_ids"]
            lp = extra.get(COMPLETION_LOGPROBS_KEY)
            if lp is None:
                assert (
                    choice.logprobs is not None and choice.logprobs.content is not None
                )
                lp = [entry.logprob for entry in choice.logprobs.content]
            for i, (token, prob) in enumerate(zip(output, lp, strict=True)):
                rows.append(
                    (
                        history.model,
                        tuple([*prompt, *output[:i]]),
                        token,
                        prob,
                        choice.finish_reason != "length"
                        and i == len(output) - 1
                        and token == 9,
                    )
                )
    return rows


def result_terms(value: tr.TokenizedMultiHistoryTrajectory) -> list[tuple]:
    return [
        (h.model, tuple(h.tokens[:i]), token, lp, bool(flag & tr.TokenFlag.STOP))
        for h in value.histories
        for i, (token, lp, flag) in enumerate(
            zip(h.tokens, h.logprobs, h.flags, strict=True)
        )
        if flag & tr.TokenFlag.SAMPLED
    ]


def claims(rows: list[tuple]) -> list[tuple]:
    seen = set()
    result = []
    for model, prompt, token, lp, stop in rows:
        key = (model, prompt, token)
        first = key not in seen
        seen.add(key)
        try:
            finite = math.isfinite(struct.unpack("!f", struct.pack("!f", lp))[0])
        except OverflowError:
            finite = False
        result.append((key, first, first and finite, stop))
    return result


@pytest.mark.parametrize("terminal_tool", [False, True])
@pytest.mark.parametrize("nested", [False, True])
async def test_tool_synthetic_stop_uses_native_authority_without_rendering(
    authority: list,
    terminal_tool: bool,
    nested: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = recorded(terminal_tool=terminal_tool, nested=nested)
    before = t.model_dump()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "The explicit native route must not call ordinary tokenization"
        )

    monkeypatch.setattr(tr.Trajectory, "tokenize", fail)
    value = (await art.tokenize_sampled([t], representation="native"))[0]
    assert value.trajectory is t
    assert _equal_with_nan(before, t.model_dump())
    assert result_terms(value) == source_terms(t)
    assert len(value.histories) == (2 if not terminal_tool and not nested else 1)
    assert authority == ["policy"]
    # The tool response has no sampled terminator: no synthetic sampled STOP
    # or renderer-owned OUTPUT tail may be invented for it.
    tool_rows = [row for row in result_terms(value) if row[2] in {30, 31}]
    assert len(tool_rows) == 2 and all(not row[-1] for row in tool_rows)
    required = (
        tr.TokenFlag.EXACT
        | tr.TokenFlag.SAMPLED
        | tr.TokenFlag.ASSISTANT
        | tr.TokenFlag.OUTPUT
    )
    assert all(
        flag == tr.TokenFlag.EXACT or flag & required == required
        for h in value.histories
        for flag in h.flags
    )


@pytest.mark.parametrize("lp", [-0.2, math.nan, -math.inf, 1e100])
async def test_repeated_sources_keep_first_ownership_before_float32_filter(
    authority: list,
    lp: float,
) -> None:
    t = recorded(repeated=True, lp=lp)
    original = source_terms(t)
    assert len(original) > 6
    value = (await art.tokenize_sampled([t], representation="native"))[0]
    assert _equal_with_nan(result_terms(value), original)
    assert claims(result_terms(value)) == claims(original)
    masks = tr.first_occurrence_masks(value.histories, where=tr.TokenFlag.SAMPLED)
    actual = [
        take
        for h, mask in zip(value.histories, masks, strict=True)
        for flag, take in zip(h.flags, mask, strict=True)
        if flag & tr.TokenFlag.SAMPLED
    ]
    assert actual == [row[1] for row in claims(original)]


async def test_two_models_with_same_ids_and_evidence_do_not_collide(
    authority: list,
) -> None:
    a, b = recorded(model="a"), recorded(model="b")
    a.exchanges.chat_completions.extend(b.exchanges.chat_completions)
    value = (await art.tokenize_sampled([a], model="*", representation="native"))[0]
    assert result_terms(value) == source_terms(a, "*")
    assert set(authority) == {"a", "b"}


async def test_multiple_nonpositional_choice_indices_keep_complete_inventory(
    authority: list,
) -> None:
    exchange = recorded().exchanges.chat_completions[0]
    data = exchange.response.model_dump()
    first = data["choices"][0]
    first["index"] = 7
    second = deepcopy(first)
    second["index"] = 3
    second["token_ids"] = [22, 23]
    for entry, token in zip(second["logprobs"]["content"], [22, 23], strict=True):
        entry["token"] = f"token_id:{token}"
    data["choices"] = [first, second]
    exchange.response = ChatCompletion.model_validate(data)
    t = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))
    value = (await art.tokenize_sampled([t], representation="native"))[0]
    assert len(value.histories) == 2 and result_terms(value) == source_terms(t)
    indices = set()
    for h in value.histories:
        assert isinstance(h.history, tr.ChatCompletionsHistory)
        indices.update(
            source.choice_index
            for source in h.history.message_sources
            if source is not None and source.choice_index is not None
        )
    assert indices == {3, 7}


@pytest.mark.parametrize("first_lp", [math.nan, 1e100])
async def test_nonfinite_first_source_owns_edge_before_later_finite_source(
    authority: list,
    first_lp: float,
) -> None:
    a = recorded(lp=first_lp).exchanges.chat_completions[0]
    b = recorded(lp=-0.2).exchanges.chat_completions[0]
    b.response.id = "distinct-finite-source"
    b.start_time += timedelta(seconds=1)
    b.end_time += timedelta(seconds=1)
    t = tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[a, b]))
    value = (await art.tokenize_sampled([t], representation="native"))[0]
    assert len(value.histories) == 2
    assert tr.first_occurrence_masks(value.histories, where=tr.TokenFlag.SAMPLED) == [
        [False, True, True],
        [False, False, False],
    ]
    assert value.histories[1].logprobs[1:] == [-0.2, -0.2]
    assert not any(row[2] for row in claims(result_terms(value)))


async def test_mixed_protocols_refuse_before_loading_stop_authority(
    authority: list,
) -> None:
    t = recorded()
    now = datetime(2026, 1, 1)
    response = Response.model_validate(
        {
            "id": "public-response",
            "object": "response",
            "created_at": 0,
            "model": "policy",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )
    t.exchanges.responses.append(
        tr.ResponsesExchange(
            request={"model": "policy", "input": "public"},
            response=response,
            start_time=now,
            end_time=now,
        )
    )
    with pytest.raises(ValueError, match="unmixed Chat"):
        await art.tokenize_sampled([t], representation="native")
    assert authority == []


async def test_packed_completion_logprobs_and_compact_roundtrip(
    authority: list,
) -> None:
    t = recorded()
    for exchange in t.exchanges.chat_completions:
        choice = exchange.response.choices[0]
        assert choice.model_extra is not None
        choice.model_extra[COMPLETION_LOGPROBS_KEY] = [-0.2] * len(
            choice.model_extra["token_ids"]
        )
        choice.logprobs = None
    packed = tr.compact_dump(t)
    restored = tr.compact_validate(packed, type=tr.Trajectory)
    value = (await art.tokenize_sampled([restored], representation="native"))[0]
    assert result_terms(value) == source_terms(restored)
    roundtrip = tr.compact_validate(
        tr.compact_dump(value), type=tr.TokenizedMultiHistoryTrajectory
    )
    assert _equal_with_nan(value.model_dump(), roundtrip.model_dump())


@pytest.mark.parametrize("thinking", [False, True, None])
@pytest.mark.parametrize("reasoning_key", ["reasoning", "reasoning_content"])
async def test_native_mode_does_not_infer_thinking_or_rewrite_literal_content(
    authority: list,
    thinking: bool | None,
    reasoning_key: str,
) -> None:
    t = recorded()
    message = {
        "role": "assistant",
        "content": "public </think> literal <think> nested </think>",
        reasoning_key: "recorded structured reasoning",
    }
    first = t.exchanges.chat_completions[0]
    data = first.response.model_dump()
    data["choices"][0]["message"] = message
    first.response = ChatCompletion.model_validate(data)
    for i, exchange in enumerate(t.exchanges.chat_completions):
        kwargs = exchange.request["chat_template_kwargs"]
        if thinking is None:
            kwargs.pop("enable_thinking")
        else:
            kwargs["enable_thinking"] = thinking
        if i:
            exchange.request["messages"][1] = cast(Any, deepcopy(message))
    before = t.model_dump()
    value = (await art.tokenize_sampled([t], representation="native"))[0]
    assert result_terms(value) == source_terms(t)
    assert _equal_with_nan(before, t.model_dump())


async def test_complete_logprob_ids_remain_valid_without_separate_output_ids(
    authority: list,
) -> None:
    t = recorded()
    for exchange in t.exchanges.chat_completions:
        extra = exchange.response.choices[0].model_extra
        assert extra is not None
        extra.pop("token_ids")
    value = (await art.tokenize_sampled([t], representation="native"))[0]
    assert [row[2] for row in result_terms(value)] == [20, 21, 30, 31, 40, 9]


@pytest.mark.parametrize(
    "bad",
    [
        "prompt",
        "output",
        "lp",
        "lp_length",
        "packed_absent",
        "packed_length",
        "duplicate",
        "additional",
        "legacy",
        "edited",
    ],
)
async def test_incomplete_or_edited_authority_refuses(
    authority: list,
    monkeypatch: pytest.MonkeyPatch,
    bad: str,
) -> None:
    t = recorded()
    choice = t.exchanges.chat_completions[0].response.choices[0]
    assert choice.model_extra is not None
    if bad == "prompt":
        choice.model_extra.pop("prompt_token_ids")
    elif bad == "output":
        choice.model_extra["token_ids"] = [999]
    elif bad == "lp":
        choice.logprobs = None
    elif bad == "lp_length":
        assert choice.logprobs is not None and choice.logprobs.content is not None
        choice.logprobs.content.pop()
    elif bad == "packed_absent":
        choice.model_extra[COMPLETION_LOGPROBS_KEY] = None
    elif bad == "packed_length":
        choice.model_extra[COMPLETION_LOGPROBS_KEY] = [-0.2]
    elif bad == "duplicate":
        t.exchanges.chat_completions.append(deepcopy(t.exchanges.chat_completions[0]))
    elif bad == "additional":
        t = tr.Trajectory(
            additional_histories=[
                tr.LegacyHistory(
                    model="policy",
                    messages_and_choices=[{"role": "user", "content": "public"}],
                )
            ]
        )
    elif bad == "legacy":
        t = tr.Trajectory(messages_and_choices=[{"role": "user", "content": "public"}])
    elif bad == "edited":
        histories = t.histories()
        assert isinstance(histories[0], tr.ChatCompletionsHistory)
        histories[0].messages[-1]["content"] = "edited public source"
        monkeypatch.setattr(
            tr.Trajectory, "histories", lambda self, **kwargs: histories
        )
    before = t.model_dump()
    with pytest.raises((ValueError, TypeError)):
        await art.tokenize_sampled([t], representation="native")
    assert _equal_with_nan(before, t.model_dump())


@pytest.mark.parametrize("bad", ["tokens", "lp", "stop", "ownership"])
async def test_constructed_output_is_independently_checked(
    authority: list,
    monkeypatch: pytest.MonkeyPatch,
    bad: str,
) -> None:
    original = _tokenize._tokenize_exact_projected_chat_history

    def corrupt(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        assert value is not None
        if bad == "tokens":
            value.tokens[0] += 1
        elif bad == "lp":
            value.logprobs[-1] -= 1
        elif bad == "stop":
            value.flags[-1] ^= tr.TokenFlag.STOP
        else:
            kwargs["_trace"].trace.source_keys[-1] = None
        return value

    monkeypatch.setattr(_tokenize, "_tokenize_exact_projected_chat_history", corrupt)
    with pytest.raises((ValueError, AssertionError)):
        await art.tokenize_sampled([recorded()], representation="native")


async def test_group_process_dispatch_rebinds_original_objects(
    authority: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda _: True)
    options = []

    async def process_map(payloads: list[bytes], trajectories: list, **_: Any) -> list:
        options.extend(pickle.loads(payload)[1] for payload in payloads)
        return [
            _parallel._deserialize_process_result(
                _parallel._tokenize_process_payload(payload), source
            )
            for payload, source in zip(payloads, trajectories, strict=True)
        ]

    monkeypatch.setattr(_parallel, "_ordered_process_map", process_map)
    t = recorded(repeated=True)
    group = tr.TrajectoryGroup([t], metadata={"order": 7}, metrics={"retained": 2})
    result = (await art.tokenize_sampled([group], representation="native"))[0]
    assert result.metadata == group.metadata and result.metrics == group.metrics
    value = result.trajectories[0]
    assert value.trajectory is t and result_terms(value) == source_terms(t)
    assert all(option.sampled and option.native_sampled for option in options)
    exchanges = {id(e) for e in t.exchanges.chat_completions}
    for h in value.histories:
        assert isinstance(h.history, tr.ChatCompletionsHistory)
        assert all(
            id(s.exchange) in exchanges
            for s in h.history.message_sources
            if s is not None
        )


@pytest.mark.parametrize(
    "change",
    [
        {"sampled": False},
        {"multi_history": False},
        {"reconcile_text_equivalent_tokenizations": True},
        {"chat_template": "override"},
        {"chat_template_kwargs": {}},
    ],
)
def test_native_process_options_reject_overrides(authority: list, change: dict) -> None:
    options = _parallel._ProcessOptions(
        True, False, None, None, None, None, sampled=True, native_sampled=True
    )
    with pytest.raises(ValueError, match="unmodified sampled options"):
        _parallel._tokenize_process_payload(
            pickle.dumps((recorded(), replace(options, **change)))
        )
    assert authority == []


async def test_native_model_selection_and_empty_containers(authority: list) -> None:
    assert await art.tokenize_sampled([], representation="native") == []
    t = recorded(model="a")
    t.exchanges.chat_completions.extend(recorded(model="b").exchanges.chat_completions)
    value = (
        await art.tokenize_sampled(
            [t], model="b", base_model="b", representation="native"
        )
    )[0]
    assert result_terms(value) == source_terms(t, "b") and authority == ["b"]
    with pytest.raises(ValueError, match="requested base"):
        await art.tokenize_sampled(
            [t], model="b", base_model="a", representation="native"
        )
    with pytest.raises(ValueError, match="Unknown sampled representation"):
        await art.tokenize_sampled([], representation=cast(Any, "unknown"))


@pytest.mark.parametrize("terminal_tool", [False, True])
@pytest.mark.parametrize("extension", [False, True])
async def test_native_request_tools_use_canonical_normalization(
    authority: list, terminal_tool: bool, extension: bool
) -> None:
    source = recorded(terminal_tool=terminal_tool)
    tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Public lookup",
            "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
        },
    }
    if extension:
        tool["x_vendor"] = {"public": True}
        tool["function"]["x_vendor"] = "public"
    for exchange in source.exchanges.chat_completions:
        exchange.request["tools"] = cast(Any, deepcopy([tool]))
    before = source.model_dump()
    canonical = source.histories()
    value = (await art.tokenize_sampled([source], representation="native"))[0]
    assert len(value.histories) == 1
    assert isinstance(value.histories[0].history, tr.ChatCompletionsHistory)
    assert isinstance(canonical[-1], tr.ChatCompletionsHistory)
    assert value.histories[0].history.tools == canonical[-1].tools
    assert result_terms(value) == source_terms(source)
    assert claims(result_terms(value)) == claims(source_terms(source))
    assert _equal_with_nan(before, source.model_dump())
    roundtrip = tr.compact_validate(
        value.compact_dump(), type=tr.TokenizedMultiHistoryTrajectory
    )
    assert _equal_with_nan(roundtrip.model_dump(), value.model_dump())


@pytest.mark.parametrize("terminal_tool", [False, True])
async def test_native_string_stop_marks_complete_sampled_suffix(
    authority: list, monkeypatch: pytest.MonkeyPatch, terminal_tool: bool
) -> None:
    class TextStop(StopOnly):
        def __call__(self, text: str, **kwargs: Any) -> list[int]:
            assert text == "END"
            assert kwargs == {"add_special_tokens": False}
            return [30, 31]

    monkeypatch.setattr(_tokenize, "_load_tokenizer", lambda config: TextStop())
    source = recorded(terminal_tool=terminal_tool)
    choice = source.exchanges.chat_completions[1].response.choices[0]
    assert choice.model_extra is not None
    choice.model_extra["stop_reason"] = "END"
    before = source.model_dump()
    value = (await art.tokenize_sampled([source], representation="native"))[0]
    assert len(value.histories) == 1
    actual = result_terms(value)
    assert [r[:-1] for r in actual] == [r[:-1] for r in source_terms(source)]
    assert [r[2] for r in actual if r[-1]] == (
        [30, 31] if terminal_tool else [30, 31, 9]
    )
    assert _equal_with_nan(source.model_dump(), before)


def test_join_keeps_singletons_when_message_view_differs() -> None:
    source = recorded()
    history = source.histories()[0]
    assert isinstance(history, tr.ChatCompletionsHistory)
    sources = [
        s
        for s in history.message_sources
        if s is not None and s.choice_index is not None
    ]
    spans = [_sampled_native._singleton(s, StopOnly()) for s in sources]
    messages = deepcopy(history.messages)
    messages[0]["content"] = "Different public context view"
    other_view = history.model_copy(update={"messages": messages})
    before = [span.value.model_dump() for span in spans]
    assert _sampled_native._join(other_view, spans, StopOnly()) is None
    assert _equal_with_nan([span.value.model_dump() for span in spans], before)
    # Direct defensive join control: retaining these complete singletons keeps
    # original sampled conditioning and encounter order without editing context.
    singles = tr.TokenizedMultiHistoryTrajectory(
        trajectory=source, histories=[s.value for s in spans]
    )
    assert result_terms(singles) == source_terms(source)


@pytest.mark.parametrize("unexpected", [False, True])
async def test_native_stop_loader_reports_authority_without_base_fallback(
    authority: list, monkeypatch: pytest.MonkeyPatch, unexpected: bool
) -> None:
    error = (
        RuntimeError("public unrelated failure")
        if unexpected
        else ValueError("pass base_model explicitly")
    )

    def fail(config: Any) -> Any:
        raise error

    monkeypatch.setattr(_tokenize, "_load_tokenizer", fail)
    with pytest.raises(type(error)) as caught:
        await art.tokenize_sampled([recorded()], representation="native")
    if unexpected:
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error
        assert "loadable tokenizer model ID" in str(caught.value)
        assert "base_model" in str(caught.value)
    assert authority == ["policy"]


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
async def test_reasoning_stripped_followup_refuses_complete_source_certification(
    authority: list, reasoning_field: str
) -> None:
    source = recorded()
    message = source.exchanges.chat_completions[0].response.choices[0].message
    assert message.model_extra is not None
    message.model_extra[reasoning_field] = "Public structured reasoning"
    before = source.model_dump()
    with pytest.raises(ValueError, match="projection is not complete and unchanged"):
        await art.tokenize_sampled([source], representation="native")
    assert authority == []
    assert _equal_with_nan(before, source.model_dump())
