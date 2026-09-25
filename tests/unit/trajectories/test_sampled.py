from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import math
import pickle
import struct
from typing import Any, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
import pytest

import art
import art.trajectories as tr
from art.trajectories import _parallel, _sampled, _tokenize


class StopTokenizer:
    eos_token_id = 9
    all_special_tokens = []
    special_tokens_map = {}

    def __call__(self, text: str, **kwargs: object) -> list[int]:
        assert text == "END"
        return [8, 9]

    def apply_chat_template(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("STOP authority must never render")


def trajectory(
    *, model: str = "policy", finish: str = "stop", reason: Any = None, lp: float = -0.2
) -> tr.Trajectory:
    response = ChatCompletion.model_validate(
        {
            "id": "response",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "stop_reason": reason,
                    "message": {"role": "assistant", "content": "answer"},
                    "prompt_token_ids": [1],
                    "token_ids": [8, 9],
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{token}",
                                "logprob": lp,
                                "bytes": [],
                                "top_logprobs": [],
                            }
                            for token in [8, 9]
                        ]
                    },
                }
            ],
        }
    )
    exchange = tr.ChatCompletionsExchange(
        request=tr.ChatCompletionsRequest(
            model=model, messages=[{"role": "user", "content": "question"}]
        ),
        response=response,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1),
    )
    return tr.Trajectory(exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]))


def native_length_output() -> tr.TokenizedMultiHistoryTrajectory:
    # Certifier boundary control; ordinary length rendering remains independent.
    t = trajectory(finish="length")
    h = t.histories()[0]
    value = tr.TokenizedHistory(
        history=h,
        model="policy",
        tokens=[1, 8, 9],
        logprobs=[math.nan, -0.2, -0.2],
        flags=[
            tr.TokenFlag.EXACT,
            *(
                [
                    tr.TokenFlag.EXACT
                    | tr.TokenFlag.SAMPLED
                    | tr.TokenFlag.ASSISTANT
                    | tr.TokenFlag.OUTPUT
                ]
                * 2
            ),
        ],
    )
    return tr.TokenizedMultiHistoryTrajectory(trajectory=t, histories=[value])


@pytest.fixture
def authority(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    calls = []

    def config(model: str, base: str | None) -> _tokenize._TokenizerConfig:
        calls.append((model, base))
        return _tokenize._TokenizerConfig(model, "revision:" + model)

    def load(config: _tokenize._TokenizerConfig) -> Any:
        assert config.revision == "revision:" + config.base_model
        return StopTokenizer()

    monkeypatch.setattr(_tokenize, "_tokenizer_config", config)
    monkeypatch.setattr(_tokenize, "_load_tokenizer", load)
    monkeypatch.setattr(_parallel, "_cpu_capacity", lambda: 2)
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: False)
    return calls


async def test_public_api_preserves_render_inputs_and_copies_only_stop(
    monkeypatch: pytest.MonkeyPatch,
    authority: list,
) -> None:
    source = trajectory()
    original = tr.Trajectory.tokenize
    observations = []
    before = pickle.dumps(source)

    def ordinary(self: tr.Trajectory, **kwargs: Any) -> Any:
        assert kwargs == dict(
            multi_history=True,
            reconcile_text_equivalent_tokenizations=False,
            model=None,
            base_model=None,
            tokenizer=None,
            chat_template=None,
            chat_template_kwargs=None,
        )
        assert authority == []
        value = original(self, **kwargs)
        observations.append(value)
        return value

    monkeypatch.setattr(tr.Trajectory, "tokenize", ordinary)
    result = (await art.tokenize_sampled([source]))[0]
    old = observations[0]
    assert result is not old and result.histories[0] is not old.histories[0]
    a, b = old.histories[0], result.histories[0]
    assert b.tokens is a.tokens and b.logprobs is a.logprobs and b.history is a.history
    assert b.flags == [*a.flags[:-1], a.flags[-1] | tr.TokenFlag.STOP]
    assert not any(f & tr.TokenFlag.STOP for f in a.flags)
    assert pickle.dumps(source) == before
    assert authority == [("policy", None)]


async def test_generic_native_path_does_not_load_stop_authority(
    authority: list,
) -> None:
    result = (await art.tokenize([trajectory()], multi_history=True))[0]
    assert authority == []
    assert not any(f & tr.TokenFlag.STOP for f in result.histories[0].flags)


@pytest.mark.parametrize(
    "finish,reason,expected",
    [
        ("stop", None, 1),
        ("stop", 9, 1),
        ("stop", "END", 2),
        ("length", None, 0),
        ("tool_calls", None, 1),
    ],
)
def test_bound_stop_suffix_and_noop_identity(
    authority: list, finish: str, reason: Any, expected: int
) -> None:
    t = trajectory(finish=finish, reason=reason)
    old = (
        native_length_output() if finish == "length" else t.tokenize(multi_history=True)
    )
    result = _sampled.reconcile_sampled_stops(old, base_model=None)
    assert (
        sum(bool(f & tr.TokenFlag.STOP) for f in result.histories[0].flags) == expected
    )
    assert (result is old) == (result.histories[0].flags == old.histories[0].flags)


async def test_groups_models_and_metadata_preserved(authority: list) -> None:
    a, b = trajectory(model="a"), trajectory(model="b")
    group = tr.TrajectoryGroup([a, b], metadata={"group": "g"}, metrics={"score": 1})
    result = (await art.tokenize_sampled([group], model="*"))[0]
    assert [x.trajectory for x in result.trajectories] == [a, b]
    assert result.metadata == group.metadata and result.metrics == group.metrics
    assert set(authority) == {("a", None), ("b", None)}


async def test_model_filter_uses_selected_history_authority(authority: list) -> None:
    a, b = trajectory(model="a"), trajectory(model="b")
    a.exchanges.chat_completions.extend(b.exchanges.chat_completions)
    result = (await art.tokenize_sampled([a], model="b", base_model="b"))[0]
    assert [h.model for h in result.histories] == ["b"]
    assert authority == [("b", None)]


async def test_original_failure_propagates_without_resolution(
    monkeypatch: pytest.MonkeyPatch, authority: list
) -> None:
    error = ValueError("ordinary refusal")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(tr.Trajectory, "tokenize", fail)
    with pytest.raises(ValueError) as caught:
        await art.tokenize_sampled([trajectory()])
    assert caught.value is error and authority == []


@pytest.mark.parametrize(
    "bad",
    [
        "prefix",
        "output",
        "lp",
        "extra_stop",
        "length_stop",
        "sampled_gap",
        "missing_sampled",
        "output_flag",
        "model",
        "partial",
        "edited",
        "unsupported",
    ],
)
def test_incomplete_or_inconsistent_native_proof_refuses(
    authority: list, bad: str
) -> None:
    old = (
        native_length_output()
        if bad == "length_stop"
        else trajectory().tokenize(multi_history=True)
    )
    h = old.histories[0]
    if bad == "prefix":
        h.tokens[0] = 777
    elif bad == "output":
        h.tokens[-1] = 777
    elif bad == "lp":
        h.logprobs[-1] = -77
    elif bad in ("extra_stop", "length_stop"):
        h.flags[1] |= tr.TokenFlag.STOP
    elif bad == "sampled_gap":
        h.flags[0] |= tr.TokenFlag.SAMPLED
    elif bad == "missing_sampled":
        h.flags[1] &= ~tr.TokenFlag.SAMPLED
    elif bad == "output_flag":
        h.flags[1] &= ~tr.TokenFlag.OUTPUT
    elif bad == "model":
        h.model = "different"
    elif bad == "partial":
        h.tokens.pop()
        h.flags.pop()
        h.logprobs.pop()
    elif bad == "edited":
        assert isinstance(h.history, tr.ChatCompletionsHistory)
        h.history.messages[-1]["content"] = "edited"
    elif bad == "unsupported":
        h.history = tr.LegacyHistory(messages_and_choices=[])
    with pytest.raises(ValueError):
        _sampled.reconcile_sampled_stops(old, base_model=None)


def test_source_less_nonsampled_history_preserves_identity(authority: list) -> None:
    value = tr.TokenizedHistory(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="x",
        tokens=[1],
        logprobs=[math.nan],
        flags=[tr.TokenFlag.EXACT],
    )
    old = tr.TokenizedMultiHistoryTrajectory(
        trajectory=tr.Trajectory(), histories=[value]
    )
    assert _sampled.reconcile_sampled_stops(old, base_model=None) is old
    assert authority == []


def test_wrong_base_and_absent_authority_refuse(
    monkeypatch: pytest.MonkeyPatch, authority: list
) -> None:
    old = trajectory().tokenize(multi_history=True)
    with pytest.raises(ValueError, match="base model"):
        _sampled.reconcile_sampled_stops(old, base_model="other")
    monkeypatch.setattr(_tokenize, "_load_tokenizer", lambda _: None)
    with pytest.raises(ValueError, match="unavailable"):
        _sampled.reconcile_sampled_stops(old, base_model=None)


@pytest.mark.parametrize("lp", [math.nan, 1e100, -0.2])
def test_first_owner_before_float32_finite_is_unchanged(
    authority: list, lp: float
) -> None:
    a = trajectory(lp=lp).tokenize(multi_history=True)
    b = trajectory(lp=-0.3).tokenize(multi_history=True)
    old = [*a.histories, *b.histories]
    new = [
        *_sampled.reconcile_sampled_stops(a, base_model=None).histories,
        *_sampled.reconcile_sampled_stops(b, base_model=None).histories,
    ]

    def terms(histories: list) -> tuple[list, list]:
        masks = tr.first_occurrence_masks(histories, where=tr.TokenFlag.SAMPLED)
        selected = []
        for h, mask in zip(histories, masks):
            for i, (claim, prob) in enumerate(zip(mask, h.logprobs)):
                try:
                    finite = math.isfinite(
                        struct.unpack("!f", struct.pack("!f", prob))[0]
                    )
                except OverflowError:
                    finite = False
                if i and claim and finite:
                    selected.append((h.tokens[:i], h.tokens[i], prob))
        return masks, selected

    assert terms(old) == terms(new)
    assert terms(new)[0][1] == [False, False, False]
    assert len(terms(new)[1]) == (2 if lp == -0.2 else 0)


def test_real_process_payload_roundtrip_and_generic_default(authority: list) -> None:
    t = trajectory()
    options = _parallel._ProcessOptions(
        True, False, None, None, None, None, sampled=True
    )
    payload = pickle.dumps((t, options))
    result = pickle.loads(_parallel._tokenize_process_payload(payload))
    assert result.histories[0].tokens == [1, 8, 9]
    assert result.histories[0].flags[-1] & tr.TokenFlag.STOP
    assert pickle.dumps((t, options)) == payload
    assert (
        result.histories[0].history.message_sources[-1].exchange
        is result.trajectory.exchanges.chat_completions[0]
    )
    authority.clear()
    options = replace(options, sampled=False)
    generic = pickle.loads(
        _parallel._tokenize_process_payload(pickle.dumps((t, options)))
    )
    assert not generic.histories[0].flags[-1] & tr.TokenFlag.STOP
    assert authority == []


async def test_empty_inputs_and_ordinary_empty_failure(authority: list) -> None:
    assert await art.tokenize_sampled([]) == []
    result = await art.tokenize_sampled(
        [tr.TrajectoryGroup([], metadata={"empty": True})]
    )
    assert result[0].trajectories == [] and result[0].metadata == {"empty": True}
    with pytest.raises(ValueError, match="no trainable choices"):
        await art.tokenize_sampled([tr.Trajectory()], model="policy")
    assert authority == []


async def test_public_process_dispatch_carries_optin_and_rebinds_sources(
    monkeypatch: pytest.MonkeyPatch,
    authority: list,
) -> None:
    monkeypatch.setattr(_parallel, "_supports_processes", lambda **_: True)
    monkeypatch.setattr(_parallel, "_processes_enabled", lambda _: True)
    calls = []

    async def process_map(payloads: list[bytes], trajectories: list, **_: Any) -> list:
        calls.extend(pickle.loads(p)[1] for p in payloads)
        return [
            _parallel._deserialize_process_result(
                _parallel._tokenize_process_payload(payload), source
            )
            for payload, source in zip(payloads, trajectories, strict=True)
        ]

    monkeypatch.setattr(_parallel, "_ordered_process_map", process_map)
    sources = [trajectory(model="a"), trajectory(model="b")]
    result = await art.tokenize_sampled(sources)
    assert len(calls) == 2 and all(c.sampled and c.multi_history for c in calls)
    for source, value in zip(sources, result, strict=True):
        assert value.trajectory is source
        history = value.histories[0].history
        assert isinstance(history, tr.ChatCompletionsHistory)
        message_source = history.message_sources[-1]
        assert message_source is not None
        assert message_source.exchange is source.exchanges.chat_completions[0]
        assert value.histories[0].flags[-1] & tr.TokenFlag.STOP


@pytest.mark.parametrize(
    "carrier", ["absent", "null", "empty", "packed_null", "packed_null_with_raw"]
)
async def test_missing_recorded_logprobs_refuse_public_certification(
    monkeypatch: pytest.MonkeyPatch, authority: list, carrier: str
) -> None:
    from art.preprocessing.dynamo_tokens import COMPLETION_LOGPROBS_KEY

    source = trajectory()
    choice = source.exchanges.chat_completions[0].response.choices[0]
    if carrier == "absent":
        choice = type(choice).model_validate(choice.model_dump(exclude={"logprobs"}))
        source.exchanges.chat_completions[0].response.choices[0] = choice
    elif carrier == "empty":
        assert choice.logprobs is not None
        choice.logprobs.content = []
    elif carrier != "packed_null_with_raw":
        choice.logprobs = None
    if carrier.startswith("packed_null"):
        assert choice.model_extra is not None
        choice.model_extra[COMPLETION_LOGPROBS_KEY] = None
    before = pickle.dumps(source)
    returned = []
    original = tr.Trajectory.tokenize

    def observe(self: tr.Trajectory, **kwargs: Any) -> Any:
        value = original(self, **kwargs)
        returned.append(value)
        return value

    monkeypatch.setattr(tr.Trajectory, "tokenize", observe)
    with pytest.raises(ValueError, match="recorded logprobs"):
        await art.tokenize_sampled([source])
    assert len(returned) == 1  # Real generic exact-token path completed first.
    assert all(math.isnan(lp) for lp in returned[0].histories[0].logprobs[1:])
    assert authority == []  # Reject missing evidence before loading STOP authority.
    assert pickle.dumps(source) == before
    assert not any(flag & tr.TokenFlag.STOP for flag in returned[0].histories[0].flags)


async def test_unsupported_finish_refuses_public_certification(
    monkeypatch: pytest.MonkeyPatch, authority: list
) -> None:
    source = trajectory(finish="content_filter")
    before = pickle.dumps(source)
    returned = []
    original = tr.Trajectory.tokenize

    def observe(self: tr.Trajectory, **kwargs: Any) -> Any:
        value = original(self, **kwargs)
        returned.append(value)
        return value

    monkeypatch.setattr(tr.Trajectory, "tokenize", observe)
    with pytest.raises(ValueError, match="supported STOP evidence"):
        await art.tokenize_sampled([source])
    assert len(returned) == 1
    assert returned[0].histories[0].tokens == [1, 8, 9]
    assert returned[0].histories[0].logprobs[1:] == [-0.2, -0.2]
    assert not any(flag & tr.TokenFlag.STOP for flag in returned[0].histories[0].flags)
    assert authority == []
    assert pickle.dumps(source) == before


@pytest.mark.parametrize("carrier", ["raw", "recorded_nan", "packed"])
@pytest.mark.parametrize("finish", ["stop", "tool_calls", "function_call"])
async def test_recorded_evidence_public_controls(
    authority: list, carrier: str, finish: str
) -> None:
    from art.preprocessing.dynamo_tokens import COMPLETION_LOGPROBS_KEY

    source = trajectory(
        finish=finish, lp=math.nan if carrier == "recorded_nan" else -0.2
    )
    choice = source.exchanges.chat_completions[0].response.choices[0]
    if carrier == "packed":
        choice.logprobs = None
        assert choice.model_extra is not None
        choice.model_extra[COMPLETION_LOGPROBS_KEY] = [-0.3, -0.4]
    before = pickle.dumps(source)
    result = (await art.tokenize_sampled([source]))[0]
    value = result.histories[0]
    assert value.tokens == [1, 8, 9]
    assert [bool(f & tr.TokenFlag.STOP) for f in value.flags] == [False, False, True]
    if carrier == "recorded_nan":
        assert all(math.isnan(lp) for lp in value.logprobs[1:])
    else:
        assert value.logprobs[1:] == (
            [-0.3, -0.4] if carrier == "packed" else [-0.2, -0.2]
        )
    assert tr.first_occurrence_masks([value], where=tr.TokenFlag.SAMPLED) == [
        [False, True, True]
    ]
    assert pickle.dumps(source) == before
    assert authority == [("policy", None)]
    roundtrip = tr.compact_validate(
        result.compact_dump(), type=tr.TokenizedMultiHistoryTrajectory
    )
    from art.trajectories._serialization import _equal_with_nan

    assert _equal_with_nan(roundtrip.model_dump(), result.model_dump())


@pytest.mark.parametrize("carrier", ["raw", "recorded_nan", "packed"])
def test_recorded_length_evidence_certifier_control(
    authority: list, carrier: str
) -> None:
    from art.preprocessing.dynamo_tokens import COMPLETION_LOGPROBS_KEY

    old = native_length_output()
    choice = old.trajectory.exchanges.chat_completions[0].response.choices[0]
    if carrier == "recorded_nan":
        assert choice.logprobs is not None and choice.logprobs.content is not None
        for entry in choice.logprobs.content:
            entry.logprob = math.nan
        old.histories[0].logprobs[1:] = [math.nan, math.nan]
    elif carrier == "packed":
        choice.logprobs = None
        assert choice.model_extra is not None
        choice.model_extra[COMPLETION_LOGPROBS_KEY] = [-0.2, -0.2]
    result = _sampled.reconcile_sampled_stops(old, base_model=None)
    assert result is old
    assert not any(f & tr.TokenFlag.STOP for f in result.histories[0].flags)


@pytest.mark.parametrize("where", ["public", "final_guard"])
def test_sampled_trace_data_refusal_is_value_error(
    monkeypatch: pytest.MonkeyPatch, authority: list, where: str
) -> None:
    source = trajectory()
    value = source.tokenize(multi_history=True)
    history = value.histories[0]
    assert isinstance(history.history, tr.ChatCompletionsHistory)
    history.flags[1] &= ~tr.TokenFlag.SAMPLED
    before = list(history.flags)
    if where == "public":
        monkeypatch.setattr(tr.Trajectory, "tokenize", lambda *args, **kwargs: value)
        import asyncio

        with pytest.raises(
            ValueError, match="complete conditioned source proof"
        ) as caught:
            asyncio.run(art.tokenize_sampled([source]))
    else:
        message_source = history.history.message_sources[-1]
        key = _tokenize._sampled_source_key(message_source)
        trace = _tokenize._HistoryTokenizationTrace(
            [None, key, key], {key: message_source}
        )
        with pytest.raises(
            ValueError, match="complete conditioned source proof"
        ) as caught:
            _sampled._require_exact_chat_source_edges(
                history.history, history, trace, StopTokenizer()
            )
    assert isinstance(caught.value.__cause__, AssertionError)
    assert history.flags == before and authority == []


async def test_sampled_unresolvable_alias_has_authority_specific_guidance(
    monkeypatch: pytest.MonkeyPatch, authority: list
) -> None:
    failure = ValueError("Could not load tokenizer; pass base_model explicitly")

    def unavailable(config: Any) -> Any:
        raise failure

    monkeypatch.setattr(_tokenize, "_load_tokenizer", unavailable)
    with pytest.raises(ValueError, match="loadable tokenizer model ID") as caught:
        await art.tokenize_sampled([trajectory(model="served-alias")])
    assert caught.value.__cause__ is failure
    assert "cannot obtain STOP authority from base_model" in str(caught.value)
    with pytest.raises(ValueError, match="differs from the requested base model"):
        await art.tokenize_sampled(
            [trajectory(model="served-alias")], base_model="different-tokenizer"
        )


async def test_sampled_stop_loader_other_error_identity_is_preserved(
    monkeypatch: pytest.MonkeyPatch, authority: list
) -> None:
    failure = RuntimeError("public sentinel")

    def broken(config: Any) -> Any:
        raise failure

    monkeypatch.setattr(_tokenize, "_load_tokenizer", broken)
    with pytest.raises(RuntimeError) as caught:
        await art.tokenize_sampled([trajectory()])
    assert caught.value is failure


def two_source_trajectory(*, reason: int | None = None) -> tr.Trajectory:
    first = trajectory(reason=reason)
    second = trajectory(reason=reason)
    exchange = second.exchanges.chat_completions[0]
    exchange.response.id = "response-second"
    exchange.start_time = exchange.end_time = datetime(2026, 1, 1, 0, 0, 1)
    exchange.request["messages"] = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ],
    )
    choice = exchange.response.choices[0]
    choice.message.content = "another answer"
    assert choice.model_extra is not None
    choice.model_extra["prompt_token_ids"] = [1, 8, 9, 2]
    choice.model_extra["token_ids"] = [10, 9]
    assert choice.logprobs is not None and choice.logprobs.content is not None
    choice.logprobs.content[0].token = "token_id:10"
    first.exchanges.chat_completions.append(exchange)
    return first


@pytest.mark.parametrize("reason", [None, 9])
async def test_rendered_default_two_sampled_spans(
    monkeypatch: pytest.MonkeyPatch, authority: list, reason: int | None
) -> None:
    source = two_source_trajectory(reason=reason)
    assert len(source.histories()) == 1
    before = pickle.dumps(source)
    returned = []
    original = tr.Trajectory.tokenize

    def observe(self: tr.Trajectory, **kwargs: Any) -> Any:
        value = original(self, **kwargs)
        returned.append(value)
        return value

    monkeypatch.setattr(tr.Trajectory, "tokenize", observe)
    result = (await art.tokenize_sampled([source]))[0]
    assert len(returned) == len(result.histories) == 1
    old, new = returned[0].histories[0], result.histories[0]
    assert new.tokens == [1, 8, 9, 2, 10, 9]
    assert [bool(f & tr.TokenFlag.SAMPLED) for f in new.flags] == [
        False,
        True,
        True,
        False,
        True,
        True,
    ]
    assert [bool(f & tr.TokenFlag.STOP) for f in new.flags] == [
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert new.tokens is old.tokens and new.logprobs is old.logprobs
    assert [
        (int(a) ^ int(b)) & ~int(tr.TokenFlag.STOP)
        for a, b in zip(old.flags, new.flags, strict=True)
    ] == [0] * 6
    assert (result is returned[0]) == (reason is not None)
    assert pickle.dumps(source) == before and authority == [("policy", None)]


def test_two_source_overlapping_prompt_refuses(authority: list) -> None:
    source = two_source_trajectory()
    value = source.tokenize(multi_history=True)
    second = source.exchanges.chat_completions[1].response.choices[0]
    assert second.model_extra is not None
    second.model_extra["prompt_token_ids"] = [1, 8]
    with pytest.raises(ValueError):
        _sampled.reconcile_sampled_stops(value, base_model=None)
    assert authority == []
