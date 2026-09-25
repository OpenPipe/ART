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
    with pytest.raises((ValueError, AssertionError)):
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
