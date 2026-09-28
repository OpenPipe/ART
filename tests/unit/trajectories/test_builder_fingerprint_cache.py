from __future__ import annotations

import copy
from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories._tokenize as core


def test_incremental_consumption_reuses_only_validated_serialization(monkeypatch):
    sources = [_chat_exchange([1], [2], offset=index) for index in range(3)]
    keys = [core._exchange_sampled_source_key(source) for source in sources]
    fingerprint = core._fingerprint
    encoded = []

    def observe(value):
        encoded.append(True)
        return fingerprint(value)

    monkeypatch.setattr(core, "_fingerprint", observe)
    ledger = core._TraceBuilder()
    for key, source in zip(keys, sources, strict=True):
        ledger.consume_sources({key: source})
        ledger.checked(lambda: None)
    assert len(encoded) == len(sources)
    # Adding renderer authority rebuilds the guard, not its stable encoding.
    ledger.consume_sources([], rendered_evidence=True)
    ledger.checked(lambda: None)
    assert len(encoded) == len(sources)


@pytest.mark.parametrize(
    "field", ["prompt", "output", "bytes", "logprob", "request", "stop", "model"]
)
def test_extension_does_not_bless_mutation_of_earlier_sources(field):
    first = cast(Any, _chat_exchange([1], [2]))
    second = _chat_exchange([1], [3], offset=1)
    ledger = core._TraceBuilder()
    ledger.consume_sources({core._exchange_sampled_source_key(first): first})
    ledger.checked(lambda: None)
    ledger.consume_sources({core._exchange_sampled_source_key(second): second})
    ledger.checked(lambda: None)
    choice = first.response.choices[0]
    if field == "prompt":
        choice.model_extra["prompt_token_ids"][0] = 8
    elif field == "output":
        choice.model_extra["token_ids"][0] = 8
    elif field == "bytes":
        choice.logprobs.content[0].bytes = [8]
    elif field == "logprob":
        choice.logprobs.content[0].logprob = -9.0
    elif field == "request":
        first.request["messages"] = [{"role": "user", "content": "changed"}]
    elif field == "stop":
        choice.model_extra["stop_reason"] = 999
    else:
        first.request["model"] = "changed"
    with pytest.raises(ValueError, match="changed during tokenization callback"):
        ledger.checked(lambda: None)


def test_aliases_keep_independent_observations_after_extension():
    first = cast(Any, _chat_exchange([1], [2]))
    second = copy.deepcopy(first)
    key = core._exchange_sampled_source_key(first)
    ledger = core._TraceBuilder()
    ledger.consume_sources([(key, first)])
    ledger.checked(lambda: None)
    ledger.consume_sources([(key, second)])
    ledger.checked(lambda: None)
    assert len(ledger.fingerprints.observations) == 2
    second.response.choices[0].logprobs.content[0].bytes = [8]
    with pytest.raises(ValueError, match="Sampled source changed"):
        ledger.checked(lambda: None)


def test_distinct_builders_do_not_share_observations():
    first, second = core._TraceBuilder(), core._TraceBuilder()
    assert first.fingerprints is not second.fingerprints
    assert not first.fingerprints.observations
    assert not second.fingerprints.observations
