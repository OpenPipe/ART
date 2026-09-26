from __future__ import annotations

from collections import Counter
import copy
from typing import Any, cast

import pytest
from test_tokenize import _chat_exchange

import art.trajectories as tr
from art.trajectories import _tokenize as module


def trajectory(*exchanges):
    return tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=list(exchanges))
    )


def logprobs(exchange):
    value = exchange.response.choices[0].logprobs
    assert value is not None and value.content is not None
    return value.content


def extras(exchange):
    value = exchange.response.choices[0].model_extra
    assert value is not None
    return value


def sources(history):
    return [s for s in history.message_sources if module._source_is_sampled(s)]


def test_exact_assembly_reuses_evidence_only_within_one_call(monkeypatch):
    first = _chat_exchange([1], [2, 3])
    second = _chat_exchange([1, 2, 3, 4], [5, 6], offset=1)
    first.response.choices[0].index = 7
    value = trajectory(first, second)
    calls = []
    fingerprint = module._fingerprint

    def observed(evidence):
        calls.append(evidence)
        return fingerprint(evidence)

    monkeypatch.setattr(module, "_fingerprint", observed)
    before = value.model_dump_json()
    result = value.tokenize()
    assert result.tokens == [1, 2, 3, 4, 5, 6]
    assert len(calls) == 2  # One callback-free decision/assembly phase per source.
    assert value.model_dump_json() == before
    lp = logprobs(first)
    lp[1].logprob = -7.5
    assert value.tokenize().logprobs[2] == -7.5
    assert result.logprobs[2] == -0.3
    assert len(calls) == 4


def test_supplied_tokenizer_stop_probe_cannot_lend_stale_evidence(monkeypatch):
    first = _chat_exchange([1], [2, 3])
    extras(first)["stop_reason"] = "public-stop"
    second = _chat_exchange([1, 2, 3, 4], [5, 9], offset=1)
    value = trajectory(first, second)
    history = value.histories()[0]
    second_source = sources(history)[1]
    original_key = module._sampled_source_key(second_source)

    class Tokenizer:
        eos_token_id = 9
        all_special_ids = []
        calls = 0

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("complete records need no rendering")

        def __call__(self, text, **kwargs):
            assert text == "public-stop"
            self.calls += 1
            logprobs(second)[0].logprob = -8.5
            return {"input_ids": [3]}

    tokenizer = Tokenizer()
    trace = module._TraceBuilder()
    result = module.tokenize_history(
        history,
        model=history.model,
        base_model=None,
        tokenizer=cast(Any, tokenizer),
        chat_template=None,
        chat_template_kwargs=None,
        _trace=trace,
    )
    assert tokenizer.calls >= 1 and trace.trace is not None
    assert result.tokens == [1, 2, 3, 4, 5, 9]
    assert result.logprobs[4] == -8.5
    assert trace.trace.source_keys[4] == module._sampled_source_key(second_source)
    assert trace.trace.source_keys[4] != original_key
    assert result.flags[2] & tr.TokenFlag.STOP
    assert result.flags[-1] & tr.TokenFlag.STOP


@pytest.mark.parametrize("override", [False, True])
def test_render_fallback_does_not_receive_decision_evidence(monkeypatch, override):
    from test_tokenize import _character_template_history

    history, tokenizer, _ = _character_template_history()
    first_source = sources(history)[0]
    exchange = first_source.exchange
    old_key = module._sampled_source_key(first_source)
    inner = trajectory(_chat_exchange([88], [99]))

    def load(config):
        # A nested tokenization and a source edit happen after the original
        # length decision, at an existing renderer-loader callback boundary.
        assert inner.tokenize().tokens == [88, 99]
        logprobs(exchange)[0].logprob = -7.5
        return tokenizer

    monkeypatch.setattr(module, "_load_tokenizer", load)
    monkeypatch.setattr(
        module,
        "_tokenizer_config",
        lambda *args: module._TokenizerConfig("public/base"),
    )
    trace = module._TraceBuilder()
    result = module.tokenize_history(
        history,
        model=history.model,
        base_model="public/base",
        tokenizer=None,
        chat_template="explicit public template" if override else None,
        chat_template_kwargs=None,
        _trace=trace,
    )
    assert trace.trace is not None
    new_key = module._sampled_source_key(first_source)
    assert new_key != old_key
    indices = [i for i, key in enumerate(trace.trace.source_keys) if key == new_key]
    assert indices and result.logprobs[indices[0]] == -7.5
    assert old_key not in trace.trace.source_keys


@pytest.mark.parametrize(
    "field", ["prompt", "output", "logprob", "content", "reason", "index"]
)
def test_signature_remains_fresh_after_source_edit(field):
    exchange = _chat_exchange([1], [2, 3])
    source = sources(trajectory(exchange).histories()[0])[0]
    before = module._source_signature(source)
    choice = exchange.response.choices[0]
    if field == "prompt":
        extras(exchange)["prompt_token_ids"][0] = 9
    elif field == "output":
        extras(exchange)["token_ids"][0] = 9
    elif field == "logprob":
        logprobs(exchange)[0].logprob = float("nan")
    elif field == "content":
        choice.message.content = "edited public view"
    elif field == "reason":
        choice.finish_reason = "length"
    else:
        choice.index = 7
        source = source.model_copy(update={"choice_index": 7})
    assert module._source_signature(source) != before


def test_source_match_caches_exchange_identity_not_response_id(monkeypatch):
    first = _chat_exchange([1], [2, 3])
    source = sources(trajectory(first).histories()[0])[0]
    same = copy.copy(source)
    different = copy.deepcopy(source)
    calls = []
    fingerprint = module._fingerprint

    def observed(evidence):
        calls.append(evidence)
        return fingerprint(evidence)

    monkeypatch.setattr(module, "_fingerprint", observed)
    assert module._sources_match([source, same], [same, source])
    assert len(calls) == 1
    assert module._sources_match([source], [different])
    assert len(calls) == 3
    different.exchange.response.choices[0].logprobs.content[0].logprob = -9
    assert not module._sources_match([source], [different])
    assert len(calls) == 5
    source.exchange.response.choices[0].message.content = "fresh mutation"
    assert not module._sources_match([source], [different])
    assert len(calls) == 7


def test_fingerprint_cache_bound_and_identity():
    cache: dict[tuple[int, str, int], tuple[module.Exchange, str]] = {}
    first = _chat_exchange([1], [2])
    expected = module._sampled_evidence_fingerprint(
        first, protocol="chat_completions", index=0
    )
    for i in range(258):
        exchange = _chat_exchange([i], [i + 1])
        module._sampled_evidence_fingerprint(
            exchange, protocol="chat_completions", index=0, _cache=cache
        )
    assert len(cache) == 256
    assert (
        module._sampled_evidence_fingerprint(
            first, protocol="chat_completions", index=0, _cache=cache
        )
        == expected
    )
    # Even a stale identity slot cannot borrow another exchange's evidence.
    cache = {(id(first), "chat_completions", 0): (exchange, "wrong")}
    assert (
        module._sampled_evidence_fingerprint(
            first, protocol="chat_completions", index=0, _cache=cache
        )
        == expected
    )


def test_whitespace_decoder_clears_evidence_and_record_caches():
    first = _chat_exchange([1], [2, 3])
    first.response.choices[0].finish_reason = "length"
    second = _chat_exchange([1, 2, 3, 32, 9, 8], [4, 5], offset=1)
    third = _chat_exchange([1, 2, 3, 32, 9, 8, 4, 5, 7], [6], offset=2)
    history = trajectory(first, second, third).histories()[0]
    first_source, second_source, _ = sources(history)
    second_key = module._sampled_source_key(second_source)
    nested = trajectory(_chat_exchange([88], [99]))

    class Decoder:
        eos_token_id = 9
        all_special_ids = []
        calls = 0

        def decode(self, ids):
            assert ids == [32]
            self.calls += 1
            assert nested.tokenize().tokens == [88, 99]
            logprobs(second)[0].logprob = -7.5
            return " "

    decoder = Decoder()
    trace = module._TraceBuilder()
    value = module._tokenize_exact_projected_chat_history(
        history,
        tokenizer=cast(Any, decoder),
        projection_validated=True,
        _trace=trace,
        length_stop_boundaries={
            module._sampled_source_key(
                first_source
            ): module._RenderedLengthStopBoundary(tail=(9,), following=(8,))
        },
    )
    assert value is not None and decoder.calls == 1
    assert trace.trace is not None
    assert value.logprobs[6] == -7.5
    assert trace.trace.source_keys[6] == module._sampled_source_key(second_source)
    assert trace.trace.source_keys[6] != second_key


def test_copied_context_stop_callback_clears_evidence_and_records():
    first = _chat_exchange([1], [2, 3])
    extras(first)["stop_reason"] = "public-stop"
    second = _chat_exchange([1, 3, 4], [5, 6], offset=1)
    third = _chat_exchange([1, 3, 4, 5, 6, 7], [8], offset=2)
    histories = trajectory(first, second, third).histories()
    assert len(histories) == 2
    original_trace = module._TraceBuilder()
    original = module._tokenize_exact_projected_chat_history(
        histories[0], tokenizer=None, projection_validated=True, _trace=original_trace
    )
    assert original is not None and original_trace.trace is not None
    second_source = next(s for s in sources(histories[1]) if s.exchange is second)
    old_key = module._sampled_source_key(second_source)

    class StopTokenizer:
        eos_token_id = 3
        all_special_ids = []
        calls = 0

        def __call__(self, text, **kwargs):
            assert text == "public-stop"
            self.calls += 1
            logprobs(second)[0].logprob = -8.5
            return {"input_ids": [3]}

    decoder = StopTokenizer()
    trace = module._TraceBuilder()
    value = module._tokenize_exact_projected_chat_history(
        histories[1],
        tokenizer=cast(Any, decoder),
        projection_validated=True,
        _trace=trace,
        _strict_sources=True,
        _prior=[(original, original_trace.trace)],
    )
    assert value is not None and decoder.calls == 1
    assert trace.trace is not None
    assert value.logprobs[3] == -8.5
    assert trace.trace.source_keys[3] == module._sampled_source_key(second_source)
    assert trace.trace.source_keys[3] != old_key
    assert not value.flags[1] & tr.TokenFlag.SAMPLED


def test_decoder_exception_identity_and_next_call_fresh():
    first = _chat_exchange([1], [2, 3])
    extras(first)["stop_reason"] = "public-stop"
    value = trajectory(first)
    failure = ValueError("public stop decoder failed")

    class Broken:
        def __call__(self, text, **kwargs):
            raise failure

    with pytest.raises(ValueError) as caught:
        value.tokenize(tokenizer=Broken())
    assert caught.value is failure
    logprobs(first)[0].logprob = -10
    assert value.tokenize().logprobs[1] == -10


@pytest.mark.parametrize("edit", ["model", "source", "prompt"])
def test_edited_history_does_not_reuse_projection_proof(edit):
    from test_tokenize import _CharacterTemplateTokenizer

    value = trajectory(_chat_exchange([1], [2, 3]))
    history = value.histories()[0]
    history.tokenize()
    if edit == "model":
        history.model = "different/model"
    elif edit == "source":
        history.message_sources[-1] = history.message_sources[-1].model_copy(
            update={"choice_index": 99}
        )
    else:
        history.messages[0]["content"] = "new question"
    with pytest.raises((ValueError, AssertionError)):
        history.tokenize(tokenizer=_CharacterTemplateTokenizer())
