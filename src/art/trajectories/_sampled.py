"""Opt-in certification of ordinary sampled output and model-bound STOP flags."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..preprocessing.dynamo_tokens import (
    COMPLETION_LOGPROBS_KEY,
    choice_completion_logprobs,
)
from . import (
    ChatCompletionsExchange,
    ChatCompletionsHistory,
    TokenFlag,
    TokenizedHistory,
    TokenizedMultiHistoryTrajectory,
    TokenizedTrajectory,
)
from . import _tokenize as original
from ._tokenize import (
    _chat_source_full_tokens,
    _chat_source_prompt_tokens,
    _HistoryTokenizationTrace,
    _sampled_source_key,
    _sampled_stop_suffix,
    _SampledSourceKey,
    _source_covers_complete_sampled_message,
    _source_is_sampled,
    _source_output_tokens,
    _source_stop_evidence,
)

if TYPE_CHECKING:
    from . import Tokenizer


def _validate_sampled_trace(
    trace: _HistoryTokenizationTrace, tokenized: TokenizedHistory
) -> None:
    try:
        trace.validate(tokenized)
    except AssertionError as error:
        raise ValueError(
            "Sampled output lacks complete conditioned source proof"
        ) from error


def _load_sampled_stop_tokenizer(model: str, *, base_model: str | None) -> Tokenizer:
    config = original._tokenizer_config(model, None)
    if base_model is not None and config.base_model != base_model:
        raise ValueError("Sampled STOP authority differs from the requested base model")
    try:
        bound = original._load_tokenizer(config)
    except ValueError as error:
        raise ValueError(
            "Sampled STOP certification requires a loadable tokenizer model ID or "
            "an artifact with recorded tokenizer configuration; an unconfigured "
            "served alias cannot obtain STOP authority from base_model"
        ) from error
    if bound is None:
        raise ValueError("Sampled STOP authority is unavailable")
    return bound


def _require_sampled_source_evidence(
    source: object, key: _SampledSourceKey, output: list[int] | None
) -> None:
    choice = original._chat_choice(source)
    recorded = (
        choice_completion_logprobs(choice)
        if COMPLETION_LOGPROBS_KEY in (choice.model_extra or {})
        else original._logprob_values(original._chat_logprob_entries(choice))
    )
    if not output or recorded is None or len(recorded) != len(output):
        raise ValueError("Sampled output requires complete recorded logprobs")
    if _source_stop_evidence(source, key)[0] not in {"stop", "length"}:
        raise ValueError("Sampled output requires supported STOP evidence")


def _require_exact_chat_source_edges(
    history: ChatCompletionsHistory,
    tokenized: TokenizedHistory,
    trace: _HistoryTokenizationTrace | None,
    tokenizer: Tokenizer,
) -> None:
    def refuse() -> None:
        raise ValueError("Sampled output lacks complete conditioned source proof")

    if (
        trace is None
        or tokenized.history is not history
        or len(tokenized.tokens) != len(tokenized.logprobs)
        or len(tokenized.tokens) != len(tokenized.flags)
    ):
        refuse()
    assert trace is not None
    _validate_sampled_trace(trace, tokenized)
    expected = {
        _sampled_source_key(source): source
        for message, source in zip(
            history.messages, history.message_sources, strict=True
        )
        if message.get("role") == "assistant"
        and source is not None
        and _source_is_sampled(source)
    }
    positions: dict[_SampledSourceKey, list[int]] = {}
    for index, key in enumerate(trace.source_keys):
        if key is not None:
            positions.setdefault(key, []).append(index)
    if (
        not expected
        or expected.keys() != positions.keys()
        or expected.keys() != trace.sources.keys()
    ):
        refuse()
    required = (
        TokenFlag.EXACT | TokenFlag.SAMPLED | TokenFlag.ASSISTANT | TokenFlag.OUTPUT
    )
    for key, source in expected.items():
        indices = positions[key]
        start, end = indices[0], indices[-1] + 1
        prompt = _chat_source_prompt_tokens(source)
        output = _source_output_tokens(source, key)
        lp_ids, logprobs = _chat_source_full_tokens(source)
        _require_sampled_source_evidence(source, key, output)
        if (
            indices != list(range(start, end))
            or prompt is None
            or output is None
            or tokenized.tokens[:start] != prompt
            or tokenized.tokens[start:end] != output
            or lp_ids != output
            or len(logprobs) != end - start
            or not all(
                a == b or (math.isnan(a) and math.isnan(b))
                for a, b in zip(tokenized.logprobs[start:end], logprobs, strict=True)
            )
            or any(flag & required != required for flag in tokenized.flags[start:end])
        ):
            refuse()
        assert output is not None
        stop_count = _sampled_stop_suffix(
            output, source=source, source_key=key, tokenizer=tokenizer
        )
        if (
            any(
                bool(tokenized.flags[index] & TokenFlag.STOP)
                != (index >= end - stop_count)
                for index in indices
            )
            or (
                _source_stop_evidence(source, key)[0] == "length"
                and any(tokenized.flags[index] & TokenFlag.STOP for index in indices)
            )
            or (
                stop_count
                and end < len(tokenized.tokens)
                and tokenized.flags[end] & TokenFlag.STOP
                and not tokenized.flags[end] & TokenFlag.SAMPLED
            )
        ):
            refuse()


def reconcile_sampled_stops(
    tokenized: TokenizedMultiHistoryTrajectory | TokenizedTrajectory,
    *,
    base_model: str | None,
) -> TokenizedMultiHistoryTrajectory:
    """Certify complete native spans after rendering; never change ordinary inputs."""
    if not isinstance(tokenized, TokenizedMultiHistoryTrajectory):
        raise TypeError("Sampled tokenization requires multiple-history output")
    assembled: list[TokenizedHistory] = []
    resolved: dict[str, Tokenizer] = {}
    changed = False
    for value in tokenized.histories:
        history = value.history
        sources: dict[_SampledSourceKey, object] = {}
        if isinstance(history, ChatCompletionsHistory):
            original._validate_history_sources(history)
            for message, source in zip(
                history.messages, history.message_sources, strict=True
            ):
                if (
                    message.get("role") != "assistant"
                    or source is None
                    or not _source_is_sampled(source)
                ):
                    continue
                if not isinstance(source.exchange, ChatCompletionsExchange):
                    raise ValueError(
                        "Sampled STOP certification supports Chat Completions sources only"
                    )
                if not _source_covers_complete_sampled_message(message, source):
                    raise ValueError(
                        "Sampled output requires a complete source message"
                    )
                key = _sampled_source_key(source)
                previous = sources.setdefault(key, source)
                if (
                    getattr(previous, "exchange") is not source.exchange
                    or getattr(previous, "choice_index") != source.choice_index
                ):
                    raise ValueError("Sampled source identity conflict")
        if not sources:
            if any(flag & TokenFlag.SAMPLED for flag in value.flags):
                raise ValueError("Sampled output lacks supported source authority")
            assembled.append(value)
            continue
        assert isinstance(history, ChatCompletionsHistory)
        if (
            not history.model
            or value.model != history.model
            or not original._history_matches_projection(history)
        ):
            raise ValueError(
                "Sampled output requires an unchanged source projection and model"
            )
        keys: list[_SampledSourceKey | None] = [None] * len(value.tokens)
        previous_end = 0
        for key, source in sources.items():
            prompt = _chat_source_prompt_tokens(source)
            output, lp = _chat_source_full_tokens(source)
            if (
                not prompt
                or not output
                or len(output) != len(lp)
                or len(prompt) < previous_end
                or len(prompt) + len(output) > len(keys)
            ):
                raise ValueError(
                    "Sampled output requires complete nonoverlapping native spans"
                )
            _require_sampled_source_evidence(source, key, output)
            start, end = len(prompt), len(prompt) + len(output)
            keys[start:end] = [key] * len(output)
            previous_end = end
        trace = _HistoryTokenizationTrace(keys, sources)
        _validate_sampled_trace(trace, value)
        # STOP authority is deliberately separate from ordinary renderer selection.
        # Resolve every exact source model, never a shared caller base/revision.
        if history.model not in resolved:
            resolved[history.model] = _load_sampled_stop_tokenizer(
                history.model, base_model=base_model
            )
        bound = resolved[history.model]
        flags = list(value.flags)
        original._mark_sampled_stops(
            value.tokens, flags, keys, sources, tokenizer=bound
        )
        if flags != value.flags:
            value = value.model_copy(update={"flags": flags})
            changed = True
        _require_exact_chat_source_edges(history, value, trace, bound)
        assembled.append(value)
    return (
        tokenized.model_copy(update={"histories": assembled}) if changed else tokenized
    )
