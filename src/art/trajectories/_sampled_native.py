"""Explicit native sampled representation; never a generic rendering fallback."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

from ..preprocessing.dynamo_tokens import (
    COMPLETION_LOGPROBS_KEY,
    choice_completion_logprobs,
)
from . import (
    ChatCompletionsExchange,
    ChatCompletionsHistory,
    ChatCompletionsMessageSource,
    TokenFlag,
    TokenizedHistory,
    TokenizedMultiHistoryTrajectory,
    Tokenizer,
    Trajectory,
)
from . import _tokenize as original
from ._history import _TOOLS, normalize_chat_message
from ._sampled import _load_sampled_stop_tokenizer, _require_exact_chat_source_edges
from ._serialization import _equal_with_nan


@dataclass
class _Span:
    source: ChatCompletionsMessageSource
    key: original._SampledSourceKey
    value: TokenizedHistory
    start: int


def _singleton(source: ChatCompletionsMessageSource, bound: Tokenizer) -> _Span:
    exchange = source.exchange
    assert isinstance(exchange, ChatCompletionsExchange)
    prompt = original._chat_source_prompt_tokens(source)
    output, logprobs = original._chat_source_full_tokens(source)
    choice = original._chat_choice(source)
    recorded_logprobs = (
        choice_completion_logprobs(choice)
        if COMPLETION_LOGPROBS_KEY in (choice.model_extra or {})
        else original._logprob_values(original._chat_logprob_entries(choice))
    )
    if (
        not prompt
        or not output
        or len(output) != len(logprobs)
        or recorded_logprobs is None
        or len(recorded_logprobs) != len(output)
    ):
        raise ValueError(
            "Native sampled sources require complete prompt, output and logprobs"
        )
    key = original._sampled_source_key(source)
    if original._source_stop_evidence(source, key)[0] not in {"stop", "length"}:
        raise ValueError("Native sampled sources require supported STOP evidence")
    request = exchange.request.get("messages")
    message = original._chat_choice_message(source)
    if not isinstance(request, list) or message is None:
        raise ValueError("Native sampled source message is unavailable")
    history = ChatCompletionsHistory(
        model=exchange.model,
        messages=[
            *(normalize_chat_message(m) for m in request),
            normalize_chat_message(message),
        ],
        message_sources=[
            *(
                ChatCompletionsMessageSource(exchange=exchange, request_index=i)
                for i in range(len(request))
            ),
            source,
        ],
        tools=deepcopy(_TOOLS.validate_python(exchange.request.get("tools"))),
        chat_template=exchange.request.get("chat_template"),
        chat_template_kwargs=deepcopy(exchange.request.get("chat_template_kwargs")),
    )
    original._validate_history_sources(history)
    trace = original._TraceBuilder()
    value = original._tokenize_exact_projected_chat_history(
        history, tokenizer=bound, _trace=trace
    )
    if value is None or value.tokens != [*prompt, *output]:
        raise ValueError("Native sampled source cannot be represented exactly")
    _require_exact_chat_source_edges(history, value, trace.trace, bound)
    return _Span(source, key, value, len(prompt))


def _join(
    history: ChatCompletionsHistory, run: list[_Span], bound: Tokenizer
) -> TokenizedHistory | None:
    if len(run) == 1:
        return run[0].value
    last = run[-1].value.history
    assert isinstance(last, ChatCompletionsHistory)
    # Token nesting does not authorize reassigning a captured message's source.
    # Use the final request's real view only when it also matches the canonical
    # message prefix in which these source encounters occurred.
    if not _equal_with_nan(last.messages, history.messages[: len(last.messages)]):
        return None
    selected = {span.key for span in run}
    message_sources = list(last.message_sources)
    for i, source in enumerate(history.message_sources[: len(last.messages)]):
        if (
            source is not None
            and original._source_is_sampled(source)
            and original._sampled_source_key(source) in selected
        ):
            message_sources[i] = source
    scoped = last.model_copy(update={"message_sources": message_sources})
    original._validate_history_sources(scoped)
    tokens = list(run[-1].value.tokens)
    flags = [TokenFlag.EXACT] * len(tokens)
    logprobs = [math.nan] * len(tokens)
    keys: list[original._SampledSourceKey | None] = [None] * len(tokens)
    sources: dict[original._SampledSourceKey, object] = {}
    previous_end = 0
    for span in run:
        start, end = span.start, len(span.value.tokens)
        if start < previous_end or tokens[:end] != span.value.tokens:
            raise ValueError("Native sampled chain lost complete conditioning")
        flags[start:end] = span.value.flags[start:end]
        logprobs[start:end] = span.value.logprobs[start:end]
        keys[start:end] = [span.key] * (end - start)
        sources[span.key] = span.source
        previous_end = end
    value = TokenizedHistory(
        history=scoped,
        model=run[-1].value.model,
        tokens=tokens,
        logprobs=logprobs,
        flags=flags,
    )
    trace = original._HistoryTokenizationTrace(keys, sources)
    _require_exact_chat_source_edges(scoped, value, trace, bound)
    return value


def tokenize_native(
    trajectory: Trajectory, *, model: str | None, base_model: str | None
) -> TokenizedMultiHistoryTrajectory:
    """Construct complete native Chat sources in canonical encounter order.

    Only SAMPLED first-occurrence-before-finite-filter loss is represented.
    Nonsampled request gaps are EXACT context, without invented assistant roles,
    OUTPUT flags or rendered STOP tails. No renderer is invoked and no ordinary
    exception is caught. Complete source validation precedes returning any value.
    """
    exchanges = trajectory.exchanges
    if (
        not exchanges.chat_completions
        or exchanges.completions
        or exchanges.responses
        or exchanges.messages
        or trajectory.messages_and_choices
        or trajectory.additional_histories
        or trajectory.tools is not None
    ):
        raise ValueError(
            "Native sampled representation requires unmixed Chat Completions sources"
        )
    histories: list[ChatCompletionsHistory] = []
    for history in trajectory.histories(model=model):
        if not isinstance(history, ChatCompletionsHistory) or not history.model:
            raise ValueError(
                "Native sampled representation requires selected Chat histories"
            )
        histories.append(history)
    if not histories:
        raise ValueError(
            "Native sampled representation requires selected Chat histories"
        )
    selected_models = {h.model for h in histories}
    expected: dict[
        tuple[str, original._SampledSourceKey], ChatCompletionsMessageSource
    ] = {}
    identities: set[tuple[str, str, int]] = set()
    for exchange in exchanges.chat_completions:
        if exchange.model not in selected_models:
            continue
        if not exchange.model or not exchange.response.id:
            raise ValueError("Native sampled source identity is unavailable")
        for choice in exchange.response.choices:
            identity = (exchange.model, exchange.response.id, choice.index)
            if identity in identities:
                raise ValueError("Native sampled source identity is duplicated")
            identities.add(identity)
            source = ChatCompletionsMessageSource(
                exchange=exchange, choice_index=choice.index
            )
            expected[(exchange.model, original._sampled_source_key(source))] = source
    encountered: list[
        tuple[ChatCompletionsHistory, list[ChatCompletionsMessageSource]]
    ] = []
    seen: set[tuple[str, original._SampledSourceKey]] = set()
    for history in histories:
        assert history.model is not None
        original._validate_history_sources(history)
        sources: dict[original._SampledSourceKey, ChatCompletionsMessageSource] = {}
        for message, source in zip(
            history.messages, history.message_sources, strict=True
        ):
            if (
                message.get("role") != "assistant"
                or source is None
                or not original._source_is_sampled(source)
            ):
                continue
            key = original._sampled_source_key(source)
            authority = expected.get((history.model, key))
            if (
                not isinstance(source.exchange, ChatCompletionsExchange)
                or authority is None
                or source.exchange is not authority.exchange
                or history.model != source.exchange.model
                or not original._source_covers_complete_sampled_message(message, source)
            ):
                raise ValueError(
                    "Native sampled source projection is not complete and unchanged"
                )
            sources.setdefault(key, source)
            seen.add((history.model, key))
        if not sources:
            raise ValueError(
                "Native sampled history contains no supported sampled sources"
            )
        encountered.append((history, list(sources.values())))
    if not expected or seen != expected.keys():
        raise ValueError("Native sampled source inventory is incomplete")

    resolved: dict[str, Tokenizer] = {}
    assembled: list[TokenizedHistory] = []
    for history, source_list in encountered:
        selected_model = history.model
        assert selected_model is not None
        if selected_model not in resolved:
            resolved[selected_model] = _load_sampled_stop_tokenizer(
                selected_model, base_model=base_model
            )
        bound = resolved[selected_model]
        runs: list[list[_Span]] = []
        for source in source_list:
            span = _singleton(source, bound)
            previous = runs[-1][-1] if runs else None
            if (
                previous is not None
                and len(previous.value.tokens) <= span.start
                and span.value.tokens[: len(previous.value.tokens)]
                == previous.value.tokens
            ):
                runs[-1].append(span)
            else:
                runs.append([span])
        for run in runs:
            joined = _join(history, run, bound)
            assembled.extend(
                [joined] if joined is not None else [span.value for span in run]
            )
    return TokenizedMultiHistoryTrajectory(trajectory=trajectory, histories=assembled)
