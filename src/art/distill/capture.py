"""Exact, immutable generation capture helpers for distillation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from art.preprocessing.policy_spans import choice_policy_token_spans
from art.preprocessing.vllm_tokens import choice_vllm_token_metadata
from art.trajectories import ChatCompletionsExchange, Trajectory

from .types import (
    CapturedGeneration,
    GenerationPart,
    PartSpan,
    RolloutRevisionSpan,
    TeacherView,
    canonical_json,
    sha256_text,
)


def generations(trajectory: Trajectory) -> tuple[CapturedGeneration, ...]:
    """Capture exact output-only Chat Completions generations in trajectory order.

    V1 deliberately fails closed for legacy trajectories, non-chat protocols,
    multiple choices, and outputs whose token-level part boundaries are ambiguous.
    It never tokenizes rendered response text.
    """

    exchanges = _chat_exchanges(trajectory)
    events: list[_CapturedEvent] = []
    history_token_ids: list[int] = []

    for event_index, exchange in enumerate(exchanges):
        if len(exchange.response.choices) != 1:
            raise ValueError(
                "Distillation capture supports exactly one choice per "
                "Chat Completions exchange"
            )
        choice = exchange.response.choices[0]
        prompt_and_completion = choice_vllm_token_metadata(choice)
        if prompt_and_completion is None:
            raise ValueError(
                "Distillation capture requires exact prompt and completion token IDs"
            )
        prompt_token_ids, continuation_token_ids = prompt_and_completion
        if not continuation_token_ids:
            raise ValueError(
                "Distillation capture requires at least one exact completion token"
            )
        _require_output_only_assistant_text(choice.message)

        if not history_token_ids:
            history_token_ids.extend(prompt_token_ids)
        elif (
            len(prompt_token_ids) < len(history_token_ids)
            or prompt_token_ids[: len(history_token_ids)] != history_token_ids
        ):
            raise ValueError(
                "Chat Completions exchanges do not form one append-only exact "
                "token history"
            )
        else:
            history_token_ids.extend(prompt_token_ids[len(history_token_ids) :])

        trajectory_token_start = len(history_token_ids)
        history_token_ids.extend(continuation_token_ids)
        rollout_spans = _rollout_spans(
            trajectory,
            choice,
            completion_tokens=len(continuation_token_ids),
        )
        context = TeacherView.from_request(
            "chat_completions",
            exchange.request,
        )
        event_payload = {
            "protocol": "chat_completions",
            "request": context.request(),
            "response_id": exchange.response.id,
            "choice_index": choice.index,
            "prompt_token_ids": prompt_token_ids,
            "continuation_token_ids": continuation_token_ids,
            "rollout_spans": [span.model_dump(mode="json") for span in rollout_spans],
        }
        events.append(
            _CapturedEvent(
                event_index=event_index,
                trajectory_token_start=trajectory_token_start,
                continuation_token_ids=tuple(continuation_token_ids),
                context=context,
                rollout_spans=rollout_spans,
                fingerprint=sha256_text(canonical_json(event_payload)),
            )
        )

    trajectory_fingerprint = sha256_text(
        canonical_json([event.fingerprint for event in events])
    )
    return tuple(
        CapturedGeneration.create(
            generation_id=sha256_text(
                f"chat_completions\0{trajectory_fingerprint}\0"
                f"{event.event_index}\0{event.fingerprint}"
            ),
            trajectory_fingerprint=trajectory_fingerprint,
            event_index=event.event_index,
            trajectory_token_start=event.trajectory_token_start,
            protocol="chat_completions",
            continuation_token_ids=event.continuation_token_ids,
            context=event.context,
            part_spans=(
                PartSpan(
                    start=0,
                    end=len(event.continuation_token_ids),
                    part=GenerationPart.ASSISTANT_TEXT,
                ),
            ),
            rollout_spans=event.rollout_spans,
        )
        for event in events
    )


def last_generation(trajectory: Trajectory) -> CapturedGeneration:
    """Return the final captured model-output event."""

    captured = generations(trajectory)
    if not captured:
        raise ValueError("Trajectory contains no captured generations")
    return captured[-1]


def captured_context(generation: CapturedGeneration) -> TeacherView:
    """Return the immutable semantic request captured before this generation."""

    return generation.context


def prepend_message(
    view: TeacherView,
    message: Mapping[str, Any],
) -> TeacherView:
    """Return a new teacher view with one message prepended."""

    request = _chat_request(view)
    messages = request.get("messages")
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise ValueError("Chat Completions teacher view messages must be a list")
    request["messages"] = [dict(message), *messages]
    return TeacherView.from_request("chat_completions", request)


def append_message(
    view: TeacherView,
    message: Mapping[str, Any],
) -> TeacherView:
    """Return a new teacher view with one message appended."""

    request = _chat_request(view)
    messages = request.get("messages")
    if messages is None:
        messages = []
    if not isinstance(messages, list):
        raise ValueError("Chat Completions teacher view messages must be a list")
    request["messages"] = [*messages, dict(message)]
    return TeacherView.from_request("chat_completions", request)


def with_tools(view: TeacherView, tools: Any) -> TeacherView:
    """Return a new teacher view with its tools replaced."""

    if tools is not None and not isinstance(tools, list):
        raise ValueError("Chat Completions teacher view tools must be a list or None")
    request = _chat_request(view)
    request["tools"] = tools
    return TeacherView.from_request("chat_completions", request)


class _CapturedEvent:
    def __init__(
        self,
        *,
        event_index: int,
        trajectory_token_start: int,
        continuation_token_ids: tuple[int, ...],
        context: TeacherView,
        rollout_spans: tuple[RolloutRevisionSpan, ...],
        fingerprint: str,
    ) -> None:
        self.event_index = event_index
        self.trajectory_token_start = trajectory_token_start
        self.continuation_token_ids = continuation_token_ids
        self.context = context
        self.rollout_spans = rollout_spans
        self.fingerprint = fingerprint


def _chat_exchanges(trajectory: Trajectory) -> list[ChatCompletionsExchange]:
    if not trajectory.exchanges:
        raise ValueError(
            "Distillation capture requires exchange-backed trajectories; "
            "legacy histories are unsupported"
        )
    if (
        trajectory.exchanges.completions
        or trajectory.exchanges.responses
        or trajectory.exchanges.messages
    ):
        raise ValueError(
            "Distillation capture currently supports only Chat Completions exchanges"
        )
    exchanges = sorted(
        trajectory.exchanges.chat_completions,
        key=lambda exchange: (exchange.start_time, exchange.end_time),
    )
    for previous, current in zip(exchanges, exchanges[1:], strict=False):
        if (previous.start_time, previous.end_time) == (
            current.start_time,
            current.end_time,
        ):
            raise ValueError(
                "Distillation capture requires an unambiguous exchange order; "
                "two exchanges have identical timestamps"
            )
    return exchanges


def _require_output_only_assistant_text(message: Any) -> None:
    data = message.model_dump(mode="python", exclude_none=True)
    extra = message.model_extra or {}
    reasoning = any(
        extra.get(key) not in (None, "", [], {})
        for key in ("reasoning", "reasoning_content")
    )
    content = data.get("content")
    if (
        not isinstance(content, str)
        or not content
        or data.get("tool_calls")
        or data.get("refusal")
        or reasoning
    ):
        raise ValueError(
            "Distillation capture cannot infer token spans for mixed reasoning, "
            "refusal, or tool-call output; V1 supports assistant text only"
        )


def _rollout_spans(
    trajectory: Trajectory,
    choice: Any,
    *,
    completion_tokens: int,
) -> tuple[RolloutRevisionSpan, ...]:
    policy_spans = choice_policy_token_spans(choice)
    if policy_spans:
        cursor = 0
        captured: list[RolloutRevisionSpan] = []
        for span in policy_spans:
            if span.start_token != cursor or span.end_token > completion_tokens:
                raise ValueError(
                    "Policy token spans must form a contiguous completion partition"
                )
            captured.append(
                RolloutRevisionSpan(
                    start=span.start_token,
                    end=span.end_token,
                    revision=span.policy_version,
                    inference_name=span.lora_slot,
                    update_seq=span.update_seq,
                )
            )
            cursor = span.end_token
        if cursor != completion_tokens:
            raise ValueError(
                "Policy token spans must cover every captured completion token"
            )
        return tuple(captured)

    if (
        trajectory.initial_policy_version is not None
        and trajectory.initial_policy_version == trajectory.final_policy_version
    ):
        revision = trajectory.initial_policy_version
        return (
            RolloutRevisionSpan(
                start=0,
                end=completion_tokens,
                revision=revision,
            ),
        )
    return ()


def _chat_request(view: TeacherView) -> dict[str, Any]:
    if view.protocol != "chat_completions":
        raise ValueError(
            "Teacher-view editing currently supports only Chat Completions"
        )
    request = view.request()
    if not isinstance(request, dict):
        raise ValueError("Chat Completions teacher view must contain a JSON object")
    return request
