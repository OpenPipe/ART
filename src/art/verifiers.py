"""Interop helpers for moving rollout data between ART and Verifiers.

The helpers keep Verifiers optional: callers pass and receive plain mapping
objects that match the public Verifiers state/trajectory-step shape.
"""

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, cast

from openai.types.chat.chat_completion import Choice

from .trajectories import Trajectory, TrajectoryGroup
from .types import Message, MessageOrChoice, Messages, MessagesAndChoices

VerifiersState = dict[str, Any]
VerifiersStep = dict[str, Any]
ChoiceFactory = Callable[[dict[str, Any], Mapping[str, Any]], MessageOrChoice]


def trajectory_to_verifiers_state(
    trajectory: Trajectory,
    *,
    task: Mapping[str, Any] | None = None,
    example_id: str | int | None = None,
    answer: Any | None = None,
) -> VerifiersState:
    """Convert one ART trajectory into a Verifiers-compatible rollout state.

    ART stores a single flat ``messages_and_choices`` transcript. Verifiers
    stores rollout turns as steps containing the prompt that produced a model
    response plus the completion for that turn. This function creates one
    Verifiers step for each ART ``Choice`` and keeps the final ART reward and
    metrics on the state and final step.
    """

    task_data = dict(task or {})
    messages = _messages_and_choices_to_messages(trajectory.messages_and_choices)
    steps = _messages_and_choices_to_steps(trajectory.messages_and_choices)
    prompt = steps[0]["prompt"] if steps else messages
    completion = messages[len(prompt) :] if len(prompt) <= len(messages) else []

    if steps:
        steps[-1]["reward"] = float(trajectory.reward)
        steps[-1]["extras"]["art_metrics"] = dict(trajectory.metrics)
        steps[-1]["extras"]["art_metadata"] = dict(trajectory.metadata)
    is_truncated = any(bool(step.get("is_truncated")) for step in steps)

    state: VerifiersState = {
        "task": task_data,
        "prompt": deepcopy(prompt),
        "completion": deepcopy(completion),
        "trajectory": steps,
        "reward": float(trajectory.reward),
        "metrics": dict(trajectory.metrics),
        "is_completed": True,
        "is_truncated": is_truncated,
        "stop_condition": "art_trajectory_imported",
        "error": None,
    }
    if trajectory.metadata:
        state["metadata"] = dict(trajectory.metadata)
    if trajectory.logs:
        state["logs"] = list(trajectory.logs)
    if example_id is not None:
        state["example_id"] = example_id
    if answer is not None:
        state["answer"] = answer
    if "info" in task_data:
        state["info"] = deepcopy(task_data["info"])
    return state


def verifiers_state_to_trajectory(
    state: Mapping[str, Any],
    *,
    choice_factory: ChoiceFactory | None = None,
) -> Trajectory:
    """Convert a Verifiers rollout state into an ART trajectory.

    Verifiers states usually contain serializable assistant messages rather
    than OpenAI ``Choice`` objects with logprobs. By default those assistant
    turns remain plain, non-trainable messages. Pass ``choice_factory`` when a
    caller has the original response payload and wants selected assistant
    messages to become trainable ART choices.
    """

    messages_and_choices = _state_messages_and_choices(
        state, choice_factory=choice_factory
    )
    return Trajectory(
        messages_and_choices=messages_and_choices,
        reward=float(state.get("reward") or 0.0),
        metrics=dict(cast(Mapping[str, Any], state.get("metrics") or {})),
        metadata=_state_metadata(state),
        logs=list(cast(Iterable[str], state.get("logs") or [])),
    )


def verifiers_states_to_trajectory_group(
    states: Iterable[Mapping[str, Any]],
    *,
    choice_factory: ChoiceFactory | None = None,
) -> TrajectoryGroup:
    """Convert Verifiers rollout states into an ART trajectory group."""

    return TrajectoryGroup(
        [
            verifiers_state_to_trajectory(state, choice_factory=choice_factory)
            for state in states
        ]
    )


def _messages_and_choices_to_steps(
    messages_and_choices: MessagesAndChoices,
) -> list[VerifiersStep]:
    steps: list[VerifiersStep] = []
    transcript: Messages = []
    for item in messages_and_choices:
        if isinstance(item, Choice):
            completion = [_choice_to_message(item)]
            steps.append(
                {
                    "prompt": deepcopy(transcript),
                    "completion": deepcopy(completion),
                    "response": _choice_response_payload(item),
                    "tokens": None,
                    "reward": None,
                    "advantage": None,
                    "is_truncated": item.finish_reason == "length",
                    "extras": {},
                }
            )
            transcript.extend(completion)
        else:
            transcript.append(_normalize_message(item))
    return steps


def _messages_and_choices_to_messages(
    messages_and_choices: MessagesAndChoices,
) -> Messages:
    messages: Messages = []
    for item in messages_and_choices:
        if isinstance(item, Choice):
            messages.append(_choice_to_message(item))
        else:
            messages.append(_normalize_message(item))
    return messages


def _state_messages_and_choices(
    state: Mapping[str, Any],
    *,
    choice_factory: ChoiceFactory | None,
) -> MessagesAndChoices:
    steps = state.get("trajectory") or []
    if isinstance(steps, list) and steps:
        return _append_state_completion_tail(
            _steps_to_messages_and_choices(steps, choice_factory=choice_factory),
            state,
        )
    return _append_completion_messages(
        [_normalize_message(message) for message in state.get("prompt") or []],
        state.get("completion") or [],
        {},
        choice_factory=choice_factory,
    )


def _steps_to_messages_and_choices(
    steps: list[Any],
    *,
    choice_factory: ChoiceFactory | None,
) -> MessagesAndChoices:
    transcript: MessagesAndChoices = []
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            continue
        prompt = [
            _normalize_message(message) for message in raw_step.get("prompt") or []
        ]
        if not transcript:
            transcript.extend(prompt)
        elif _transcript_has_prefix(transcript, prompt):
            transcript.extend(prompt[len(transcript) :])
        else:
            transcript.extend(prompt)
        transcript = _append_completion_messages(
            transcript,
            raw_step.get("completion") or [],
            raw_step,
            choice_factory=choice_factory,
        )
    return transcript


def _append_state_completion_tail(
    transcript: MessagesAndChoices,
    state: Mapping[str, Any],
) -> MessagesAndChoices:
    prompt = [_normalize_message(message) for message in state.get("prompt") or []]
    completion = [
        _normalize_message(message) for message in state.get("completion") or []
    ]
    state_transcript = prompt + completion
    if not state_transcript:
        return transcript
    if _transcript_has_prefix(transcript, state_transcript):
        transcript.extend(state_transcript[len(transcript) :])
    return transcript


def _append_completion_messages(
    transcript: MessagesAndChoices,
    completion: Any,
    step: Mapping[str, Any],
    *,
    choice_factory: ChoiceFactory | None,
) -> MessagesAndChoices:
    for message in completion:
        normalized = _normalize_message(message)
        if choice_factory is not None and normalized.get("role") == "assistant":
            transcript.append(choice_factory(normalized, step))
        else:
            transcript.append(normalized)
    return transcript


def _transcript_has_prefix(
    transcript: MessagesAndChoices,
    prompt: Messages,
) -> bool:
    if len(prompt) < len(transcript):
        return False
    prefix = [_message_or_choice_to_dict(item) for item in transcript]
    return prefix == [dict(message) for message in prompt[: len(transcript)]]


def _choice_to_message(choice: Choice) -> Message:
    message = choice.message.model_dump(mode="json", exclude_none=True)
    message["role"] = "assistant"
    if message.get("content") is None:
        message["content"] = ""
    return cast(Message, message)


def _choice_response_payload(choice: Choice) -> dict[str, Any]:
    return {
        "choices": [choice.model_dump(mode="json", exclude_none=True)],
    }


def _normalize_message(message: Any) -> Message:
    if hasattr(message, "model_dump"):
        data = message.model_dump(mode="json", exclude_none=True)
    else:
        data = dict(message)
    if data.get("content") is None:
        data["content"] = ""
    return cast(Message, data)


def _message_or_choice_to_dict(item: MessageOrChoice) -> dict[str, Any]:
    if isinstance(item, Choice):
        return dict(_choice_to_message(item))
    return dict(_normalize_message(item))


def _state_metadata(
    state: Mapping[str, Any],
) -> dict[str, str | int | float | bool | None]:
    metadata = dict(cast(Mapping[str, Any], state.get("metadata") or {}))
    for key in ("example_id", "stop_condition"):
        if key not in state:
            continue
        value = state.get(key)
        if isinstance(value, str | int | float | bool) or value is None:
            metadata.setdefault(key, value)
    return metadata
