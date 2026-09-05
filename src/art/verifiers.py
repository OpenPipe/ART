from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import time
from typing import Any, cast

from openai.types.chat.chat_completion import Choice

from .trajectories import Trajectory, TrajectoryGroup
from .types import Messages, Tools


def trajectory_from_verifiers_rollout(output: Mapping[str, Any]) -> Trajectory:
    """Convert a verifiers RolloutOutput or serialized State into an ART Trajectory.

    When the output includes `trajectory` (for example from
    `env.run_rollout(..., state_columns=["trajectory"])`), the full multi-turn
    transcript is reconstructed from the verifiers steps. Otherwise the
    conversion falls back to `prompt + completion`.
    """

    messages = _messages_from_verifiers_output(output)
    metrics = dict(cast(Mapping[str, Any], output.get("metrics") or {}))
    metadata = _verifiers_metadata(output)
    tools = _openai_tools_from_verifiers_tools(output.get("tool_defs"))
    return Trajectory(
        messages_and_choices=messages,
        tools=tools,
        reward=float(output.get("reward") or 0.0),
        metrics=_numeric_metrics(metrics),
        metadata=metadata,
        logs=_string_list(output.get("logs")),
    ).finish()


def trajectory_group_from_verifiers_outputs(
    outputs: Iterable[Mapping[str, Any]],
) -> TrajectoryGroup:
    """Convert a group of verifiers rollout outputs into an ART TrajectoryGroup."""

    return TrajectoryGroup(
        [trajectory_from_verifiers_rollout(output) for output in outputs]
    )


def rollout_output_from_trajectory(
    trajectory: Trajectory,
    *,
    example_id: int = 0,
    prompt_length: int | None = None,
    include_trajectory: bool = True,
) -> dict[str, Any]:
    """Convert an ART Trajectory into a verifiers-compatible RolloutOutput dict.

    `prompt_length` controls the split between the initial prompt and the
    generated completion. If omitted, the split occurs before the first
    assistant message.
    """

    messages = _messages_from_art_items(trajectory.messages_and_choices)
    split_at = _prompt_length(messages, prompt_length)
    prompt = messages[:split_at]
    completion = messages[split_at:]
    output: dict[str, Any] = {
        "example_id": example_id,
        "prompt": prompt,
        "completion": completion,
        "reward": trajectory.reward,
        "timing": {
            "start_time": time.time(),
            "setup": {"start": 0.0, "end": 0.0, "duration": 0.0},
            "generation": {"start": 0.0, "end": 0.0, "duration": 0.0},
            "scoring": {"start": 0.0, "end": 0.0, "duration": 0.0},
            "model": {"spans": [], "duration": 0.0},
            "env": {"spans": [], "duration": 0.0},
            "total": 0.0,
            "overhead": 0.0,
        },
        "is_completed": True,
        "is_truncated": bool(trajectory.metadata.get("is_truncated", False)),
        "metrics": dict(trajectory.metrics),
        "tool_defs": _verifiers_tools_from_openai_tools(trajectory.tools),
    }
    if include_trajectory:
        output["trajectory"] = [
            {
                "prompt": prompt,
                "completion": completion,
                "response": None,
                "tokens": None,
                "reward": trajectory.reward,
                "advantage": None,
                "is_truncated": output["is_truncated"],
                "trajectory_id": str(trajectory.metadata.get("trajectory_id", "")),
                "extras": {"art_metadata": dict(trajectory.metadata)},
            }
        ]
    return output


def rollout_outputs_from_trajectory_group(
    group: TrajectoryGroup,
    *,
    first_example_id: int = 0,
    prompt_length: int | None = None,
    include_trajectory: bool = True,
) -> list[dict[str, Any]]:
    """Convert an ART TrajectoryGroup into verifiers-compatible outputs."""

    return [
        rollout_output_from_trajectory(
            trajectory,
            example_id=first_example_id + index,
            prompt_length=prompt_length,
            include_trajectory=include_trajectory,
        )
        for index, trajectory in enumerate(group.trajectories)
    ]


def normalize_verifiers_rollout_output(
    output: Mapping[str, Any],
    *,
    prompt_length: int | None = None,
    include_trajectory: bool = True,
) -> dict[str, Any]:
    """Round-trip a verifiers output through ART and back to verifiers shape.

    This is useful for portability checks and for tools that need a normalized
    RolloutOutput-compatible payload after ART has inspected or transformed the
    trajectory.
    """

    trajectory = trajectory_from_verifiers_rollout(output)
    example_id_value = output.get("example_id")
    example_id = example_id_value if isinstance(example_id_value, int) else 0
    normalized = rollout_output_from_trajectory(
        trajectory,
        example_id=example_id,
        prompt_length=prompt_length,
        include_trajectory=include_trajectory,
    )
    if "timing" in output:
        normalized["timing"] = deepcopy(output["timing"])
    if "logs" in output:
        normalized["logs"] = _string_list(output.get("logs"))
    if "answer" in output:
        normalized["answer"] = output.get("answer")
    if "stop_condition" in output:
        normalized["stop_condition"] = output.get("stop_condition")
    normalized["is_completed"] = bool(output.get("is_completed", True))
    normalized["is_truncated"] = bool(output.get("is_truncated", normalized["is_truncated"]))
    return normalized


def normalize_verifiers_rollout_outputs(
    outputs: Iterable[Mapping[str, Any]],
    *,
    prompt_length: int | None = None,
    include_trajectory: bool = True,
) -> list[dict[str, Any]]:
    """Normalize a collection of verifiers outputs through ART trajectories."""

    return [
        normalize_verifiers_rollout_output(
            output,
            prompt_length=prompt_length,
            include_trajectory=include_trajectory,
        )
        for output in outputs
    ]


async def rollout_with_verifiers_environment(
    env: Any,
    model: Any,
    input: Mapping[str, Any],
    *,
    sampling_args: Mapping[str, Any] | None = None,
    max_retries: int = 0,
    state_columns: Sequence[str] = ("trajectory",),
) -> Trajectory:
    """Run a verifiers Environment with an ART model and return an ART Trajectory."""

    output = await _run_verifiers_rollout(
        env,
        model,
        input,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
    )
    return trajectory_from_verifiers_rollout(output)


async def trajectory_group_with_verifiers_environment(
    env: Any,
    model: Any,
    group_inputs: Sequence[Mapping[str, Any]],
    *,
    sampling_args: Mapping[str, Any] | None = None,
    max_retries: int = 0,
    state_columns: Sequence[str] = ("trajectory",),
) -> TrajectoryGroup:
    """Run a verifiers Environment group with an ART model."""

    try:
        from verifiers.clients.openai_chat_completions_client import (
            OpenAIChatCompletionsClient,
        )
    except ImportError as exc:
        raise ImportError(
            "art.verifiers requires the optional `verifiers` package. "
            "Install it with `pip install verifiers`."
        ) from exc

    client = OpenAIChatCompletionsClient(model.openai_client())
    outputs = await env.run_group(
        group_inputs=list(group_inputs),
        client=client,
        model=model.get_inference_name(),
        sampling_args=dict(sampling_args or {"n": 1}),
        max_retries=max_retries,
        state_columns=list(state_columns),
    )
    return trajectory_group_from_verifiers_outputs(outputs)


async def _run_verifiers_rollout(
    env: Any,
    model: Any,
    input: Mapping[str, Any],
    *,
    sampling_args: Mapping[str, Any] | None,
    max_retries: int,
    state_columns: Sequence[str],
) -> Mapping[str, Any]:
    try:
        from verifiers.clients.openai_chat_completions_client import (
            OpenAIChatCompletionsClient,
        )
    except ImportError as exc:
        raise ImportError(
            "art.verifiers requires the optional `verifiers` package. "
            "Install it with `pip install verifiers`."
        ) from exc

    client = OpenAIChatCompletionsClient(model.openai_client())
    return await env.run_rollout(
        input=dict(input),
        client=client,
        model=model.get_inference_name(),
        sampling_args=dict(sampling_args or {"n": 1}),
        max_retries=max_retries,
        state_columns=list(state_columns),
    )


def _messages_from_verifiers_output(output: Mapping[str, Any]) -> Messages:
    trajectory = output.get("trajectory")
    if isinstance(trajectory, Sequence) and not isinstance(trajectory, (str, bytes)):
        messages = _messages_from_verifiers_steps(trajectory)
        if messages:
            return messages

    prompt = _coerce_messages(output.get("prompt"))
    completion = _coerce_messages(output.get("completion"))
    return [*prompt, *completion]


def _messages_from_verifiers_steps(steps: Sequence[Any]) -> Messages:
    messages: list[dict[str, Any]] = []
    for raw_step in steps:
        if not isinstance(raw_step, Mapping):
            continue
        prompt = _coerce_messages(raw_step.get("prompt"))
        completion = _coerce_messages(raw_step.get("completion"))
        _append_with_prefix_dedupe(messages, prompt)
        messages.extend(completion)
    return cast(Messages, messages)


def _append_with_prefix_dedupe(
    messages: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> None:
    if not incoming:
        return
    if not messages:
        messages.extend(incoming)
        return
    if incoming[: len(messages)] == messages:
        messages.extend(incoming[len(messages) :])
        return
    messages.extend(incoming)


def _coerce_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(f"Expected a message list or string, got {type(value).__name__}")

    messages: list[dict[str, Any]] = []
    for message in value:
        messages.append(_message_to_dict(message))
    return messages


def _messages_from_art_items(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [_message_to_dict(item) for item in items]


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, Choice):
        data = message.message.model_dump(mode="json", exclude_none=True)
        data["role"] = "assistant"
        return data
    if hasattr(message, "model_dump"):
        return cast(dict[str, Any], message.model_dump(mode="json", exclude_none=True))
    if isinstance(message, Mapping):
        return deepcopy(dict(message))
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def _prompt_length(messages: Sequence[Mapping[str, Any]], prompt_length: int | None) -> int:
    if prompt_length is not None:
        if prompt_length < 0 or prompt_length > len(messages):
            raise ValueError("prompt_length must be between 0 and the number of messages")
        return prompt_length
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            return index
    return len(messages)


def _numeric_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int | bool]:
    numeric: dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int, bool)):
            numeric[key] = value
    return numeric


def _verifiers_metadata(output: Mapping[str, Any]) -> dict[str, float | int | str | bool | None]:
    metadata: dict[str, float | int | str | bool | None] = {}
    fields = (
        "example_id",
        "is_completed",
        "is_truncated",
        "stop_condition",
        "answer",
    )
    for field in fields:
        value = output.get(field)
        if isinstance(value, (float, int, str, bool)) or value is None:
            metadata[f"verifiers_{field}"] = value
    return metadata


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, str)]


def _openai_tools_from_verifiers_tools(value: Any) -> Tools | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    tools: list[dict[str, Any]] = []
    for tool in value:
        tool_dict = _model_or_mapping_to_dict(tool)
        if tool_dict.get("type") == "function":
            tools.append(tool_dict)
            continue
        name = tool_dict.get("name")
        parameters = tool_dict.get("parameters")
        if isinstance(name, str) and isinstance(parameters, Mapping):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool_dict.get("description", "")),
                        "parameters": dict(parameters),
                    },
                }
            )
    return cast(Tools, tools) or None


def _verifiers_tools_from_openai_tools(value: Tools | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    tools: list[dict[str, Any]] = []
    for tool in value:
        tool_dict = _model_or_mapping_to_dict(tool)
        function = tool_dict.get("function")
        if tool_dict.get("type") == "function" and isinstance(function, Mapping):
            tools.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
            )
            continue
        if {"name", "parameters"} <= set(tool_dict):
            tools.append(tool_dict)
    return tools


def _model_or_mapping_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    return {}
