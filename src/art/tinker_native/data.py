from __future__ import annotations

import json
import re
from typing import Any, Iterable, cast

from openai.types.chat.chat_completion import Choice
import tinker
from tinker_cookbook import renderers
import torch

from ..trajectories import History, Trajectory, TrajectoryGroup, get_messages
from ..types import MessagesAndChoices


def _create_conversation_prefix_with_tools_fallback(
    tools: list[dict[str, Any]], system_prompt: str = ""
) -> list[dict[str, Any]]:
    """Fallback implementation for create_conversation_prefix_with_tools.

    Used when the installed tinker_cookbook version doesn't have this method.
    Implements the Qwen3 tool format.
    """
    tools_text = ""
    if tools:
        # Each tool is wrapped in {"type": "function", "function": {...}} per OpenAI format
        tool_lines = "\n".join(
            json.dumps({"type": "function", "function": tool}, separators=(", ", ": "))
            for tool in tools
        )
        tools_text = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

    # Add separator between system prompt and tools if system prompt exists
    if system_prompt:
        content = system_prompt + "\n\n" + tools_text
    else:
        content = tools_text

    return [{"role": "system", "content": content}]


def create_conversation_prefix_with_tools(
    renderer: Any, tools: list[dict[str, Any]], system_prompt: str = ""
) -> list[dict[str, Any]]:
    """Create conversation prefix with tools, using renderer method or fallback."""
    if hasattr(renderer, "create_conversation_prefix_with_tools"):
        return renderer.create_conversation_prefix_with_tools(tools, system_prompt)
    return _create_conversation_prefix_with_tools_fallback(tools, system_prompt)


def compute_advantages(
    rewards: list[float], normalize_advantages: bool = True
) -> list[float]:
    if not rewards:
        return []
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    centered = rewards_tensor - rewards_tensor.mean()
    if not normalize_advantages:
        return centered.tolist()
    std_reward = rewards_tensor.std()
    if std_reward > 1e-8:
        return (centered / std_reward).tolist()
    return [0.0] * len(rewards)


def convert_openai_messages_to_renderer_format(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    renderer: Any,
) -> list[dict[str, Any]]:
    if tools and len(messages) > 0 and messages[0].get("role") == "system":
        original_system = messages[0].get("content", "")

        tool_specs = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tool_specs.append(func)
            else:
                tool_specs.append(tool)

        tool_messages = create_conversation_prefix_with_tools(
            renderer, tool_specs, system_prompt=original_system
        )

        converted = list(tool_messages)
        messages = messages[1:]
    else:
        converted = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            converted.append({"role": "system", "content": content})

        elif role == "user":
            converted.append({"role": "user", "content": content})

        elif role == "assistant":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content or "",
            }

            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = []
                for tool_call in msg["tool_calls"]:
                    func = tool_call.get("function", {})
                    tool_calls.append(
                        renderers.ToolCall(
                            id=tool_call.get("id", ""),
                            function=renderers.ToolCall.FunctionBody(
                                name=func.get("name", ""),
                                arguments=func.get("arguments", "{}"),
                            ),
                        )
                    )
                assistant_msg["tool_calls"] = tool_calls

            converted.append(assistant_msg)

        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "name": msg.get("name", ""),
                }
            )

    return converted


def _extract_gpt_oss_tool_calls(content: str) -> tuple[str, list[dict[str, Any]]]:
    tool_calls = []
    cleaned_content = content

    pattern = r"<assistant to=functions\.(\w+)>(\{[^}]*\})(?:<\|call\|>)?"

    matches = list(re.finditer(pattern, content))
    for i, match in enumerate(matches):
        func_name = match.group(1)
        args_json = match.group(2)

        tool_calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": args_json,
                },
            }
        )

        cleaned_content = cleaned_content.replace(match.group(0), "").strip()

    return cleaned_content, tool_calls


def parse_completion_to_openai_message(
    completion_tokens: list[int],
    renderer: Any,
) -> dict[str, Any]:
    message, _ = renderer.parse_response(completion_tokens)

    result: dict[str, Any] = {"role": "assistant"}

    content = message.get("content", "")
    if isinstance(content, str):
        result["content"] = content
    else:
        text_parts = []
        for part in content:
            if part["type"] == "text":
                text_parts.append(part["text"])
            elif part["type"] == "thinking":
                text_parts.append(part["thinking"])
        result["content"] = "".join(text_parts)

    if "tool_calls" in message and message["tool_calls"]:
        result["tool_calls"] = [
            {
                "id": tool_call.id or f"call_{i}",
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for i, tool_call in enumerate(message["tool_calls"])
        ]
    else:
        if result.get("content") and "<assistant to=functions." in result["content"]:
            cleaned_content, extracted_tool_calls = _extract_gpt_oss_tool_calls(
                result["content"]
            )
            if extracted_tool_calls:
                result["content"] = cleaned_content
                result["tool_calls"] = extracted_tool_calls

    return result


def _trajectory_has_choice(trajectory: Trajectory) -> bool:
    for message_or_choice in trajectory.messages_and_choices:
        if isinstance(message_or_choice, Choice):
            return True
    for history in trajectory.additional_histories:
        for message_or_choice in history.messages_and_choices:
            if isinstance(message_or_choice, Choice):
                return True
    return False


def trajectory_groups_to_datums(
    trajectory_groups: Iterable[TrajectoryGroup],
    renderer: Any,
    tokenizer: Any,
    normalize_advantages: bool = True,
) -> list[tinker.Datum]:
    datums: list[tinker.Datum] = []

    for group in trajectory_groups:
        if not group.trajectories:
            continue
        for trajectory in group.trajectories:
            if not _trajectory_has_choice(trajectory):
                raise ValueError(
                    "Trajectory is missing a Choice object. Training requires at least one Choice "
                    "to compute logprobs. Ensure your rollout includes an OpenAI Choice in "
                    "Trajectory.messages_and_choices."
                )
        rewards = [trajectory.reward for trajectory in group.trajectories]
        advantages = compute_advantages(rewards, normalize_advantages)

        if all(advantage == 0.0 for advantage in advantages):
            continue
        for trajectory, advantage in zip(group.trajectories, advantages):
            for history in iter_trajectory_histories(trajectory):
                datum = history_to_datum(history, advantage, renderer, tokenizer)
                if datum is not None:
                    datums.append(datum)

    return datums


def iter_trajectory_histories(trajectory: Trajectory) -> Iterable[History]:
    yield History(
        messages_and_choices=trajectory.messages_and_choices,
        tools=trajectory.tools,
    )
    yield from trajectory.additional_histories


def find_last_choice(
    messages_and_choices: MessagesAndChoices,
) -> tuple[int, Choice] | None:
    for idx in range(len(messages_and_choices) - 1, -1, -1):
        message = messages_and_choices[idx]
        if isinstance(message, Choice):
            return idx, message
    return None


def extract_logprobs_from_choice(
    choice: Choice, tokenizer: Any
) -> tuple[list[int], list[float]]:
    if choice.logprobs is None:
        return [], []
    token_logprobs = choice.logprobs.content or choice.logprobs.refusal or []
    tokens: list[int] = []
    logprobs: list[float] = []
    for token_logprob in token_logprobs:
        token_str = token_logprob.token or ""
        if token_str.startswith("token_id:"):
            try:
                token_id = int(token_str.split(":")[1])
            except ValueError:
                continue
            tokens.append(token_id)
            logprobs.append(token_logprob.logprob)
        else:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None:
                continue
            tokens.append(int(token_id))
            logprobs.append(token_logprob.logprob)
    return tokens, logprobs


def history_to_datum(
    history: History,
    advantage: float,
    renderer: Any,
    tokenizer: Any,
) -> tinker.Datum | None:
    choice_info = find_last_choice(history.messages_and_choices)
    if choice_info is None:
        return None
    choice_index, choice = choice_info

    completion_tokens, logprobs = extract_logprobs_from_choice(choice, tokenizer)
    if not completion_tokens or len(completion_tokens) != len(logprobs):
        return None

    prompt_messages = cast(
        list[dict[str, Any]], get_messages(history.messages_and_choices[:choice_index])
    )
    renderer_messages = convert_openai_messages_to_renderer_format(
        messages=prompt_messages,
        tools=cast(list[dict[str, Any]] | None, history.tools),
        renderer=renderer,
    )
    prompt_input = renderer.build_generation_prompt(renderer_messages)
    prompt_tokens = list(prompt_input.to_ints())

    return build_datum(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        logprobs=logprobs,
        advantage=advantage,
    )


def build_datum(
    prompt_tokens: list[int],
    completion_tokens: list[int],
    logprobs: list[float],
    advantage: float,
) -> tinker.Datum | None:
    if not prompt_tokens or not completion_tokens:
        return None
    ob_len = max(len(prompt_tokens) - 1, 0)

    all_tokens = prompt_tokens + completion_tokens
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]

    padded_logprobs = [0.0] * ob_len + list(logprobs)
    padded_advantages = [0.0] * ob_len + [advantage] * len(completion_tokens)
    action_mask = [0.0] * ob_len + [1.0] * len(completion_tokens)

    if not (
        len(input_tokens)
        == len(target_tokens)
        == len(padded_logprobs)
        == len(padded_advantages)
        == len(action_mask)
    ):
        return None

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": tinker.TensorData.from_torch(
                torch.tensor(padded_logprobs, dtype=torch.float32)
            ),
            "advantages": tinker.TensorData.from_torch(
                torch.tensor(padded_advantages, dtype=torch.float32)
            ),
            "mask": tinker.TensorData.from_torch(
                torch.tensor(action_mask, dtype=torch.float32)
            ),
        },
    )
