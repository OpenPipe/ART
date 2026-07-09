from __future__ import annotations

import json
from typing import Any, cast

from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from art.preprocessing.tokenize import (
    _apply_chat_template_token_ids,
    _messages_for_chat_template,
)
from art.trajectories import History, Trajectory, TrajectoryGroup
from art.types import MessagesAndChoices, Tools


def _tool_schema() -> Tools:
    return cast(
        Tools,
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up the weather forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                        },
                        "required": ["city", "days"],
                    },
                },
            }
        ],
    )


def _tool_call(*, city: str) -> dict[str, Any]:
    return {
        "id": "call_weather",
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "arguments": json.dumps({"city": city, "days": 3}),
        },
    }


def _tool_message(*, forecast: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": "call_weather",
        "content": json.dumps({"forecast": forecast}),
    }


def _choice_for_text(
    text: str,
    token_ids: list[int],
    *,
    tool_calls: list[dict[str, Any]] | None = None,
) -> Choice:
    return Choice.model_validate(
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": {
                "content": [
                    {
                        "token": f"token_id:{token_id}",
                        "bytes": list(str(token_id).encode("utf-8")),
                        "logprob": -0.1,
                        "top_logprobs": [],
                    }
                    for token_id in token_ids
                ],
                "refusal": None,
            },
            "message": {
                "content": "" if tool_calls else text,
                "refusal": None,
                "role": "assistant",
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": tool_calls or [],
            },
        }
    )


def _logprob_content(token_ids: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "token": f"token_id:{token_id}",
            "bytes": list(str(token_id).encode("utf-8")),
            "logprob": -0.1,
            "top_logprobs": [],
        }
        for token_id in token_ids
    ]


def _choice_with_token_metadata(
    choice: Choice,
    *,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
) -> Choice:
    payload = choice.model_dump(mode="python")
    payload["logprobs"]["content"] = _logprob_content(completion_token_ids)
    payload["prompt_token_ids"] = prompt_token_ids
    payload["token_ids"] = completion_token_ids
    return Choice.model_validate(payload)


def _rendered_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages_and_choices: MessagesAndChoices,
    tools: Tools | None,
) -> list[int]:
    return _apply_chat_template_token_ids(
        tokenizer,
        _messages_for_chat_template(tokenizer, messages_and_choices),
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
    )


def _attach_token_metadata_to_history(
    tokenizer: PreTrainedTokenizerBase,
    history: Trajectory | History,
) -> None:
    items = history.messages_and_choices
    for index, item in enumerate(items):
        if not isinstance(item, Choice):
            continue
        prompt_token_ids = _rendered_ids(tokenizer, items[:index], history.tools)
        rendered_ids = _rendered_ids(tokenizer, items[: index + 1], history.tools)
        completion_token_ids = rendered_ids[len(prompt_token_ids) :]
        items[index] = _choice_with_token_metadata(
            item,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
        )


def _attach_token_metadata(
    tokenizer: PreTrainedTokenizerBase,
    inputs: "ChatTemplateConformanceInputs",
) -> "ChatTemplateConformanceInputs":
    groups = (
        inputs.text_pack_group,
        inputs.tool_conversation_group,
        inputs.additional_histories_group,
    )
    trajectories = [
        inputs.non_final_tool_call_base,
        inputs.non_final_tool_call_mutated,
        inputs.unsupported_assistant_tool_calls,
        *(trajectory for group in groups for trajectory in group.trajectories),
    ]
    for trajectory in trajectories:
        _attach_token_metadata_to_history(tokenizer, trajectory)
        for history in trajectory.additional_histories:
            _attach_token_metadata_to_history(tokenizer, history)
    return inputs


def _messages_and_choices(*items: Any) -> MessagesAndChoices:
    return cast(MessagesAndChoices, list(items))


class ChatTemplateConformanceInputs(BaseModel):
    text_pack_group: TrajectoryGroup
    non_final_tool_call_base: Trajectory
    non_final_tool_call_mutated: Trajectory
    tool_conversation_group: TrajectoryGroup
    additional_histories_group: TrajectoryGroup
    sft_tool_conversation: Trajectory
    sft_tool_conversation_mutated: Trajectory
    unsupported_assistant_tool_calls: Trajectory


def build_chat_template_conformance_inputs(
    tokenizer: PreTrainedTokenizerBase,
) -> ChatTemplateConformanceInputs:
    maybe_ids = tokenizer.encode("maybe", add_special_tokens=False)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    lookup_ids = tokenizer.encode("lookup_weather", add_special_tokens=False)
    sunny_ids = tokenizer.encode("sunny", add_special_tokens=False)
    rainy_ids = tokenizer.encode("rainy", add_special_tokens=False)
    prior_yes_ids = tokenizer.encode("prior yes", add_special_tokens=False)

    tools = _tool_schema()

    inputs = ChatTemplateConformanceInputs(
        text_pack_group=TrajectoryGroup(
            [
                Trajectory(
                    messages_and_choices=_messages_and_choices(
                        {"role": "user", "content": "Respond with one word."},
                        _choice_for_text("maybe", maybe_ids),
                    ),
                    reward=1.0,
                ),
                Trajectory(
                    messages_and_choices=_messages_and_choices(
                        {"role": "user", "content": "Respond with one word."},
                        _choice_for_text("yes", yes_ids),
                    ),
                    reward=0.0,
                ),
            ]
        ),
        non_final_tool_call_base=Trajectory(
            messages_and_choices=_messages_and_choices(
                {"role": "user", "content": "What is the weather forecast?"},
                _choice_for_text(
                    "lookup_weather",
                    lookup_ids,
                    tool_calls=[_tool_call(city="San Francisco")],
                ),
                _tool_message(forecast="sunny"),
                _choice_for_text("sunny", sunny_ids),
            ),
            reward=1.0,
            tools=tools,
        ),
        non_final_tool_call_mutated=Trajectory(
            messages_and_choices=_messages_and_choices(
                {"role": "user", "content": "What is the weather forecast?"},
                _choice_for_text(
                    "lookup_weather",
                    lookup_ids,
                    tool_calls=[_tool_call(city="New York")],
                ),
                _tool_message(forecast="sunny"),
                _choice_for_text("sunny", sunny_ids),
            ),
            reward=1.0,
            tools=tools,
        ),
        tool_conversation_group=TrajectoryGroup(
            [
                Trajectory(
                    messages_and_choices=_messages_and_choices(
                        {
                            "role": "user",
                            "content": "What is the weather in San Francisco?",
                        },
                        _choice_for_text(
                            "lookup_weather",
                            lookup_ids,
                            tool_calls=[_tool_call(city="San Francisco")],
                        ),
                        _tool_message(forecast="sunny"),
                        _choice_for_text("sunny", sunny_ids),
                    ),
                    reward=1.0,
                    tools=tools,
                ),
                Trajectory(
                    messages_and_choices=_messages_and_choices(
                        {
                            "role": "user",
                            "content": "What is the weather in New York?",
                        },
                        _choice_for_text(
                            "lookup_weather",
                            lookup_ids,
                            tool_calls=[_tool_call(city="New York")],
                        ),
                        _tool_message(forecast="rainy"),
                        _choice_for_text("rainy", rainy_ids),
                    ),
                    reward=0.0,
                    tools=tools,
                ),
            ]
        ),
        additional_histories_group=TrajectoryGroup(
            [
                Trajectory(
                    messages_and_choices=_messages_and_choices(
                        {"role": "user", "content": "Answer with one word."},
                        _choice_for_text("maybe", maybe_ids),
                    ),
                    additional_histories=[
                        History(
                            messages_and_choices=_messages_and_choices(
                                {"role": "user", "content": "Previous turn."},
                                _choice_for_text("prior yes", prior_yes_ids),
                            ),
                        )
                    ],
                    reward=1.0,
                ),
                Trajectory(
                    messages_and_choices=_messages_and_choices(
                        {"role": "user", "content": "Answer with one word."},
                        _choice_for_text("yes", yes_ids),
                    ),
                    additional_histories=[
                        History(
                            messages_and_choices=_messages_and_choices(
                                {"role": "user", "content": "Previous turn."},
                                _choice_for_text("prior yes", prior_yes_ids),
                            ),
                        )
                    ],
                    reward=0.0,
                ),
            ]
        ),
        sft_tool_conversation=Trajectory(
            messages_and_choices=_messages_and_choices(
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call(city="San Francisco")],
                },
                _tool_message(forecast="sunny"),
                {"role": "assistant", "content": "It will be sunny."},
            ),
            tools=tools,
        ),
        sft_tool_conversation_mutated=Trajectory(
            messages_and_choices=_messages_and_choices(
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call(city="New York")],
                },
                _tool_message(forecast="sunny"),
                {"role": "assistant", "content": "It will be sunny."},
            ),
            tools=tools,
        ),
        unsupported_assistant_tool_calls=Trajectory(
            messages_and_choices=_messages_and_choices(
                {"role": "user", "content": "Use the weather tool."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call(city="San Francisco")],
                },
            ),
            tools=tools,
        ),
    )
    return _attach_token_metadata(tokenizer, inputs)
