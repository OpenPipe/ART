from copy import deepcopy
import json
from typing import Iterable, List, Literal, Tuple

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pydantic import create_model


def freeze_tool_schema(tool: dict, fixed_args: dict) -> ChatCompletionToolParam:
    """
    Return a clone of *tool* whose parameters schema permits *only* `fixed_args`.
    Each field is cast to typing.Literal[value] so Pydantic emits an
    enum-of-one in the JSON schema, which vLLM's `guided_json` accepts.
    """
    fields = {k: (Literal[v], ...) for k, v in fixed_args.items()}  # ty:ignore[invalid-type-form]
    FrozenModel = create_model(
        f"{tool['function']['name'].title()}FrozenArgs",
        **fields,
    )  # ty:ignore[no-matching-overload]

    locked = deepcopy(tool)
    locked["function"]["parameters"] = FrozenModel.model_json_schema()
    return locked  # type: ignore


def get_guided_completion_params(
    completion: ChatCompletion,
    base_tools: Iterable[ChatCompletionToolParam] | None = None,
) -> Tuple[
    List[str] | None,
    ChatCompletionToolChoiceOptionParam | None,
    ChatCompletionToolParam | None,
]:
    """
    Given a completion from a teacher model, returns chat completion params that can be used to guide a student model's response.
    Useful for RL-based distillation.

    When guiding the student model's completion, remember to set `num_scheduler_steps` to 1.

    Args:
        completion: The completion of a teacher model
        base_tools: The base tools available to the teacher model

    Returns a tuple of (guided_choice, tool_choice, tool_params).
    """
    guided_choice, tool_choice, tool_params = None, None, None

    if (
        completion.choices[0].message.tool_calls
        and len(completion.choices[0].message.tool_calls) > 0
    ):
        tool_call = completion.choices[0].message.tool_calls[0]
        if not tool_call:
            raise ValueError("No tool call found in completion")
        if base_tools is None:
            raise ValueError("No base tools provided")
        assert isinstance(tool_call, ChatCompletionMessageFunctionToolCall), (
            "Only function tool calls are supported"
        )
        tool_name = tool_call.function.name
        tool_choice = {
            "type": "function",  # ← must call it
            "function": {"name": tool_name},
        }
        chosen_tool = next(t for t in base_tools if t["function"]["name"] == tool_name)
        tool_params = [
            freeze_tool_schema(chosen_tool, json.loads(tool_call.function.arguments))
        ]
    else:
        content = completion.choices[0].message.content
        guided_choice = [content]
    return (guided_choice, tool_choice, tool_params)  # type: ignore
