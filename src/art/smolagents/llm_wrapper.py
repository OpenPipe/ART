"""LLM wrapper with logging functionality for smolagents."""

import contextvars
import os
import uuid
from typing import Any

from smolagents import ChatMessage, MessageRole, Model, Tool

from art.trajectories import History, Trajectory

from .logging import FileLogger
from .message_utils import convert_smolagents_input_messages, convert_smolagents_output_to_choice

CURRENT_CONFIG = contextvars.ContextVar("CURRENT_CONFIG")


def add_thread(thread_id, base_url, api_key, model):
    log_path = f".art/smolagents/{thread_id}"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    CURRENT_CONFIG.set(
        {
            "logger": FileLogger(log_path),
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }
    )
    return log_path


def create_messages_from_logs(log_path: str, trajectory: Trajectory):
    """Load logs and convert them to trajectory format."""
    logs = FileLogger(log_path).load_logs()

    if not logs:
        return trajectory

    conversations = []
    tools_list = []

    for log_entry in logs:
        try:
            entry_data = log_entry[1]
            input_msgs = entry_data["input"]       # list[ChatMessage]
            output_msg = entry_data["output"]      # ChatMessage from model
            tools = entry_data["tools"]

            new_conversation = {
                "input": input_msgs,
                "output": output_msg,
            }

            # Try to match with existing conversations by comparing input
            matched = False
            for idx, existing in enumerate(conversations):
                existing_input = existing["input"]
                # Compare non-TOOL_RESPONSE messages
                existing_non_tool = [m for m in existing_input if m.role != MessageRole.TOOL_RESPONSE]
                new_non_tool = [m for m in input_msgs if m.role != MessageRole.TOOL_RESPONSE]

                if len(existing_non_tool) == len(new_non_tool) and all(
                    e.content == n.content and e.role == n.role
                    for e, n in zip(existing_non_tool, new_non_tool)
                ):
                    # Replace with the longer conversation
                    conversations[idx] = new_conversation
                    tools_list[idx] = tools
                    matched = True
                    break

            if not matched:
                conversations.append(new_conversation)
                tools_list.append(tools)
        except Exception as e:
            print(f"Warning: Failed to parse log entry: {e}")
            continue

    # Convert conversations to trajectory format
    for idx, conv in enumerate(conversations):
        try:
            # Convert input messages
            input_converted = convert_smolagents_input_messages(conv["input"])
            # Convert output to Choice
            output_choice = convert_smolagents_output_to_choice(conv["output"])
            # Combine: input messages + output choice
            converted = input_converted + [output_choice]

            if idx == 0:
                trajectory.messages_and_choices = converted
                trajectory.tools = tools_list[idx]
            else:
                trajectory.additional_histories.append(
                    History(messages_and_choices=converted, tools=tools_list[idx])
                )
        except Exception as e:
            print(f"Warning: Failed to convert conversation {idx}: {e}")
            continue

    return trajectory


def wrap_rollout(model, fn):
    """Wrap a rollout function to log all model interactions."""
    async def wrapper(*args, **kwargs):
        thread_id = str(uuid.uuid4())
        log_path = add_thread(
            thread_id,
            model.inference_base_url,
            model.inference_api_key,
            model.inference_model_name,
        )
        result = await fn(*args, **kwargs)
        return create_messages_from_logs(log_path, result)

    return wrapper


def init_chat_model(**kwargs: Any) -> Model:
    """Get a LoggingModel instance with the current config."""
    config = CURRENT_CONFIG.get()

    from smolagents import LiteLLMModel

    base_model = LiteLLMModel(
        model_id=config["model"],
        api_base=config["base_url"],
        api_key=config["api_key"],
        **kwargs
    )

    return LoggingModel(base_model, config["logger"])


class LoggingModel(Model):
    """A Model wrapper that logs all generate() calls."""

    def __init__(self, base_model: Model, logger: FileLogger):
        super().__init__(
            flatten_messages_as_text=base_model.flatten_messages_as_text,
            tool_name_key=base_model.tool_name_key,
            tool_arguments_key=base_model.tool_arguments_key,
            model_id=base_model.model_id,
        )
        self.base_model = base_model
        self.logger = logger

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Generate a response and log it."""
        completion_id = str(uuid.uuid4())

        # Store tools for logging
        tools_for_log = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputs,
                }
            }
            for tool in tools_to_call_from
        ] if tools_to_call_from else None

        # Generate the response
        result = self.base_model.generate(
            messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs
        )

        # Log the interaction
        if self.logger:
            entry = {
                "input": messages,
                "output": result,
                "tools": tools_for_log
            }
            self.logger.log(completion_id, entry)

        return result
