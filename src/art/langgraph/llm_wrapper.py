"""LLM wrapper with logging functionality."""

import asyncio
from collections.abc import Callable
import contextvars
import json
import os
from typing import Any, Literal
import uuid

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from art.trajectories import History, Trajectory

from .logging import FileLogger
from .message_utils import convert_langgraph_messages

CURRENT_CONFIG = contextvars.ContextVar("CURRENT_CONFIG")

mappings = {}

DEFAULT_INVOKE_TIMEOUT = 10 * 60
OPENAI_COMPATIBLE_PROVIDERS = {None, "openai", "openai-compatible", "openai_compatible"}


def add_thread(thread_id, base_url, api_key, model):
    log_path = f".art/langgraph/{thread_id}"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = FileLogger(log_path)
    CURRENT_CONFIG.set(
        {
            "logger": logger,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }
    )
    return logger


def create_messages_from_logs(logger: FileLogger, trajectory: Trajectory):
    logs = logger.load_logs()
    conversations = []
    tools = []

    for log_entry in logs:
        output = log_entry[1]["output"]
        new_tools = log_entry[1]["tools"]
        raw_output = output.get("raw") if hasattr(output, "get") else output

        input_msgs = (
            log_entry[1]["input"].to_messages()
            if isinstance(log_entry[1]["input"], ChatPromptValue)
            else log_entry[1]["input"]
        )
        new_conversation = input_msgs + [raw_output]

        # Try to match with existing conversations
        matched = False
        for idx, existing in enumerate(conversations):
            existing_non_tool = [m for m in existing if not isinstance(m, ToolMessage)]
            new_non_tool = [m for m in input_msgs if not isinstance(m, ToolMessage)]
            new_non_tool = (
                new_non_tool[:-1]
                if new_non_tool and isinstance(new_non_tool[-1], HumanMessage)
                else new_non_tool
            )

            if existing_non_tool == new_non_tool:
                # Replace with the longer one
                conversations[idx] = new_conversation
                tools[idx] = new_tools
                matched = True
                break

        if not matched:
            conversations.append(new_conversation)
            tools.append(new_tools)

    for idx, conv in enumerate(conversations):
        try:
            converted = convert_langgraph_messages(conv)
            if idx == 0:
                trajectory.messages_and_choices = converted
                trajectory.tools = tools[idx]
            else:
                trajectory.additional_histories.append(
                    History(messages_and_choices=converted, tools=tools[idx])
                )
        except Exception:
            pass

    return trajectory


def wrap_rollout(model, fn):
    async def wrapper(*args, **kwargs):
        thread_id = str(uuid.uuid4())
        logger = add_thread(
            thread_id,
            model.inference_base_url,
            model.inference_api_key,
            model.inference_model_name,
        )
        result = await fn(*args, **kwargs)
        return create_messages_from_logs(logger, result)

    return wrapper


def init_chat_model(
    model: str | Runnable | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal[None] = None,
    config_prefix: str | None = None,
    invoke_timeout: float | None = DEFAULT_INVOKE_TIMEOUT,
    **kwargs: Any,
):
    """Create a logged LangChain chat model for ART LangGraph rollouts.

    By default ART constructs a ChatOpenAI client pointed at the
    OpenAI-compatible endpoint from the active rollout context. For other
    LangChain providers, pass an already constructed chat model instance as
    ``model``. Provider kwargs such as ``temperature`` and ``timeout`` are
    forwarded to ChatOpenAI; ``invoke_timeout`` controls only ART's outer
    ``asyncio.wait_for`` timeout.
    """
    config = CURRENT_CONFIG.get()

    if configurable_fields is not None:
        raise ValueError(
            "configurable_fields is not supported by ART's init_chat_model"
        )
    if config_prefix is not None:
        raise ValueError("config_prefix is not supported by ART's init_chat_model")

    if model is not None and not isinstance(model, str):
        return LoggingLLM(
            model,
            config["logger"],
            invoke_timeout=invoke_timeout,
        )

    if model_provider not in OPENAI_COMPATIBLE_PROVIDERS:
        raise ValueError(
            "ART's init_chat_model can construct only OpenAI-compatible chat "
            "models. Pass a LangChain chat model instance as `model` to use "
            f"provider {model_provider!r}."
        )

    model_name = model

    def chat_openai_factory(art_config: dict[str, Any]):
        chat_model_kwargs: dict[str, Any] = {
            "base_url": art_config["base_url"],
            "api_key": art_config["api_key"],
            "model": model_name or art_config["model"],
            "temperature": 1.0,
        }
        chat_model_kwargs.update(kwargs)
        return ChatOpenAI(**chat_model_kwargs)

    return LoggingLLM(
        chat_openai_factory(config),
        config["logger"],
        invoke_timeout=invoke_timeout,
        chat_model_factory=chat_openai_factory,
    )


class LoggingLLM(Runnable):
    def __init__(
        self,
        llm,
        logger,
        structured_output=None,
        tools=None,
        invoke_timeout: float | None = DEFAULT_INVOKE_TIMEOUT,
        chat_model_factory: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.llm = llm
        self.logger = logger
        self.structured_output = structured_output
        self.tools = [convert_to_openai_tool(t) for t in tools] if tools else None
        self.invoke_timeout = invoke_timeout
        self.chat_model_factory = chat_model_factory

    def _log(self, completion_id, input, output):
        if self.logger:
            entry = {"input": input, "output": output, "tools": self.tools}
            self.logger.log(f"{completion_id}", entry)

    def invoke(self, input, config=None, **kwargs):
        completion_id = str(uuid.uuid4())

        def execute():
            result = self.llm.invoke(input, config=config, **kwargs)
            self._log(completion_id, input, result)
            return result

        result = execute()

        tool_calls = getattr(result, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                if isinstance(tool_call["args"], str):
                    tool_call["args"] = json.loads(tool_call["args"])

        if self.structured_output:
            return self.structured_output.model_validate(
                tool_calls[0]["args"] if tool_calls else None
            )
        return result

    async def ainvoke(self, input, config=None, **kwargs):
        completion_id = str(uuid.uuid4())

        async def execute():
            try:
                call = self.llm.ainvoke(input, config=config, **kwargs)
                if self.invoke_timeout is None:
                    result = await call
                else:
                    result = await asyncio.wait_for(call, timeout=self.invoke_timeout)
                self._log(completion_id, input, result)
            except asyncio.TimeoutError as e:
                raise e
            return result

        result = await execute()

        tool_calls = getattr(result, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                if isinstance(tool_call["args"], str):
                    tool_call["args"] = json.loads(tool_call["args"])

        if self.structured_output:
            return self.structured_output.model_validate(
                tool_calls[0]["args"] if tool_calls else None
            )
        return result

    def with_structured_output(self, tools):
        return LoggingLLM(
            self.llm.bind_tools([tools]),
            self.logger,
            structured_output=tools,
            tools=[tools],
            invoke_timeout=self.invoke_timeout,
            chat_model_factory=self.chat_model_factory,
        )

    def bind_tools(self, tools):
        return LoggingLLM(
            self.llm.bind_tools(tools),
            self.logger,
            tools=tools,
            invoke_timeout=self.invoke_timeout,
            chat_model_factory=self.chat_model_factory,
        )

    def with_retry(
        self,
        *,
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        exponential_jitter_params=None,
        stop_after_attempt=3,
    ):
        return self

    def with_config(
        self,
        config=None,
        **kwargs: Any,
    ):
        art_config = CURRENT_CONFIG.get()
        self.logger = art_config["logger"]

        if self.chat_model_factory is not None:
            configured_llm = self.chat_model_factory(art_config)
            if hasattr(self.llm, "bound"):
                setattr(self.llm, "bound", configured_llm)
            else:
                self.llm = configured_llm
        elif hasattr(self.llm, "with_config"):
            self.llm = self.llm.with_config(config=config, **kwargs)

        return self
