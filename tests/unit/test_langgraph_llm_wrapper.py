import importlib
import sys
import types
from typing import Any

import pytest


class _FakeRunnable:
    pass


class _FakeMessage:
    pass


class _FakePromptValue:
    pass


class _FakeBoundLLM:
    def __init__(self, bound: Any, tools: list[Any]) -> None:
        self.bound = bound
        self.tools = tools

    def bind_tools(self, tools: list[Any]) -> "_FakeBoundLLM":
        return _FakeBoundLLM(self.bound, tools)


class _FakeChatOpenAI:
    instances: list["_FakeChatOpenAI"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.instances.append(self)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls.append(("invoke", input, {"config": config, **kwargs}))
        return types.SimpleNamespace(tool_calls=None)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls.append(("ainvoke", input, {"config": config, **kwargs}))
        return types.SimpleNamespace(tool_calls=None)

    def bind_tools(self, tools: list[Any]) -> _FakeBoundLLM:
        return _FakeBoundLLM(self, tools)


class _FakeLogger:
    def __init__(self) -> None:
        self.entries: list[tuple[str, Any]] = []

    def log(self, key: str, entry: Any) -> None:
        self.entries.append((key, entry))


@pytest.fixture
def llm_wrapper(monkeypatch: pytest.MonkeyPatch):
    _FakeChatOpenAI.instances.clear()

    messages_module = types.ModuleType("langchain_core.messages")
    setattr(messages_module, "AIMessage", _FakeMessage)
    setattr(messages_module, "BaseMessage", _FakeMessage)
    setattr(messages_module, "FunctionMessage", _FakeMessage)
    setattr(messages_module, "HumanMessage", _FakeMessage)
    setattr(messages_module, "SystemMessage", _FakeMessage)
    setattr(messages_module, "ToolMessage", _FakeMessage)

    prompt_values_module = types.ModuleType("langchain_core.prompt_values")
    setattr(prompt_values_module, "ChatPromptValue", _FakePromptValue)

    runnables_module = types.ModuleType("langchain_core.runnables")
    setattr(runnables_module, "Runnable", _FakeRunnable)

    function_calling_module = types.ModuleType("langchain_core.utils.function_calling")
    setattr(function_calling_module, "convert_to_openai_tool", lambda tool: tool)

    utils_module = types.ModuleType("langchain_core.utils")
    core_module = types.ModuleType("langchain_core")
    openai_module = types.ModuleType("langchain_openai")
    setattr(openai_module, "ChatOpenAI", _FakeChatOpenAI)

    for module_name, module in {
        "langchain_core": core_module,
        "langchain_core.messages": messages_module,
        "langchain_core.prompt_values": prompt_values_module,
        "langchain_core.runnables": runnables_module,
        "langchain_core.utils": utils_module,
        "langchain_core.utils.function_calling": function_calling_module,
        "langchain_openai": openai_module,
    }.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    for module_name in [
        "art.langgraph",
        "art.langgraph.llm_wrapper",
        "art.langgraph.message_utils",
    ]:
        sys.modules.pop(module_name, None)

    return importlib.import_module("art.langgraph.llm_wrapper")


def _set_current_config(module: Any, **overrides: Any) -> _FakeLogger:
    logger = overrides.pop("logger", _FakeLogger())
    module.CURRENT_CONFIG.set(
        {
            "logger": logger,
            "base_url": overrides.pop("base_url", "http://rollout.test/v1"),
            "api_key": overrides.pop("api_key", "test-key"),
            "model": overrides.pop("model", "context-model"),
            **overrides,
        }
    )
    return logger


def test_init_chat_model_forwards_model_and_provider_kwargs(llm_wrapper: Any) -> None:
    _set_current_config(llm_wrapper)

    logged_llm = llm_wrapper.init_chat_model(
        "explicit-model",
        temperature=0.2,
        timeout=123,
        max_tokens=42,
        invoke_timeout=5,
    )

    assert logged_llm.invoke_timeout == 5
    assert logged_llm.llm.kwargs == {
        "base_url": "http://rollout.test/v1",
        "api_key": "test-key",
        "model": "explicit-model",
        "temperature": 0.2,
        "timeout": 123,
        "max_tokens": 42,
    }


def test_with_config_preserves_kwargs_and_uses_new_context(
    llm_wrapper: Any,
) -> None:
    first_logger = _set_current_config(
        llm_wrapper,
        base_url="http://first.test/v1",
        api_key="first-key",
        model="first-model",
    )

    logged_llm = llm_wrapper.init_chat_model(temperature=0.3, invoke_timeout=None)
    assert logged_llm.logger is first_logger
    assert logged_llm.llm.kwargs["model"] == "first-model"

    second_logger = _set_current_config(
        llm_wrapper,
        base_url="http://second.test/v1",
        api_key="second-key",
        model="second-model",
    )

    assert logged_llm.with_config() is logged_llm
    assert logged_llm.logger is second_logger
    assert logged_llm.invoke_timeout is None
    assert logged_llm.llm.kwargs == {
        "base_url": "http://second.test/v1",
        "api_key": "second-key",
        "model": "second-model",
        "temperature": 0.3,
    }


def test_with_config_updates_bound_chat_model(llm_wrapper: Any) -> None:
    _set_current_config(llm_wrapper, model="first-model")
    logged_llm = llm_wrapper.init_chat_model("explicit-model", temperature=0.4)
    bound_llm = logged_llm.bind_tools(["tool"])

    _set_current_config(llm_wrapper, model="second-context-model")

    bound_llm.with_config()

    assert bound_llm.llm.tools == ["tool"]
    assert bound_llm.llm.bound.kwargs == {
        "base_url": "http://rollout.test/v1",
        "api_key": "test-key",
        "model": "explicit-model",
        "temperature": 0.4,
    }


def test_custom_chat_model_is_wrapped_without_chat_openai(llm_wrapper: Any) -> None:
    class CustomChatModel:
        pass

    _set_current_config(llm_wrapper)
    custom_model = CustomChatModel()

    logged_llm = llm_wrapper.init_chat_model(
        custom_model,
        model_provider="ollama",
        invoke_timeout=7,
    )

    assert logged_llm.llm is custom_model
    assert logged_llm.invoke_timeout == 7
    assert _FakeChatOpenAI.instances == []


def test_unsupported_provider_raises_instead_of_using_chat_openai(
    llm_wrapper: Any,
) -> None:
    _set_current_config(llm_wrapper)

    with pytest.raises(ValueError, match="OpenAI-compatible"):
        llm_wrapper.init_chat_model("llama3", model_provider="ollama")

    assert _FakeChatOpenAI.instances == []


@pytest.mark.asyncio
async def test_ainvoke_uses_wrapper_timeout_and_forwards_kwargs(
    llm_wrapper: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_wait_for(awaitable: Any, timeout: float | None) -> Any:
        seen["timeout"] = timeout
        return await awaitable

    monkeypatch.setattr(llm_wrapper.asyncio, "wait_for", fake_wait_for)

    logger = _set_current_config(llm_wrapper)
    logged_llm = llm_wrapper.init_chat_model(invoke_timeout=2)

    await logged_llm.ainvoke("hello", config={"run": 1}, stop=["END"])

    assert seen["timeout"] == 2
    assert logged_llm.llm.calls == [
        ("ainvoke", "hello", {"config": {"run": 1}, "stop": ["END"]})
    ]
    assert len(logger.entries) == 1
