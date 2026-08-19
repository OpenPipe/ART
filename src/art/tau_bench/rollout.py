from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
import logging
import os
import time
from typing import Any, cast, overload

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    BadRequestError,
    DefaultAsyncHttpxClient,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

from art.costs import get_model_pricing, tokens_to_cost
from art.model import Model
from art.openai import (
    IncompleteChatCompletionStreamError,
    consume_chat_completion_stream,
)
from art.trajectories import Trajectory

from .client import Scenario, TauBenchClient, _get_default_client

openai_clients: dict[tuple[str, str], AsyncOpenAI] = {}
CONTEXT_TOKEN_LIMIT = 32_768
DEFAULT_MAX_COMPLETION_TOKENS = 4096
_POLICY_CONNECTION_LIMIT = 2048
_STREAM_RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@overload
async def rollout(
    scenario: Scenario,
    model: Model,
    /,
    *,
    client: TauBenchClient | None = None,
    max_turns: int | None = None,
    chat_completion_kwargs: dict[str, Any] | None = None,
    user_model_name: str = "gpt-4.1-2025-04-14",
    user_chat_completion_kwargs: dict[str, Any] | None = None,
    assert_costs: bool = False,
    retrieval_config: str | None = None,
    retrieval_config_kwargs: dict[str, Any] | None = None,
) -> Trajectory: ...


@overload
async def rollout(
    scenario: Scenario,
    base_url: str,
    api_key: str,
    model: str,
    /,
    *,
    client: TauBenchClient | None = None,
    base_model: str | None = None,
    max_turns: int | None = None,
    chat_completion_kwargs: dict[str, Any] | None = None,
    user_model_name: str = "gpt-4.1-2025-04-14",
    user_chat_completion_kwargs: dict[str, Any] | None = None,
    assert_costs: bool = False,
    retrieval_config: str | None = None,
    retrieval_config_kwargs: dict[str, Any] | None = None,
) -> Trajectory: ...


async def rollout(
    scenario: Scenario,
    base_url_or_model: str | Model,
    api_key: str | None = None,
    model: str | None = None,
    /,
    *,
    client: TauBenchClient | None = None,
    base_model: str | None = None,
    max_turns: int | None = None,
    chat_completion_kwargs: dict[str, Any] | None = None,
    user_model_name: str = "gpt-4.1-2025-04-14",
    user_chat_completion_kwargs: dict[str, Any] | None = None,
    assert_costs: bool = False,
    retrieval_config: str | None = None,
    retrieval_config_kwargs: dict[str, Any] | None = None,
) -> Trajectory:
    started = time.perf_counter()
    client = _get_default_client(client)
    task_id = scenario.task.id
    async with client.environment(
        domain=scenario.domain,
        task_id=task_id,
        user_llm=user_model_name,
        user_llm_args=(
            user_chat_completion_kwargs
            if user_chat_completion_kwargs is not None
            else default_user_llm_args(user_model_name)
        ),
        retrieval_config=retrieval_config,
        retrieval_config_kwargs=retrieval_config_kwargs,
    ) as env:
        environment_startup = time.perf_counter() - started
        chat_completion_kwargs = chat_completion_kwargs or {}
        openai_client, model_name, cost_model = _completion_client_and_model(
            base_url_or_model,
            api_key=api_key,
            model=model,
            base_model=base_model,
        )
        policy_completion_kwargs = dict(chat_completion_kwargs)
        stream_policy = bool(
            policy_completion_kwargs.setdefault(
                "stream", isinstance(base_url_or_model, str)
            )
        )
        if stream_policy:
            stream_options = dict(
                cast(
                    Mapping[str, Any],
                    policy_completion_kwargs.get("stream_options") or {},
                )
            )
            stream_options["include_usage"] = True
            policy_completion_kwargs["stream_options"] = stream_options
            extra_headers = dict(
                cast(
                    Mapping[str, str],
                    policy_completion_kwargs.get("extra_headers") or {},
                )
            )
            extra_headers.setdefault("X-ART-Stream-Progress", "sse-comments-v1")
            policy_completion_kwargs["extra_headers"] = extra_headers
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": env.info["policy"]},
            {"role": "user", "content": env.observation.removeprefix("user: ")},
        ]
        tools = env.info.get("tools") or []
        trajectory = Trajectory(
            reward=0,
            metrics={
                "cost/tinker/prefill": 0.0,
                "cost/tinker/sample": 0.0,
                "cost/user": 0.0,
            },
            metadata={"scenario_id": task_id},
        )
        terminated = False
        num_turns = 0
        with trajectory:
            while not terminated:
                if max_turns is not None and num_turns >= max_turns:
                    break
                try:
                    policy_started = time.perf_counter()
                    chat_completion = await _create_policy_completion(
                        openai_client,
                        messages=messages,
                        model=model_name,
                        stream_policy=stream_policy,
                        tool_choice="auto",
                        tools=tools,
                        **policy_completion_kwargs,
                    )
                    policy_latency = time.perf_counter() - policy_started
                    trajectory.metrics["latency/policy"] = (
                        trajectory.metrics.get("latency/policy", 0.0) + policy_latency
                    )
                    trajectory.metrics["latency/policy_max"] = max(
                        trajectory.metrics.get("latency/policy_max", 0.0),
                        policy_latency,
                    )
                except (BadRequestError, APIError) as exc:
                    if _is_max_tokens_error(exc):
                        break
                    raise
                _record_tinker_costs(
                    trajectory,
                    cost_model,
                    chat_completion.usage,
                    assert_costs=assert_costs,
                )
                if chat_completion.usage is not None:
                    trajectory.metrics["tokens/prompt"] = (
                        trajectory.metrics.get("tokens/prompt", 0.0)
                        + chat_completion.usage.prompt_tokens
                    )
                    trajectory.metrics["tokens/completion"] = (
                        trajectory.metrics.get("tokens/completion", 0.0)
                        + chat_completion.usage.completion_tokens
                    )
                choice = chat_completion.choices[0]
                messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        choice.message.model_dump(exclude_none=True),
                    )
                )
                tool_calls = getattr(choice.message, "tool_calls", None)
                if tool_calls:
                    for tool_call in tool_calls:
                        action = _tool_call_action(tool_call)
                        environment_started = time.perf_counter()
                        step = await client.step_environment(env.id, action)
                        environment_latency = time.perf_counter() - environment_started
                        trajectory.metrics["latency/environment"] = (
                            trajectory.metrics.get("latency/environment", 0.0)
                            + environment_latency
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "content": step.observation.removeprefix("tool: "),
                                "tool_call_id": tool_call.id,
                            }
                        )
                        trajectory.reward += step.reward
                        terminated = step.terminated
                else:
                    environment_started = time.perf_counter()
                    step = await client.step_environment(
                        env.id,
                        choice.message.content or "",
                    )
                    environment_latency = time.perf_counter() - environment_started
                    trajectory.metrics["latency/environment"] = (
                        trajectory.metrics.get("latency/environment", 0.0)
                        + environment_latency
                    )
                    if "user_message_cost" in step.info:
                        trajectory.metrics["cost/user"] += step.info[
                            "user_message_cost"
                        ]
                    elif assert_costs:
                        raise ValueError("Costs are not supported for the user model")
                    messages.append(
                        {
                            "role": "user",
                            "content": step.observation.removeprefix("user: "),
                        }
                    )
                    trajectory.reward += step.reward
                    terminated = step.terminated
                num_turns += 1
                usage = chat_completion.usage
                if usage is not None and _would_exceed_context_limit(
                    usage.total_tokens,
                    _requested_completion_tokens(chat_completion_kwargs),
                ):
                    break
        trajectory.metrics["num_turns"] = num_turns
        trajectory.metrics["latency/environment_startup"] = environment_startup
        trajectory.metrics["latency/active"] = time.perf_counter() - started
        return trajectory


async def _create_policy_completion(
    openai_client: AsyncOpenAI,
    *,
    stream_policy: bool,
    **kwargs: Any,
) -> ChatCompletion:
    attempts = max(1, int(getattr(openai_client, "max_retries", 2)) + 1)
    for attempt in range(attempts):
        completion = await openai_client.chat.completions.create(**kwargs)
        if not stream_policy:
            return cast(ChatCompletion, completion)
        try:
            return await consume_chat_completion_stream(
                cast(AsyncStream[ChatCompletionChunk], completion),
                require_usage=True,
            )
        except Exception as error:
            if attempt == attempts - 1 or not _retryable_stream_error(error):
                raise
            delay = 0.25 * (2**attempt)
            logging.warning(
                "Retrying streamed policy completion after %s",
                type(error).__name__,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def _retryable_stream_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            IncompleteChatCompletionStreamError,
            APIConnectionError,
            APITimeoutError,
            httpx.TransportError,
            json.JSONDecodeError,
        ),
    ):
        return True
    if not isinstance(error, APIError):
        return False
    body = getattr(error, "body", None)
    code = body.get("code") if isinstance(body, Mapping) else None
    if not isinstance(code, (int, str)):
        return False
    try:
        return int(code) in _STREAM_RETRY_STATUS_CODES
    except (TypeError, ValueError):
        return False


def _completion_client_and_model(
    base_url_or_model: str | Model,
    *,
    api_key: str | None,
    model: str | None,
    base_model: str | None,
) -> tuple[AsyncOpenAI, str, str | None]:
    if isinstance(base_url_or_model, Model):
        art_model = base_url_or_model
        return (
            art_model.openai_client(),
            art_model.get_inference_name(),
            getattr(art_model, "base_model", None),
        )
    if api_key is None or model is None:
        raise TypeError("base_url, api_key, and model are required for string rollouts")
    key = (base_url_or_model, api_key)
    if key not in openai_clients:
        openai_clients[key] = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url_or_model,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_connections=_POLICY_CONNECTION_LIMIT,
                    max_keepalive_connections=_POLICY_CONNECTION_LIMIT,
                )
            ),
        )
    return openai_clients[key], model, base_model


def _tool_call_action(tool_call: Any) -> str:
    arguments = json.loads(tool_call.function.arguments)
    args_str = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{tool_call.function.name}({args_str})"


def _record_tinker_costs(
    trajectory: Trajectory,
    base_model: str | None,
    usage: CompletionUsage | None,
    *,
    assert_costs: bool,
) -> None:
    if usage is None:
        if assert_costs:
            raise ValueError("Costs are not supported for this model")
        return
    pricing = get_model_pricing(base_model)
    if pricing is None:
        if assert_costs:
            raise ValueError("Costs are not supported for this model")
        return
    trajectory.metrics["cost/tinker/prefill"] += tokens_to_cost(
        usage.prompt_tokens,
        pricing.prefill,
    )
    trajectory.metrics["cost/tinker/sample"] += tokens_to_cost(
        usage.completion_tokens,
        pricing.sample,
    )


def _is_max_tokens_error(exc: APIError) -> bool:
    message = getattr(exc, "message", str(exc))
    return "max_tokens" in message or "max_completion_tokens" in message


def _would_exceed_context_limit(
    total_tokens: int,
    requested_completion_tokens: int,
) -> bool:
    return total_tokens + requested_completion_tokens > CONTEXT_TOKEN_LIMIT


def _requested_completion_tokens(
    chat_completion_kwargs: Mapping[str, object],
) -> int:
    requested_completion_tokens = (
        chat_completion_kwargs.get("max_tokens")
        or chat_completion_kwargs.get("max_completion_tokens")
        or DEFAULT_MAX_COMPLETION_TOKENS
    )
    if not isinstance(requested_completion_tokens, int):
        raise TypeError("max_tokens and max_completion_tokens must be integers")
    return requested_completion_tokens


def default_user_llm_args(user_model_name: str) -> dict[str, Any]:
    args: dict[str, Any] = {"temperature": 0.0}
    normalized = user_model_name.lower()

    api_key_env: str | None = None
    if normalized.startswith("openrouter/"):
        api_key_env = "OPENROUTER_API_KEY"
    elif normalized.startswith(("openai/", "gpt-")):
        api_key_env = "OPENAI_API_KEY"
    elif normalized.startswith(("anthropic/", "claude")):
        api_key_env = "ANTHROPIC_API_KEY"
    elif normalized.startswith(("gemini/", "google/")):
        api_key_env = (
            "GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "GOOGLE_API_KEY"
        )

    if api_key_env and (api_key := os.getenv(api_key_env)):
        args["api_key"] = api_key
    return args
