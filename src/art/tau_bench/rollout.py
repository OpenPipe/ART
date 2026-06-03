from __future__ import annotations

import json
import os
import time
from typing import Any, overload

from openai import AsyncOpenAI

from art.costs import get_model_pricing, tokens_to_cost
from art.model import Model
from art.trajectories import Trajectory

from .client import Scenario, TauBenchClient, _get_default_client

openai_clients: dict[tuple[str, str], AsyncOpenAI] = {}


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
    client = _get_default_client(client)
    task_id = scenario.task.id
    env_create_started_at = time.perf_counter()
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
        env_create_wait_seconds = time.perf_counter() - env_create_started_at
        chat_completion_kwargs = chat_completion_kwargs or {}
        openai_client, model_name, cost_model = _completion_client_and_model(
            base_url_or_model,
            api_key=api_key,
            model=model,
            base_model=base_model,
        )
        trajectory = Trajectory(
            messages_and_choices=[
                {"role": "system", "content": env.info["policy"]},
                {"role": "user", "content": env.observation.removeprefix("user: ")},
            ],
            tools=env.info.get("tools"),
            reward=0,
            metrics={
                "cost/tinker/prefill": 0.0,
                "cost/tinker/sample": 0.0,
                "cost/user": 0.0,
                "time/policy_chat_completion": 0.0,
                "time/tau_bench/env_create_wait": env_create_wait_seconds,
                "time/tau_bench/env_step_wait": 0.0,
                "time/tau_bench/env_wait": 0.0,
            },
            metadata={"scenario_id": task_id},
        )
        _record_environment_timing(
            trajectory,
            env.info,
            env_create_wait_seconds,
            is_step=False,
        )
        terminated = False
        num_turns = 0
        while not terminated:
            if max_turns is not None and num_turns >= max_turns:
                break
            policy_started_at = time.perf_counter()
            chat_completion = await openai_client.chat.completions.create(
                messages=trajectory.messages(),
                model=model_name,
                stream=False,
                tool_choice="auto",
                tools=trajectory.tools or [],
                **chat_completion_kwargs,
            )
            trajectory.metrics["time/policy_chat_completion"] += (
                time.perf_counter() - policy_started_at
            )
            _record_tinker_costs(
                trajectory,
                cost_model,
                getattr(chat_completion, "usage", None),
                assert_costs=assert_costs,
            )
            choice = chat_completion.choices[0]
            trajectory.messages_and_choices.append(choice)
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    action = _tool_call_action(tool_call)
                    env_step_started_at = time.perf_counter()
                    step = await client.step_environment(
                        env.id,
                        action,
                        include_info=False,
                    )
                    env_step_wait_seconds = time.perf_counter() - env_step_started_at
                    _record_environment_timing(
                        trajectory,
                        step.info,
                        env_step_wait_seconds,
                        is_step=True,
                    )
                    trajectory.messages_and_choices.append(
                        {
                            "role": "tool",
                            "content": step.observation.removeprefix("tool: "),
                            "tool_call_id": tool_call.id,
                        }
                    )
                    trajectory.reward += step.reward
                    terminated = step.terminated
            else:
                env_step_started_at = time.perf_counter()
                step = await client.step_environment(
                    env.id,
                    choice.message.content or "",
                    include_info=False,
                )
                env_step_wait_seconds = time.perf_counter() - env_step_started_at
                _record_environment_timing(
                    trajectory,
                    step.info,
                    env_step_wait_seconds,
                    is_step=True,
                )
                if "user_message_cost" in step.info:
                    trajectory.metrics["cost/user"] += step.info["user_message_cost"]
                elif assert_costs:
                    raise ValueError("Costs are not supported for the user model")
                trajectory.messages_and_choices.append(
                    {"role": "user", "content": step.observation.removeprefix("user: ")}
                )
                trajectory.reward += step.reward
                terminated = step.terminated
            num_turns += 1
        trajectory.metrics["num_turns"] = num_turns
        return trajectory


def _completion_client_and_model(
    base_url_or_model: str | Model,
    *,
    api_key: str | None,
    model: str | None,
    base_model: str | None,
) -> tuple[Any, str, str | None]:
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
        openai_clients[key] = AsyncOpenAI(api_key=api_key, base_url=base_url_or_model)
    return openai_clients[key], model, base_model


def _tool_call_action(tool_call: Any) -> str:
    arguments = json.loads(tool_call.function.arguments)
    args_str = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{tool_call.function.name}({args_str})"


def _record_tinker_costs(
    trajectory: Trajectory,
    base_model: str | None,
    usage: Any,
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
        getattr(usage, "prompt_tokens", None) or 0,
        pricing.prefill,
    )
    trajectory.metrics["cost/tinker/sample"] += tokens_to_cost(
        getattr(usage, "completion_tokens", None) or 0,
        pricing.sample,
    )


def _record_environment_timing(
    trajectory: Trajectory,
    info: dict[str, Any],
    client_wait_seconds: float,
    *,
    is_step: bool,
) -> None:
    trajectory.metrics["time/tau_bench/env_wait"] += client_wait_seconds
    if is_step:
        trajectory.metrics["time/tau_bench/env_step_wait"] += client_wait_seconds
    timing = info.get("timing")
    if not isinstance(timing, dict):
        return
    for key, value in timing.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        metric_key = f"time/tau_bench/{key}"
        trajectory.metrics[metric_key] = trajectory.metrics.get(
            metric_key, 0.0
        ) + float(value)


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
