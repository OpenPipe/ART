from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, cast
import uuid

from dotenv import load_dotenv
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import polars as pl

import art
from art.tinker_native import TinkerNativeBackend

from . import PipelineTrainer, make_group_rollout_fn

Scenario = dict[str, Any]


@dataclass
class PipelineConfig:
    temperature: float
    eval_temperature: float
    max_tokens: int


TOOL_NAME = "make_guess"
SECRET_BITS = "110000110100101011111010101011"
SECRET_LEN = len(SECRET_BITS)

TOOLS: list[ChatCompletionToolParam] = [
    cast(
        ChatCompletionToolParam,
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": "Submit a binary guess for the secret string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "guess": {
                            "type": "string",
                            "description": (
                                "A binary string of length "
                                f"{SECRET_LEN} consisting of 0 and 1."
                            ),
                        }
                    },
                    "required": ["guess"],
                    "additionalProperties": False,
                },
            },
        },
    )
]
TOOL_CHOICE: ChatCompletionToolChoiceOptionParam = {
    "type": "function",
    "function": {"name": TOOL_NAME},
}

SYSTEM_PROMPT = (
    "You are playing a prefix-guessing game. You must call the tool "
    f"{TOOL_NAME} exactly once with your best guess. The argument must be a "
    f"{SECRET_LEN}-character string of only 0 and 1. Do not output any other text. "
    "Your reward is the length of the shared prefix with the secret string."
)
USER_PROMPT = "Call the tool with your best binary guess."


def is_valid_guess(guess: str) -> bool:
    return all(ch in {"0", "1"} for ch in guess)


def shared_prefix_len(guess: str, secret: str) -> int:
    matched = 0
    for guessed, actual in zip(guess, secret):
        if guessed != actual:
            break
        matched += 1
    return matched


def _parse_guess_args(arguments: str | None) -> str | None:
    if not arguments:
        return None
    text = arguments.strip()
    if not text:
        return None
    payload: Any | None = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        guess = payload.get("guess")
        if isinstance(guess, str):
            return guess
        if guess is not None:
            return str(guess)

    match = re.search(r'guess\s*[:=]\s*"([^"]*)"', text)
    if match:
        return match.group(1)
    match = re.search(r"guess\s*[:=]\s*'([^']*)'", text)
    if match:
        return match.group(1)
    if re.fullmatch(r"[01\s]+", text):
        return text
    return None


def _tool_name_and_args(tool_call: Any) -> tuple[str | None, str | None]:
    if hasattr(tool_call, "function"):
        function = tool_call.function
        return getattr(function, "name", None), getattr(function, "arguments", None)
    if isinstance(tool_call, dict):
        func = tool_call.get("function") or {}
        return func.get("name"), func.get("arguments")
    return None, None


def extract_guess(choice: Any) -> tuple[str | None, str]:
    tool_calls = getattr(choice.message, "tool_calls", None) or []
    for tool_call in tool_calls:
        name, args = _tool_name_and_args(tool_call)
        if name != TOOL_NAME:
            continue
        guess = _parse_guess_args(args)
        if guess is not None:
            return guess, "tool_call"

    return None, "missing"


def get_model_output_dir(model: art.TrainableModel) -> Path:
    return Path(model.base_path) / model.project / "models" / model.name


def print_history_summary(model: art.TrainableModel, tail: int = 5) -> None:
    history_path = get_model_output_dir(model) / "history.jsonl"
    if not history_path.exists():
        print(f"No history found at {history_path}")
        return

    rows = pl.read_ndjson(str(history_path)).to_dicts()

    train_rows = [row for row in rows if "train/reward" in row]
    print("\nRecent training metrics:")
    for row in train_rows[-tail:]:
        step = row["step"]
        reward = row["train/reward"]
        std_dev = row["train/reward_std_dev"]
        discarded = row["train/discarded_stale_samples"]
        off_policy = row["train/steps_off_policy"]
        print(
            f"  step={step} reward={reward} std={std_dev} "
            f"discarded={discarded} off_policy={off_policy}"
        )


async def main() -> None:
    load_dotenv()

    base_model = os.environ.get(
        "BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
    )  # Qwen/Qwen3-30B-A3B-Instruct-2507
    model_name = os.environ.get("MODEL_NAME", "pipeline-binary-prefix-tool")
    run_suffix = os.environ.get("RUN_SUFFIX") or uuid.uuid4().hex[:8]
    model_name = f"{model_name}-{run_suffix}"
    project = os.environ.get("PROJECT", "binary-prefix-tool-pipeline")
    art_path = os.environ.get("ART_PATH")

    min_batch_size = int(os.environ.get("MIN_BATCH_SIZE", "4"))
    num_rollout_workers = int(os.environ.get("NUM_ROLLOUT_WORKERS", "8"))
    rollouts_per_scenario = int(os.environ.get("ROLLOUTS_PER_SCENARIO", "8"))
    max_steps_off_policy = int(os.environ.get("MAX_STEPS_OFF_POLICY", "6"))
    max_batch_size_env = os.environ.get("MAX_BATCH_SIZE")
    max_batch_size = int(max_batch_size_env) if max_batch_size_env else None
    eval_every_n_steps = int(os.environ.get("EVAL_EVERY_N_STEPS", "2"))
    eval_step_0 = os.environ.get("EVAL_STEP_0", "1") == "1"
    max_steps = int(os.environ.get("MAX_STEPS", "10"))
    save_checkpoint = os.environ.get("SAVE_CHECKPOINT", "0") == "1"
    resume_env = os.environ.get("RESUME")
    resume = (resume_env == "1") if resume_env is not None else save_checkpoint
    if resume and not save_checkpoint:
        print("RESUME=1 but SAVE_CHECKPOINT=0; disabling resume for a clean run.")
        resume = False

    temperature = float(os.environ.get("ROLLOUT_TEMPERATURE", "1.0"))
    eval_temperature = float(os.environ.get("EVAL_TEMPERATURE", "0.0"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "300"))
    request_timeout = float(os.environ.get("REQUEST_TIMEOUT", "60"))
    log_interval_seconds = float(os.environ.get("STATUS_LOG_INTERVAL_SECONDS", "60"))

    internal_config: art.dev.InternalModelConfig | None = None
    lora_rank = os.environ.get("LORA_RANK")
    if lora_rank is not None:
        internal_config = {
            "tinker_native_args": {
                "renderer_name": os.environ.get("RENDERER_NAME", "qwen3_instruct"),
                "training_client_args": {"rank": int(lora_rank)},
            }
        }

    backend = TinkerNativeBackend(path=art_path)
    model = art.TrainableModel(
        name=model_name,
        project=project,
        base_model=base_model,
        _internal_config=internal_config,
        report_metrics=[],  # Disable wandb logging
    )
    await model.register(backend)

    openai_client = model.openai_client()

    async def do_rollout(scenario: Scenario, temp: float) -> art.Trajectory:
        """Core rollout logic used by both training and eval."""
        messages: art.Messages = scenario["messages"]
        response = await openai_client.chat.completions.create(
            messages=messages,
            model=model.name,
            max_tokens=max_tokens,
            timeout=request_timeout,
            temperature=temp,
            tools=TOOLS,
            tool_choice=TOOL_CHOICE,
        )
        choice = response.choices[0]
        raw_guess, source = extract_guess(choice)
        sampled_content = choice.message.content or ""
        guess = raw_guess or ""
        valid_guess = is_valid_guess(guess)
        prefix_len = shared_prefix_len(guess, SECRET_BITS) if valid_guess else 0
        reward = float(prefix_len)
        metrics = {
            "prefix_len": prefix_len,
            "guess_len": len(guess),
            "secret_len": SECRET_LEN,
            "valid_guess": 1.0 if valid_guess else 0.0,
            "tool_call_count": float(
                len(getattr(choice.message, "tool_calls", None) or [])
            ),
            "tool_call_found": 1.0 if source != "missing" else 0.0,
            "tool_call_structured": 1.0 if source == "tool_call" else 0.0,
        }
        return art.Trajectory(
            messages_and_choices=[*messages, choice],
            tools=TOOLS,
            reward=reward,
            logs=[f"sampled_content:\n{sampled_content}"],
            metrics=metrics,
        )

    async def single_rollout(
        _model: art.TrainableModel,
        scenario: Scenario,
        _config: PipelineConfig,
    ) -> art.Trajectory:
        return await do_rollout(scenario, temperature)

    rollout_fn = make_group_rollout_fn(single_rollout, n=rollouts_per_scenario)

    last_eval: dict[str, float | None] = {"avg_reward": None}

    async def eval_fn(
        _model: art.TrainableModel, _step: int, _config: PipelineConfig
    ) -> list[art.Trajectory]:
        tasks = [do_rollout(build_scenario(), eval_temperature)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        trajectories = [r for r in results if isinstance(r, art.Trajectory)]
        if trajectories:
            avg_reward = sum(t.reward for t in trajectories) / len(trajectories)
            last_eval["avg_reward"] = avg_reward
        return trajectories

    scenario_count = int(os.environ.get("SCENARIO_COUNT", "1000"))
    scenario_count = max(1, scenario_count)

    def build_scenario() -> Scenario:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
        }

    async def scenario_iter():
        for i in range(scenario_count):
            scenario = build_scenario()
            scenario["metadata"] = {"scenario_idx": i}
            yield scenario

    config = PipelineConfig(
        temperature=temperature,
        eval_temperature=eval_temperature,
        max_tokens=max_tokens,
    )

    trainer = PipelineTrainer(
        model=model,
        backend=backend,
        rollout_fn=rollout_fn,
        scenarios=scenario_iter(),
        config=config,
        eval_fn=eval_fn,
        num_rollout_workers=num_rollout_workers,
        min_batch_size=min_batch_size,
        max_steps_off_policy=max_steps_off_policy,
        max_batch_size=max_batch_size,
        learning_rate=float(os.environ.get("LEARNING_RATE", "1e-4")),
        log_interval_seconds=log_interval_seconds,
        eval_every_n_steps=eval_every_n_steps,
        eval_step_0=eval_step_0,
        save_checkpoint=save_checkpoint,
        resume=resume,
        max_steps=max_steps,
        total_scenarios=scenario_count,
    )

    print(
        "Starting pipeline trainer test: "
        f"max_steps={max_steps} scenarios={scenario_count} "
        f"rollouts_per_scenario={rollouts_per_scenario} "
        f"secret_len={SECRET_LEN}"
    )
    await trainer.train()
    print("Training completed.")
    if last_eval["avg_reward"] is not None:
        print(f"Last eval avg reward: {last_eval['avg_reward']:.3f}")

    print_history_summary(model, tail=5)
    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
