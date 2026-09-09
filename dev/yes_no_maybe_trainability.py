from __future__ import annotations

import asyncio
from itertools import permutations
import json
import os
from pathlib import Path
import re
import time
from typing import cast

from dotenv import load_dotenv
import openai

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import art
from art.local import LocalBackend
from art.megatron import MegatronBackend


def _disable_wandb() -> None:
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_SILENT"] = "true"
    os.environ.pop("WANDB_API_KEY", None)


def _get_env_bool(name: str, default: bool | None = None) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {value!r}")


def _get_env_int_list(name: str) -> list[int] | None:
    value = os.environ.get(name)
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid GPU ID list for {name}: {value!r}")
    return [int(part) for part in parts]


def _with_quotes(word: str) -> str:
    return f"'{word}'"


def build_prompts() -> list[str]:
    prompts: list[str] = []
    for prefix in ["respond", "just respond"]:
        for use_quotes in [True, False]:
            for length in [3, 2]:
                for words in permutations(["yes", "no", "maybe"], length):
                    rendered_words = (
                        [_with_quotes(word) for word in words]
                        if use_quotes
                        else list(words)
                    )
                    suffix = (
                        ", ".join(rendered_words)
                        if length == 3
                        else f"{rendered_words[0]} or {rendered_words[1]}"
                    )
                    prompts.append(f"{prefix} with {suffix}")
    return prompts


def reward_for_answer(answer: str) -> float:
    if answer == "yes":
        return 0.5
    if answer == "no":
        return 0.75
    if answer == "maybe":
        return 1.0
    return 0.0


def first_word_for_answer(content: str | None) -> str:
    if not content:
        return ""
    content = re.sub(
        r"<think>.*?</think>\s*",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    words = content.strip().lower().split(maxsplit=1)
    if not words:
        return ""
    return words[0].strip(".,!?:;\"'()[]{}")


def scenario_id_for_prompt(prompt: str) -> str:
    return prompt.replace(" ", "_").replace("'", "")


def response_total_tokens(
    response: openai.types.chat.chat_completion.ChatCompletion,
) -> int:
    usage = response.usage
    if usage is None:
        return 0
    return int(usage.prompt_tokens or 0) + int(usage.completion_tokens or 0)


def total_actor_tokens(groups: list[art.TrajectoryGroup]) -> int:
    return sum(
        int(trajectory.metadata.get("actor_total_tokens", 0) or 0)
        for group in groups
        for trajectory in group.trajectories
    )


def mean_reward(groups: list[art.TrajectoryGroup]) -> float:
    rewards = [
        trajectory.reward for group in groups for trajectory in group.trajectories
    ]
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


async def rollout(
    client: openai.AsyncOpenAI,
    model: art.TrainableModel,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool,
) -> art.Trajectory:
    messages: art.Messages = [{"role": "user", "content": prompt}]
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model.get_inference_name(),
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    choice = chat_completion.choices[0]
    answer = first_word_for_answer(choice.message.content)
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward_for_answer(answer),
        metadata={
            "scenario_id": scenario_id_for_prompt(prompt),
            "actor_total_tokens": response_total_tokens(chat_completion),
        },
        metrics={
            "valid_answer": answer in {"yes", "no", "maybe"},
            "answer_is_yes": answer == "yes",
            "answer_is_no": answer == "no",
            "answer_is_maybe": answer == "maybe",
        },
    )


async def gather_groups(
    client: openai.AsyncOpenAI,
    model: art.TrainableModel,
    prompts: list[str],
    *,
    rollouts_per_prompt: int,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool,
) -> list[art.TrajectoryGroup]:
    return await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                rollout(
                    client,
                    model,
                    prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    enable_thinking=enable_thinking,
                )
                for _ in range(rollouts_per_prompt)
            )
            for prompt in prompts
        )
    )


def build_internal_config() -> art.dev.InternalModelConfig:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_gpu_count = (
        len([device for device in visible_devices.split(",") if device.strip()])
        if visible_devices
        else 1
    )
    init_args: art.dev.InitArgs = {
        "max_seq_length": int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
    }
    load_in_4bit = _get_env_bool("LOAD_IN_4BIT")
    if load_in_4bit is not None:
        init_args["load_in_4bit"] = load_in_4bit
    load_in_16bit = _get_env_bool("LOAD_IN_16BIT")
    if load_in_16bit is not None:
        init_args["load_in_16bit"] = load_in_16bit

    config = art.dev.InternalModelConfig(
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=float(
                os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
            ),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "8")),
            enforce_eager=_get_env_bool("ENFORCE_EAGER", True),
            tensor_parallel_size=int(
                os.environ.get("TENSOR_PARALLEL_SIZE", str(max(1, visible_gpu_count)))
            ),
        ),
        init_args=init_args,
    )

    trainer_gpu_ids = _get_env_int_list("TRAINER_GPU_IDS")
    inference_gpu_ids = _get_env_int_list("INFERENCE_GPU_IDS")
    if (trainer_gpu_ids is None) != (inference_gpu_ids is None):
        raise ValueError(
            "TRAINER_GPU_IDS and INFERENCE_GPU_IDS must both be set or both unset"
        )
    if trainer_gpu_ids is not None and inference_gpu_ids is not None:
        config["trainer_gpu_ids"] = trainer_gpu_ids
        config["inference_gpu_ids"] = inference_gpu_ids

    rollout_weights_mode = os.environ.get("ROLLOUT_WEIGHTS_MODE")
    if rollout_weights_mode is not None:
        config["rollout_weights_mode"] = rollout_weights_mode
    return config


def make_backend(
    backend_name: str, art_path: str, *, in_process: bool
) -> LocalBackend | MegatronBackend:
    if backend_name == "local":
        return LocalBackend(path=art_path, in_process=in_process)
    if backend_name == "megatron":
        return MegatronBackend(path=art_path)
    raise ValueError(f"Unsupported BACKEND={backend_name!r}")


def output_dir_for_model(model: art.TrainableModel) -> Path:
    return Path(model.base_path) / model.project / "models" / model.name


async def main() -> None:
    load_dotenv()
    _disable_wandb()

    backend_name = os.environ.get("BACKEND", "local")
    run_id = os.environ.get("RUN_ID", str(int(time.time())))
    project = os.environ.get("PROJECT", f"yes-no-maybe-{backend_name}")
    model_name = os.environ.get("MODEL_NAME", f"{backend_name}-{run_id}")
    art_path = os.environ.get(
        "ART_PATH",
        f"/tmp/art_yes_no_maybe_trainability/{backend_name}/{run_id}",
    )
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    in_process = bool(_get_env_bool("IN_PROCESS", False))
    num_steps = int(os.environ.get("NUM_STEPS", "20"))
    rollouts_per_prompt = int(os.environ.get("ROLLOUTS_PER_PROMPT", "32"))
    eval_rollouts_per_prompt = int(os.environ.get("EVAL_ROLLOUTS_PER_PROMPT", "4"))
    eval_prompts = int(os.environ.get("EVAL_PROMPTS", "12"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
    timeout = float(os.environ.get("TIMEOUT", "100"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
    packed_sequence_length = os.environ.get("PACKED_SEQUENCE_LENGTH")
    enable_thinking = bool(_get_env_bool("ENABLE_THINKING", False))

    os.makedirs(art_path, exist_ok=True)
    backend = make_backend(backend_name, art_path, in_process=in_process)
    model = art.TrainableModel(
        run_name=model_name,
        name=model_name,
        project=project,
        base_model=base_model,
        report_metrics=[],
        _internal_config=build_internal_config(),
    )

    prompts = build_prompts()
    eval_prompt_subset = prompts[:eval_prompts]
    run_summary: dict[str, object] = {
        "backend": backend_name,
        "art_path": art_path,
        "project": project,
        "model_name": model_name,
        "base_model": base_model,
        "in_process": in_process,
        "num_steps": num_steps,
        "rollouts_per_prompt": rollouts_per_prompt,
        "eval_rollouts_per_prompt": eval_rollouts_per_prompt,
        "eval_prompts": eval_prompts,
        "max_tokens": max_tokens,
        "learning_rate": learning_rate,
        "packed_sequence_length": (
            None if packed_sequence_length is None else int(packed_sequence_length)
        ),
        "steps": [],
    }

    try:
        await model.register(backend)
        client = model.openai_client()
        start_step = await model.get_step()
        summary_path = output_dir_for_model(model) / "trainability_summary.json"

        for offset in range(num_steps):
            current_step = start_step + offset
            val_groups = await gather_groups(
                client,
                model,
                eval_prompt_subset,
                rollouts_per_prompt=eval_rollouts_per_prompt,
                max_tokens=max_tokens,
                timeout=timeout,
                enable_thinking=enable_thinking,
            )
            await model.log(val_groups, split="val", step=current_step)

            train_groups = await gather_groups(
                client,
                model,
                prompts,
                rollouts_per_prompt=rollouts_per_prompt,
                max_tokens=max_tokens,
                timeout=timeout,
                enable_thinking=enable_thinking,
            )
            train_kwargs: dict[str, object] = {"learning_rate": learning_rate}
            if packed_sequence_length is not None:
                train_kwargs["packed_sequence_length"] = int(packed_sequence_length)
            result = await backend.train(model, train_groups, **train_kwargs)
            await model.log(
                train_groups,
                split="train",
                step=result.step,
                metrics=result.metrics,
            )

            step_summary = {
                "step": result.step,
                "pre_train_val_reward": mean_reward(val_groups),
                "train_reward": mean_reward(train_groups),
                "val_actor_tokens": total_actor_tokens(val_groups),
                "train_actor_tokens": total_actor_tokens(train_groups),
                "train_metrics": result.metrics,
            }
            cast(list[dict[str, object]], run_summary["steps"]).append(step_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(run_summary, indent=2) + "\n")
            print(json.dumps(step_summary, sort_keys=True))

        print(f"SUMMARY_PATH={summary_path}")
        print(f"HISTORY_PATH={output_dir_for_model(model) / 'history.jsonl'}")
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
