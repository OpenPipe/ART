"""Yes-no-maybe metrics demo for the Megatron backend."""

from __future__ import annotations

import asyncio
from itertools import permutations
import json
import os
import time

from dotenv import load_dotenv
import openai

import art
from art.megatron import MegatronBackend


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


def _get_env_int_list(name: str, default: list[int] | None = None) -> list[int] | None:
    value = os.environ.get(name)
    if value is None:
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid GPU ID list for {name}: {value!r}")
    return [int(part) for part in parts]


def _chat_completion_extra_body(base_model: str) -> dict[str, object] | None:
    if base_model.startswith("Qwen/Qwen3"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def with_quotes(word: str) -> str:
    return f"'{word}'"


def build_prompts() -> list[str]:
    prompts: list[str] = []
    for prefix in ["respond", "just respond"]:
        for use_quotes in [True, False]:
            for length in [3, 2]:
                for words in permutations(["yes", "no", "maybe"], length):
                    rendered_words = (
                        [with_quotes(word) for word in words]
                        if use_quotes
                        else list(words)
                    )
                    if length == 3:
                        suffix = ", ".join(rendered_words)
                    else:
                        suffix = f"{rendered_words[0]} or {rendered_words[1]}"
                    prompts.append(f"{prefix} with {suffix}")
    return prompts


def first_word(content: str | None) -> str:
    if not content:
        return ""
    words = content.strip().lower().split(maxsplit=1)
    if not words:
        return ""
    return words[0].strip(".,!?:;\"'()[]{}")


def reward_for_answer(answer: str) -> float:
    if answer == "yes":
        return 0.5
    if answer == "no":
        return 0.75
    if answer == "maybe":
        return 1.0
    return 0.0


def summarize(groups: list[art.TrajectoryGroup]) -> dict[str, float]:
    trajectories = [trajectory for group in groups for trajectory in group.trajectories]
    answers = [str(trajectory.metadata["answer"]) for trajectory in trajectories]
    rewards = [trajectory.reward for trajectory in trajectories]
    total = len(trajectories)
    assert total > 0
    return {
        "num_rollouts": float(total),
        "avg_reward": sum(rewards) / total,
        "yes_rate": answers.count("yes") / total,
        "no_rate": answers.count("no") / total,
        "maybe_rate": answers.count("maybe") / total,
        "invalid_rate": sum(answer not in {"yes", "no", "maybe"} for answer in answers)
        / total,
    }


async def rollout(
    client: openai.AsyncOpenAI,
    model: art.TrainableModel,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
) -> art.Trajectory:
    messages: art.Messages = [{"role": "user", "content": prompt}]
    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=messages,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=_chat_completion_extra_body(model.base_model),
    )
    choice = completion.choices[0]
    answer = first_word(choice.message.content)
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward_for_answer(answer),
        metadata={"answer": answer},
    )


async def evaluate(
    client: openai.AsyncOpenAI,
    model: art.TrainableModel,
    prompts: list[str],
    *,
    max_tokens: int,
    timeout: float,
) -> dict[str, float]:
    groups = await art.gather_trajectory_groups(
        art.TrajectoryGroup(
            [rollout(client, model, prompt, max_tokens=max_tokens, timeout=timeout)]
        )
        for prompt in prompts
    )
    return summarize(groups)


def build_internal_config() -> art.dev.InternalModelConfig:
    trainer_gpu_ids = _get_env_int_list("TRAINER_GPU_IDS")
    inference_gpu_ids = _get_env_int_list("INFERENCE_GPU_IDS")
    rollout_weights_mode = os.environ.get("ROLLOUT_WEIGHTS_MODE")

    internal_config = art.dev.InternalModelConfig(
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=float(
                os.environ.get("GPU_MEMORY_UTILIZATION", "0.8")
            ),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "8")),
            tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
            enforce_eager=_get_env_bool("ENFORCE_EAGER"),
        ),
    )
    max_seq_length = os.environ.get("MAX_SEQ_LENGTH")
    if max_seq_length is not None:
        init_args: art.dev.InitArgs = {"max_seq_length": int(max_seq_length)}
        load_in_16bit = _get_env_bool("LOAD_IN_16BIT")
        if load_in_16bit is not None:
            init_args["load_in_16bit"] = load_in_16bit
        load_in_4bit = _get_env_bool("LOAD_IN_4BIT")
        if load_in_4bit is not None:
            init_args["load_in_4bit"] = load_in_4bit
        internal_config["init_args"] = init_args
    if trainer_gpu_ids is not None:
        assert inference_gpu_ids is not None
        internal_config["trainer_gpu_ids"] = trainer_gpu_ids
        internal_config["inference_gpu_ids"] = inference_gpu_ids
    if rollout_weights_mode is not None:
        internal_config["rollout_weights_mode"] = rollout_weights_mode
    return internal_config


async def main() -> None:
    load_dotenv()

    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    project = os.environ.get("PROJECT", "yes-no-maybe-megatron")
    model_name = os.environ.get("MODEL_NAME", f"megatron-ynm-{int(time.time())}")
    num_steps = int(os.environ.get("NUM_STEPS", "20"))
    rollouts_per_prompt = int(os.environ.get("ROLLOUTS_PER_PROMPT", "32"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
    timeout = float(os.environ.get("TIMEOUT", "100"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
    packed_sequence_length = int(
        os.environ.get(
            "PACKED_SEQUENCE_LENGTH",
            os.environ.get("MAX_SEQ_LENGTH", "4096"),
        )
    )

    backend = MegatronBackend()
    model = art.TrainableModel(
        name=model_name,
        project=project,
        base_model=base_model,
        report_metrics=[],
        _internal_config=build_internal_config(),
    )
    prompts = build_prompts()
    prompts = prompts[: int(os.environ.get("PROMPTS_LIMIT", str(len(prompts))))]
    eval_prompts = prompts[: int(os.environ.get("EVAL_PROMPTS", "24"))]

    try:
        print(json.dumps({"event": "register_start"}), flush=True)
        await model.register(backend)
        print(
            json.dumps(
                {
                    "event": "register_done",
                    "step": int(await model.get_step()),
                    "model": model.get_inference_name(),
                }
            ),
            flush=True,
        )
        client = model.openai_client()

        print(
            json.dumps({"event": "eval_start", "step": int(await model.get_step())}),
            flush=True,
        )
        initial_eval = await evaluate(
            client,
            model,
            eval_prompts,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        print(
            json.dumps(
                {
                    "event": "eval",
                    "step": int(await model.get_step()),
                    "model": model.get_inference_name(),
                    **initial_eval,
                }
            ),
            flush=True,
        )

        start_step = await model.get_step()
        for offset in range(num_steps):
            current_step = start_step + offset
            print(
                json.dumps(
                    {
                        "event": "rollout_start",
                        "step": current_step,
                        "model": model.get_inference_name(),
                    }
                ),
                flush=True,
            )
            train_groups = await art.gather_trajectory_groups(
                art.TrajectoryGroup(
                    rollout(
                        client,
                        model,
                        prompt,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )
                    for _ in range(rollouts_per_prompt)
                )
                for prompt in prompts
            )
            train_summary = summarize(train_groups)
            print(
                json.dumps(
                    {
                        "event": "train_start",
                        "step": current_step,
                        "model": model.get_inference_name(),
                        **train_summary,
                    }
                ),
                flush=True,
            )
            result = await backend.train(
                model,
                train_groups,
                learning_rate=learning_rate,
                packed_sequence_length=packed_sequence_length,
            )
            print(
                json.dumps(
                    {
                        "event": "train_step",
                        "step": result.step,
                        "model": model.get_inference_name(),
                        **train_summary,
                        "backend_metrics": result.metrics,
                    }
                ),
                flush=True,
            )

            eval_summary = await evaluate(
                client,
                model,
                eval_prompts,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "step": current_step + 1,
                        "model": model.get_inference_name(),
                        **eval_summary,
                    }
                ),
                flush=True,
            )
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
