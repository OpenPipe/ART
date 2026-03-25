"""Yes-no-maybe metrics demo for the LocalBackend `model.train()` path.

This keeps the same prompt family, rollout structure, and reward ordering as
`dev/yes-no-maybe.py` while adding explicit metrics taxonomy instrumentation for
actor/eval timing and data metrics, while relying on LocalBackend for automatic
step wall time and GPU cost logging.
"""

from __future__ import annotations

import asyncio
from itertools import permutations
import os
import re
import time

from dotenv import load_dotenv
import openai

try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import art
from art.local import LocalBackend


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


def with_quotes(word: str) -> str:
    return f"'{word}'"


def build_prompts() -> list[str]:
    prompts: list[str] = []
    for prefix in ["respond", "just respond"]:
        for use_quotes in [True, False]:
            for length in [3, 2]:
                for words in permutations(["yes", "no", "maybe"], length):
                    if use_quotes:
                        rendered_words = [with_quotes(word) for word in words]
                    else:
                        rendered_words = list(words)
                    if length == 3:
                        suffix = ", ".join(rendered_words)
                    else:
                        suffix = f"{rendered_words[0]} or {rendered_words[1]}"
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
    prompt_tokens = int(usage.prompt_tokens or 0)
    completion_tokens = int(usage.completion_tokens or 0)
    return prompt_tokens + completion_tokens


def total_actor_tokens(groups: list[art.TrajectoryGroup]) -> int:
    return sum(
        int(trajectory.metadata.get("actor_total_tokens", 0) or 0)
        for group in groups
        for trajectory in group.trajectories
    )


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
    content = choice.message.content
    answer = first_word_for_answer(content)
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


async def evaluate(
    client: openai.AsyncOpenAI,
    model: art.TrainableModel,
    prompts: list[str],
    *,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool,
) -> list[art.TrajectoryGroup]:
    groups = await art.gather_trajectory_groups(
        art.TrajectoryGroup(
            [
                rollout(
                    client,
                    model,
                    prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    enable_thinking=enable_thinking,
                )
            ],
            metadata={"scenario_id": scenario_id_for_prompt(prompt)},
        )
        for prompt in prompts
    )
    return groups


def print_history_summary(model: art.TrainableModel) -> None:
    history_path = (
        model.base_path + f"/{model.project}/models/{model.name}/history.jsonl"
    )
    print(f"History: {history_path}")


def build_internal_config() -> art.dev.InternalModelConfig:
    init_args: art.dev.InitArgs = {
        "max_seq_length": int(os.environ.get("MAX_SEQ_LENGTH", "4096"))
    }

    load_in_4bit = _get_env_bool("LOAD_IN_4BIT")
    if load_in_4bit is not None:
        init_args["load_in_4bit"] = load_in_4bit

    load_in_16bit = _get_env_bool("LOAD_IN_16BIT")
    if load_in_16bit is not None:
        init_args["load_in_16bit"] = load_in_16bit

    result = art.dev.InternalModelConfig(
        engine_args=art.dev.EngineArgs(
            gpu_memory_utilization=float(
                os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
            ),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "8")),
            enforce_eager=_get_env_bool("ENFORCE_EAGER", True),
        ),
        init_args=init_args,
    )

    trainer_gpu_ids = _get_env_int_list("TRAINER_GPU_IDS")
    inference_gpu_ids = _get_env_int_list("INFERENCE_GPU_IDS")
    assert (trainer_gpu_ids is None) == (inference_gpu_ids is None), (
        "TRAINER_GPU_IDS and INFERENCE_GPU_IDS must both be set or both unset"
    )
    if trainer_gpu_ids is not None:
        result["trainer_gpu_ids"] = trainer_gpu_ids
        result["inference_gpu_ids"] = inference_gpu_ids

    rollout_weights_mode = os.environ.get("ROLLOUT_WEIGHTS_MODE")
    if rollout_weights_mode is not None:
        if rollout_weights_mode not in {"lora", "merged"}:
            raise ValueError("ROLLOUT_WEIGHTS_MODE must be either 'lora' or 'merged'")
        result["rollout_weights_mode"] = rollout_weights_mode

    return result


async def main() -> None:
    load_dotenv()

    backend = LocalBackend()
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
    project = os.environ.get("PROJECT", "yes-no-maybe-metrics")
    model = art.TrainableModel(
        name=os.environ.get("MODEL_NAME", f"yes-no-maybe-metrics-{int(time.time())}"),
        project=project,
        base_model=base_model,
        report_metrics=["wandb"],
        _internal_config=build_internal_config(),
    )
    try:
        await model.register(backend)
        if wandb_run := model._get_wandb_run():
            print(f"W&B run: {wandb_run.url}")

        prompts = build_prompts()
        eval_prompts = prompts[: int(os.environ.get("EVAL_PROMPTS", "12"))]
        openai_client = model.openai_client()
        max_steps = int(os.environ.get("NUM_STEPS", "20"))
        rollouts_per_prompt = int(os.environ.get("ROLLOUTS_PER_PROMPT", "32"))
        max_tokens = int(os.environ.get("MAX_TOKENS", "100"))
        timeout = float(os.environ.get("TIMEOUT", "100"))
        eval_every_n_steps = int(os.environ.get("EVAL_EVERY_N_STEPS", "1"))
        learning_rate = float(os.environ.get("LEARNING_RATE", "1e-4"))
        enable_thinking = _get_env_bool("ENABLE_THINKING", False)

        start_step = await model.get_step()
        for offset in range(max_steps):
            current_step = start_step + offset

            if (
                eval_every_n_steps > 0
                and (current_step - start_step) % eval_every_n_steps == 0
            ):
                eval_builder = model.metrics_builder("eval")
                with eval_builder.activate_context():
                    with eval_builder.measure("time/step_eval_s"):
                        val_groups = await evaluate(
                            openai_client,
                            model,
                            eval_prompts,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            enable_thinking=enable_thinking,
                        )
                    eval_builder.add_data(
                        step_actor_tokens=total_actor_tokens(val_groups)
                    )
                await model.log(val_groups, split="val", step=current_step)

            train_builder = model.metrics_builder("train")
            with train_builder.activate_context():
                with train_builder.measure("time/step_actor_s"):
                    train_groups = await art.gather_trajectory_groups(
                        (
                            art.TrajectoryGroup(
                                rollout(
                                    openai_client,
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
                train_builder.add_data(
                    step_actor_tokens=total_actor_tokens(train_groups)
                )
                result = await backend.train(
                    model,
                    train_groups,
                    learning_rate=learning_rate,
                )

            await model.log(
                split="train",
                step=result.step,
                trajectories=train_groups,
                metrics=result.metrics,
            )
            print(f"step {result.step} complete")

        print_history_summary(model)
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
