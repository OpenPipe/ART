from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, AsyncIterator, Literal, cast
import uuid

from pydantic import BaseModel, Field
import pytest

import art
from art.megatron.model_support.registry import model_uses_expert_parallel
from art.pipeline_trainer import PipelineTrainer

from ..model_support.oracle_harness import Topology
from .yes_no_trainability import (
    _backend_context,
    _build_internal_config,
    _build_variant,
    _get_env_bool,
    _get_env_float,
    _get_env_int,
    _init_megatron_runtime_config,
    _list_model_ids,
)

torch = pytest.importorskip("torch")

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"
LIVE_ENV = "ART_RUN_LIVE_LENGTH_TRAINABILITY"
TRAINER_GPU_IDS_ENV = "ART_MODEL_SUPPORT_TRAINER_GPU_IDS"
INFERENCE_GPU_IDS_ENV = "ART_MODEL_SUPPORT_INFERENCE_GPU_IDS"
REPO_ROOT = Path(__file__).resolve().parents[4]
LATEST_SUMMARY_LOG_PATH = REPO_ROOT / ".local" / "length_trainability.log"
INITIAL_ABS_ERROR_MIN = 5.0
SUCCESS_ABS_ERROR_MAX = 1.5
MOE_DEDICATED_TRAINING_TOPOLOGY = Topology(
    tp=1,
    cp=2,
    ep=2,
    etp=1,
    dp=1,
    sp=False,
)
BASE_PROMPT = (
    "Write a plain answer about a quiet harbor. Use the unrelated notes below "
    "only as background texture. Use one sentence. Do not use bullets, numbering, "
    "code, or a preface."
)
FILLER_SENTENCES = (
    "The morning ledger mentioned a bicycle bell near the old customs window.",
    "A folded receipt waited beside three dull pencils and a chipped mug.",
    "Someone had drawn a small square around Thursday on the calendar.",
    "The storage room smelled faintly of rope, dust, and yesterday's rain.",
    "A green notebook listed errands that no one seemed eager to finish.",
    "The clock above the doorway ticked with a patient mechanical rhythm.",
    "Two mismatched gloves rested under the bench near the umbrella stand.",
    "A paper tag fluttered from a crate of spare brass hinges.",
    "The shop radio murmured about traffic far from the waterfront.",
    "A narrow envelope contained a map with several coffee stains.",
    "The caretaker had stacked clean towels beside a basket of loose keys.",
    "A faded poster advertised a lecture about practical knot repairs.",
    "Someone left a blue scarf draped over the back of a wooden chair.",
    "The rain gauge showed a modest line from a storm before dawn.",
    "A quiet clerk sorted stamps into a tin marked for later use.",
    "The window latch clicked softly whenever a colder breeze arrived.",
    "A jar of buttons sat near the lamp with no label attached.",
    "The floorboards held a faint shine where people usually turned left.",
    "A postcard showed a bridge, though no bridge could be seen nearby.",
    "The supply shelf included chalk, twine, soap, and several blank cards.",
    "A small toolbox waited open with every socket arranged by size.",
    "The notice board carried old schedules with careful handwritten corrections.",
    "A kettle cooled on the counter beside a plate of plain biscuits.",
    "The narrow hallway displayed framed photographs of ordinary cloudy afternoons.",
    "A stack of forms leaned against a vase holding one dry reed.",
    "The back office kept a spare lantern wrapped in brown paper.",
    "A silver whistle hung from a nail beside the maintenance checklist.",
    "The cupboard door closed unevenly unless pressed near the lower hinge.",
    "A receipt book recorded purchases of candles, nails, and black ink.",
    "The stair rail felt smooth where many hands had passed over it.",
    "A shallow drawer contained string, labels, and a forgotten measuring tape.",
    "The wall map used faded pins to mark unimportant delivery stops.",
    "A wool cap lay on a crate beside a coil of clean line.",
    "The afternoon light made the dust above the desk look almost orderly.",
    "A clipboard noted that the north window should be painted soon.",
    "The brass hook near the door held only an empty canvas bag.",
    "A stack of newspapers waited under a stone used as a weight.",
    "The broom leaned in a corner beside a cardboard box of washers.",
    "A shallow bowl held wrapped peppermints for visitors who rarely arrived.",
    "The gray filing cabinet opened with a scrape and a small sigh.",
    "A pencil sharpener was screwed to the wall beside a crooked shelf.",
    "The old ledger contained careful columns and very little useful drama.",
    "A canvas cover protected the spare chair from dust and sunlight.",
    "The side table held a ruler, a thimble, and a sealed jar.",
    "A neat row of jars preserved screws sorted by uncertain categories.",
    "The calendar showed local holidays in red and market days in blue.",
    "A small bell above the entrance moved only when the door stuck.",
    "The envelope tray was empty except for a note about lamp oil.",
    "The desk drawer included a spare button and two brittle rubber bands.",
    "A plain brown box carried the words archive later in pencil.",
)


class LengthScenario(BaseModel):
    scenario_index: int
    target_step: int
    target_tokens: int
    max_tokens: int
    prompt: str
    prompt_word_count: int
    metadata: dict[str, int | float | str | None] = Field(default_factory=dict)


class LengthSampleReport(BaseModel):
    split: Literal["train"]
    step: int | None
    scenario_index: int
    target_step: int
    target_tokens: int
    max_tokens: int
    prompt_word_count: int
    generated_tokens: int
    abs_error: int
    reward: float
    text: str


class LengthTrainabilityReport(BaseModel):
    base_model: str
    max_steps: int
    max_steps_off_policy: int
    latest_step: int
    variant_name: str
    trainer_gpu_ids: list[int]
    inference_gpu_ids: list[int]
    training_topology: dict[str, int | bool]
    rollout_weights_mode: str
    rollouts_per_prompt: int
    normalize_advantages: bool
    summary_log_path: str
    latest_summary_log_path: str
    initial_train_abs_error: float | None
    best_train_abs_error: float | None
    success_step: int | None
    final_train_reward: float | None
    final_train_abs_error: float | None
    model_ids_after: list[str]
    samples: list[LengthSampleReport]


def _require_opt_in() -> None:
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(f"set {LIVE_ENV}=1 to run live length trainability")


def _base_model() -> str:
    return os.environ.get(
        "ART_LIVE_LENGTH_BASE_MODEL",
        os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
    )


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "_").replace("-", "_")


def _artifact_dir(base_model: str) -> Path:
    path = (
        REPO_ROOT
        / ".local"
        / "model_support_validation"
        / _slugify(base_model)
        / "length_trainability"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _word_count(text: str) -> int:
    return len(text.split())


def _target_tokens() -> int:
    return _get_env_int("ART_MODEL_SUPPORT_LENGTH_TARGET_TOKENS", 10)


def _use_default_moe_dedicated_placement(variant: Any, *, base_model: str) -> None:
    if not model_uses_expert_parallel(base_model, allow_unvalidated_arch=True):
        return
    if os.environ.get(TRAINER_GPU_IDS_ENV) or os.environ.get(INFERENCE_GPU_IDS_ENV):
        return
    if torch.cuda.device_count() < 3:
        pytest.skip(
            "Need at least 3 visible CUDA GPUs for default dedicated MoE length "
            "trainability: 2 trainer GPUs and 1 inference GPU."
        )
    variant.trainer_gpu_ids = [0, 1]
    variant.inference_gpu_ids = [2]
    variant.topology = MOE_DEDICATED_TRAINING_TOPOLOGY


def _check_prompt_hides_target(prompt: str) -> None:
    lowered = prompt.lower()
    leaked = [
        phrase
        for phrase in ("generated tokens", "target tokens", "target length", "exactly")
        if phrase in lowered
    ]
    if leaked:
        raise RuntimeError(f"Length prompt leaks target wording: {leaked}")


def _prompt_for_index(index: int) -> tuple[str, int]:
    target_words = _get_env_int("ART_MODEL_SUPPORT_LENGTH_PROMPT_WORDS", 300)
    rng = random.Random(index)
    sentences = list(FILLER_SENTENCES)
    rng.shuffle(sentences)
    selected: list[str] = []
    prompt = BASE_PROMPT
    for sentence in sentences:
        if _word_count(prompt) >= target_words:
            break
        selected.append(sentence)
        prompt = f"{BASE_PROMPT}\n\nNotes: {' '.join(selected)}"
    _check_prompt_hides_target(prompt)
    return prompt, _word_count(prompt)


def _scenario(index: int, *, target_step: int | None = None) -> LengthScenario:
    target_tokens = _target_tokens()
    max_tokens = max(
        target_tokens + 1,
        math.ceil(
            target_tokens
            * _get_env_float("ART_MODEL_SUPPORT_LENGTH_MAX_TOKENS_MULTIPLIER", 1.4)
        )
        + 128,
    )
    prompt, prompt_word_count = _prompt_for_index(index)
    return LengthScenario(
        scenario_index=index,
        target_step=index if target_step is None else target_step,
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        prompt=prompt,
        prompt_word_count=prompt_word_count,
        metadata={
            "scenario_index": index,
            "target_step": index if target_step is None else target_step,
            "target_tokens": target_tokens,
            "max_tokens": max_tokens,
            "prompt_word_count": prompt_word_count,
        },
    )


def _step_from_model_name(model_name: str) -> int | None:
    if "@" not in model_name:
        return None
    try:
        return int(model_name.rsplit("@", 1)[1])
    except ValueError:
        return None


def _scenario_for_training_step(
    scenario: LengthScenario | dict[str, object],
    step: int,
) -> LengthScenario:
    parsed = LengthScenario.model_validate(scenario)
    return _scenario(parsed.scenario_index, target_step=step)


def _messages(scenario: LengthScenario) -> art.Messages:
    return [{"role": "user", "content": scenario.prompt}]


def _extra_body() -> dict[str, object]:
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _generated_token_count(choice: object) -> int:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None)
    if content is not None:
        return len(content)
    message = getattr(choice, "message", None)
    return len((getattr(message, "content", "") or "").split())


def _reward(generated_tokens: int, target_tokens: int) -> float:
    # Do not clamp: early generations can be far from target, and CISPO still
    # needs within-group reward differences to produce trainable advantages.
    return -abs(generated_tokens - target_tokens) / max(1, target_tokens)


def _sample_report(
    *,
    split: Literal["train"],
    step: int | None,
    scenario: LengthScenario,
    choice: object,
) -> LengthSampleReport:
    generated_tokens = _generated_token_count(choice)
    message = getattr(choice, "message", None)
    text = getattr(message, "content", "") or ""
    return LengthSampleReport(
        split=split,
        step=step,
        scenario_index=scenario.scenario_index,
        target_step=scenario.target_step,
        target_tokens=scenario.target_tokens,
        max_tokens=scenario.max_tokens,
        prompt_word_count=scenario.prompt_word_count,
        generated_tokens=generated_tokens,
        abs_error=abs(generated_tokens - scenario.target_tokens),
        reward=_reward(generated_tokens, scenario.target_tokens),
        text=text,
    )


async def _length_group(
    model: art.TrainableModel,
    *,
    scenario: LengthScenario,
    model_name: str,
    split: Literal["train"],
    step: int | None,
    n: int,
    temperature: float,
    samples: list[LengthSampleReport],
    summary_log_path: Path | None = None,
) -> art.TrajectoryGroup:
    messages = _messages(scenario)
    completion = await model.openai_client().chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=scenario.max_tokens,
        n=n,
        temperature=temperature,
        extra_body=_extra_body(),
        logprobs=True,
        top_logprobs=0,
        timeout=_get_env_float("ART_MODEL_SUPPORT_LENGTH_REQUEST_TIMEOUT", 900.0),
    )
    trajectories: list[art.Trajectory] = []
    for choice in completion.choices:
        report = _sample_report(
            split=split,
            step=step,
            scenario=scenario,
            choice=choice,
        )
        samples.append(report)
        trajectories.append(
            art.Trajectory(
                messages_and_choices=[*messages, choice],
                reward=report.reward,
                metrics={
                    "length/generated_tokens": report.generated_tokens,
                    "length/target_tokens": scenario.target_tokens,
                    "length/max_tokens": scenario.max_tokens,
                    "length/prompt_word_count": scenario.prompt_word_count,
                    "length/abs_error": report.abs_error,
                },
                metadata=scenario.metadata,
            )
        )
    _append_step_summary(summary_log_path, samples, split=split, step=step)
    return art.TrajectoryGroup(trajectories)


def _mean_reward(samples: list[LengthSampleReport]) -> float:
    return sum(sample.reward for sample in samples) / max(1, len(samples))


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _mean_abs_error_by_step(samples: list[LengthSampleReport]) -> dict[int, float]:
    steps = sorted({sample.step for sample in samples if sample.step is not None})
    return {
        step: _mean(
            [float(sample.abs_error) for sample in samples if sample.step == step]
        )
        for step in steps
    }


def _init_summary_log(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "# length trainability summary",
                "# rows append when a rollout/eval group completes; n is cumulative for split+step",
                (
                    "split      step target max_tok prompt_w     n reward_mean "
                    "gen_mean abs_err_mean gen_min gen_max reward_min reward_max"
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _copy_latest_summary_log(path)


def _copy_latest_summary_log(path: Path) -> None:
    LATEST_SUMMARY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, LATEST_SUMMARY_LOG_PATH)


def _append_step_summary(
    path: Path | None,
    samples: list[LengthSampleReport],
    *,
    split: Literal["train"],
    step: int | None,
) -> None:
    if path is None:
        return
    matching = [
        sample for sample in samples if sample.split == split and sample.step == step
    ]
    if not matching:
        return
    generated = [float(sample.generated_tokens) for sample in matching]
    abs_errors = [float(sample.abs_error) for sample in matching]
    rewards = [sample.reward for sample in matching]
    latest = matching[-1]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{split:<9} {step if step is not None else '-':>4} "
            f"{latest.target_tokens:>6} {latest.max_tokens:>7} "
            f"{latest.prompt_word_count:>8} {len(matching):>5} "
            f"{_mean(rewards):>11.4f} {_mean(generated):>8.1f} "
            f"{_mean(abs_errors):>12.1f} {int(min(generated)):>7} "
            f"{int(max(generated)):>7} {min(rewards):>10.4f} "
            f"{max(rewards):>10.4f}\n"
        )
    _copy_latest_summary_log(path)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="Need at least 3 CUDA GPUs for live dedicated length trainability",
)
@pytest.mark.asyncio
async def test_megatron_dedicated_length_trainability_live(artifact_dir: Path) -> None:
    _require_opt_in()
    report = await run_length_trainability_async(
        base_model=_base_model(),
        artifact_dir=artifact_dir,
        allow_unvalidated_arch=True,
    )
    assert_length_trainability_passed(report)


async def run_length_trainability_async(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    artifact_dir: Path | None = None,
    allow_unvalidated_arch: bool = False,
) -> LengthTrainabilityReport:
    artifact_dir = artifact_dir or _artifact_dir(base_model)
    variant = _build_variant(
        "megatron_dedicated",
        base_model=base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    _use_default_moe_dedicated_placement(variant, base_model=base_model)
    max_steps = _get_env_int("ART_MODEL_SUPPORT_LENGTH_MAX_STEPS", 10)
    max_steps_off_policy = _get_env_int(
        "ART_MODEL_SUPPORT_LENGTH_MAX_STEPS_OFF_POLICY",
        0,
    )
    rollouts_per_prompt = _get_env_int(
        "ART_MODEL_SUPPORT_LENGTH_ROLLOUTS_PER_PROMPT",
        4,
    )
    normalize_advantages = _get_env_bool(
        "ART_MODEL_SUPPORT_LENGTH_NORMALIZE_ADVANTAGES",
        True,
    )
    rollout_workers = _get_env_int(
        "ART_MODEL_SUPPORT_LENGTH_ROLLOUT_WORKERS",
        max(1, max_steps_off_policy + 1),
    )
    scenario_count = _get_env_int(
        "ART_MODEL_SUPPORT_LENGTH_SCENARIOS",
        max_steps * max(rollouts_per_prompt, 2) + rollout_workers + 4,
    )
    success_hit = False
    samples: list[LengthSampleReport] = []
    backend_root = artifact_dir / "megatron_dedicated_workspace"
    summary_log_path = artifact_dir / "length_trainability.log"
    _init_summary_log(summary_log_path)
    internal_config = _build_internal_config(
        variant,
        base_model=base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    internal_config["engine_args"]["max_model_len"] = _get_env_int(
        "ART_MODEL_SUPPORT_LENGTH_MAX_MODEL_LEN",
        1024,
    )
    internal_config["engine_args"]["max_num_seqs"] = _get_env_int(
        "ART_MODEL_SUPPORT_LENGTH_MAX_NUM_SEQS",
        4,
    )
    rollout_weights_mode = internal_config["rollout_weights_mode"]
    _init_megatron_runtime_config(variant)

    async with _backend_context(variant, backend_root=backend_root) as backend:
        model = art.TrainableModel(
            name=f"length-{uuid.uuid4().hex[:8]}",
            project="integration-tests",
            base_model=base_model,
            _internal_config=internal_config,
            report_metrics=[],
        )
        await model.register(backend)

        async def scenarios() -> AsyncIterator[dict[str, object]]:
            for index in range(scenario_count):
                if success_hit:
                    break
                yield _scenario(index, target_step=0).model_dump()

        async def rollout_fn(
            rollout_model: art.TrainableModel,
            scenario: dict[str, object],
            _config: None,
        ) -> art.TrajectoryGroup:
            nonlocal success_hit
            model_name = rollout_model.get_inference_name()
            target_step = _step_from_model_name(model_name)
            if target_step is None:
                target_step = await rollout_model.get_step()
            group = await _length_group(
                rollout_model,
                scenario=_scenario_for_training_step(scenario, target_step),
                model_name=model_name,
                split="train",
                step=target_step,
                n=rollouts_per_prompt,
                temperature=_get_env_float(
                    "ART_MODEL_SUPPORT_LENGTH_ROLLOUT_TEMPERATURE",
                    1.1,
                ),
                samples=samples,
                summary_log_path=summary_log_path,
            )
            if (
                _mean_abs_error_by_step(
                    [sample for sample in samples if sample.split == "train"]
                )[target_step]
                <= SUCCESS_ABS_ERROR_MAX
            ):
                success_hit = True
            return group

        trainer = PipelineTrainer(
            model=model,
            backend=backend,
            rollout_fn=rollout_fn,
            scenarios=scenarios(),
            config=None,
            num_rollout_workers=rollout_workers,
            min_batch_size=1,
            max_batch_size=1,
            max_steps_off_policy=max_steps_off_policy,
            learning_rate=_get_env_float(
                "ART_MODEL_SUPPORT_LENGTH_LEARNING_RATE",
                1e-4,
            ),
            loss_fn="cispo",
            normalize_advantages=normalize_advantages,
            max_steps=max_steps,
            eval_every_n_steps=0,
            eval_at_start=False,
            save_checkpoint=False,
            total_scenarios=scenario_count,
            log_interval_seconds=30.0,
            discard_queue_multiplier=1000,
            resume=False,
        )
        await trainer.train(handle_signals=False)

        latest_step = await model.get_step()
        model_ids_after = await _list_model_ids(model)

    train_samples = [sample for sample in samples if sample.split == "train"]
    train_rewards_by_step = {
        step: [sample.reward for sample in train_samples if sample.step == step]
        for step in {sample.step for sample in train_samples}
    }
    train_abs_error_by_step = _mean_abs_error_by_step(train_samples)
    initial_train_abs_error = train_abs_error_by_step.get(0)
    best_train_abs_error = (
        min(train_abs_error_by_step.values()) if train_abs_error_by_step else None
    )
    success_step = next(
        (
            step
            for step, abs_error in train_abs_error_by_step.items()
            if abs_error <= SUCCESS_ABS_ERROR_MAX
        ),
        None,
    )
    final_train_samples = [
        sample for sample in train_samples if sample.step == latest_step - 1
    ]
    final_train_reward = (
        _mean_reward(final_train_samples) if final_train_samples else None
    )
    final_train_abs_error = (
        _mean([float(sample.abs_error) for sample in final_train_samples])
        if final_train_samples
        else None
    )
    topology = cast(Topology, variant.topology)
    report = LengthTrainabilityReport(
        base_model=base_model,
        max_steps=max_steps,
        max_steps_off_policy=max_steps_off_policy,
        latest_step=latest_step,
        variant_name=variant.name,
        trainer_gpu_ids=variant.trainer_gpu_ids,
        inference_gpu_ids=variant.inference_gpu_ids,
        training_topology=cast(dict[str, int | bool], topology.model_dump()),
        rollout_weights_mode=rollout_weights_mode,
        rollouts_per_prompt=rollouts_per_prompt,
        normalize_advantages=normalize_advantages,
        summary_log_path=str(summary_log_path),
        latest_summary_log_path=str(LATEST_SUMMARY_LOG_PATH),
        initial_train_abs_error=initial_train_abs_error,
        best_train_abs_error=best_train_abs_error,
        success_step=success_step,
        final_train_reward=final_train_reward,
        final_train_abs_error=final_train_abs_error,
        model_ids_after=model_ids_after,
        samples=samples,
    )
    (artifact_dir / "length_trainability.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_length_trainability(
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    allow_unvalidated_arch: bool = False,
) -> LengthTrainabilityReport:
    return asyncio.run(
        run_length_trainability_async(
            base_model=base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
    )


def length_trainability_passed(report: LengthTrainabilityReport) -> bool:
    train_samples = [sample for sample in report.samples if sample.split == "train"]
    train_rewards_by_step = {
        step: [sample.reward for sample in train_samples if sample.step == step]
        for step in {sample.step for sample in train_samples}
    }
    return (
        bool(train_samples)
        and report.latest_step <= report.max_steps
        and report.initial_train_abs_error is not None
        and report.initial_train_abs_error >= INITIAL_ABS_ERROR_MIN
        and report.best_train_abs_error is not None
        and report.best_train_abs_error <= SUCCESS_ABS_ERROR_MAX
        and report.success_step is not None
        and len(train_rewards_by_step) <= report.max_steps
        and all(sample.max_tokens > sample.target_tokens for sample in train_samples)
        and any(sample.generated_tokens < sample.max_tokens for sample in train_samples)
        and any(len(set(rewards)) > 1 for rewards in train_rewards_by_step.values())
        and any(
            name.endswith(f"@{report.latest_step}") for name in report.model_ids_after
        )
    )


def assert_length_trainability_passed(report: LengthTrainabilityReport) -> None:
    train_samples = [sample for sample in report.samples if sample.split == "train"]
    train_rewards_by_step = {
        step: [sample.reward for sample in train_samples if sample.step == step]
        for step in {sample.step for sample in train_samples}
    }
    assert train_samples
    assert report.latest_step <= report.max_steps
    assert report.initial_train_abs_error is not None
    assert report.initial_train_abs_error >= INITIAL_ABS_ERROR_MIN
    assert report.best_train_abs_error is not None
    assert report.best_train_abs_error <= SUCCESS_ABS_ERROR_MAX
    assert report.success_step is not None
    assert len(train_rewards_by_step) <= report.max_steps
    assert all(sample.max_tokens > sample.target_tokens for sample in train_samples)
    assert any(sample.generated_tokens < sample.max_tokens for sample in train_samples)
    assert any(len(set(rewards)) > 1 for rewards in train_rewards_by_step.values())
    assert any(
        name.endswith(f"@{report.latest_step}") for name in report.model_ids_after
    )
