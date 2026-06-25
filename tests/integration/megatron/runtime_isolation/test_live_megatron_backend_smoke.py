import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, cast
import uuid

import httpx
import pytest

import art
from art import dev
from art.megatron.backend import MegatronBackend
from art.megatron.service import MegatronService

from ..model_support.oracle_harness import ORACLE_TOPOLOGY, Topology
from ..model_support.oracle_worker import provider_topology_env
from ..trainability import (
    _build_trainable_groups,
    _build_training_groups,
    _engine_args_for_yes_no_trainability,
    _evaluate_model,
    _wandb_disabled,
    _warmup_model,
    build_prompts,
)

torch = pytest.importorskip("torch")

DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_MAX_SEQ_LENGTH = 1024
DEFAULT_PACKED_SEQUENCE_LENGTH = 1024
DEDICATED_MERGED_ENV = "ART_RUN_LIVE_MEGATRON_MERGED_SMOKE"
DEDICATED_MULTIRANK_MERGED_ENV = "ART_RUN_LIVE_MEGATRON_MULTIRANK_MERGED_SMOKE"
SHARED_LORA_ENV = "ART_RUN_LIVE_MEGATRON_SHARED_SMOKE"
SHARED_LONG_LORA_ENV = "ART_RUN_LIVE_MEGATRON_SHARED_LONG_SMOKE"
SHARED_TOPOLOGY = Topology(tp=1, ep=2, etp=1, dp=1, cp=2, sp=False)


def _base_model() -> str:
    return os.environ.get(
        "ART_LIVE_MEGATRON_BASE_MODEL",
        os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL),
    )


def _max_seq_length() -> int:
    return int(os.environ.get("ART_TEST_MAX_SEQ_LENGTH", str(DEFAULT_MAX_SEQ_LENGTH)))


def _packed_sequence_length() -> int:
    return int(
        os.environ.get(
            "ART_TEST_PACKED_SEQUENCE_LENGTH",
            str(DEFAULT_PACKED_SEQUENCE_LENGTH),
        )
    )


def _train_group_prompts() -> list[str]:
    prompt_count = int(os.environ.get("ART_TEST_MEGATRON_PROMPT_COUNT", "2"))
    return build_prompts()[: max(1, prompt_count)]


def _rollouts_per_prompt() -> int:
    return int(os.environ.get("ART_TEST_MEGATRON_ROLLOUTS_PER_PROMPT", "2"))


def _trainer_gpu_ids() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 visible CUDA GPUs for Megatron live smokes")
    return [0]


def _inference_gpu_ids() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 visible CUDA GPUs for Megatron live smokes")
    return [1]


def _multirank_trainer_gpu_ids() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError(
            "Need at least 3 visible CUDA GPUs for multi-rank Megatron merged smoke"
        )
    return [0, 1]


def _multirank_inference_gpu_ids() -> list[int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError(
            "Need at least 3 visible CUDA GPUs for multi-rank Megatron merged smoke"
        )
    return [2]


def _require_opt_in(env_name: str) -> None:
    if os.environ.get(env_name) != "1":
        pytest.skip(f"set {env_name}=1 to run this live Megatron smoke")


def _shared_live_config() -> dev.InternalModelConfig:
    return cast(
        dev.InternalModelConfig,
        {
            "rollout_weights_mode": "lora",
            "engine_args": {
                **_engine_args_for_yes_no_trainability(inference_gpu_ids=[0, 1]),
                "tensor_parallel_size": 2,
                "enable_expert_parallel": True,
                "enable_sleep_mode": True,
            },
            "init_args": {"max_seq_length": _max_seq_length()},
        },
    )


def _dedicated_merged_config() -> dev.InternalModelConfig:
    return {
        "trainer_gpu_ids": _trainer_gpu_ids(),
        "inference_gpu_ids": _inference_gpu_ids(),
        "rollout_weights_mode": "merged",
        "engine_args": {
            **_engine_args_for_yes_no_trainability(
                inference_gpu_ids=_inference_gpu_ids()
            ),
        },
        "init_args": {"max_seq_length": _max_seq_length()},
    }


def _dedicated_multirank_merged_config() -> dev.InternalModelConfig:
    return {
        "trainer_gpu_ids": _multirank_trainer_gpu_ids(),
        "inference_gpu_ids": _multirank_inference_gpu_ids(),
        "rollout_weights_mode": "merged",
        "engine_args": {
            **_engine_args_for_yes_no_trainability(
                inference_gpu_ids=_multirank_inference_gpu_ids()
            ),
        },
        "init_args": {"max_seq_length": _max_seq_length()},
    }


def _shared_long_steps() -> int:
    return int(os.environ.get("ART_TEST_MEGATRON_SHARED_LONG_STEPS", "10"))


async def _list_model_ids(model: art.TrainableModel) -> list[str]:
    client = model.openai_client()
    return [model_info.id async for model_info in client.models.list()]


async def _chat_snapshot(model: art.TrainableModel, *, step: int) -> dict[str, object]:
    client = model.openai_client()
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello."}],
        model=model.get_inference_name(step=step),
        max_tokens=8,
        timeout=180.0,
        logprobs=True,
        top_logprobs=0,
    )
    return {
        "text": completion.choices[0].message.content,
        "has_logprobs": completion.choices[0].logprobs is not None,
    }


async def _runtime_is_sleeping(service: MegatronService) -> bool:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{service._vllm_base_url}/is_sleeping")
        response.raise_for_status()
        return bool(response.json()["is_sleeping"])


async def _wait_until_runtime_sleeping(
    service: MegatronService,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 0.5,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if await _runtime_is_sleeping(service):
            return True
        await asyncio.sleep(poll_s)
    return False


@asynccontextmanager
async def _megatron_backend_context(
    *,
    backend_root: Path,
    topology: Topology,
) -> AsyncIterator[MegatronBackend]:
    with _wandb_disabled():
        with provider_topology_env(topology):
            art.init_megatron_runtime_config(
                topology=art.MegatronTopologyConfig(
                    tp=topology.tp,
                    cp=topology.cp,
                    ep=topology.ep,
                    pp=topology.pp,
                    vpp=topology.vpp if topology.vpp != 1 else None,
                    etp=topology.etp,
                ),
                packed_sequence_length=_packed_sequence_length(),
            )
            async with MegatronBackend(
                path=str(backend_root), in_process=False
            ) as backend:
                yield backend


def _jitter_training_groups(
    groups: list[art.TrajectoryGroup],
    *,
    step: int,
) -> list[art.TrajectoryGroup]:
    jittered_groups: list[art.TrajectoryGroup] = []
    for group_index, group in enumerate(groups):
        jittered_trajectories: list[art.Trajectory] = []
        for trajectory_index, trajectory in enumerate(group.trajectories):
            reward = float(trajectory.reward) + 1e-3 * (
                1 + step + group_index + trajectory_index
            )
            jittered_trajectories.append(
                art.Trajectory(
                    messages_and_choices=trajectory.messages_and_choices,
                    reward=reward,
                )
            )
        jittered_groups.append(art.TrajectoryGroup(jittered_trajectories))
    return jittered_groups


async def _build_jittered_training_groups(
    model: art.TrainableModel,
    *,
    step: int,
    rollouts_per_prompt: int,
) -> list[art.TrajectoryGroup]:
    if rollouts_per_prompt < 2:
        raise ValueError("Shared Megatron long smoke requires rollouts_per_prompt >= 2")
    return _jitter_training_groups(
        await _build_training_groups(
            model,
            base_model=model.base_model,
            prompts=_train_group_prompts(),
            rollouts_per_prompt=rollouts_per_prompt,
        ),
        step=step,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for Megatron live smokes",
)
@pytest.mark.asyncio
async def test_megatron_backend_shared_lora_runtime_sleep_wake_live_smoke(
    artifact_dir: Path,
) -> None:
    _require_opt_in(SHARED_LORA_ENV)
    backend_root = artifact_dir / "art_workspace"
    backend_root.mkdir(parents=True, exist_ok=True)

    async with _megatron_backend_context(
        backend_root=backend_root,
        topology=SHARED_TOPOLOGY,
    ) as backend:
        model = art.TrainableModel(
            name=f"megatron-shared-live-{uuid.uuid4().hex[:8]}",
            project="integration-tests",
            base_model=_base_model(),
            _internal_config=_shared_live_config(),
            report_metrics=[],
        )
        await model.register(backend)
        service = cast(MegatronService, await backend._get_service(model))
        prompts = _train_group_prompts()
        await _warmup_model(model, base_model=model.base_model, prompt=prompts[0])
        step0_name = model.get_inference_name(step=0)
        model_ids_before = await _list_model_ids(model)
        train_groups = await _build_trainable_groups(
            model,
            base_model=model.base_model,
            prompts=prompts,
            rollouts_per_prompt=_rollouts_per_prompt(),
        )
        train_task = asyncio.create_task(
            backend.train(
                model,
                train_groups,
                learning_rate=float(os.environ.get("ART_TEST_MEGATRON_LR", "1e-4")),
                loss_fn="cispo",
            )
        )
        observed_sleep = False
        try:
            while not train_task.done():
                if await _runtime_is_sleeping(service):
                    observed_sleep = True
                    break
                await asyncio.sleep(0.5)
            assert observed_sleep or train_task.done()
            result = await train_task
        finally:
            if not train_task.done():
                await train_task

        latest_step = int(result.step)
        latest_name = model.get_inference_name(step=latest_step)
        model_ids_after = await _list_model_ids(model)
        eval_reward = await _evaluate_model(
            model,
            base_model=model.base_model,
            prompts=prompts,
            step=latest_step,
        )
        latest_snapshot = await _chat_snapshot(model, step=latest_step)
        runtime_sleep_after = await _runtime_is_sleeping(service)
        payload = {
            "base_model": model.base_model,
            "output_dir": service.output_dir,
            "step0_name": step0_name,
            "latest_name": latest_name,
            "latest_step": latest_step,
            "model_ids_before": model_ids_before,
            "model_ids_after": model_ids_after,
            "observed_sleep": observed_sleep,
            "runtime_sleep_after": runtime_sleep_after,
            "eval_reward": eval_reward,
            "latest_snapshot": latest_snapshot,
        }
        (artifact_dir / "shared_megatron_live_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert observed_sleep
        assert runtime_sleep_after is False
        assert latest_step > 0
        assert step0_name in model_ids_before
        assert step0_name in model_ids_after
        assert latest_name in model_ids_after
        assert latest_snapshot["has_logprobs"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for Megatron live smokes",
)
@pytest.mark.asyncio
async def test_megatron_backend_dedicated_merged_live_smoke(
    artifact_dir: Path,
) -> None:
    _require_opt_in(DEDICATED_MERGED_ENV)
    backend_root = artifact_dir / "art_workspace"
    backend_root.mkdir(parents=True, exist_ok=True)

    async with _megatron_backend_context(
        backend_root=backend_root,
        topology=ORACLE_TOPOLOGY,
    ) as backend:
        model = art.TrainableModel(
            name=f"megatron-merged-live-{uuid.uuid4().hex[:8]}",
            project="integration-tests",
            base_model=_base_model(),
            _internal_config=_dedicated_merged_config(),
            report_metrics=[],
        )
        await model.register(backend)
        service = cast(MegatronService, await backend._get_service(model))
        prompts = _train_group_prompts()
        await _warmup_model(model, base_model=model.base_model, prompt=prompts[0])
        step0_name = model.get_inference_name(step=0)
        model_ids_before = await _list_model_ids(model)
        train_groups = await _build_trainable_groups(
            model,
            base_model=model.base_model,
            prompts=prompts,
            rollouts_per_prompt=_rollouts_per_prompt(),
        )
        result = await backend.train(
            model,
            train_groups,
            learning_rate=float(os.environ.get("ART_TEST_MEGATRON_LR", "1e-4")),
            loss_fn="cispo",
        )
        latest_step = int(result.step)
        latest_name = model.get_inference_name(step=latest_step)
        model_ids_after = await _list_model_ids(model)
        eval_reward = await _evaluate_model(
            model,
            base_model=model.base_model,
            prompts=prompts,
            step=latest_step,
        )
        latest_snapshot = await _chat_snapshot(model, step=latest_step)
        payload = {
            "base_model": model.base_model,
            "output_dir": service.output_dir,
            "step0_name": step0_name,
            "latest_name": latest_name,
            "latest_step": latest_step,
            "model_ids_before": model_ids_before,
            "model_ids_after": model_ids_after,
            "eval_reward": eval_reward,
            "latest_snapshot": latest_snapshot,
        }
        (artifact_dir / "dedicated_megatron_merged_live_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert latest_step > 0
        assert step0_name in model_ids_before
        assert latest_name in model_ids_after
        assert step0_name not in model_ids_after
        assert latest_snapshot["has_logprobs"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 3,
    reason="Need at least 3 CUDA GPUs for multi-rank Megatron merged smoke",
)
@pytest.mark.asyncio
async def test_megatron_backend_dedicated_multirank_merged_live_smoke(
    artifact_dir: Path,
) -> None:
    _require_opt_in(DEDICATED_MULTIRANK_MERGED_ENV)
    backend_root = artifact_dir / "art_workspace"
    backend_root.mkdir(parents=True, exist_ok=True)

    async with _megatron_backend_context(
        backend_root=backend_root,
        topology=SHARED_TOPOLOGY,
    ) as backend:
        model = art.TrainableModel(
            name=f"megatron-multirank-merged-live-{uuid.uuid4().hex[:8]}",
            project="integration-tests",
            base_model=_base_model(),
            _internal_config=_dedicated_multirank_merged_config(),
            report_metrics=[],
        )
        await model.register(backend)
        service = cast(MegatronService, await backend._get_service(model))
        prompts = _train_group_prompts()
        await _warmup_model(model, base_model=model.base_model, prompt=prompts[0])
        step0_name = model.get_inference_name(step=0)
        model_ids_before = await _list_model_ids(model)
        train_groups = await _build_trainable_groups(
            model,
            base_model=model.base_model,
            prompts=prompts,
            rollouts_per_prompt=_rollouts_per_prompt(),
        )
        result = await backend.train(
            model,
            train_groups,
            learning_rate=float(os.environ.get("ART_TEST_MEGATRON_LR", "1e-4")),
            loss_fn="cispo",
        )
        latest_step = int(result.step)
        latest_name = model.get_inference_name(step=latest_step)
        model_ids_after = await _list_model_ids(model)
        eval_reward = await _evaluate_model(
            model,
            base_model=model.base_model,
            prompts=prompts,
            step=latest_step,
        )
        latest_snapshot = await _chat_snapshot(model, step=latest_step)
        payload = {
            "base_model": model.base_model,
            "output_dir": service.output_dir,
            "step0_name": step0_name,
            "latest_name": latest_name,
            "latest_step": latest_step,
            "model_ids_before": model_ids_before,
            "model_ids_after": model_ids_after,
            "eval_reward": eval_reward,
            "latest_snapshot": latest_snapshot,
            "trainer_gpu_ids": _multirank_trainer_gpu_ids(),
            "inference_gpu_ids": _multirank_inference_gpu_ids(),
            "topology": SHARED_TOPOLOGY.model_dump(),
        }
        (
            artifact_dir / "dedicated_megatron_multirank_merged_live_result.json"
        ).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert latest_step > 0
        assert step0_name in model_ids_before
        assert latest_name in model_ids_after
        assert step0_name not in model_ids_after
        assert latest_snapshot["has_logprobs"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need at least 2 CUDA GPUs for Megatron live smokes",
)
@pytest.mark.asyncio
async def test_megatron_backend_shared_lora_ten_step_live_smoke(
    artifact_dir: Path,
) -> None:
    _require_opt_in(SHARED_LONG_LORA_ENV)
    backend_root = artifact_dir / "art_workspace"
    backend_root.mkdir(parents=True, exist_ok=True)

    async with _megatron_backend_context(
        backend_root=backend_root,
        topology=SHARED_TOPOLOGY,
    ) as backend:
        model = art.TrainableModel(
            name=f"megatron-shared-long-live-{uuid.uuid4().hex[:8]}",
            project="integration-tests",
            base_model=_base_model(),
            _internal_config=_shared_live_config(),
            report_metrics=[],
        )
        await model.register(backend)
        service = cast(MegatronService, await backend._get_service(model))
        prompts = _train_group_prompts()
        await _warmup_model(model, base_model=model.base_model, prompt=prompts[0])
        step0_name = model.get_inference_name(step=0)
        model_ids_before = await _list_model_ids(model)
        step_reports: list[dict[str, object]] = []

        for step_index in range(_shared_long_steps()):
            train_groups = await _build_jittered_training_groups(
                model,
                step=step_index,
                rollouts_per_prompt=_rollouts_per_prompt(),
            )
            train_task = asyncio.create_task(
                backend.train(
                    model,
                    train_groups,
                    learning_rate=float(os.environ.get("ART_TEST_MEGATRON_LR", "1e-4")),
                    loss_fn="cispo",
                )
            )
            observed_sleep = False
            try:
                while not train_task.done():
                    if await _runtime_is_sleeping(service):
                        observed_sleep = True
                        break
                    await asyncio.sleep(0.5)
                assert observed_sleep or train_task.done()
                result = await train_task
            finally:
                if not train_task.done():
                    await train_task

            latest_step = int(result.step)
            eval_reward = await _evaluate_model(
                model,
                base_model=model.base_model,
                prompts=prompts,
                step=latest_step,
            )
            step_reports.append(
                {
                    "step": latest_step,
                    "observed_sleep": observed_sleep,
                    "eval_reward": eval_reward,
                    "train_reward": sum(
                        trajectory.reward
                        for group in train_groups
                        for trajectory in group.trajectories
                    )
                    / max(1, sum(len(group.trajectories) for group in train_groups)),
                }
            )

        latest_step = int(cast(Any, step_reports[-1]["step"]))
        latest_name = model.get_inference_name(step=latest_step)
        model_ids_after = await _list_model_ids(model)
        latest_snapshot = await _chat_snapshot(model, step=latest_step)
        runtime_sleep_after = await _runtime_is_sleeping(service)
        payload = {
            "base_model": model.base_model,
            "output_dir": service.output_dir,
            "step0_name": step0_name,
            "latest_name": latest_name,
            "latest_step": latest_step,
            "model_ids_before": model_ids_before,
            "model_ids_after": model_ids_after,
            "runtime_sleep_after": runtime_sleep_after,
            "latest_snapshot": latest_snapshot,
            "step_reports": step_reports,
        }
        (artifact_dir / "shared_megatron_ten_step_live_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert all(bool(step_report["observed_sleep"]) for step_report in step_reports)
        assert runtime_sleep_after is False
        assert latest_step >= _shared_long_steps()
        assert step0_name in model_ids_before
        assert step0_name in model_ids_after
        assert latest_name in model_ids_after
        assert latest_snapshot["has_logprobs"] is True
