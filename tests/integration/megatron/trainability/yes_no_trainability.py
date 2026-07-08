from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager, nullcontext
import gc
import os
from pathlib import Path
import re
import time
from typing import Any, AsyncIterator, Iterator, Literal, TypedDict, cast
import uuid

from pydantic import BaseModel, Field
import torch

import art
from art import dev
from art.local import LocalBackend
from art.megatron.backend import MegatronBackend
from art.megatron.model_support.registry import (
    get_model_support_spec,
    model_supports_context_parallel,
    model_uses_expert_parallel,
)
from art.megatron.model_support.spec import RolloutWeightsMode

from ..model_support.oracle_harness import Topology, oracle_topology
from ..model_support.oracle_worker import provider_topology_env
from ..model_support.workflow_resources import (
    handler_workflow_resources_for_base_model,
    resolve_stage_resources_for_current_host,
)

_TRAINER_GPU_IDS_ENV = "ART_MODEL_SUPPORT_TRAINER_GPU_IDS"
_INFERENCE_GPU_IDS_ENV = "ART_MODEL_SUPPORT_INFERENCE_GPU_IDS"
_SHARED_GPU_IDS_ENV = "ART_MODEL_SUPPORT_SHARED_GPU_IDS"
_VARIANT_ENV = "ART_MODEL_SUPPORT_YES_NO_VARIANT"
_EXTERNAL_VLLM_URL_ENV = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_URL"
_EXTERNAL_VLLM_API_KEY_ENV = "ART_MODEL_SUPPORT_EXTERNAL_VLLM_API_KEY"
_TRAINABILITY_ROOT = (
    Path(__file__).resolve().parents[4] / ".local" / "model_support_validation"
)
_SHARED_MEGATRON_TOPOLOGY = Topology(tp=1, ep=2, etp=1, dp=1, cp=2, sp=False)
_DENSE_SHARED_MEGATRON_TOPOLOGY = Topology(tp=1, ep=1, etp=1, dp=1, cp=2, sp=False)
_VARIANT_NAME = Literal[
    "megatron_shared",
    "megatron_dedicated",
    "unsloth_dedicated",
]
_RESOURCE_STAGE_NAME = Literal["yes_no_trainability", "length_trainability"]


class _TrainKwargs(TypedDict):
    packed_sequence_length: int


class TrainabilityStepReport(BaseModel):
    step: int
    eval_reward: float
    train_reward: float
    train_metrics: dict[str, float] = Field(default_factory=dict)


class YesNoTrainabilityReport(BaseModel):
    variant: _VARIANT_NAME
    backend_name: Literal["megatron", "local"]
    placement_mode: Literal["shared", "dedicated"]
    base_model: str
    output_dir: str
    trainer_gpu_ids: list[int]
    inference_gpu_ids: list[int]
    rollout_weights_mode: str
    reward_threshold: float
    max_steps: int
    prompt_count: int
    eval_prompt_count: int
    rollouts_per_prompt: int
    prompt_tree_depth: int = 0
    prompt_tree_branch_count: int = 0
    latest_step: int
    initial_eval_reward: float
    final_eval_reward: float | None = None
    saturated_step: int | None = None
    step0_name: str
    latest_name: str
    model_ids_before: list[str] = Field(default_factory=list)
    model_ids_after: list[str] = Field(default_factory=list)
    latest_snapshot: dict[str, object] = Field(default_factory=dict)
    steps: list[TrainabilityStepReport] = Field(default_factory=list)


class _TrainabilityVariant(BaseModel):
    name: _VARIANT_NAME
    backend_name: Literal["megatron", "local"]
    placement_mode: Literal["shared", "dedicated"]
    topology: Topology | None = None
    trainer_gpu_ids: list[int] = Field(default_factory=list)
    inference_gpu_ids: list[int] = Field(default_factory=list)


_YES_NO_PROMPT_ROOT = (
    "Read the validation card and answer with one word from yes, no, or maybe."
)
_YES_NO_PROMPT_MIDS = (
    "Branch alpha: the card is about deployment readiness.",
    "Branch beta: the card is about metric interpretation.",
)
_YES_NO_PROMPT_LEAVES = (
    "Case one: the safest answer is uncertain.",
    "Case two: the report contains a contradiction.",
    "Case three: the check has partial evidence.",
    "Case four: the reviewer needs a cautious final word.",
)


def build_prompts() -> list[str]:
    prompt = os.environ.get("ART_MODEL_SUPPORT_YES_NO_PROMPT", "").strip()
    prompt_count = _get_env_int("ART_MODEL_SUPPORT_YES_NO_PROMPT_COUNT", 8)
    if prompt:
        return [prompt] * max(1, prompt_count)
    prompts = [
        "\n\n".join(
            (
                _YES_NO_PROMPT_ROOT,
                _YES_NO_PROMPT_MIDS[(index // 2) % len(_YES_NO_PROMPT_MIDS)],
                _YES_NO_PROMPT_LEAVES[index % len(_YES_NO_PROMPT_LEAVES)],
                "Return only yes, no, or maybe.",
            )
        )
        for index in range(max(1, prompt_count))
    ]
    return prompts


def _prompt_tree_shape(prompts: list[str]) -> tuple[int, int]:
    mid_count = len(
        {mid for mid in _YES_NO_PROMPT_MIDS if any(mid in prompt for prompt in prompts)}
    )
    leaf_count = len(
        {
            leaf
            for leaf in _YES_NO_PROMPT_LEAVES
            if any(leaf in prompt for prompt in prompts)
        }
    )
    return (3 if mid_count and leaf_count else 1, mid_count + leaf_count)


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "_").replace("-", "_")


def _parse_gpu_id_env(name: str) -> list[int] | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _external_vllm_runtime_config() -> dev.VllmRuntimeArgs | None:
    server_url = os.environ.get(_EXTERNAL_VLLM_URL_ENV)
    if server_url is None or server_url.strip() == "":
        return None
    return {
        "mode": "external",
        "server_url": server_url,
        "api_key": os.environ.get(_EXTERNAL_VLLM_API_KEY_ENV, "art-external-vllm"),
    }


def _topology_with_env_overrides(topology: Topology) -> Topology:
    updates: dict[str, int | bool] = {}
    for env_name, attr in (
        ("ART_MODEL_SUPPORT_TP", "tp"),
        ("ART_MODEL_SUPPORT_EP", "ep"),
        ("ART_MODEL_SUPPORT_ETP", "etp"),
        ("ART_MODEL_SUPPORT_DP", "dp"),
        ("ART_MODEL_SUPPORT_CP", "cp"),
        ("ART_MODEL_SUPPORT_PP", "pp"),
        ("ART_MODEL_SUPPORT_VPP", "vpp"),
    ):
        if raw_value := os.environ.get(env_name):
            updates[attr] = int(raw_value)
    if raw_sp := os.environ.get("ART_MODEL_SUPPORT_SP"):
        updates["sp"] = raw_sp.strip().lower() in {"1", "true", "yes", "on"}
    return topology.model_copy(update=updates) if updates else topology


def _variant_with_env_overrides(
    variant: _TrainabilityVariant,
) -> _TrainabilityVariant:
    trainer_gpu_ids = _parse_gpu_id_env(_TRAINER_GPU_IDS_ENV)
    inference_gpu_ids = _parse_gpu_id_env(_INFERENCE_GPU_IDS_ENV)
    updates: dict[str, object] = {}
    if trainer_gpu_ids is not None:
        updates["trainer_gpu_ids"] = trainer_gpu_ids
    if inference_gpu_ids is not None:
        updates["inference_gpu_ids"] = inference_gpu_ids
    if variant.topology is not None:
        updates["topology"] = _topology_with_env_overrides(variant.topology)
    return variant.model_copy(update=updates) if updates else variant


def _resolve_shared_gpu_ids() -> list[int]:
    if shared_gpu_ids := _parse_gpu_id_env(_SHARED_GPU_IDS_ENV):
        return shared_gpu_ids
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Need at least 2 visible CUDA GPUs for shared trainability")
    return [0, 1]


def _resolve_dedicated_gpu_ids() -> tuple[list[int], list[int]]:
    trainer_gpu_ids = _parse_gpu_id_env(_TRAINER_GPU_IDS_ENV)
    inference_gpu_ids = _parse_gpu_id_env(_INFERENCE_GPU_IDS_ENV)
    if trainer_gpu_ids is not None or inference_gpu_ids is not None:
        if trainer_gpu_ids is None or inference_gpu_ids is None:
            raise RuntimeError(
                f"{_TRAINER_GPU_IDS_ENV} and {_INFERENCE_GPU_IDS_ENV} must both be set"
            )
        return trainer_gpu_ids, inference_gpu_ids
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "Need at least 2 visible CUDA GPUs for dedicated trainability"
        )
    return [0], [1]


def _safe_gpu_memory_utilization(device_ids: list[int]) -> float:
    requested = float(
        os.environ.get("ART_MODEL_SUPPORT_YES_NO_GPU_MEMORY_UTILIZATION", "0.85")
    )
    min_free_gib = float(
        os.environ.get("ART_MODEL_SUPPORT_YES_NO_MIN_FREE_GPU_GIB", "8")
    )
    min_utilization = min(
        requested,
        float(
            os.environ.get(
                "ART_MODEL_SUPPORT_YES_NO_MIN_GPU_MEMORY_UTILIZATION",
                "0.5",
            )
        ),
    )
    attempts = _get_env_int("ART_MODEL_SUPPORT_YES_NO_GPU_MEMORY_RETRY_ATTEMPTS", 12)
    sleep_s = _get_env_float("ART_MODEL_SUPPORT_YES_NO_GPU_MEMORY_RETRY_SLEEP_S", 5.0)
    devices = sorted(set(device_ids))
    last_message = "no GPU memory samples collected"

    for attempt in range(attempts):
        free_ratios: list[float] = []
        low_free: list[str] = []
        for device in devices:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            free_gib = free_bytes / (1024**3)
            if free_gib < min_free_gib:
                low_free.append(
                    f"GPU {device} has only {free_gib:.1f} GiB free < {min_free_gib:.1f} GiB required"
                )
            free_ratios.append(free_bytes / total_bytes)

        utilization = max(0.02, min(requested, min(free_ratios) * 0.95))
        if not low_free and utilization >= min_utilization:
            return utilization

        ratio_summary = ", ".join(
            f"GPU {device}: free_ratio={ratio:.3f}"
            for device, ratio in zip(devices, free_ratios, strict=True)
        )
        last_message = "; ".join(
            [
                *low_free,
                f"computed gpu_memory_utilization={utilization:.3f}",
                ratio_summary,
            ]
        )
        if attempt == attempts - 1:
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        time.sleep(sleep_s)

    raise RuntimeError(
        "Unable to recover enough free GPU memory for yes/no validation runtime startup. "
        f"{last_message}"
    )


def reward_for_answer(text: str) -> float:
    return {"yes": 0.5, "no": 0.75, "maybe": 1.0}.get(
        first_word_for_answer(text).lower(),
        0.0,
    )


def first_word_for_answer(text: str | None) -> str:
    if not text:
        return ""
    stripped = re.sub(
        r"<think>.*?</think>\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    first_word = stripped.strip().split(maxsplit=1)
    if not first_word:
        return ""
    return first_word[0].strip(".,!?:;\"'()[]{}")


def _get_env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _get_env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


def _max_tokens() -> int:
    return _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_TOKENS", 5)


def _render_chat_messages(base_model: str, prompt: str) -> art.Messages:
    del base_model
    return [{"role": "user", "content": prompt}]


def _enable_thinking() -> bool:
    return os.environ.get(
        "ART_MODEL_SUPPORT_YES_NO_ENABLE_THINKING", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _extra_body() -> dict[str, object]:
    return {"chat_template_kwargs": {"enable_thinking": _enable_thinking()}}


def _request_timeout(name: str, default: float) -> float:
    return _get_env_float(name, default)


def _engine_args_for_yes_no_trainability(
    *,
    inference_gpu_ids: list[int],
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    enable_sleep_mode: bool | None = None,
) -> dev.EngineArgs:
    engine_args: dict[str, object] = {
        "gpu_memory_utilization": _safe_gpu_memory_utilization(inference_gpu_ids),
        "max_model_len": _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_MODEL_LEN", 128),
        "max_num_seqs": _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_NUM_SEQS", 4),
        "enforce_eager": True,
        "tensor_parallel_size": tensor_parallel_size,
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
    }
    if enable_expert_parallel:
        engine_args["enable_expert_parallel"] = True
    if enable_sleep_mode is not None:
        engine_args["enable_sleep_mode"] = enable_sleep_mode
    return cast(dev.EngineArgs, engine_args)


@contextmanager
def _wandb_disabled() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in ("WANDB_API_KEY", "WANDB_MODE")}
    os.environ.pop("WANDB_API_KEY", None)
    os.environ["WANDB_MODE"] = "disabled"
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _temporary_env(updates: dict[str, str] | None) -> Iterator[None]:
    if not updates:
        yield
        return
    saved = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _artifact_dir(base_model: str, variant_name: _VARIANT_NAME) -> Path:
    path = (
        _TRAINABILITY_ROOT / _slugify(base_model) / variant_name / uuid.uuid4().hex[:8]
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _trainability_stage_resources(
    base_model: str,
    *,
    stage_name: _RESOURCE_STAGE_NAME,
    allow_unvalidated_arch: bool = False,
):
    workflow_resources = handler_workflow_resources_for_base_model(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    if workflow_resources is None:
        return None
    stage_resources = getattr(workflow_resources, stage_name)
    if stage_resources is None:
        return None
    return resolve_stage_resources_for_current_host(stage_name, stage_resources)


def _build_variant(
    variant_name: _VARIANT_NAME,
    *,
    base_model: str,
    allow_unvalidated_arch: bool = False,
    resource_stage_name: _RESOURCE_STAGE_NAME = "yes_no_trainability",
) -> _TrainabilityVariant:
    stage_resources = _trainability_stage_resources(
        base_model,
        stage_name=resource_stage_name,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    is_moe = model_uses_expert_parallel(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    cp_supported = model_supports_context_parallel(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    if variant_name == "megatron_shared":
        if (
            stage_resources is not None
            and stage_resources.megatron is not None
            and stage_resources.vllm is not None
        ):
            shared_gpu_ids = sorted(
                {*stage_resources.megatron.gpu_ids, *stage_resources.vllm.gpu_ids}
            )
        else:
            shared_gpu_ids = _resolve_shared_gpu_ids()
        if not cp_supported:
            shared_world_size = len(shared_gpu_ids)
            return _variant_with_env_overrides(
                _TrainabilityVariant(
                    name=variant_name,
                    backend_name="megatron",
                    placement_mode="shared",
                    topology=Topology(
                        tp=shared_world_size,
                        ep=shared_world_size if is_moe else 1,
                        etp=1,
                        dp=1,
                        cp=1,
                        sp=shared_world_size > 1,
                    ),
                    trainer_gpu_ids=shared_gpu_ids,
                    inference_gpu_ids=shared_gpu_ids,
                )
            )
        return _variant_with_env_overrides(
            _TrainabilityVariant(
                name=variant_name,
                backend_name="megatron",
                placement_mode="shared",
                topology=(
                    _SHARED_MEGATRON_TOPOLOGY
                    if is_moe
                    else _DENSE_SHARED_MEGATRON_TOPOLOGY
                ),
                trainer_gpu_ids=shared_gpu_ids,
                inference_gpu_ids=shared_gpu_ids,
            )
        )
    if (
        variant_name == "megatron_dedicated"
        and stage_resources is not None
        and stage_resources.megatron is not None
        and stage_resources.vllm is not None
    ):
        workflow_topology = stage_resources.megatron.topology
        return _variant_with_env_overrides(
            _TrainabilityVariant(
                name=variant_name,
                backend_name="megatron",
                placement_mode="dedicated",
                topology=Topology(
                    tp=workflow_topology.tp,
                    ep=workflow_topology.ep,
                    etp=workflow_topology.etp,
                    dp=workflow_topology.dp,
                    sp=workflow_topology.sp,
                    cp=workflow_topology.cp,
                    pp=workflow_topology.pp,
                ),
                trainer_gpu_ids=list(stage_resources.megatron.gpu_ids),
                inference_gpu_ids=list(stage_resources.vllm.gpu_ids),
            )
        )
    trainer_gpu_ids, inference_gpu_ids = _resolve_dedicated_gpu_ids()
    if variant_name == "megatron_dedicated":
        return _variant_with_env_overrides(
            _TrainabilityVariant(
                name=variant_name,
                backend_name="megatron",
                placement_mode="dedicated",
                topology=oracle_topology(is_moe=is_moe),
                trainer_gpu_ids=trainer_gpu_ids,
                inference_gpu_ids=inference_gpu_ids,
            )
        )
    return _variant_with_env_overrides(
        _TrainabilityVariant(
            name=variant_name,
            backend_name="local",
            placement_mode="dedicated",
            trainer_gpu_ids=trainer_gpu_ids,
            inference_gpu_ids=inference_gpu_ids,
        )
    )


def _variant_packed_sequence_length(variant: _TrainabilityVariant) -> int:
    return _get_env_int("ART_MODEL_SUPPORT_YES_NO_PACKED_SEQUENCE_LENGTH", 1024)


def _variant_train_kwargs(variant: _TrainabilityVariant) -> _TrainKwargs:
    return {"packed_sequence_length": _variant_packed_sequence_length(variant)}


def _variant_init_args(variant: _TrainabilityVariant) -> dev.InitArgs:
    return {"max_seq_length": _variant_packed_sequence_length(variant)}


def _init_megatron_runtime_config(variant: _TrainabilityVariant) -> None:
    if variant.topology is None:
        return
    init_runtime_config = getattr(art, "init_megatron_runtime_config", None)
    if init_runtime_config is None:
        return
    init_runtime_config(
        topology=art.MegatronTopologyConfig(
            tp=variant.topology.tp,
            cp=variant.topology.cp,
            ep=variant.topology.ep,
            etp=variant.topology.etp,
        ),
        packed_sequence_length=_variant_packed_sequence_length(variant),
    )


def _variant_max_steps(variant: _TrainabilityVariant) -> int:
    default = 12 if variant.backend_name == "local" else 4
    return _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_STEPS", default)


def _variant_rollouts_per_prompt(variant: _TrainabilityVariant) -> int:
    default = 8 if variant.backend_name == "local" else 4
    return _get_env_int("ART_MODEL_SUPPORT_YES_NO_ROLLOUTS_PER_PROMPT", default)


def _rollout_weights_mode(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> RolloutWeightsMode:
    return get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    ).default_rollout_weights_mode


def _default_variant_name(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> _VARIANT_NAME:
    if override := os.environ.get(_VARIANT_ENV, "").strip():
        if override not in {"megatron_shared", "megatron_dedicated"}:
            raise ValueError(
                f"Unsupported {_VARIANT_ENV}={override!r}. "
                "Expected 'megatron_shared' or 'megatron_dedicated'."
            )
        return cast(_VARIANT_NAME, override)
    workflow_resources = handler_workflow_resources_for_base_model(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    if (
        workflow_resources is not None
        and workflow_resources.yes_no_trainability_variant is not None
    ):
        return workflow_resources.yes_no_trainability_variant
    is_moe = model_uses_expert_parallel(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    rollout_weights_mode = _rollout_weights_mode(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    if rollout_weights_mode == "merged" or not is_moe:
        return "megatron_dedicated"
    return "megatron_shared"


def _build_internal_config(
    variant: _TrainabilityVariant,
    *,
    base_model: str,
    rollout_weights_mode: RolloutWeightsMode | None = None,
    allow_unvalidated_arch: bool = False,
    resource_stage_name: _RESOURCE_STAGE_NAME = "yes_no_trainability",
) -> dev.InternalModelConfig:
    shared = variant.placement_mode == "shared"
    inference_gpu_ids = variant.inference_gpu_ids
    stage_resources = _trainability_stage_resources(
        base_model,
        stage_name=resource_stage_name,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    stage_resources_apply = (
        not shared
        and variant.backend_name == "megatron"
        and stage_resources is not None
        and stage_resources.megatron is not None
        and stage_resources.vllm is not None
        and variant.trainer_gpu_ids == stage_resources.megatron.gpu_ids
        and variant.inference_gpu_ids == stage_resources.vllm.gpu_ids
    )
    if stage_resources_apply:
        assert stage_resources is not None
        assert stage_resources.vllm is not None
        vllm_resources = stage_resources.vllm
    else:
        vllm_resources = None
    engine_args = _engine_args_for_yes_no_trainability(
        inference_gpu_ids=inference_gpu_ids,
        tensor_parallel_size=(
            vllm_resources.tensor_parallel_size
            if vllm_resources is not None
            else len(inference_gpu_ids)
            if shared
            else 1
        ),
        enable_expert_parallel=(
            vllm_resources.enable_expert_parallel
            if vllm_resources is not None
            else shared
            and variant.backend_name == "megatron"
            and model_uses_expert_parallel(
                base_model,
                allow_unvalidated_arch=allow_unvalidated_arch,
            )
        ),
        enable_sleep_mode=True if shared else None,
    )
    if vllm_resources is not None:
        engine_args.update(vllm_resources.engine_args())
    elif shared and stage_resources is not None and stage_resources.vllm is not None:
        engine_args.update(stage_resources.vllm.extra_engine_args)
    engine_args["model"] = base_model
    internal_config = dev.InternalModelConfig(
        rollout_weights_mode=rollout_weights_mode
        or _rollout_weights_mode(
            base_model,
            allow_unvalidated_arch=allow_unvalidated_arch,
        ),
        engine_args=engine_args,
        init_args=_variant_init_args(variant),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    external_runtime = _external_vllm_runtime_config()
    if (
        stage_resources is not None
        and stage_resources.requires_external_vllm
        and external_runtime is None
    ):
        raise RuntimeError(
            f"{resource_stage_name} for this model requires an external vLLM server. "
            f"Set {_EXTERNAL_VLLM_URL_ENV}."
        )
    if external_runtime is not None:
        internal_config["vllm_runtime"] = external_runtime
    if not shared:
        internal_config["trainer_gpu_ids"] = variant.trainer_gpu_ids
        internal_config["inference_gpu_ids"] = variant.inference_gpu_ids
    if not stage_resources_apply:
        dev.validate_dedicated_config(internal_config)
    return internal_config


@asynccontextmanager
async def _backend_context(
    variant: _TrainabilityVariant,
    *,
    backend_root: Path,
    extra_env: dict[str, str] | None = None,
) -> AsyncIterator[LocalBackend | MegatronBackend]:
    with _wandb_disabled(), _temporary_env(extra_env):
        topology_context = (
            provider_topology_env(variant.topology)
            if variant.topology is not None
            else nullcontext()
        )
        with topology_context:
            if variant.backend_name == "megatron":
                async with MegatronBackend(
                    path=str(backend_root),
                    in_process=False,
                ) as backend:
                    yield backend
                return
            async with LocalBackend(path=str(backend_root)) as backend:
                yield backend


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


async def _evaluate_groups(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompts: list[str],
    step: int,
) -> list[art.TrajectoryGroup]:
    client = model.openai_client()

    async def _group_for_prompt(prompt: str) -> art.TrajectoryGroup:
        messages = _render_chat_messages(base_model, prompt)
        completion = await client.chat.completions.create(
            messages=messages,
            model=model.get_inference_name(step=step),
            max_tokens=_max_tokens(),
            extra_body=_extra_body(),
            temperature=_get_env_float(
                "ART_MODEL_SUPPORT_YES_NO_EVAL_TEMPERATURE",
                0.0,
            ),
            timeout=_request_timeout("ART_MODEL_SUPPORT_YES_NO_EVAL_TIMEOUT", 180.0),
        )
        choice = completion.choices[0]
        return art.TrajectoryGroup(
            [
                art.Trajectory(
                    messages_and_choices=[*messages, choice],
                    reward=reward_for_answer(choice.message.content or ""),
                )
            ]
        )

    return await art.gather_trajectory_groups(
        [_group_for_prompt(prompt) for prompt in prompts]  # ty: ignore[invalid-argument-type]
    )


def _mean_group_reward(groups: list[art.TrajectoryGroup]) -> float:
    rewards = [
        trajectory.reward for group in groups for trajectory in group.trajectories
    ]
    return sum(rewards) / max(1, len(rewards))


async def _evaluate_model(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompts: list[str],
    step: int,
) -> float:
    return _mean_group_reward(
        await _evaluate_groups(
            model,
            base_model=base_model,
            prompts=prompts,
            step=step,
        )
    )


async def _build_training_groups(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompts: list[str],
    rollouts_per_prompt: int,
) -> list[art.TrajectoryGroup]:
    client = model.openai_client()

    async def _group_for_prompt(prompt: str) -> art.TrajectoryGroup:
        messages = _render_chat_messages(base_model, prompt)
        completion = await client.chat.completions.create(
            messages=messages,
            model=model.get_inference_name(),
            max_tokens=_max_tokens(),
            n=rollouts_per_prompt,
            extra_body=_extra_body(),
            temperature=_get_env_float(
                "ART_MODEL_SUPPORT_YES_NO_ROLLOUT_TEMPERATURE",
                1.2,
            ),
            timeout=_request_timeout(
                "ART_MODEL_SUPPORT_YES_NO_ROLLOUT_TIMEOUT",
                180.0,
            ),
        )
        return art.TrajectoryGroup(
            [
                art.Trajectory(
                    messages_and_choices=[*messages, choice],
                    reward=reward_for_answer(choice.message.content or ""),
                )
                for choice in completion.choices
            ]
        )

    return await art.gather_trajectory_groups(
        [_group_for_prompt(prompt) for prompt in prompts]  # ty: ignore[invalid-argument-type]
    )


def _group_has_reward_variance(group: art.TrajectoryGroup) -> bool:
    return len({trajectory.reward for trajectory in group.trajectories}) > 1


async def _build_trainable_groups(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompts: list[str],
    rollouts_per_prompt: int,
) -> list[art.TrajectoryGroup]:
    max_attempts = _get_env_int("ART_MODEL_SUPPORT_YES_NO_MAX_ROLLOUT_ATTEMPTS", 4)
    for _ in range(max_attempts):
        groups = await _build_training_groups(
            model,
            base_model=base_model,
            prompts=prompts,
            rollouts_per_prompt=rollouts_per_prompt,
        )
        trainable_groups = [
            group for group in groups if _group_has_reward_variance(group)
        ]
        if trainable_groups:
            return trainable_groups
    raise RuntimeError(
        "No reward-variant trajectory groups were produced for yes/no trainability"
    )


async def _warmup_model(
    model: art.TrainableModel,
    *,
    base_model: str,
    prompt: str,
) -> None:
    client = model.openai_client()
    await client.chat.completions.create(
        messages=_render_chat_messages(base_model, prompt),
        model=model.get_inference_name(step=0),
        max_tokens=1,
        extra_body=_extra_body(),
        temperature=0.0,
        timeout=_request_timeout("ART_MODEL_SUPPORT_YES_NO_WARMUP_TIMEOUT", 900.0),
    )


async def run_yes_no_trainability_async(
    *,
    base_model: str,
    variant_name: _VARIANT_NAME = "megatron_shared",
    artifact_root: Path | None = None,
    rollout_weights_mode: RolloutWeightsMode | None = None,
    allow_unvalidated_arch: bool = False,
    extra_env: dict[str, str] | None = None,
) -> YesNoTrainabilityReport:
    variant = _build_variant(
        variant_name,
        base_model=base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    backend_root = artifact_root or _artifact_dir(base_model, variant.name)
    backend_root.mkdir(parents=True, exist_ok=True)
    reward_threshold = _get_env_float("ART_MODEL_SUPPORT_YES_NO_REWARD_THRESHOLD", 0.9)
    max_steps = _variant_max_steps(variant)
    rollouts_per_prompt = _variant_rollouts_per_prompt(variant)
    eval_prompt_count = _get_env_int("ART_MODEL_SUPPORT_YES_NO_EVAL_PROMPTS", 8)
    prompts = build_prompts()
    eval_prompts = prompts[:eval_prompt_count]
    prompt_tree_depth, prompt_tree_branch_count = _prompt_tree_shape(prompts)
    internal_config = _build_internal_config(
        variant,
        base_model=base_model,
        rollout_weights_mode=rollout_weights_mode,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    rollout_weights_mode = internal_config["rollout_weights_mode"]
    _init_megatron_runtime_config(variant)
    model = art.TrainableModel(
        name=f"{variant.name}-{uuid.uuid4().hex[:8]}",
        project="model-support-validation",
        base_model=base_model,
        _internal_config=internal_config,
        report_metrics=[],
    )
    train_kwargs = _variant_train_kwargs(variant)
    workflow_resources = handler_workflow_resources_for_base_model(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    stage_resources = (
        workflow_resources.yes_no_trainability
        if workflow_resources is not None
        else None
    )
    if stage_resources is not None:
        stage_resources = resolve_stage_resources_for_current_host(
            "yes_no_trainability",
            stage_resources,
        )
    backend_env = {
        **(stage_resources.megatron_env if stage_resources is not None else {}),
        **(extra_env or {}),
    }

    async with _backend_context(
        variant, backend_root=backend_root, extra_env=backend_env
    ) as backend:
        await model.register(backend)
        output_dir = Path(model.base_path) / model.project / "models" / model.name
        await _warmup_model(model, base_model=base_model, prompt=prompts[0])
        step0_name = model.get_inference_name(step=0)
        model_ids_before = await _list_model_ids(model)
        initial_eval_groups = await _evaluate_groups(
            model,
            base_model=base_model,
            prompts=eval_prompts,
            step=0,
        )
        initial_eval_reward = _mean_group_reward(initial_eval_groups)
        await model.log(initial_eval_groups, step=0, split="val")
        report = YesNoTrainabilityReport(
            variant=variant.name,
            backend_name=variant.backend_name,
            placement_mode=variant.placement_mode,
            base_model=base_model,
            output_dir=str(output_dir),
            trainer_gpu_ids=variant.trainer_gpu_ids,
            inference_gpu_ids=variant.inference_gpu_ids,
            rollout_weights_mode=rollout_weights_mode,
            reward_threshold=reward_threshold,
            max_steps=max_steps,
            prompt_count=len(prompts),
            eval_prompt_count=len(eval_prompts),
            rollouts_per_prompt=rollouts_per_prompt,
            prompt_tree_depth=prompt_tree_depth,
            prompt_tree_branch_count=prompt_tree_branch_count,
            latest_step=0,
            initial_eval_reward=initial_eval_reward,
            step0_name=step0_name,
            latest_name=step0_name,
            model_ids_before=model_ids_before,
        )

        for _ in range(max_steps):
            train_groups = await _build_trainable_groups(
                model,
                base_model=base_model,
                prompts=prompts,
                rollouts_per_prompt=rollouts_per_prompt,
            )
            result = await backend.train(
                model,
                train_groups,
                learning_rate=_get_env_float(
                    "ART_MODEL_SUPPORT_YES_NO_LEARNING_RATE",
                    1e-4,
                ),
                loss_fn="cispo",
                packed_sequence_length=train_kwargs["packed_sequence_length"],
            )
            await model.log(
                train_groups,
                metrics=result.metrics,
                step=result.step,
                split="train",
            )
            eval_groups = await _evaluate_groups(
                model,
                base_model=base_model,
                prompts=eval_prompts,
                step=result.step,
            )
            eval_reward = _mean_group_reward(eval_groups)
            await model.log(eval_groups, step=result.step, split="val")
            report.latest_step = int(result.step)
            report.latest_name = model.get_inference_name(step=result.step)
            report.final_eval_reward = float(eval_reward)
            report.steps.append(
                TrainabilityStepReport(
                    step=int(result.step),
                    eval_reward=float(eval_reward),
                    train_reward=sum(
                        trajectory.reward
                        for group in train_groups
                        for trajectory in group.trajectories
                    )
                    / max(1, sum(len(group.trajectories) for group in train_groups)),
                    train_metrics={
                        key: float(value)
                        for key, value in result.metrics.items()
                        if isinstance(value, int | float)
                    },
                )
            )
            if eval_reward >= reward_threshold:
                report.saturated_step = int(result.step)
                break

        report.model_ids_after = await _list_model_ids(model)
        report.latest_snapshot = await _chat_snapshot(model, step=report.latest_step)

    output_dir = Path(report.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report


def run_yes_no_trainability(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> YesNoTrainabilityReport:
    return asyncio.run(
        run_yes_no_trainability_async(
            base_model=base_model,
            variant_name=_default_variant_name(
                base_model,
                allow_unvalidated_arch=allow_unvalidated_arch,
            ),
            allow_unvalidated_arch=allow_unvalidated_arch,
        )
    )


def yes_no_trainability_passed(report: YesNoTrainabilityReport) -> bool:
    learned_from_below_threshold = (
        report.saturated_step is not None
        and report.saturated_step > 0
        and report.initial_eval_reward < report.reward_threshold
        and report.final_eval_reward is not None
        and report.final_eval_reward >= report.reward_threshold
        and report.final_eval_reward > report.initial_eval_reward
    )
    already_saturated_and_stable = (
        report.initial_eval_reward >= report.reward_threshold
        and report.latest_step > 0
        and report.final_eval_reward is not None
        and report.final_eval_reward >= report.reward_threshold
        and bool(report.steps)
        and any(step.train_metrics.get("grad_norm", 0.0) > 0.0 for step in report.steps)
    )
    return learned_from_below_threshold or already_saturated_and_stable


def run_megatron_dedicated_yes_no_trainability(
    base_model: str,
    *,
    rollout_weights_mode: RolloutWeightsMode | None = None,
) -> YesNoTrainabilityReport:
    return asyncio.run(
        run_yes_no_trainability_async(
            base_model=base_model,
            variant_name="megatron_dedicated",
            rollout_weights_mode=rollout_weights_mode,
        )
    )


def run_unsloth_dedicated_yes_no_trainability(
    base_model: str,
) -> YesNoTrainabilityReport:
    return asyncio.run(
        run_yes_no_trainability_async(
            base_model=base_model,
            variant_name="unsloth_dedicated",
        )
    )
