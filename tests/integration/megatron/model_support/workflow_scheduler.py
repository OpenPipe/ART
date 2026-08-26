from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from pathlib import Path
import socket
from threading import Lock
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from art.megatron.lora_config import (
    MEGATRON_LORA_RANK_ENV,
    default_lora_rank_for_handler,
)
from art.megatron.model_support.registry import (
    get_model_support_handler_for_spec,
    get_model_support_spec,
    model_uses_expert_parallel,
)
from art.megatron.model_support.spec import ArchitectureReport
from art.vllm_runtime import VllmRuntimeLaunchConfig

from .validation_spec import ValidationReport, ValidationStageResult
from .workflow import (
    CORRECTNESS_ARTIFACT_ROOT_ENV,
    CORRECTNESS_PHASE_ENV,
    CORRECTNESS_REFERENCE_STAGE,
)
from .workflow_fixtures import (
    FIXTURE_PATH_ENV,
    WorkflowFixture,
    ensure_workflow_fixture,
)
from .workflow_forkserver import WorkflowForkserverPool
from .workflow_resources import (
    HANDLER_WORKFLOW_RESOURCES,
    WorkflowStageResources,
    resolve_stage_resources_for_visible_gpus,
)
from .workflow_runtime import (
    WorkflowDevice,
    WorkflowOperation,
    WorkflowOperationFailed,
    WorkflowPlacement,
    WorkflowResourceRequest,
    WorkflowRolePlacement,
    WorkflowRuntimeKey,
    WorkflowRuntimeTopology,
    WorkflowSession,
    WorkflowTrainerTopology,
    WorkflowVllmTopology,
    compile_workflow,
    execute_workflow,
)
from .workflow_stage_worker import (
    BASE_MEGATRON_MODE,
    BASE_MEGATRON_STAGES,
    RESIDENT_FUNCTIONAL_MODE,
    RESIDENT_FUNCTIONAL_STAGES,
    ResidentFunctionalSessionSpec,
    WorkflowStageWorkerItem,
    WorkflowStageWorkerSession,
)

_REDUCED_FIXTURE_GPU_SHARE = 0.125
_LIGHTWEIGHT_GPU_SHARE_OVERRIDES = {("glm52", "hf_parity"): 1.0}
_STAGE_DURATION_ESTIMATES_S = {
    "hf_parity": 120.0,
    "lora_coverage": 60.0,
    "train_inf_mismatch": 360.0,
    "correctness_sensitivity": 480.0,
    CORRECTNESS_REFERENCE_STAGE: 120.0,
    "chat_template_rollout": 45.0,
    "packing_invariance": 90.0,
    "length_trainability": 360.0,
    "e2e_throughput": 360.0,
}
_STAGE_DURATION_ESTIMATE_OVERRIDES_S = {
    # Rounded B300 workflow measurements; DSV4 is a deliberate architecture outlier.
    ("e2e_throughput", "gpt_oss_moe"): 300.0,
    ("e2e_throughput", "dsv4"): 690.0,
}
_SHARED_STARTUP_ESTIMATES_S = {
    # Base phases measured 74-79s standalone and share the same model startup.
    BASE_MEGATRON_MODE: 60.0,
    # Conservative cross-handler budget for resident trainer and serving startup.
    RESIDENT_FUNCTIONAL_MODE: 200.0,
}
_LIGHTWEIGHT_GPU_STAGES = frozenset(
    {"hf_parity", "lora_coverage", "packing_invariance"}
)
_CPU_STAGES = frozenset({"chat_template_rollout"})
_DEFAULT_STAGE_GPU_COUNTS = {
    "train_inf_mismatch": 4,
}
_WORKFLOW_HOSTS_ENV = "ART_MODEL_SUPPORT_WORKFLOW_HOSTS"
_VLLM_CAPACITY_ARGS = (
    "gpu_memory_utilization",
    "max_model_len",
    "max_logprobs",
    "max_num_seqs",
    "max_loras",
    "max_lora_rank",
    "max_num_batched_tokens",
)
_VLLM_PARALLEL_ARGS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
)


class PreparedWorkflow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    report: ValidationReport
    architecture: ArchitectureReport
    fixture: WorkflowFixture
    run_dir: Path
    output_json: Path | None
    stages: tuple[str, ...]
    allow_unvalidated_arch: bool
    include_sensitivity: bool | None
    fixture_provisioning_s: float
    _fixture_metric_recorded: bool = PrivateAttr(default=False)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    def record(self, result: ValidationStageResult) -> None:
        with self._lock:
            stage = next(
                stage for stage in self.report.stages if stage.name == result.name
            )
            stage.passed = result.passed
            stage.skipped = result.skipped
            stage.metrics = dict(result.metrics)
            stage.artifact_dir = result.artifact_dir
            if self.output_json is not None:
                self.output_json.parent.mkdir(parents=True, exist_ok=True)
                self.output_json.write_text(
                    self.report.model_dump_json(indent=2), encoding="utf-8"
                )

    def record_fixture_metric(self, metrics: dict[str, Any]) -> None:
        with self._lock:
            if self._fixture_metric_recorded:
                return
            metrics["fixture_provisioning_s"] = self.fixture_provisioning_s
            self._fixture_metric_recorded = True


class _FunctionalStageSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    trainer_gpu_ids: tuple[int, ...]
    inference_gpu_ids: tuple[int, ...]
    trainer_topology: WorkflowTrainerTopology
    vllm_topology: WorkflowVllmTopology
    vllm_external: bool
    engine_args: dict[str, object]


class _SharedFunctionalSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker: ResidentFunctionalSessionSpec
    topology: WorkflowRuntimeTopology


class _SharedBaseSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_count: int
    fixture: str
    topology: WorkflowRuntimeTopology


def _stage_duration_estimate(model_key: str, stage: str) -> float:
    return _STAGE_DURATION_ESTIMATE_OVERRIDES_S.get(
        (stage, model_key), _STAGE_DURATION_ESTIMATES_S[stage]
    )


def _shared_startup_estimate(mode: str, stage: str) -> float:
    if mode == BASE_MEGATRON_MODE:
        return _SHARED_STARTUP_ESTIMATES_S[mode]
    if mode == RESIDENT_FUNCTIONAL_MODE and stage != "lora_coverage":
        return _SHARED_STARTUP_ESTIMATES_S[mode]
    return 0.0


def _initialize_workflow(
    *,
    base_model: str,
    output_json: Path | None,
    allow_unvalidated_arch: bool,
) -> tuple[ValidationReport, Path]:
    from . import workflow

    report = workflow.initialize_validation_report(
        base_model=base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return report, workflow._new_workflow_run_dir(
        output_json=output_json,
        model_key=report.model_key,
    )


def prepare_workflow(
    *,
    base_model: str,
    include_sensitivity: bool | None,
    output_json: Path | None,
    skip_stages: set[str],
    allow_unvalidated_arch: bool,
    initialized: tuple[ValidationReport, Path] | None = None,
) -> PreparedWorkflow:
    from . import workflow

    report, run_dir = initialized or _initialize_workflow(
        base_model=base_model,
        output_json=output_json,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    dependency = next(
        stage for stage in report.stages if stage.name == "dependency_resolution"
    )
    started = time.monotonic()
    dependency.passed = True
    dependency.metrics = dict(report.dependency_versions)
    workflow._record_stage_duration(dependency, started=started)

    architecture_stage = next(
        stage for stage in report.stages if stage.name == "architecture_discovery"
    )
    started = time.monotonic()
    architecture = workflow._inspect_architecture_for_workflow(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    architecture_stage.passed = not architecture.unresolved_risks
    architecture_stage.metrics = {
        "recommended_min_layers": architecture.recommended_min_layers,
        "layer_families": [
            family.model_dump() for family in architecture.layer_families
        ],
        "unresolved_risks": list(architecture.unresolved_risks),
    }
    workflow._record_stage_duration(architecture_stage, started=started)

    stages = tuple(
        stage.name
        for stage in report.stages
        if stage.name not in {"dependency_resolution", "architecture_discovery"}
        and stage.name not in skip_stages
    )
    for stage in report.stages:
        if stage.name not in skip_stages:
            continue
        stage.skipped = True
        stage.metrics = {
            "skipped": True,
            "reason": "--skip-stage",
            "workflow_stage_duration_s": 0.0,
        }
    fixture_started = time.monotonic()
    fixture = ensure_workflow_fixture(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
        required_stages=frozenset(stages),
    )
    prepared = PreparedWorkflow(
        report=report,
        architecture=architecture,
        fixture=fixture,
        run_dir=run_dir,
        output_json=output_json,
        stages=stages,
        allow_unvalidated_arch=allow_unvalidated_arch,
        include_sensitivity=include_sensitivity,
        fixture_provisioning_s=time.monotonic() - fixture_started,
    )
    workflow._write_validation_report(report, output_json)
    return prepared


def _minimum_gpu_count(
    stage_name: str,
    resources: WorkflowStageResources,
    *,
    visible_gpu_count: int,
) -> int:
    minimum = resources.required_physical_gpus or 1
    for count in range(minimum, visible_gpu_count + 1):
        try:
            resolve_stage_resources_for_visible_gpus(
                stage_name,
                resources,
                visible_gpu_count=count,
            )
        except RuntimeError:
            continue
        return count
    raise RuntimeError(
        f"{stage_name} does not fit any allocation up to {visible_gpu_count} GPUs"
    )


def _stage_gpu_count(
    prepared: PreparedWorkflow, stage_name: str, available: int
) -> int:
    if stage_name in _CPU_STAGES:
        return 0
    if stage_name in _LIGHTWEIGHT_GPU_STAGES:
        return 1
    if stage_name == CORRECTNESS_REFERENCE_STAGE:
        return 1
    if stage_name == "correctness_sensitivity":
        from .oracle_harness import selected_suite_topologies

        handler = get_model_support_handler_for_spec(
            get_model_support_spec(
                prepared.report.base_model,
                allow_unvalidated_arch=prepared.allow_unvalidated_arch,
            )
        )
        return max(
            topology.world_size()
            for topology in selected_suite_topologies(
                is_moe=handler.is_moe,
                cp_supported=bool(handler.cp_supported),
            )
        )
    resources = getattr(
        HANDLER_WORKFLOW_RESOURCES.get(prepared.report.model_key), stage_name, None
    )
    if resources is None:
        if stage_name == "length_trainability":
            return (
                3
                if model_uses_expert_parallel(
                    prepared.report.base_model,
                    allow_unvalidated_arch=prepared.allow_unvalidated_arch,
                )
                else 2
            )
        try:
            return _DEFAULT_STAGE_GPU_COUNTS[stage_name]
        except KeyError:
            raise RuntimeError(
                "missing workflow resources for "
                f"{prepared.report.model_key}/{stage_name}"
            ) from None
    return _minimum_gpu_count(stage_name, resources, visible_gpu_count=available)


def _trainer_topology(
    variant: str, topology: Any | None = None, **updates: Any
) -> WorkflowTrainerTopology:
    values = {
        name: getattr(topology, name, default)
        for name, default in (
            ("tp", 1),
            ("cp", 1),
            ("ep", 1),
            ("etp", 1),
            ("dp", 1),
            ("pp", 1),
            ("vpp", 1),
            ("sp", False),
        )
    }
    return WorkflowTrainerTopology(variant=variant, **(values | updates))


def _trainer_environment(
    topology: WorkflowTrainerTopology,
    trainer_gpu_ids: tuple[int, ...],
    inference_gpu_ids: tuple[int, ...],
) -> dict[str, str]:
    values = {
        "TRAINER_GPU_IDS": ",".join(map(str, trainer_gpu_ids)),
        "INFERENCE_GPU_IDS": ",".join(map(str, inference_gpu_ids)),
        **{
            name.upper(): str(getattr(topology, name))
            for name in ("tp", "cp", "ep", "etp", "dp", "pp")
        },
    }
    return {
        **{f"ART_TRAIN_INF_MISMATCH_{name}": value for name, value in values.items()},
        **{
            f"ART_MODEL_SUPPORT_{name}": value
            for name, value in (
                values | {"VPP": str(topology.vpp), "SP": "1" if topology.sp else "0"}
            ).items()
        },
    }


def _vllm_topology(
    variant: str, engine_args: dict[str, object], *, gpu_count: int
) -> WorkflowVllmTopology:
    values = {}
    for field, key in zip(("tp", "pp", "dp"), _VLLM_PARALLEL_ARGS, strict=True):
        value = engine_args.get(key, 1)
        if type(value) is not int:
            raise RuntimeError(f"{variant} {key} must be an integer")
        values[field] = value
    expert_parallel = engine_args.get("enable_expert_parallel", False)
    if type(expert_parallel) is not bool:
        raise RuntimeError(f"{variant} enable_expert_parallel must be a boolean")
    topology = WorkflowVllmTopology(variant=variant, ep=expert_parallel, **values)
    if topology.tp * topology.pp * topology.dp != gpu_count:
        raise RuntimeError(
            f"{variant} vLLM topology {(topology.tp, topology.pp, topology.dp)} "
            f"does not match {gpu_count} inference GPUs"
        )
    return topology


def _resolved_stage_resources(
    prepared: PreparedWorkflow, stage_name: str, *, gpu_count: int
) -> WorkflowStageResources | None:
    configured = HANDLER_WORKFLOW_RESOURCES.get(prepared.report.model_key)
    resources = getattr(configured, stage_name, None)
    return (
        resolve_stage_resources_for_visible_gpus(
            stage_name, resources, visible_gpu_count=gpu_count
        )
        if resources is not None
        else None
    )


def _default_mismatch_trainer_topology(
    prepared: PreparedWorkflow, variant: str
) -> WorkflowTrainerTopology:
    handler = get_model_support_handler_for_spec(
        get_model_support_spec(
            prepared.report.base_model,
            allow_unvalidated_arch=prepared.allow_unvalidated_arch,
        )
    )
    cp = 2 if bool(handler.cp_supported) else 1
    return _trainer_topology(
        variant,
        cp=cp,
        ep=2 if handler.is_moe else 1,
        dp=1 if cp == 2 else 2,
    )


def _stage_runtime_topology(
    prepared: PreparedWorkflow, stage_name: str, *, gpu_count: int
) -> WorkflowRuntimeTopology:
    if stage_name in _CPU_STAGES:
        return WorkflowRuntimeTopology()
    if stage_name in {"correctness_sensitivity", CORRECTNESS_REFERENCE_STAGE}:
        from .oracle_harness import selected_suite_topologies

        handler = get_model_support_handler_for_spec(
            get_model_support_spec(
                prepared.report.base_model,
                allow_unvalidated_arch=prepared.allow_unvalidated_arch,
            )
        )
        variants = tuple(
            selected_suite_topologies(
                is_moe=handler.is_moe,
                cp_supported=bool(handler.cp_supported),
            )
        )
        if stage_name == CORRECTNESS_REFERENCE_STAGE:
            variants = variants[:1]
        else:
            variants = variants[1:]
        names = tuple(topology.slug() for topology in variants)
        return WorkflowRuntimeTopology(
            trainer_variants=tuple(
                _trainer_topology(name, topology)
                for name, topology in zip(names, variants, strict=True)
            ),
            role_placements=tuple(
                WorkflowRolePlacement(
                    variant=name,
                    trainer_gpu_ids=tuple(range(topology.world_size())),
                )
                for name, topology in zip(names, variants, strict=True)
            ),
        )
    if stage_name in {"train_inf_mismatch", "length_trainability"}:
        spec = _functional_stage_spec(prepared, stage_name, gpu_count)
        return WorkflowRuntimeTopology(
            trainer_variants=(spec.trainer_topology,),
            vllm_variants=(spec.vllm_topology,),
            role_placements=(
                WorkflowRolePlacement(
                    variant=stage_name,
                    trainer_gpu_ids=spec.trainer_gpu_ids,
                    vllm_gpu_ids=spec.inference_gpu_ids,
                    vllm_external=spec.vllm_external,
                ),
            ),
        )

    resources = _resolved_stage_resources(prepared, stage_name, gpu_count=gpu_count)
    if resources is not None:
        trainer = resources.megatron
        vllm = resources.vllm
        return WorkflowRuntimeTopology(
            trainer_variants=(
                (_trainer_topology(stage_name, trainer.topology),) if trainer else ()
            ),
            vllm_variants=(
                _vllm_topology(
                    stage_name,
                    vllm.engine_args(),
                    gpu_count=len(vllm.gpu_ids),
                ),
            )
            if vllm
            else (),
            role_placements=(
                WorkflowRolePlacement(
                    variant=stage_name,
                    trainer_gpu_ids=tuple(trainer.gpu_ids) if trainer else (),
                    vllm_gpu_ids=tuple(vllm.gpu_ids) if vllm else (),
                    vllm_external=resources.requires_external_vllm,
                ),
            ),
        )

    trainer = _trainer_topology(stage_name)
    trainer_gpu_ids = (0,)
    vllm_gpu_ids: tuple[int, ...] = ()
    vllm: WorkflowVllmTopology | None = None
    if stage_name == "train_inf_mismatch":
        trainer = _default_mismatch_trainer_topology(prepared, stage_name)
        trainer_gpu_ids = (0, 1)
        vllm_gpu_ids = (2, 3)
        vllm = WorkflowVllmTopology(
            variant=stage_name,
            tp=2,
            ep=trainer.ep > 1,
        )
    return WorkflowRuntimeTopology(
        trainer_variants=(trainer,),
        vllm_variants=(vllm,) if vllm else (),
        role_placements=(
            WorkflowRolePlacement(
                variant=stage_name,
                trainer_gpu_ids=trainer_gpu_ids,
                vllm_gpu_ids=vllm_gpu_ids,
            ),
        ),
    )


def _runtime_key(
    prepared: PreparedWorkflow,
    stage_name: str,
    *,
    gpu_count: int,
) -> WorkflowRuntimeKey:
    fixture_stage = (
        "correctness_sensitivity"
        if stage_name == CORRECTNESS_REFERENCE_STAGE
        else stage_name
    )
    environment = prepared.fixture.environment(fixture_stage)
    if stage_name in _CPU_STAGES:
        kind = "cpu"
        mode = stage_name
    elif stage_name in {
        "lora_coverage",
        "correctness_sensitivity",
        CORRECTNESS_REFERENCE_STAGE,
    }:
        kind = "megatron"
        mode = stage_name
    elif stage_name == "e2e_throughput":
        kind = "joint"
        mode = "throughput"
    else:
        kind = "joint"
        mode = stage_name
    return WorkflowRuntimeKey(
        source_fingerprint=str(prepared.report.git["commit"]),
        handler=prepared.report.model_key,
        fixture=environment["ART_MODEL_SUPPORT_FIXTURE_PATH"],
        kind=kind,
        topology=_stage_runtime_topology(prepared, stage_name, gpu_count=gpu_count),
        mode=mode,
        static_options=stage_name if mode == stage_name else "",
    )


def _stage_gpu_share(prepared: PreparedWorkflow, stage_name: str) -> float:
    if stage_name not in _LIGHTWEIGHT_GPU_STAGES:
        return 1.0
    if override := _LIGHTWEIGHT_GPU_SHARE_OVERRIDES.get(
        (prepared.report.model_key, stage_name)
    ):
        return override
    fixture_path = prepared.fixture.environment(stage_name)[
        "ART_MODEL_SUPPORT_FIXTURE_PATH"
    ]
    return (
        1.0
        if fixture_path == prepared.fixture.canonical_path
        else _REDUCED_FIXTURE_GPU_SHARE
    )


def _ordered_stages(stages: tuple[str, ...], selected: tuple[str, ...]) -> bool:
    return tuple(stage for stage in stages if stage in selected) == selected


def _topology_shape(topology: WorkflowRuntimeTopology) -> tuple[object, ...]:
    return (
        tuple(
            variant.model_dump(exclude={"variant"})
            for variant in topology.trainer_variants
        ),
        tuple(
            variant.model_dump(exclude={"variant"})
            for variant in topology.vllm_variants
        ),
        tuple(
            placement.model_dump(exclude={"variant"})
            for placement in topology.role_placements
        ),
    )


def _base_session(
    prepared: PreparedWorkflow,
    stage_gpu_counts: dict[str, int],
    *,
    visible_gpu_count: int,
) -> _SharedBaseSession | None:
    if not _ordered_stages(prepared.stages, BASE_MEGATRON_STAGES):
        return None
    handler = get_model_support_handler_for_spec(
        get_model_support_spec(
            prepared.report.base_model,
            allow_unvalidated_arch=prepared.allow_unvalidated_arch,
        )
    )
    if handler.is_moe:
        return None
    gpu_counts = {stage_gpu_counts[stage] for stage in BASE_MEGATRON_STAGES}
    fixtures = {
        prepared.fixture.environment(stage)[FIXTURE_PATH_ENV]
        for stage in BASE_MEGATRON_STAGES
    }
    topologies = tuple(
        _stage_runtime_topology(prepared, stage, gpu_count=stage_gpu_counts[stage])
        for stage in BASE_MEGATRON_STAGES
    )
    if (
        len(gpu_counts) != 1
        or next(iter(gpu_counts)) > visible_gpu_count
        or len(fixtures) != 1
        or _stage_gpu_share(prepared, BASE_MEGATRON_STAGES[0])
        != _stage_gpu_share(prepared, BASE_MEGATRON_STAGES[1])
        or any(
            _topology_shape(topology) != _topology_shape(topologies[0])
            for topology in topologies[1:]
        )
    ):
        return None
    topology = topologies[0]
    return _SharedBaseSession(
        gpu_count=gpu_counts.pop(),
        fixture=fixtures.pop(),
        topology=WorkflowRuntimeTopology(
            trainer_variants=tuple(
                variant.model_copy(update={"variant": BASE_MEGATRON_MODE})
                for variant in topology.trainer_variants
            ),
            vllm_variants=tuple(
                variant.model_copy(update={"variant": BASE_MEGATRON_MODE})
                for variant in topology.vllm_variants
            ),
            role_placements=tuple(
                placement.model_copy(update={"variant": BASE_MEGATRON_MODE})
                for placement in topology.role_placements
            ),
        ),
    )


def _functional_stage_spec(
    prepared: PreparedWorkflow, stage: str, gpu_count: int
) -> _FunctionalStageSpec:
    handler = get_model_support_handler_for_spec(
        get_model_support_spec(
            prepared.report.base_model,
            allow_unvalidated_arch=prepared.allow_unvalidated_arch,
        )
    )
    configured = HANDLER_WORKFLOW_RESOURCES.get(prepared.report.model_key)
    resources = getattr(configured, stage, None)
    if resources is None:
        if stage == "train_inf_mismatch":
            trainer_topology = _default_mismatch_trainer_topology(prepared, stage)
            trainer_gpu_count = (
                trainer_topology.tp
                * trainer_topology.cp
                * trainer_topology.pp
                * trainer_topology.dp
            )
            inference_gpu_count = gpu_count - trainer_gpu_count
        else:
            inference_gpu_count = 1
            trainer_gpu_count = gpu_count - 1
            trainer_topology = _trainer_topology(
                stage,
                cp=2 if handler.is_moe else 1,
                ep=2 if handler.is_moe else 1,
            )
        trainer_gpu_ids = tuple(range(trainer_gpu_count))
        inference_gpu_ids = tuple(range(trainer_gpu_count, gpu_count))
        engine_args: dict[str, object] = {
            "tensor_parallel_size": inference_gpu_count,
        }
        if handler.is_moe and inference_gpu_count > 1:
            engine_args["enable_expert_parallel"] = True
    else:
        resolved = resolve_stage_resources_for_visible_gpus(
            stage, resources, visible_gpu_count=gpu_count
        )
        if resolved.vllm is None:
            raise RuntimeError(f"{stage} resources require vLLM")
        inference_gpu_count = len(resolved.vllm.gpu_ids)
        trainer_gpu_count = (
            len(resolved.megatron.gpu_ids) if resolved.megatron is not None else 1
        )
        trainer_gpu_ids = (
            tuple(resolved.megatron.gpu_ids) if resolved.megatron is not None else (0,)
        )
        inference_gpu_ids = tuple(resolved.vllm.gpu_ids)
        vllm_external = resolved.requires_external_vllm
        trainer_topology = _trainer_topology(
            stage,
            resolved.megatron.topology if resolved.megatron is not None else None,
        )
        engine_args = resolved.vllm.engine_args()
    if trainer_gpu_count < 1 or inference_gpu_count < 1:
        raise RuntimeError(f"{stage} requires non-empty trainer and inference roles")
    trainer_world_size = (
        trainer_topology.tp
        * trainer_topology.cp
        * trainer_topology.pp
        * trainer_topology.dp
    )
    if trainer_world_size != trainer_gpu_count:
        raise RuntimeError(
            f"{stage} trainer topology has world size {trainer_world_size} "
            f"but {trainer_gpu_count} GPU ids"
        )
    engine_args = {
        **handler.vllm_engine_args(),
        **engine_args,
    }
    if handler.is_moe:
        engine_args["enable_return_routed_experts"] = True
    return _FunctionalStageSpec(
        stage=stage,
        trainer_gpu_ids=trainer_gpu_ids,
        inference_gpu_ids=inference_gpu_ids,
        trainer_topology=trainer_topology,
        vllm_topology=_vllm_topology(stage, engine_args, gpu_count=inference_gpu_count),
        vllm_external=vllm_external if resources is not None else False,
        engine_args=engine_args,
    )


def _merge_functional_engine_args(
    specs: tuple[_FunctionalStageSpec, ...], capacities: dict[str, int | float]
) -> dict[str, object] | None:
    static_options: dict[str, object] | None = None
    topology = specs[0].vllm_topology
    for spec in specs:
        engine_args = dict(spec.engine_args)
        for key in _VLLM_PARALLEL_ARGS:
            engine_args.pop(key, None)
        engine_args.pop("enable_expert_parallel", None)
        for key in _VLLM_CAPACITY_ARGS:
            value = engine_args.pop(key, None)
            if value is not None:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise RuntimeError(f"{spec.stage} {key} must be numeric")
                capacities[key] = max(capacities.get(key, 0), value)
        if static_options is None:
            static_options = engine_args
        elif engine_args != static_options:
            return None
    parallel_args: dict[str, object] = {
        "tensor_parallel_size": topology.tp,
        "pipeline_parallel_size": topology.pp,
        "data_parallel_size": topology.dp,
    }
    if topology.ep:
        parallel_args["enable_expert_parallel"] = True
    return (static_options or {}) | parallel_args | capacities


def _functional_session(
    prepared: PreparedWorkflow,
    stage_gpu_counts: dict[str, int],
    *,
    visible_gpu_count: int,
) -> _SharedFunctionalSession | None:
    if not _ordered_stages(prepared.stages, RESIDENT_FUNCTIONAL_STAGES):
        return None
    support = get_model_support_spec(
        prepared.report.base_model,
        allow_unvalidated_arch=prepared.allow_unvalidated_arch,
    )
    if support.native_vllm_lora_status == "disabled":
        return None
    handler = get_model_support_handler_for_spec(support)
    joint_stages = ("train_inf_mismatch", "length_trainability")
    specs = tuple(
        _functional_stage_spec(prepared, stage, stage_gpu_counts[stage])
        for stage in joint_stages
    )
    functional_env = prepared.fixture.resident_functional_environment()
    length_env = os.environ | functional_env
    capacities: dict[str, int | float] = {
        "max_model_len": int(
            length_env.get("ART_MODEL_SUPPORT_LENGTH_MAX_MODEL_LEN", 1024)
        ),
        "max_num_seqs": int(
            length_env.get("ART_MODEL_SUPPORT_LENGTH_MAX_NUM_SEQS", 32)
        ),
        "max_loras": 2,
        "max_logprobs": 20,
        "max_lora_rank": int(
            length_env.get(
                MEGATRON_LORA_RANK_ENV, default_lora_rank_for_handler(handler)
            )
        ),
    }
    resource_args = _merge_functional_engine_args(specs, capacities)
    if resource_args is None:
        return None
    mismatch = specs[0]
    trainer_gpu_count = len(mismatch.trainer_gpu_ids)
    inference_gpu_count = len(mismatch.inference_gpu_ids)
    gpu_count = inference_gpu_count + trainer_gpu_count
    if gpu_count > visible_gpu_count:
        return None
    inference_gpu_ids = tuple(range(trainer_gpu_count, gpu_count))
    trainer = mismatch.trainer_topology
    return _SharedFunctionalSession(
        worker=ResidentFunctionalSessionSpec(
            gpu_count=gpu_count,
            launch=VllmRuntimeLaunchConfig(
                base_model=functional_env[FIXTURE_PATH_ENV],
                port=0,
                cuda_visible_devices=",".join(map(str, inference_gpu_ids)),
                served_model_name="__art_functional_base__",
                engine_args={
                    "enforce_eager": True,
                    "generation_config": "vllm",
                    "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
                    **resource_args,
                },
                server_args={
                    "return_tokens_as_token_ids": True,
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                    **handler.vllm_server_args(),
                    "api_key": "art-functional-vllm",
                },
            ),
            trainer_gpu_ids=tuple(range(trainer_gpu_count)),
            trainer_environment=_trainer_environment(
                trainer,
                tuple(range(trainer_gpu_count)),
                inference_gpu_ids,
            ),
        ),
        topology=WorkflowRuntimeTopology(
            trainer_variants=(
                trainer.model_copy(update={"variant": RESIDENT_FUNCTIONAL_MODE}),
            ),
            vllm_variants=(
                mismatch.vllm_topology.model_copy(
                    update={"variant": RESIDENT_FUNCTIONAL_MODE}
                ),
            ),
            role_placements=(
                WorkflowRolePlacement(
                    variant=RESIDENT_FUNCTIONAL_MODE,
                    trainer_gpu_ids=tuple(range(trainer_gpu_count)),
                    vllm_gpu_ids=inference_gpu_ids,
                    vllm_external=mismatch.vllm_external,
                ),
            ),
        ),
    )


def compile_prepared_workflows(
    workflows: list[PreparedWorkflow], *, visible_gpu_count: int
):
    operations = []
    for prepared in workflows:
        stage_gpu_counts = {
            stage_name: _stage_gpu_count(prepared, stage_name, visible_gpu_count)
            for stage_name in prepared.stages
        }
        functional_session = _functional_session(
            prepared,
            stage_gpu_counts,
            visible_gpu_count=visible_gpu_count,
        )
        base_megatron = _base_session(
            prepared,
            stage_gpu_counts,
            visible_gpu_count=visible_gpu_count,
        )
        for stage_name in prepared.stages:
            shared = (
                functional_session
                if functional_session is not None
                and stage_name in RESIDENT_FUNCTIONAL_STAGES
                else None
            )
            stage_gpu_count = stage_gpu_counts[stage_name]
            runtime = _runtime_key(prepared, stage_name, gpu_count=stage_gpu_count)
            gpu_count = stage_gpu_count
            if shared is not None:
                gpu_count = shared.worker.gpu_count
            elif base_megatron is not None and stage_name in BASE_MEGATRON_STAGES:
                gpu_count = base_megatron.gpu_count
            if shared is not None:
                runtime = runtime.model_copy(
                    update={
                        "fixture": prepared.fixture.resident_functional_environment()[
                            FIXTURE_PATH_ENV
                        ],
                        "kind": "joint",
                        "topology": shared.topology,
                        "mode": RESIDENT_FUNCTIONAL_MODE,
                        "static_options": shared.worker.model_dump_json(),
                    }
                )
            elif base_megatron is not None and stage_name in BASE_MEGATRON_STAGES:
                runtime = runtime.model_copy(
                    update={
                        "fixture": base_megatron.fixture,
                        "kind": "megatron",
                        "mode": BASE_MEGATRON_MODE,
                        "topology": base_megatron.topology,
                        "static_options": "",
                    }
                )
            dependencies: tuple[str, ...] = ()
            if stage_name == "correctness_sensitivity":
                reference_id = (
                    f"{prepared.report.model_key}:{CORRECTNESS_REFERENCE_STAGE}"
                )
                correctness_affinity = f"{prepared.report.model_key}:correctness"
                operations.append(
                    WorkflowOperation(
                        id=reference_id,
                        stage=CORRECTNESS_REFERENCE_STAGE,
                        runtime=_runtime_key(
                            prepared, CORRECTNESS_REFERENCE_STAGE, gpu_count=1
                        ),
                        resources=WorkflowResourceRequest(
                            gpu_count=1,
                            host_affinity=correctness_affinity,
                        ),
                        estimated_duration_s=_stage_duration_estimate(
                            prepared.report.model_key, CORRECTNESS_REFERENCE_STAGE
                        ),
                    )
                )
                dependencies = (reference_id,)
            if shared is not None:
                stage_index = RESIDENT_FUNCTIONAL_STAGES.index(stage_name)
                if stage_index:
                    dependencies = (
                        f"{prepared.report.model_key}:"
                        f"{RESIDENT_FUNCTIONAL_STAGES[stage_index - 1]}",
                    )
            elif (
                stage_name == BASE_MEGATRON_STAGES[1]
                and BASE_MEGATRON_STAGES[0] in prepared.stages
            ):
                dependencies = (
                    f"{prepared.report.model_key}:{BASE_MEGATRON_STAGES[0]}",
                )
            operations.append(
                WorkflowOperation(
                    id=f"{prepared.report.model_key}:{stage_name}",
                    stage=stage_name,
                    runtime=runtime,
                    resources=WorkflowResourceRequest(
                        gpu_count=gpu_count,
                        gpu_share=(
                            1.0
                            if shared is not None
                            else _stage_gpu_share(prepared, stage_name)
                        ),
                        host_affinity=(
                            correctness_affinity
                            if stage_name == "correctness_sensitivity"
                            else None
                        ),
                    ),
                    dependencies=dependencies,
                    estimated_duration_s=_stage_duration_estimate(
                        prepared.report.model_key, stage_name
                    ),
                    estimated_shared_startup_s=_shared_startup_estimate(
                        runtime.mode, stage_name
                    ),
                )
            )
    return compile_workflow(operations)


def _workflow_hosts() -> list[str]:
    hosts = [
        host.strip()
        for host in os.environ.get(_WORKFLOW_HOSTS_ENV, socket.gethostname()).split(",")
        if host.strip()
    ]
    if not hosts or len(set(hosts)) != len(hosts):
        raise RuntimeError(f"{_WORKFLOW_HOSTS_ENV} must contain unique host names")
    return hosts


def _visible_devices() -> list[WorkflowDevice]:
    import torch

    count = int(torch.cuda.device_count())
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = raw.split(",") if raw else [str(index) for index in range(count)]
    if len(gpu_ids) != count:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES exposes {len(gpu_ids)} ids but torch sees {count} GPUs"
        )
    return [
        WorkflowDevice(host=host, gpu=gpu_id)
        for host in _workflow_hosts()
        for gpu_id in gpu_ids
    ]


def run_prepared_workflows(
    workflows: list[PreparedWorkflow],
    *,
    forkservers: WorkflowForkserverPool | None = None,
) -> list[ValidationReport]:
    import torch

    from . import workflow

    devices = _visible_devices()
    if not devices:
        raise RuntimeError("the scheduled model-support workflow requires CUDA GPUs")
    gpu_counts = {
        host: sum(device.host == host for device in devices)
        for host in {device.host for device in devices}
    }
    if len(set(gpu_counts.values())) != 1:
        raise RuntimeError(f"workflow hosts expose different GPU counts: {gpu_counts}")
    by_model_key = {prepared.report.model_key: prepared for prepared in workflows}
    plan = compile_prepared_workflows(
        workflows, visible_gpu_count=next(iter(gpu_counts.values()))
    )
    owns_forkservers = forkservers is None
    active_forkservers = forkservers or WorkflowForkserverPool(
        hosts=sorted(gpu_counts),
        repo_root=workflow.REPO_ROOT,
        tests_dir=workflow.TESTS_DIR,
        log_dir=workflows[0].run_dir / ".forkservers",
    )
    failure_evidence: dict[str, dict[str, object]] = {}
    failure_evidence_lock = Lock()

    def remember_failure(
        operation: WorkflowOperation,
        output_json: Path,
        result: ValidationStageResult,
    ) -> None:
        evidence: dict[str, object] = {"stage": operation.stage}
        if output_json.is_file():
            evidence["stage_result_json"] = str(output_json)
        if error := result.metrics.get("error"):
            evidence["error"] = error
        if result.artifact_dir is not None:
            evidence["artifact_dir"] = result.artifact_dir
        with failure_evidence_lock:
            failure_evidence[operation.id] = evidence

    def record_blocked(
        operations: tuple[WorkflowOperation, ...],
        failed_operations: tuple[str, ...],
    ) -> None:
        with failure_evidence_lock:
            evidence = {
                operation_id: dict(failure_evidence.get(operation_id, {}))
                for operation_id in failed_operations
            }
        for operation in operations:
            if operation.stage == CORRECTNESS_REFERENCE_STAGE:
                continue
            by_model_key[operation.runtime.handler].record(
                ValidationStageResult(
                    name=operation.stage,
                    metrics={
                        "blocked": True,
                        "reason": "blocked by failed workflow operation(s): "
                        + ", ".join(failed_operations),
                        "workflow_failed_dependencies": list(failed_operations),
                        "workflow_failure_evidence": evidence,
                        "workflow_stage_duration_s": 0.0,
                    },
                )
            )

    def runner(session: WorkflowSession, placement: WorkflowPlacement) -> None:
        owner = by_model_key[session.runtime.handler]
        session_dir = owner.run_dir / ".sessions" / session.id
        session_dir.mkdir(parents=True, exist_ok=False)
        architecture_json = session_dir / "architecture.json"
        request_json = session_dir / "request.json"
        session_log = session_dir / "worker.log"
        architecture_json.write_text(
            owner.architecture.model_dump_json(indent=2), encoding="utf-8"
        )
        items = []
        for operation in session.operations:
            stage_dir = owner.run_dir / operation.stage
            stage_dir.mkdir(parents=True, exist_ok=False)
            fixture_stage = (
                None
                if session.runtime.mode == RESIDENT_FUNCTIONAL_MODE
                else "correctness_sensitivity"
                if operation.stage == CORRECTNESS_REFERENCE_STAGE
                else operation.stage
            )
            environment = (
                owner.fixture.resident_functional_environment()
                if session.runtime.mode == RESIDENT_FUNCTIONAL_MODE
                else owner.fixture.environment(fixture_stage)
            )
            environment[workflow.WORKFLOW_RUN_DIR_ENV] = str(owner.run_dir)
            environment.update(
                {
                    "ART_MEGATRON_CACHE_ROOT": os.environ.get(
                        "ART_MEGATRON_CACHE_ROOT",
                        "/tmp/art-model-support-workflow/cache",
                    ),
                    "ART_MEGATRON_COMPILE_CACHE": "1",
                }
            )
            if operation.stage in {
                CORRECTNESS_REFERENCE_STAGE,
                "correctness_sensitivity",
            }:
                environment[CORRECTNESS_ARTIFACT_ROOT_ENV] = str(
                    owner.run_dir / ".correctness" / "artifacts"
                )
                environment[CORRECTNESS_PHASE_ENV] = (
                    "reference"
                    if operation.stage == CORRECTNESS_REFERENCE_STAGE
                    else "variants"
                )
            if owner.include_sensitivity is not None:
                environment[workflow.SKIP_SENSITIVITY_ENV] = (
                    "0" if owner.include_sensitivity else "1"
                )
            items.append(
                WorkflowStageWorkerItem(
                    stage=operation.stage,
                    stage_dir=str(stage_dir),
                    output_json=str(stage_dir / "stage_result.json"),
                    environment=environment,
                )
            )
        request = WorkflowStageWorkerSession(
            base_model=owner.report.base_model,
            architecture_json=str(architecture_json),
            allow_unvalidated_arch=owner.allow_unvalidated_arch,
            resident_functional=(
                ResidentFunctionalSessionSpec.model_validate_json(
                    session.runtime.static_options
                )
                if session.runtime.mode == RESIDENT_FUNCTIONAL_MODE
                else None
            ),
            base_megatron=session.runtime.mode == BASE_MEGATRON_MODE,
            items=tuple(items),
        )
        request_json.write_text(request.model_dump_json(indent=2), encoding="utf-8")
        if any(item.environment != items[0].environment for item in items[1:]):
            raise RuntimeError("one workflow session requires one process environment")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(
                    device.gpu for device in placement.devices
                ),
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(workflow.TESTS_DIR),
                            environment.get("PYTHONPATH"),
                        ),
                    )
                ),
                "WANDB_MODE": "disabled",
            }
        )
        placement_hosts = {device.host for device in placement.devices}
        if len(placement_hosts) > 1:
            raise RuntimeError("one workflow session cannot span hosts")
        execution_host = placement.host or socket.gethostname()
        timeout_s = sum(
            workflow._WORKFLOW_STAGE_TIMEOUT_OVERRIDES_S.get(
                (operation.stage, owner.report.base_model),
                workflow._WORKFLOW_STAGE_TIMEOUT_S,
            )
            for operation in session.operations
        )
        fork_result = active_forkservers.run(
            execution_host,
            request_json=request_json,
            log_path=session_log,
            environment=environment,
            session_environment=items[0].environment,
            torch_threads=torch.get_num_threads(),
            timeout_s=timeout_s,
        )
        returncode = fork_result["returncode"]
        worker_wall_s = float(fork_result["child_wall_s"])
        results: list[tuple[WorkflowOperation, Path, bool, ValidationStageResult]] = []
        for operation, item in zip(session.operations, items, strict=True):
            output_json = Path(item.output_json)
            produced = output_json.exists()
            if produced:
                result = ValidationStageResult.model_validate_json(
                    output_json.read_text(encoding="utf-8")
                )
            else:
                detail = (
                    f"session exceeded {timeout_s:g}s"
                    if returncode is None
                    else "session worker did not write stage_result.json"
                    if returncode == 0
                    else workflow._subprocess_log_tail(session_log)
                    or f"session exited with code {returncode}"
                )
                result = ValidationStageResult(
                    name=operation.stage,
                    passed=False,
                    metrics={"error": detail},
                )
            result.metrics["workflow_session_id"] = session.id
            result.metrics["workflow_gpu_ids"] = [
                device.gpu for device in placement.devices
            ]
            result.metrics["workflow_host"] = execution_host
            result.metrics["workflow_session_worker_s"] = worker_wall_s
            result.metrics["workflow_session_operation_count"] = len(session.operations)
            result.metrics.update(active_forkservers.metrics(execution_host))
            results.append((operation, output_json, produced, result))

        failed_operation_id: str | None = None
        for operation, output_json, produced, result in results:
            if not produced and failed_operation_id is not None:
                record_blocked((operation,), (failed_operation_id,))
                continue
            if operation.stage == CORRECTNESS_REFERENCE_STAGE:
                if not result.passed:
                    remember_failure(operation, output_json, result)
                    failed_operation_id = failed_operation_id or operation.id
                continue
            if operation.stage == "correctness_sensitivity":
                reference_result = ValidationStageResult.model_validate_json(
                    (
                        owner.run_dir
                        / CORRECTNESS_REFERENCE_STAGE
                        / "stage_result.json"
                    ).read_text(encoding="utf-8")
                )
                reference_s = float(
                    reference_result.metrics["workflow_stage_duration_s"]
                )
                composition_s = float(result.metrics["workflow_stage_duration_s"])
                result.metrics.update(
                    {
                        "correctness_reference_duration_s": reference_s,
                        "correctness_composition_duration_s": composition_s,
                        "correctness_total_compute_s": reference_s + composition_s,
                    }
                )
            owner.record_fixture_metric(result.metrics)
            if operation.stage in workflow._RUNTIME_CLEANUP_STAGES:
                try:
                    result.metrics.update(
                        workflow._prune_runtime_artifacts(
                            owner.run_dir / operation.stage
                        )
                    )
                except Exception as exc:
                    result.passed = False
                    result.metrics["runtime_artifact_cleanup_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            owner.record(result)
            if not result.passed:
                remember_failure(operation, output_json, result)
                failed_operation_id = failed_operation_id or operation.id
        if failed_operation_id is not None:
            raise WorkflowOperationFailed(failed_operation_id)

    context = (
        active_forkservers if owns_forkservers else nullcontext(active_forkservers)
    )
    with context:
        execution = execute_workflow(plan, devices=devices, runner=runner)
    sessions = {session.id: session for session in plan.sessions}
    for session_id, failed_operations in execution.blocked_by_failed_operations.items():
        record_blocked(sessions[session_id].operations, failed_operations)
    for prepared in workflows:
        workflow._finalize_validation_report(prepared.report, partial=False)
        workflow._write_validation_report(prepared.report, prepared.output_json)
    return [prepared.report for prepared in workflows]


def build_scheduled_validation_reports(
    *,
    base_models: list[str],
    include_sensitivity: bool | None = None,
    output_json_by_model: dict[str, Path | None] | None = None,
    skip_stages: set[str] | None = None,
    allow_unvalidated_arch: bool = False,
) -> list[ValidationReport]:
    from . import workflow

    skip_stages = skip_stages or set()
    output_json_by_model = output_json_by_model or {}
    initialized = [
        (
            base_model,
            _initialize_workflow(
                base_model=base_model,
                output_json=output_json_by_model.get(base_model),
                allow_unvalidated_arch=allow_unvalidated_arch,
            ),
        )
        for base_model in base_models
    ]

    def prepare(item: tuple[str, tuple[ValidationReport, Path]]) -> PreparedWorkflow:
        base_model, workflow_identity = item
        return prepare_workflow(
            base_model=base_model,
            include_sensitivity=include_sensitivity,
            output_json=output_json_by_model.get(base_model),
            skip_stages=skip_stages,
            allow_unvalidated_arch=allow_unvalidated_arch,
            initialized=workflow_identity,
        )

    with ThreadPoolExecutor(max_workers=1) as forkserver_executor:
        forkserver_future = forkserver_executor.submit(
            WorkflowForkserverPool,
            hosts=sorted(_workflow_hosts()),
            repo_root=workflow.REPO_ROOT,
            tests_dir=workflow.TESTS_DIR,
            log_dir=initialized[0][1][1] / ".forkservers",
        )
        try:
            with ThreadPoolExecutor(max_workers=len(initialized)) as executor:
                workflows = list(executor.map(prepare, initialized))
        except BaseException as error:
            try:
                forkserver_future.result().close()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "workflow preparation and forkserver cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise
        forkservers = forkserver_future.result()
        with forkservers:
            return run_prepared_workflows(workflows, forkservers=forkservers)
