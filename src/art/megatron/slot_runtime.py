from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art import dev
from art.distributed import ArtLaunchContext, ArtRuntime, ArtRuntimeConfig
from art.distributed.rollout import RolloutModelSpec
from art.distributed.specs import RuntimeTopology
from art.model import TrainableModel
from art.runtime_attestation import (
    PairedSlotRuntimeAttestation,
    RuntimeArchitectureAttestation,
    RuntimeHostAttestation,
)
from art.training import TrainingInputResolver, TrainingRunSpec
from art.types import MegatronRuntimeConfig
from art.utils.output_dirs import get_step_checkpoint_dir
from art.vllm_route_transport import RouteBundleReader

from .operation_handler import MegatronCheckpointOperations, MegatronOperationConfig
from .optimizer_state import (
    optimizer_adapter,
    optimizer_model_lease,
    read_committed_optimizer_pointer,
    resolve_committed_optimizer_policy,
)
from .paired_inference import MegatronPairedInferencePublisher
from .route_retention import RouteBundleOwnershipProvider
from .runtime.build import build_trainer_runtime_spec
from .runtime.portable_snapshot import PortableSnapshotArchive
from .runtime.run_residency import RunResidencyPolicy
from .runtime.specs import TrainerGeneration, TrainerRuntimeSpec
from .runtime_config import init_megatron_runtime_config
from .slot_coordinator import (
    MegatronOperationEvidenceSink,
    MegatronSlotCoordinator,
    MegatronSlotResourceManager,
    MegatronSlotRun,
    MegatronSlotScheduleConfig,
)


class MegatronSlotLaunchConfig(BaseModel):
    """Complete immutable input for one persistent trainer allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str = Field(min_length=1)
    runtime_source_epoch: int = Field(ge=0)
    topology: RuntimeTopology
    megatron: MegatronRuntimeConfig
    base_model: str = Field(min_length=1)
    model: dict[str, Any] = Field(default_factory=dict)
    enable_moe_routing_replay: bool = True
    command_timeout_s: float = Field(default=300.0, gt=0)
    shutdown_timeout_s: float = Field(default=240.0, gt=0)
    runtime: ArtRuntimeConfig = Field(default_factory=ArtRuntimeConfig)
    schedule: MegatronSlotScheduleConfig = Field(
        default_factory=MegatronSlotScheduleConfig
    )
    residency: RunResidencyPolicy = Field(default_factory=RunResidencyPolicy)

    @model_validator(mode="after")
    def _validate_trainer_topology(self) -> "MegatronSlotLaunchConfig":
        trainer = self.topology.trainer
        if trainer is None:
            raise ValueError("Megatron slot topology requires a trainer mesh")
        if trainer.topology != self.megatron.topology:
            raise ValueError(
                "Megatron runtime topology does not match the slot trainer mesh"
            )
        return self


class MegatronSlotRuntimeDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_source_id: str = Field(min_length=1)
    runtime_source_epoch: int = Field(ge=0)
    runtime_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    trainer_architecture: RuntimeArchitectureAttestation
    paired_attestation: PairedSlotRuntimeAttestation | None = None


class MegatronRunBootstrapConfig(BaseModel):
    """Stable service input for creating or recovering one logical slot run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=255)
    training_session_id: str = Field(min_length=1, max_length=255)
    run: TrainingRunSpec
    output_dir: str = Field(min_length=1)
    initial_operation_sequence: int = Field(default=0, ge=0)
    max_retained_inputs: int = Field(default=65, ge=1, le=256)


@dataclass(slots=True)
class MegatronRunBinding:
    run: MegatronSlotRun
    config: MegatronOperationConfig


@dataclass(slots=True)
class MegatronSlotRuntime:
    """Lifecycle owner for one ART runtime and its shared trainer coordinator."""

    runtime: ArtRuntime
    coordinator: MegatronSlotCoordinator
    descriptor: MegatronSlotRuntimeDescriptor
    paired_inference: MegatronPairedInferencePublisher | None = None

    async def bind_run(
        self,
        request: MegatronRunBootstrapConfig,
        *,
        checkpoints: MegatronCheckpointOperations | None = None,
        max_retained_operations: int = 128,
        portable_archive: PortableSnapshotArchive | None = None,
    ) -> MegatronRunBinding:
        config = await asyncio.to_thread(
            prepare_megatron_run_config,
            request,
            self.coordinator.trainer.runtime_spec,
            portable_archive=portable_archive,
        )
        run = await self.coordinator.register_run(
            config,
            checkpoints=checkpoints,
            max_retained_operations=max_retained_operations,
            portable_archive=portable_archive,
        )
        return MegatronRunBinding(run=run, config=config)

    async def aclose(self) -> None:
        await self.runtime.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.aclose()


async def launch_megatron_slot(
    config: MegatronSlotLaunchConfig,
    *,
    launch: ArtLaunchContext | None = None,
    route_bundle_reader: RouteBundleReader | None = None,
    route_bundle_ownership: RouteBundleOwnershipProvider | None = None,
    input_resolver: TrainingInputResolver | None = None,
    resources: MegatronSlotResourceManager | None = None,
    operation_evidence_sink: MegatronOperationEvidenceSink | None = None,
) -> MegatronSlotRuntime:
    """Start one shared trainer and return its only service-facing command root."""

    init_megatron_runtime_config(config.megatron)
    runtime = (
        await ArtRuntime.start_local(
            config.topology,
            config=config.runtime,
            route_bundle_reader=route_bundle_reader,
        )
        if launch is None
        else await ArtRuntime.start(
            launch.host_mesh,
            config.topology,
            config=config.runtime,
            route_bundle_reader=route_bundle_reader,
        )
    )
    try:
        runtime_spec = build_trainer_runtime_spec(
            runtime,
            base_model=config.base_model,
            config=cast(dev.BackendModelConfig, dict(config.model)),
            enable_expert_replay=config.enable_moe_routing_replay,
            offload_between_jobs=False,
            run_residency=config.residency,
        )
        trainer = await runtime.start_shared_trainer(
            runtime_spec,
            launch_id=config.slot_id,
            command_timeout_s=config.command_timeout_s,
            shutdown_timeout_s=config.shutdown_timeout_s,
        )
        paired_inference = (
            await MegatronPairedInferencePublisher.start(
                runtime,
                trainer,
                base_model=config.base_model,
                config=cast(dev.BackendModelConfig, dict(config.model)),
                runtime_spec=runtime_spec,
            )
            if config.topology.model_services
            else None
        )
        coordinator = MegatronSlotCoordinator(
            runtime,
            trainer,
            resources=resources,
            schedule=config.schedule,
            publisher=paired_inference,
            route_ownership=route_bundle_ownership,
            input_resolver=input_resolver,
            operation_evidence_sink=operation_evidence_sink,
            command_timeout_s=config.command_timeout_s,
        )
        runtime.register_closeable(coordinator)
    except BaseException:
        await runtime.close()
        raise
    trainer_architecture = trainer.architecture_attestation
    paired_attestation = None
    if paired_inference is not None:
        trainer_host_ids = tuple(
            dict.fromkeys(rank.host_id for rank in runtime_spec.trainer_mesh.ranks)
        )
        inference_host_ids = tuple(
            dict.fromkeys(member.host_id for member in paired_inference.service.members)
        )
        paired_attestation = PairedSlotRuntimeAttestation(
            trainer=trainer_architecture,
            inference=paired_inference.architecture_attestation,
            trainer_hosts=tuple(
                RuntimeHostAttestation.from_admission(report)
                for report in runtime.host_admission_reports(trainer_host_ids)
            ),
            inference_hosts=tuple(
                RuntimeHostAttestation.from_admission(report)
                for report in runtime.host_admission_reports(inference_host_ids)
            ),
        )
    return MegatronSlotRuntime(
        runtime=runtime,
        coordinator=coordinator,
        descriptor=MegatronSlotRuntimeDescriptor(
            runtime_source_id=config.slot_id,
            runtime_source_epoch=config.runtime_source_epoch,
            runtime_fingerprint=runtime_spec.fingerprint,
            trainer_architecture=trainer_architecture,
            paired_attestation=paired_attestation,
        ),
        paired_inference=paired_inference,
    )


def prepare_megatron_run_config(
    request: MegatronRunBootstrapConfig,
    runtime_spec: TrainerRuntimeSpec,
    *,
    portable_archive: PortableSnapshotArchive | None = None,
) -> MegatronOperationConfig:
    """Create or recover exact ART-owned paths for one logical trainer run."""

    spec = request.run
    if (
        spec.base_model != runtime_spec.model_identifier
        or spec.dtype != runtime_spec.dtype
    ):
        raise ValueError("training run model or dtype differs from the slot runtime")
    if spec.adapter.rank > runtime_spec.lora_rank or not set(
        spec.adapter.target_modules
    ).issubset(runtime_spec.lora_target_modules):
        raise ValueError("training run adapter shape exceeds the slot runtime")

    output_dir = Path(request.output_dir).absolute()
    optimizer_state = output_dir / "optimizer_states"
    if portable_archive is not None:
        archive = PortableSnapshotArchive.model_validate(
            portable_archive.model_dump(mode="json")
        )
        portable_generation = archive.generation
        if portable_generation.training_session_id != request.training_session_id:
            raise RuntimeError("portable trainer state differs from run admission")
        generation = TrainerGeneration(
            training_session_id=portable_generation.training_session_id,
            policy_step=portable_generation.policy_step,
            generation_id=portable_generation.generation_id,
            adapter_path=str(
                output_dir / "checkpoints" / portable_generation.generation_id
            ),
        )
    else:
        generation = None

    initial_adapter = Path(get_step_checkpoint_dir(str(output_dir), 0))
    if generation is None:
        with optimizer_model_lease(optimizer_state):
            pointer = read_committed_optimizer_pointer(str(optimizer_state))
            if pointer is None and not initial_adapter.exists():
                initial_adapter.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".art-bootstrap-", dir=initial_adapter.parent
                ) as temporary:
                    staged = Path(temporary) / "adapter"
                    from .identity_lora import create_identity_lora
                    from .model_support import get_model_support_handler

                    create_identity_lora(
                        spec.base_model,
                        str(staged),
                        rank=spec.adapter.rank,
                        target_modules=list(spec.adapter.target_modules),
                        lora_alpha=int(runtime_spec.lora_alpha),
                        random_state=(
                            spec.seed
                            if spec.seed is not None
                            else runtime_spec.random_state
                        ),
                        allow_unvalidated_arch=runtime_spec.allow_unvalidated_arch,
                        handler=get_model_support_handler(
                            spec.base_model,
                            allow_unvalidated_arch=runtime_spec.allow_unvalidated_arch,
                        ),
                    )
                    os.rename(staged, initial_adapter)
                    directory_fd = os.open(initial_adapter.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            if pointer is None:
                adapter = optimizer_adapter(
                    initial_adapter,
                    0,
                    training_session_id=request.training_session_id,
                )
            else:
                adapter = resolve_committed_optimizer_policy(
                    str(optimizer_state),
                    initial_adapter_path=str(initial_adapter),
                ).policy_adapter
        from .model_support.lora_disk import (
            load_adapter_config,
            training_target_modules,
        )

        adapter_config = load_adapter_config(adapter.identity)
        if (
            adapter.training_session_id != request.training_session_id
            or int(adapter_config.get("r", 0)) != spec.adapter.rank
            or set(training_target_modules(adapter_config))
            != set(spec.adapter.target_modules)
            or adapter_config.get("base_model_name_or_path") != spec.base_model
        ):
            raise RuntimeError("recovered trainer state differs from run admission")
        generation = TrainerGeneration(
            training_session_id=adapter.training_session_id,
            policy_step=adapter.step,
            generation_id=adapter.generation_id,
            adapter_path=adapter.identity,
        )

    rollout_model = RolloutModelSpec.from_model(
        TrainableModel(
            name=spec.base_model,
            run_name=request.run_id,
            project="serverless-training",
            base_model=spec.base_model,
            lora_config={
                "rank": spec.adapter.rank,
                "target_modules": list(spec.adapter.target_modules),
            },
        )
    )
    return MegatronOperationConfig(
        run_id=request.run_id,
        training_session_id=request.training_session_id,
        adapter=spec.adapter,
        source=generation,
        initial_operation_sequence=request.initial_operation_sequence,
        optimizer_state_path=str(optimizer_state),
        rollout_model=rollout_model,
        output_adapter_root=str(output_dir / "checkpoints"),
        max_retained_inputs=request.max_retained_inputs,
    )
