from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art import dev
from art.distributed import ArtLaunchContext, ArtRuntime, ArtRuntimeConfig
from art.distributed.specs import RuntimeTopology
from art.types import MegatronRuntimeConfig
from art.vllm_route_transport import RouteBundleReader

from .runtime.build import build_trainer_runtime_spec
from .runtime_config import init_megatron_runtime_config
from .slot_coordinator import (
    MegatronSlotCoordinator,
    MegatronSlotResourceManager,
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


@dataclass(slots=True)
class MegatronSlotRuntime:
    """Lifecycle owner for one ART runtime and its shared trainer coordinator."""

    runtime: ArtRuntime
    coordinator: MegatronSlotCoordinator
    descriptor: MegatronSlotRuntimeDescriptor

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
    resources: MegatronSlotResourceManager | None = None,
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
        )
        trainer = await runtime.start_shared_trainer(
            runtime_spec,
            launch_id=config.slot_id,
            command_timeout_s=config.command_timeout_s,
            shutdown_timeout_s=config.shutdown_timeout_s,
        )
        coordinator = MegatronSlotCoordinator(
            runtime,
            trainer,
            resources=resources,
            schedule=config.schedule,
        )
        runtime.register_closeable(coordinator)
    except BaseException:
        await runtime.close()
        raise
    return MegatronSlotRuntime(
        runtime=runtime,
        coordinator=coordinator,
        descriptor=MegatronSlotRuntimeDescriptor(
            runtime_source_id=config.slot_id,
            runtime_source_epoch=config.runtime_source_epoch,
            runtime_fingerprint=runtime_spec.fingerprint,
        ),
    )
