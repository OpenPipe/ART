from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from art import dev
from art.distributed.art_runtime import ArtRuntime
from art.distributed.rollout import RolloutModelSpec
from art.model import TrainableModel
from art.types import MegatronRuntimeConfig

from ..runtime.build import build_trainer_runtime_spec
from ..runtime.local import compile_local_training_topology
from ..runtime_config import init_megatron_runtime_config
from .slot import MegatronTrainingSlot


class LocalMegatronTrainingSlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str = Field(min_length=1)
    artifact_root: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    model_identifier: str | None = Field(default=None, min_length=1)
    model_revision: str | None = Field(default=None, min_length=1)
    trainer_gpu_ids: tuple[int, ...] = Field(min_length=1)
    runtime: MegatronRuntimeConfig
    lora_rank: int | None = Field(default=None, ge=1)
    lora_alpha: float = Field(default=32.0, gt=0)
    lora_target_modules: tuple[str, ...] = ()
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    model_initialization: Literal["pretrained", "random"] = "pretrained"
    random_state: int | None = None
    allow_unvalidated_arch: bool = False
    enable_moe_routing_replay: bool = True
    cache_root: str | None = Field(default=None, min_length=1)
    startup_timeout_s: float = Field(default=600.0, gt=0)
    rpc_timeout_s: float = Field(default=60.0, gt=0)
    command_timeout_s: float = Field(default=300.0, gt=0)
    shutdown_timeout_s: float = Field(default=240.0, gt=0)
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    chat_template_tool_schema_format: Literal["default", "vllm_openai"] = "default"

    @model_validator(mode="after")
    def _validate_lora(self) -> "LocalMegatronTrainingSlotConfig":
        if self.lora_alpha != 32.0:
            raise ValueError("current Megatron LoRA semantics require lora_alpha=32")
        if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
            raise ValueError("lora_target_modules must be unique")
        return self


class LocalMegatronTrainingSlot:
    """Own one local training-only ArtRuntime and persistent Megatron slot."""

    def __init__(
        self,
        *,
        runtime: ArtRuntime,
        slot: MegatronTrainingSlot,
        model: RolloutModelSpec,
    ) -> None:
        self.runtime = runtime
        self.slot = slot
        self.model = model
        self._closed = False

    @classmethod
    async def start(
        cls, config: LocalMegatronTrainingSlotConfig
    ) -> "LocalMegatronTrainingSlot":
        import torch

        runtime_config = init_megatron_runtime_config(config.runtime)
        artifact_root = str(Path(config.artifact_root).resolve())
        init_args: dev.InitArgs = {
            "model_name": config.model_identifier or config.base_model,
            "max_seq_length": runtime_config.packed_sequence_length,
            "dtype": config.dtype,
        }
        if config.model_revision is not None:
            init_args["revision"] = config.model_revision
        if config.random_state is not None:
            init_args["random_state"] = config.random_state
        lora_config: dev.LoRAConfig = {"alpha": int(config.lora_alpha)}
        if config.lora_rank is not None:
            lora_config["rank"] = config.lora_rank
        if config.lora_target_modules:
            lora_config["target_modules"] = list(config.lora_target_modules)
        internal_config = cast(
            dev.BackendModelConfig,
            {
                "init_args": init_args,
                "lora_config": lora_config,
                "trainer_gpu_ids": list(config.trainer_gpu_ids),
                "megatron_model_initialization": config.model_initialization,
                "allow_unvalidated_arch": config.allow_unvalidated_arch,
                "chat_template_kwargs": config.chat_template_kwargs,
                "chat_template_tool_schema_format": (
                    config.chat_template_tool_schema_format
                ),
                **(
                    {"chat_template": config.chat_template}
                    if config.chat_template is not None
                    else {}
                ),
            },
        )
        model = TrainableModel(
            name=config.slot_id,
            run_name=config.slot_id,
            project="remote-training",
            base_model=config.base_model,
            lora_config=lora_config,
            _internal_config=internal_config,
            report_metrics=[],
        )
        topology = compile_local_training_topology(
            artifact_root=artifact_root,
            trainer_gpu_ids=config.trainer_gpu_ids,
            visible_gpu_count=int(torch.cuda.device_count()),
            cache_root=config.cache_root,
            startup_timeout_s=config.startup_timeout_s,
            rpc_timeout_s=config.rpc_timeout_s,
        )
        runtime = await ArtRuntime.start_local(topology)
        try:
            runtime_spec = build_trainer_runtime_spec(
                runtime,
                base_model=config.base_model,
                config=internal_config,
                enable_expert_replay=config.enable_moe_routing_replay,
                offload_between_jobs=False,
            )
            trainer = await runtime.start_trainer_slot(
                runtime_spec,
                slot_id=config.slot_id,
                command_timeout_s=config.command_timeout_s,
                shutdown_timeout_s=config.shutdown_timeout_s,
            )
            slot = MegatronTrainingSlot(
                runtime=runtime,
                trainer=trainer,
                runtime_spec=runtime_spec,
                artifact_root=artifact_root,
            )
        except BaseException:
            await runtime.close()
            raise
        return cls(runtime=runtime, slot=slot, model=RolloutModelSpec.from_model(model))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        primary: BaseException | None = None
        try:
            await self.slot.close()
        except BaseException as error:
            primary = error
        try:
            await self.runtime.close()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(
                    f"runtime close also failed: {type(error).__name__}: {error}"
                )
        if primary is not None:
            raise primary

    async def __aenter__(self) -> "LocalMegatronTrainingSlot":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
