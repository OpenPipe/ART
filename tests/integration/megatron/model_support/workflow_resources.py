from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class MegatronWorkflowTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    tp: int = 1
    ep: int = 1
    etp: int = 1
    dp: int = 1
    cp: int = 1
    pp: int = 1
    sp: bool = False

    def to_megatron_config(self) -> dict[str, int | None]:
        return {
            "tp": self.tp,
            "ep": self.ep,
            "etp": self.etp,
            "cp": self.cp,
            "pp": self.pp,
        }

    def to_oracle_topology_kwargs(self) -> dict[str, int | bool]:
        return self.model_dump()

    def to_train_inf_topology_kwargs(self) -> dict[str, int]:
        return {
            "tp": self.tp,
            "ep": self.ep,
            "etp": self.etp,
            "dp": self.dp,
            "cp": self.cp,
            "pp": self.pp,
        }


class MegatronWorkflowResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_ids: list[int]
    topology: MegatronWorkflowTopology


class VllmWorkflowResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu_ids: list[int]
    tensor_parallel_size: int
    enable_expert_parallel: bool = False

    def engine_args(self) -> dict[str, object]:
        engine_args: dict[str, object] = {
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.enable_expert_parallel:
            engine_args["enable_expert_parallel"] = True
        return engine_args


class WorkflowStageResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_world_size: int
    megatron: MegatronWorkflowResources | None = None
    vllm: VllmWorkflowResources | None = None


class HandlerWorkflowResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_inf_mismatch: WorkflowStageResources | None = None
    merged_vllm_serving: WorkflowStageResources | None = None
    native_vllm_lora: WorkflowStageResources | None = None
    yes_no_trainability: WorkflowStageResources | None = None
    yes_no_trainability_variant: (
        Literal[
            "megatron_shared",
            "megatron_dedicated",
            "unsloth_dedicated",
        ]
        | None
    ) = None


_DSV4_TP2_EP4 = MegatronWorkflowTopology(
    tp=2,
    ep=4,
    etp=1,
    dp=2,
    cp=1,
    pp=1,
    sp=True,
)
_DSV4_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0, 1, 2, 3],
    topology=_DSV4_TP2_EP4,
)
_DSV4_VLLM_EP4 = VllmWorkflowResources(
    gpu_ids=[4, 5, 6, 7],
    tensor_parallel_size=4,
    enable_expert_parallel=True,
)
_DSV4_NATIVE_VLLM_EP4 = VllmWorkflowResources(
    gpu_ids=[0, 1, 2, 3],
    tensor_parallel_size=4,
    enable_expert_parallel=True,
)

# Explicitly for large models which do not fit in the default topology.
HANDLER_WORKFLOW_RESOURCES: dict[str, HandlerWorkflowResources] = {
    "dsv4": HandlerWorkflowResources(
        train_inf_mismatch=WorkflowStageResources(
            required_world_size=8,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_VLLM_EP4,
        ),
        merged_vllm_serving=WorkflowStageResources(
            required_world_size=8,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_VLLM_EP4,
        ),
        native_vllm_lora=WorkflowStageResources(
            required_world_size=4,
            vllm=_DSV4_NATIVE_VLLM_EP4,
        ),
        yes_no_trainability=WorkflowStageResources(
            required_world_size=8,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_VLLM_EP4,
        ),
        yes_no_trainability_variant="megatron_dedicated",
    ),
}


def handler_workflow_resources_for_base_model(
    base_model: str,
    *,
    allow_unvalidated_arch: bool = False,
) -> HandlerWorkflowResources | None:
    from art.megatron.model_support.registry import get_model_support_spec

    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    return HANDLER_WORKFLOW_RESOURCES.get(spec.handler_key)


def validate_visible_gpu_count(
    stage_name: str,
    stage_resources: WorkflowStageResources,
    *,
    visible_gpu_count: int,
) -> None:
    if visible_gpu_count < stage_resources.required_world_size:
        raise RuntimeError(
            f"Need {stage_resources.required_world_size} visible GPUs for "
            f"{stage_name}, found {visible_gpu_count}"
        )


def validate_dedicated_test_resources(
    *,
    stage_name: str,
    trainer_gpu_ids: list[int],
    inference_gpu_ids: list[int],
    allow_overlap: bool = False,
) -> None:
    if not trainer_gpu_ids:
        raise RuntimeError(f"{stage_name} trainer GPU ids must be non-empty")
    if not inference_gpu_ids:
        raise RuntimeError(f"{stage_name} inference GPU ids must be non-empty")
    if not allow_overlap and set(trainer_gpu_ids) & set(inference_gpu_ids):
        raise RuntimeError(
            f"{stage_name} trainer and inference GPU ids must not overlap"
        )
