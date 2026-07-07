from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_H200_REFERENCE_VRAM_GIB = 140.0
_H200_SLOT_TOLERANCE = 0.05


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
    hf_overrides: dict[str, object] = Field(default_factory=dict)
    extra_engine_args: dict[str, object] = Field(default_factory=dict)

    def engine_args(self) -> dict[str, object]:
        engine_args: dict[str, object] = {
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.enable_expert_parallel:
            engine_args["enable_expert_parallel"] = True
        if self.hf_overrides:
            engine_args["hf_overrides"] = dict(self.hf_overrides)
        engine_args.update(self.extra_engine_args)
        return engine_args


class WorkflowStageResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_world_size: int
    required_h200_equivalent_gpus: int | None = None
    allow_gpu_overlap: bool = False
    requires_external_vllm: bool = False
    megatron: MegatronWorkflowResources | None = None
    vllm: VllmWorkflowResources | None = None
    high_vram_megatron: MegatronWorkflowResources | None = None
    high_vram_vllm: VllmWorkflowResources | None = None
    megatron_env: dict[str, str] = Field(default_factory=dict)


class HandlerWorkflowResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_inf_mismatch: WorkflowStageResources | None = None
    merged_vllm_serving: WorkflowStageResources | None = None
    native_vllm_lora: WorkflowStageResources | None = None
    yes_no_trainability: WorkflowStageResources | None = None
    length_trainability: WorkflowStageResources | None = None
    yes_no_trainability_variant: (
        Literal[
            "megatron_shared",
            "megatron_dedicated",
            "unsloth_dedicated",
        ]
        | None
    ) = None


_DSV4_TP2_EP8 = MegatronWorkflowTopology(
    tp=2,
    ep=8,
    etp=1,
    dp=4,
    cp=1,
    pp=1,
    sp=True,
)
_DSV4_TP2_EP4 = MegatronWorkflowTopology(
    tp=2,
    ep=4,
    etp=1,
    dp=2,
    cp=1,
    pp=1,
    sp=True,
)
_DSV4_TP2_EP2 = MegatronWorkflowTopology(
    tp=2,
    ep=2,
    etp=1,
    dp=1,
    cp=1,
    pp=1,
    sp=True,
)
_DSV4_REPRESENTATIVE_NUM_LAYERS = 4
_DSV4_REPRESENTATIVE_COMPRESS_RATIOS = [0, 0, 4, 128]
_DSV4_REPRESENTATIVE_LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
]
_DSV4_REPRESENTATIVE_MLP_LAYER_TYPES = ["hash_moe", "hash_moe", "hash_moe", "moe"]
_DSV4_MEGATRON_ENV = {
    "ART_DSV4_VALIDATION_NUM_LAYERS": str(_DSV4_REPRESENTATIVE_NUM_LAYERS)
}
_DSV4_STREAMING_OFFLOAD_ENV = {"ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD": "1"}
_DSV4_HF_OVERRIDES = {
    "num_hidden_layers": _DSV4_REPRESENTATIVE_NUM_LAYERS,
    "compress_ratios": _DSV4_REPRESENTATIVE_COMPRESS_RATIOS,
    "layer_types": _DSV4_REPRESENTATIVE_LAYER_TYPES,
    "mlp_layer_types": _DSV4_REPRESENTATIVE_MLP_LAYER_TYPES,
}
_DSV4_COMMON_VLLM_ENGINE_ARGS = {
    "compilation_config": {
        "cudagraph_mode": "NONE",
        "pass_config": {"fuse_allreduce_rms": False},
    },
    "disable_custom_all_reduce": True,
    "enforce_eager": True,
    "gpu_memory_utilization": 0.82,
    "kv_cache_dtype": "fp8",
    "max_num_batched_tokens": 1032,
}
_DSV4_MERGED_VLLM_ENGINE_ARGS = {
    **_DSV4_COMMON_VLLM_ENGINE_ARGS,
    "moe_backend": "triton_unfused",
}
_DSV4_LORA_VLLM_ENGINE_ARGS = {
    **_DSV4_COMMON_VLLM_ENGINE_ARGS,
    "moe_backend": "triton_unfused",
}
_DSV4_REDUCED_VLLM_ENGINE_ARGS = {
    **_DSV4_MERGED_VLLM_ENGINE_ARGS,
    # The quick DSV4 vLLM serving gates use a reduced 4-layer validation model and then
    # sync Megatron weights into vLLM through merged-weight transfer. Loading
    # the full public checkpoint before that sync is incompatible with the
    # reduced hf_overrides because vLLM still streams layer-4+ tensors.
    "load_format": "dummy",
}
_DSV4_NATIVE_LORA_VLLM_ENGINE_ARGS = {
    **_DSV4_LORA_VLLM_ENGINE_ARGS,
    "load_format": "dummy",
}
_DSV4_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    topology=_DSV4_TP2_EP8,
)
_DSV4_HIGH_VRAM_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0, 1],
    topology=_DSV4_TP2_EP2,
)
_DSV4_FULL_VLLM_EP4 = VllmWorkflowResources(
    gpu_ids=[4, 5, 6, 7],
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    extra_engine_args=_DSV4_LORA_VLLM_ENGINE_ARGS,
)
_DSV4_FULL_VLLM_EP2 = VllmWorkflowResources(
    gpu_ids=[2, 3],
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    extra_engine_args=_DSV4_LORA_VLLM_ENGINE_ARGS,
)
_DSV4_REDUCED_VLLM_EP4 = VllmWorkflowResources(
    gpu_ids=[4, 5, 6, 7],
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    hf_overrides=_DSV4_HF_OVERRIDES,
    extra_engine_args=_DSV4_REDUCED_VLLM_ENGINE_ARGS,
)
_DSV4_REDUCED_VLLM_EP2 = VllmWorkflowResources(
    gpu_ids=[2, 3],
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    hf_overrides=_DSV4_HF_OVERRIDES,
    extra_engine_args=_DSV4_REDUCED_VLLM_ENGINE_ARGS,
)
_DSV4_REDUCED_NATIVE_VLLM_EP4 = VllmWorkflowResources(
    gpu_ids=[0, 1, 2, 3],
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    hf_overrides=_DSV4_HF_OVERRIDES,
    extra_engine_args=_DSV4_NATIVE_LORA_VLLM_ENGINE_ARGS,
)

# Explicitly for large models which do not fit in the default topology.
HANDLER_WORKFLOW_RESOURCES: dict[str, HandlerWorkflowResources] = {
    "dsv4": HandlerWorkflowResources(
        train_inf_mismatch=WorkflowStageResources(
            required_world_size=8,
            required_h200_equivalent_gpus=8,
            requires_external_vllm=True,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_FULL_VLLM_EP4,
            high_vram_megatron=_DSV4_HIGH_VRAM_MEGATRON,
            high_vram_vllm=_DSV4_FULL_VLLM_EP2,
            megatron_env=_DSV4_STREAMING_OFFLOAD_ENV,
        ),
        merged_vllm_serving=WorkflowStageResources(
            required_world_size=8,
            required_h200_equivalent_gpus=8,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_REDUCED_VLLM_EP4,
            high_vram_megatron=_DSV4_HIGH_VRAM_MEGATRON,
            high_vram_vllm=_DSV4_REDUCED_VLLM_EP2,
            megatron_env=_DSV4_MEGATRON_ENV,
        ),
        native_vllm_lora=WorkflowStageResources(
            required_world_size=4,
            vllm=_DSV4_REDUCED_NATIVE_VLLM_EP4,
        ),
        yes_no_trainability=WorkflowStageResources(
            required_world_size=8,
            required_h200_equivalent_gpus=8,
            requires_external_vllm=True,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_FULL_VLLM_EP4,
            high_vram_megatron=_DSV4_HIGH_VRAM_MEGATRON,
            high_vram_vllm=_DSV4_FULL_VLLM_EP2,
            megatron_env=_DSV4_STREAMING_OFFLOAD_ENV,
        ),
        length_trainability=WorkflowStageResources(
            required_world_size=8,
            required_h200_equivalent_gpus=8,
            requires_external_vllm=True,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_FULL_VLLM_EP4,
            high_vram_megatron=_DSV4_HIGH_VRAM_MEGATRON,
            high_vram_vllm=_DSV4_FULL_VLLM_EP2,
            megatron_env=_DSV4_STREAMING_OFFLOAD_ENV,
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


def _h200_equivalent_slots_for_total_gib(total_gib: float) -> int:
    return max(0, int(total_gib / _H200_REFERENCE_VRAM_GIB + _H200_SLOT_TOLERANCE))


def _visible_h200_equivalent_gpus(*, visible_gpu_count: int) -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    equivalent = 0
    for device_index in range(visible_gpu_count):
        props = torch.cuda.get_device_properties(device_index)
        total_gib = float(props.total_memory) / (1024**3)
        equivalent += _h200_equivalent_slots_for_total_gib(total_gib)
    return equivalent


def _remap_gpu_ids_to_visible(
    gpu_ids: list[int], *, visible_gpu_count: int
) -> list[int]:
    if all(0 <= gpu_id < visible_gpu_count for gpu_id in gpu_ids):
        return list(gpu_ids)
    if len(gpu_ids) > visible_gpu_count:
        raise RuntimeError(
            "Cannot remap workflow GPU ids to visible high-VRAM devices: "
            f"gpu_ids={gpu_ids}, visible_gpu_count={visible_gpu_count}"
        )
    return list(range(len(gpu_ids)))


def _validate_gpu_ids_visible(gpu_ids: list[int], *, visible_gpu_count: int) -> None:
    invalid = [
        gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= visible_gpu_count
    ]
    if invalid:
        raise RuntimeError(
            f"Workflow GPU ids {gpu_ids} are not visible on host with "
            f"{visible_gpu_count} GPUs"
        )


def resolve_stage_resources_for_visible_gpus(
    stage_name: str,
    stage_resources: WorkflowStageResources,
    *,
    visible_gpu_count: int,
) -> WorkflowStageResources:
    if visible_gpu_count >= stage_resources.required_world_size:
        return stage_resources
    required_equivalent = stage_resources.required_h200_equivalent_gpus
    available_equivalent = _visible_h200_equivalent_gpus(
        visible_gpu_count=visible_gpu_count
    )
    if required_equivalent is None or available_equivalent < required_equivalent:
        raise RuntimeError(
            f"Need {stage_resources.required_world_size} visible GPUs for "
            f"{stage_name}, found {visible_gpu_count}. High-VRAM remapping "
            f"requires {required_equivalent or stage_resources.required_world_size} "
            f"H200-equivalent GPUs, found {available_equivalent}."
        )
    if (
        stage_resources.high_vram_megatron is not None
        or stage_resources.high_vram_vllm is not None
    ):
        megatron = stage_resources.high_vram_megatron or stage_resources.megatron
        vllm = stage_resources.high_vram_vllm or stage_resources.vllm
        if megatron is not None:
            _validate_gpu_ids_visible(
                megatron.gpu_ids,
                visible_gpu_count=visible_gpu_count,
            )
        if vllm is not None:
            _validate_gpu_ids_visible(
                vllm.gpu_ids,
                visible_gpu_count=visible_gpu_count,
            )
        return stage_resources.model_copy(update={"megatron": megatron, "vllm": vllm})
    if not stage_resources.allow_gpu_overlap:
        raise RuntimeError(
            f"Need {stage_resources.required_world_size} visible GPUs for "
            f"{stage_name}, found {visible_gpu_count}. No high-VRAM resource "
            "override is configured for this stage."
        )
    megatron = stage_resources.megatron
    if megatron is not None:
        megatron = megatron.model_copy(
            update={
                "gpu_ids": _remap_gpu_ids_to_visible(
                    megatron.gpu_ids,
                    visible_gpu_count=visible_gpu_count,
                )
            }
        )
    vllm = stage_resources.vllm
    if vllm is not None:
        vllm = vllm.model_copy(
            update={
                "gpu_ids": _remap_gpu_ids_to_visible(
                    vllm.gpu_ids,
                    visible_gpu_count=visible_gpu_count,
                )
            }
        )
    return stage_resources.model_copy(update={"megatron": megatron, "vllm": vllm})


def resolve_stage_resources_for_current_host(
    stage_name: str,
    stage_resources: WorkflowStageResources,
) -> WorkflowStageResources:
    try:
        import torch
    except ImportError:
        visible_gpu_count = 0
    else:
        visible_gpu_count = int(torch.cuda.device_count())
    return resolve_stage_resources_for_visible_gpus(
        stage_name,
        stage_resources,
        visible_gpu_count=visible_gpu_count,
    )


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
