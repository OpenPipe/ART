from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_H200_REFERENCE_VRAM_GIB = 130.0
_H200_SLOT_TOLERANCE = 0.05
THROUGHPUT_PACKED_SEQUENCE_LENGTH = 131_072
THROUGHPUT_RANDOM_INITIALIZATION_VERSION = "deterministic_random_v1"
THROUGHPUT_RANDOM_SEED = 3407


class ThroughputThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    calibration_basis: Literal["measured", "estimated"]
    calibration_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    min_isolated_train_tok_s: float = Field(gt=0.0, allow_inf_nan=False)
    min_e2e_train_tok_s: float = Field(gt=0.0, allow_inf_nan=False)
    min_accepted_train_tok_s: float = Field(gt=0.0, allow_inf_nan=False)
    min_e2e_to_isolated_ratio: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    min_matched_core_to_isolated_ratio: float = Field(
        gt=0.0, le=1.0, allow_inf_nan=False
    )
    max_matched_core_to_isolated_ratio: float = Field(
        default=1.05, gt=1.0, allow_inf_nan=False
    )
    max_mean_policy_activation_lag_s: float = Field(gt=0.0, le=3.5, allow_inf_nan=False)
    max_policy_activation_lag_s: float = Field(gt=0.0, le=3.5, allow_inf_nan=False)
    max_repeated_policy_activation_interval_s: float = Field(
        gt=0.0, allow_inf_nan=False
    )
    max_queue_ready_inter_forward_backward_gap_p50_s: float = Field(
        default=0.23, gt=0.0, le=0.23, allow_inf_nan=False
    )
    max_queue_ready_inter_forward_backward_gap_max_s: float = Field(
        default=1.0, gt=0.0, le=1.0, allow_inf_nan=False
    )
    min_queue_ready_inter_forward_backward_gap_count: int = Field(default=3, ge=3)

    @model_validator(mode="after")
    def validate_calibration_identity(self) -> "ThroughputThresholds":
        measured = self.calibration_basis == "measured"
        if measured != (self.calibration_fingerprint is not None):
            raise ValueError(
                "measured calibration requires a fingerprint and estimated "
                "calibration must not claim one"
            )
        if self.max_mean_policy_activation_lag_s > self.max_policy_activation_lag_s:
            raise ValueError("mean activation lag limit cannot exceed absolute limit")
        return self


class ThroughputWorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_layers: int = Field(ge=2)
    prompt_tokens: int = Field(default=3839, ge=1)
    completion_tokens: int = Field(default=64, ge=1)
    rollouts_per_group: int = Field(default=4, ge=2)
    groups_per_step: int = Field(default=32, ge=2)
    initial_model_calls_per_inference_gpu: int = Field(default=32, ge=1)
    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int = Field(default=65_536, ge=1)
    enable_prefix_caching: bool = False
    max_steps: int = Field(default=13, ge=7)
    max_steps_off_policy: int = Field(default=4, ge=0)
    packed_sequence_length: int = Field(
        default=THROUGHPUT_PACKED_SEQUENCE_LENGTH, ge=1024
    )
    min_vllm_pressure: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    max_trainer_underfeed: float = Field(default=0.08, ge=0.0, allow_inf_nan=False)
    max_unused_and_dummy_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0, allow_inf_nan=False
    )
    max_queue_ready_wait_s: float = Field(
        default=0.01, ge=0.0, le=0.2, allow_inf_nan=False
    )
    random_initialization_version: Literal["deterministic_random_v1"] = (
        THROUGHPUT_RANDOM_INITIALIZATION_VERSION
    )
    random_seed: int = Field(default=THROUGHPUT_RANDOM_SEED, ge=0, le=2**31 - 1)
    thresholds: dict[Literal["h200", "b300"], ThroughputThresholds] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def require_measured_b300_calibration(self) -> "ThroughputWorkflowConfig":
        b300 = self.thresholds.get("b300")
        if b300 is not None and b300.calibration_basis != "measured":
            raise ValueError("B300 throughput thresholds must be measured")
        return self


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
    extra_engine_args: dict[str, object] = Field(default_factory=dict)

    def engine_args(self) -> dict[str, object]:
        engine_args: dict[str, object] = {
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.enable_expert_parallel:
            engine_args["enable_expert_parallel"] = True
        engine_args.update(self.extra_engine_args)
        return engine_args


class WorkflowStageResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_world_size: int
    required_physical_gpus: int | None = None
    required_h200_equivalent_gpus: int | None = None
    requires_external_vllm: bool = False
    megatron: MegatronWorkflowResources | None = None
    vllm: VllmWorkflowResources | None = None
    high_vram_megatron: MegatronWorkflowResources | None = None
    high_vram_vllm: VllmWorkflowResources | None = None
    streaming_weight_offload: bool = False
    megatron_env: dict[str, str] = Field(default_factory=dict)
    throughput: ThroughputWorkflowConfig | None = None


class HandlerWorkflowResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_inf_mismatch: WorkflowStageResources | None = None
    yes_no_trainability: WorkflowStageResources | None = None
    length_trainability: WorkflowStageResources | None = None
    e2e_throughput: WorkflowStageResources | None = None
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
_DSV4_COMMON_VLLM_ENGINE_ARGS = {
    "compilation_config": {
        "cudagraph_mode": "NONE",
        "pass_config": {"fuse_allreduce_rms": False},
    },
    "disable_custom_all_reduce": True,
    "enforce_eager": True,
    "gpu_memory_utilization": 0.82,
    "kv_cache_dtype": "fp8",
    "max_model_len": 1024,
    "max_num_batched_tokens": 1032,
}
_DSV4_VLLM_ENGINE_ARGS = {
    **_DSV4_COMMON_VLLM_ENGINE_ARGS,
    "moe_backend": "auto",
}
_DSV4_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    topology=_DSV4_TP2_EP8,
)
_DSV4_FOUR_GPU_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0, 1, 2, 3],
    topology=_DSV4_TP2_EP4,
)
_DSV4_FULL_VLLM_EP4 = VllmWorkflowResources(
    gpu_ids=[4, 5, 6, 7],
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    extra_engine_args=_DSV4_VLLM_ENGINE_ARGS,
)
_DSV4_FULL_VLLM_EP2 = VllmWorkflowResources(
    gpu_ids=[2, 3],
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    extra_engine_args=_DSV4_VLLM_ENGINE_ARGS,
)
_DSV4_FUNCTIONAL_RESOURCES = WorkflowStageResources(
    required_world_size=8,
    required_h200_equivalent_gpus=8,
    requires_external_vllm=True,
    megatron=_DSV4_MEGATRON,
    vllm=_DSV4_FULL_VLLM_EP4,
    high_vram_megatron=_DSV4_FOUR_GPU_MEGATRON,
    high_vram_vllm=_DSV4_FULL_VLLM_EP2,
    streaming_weight_offload=True,
)
_GLM52_REDUCED_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0],
    topology=MegatronWorkflowTopology(),
)
_GLM52_REDUCED_VLLM = VllmWorkflowResources(
    gpu_ids=[1],
    tensor_parallel_size=1,
    # The reduced fixture is narrower than the production model. FlashMLA covers
    # its sparse attention shape while Triton avoids absent SM100 E=4 MoE tuning.
    extra_engine_args={
        "attention_backend": "FLASHMLA_SPARSE",
        "max_model_len": 1024,
        "moe_backend": "triton",
    },
)
_GLM52_FUNCTIONAL_RESOURCES = WorkflowStageResources(
    required_world_size=2,
    megatron=_GLM52_REDUCED_MEGATRON,
    vllm=_GLM52_REDUCED_VLLM,
)
# Explicitly for large models which do not fit in the default topology.
HANDLER_WORKFLOW_RESOURCES: dict[str, HandlerWorkflowResources] = {
    "dsv4": HandlerWorkflowResources(
        train_inf_mismatch=_DSV4_FUNCTIONAL_RESOURCES,
        yes_no_trainability=_DSV4_FUNCTIONAL_RESOURCES,
        length_trainability=_DSV4_FUNCTIONAL_RESOURCES,
        yes_no_trainability_variant="megatron_dedicated",
    ),
    "glm52": HandlerWorkflowResources(
        train_inf_mismatch=_GLM52_FUNCTIONAL_RESOURCES,
        yes_no_trainability=_GLM52_FUNCTIONAL_RESOURCES,
        length_trainability=_GLM52_FUNCTIONAL_RESOURCES,
        yes_no_trainability_variant="megatron_dedicated",
    ),
    "gpt_oss_moe": HandlerWorkflowResources(
        train_inf_mismatch=WorkflowStageResources(
            required_world_size=3,
            required_physical_gpus=3,
            megatron=MegatronWorkflowResources(
                gpu_ids=[0, 1],
                topology=MegatronWorkflowTopology(cp=2, ep=2),
            ),
            vllm=VllmWorkflowResources(
                gpu_ids=[2],
                tensor_parallel_size=1,
            ),
        ),
    ),
}

_THROUGHPUT_CONFIGS = {
    "llama3_dense": ThroughputWorkflowConfig(
        num_layers=24,
        prompt_tokens=3922,
        completion_tokens=256,
        rollouts_per_group=6,
        groups_per_step=23,
        initial_model_calls_per_inference_gpu=26,
        enable_prefix_caching=True,
    ),
    "qwen3_dense": ThroughputWorkflowConfig(
        num_layers=8,
        completion_tokens=144,
        rollouts_per_group=8,
        groups_per_step=25,
        initial_model_calls_per_inference_gpu=10,
    ),
    "qwen3_moe": ThroughputWorkflowConfig(
        num_layers=16,
        prompt_tokens=3884,
        completion_tokens=48,
        rollouts_per_group=5,
        groups_per_step=27,
        initial_model_calls_per_inference_gpu=20,
        max_steps=17,
    ),
    "qwen3_5_dense": ThroughputWorkflowConfig(
        num_layers=8,
        prompt_tokens=3839,
        completion_tokens=64,
        groups_per_step=31,
        initial_model_calls_per_inference_gpu=12,
        enable_prefix_caching=True,
    ),
    "qwen3_5_moe": ThroughputWorkflowConfig(
        num_layers=24,
        prompt_tokens=7600,
        completion_tokens=16,
        groups_per_step=17,
        initial_model_calls_per_inference_gpu=12,
        max_num_batched_tokens=THROUGHPUT_PACKED_SEQUENCE_LENGTH,
        enable_prefix_caching=True,
    ),
    "gemma4_dense": ThroughputWorkflowConfig(
        num_layers=12,
        completion_tokens=75,
        rollouts_per_group=7,
        groups_per_step=30,
        initial_model_calls_per_inference_gpu=11,
    ),
    "gemma4_moe": ThroughputWorkflowConfig(
        num_layers=12,
        prompt_tokens=3640,
        completion_tokens=128,
        groups_per_step=31,
        initial_model_calls_per_inference_gpu=26,
    ),
    "dsv4": ThroughputWorkflowConfig(
        num_layers=8,
        packed_sequence_length=32_768,
        prompt_tokens=14_651,
        completion_tokens=84,
        rollouts_per_group=20,
        groups_per_step=4,
        initial_model_calls_per_inference_gpu=6,
        max_num_seqs=80,
        max_num_batched_tokens=131_072,
        enable_prefix_caching=True,
    ),
    "glm52": ThroughputWorkflowConfig(
        num_layers=12,
        prompt_tokens=3836,
        completion_tokens=1024,
        groups_per_step=16,
        initial_model_calls_per_inference_gpu=19,
    ),
    "gpt_oss_moe": ThroughputWorkflowConfig(
        num_layers=4,
        initial_model_calls_per_inference_gpu=23,
        max_num_seqs=48,
        max_steps=21,
    ),
    "nemotron_h_moe": ThroughputWorkflowConfig(
        num_layers=39,
        prompt_tokens=7900,
        completion_tokens=48,
        rollouts_per_group=4,
        groups_per_step=16,
        initial_model_calls_per_inference_gpu=20,
        max_num_batched_tokens=THROUGHPUT_PACKED_SEQUENCE_LENGTH,
        max_steps=17,
    ),
}

# Floors are isolated tok/s, E2E tok/s, accepted tok/s, E2E/isolated, and
# maximum repeated policy-activation interval. B300 values are measured; H200
# values are estimates from the prior H200 workflow and remain fingerprint-free.
_B300_THROUGHPUT_FLOORS = {
    "llama3_dense": (
        "b777d6c00d6574a9445b5a460f36909ba48155355e51034867aa286be171894d",
        (38_500, 37_300, 10_500, 0.93, 4.5),
    ),
    "qwen3_dense": (
        "fde06e40ef5a363a7910b349b5364dc84992a6aa31c7b1267d63d870fd57fd69",
        (40_200, 37_600, 8_600, 0.88, 4.5),
    ),
    "qwen3_moe": (
        "d41841a7ff6d0fcca3fe9f3ce240519143da1a1e7931fc313a88b11734535a62",
        (49_900, 43_700, 2_050, 0.82, 4.5),
    ),
    "qwen3_5_dense": (
        "5617e8880591545a3281ff14d1fe5197eeefc21a81ec80d1a107fd31421d37a0",
        (64_800, 60_000, 3_750, 0.87, 3.5),
    ),
    "qwen3_5_moe": (
        "72172ac8d112af1dd7248f52e36ff5cf4cd6c2407d4a6ffb50e2b4758e8bb98d",
        (32_600, 30_800, 257, 0.89, 5.5),
    ),
    "gemma4_dense": (
        "05fc46053854bd510487296cc6923cce846f8fb3e7b57bd67ac00848625c1a78",
        (23_100, 22_700, 2_390, 0.93, 7.0),
    ),
    "gemma4_moe": (
        "23f9679170045207c3e85dbd4496cc67a14f136f769171e18ba463788c730ac8",
        (40_300, 38_500, 4_740, 0.90, 5.0),
    ),
    "dsv4": (
        "8f947ec5b5d3237ad4b6a94f8ac0333b7f486f9c2bdff2b0e014f4db3b440854",
        (14_800, 14_300, 1_300, 0.94, 6.0),
    ),
    "glm52": (
        "bf81b6800b9ea080514da72e5e3a989e72fe523a6cdde9cbe9271eaf162c0f07",
        (14_880, 14_330, 5_730, 0.91, 12.0),
    ),
    "gpt_oss_moe": (
        "79572974803f721f65252935f0f53739e3e37110790f6957c6dd6b305a5f0689",
        (81_700, 76_400, 4_850, 0.88, 2.5),
    ),
    "nemotron_h_moe": (
        "8c5a0f125a8bd8365b68753bd933c9888661cd956c07cc8e4a1e52e94be85216",
        (40_000, 34_000, 800, 0.80, 4.5),
    ),
}
_B300_THROUGHPUT_FINGERPRINT_OVERRIDES = {
    "Qwen/Qwen3.8-27B": "b07ee7ec6338ec021463a43a90fc96c5c5a036b4a04d90b80e1d22c1eef86774",
}
_H200_THROUGHPUT_FLOORS = {
    "llama3_dense": (18_300, 17_200, 4_400, 0.89, 7.0),
    "qwen3_dense": (24_100, 23_100, 5_000, 0.91, 7.0),
    "qwen3_moe": (26_400, 20_900, 930, 0.74, 10.0),
    "qwen3_5_dense": (26_500, 25_600, 1_500, 0.91, 5.5),
    "qwen3_5_moe": (13_600, 12_900, 100, 0.90, 12.0),
    "gemma4_dense": (10_600, 10_400, 1_000, 0.93, 13.0),
    "gemma4_moe": (17_900, 17_300, 2_000, 0.91, 9.5),
    "dsv4": (8_300, 8_000, 700, 0.90, 12.0),
    "glm52": (9_400, 9_000, 3_400, 0.91, 19.5),
    "gpt_oss_moe": (39_900, 37_100, 2_200, 0.88, 4.5),
    "nemotron_h_moe": (20_000, 17_000, 400, 0.78, 8.0),
}


def _throughput_threshold(
    calibration_basis: Literal["measured", "estimated"],
    floor: tuple[float, float, float, float, float],
    *,
    calibration_fingerprint: str | None = None,
    max_mean_policy_activation_lag_s: float = 1.5,
) -> ThroughputThresholds:
    isolated, e2e, accepted, ratio, cadence = floor
    return ThroughputThresholds(
        calibration_basis=calibration_basis,
        calibration_fingerprint=calibration_fingerprint,
        min_isolated_train_tok_s=isolated,
        min_e2e_train_tok_s=e2e,
        min_accepted_train_tok_s=accepted,
        min_e2e_to_isolated_ratio=ratio,
        min_matched_core_to_isolated_ratio=0.95,
        max_mean_policy_activation_lag_s=max_mean_policy_activation_lag_s,
        max_policy_activation_lag_s=3.5,
        max_repeated_policy_activation_interval_s=cadence,
    )


for _model_key, (_fingerprint, _b300_floor) in _B300_THROUGHPUT_FLOORS.items():
    _max_mean_activation_lag_s = 2.25 if _model_key == "dsv4" else 1.5
    _THROUGHPUT_CONFIGS[_model_key] = _THROUGHPUT_CONFIGS[_model_key].model_copy(
        update={
            "thresholds": {
                "b300": _throughput_threshold(
                    "measured",
                    _b300_floor,
                    calibration_fingerprint=_fingerprint,
                    max_mean_policy_activation_lag_s=_max_mean_activation_lag_s,
                ),
                "h200": _throughput_threshold(
                    "estimated",
                    _H200_THROUGHPUT_FLOORS[_model_key],
                    max_mean_policy_activation_lag_s=_max_mean_activation_lag_s,
                ),
            }
        }
    )

_DENSE_HANDLER_KEYS = {
    "llama3_dense",
    "qwen3_dense",
    "qwen3_5_dense",
    "gemma4_dense",
}


def _throughput_stage_resources(model_key: str) -> WorkflowStageResources:
    config = _THROUGHPUT_CONFIGS[model_key]
    is_moe = model_key not in _DENSE_HANDLER_KEYS
    vllm_engine_args: dict[str, object] = {
        "disable_custom_all_reduce": True,
        "load_format": "dummy",
        "gpu_memory_utilization": 0.82,
        "max_model_len": 16_384,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "max_num_seqs": config.max_num_seqs,
        "lora_dtype": "bfloat16",
    }
    if model_key in {"qwen3_moe", "qwen3_5_moe"}:
        vllm_engine_args["compilation_config"] = {
            "pass_config": {"fuse_allreduce_rms": False}
        }
    if config.enable_prefix_caching:
        vllm_engine_args["enable_prefix_caching"] = True
    if model_key == "dsv4":
        vllm_engine_args.update(
            compilation_config={
                "cudagraph_mode": "NONE",
                "pass_config": {"fuse_allreduce_rms": False},
            },
            enforce_eager=True,
            kv_cache_dtype="fp8",
        )
    return WorkflowStageResources(
        required_world_size=4,
        required_physical_gpus=4,
        megatron=MegatronWorkflowResources(
            gpu_ids=[0, 1],
            topology=MegatronWorkflowTopology(
                cp=1 if model_key == "dsv4" else 2,
                ep=2 if is_moe else 1,
            ),
        ),
        vllm=VllmWorkflowResources(
            gpu_ids=[2, 3],
            tensor_parallel_size=2,
            enable_expert_parallel=is_moe,
            extra_engine_args=vllm_engine_args,
        ),
        throughput=config,
    )


for _model_key in _THROUGHPUT_CONFIGS:
    _resources = HANDLER_WORKFLOW_RESOURCES.get(_model_key, HandlerWorkflowResources())
    HANDLER_WORKFLOW_RESOURCES[_model_key] = _resources.model_copy(
        update={"e2e_throughput": _throughput_stage_resources(_model_key)}
    )


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
    resources = HANDLER_WORKFLOW_RESOURCES.get(spec.handler_key)
    fingerprint = _B300_THROUGHPUT_FINGERPRINT_OVERRIDES.get(base_model)
    if resources is None or resources.e2e_throughput is None or fingerprint is None:
        return resources
    stage = resources.e2e_throughput
    config = stage.throughput
    if config is None:
        raise RuntimeError(f"missing throughput config for {base_model}")
    thresholds = dict(config.thresholds)
    thresholds["b300"] = thresholds["b300"].model_copy(
        update={"calibration_fingerprint": fingerprint}
    )
    config = config.model_copy(update={"thresholds": thresholds})
    return resources.model_copy(
        update={"e2e_throughput": stage.model_copy(update={"throughput": config})}
    )


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
    required_physical = stage_resources.required_physical_gpus
    if required_physical is not None and visible_gpu_count < required_physical:
        raise RuntimeError(
            f"Need {required_physical} physical GPUs for {stage_name}, found "
            f"{visible_gpu_count}; H200-equivalent capacity cannot coalesce "
            "distinct workflow roles."
        )
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
    megatron = stage_resources.high_vram_megatron
    vllm = stage_resources.high_vram_vllm
    if megatron is None and vllm is None:
        raise RuntimeError(
            f"Need {stage_resources.required_world_size} visible GPUs for "
            f"{stage_name}, found {visible_gpu_count}. No high-VRAM resource "
            "override is configured for this stage."
        )
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
    return stage_resources.model_copy(
        update={
            "megatron": megatron or stage_resources.megatron,
            "vllm": vllm or stage_resources.vllm,
        }
    )


def _current_visible_gpu_count() -> int:
    try:
        import torch
    except ImportError:
        return 0
    return int(torch.cuda.device_count())


def resolve_stage_resources_for_current_host(
    stage_name: str,
    stage_resources: WorkflowStageResources,
) -> WorkflowStageResources:
    return resolve_stage_resources_for_visible_gpus(
        stage_name,
        stage_resources,
        visible_gpu_count=_current_visible_gpu_count(),
    )
