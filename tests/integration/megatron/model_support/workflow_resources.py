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
    max_mean_policy_activation_lag_s: float = Field(gt=0.0, le=1.5, allow_inf_nan=False)
    max_policy_activation_lag_s: float = Field(gt=0.0, le=3.5, allow_inf_nan=False)
    max_repeated_policy_activation_interval_s: float = Field(
        gt=0.0, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_calibration_identity(self) -> "ThroughputThresholds":
        measured = self.calibration_basis == "measured"
        if measured != (self.calibration_fingerprint is not None):
            raise ValueError(
                "measured calibration requires a fingerprint and estimated "
                "calibration must not claim one"
            )
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
    max_steps: int = Field(default=31, ge=11)
    max_steps_off_policy: int = Field(default=4, ge=0)
    packed_sequence_length: Literal[131072] = THROUGHPUT_PACKED_SEQUENCE_LENGTH
    min_vllm_pressure: float = Field(default=0.5, ge=0.0, allow_inf_nan=False)
    max_trainer_underfeed: float = Field(default=0.08, ge=0.0, allow_inf_nan=False)
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
    required_physical_gpus: int | None = None
    required_h200_equivalent_gpus: int | None = None
    allow_gpu_overlap: bool = False
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
    merged_vllm_serving: WorkflowStageResources | None = None
    native_vllm_lora: WorkflowStageResources | None = None
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
_DSV4_REPRESENTATIVE_MLP_LAYER_TYPES = ["moe"] * 4
_DSV4_MEGATRON_ENV = {
    "ART_DSV4_VALIDATION_NUM_LAYERS": str(_DSV4_REPRESENTATIVE_NUM_LAYERS)
}
_DSV4_HF_OVERRIDES = {
    "num_hidden_layers": _DSV4_REPRESENTATIVE_NUM_LAYERS,
    "num_hash_layers": 0,
    # Keep DSV4's required FP8 linear path, but avoid the public checkpoint's
    # MXFP4 experts, which cannot represent the reduced BF16 trainer fixture.
    "expert_dtype": "fp8",
    "compress_ratios": _DSV4_REPRESENTATIVE_COMPRESS_RATIOS,
    "layer_types": _DSV4_REPRESENTATIVE_LAYER_TYPES,
    "mlp_layer_types": _DSV4_REPRESENTATIVE_MLP_LAYER_TYPES,
    "rope_parameters": {
        "partial_rotary_factor": 0.125,
        "rope_theta": 10000,
        "rope_type": "default",
    },
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
    "max_model_len": 1024,
    "max_num_batched_tokens": 1032,
}
_DSV4_MERGED_VLLM_ENGINE_ARGS = {
    **_DSV4_COMMON_VLLM_ENGINE_ARGS,
    "moe_backend": "triton",
}
_DSV4_LORA_VLLM_ENGINE_ARGS = {
    **_DSV4_COMMON_VLLM_ENGINE_ARGS,
    "moe_backend": "triton",
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
_DSV4_REDUCED_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0, 1, 2, 3],
    topology=_DSV4_TP2_EP4,
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
_GPT_OSS_REDUCED_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0],
    topology=MegatronWorkflowTopology(),
)
_GPT_OSS_REDUCED_VLLM = VllmWorkflowResources(
    gpu_ids=[1],
    tensor_parallel_size=1,
    extra_engine_args={
        "enforce_eager": True,
        "load_format": "dummy",
        "max_model_len": 1024,
    },
)
_QWEN_MOE_REDUCED_MEGATRON = MegatronWorkflowResources(
    gpu_ids=[0],
    topology=MegatronWorkflowTopology(),
)
_QWEN_MOE_REDUCED_VLLM = VllmWorkflowResources(
    gpu_ids=[1],
    tensor_parallel_size=1,
    extra_engine_args={
        "enforce_eager": True,
        "max_model_len": 1024,
        "moe_backend": "triton",
    },
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
            streaming_weight_offload=True,
        ),
        merged_vllm_serving=WorkflowStageResources(
            required_world_size=8,
            required_h200_equivalent_gpus=8,
            megatron=_DSV4_REDUCED_MEGATRON,
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
            streaming_weight_offload=True,
        ),
        length_trainability=WorkflowStageResources(
            required_world_size=8,
            required_h200_equivalent_gpus=8,
            requires_external_vllm=True,
            megatron=_DSV4_MEGATRON,
            vllm=_DSV4_FULL_VLLM_EP4,
            high_vram_megatron=_DSV4_HIGH_VRAM_MEGATRON,
            high_vram_vllm=_DSV4_FULL_VLLM_EP2,
            streaming_weight_offload=True,
        ),
        yes_no_trainability_variant="megatron_dedicated",
    ),
    "glm52": HandlerWorkflowResources(
        train_inf_mismatch=WorkflowStageResources(
            required_world_size=2,
            megatron=_GLM52_REDUCED_MEGATRON,
            vllm=_GLM52_REDUCED_VLLM,
        ),
        merged_vllm_serving=WorkflowStageResources(
            required_world_size=2,
            megatron=_GLM52_REDUCED_MEGATRON,
            vllm=_GLM52_REDUCED_VLLM,
        ),
        native_vllm_lora=WorkflowStageResources(
            required_world_size=2,
            vllm=_GLM52_REDUCED_VLLM,
        ),
        yes_no_trainability=WorkflowStageResources(
            required_world_size=2,
            megatron=_GLM52_REDUCED_MEGATRON,
            vllm=_GLM52_REDUCED_VLLM,
        ),
        length_trainability=WorkflowStageResources(
            required_world_size=2,
            megatron=_GLM52_REDUCED_MEGATRON,
            vllm=_GLM52_REDUCED_VLLM,
        ),
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
        merged_vllm_serving=WorkflowStageResources(
            required_world_size=2,
            megatron=_GPT_OSS_REDUCED_MEGATRON,
            vllm=_GPT_OSS_REDUCED_VLLM,
        ),
        native_vllm_lora=WorkflowStageResources(
            required_world_size=2,
            vllm=_GPT_OSS_REDUCED_VLLM,
        ),
    ),
    **{
        handler_key: HandlerWorkflowResources(
            merged_vllm_serving=WorkflowStageResources(
                required_world_size=2,
                megatron=_QWEN_MOE_REDUCED_MEGATRON,
                vllm=_QWEN_MOE_REDUCED_VLLM,
            ),
            native_vllm_lora=WorkflowStageResources(
                required_world_size=2,
                vllm=_QWEN_MOE_REDUCED_VLLM,
            ),
        )
        for handler_key in ("qwen3_moe", "qwen3_5_moe")
    },
}

_THROUGHPUT_CONFIGS = {
    "llama3_dense": ThroughputWorkflowConfig(
        num_layers=16,
        prompt_tokens=3922,
        completion_tokens=256,
        rollouts_per_group=6,
        groups_per_step=24,
        initial_model_calls_per_inference_gpu=20,
        max_steps=35,
    ),
    "qwen3_dense": ThroughputWorkflowConfig(
        num_layers=8,
        completion_tokens=144,
        rollouts_per_group=8,
        groups_per_step=25,
        initial_model_calls_per_inference_gpu=10,
        max_steps=35,
    ),
    "qwen3_moe": ThroughputWorkflowConfig(
        num_layers=16,
        prompt_tokens=3884,
        completion_tokens=48,
        rollouts_per_group=5,
        groups_per_step=27,
        initial_model_calls_per_inference_gpu=20,
        max_steps=35,
    ),
    "qwen3_5_dense": ThroughputWorkflowConfig(
        num_layers=8,
        prompt_tokens=3839,
        completion_tokens=64,
        groups_per_step=31,
        initial_model_calls_per_inference_gpu=12,
        max_steps=35,
        enable_prefix_caching=True,
    ),
    "qwen3_5_moe": ThroughputWorkflowConfig(
        num_layers=24,
        prompt_tokens=7600,
        completion_tokens=16,
        groups_per_step=17,
        initial_model_calls_per_inference_gpu=12,
        max_num_batched_tokens=THROUGHPUT_PACKED_SEQUENCE_LENGTH,
        max_steps=35,
        enable_prefix_caching=True,
    ),
    "gemma4_dense": ThroughputWorkflowConfig(
        num_layers=12,
        completion_tokens=75,
        rollouts_per_group=7,
        groups_per_step=30,
        initial_model_calls_per_inference_gpu=11,
        max_steps=35,
    ),
    "gemma4_moe": ThroughputWorkflowConfig(
        num_layers=12,
        prompt_tokens=3640,
        completion_tokens=128,
        groups_per_step=31,
        initial_model_calls_per_inference_gpu=26,
        max_steps=35,
    ),
    "dsv4": ThroughputWorkflowConfig(
        num_layers=8,
        prompt_tokens=12_800,
        completion_tokens=736,
        groups_per_step=8,
        initial_model_calls_per_inference_gpu=20,
        max_num_seqs=60,
        max_num_batched_tokens=24_576,
    ),
    "glm52": ThroughputWorkflowConfig(
        num_layers=12,
        prompt_tokens=3836,
        completion_tokens=640,
        groups_per_step=20,
        initial_model_calls_per_inference_gpu=19,
    ),
    "gpt_oss_moe": ThroughputWorkflowConfig(
        num_layers=4,
        initial_model_calls_per_inference_gpu=21,
        max_num_seqs=48,
        max_steps=35,
    ),
}

# Floors are isolated tok/s, E2E tok/s, accepted tok/s, E2E/isolated, and
# maximum repeated policy-activation interval. B300 values are measured; H200
# values are estimates from the prior H200 workflow and remain fingerprint-free.
_B300_THROUGHPUT_FLOORS = {
    "llama3_dense": (
        "421297e73e35aacdd0f2dc321dec0548adf354789840debd1b2b5b546d3bd7ba",
        (49_500, 47_300, 12_900, 0.90, 4.5),
    ),
    "qwen3_dense": (
        "1dc5ba0da1547fb17535549ccdeabc9b18a8d0a70bbcf187f4a0ff7957ed5135",
        (40_200, 37_600, 8_600, 0.88, 4.5),
    ),
    "qwen3_moe": (
        "999e014167fbe2b8570d124e5ea38450104041d5f735a08433616cb9d19d91c8",
        (49_900, 43_700, 2_050, 0.82, 4.5),
    ),
    "qwen3_5_dense": (
        "b2c0a9395ad09d3821082a4aac0337a5be5e9b3580953ece61079785d4b158d4",
        (64_800, 60_000, 3_750, 0.87, 3.5),
    ),
    "qwen3_5_moe": (
        "38475e5ec324f9fcf21cb0da0e202247814ba317c7f2dbb9b6c1646c8732b761",
        (32_600, 30_800, 257, 0.89, 5.5),
    ),
    "gemma4_dense": (
        "74228dd6f0d2c4872b0b68ba90f4f72a99aace2c622b4adb1b47a5260416c374",
        (23_100, 22_700, 2_390, 0.93, 7.0),
    ),
    "gemma4_moe": (
        "934a6106a961eb1eedc0f50059f389b223b3b7a5538ab10a7086d44ac440bd10",
        (40_300, 38_500, 4_740, 0.90, 5.0),
    ),
    "dsv4": (
        "94cc2ad031210a54ad0f46a83d8fe277eae9e77068acc91055b5cd420ff77901",
        (7_050, 7_020, 1_350, 0.94, 43.0),
    ),
    "glm52": (
        "702f7d8f96ba8c4501c6e2e386e5c9b95d11fe1c3ab760bd5c9df9600572627e",
        (14_880, 14_330, 5_730, 0.91, 12.0),
    ),
    "gpt_oss_moe": (
        "c69902095a0b42735d59ad29a8b098483966f1677cc27ec3a4b4be2375d05a2d",
        (81_700, 76_400, 4_850, 0.88, 2.5),
    ),
}
_H200_THROUGHPUT_FLOORS = {
    "llama3_dense": (27_500, 25_900, 6_600, 0.89, 7.0),
    "qwen3_dense": (24_100, 23_100, 5_000, 0.91, 7.0),
    "qwen3_moe": (26_400, 20_900, 930, 0.74, 10.0),
    "qwen3_5_dense": (26_500, 25_600, 1_500, 0.91, 5.5),
    "qwen3_5_moe": (13_600, 12_900, 100, 0.90, 12.0),
    "gemma4_dense": (10_600, 10_400, 1_000, 0.93, 13.0),
    "gemma4_moe": (17_900, 17_300, 2_000, 0.91, 9.5),
    "dsv4": (3_500, 3_400, 620, 0.94, 80.0),
    "glm52": (9_400, 9_000, 3_400, 0.91, 19.5),
    "gpt_oss_moe": (39_900, 37_100, 2_200, 0.88, 4.5),
}


def _throughput_threshold(
    calibration_basis: Literal["measured", "estimated"],
    floor: tuple[float, float, float, float, float],
    *,
    calibration_fingerprint: str | None = None,
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
        max_mean_policy_activation_lag_s=1.5,
        max_policy_activation_lag_s=3.5,
        max_repeated_policy_activation_interval_s=cadence,
    )


for _model_key, (_fingerprint, _b300_floor) in _B300_THROUGHPUT_FLOORS.items():
    _THROUGHPUT_CONFIGS[_model_key] = _THROUGHPUT_CONFIGS[_model_key].model_copy(
        update={
            "thresholds": {
                "b300": _throughput_threshold(
                    "measured",
                    _b300_floor,
                    calibration_fingerprint=_fingerprint,
                ),
                "h200": _throughput_threshold(
                    "estimated", _H200_THROUGHPUT_FLOORS[_model_key]
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
