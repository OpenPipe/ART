# isort: off
from art.megatron.runtime.runtime_env import configure_megatron_runtime_env

configure_megatron_runtime_env()
from art.megatron.runtime.bridge_runtime import install_art_bridge_runtime_patches

install_art_bridge_runtime_patches()
# isort: on

"""Megatron training runtime and typed executor API.

Public cross-repo API consumed by serverless-training:
- build_training_runtime
- execute_megatron_rl_job
- execute_megatron_score_job
- execute_megatron_sft_job
- inspect_resident_lora
"""

from contextlib import contextmanager
import hashlib
import math
import os
import random
from threading import Event
import time
from typing import Any, Callable, Iterator, Literal, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import MegatronModule
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir as inductor_cache_dir

from art import dev, types
from art.loss import (
    AlignedLossInputs,
    LossInputs,
    LossOffPolicyDiagnostics,
    LossOffPolicyDiagnosticsAccumulator,
    compute_probs_corr,
    loss_fn,
    shift_tensor,
)
from art.megatron.context_parallel.types import (
    ParallelTopology,
    PreparedMegatronBatch,
    TrainingStepWorkload,
)
from art.megatron.lora import LoRA, apply_lora_adapters
from art.megatron.megatron_patches import install_fast_frozen_output_backward
from art.megatron.model_support.lora_disk import (
    load_adapter_config,
    load_lora_tensors_for_megatron,
)
from art.megatron.optimizer_state import (
    ALLOW_UNPAIRED_MEGATRON_RESUME_ENV,
    _model_runtime_sha256,
    load_optimizer_state,
)
from art.megatron.provider import (
    ProviderBundle,
    finalize_provider_bundle,
    prepare_provider_bundle,
)
from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    MoeRoutingReplayController,
    build_moe_routing_replay_bundle_from_packed_tensors,
    prepare_moe_routing_replay_boundaries,
)
from art.megatron.runtime.data_plane import SFTBatchData
from art.megatron.runtime.device_usage import measure_cuda_call
from art.megatron.runtime.specs import (
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    OptimizerJobSpec,
    PackedTokenScore,
    ResidentLoraExport,
    ResidentLoraInspectionShard,
    ResidentLoraInspectionSpec,
    ResidentScoreJobSpec,
    ResidentScoreShard,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    SFTJobSpec,
    TrainJobSpec,
)
from art.megatron.selective_lm_head import (
    TokenLossOutput,
    forward_token_logits,
    forward_token_losses,
    vocab_parallel_selected_logprobs,
)
from art.megatron.tensor_snapshot import SnapshotReadBarrier
from art.megatron.training.compile import configure_training_compile
from art.megatron.training.finalize_grads import (
    finalize_model_grads_extended,
    flush_param_grads_to_main_grads,
)
from art.megatron.training.microbatches import (
    CpBatchLookaheadState,
    PreparedRLMicroInputs,
    PreparedSFTMicroInputs,
    _clone_packed_tensors,
    _clone_sft_tensors,
    _count_sft_trainable_tokens,
    _local_trainable_sft_token_count_tensor,
    _local_trainable_token_count_tensor,
    _next_micro_lookahead,
    _prepare_current_rl_micro,
    _prepare_current_sft_micro,
    _prepare_dense_sft_micro,
    _prepare_next_rl_cp_micro,
    _prepare_next_sft_cp_micro,
    _select_next_step_first_micro,
    _zero_contribution_inputs,
    _zero_contribution_sft_inputs,
    build_micro_sample_indices,
    build_rl_hybridep_token_counts,
    build_sft_hybridep_token_counts,
    resolve_global_grad_accumulation_sequences,
    select_indexed_inputs,
    select_micro_inputs,
    select_sft_micro_inputs,
)
from art.megatron.training.model_chunks import (
    ModelChunks,
    as_megatron_api_chunks,
    validate_model_chunks,
)
from art.megatron.training.pipeline_schedule import (
    MCoreScheduleAdapter,
    PipelineMicrobatchState,
    ScheduleMicrobatch,
    _set_hybridep_token_count,
    _validate_hybridep_token_counts,
    chunk_post_process,
    validate_pipeline_topology,
)
from art.megatron.training.trace import (
    attach_trace_token_uids,
    context_parallel_trace_token_uids_enabled,
)
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.preprocessing.pack import PackedTensors
from art.training.contracts import LossConfig
from art.training.tokenized import tokenized_clip_bounds
from art.training.tokenized_loss import tokenized_loss

DEFAULT_MODEL_IDENTIFIER = "Qwen/Qwen3-30B-A3B-Instruct-2507"
_optimizer_stats_printed = False
_INTER_FORWARD_BACKWARD_GAP_PREFIX = "time/inter_forward_backward_gap_rank_"
_INTER_FORWARD_BACKWARD_GPU_GAP_PREFIX = "time/inter_forward_backward_gpu_gap_rank_"
_INTER_FORWARD_BACKWARD_PHASE_PREFIX = "time/inter_forward_backward_"

__all__ = [
    "DEFAULT_MODEL_IDENTIFIER",
    "TrainingRuntime",
    "build_training_runtime",
    "execute_megatron_rl_job",
    "execute_megatron_score_job",
    "execute_megatron_sft_job",
    "inspect_resident_lora",
]


class _InterForwardBackwardTiming(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    metrics_group: Any | None = None
    previous_schedule_end_s: float | None = None
    previous_schedule_cuda_end: torch.cuda.Event | None = None
    previous_job_complete_s: float | None = None
    current_job_start_s: float | None = None


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_bundle: ProviderBundle
    provider: Any
    model: ModelChunks
    optimizer: Any | None
    optimizer_config: OptimizerConfig
    optimizer_runtime_sha256: str | None = None
    optimizer_persistent: bool = True
    resident_run_id: str | None = None
    resident_training_session_id: str | None = None
    resident_policy_step: int | None = None
    resident_generation_id: str | None = None
    optimizer_state_loaded: bool = False
    adapter_export_dtypes: dict[str, torch.dtype] | None = None
    adapter_export_config: dict[str, Any] | None = None
    snapshot_pool_capacity: int = Field(default=2, ge=1, le=4)
    optimizer_snapshot_barrier: SnapshotReadBarrier = Field(
        default_factory=SnapshotReadBarrier
    )
    transformer_layers_compiled: bool = False
    rank: int
    world_size: int
    moe_routing_replay_controller: MoeRoutingReplayController | None = None
    inter_forward_backward_timing: _InterForwardBackwardTiming = Field(
        default_factory=_InterForwardBackwardTiming
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: ModelChunks) -> ModelChunks:
        validate_model_chunks(value)
        return value

    @property
    def bridge(self) -> Any:
        return self.provider_bundle.bridge

    @property
    def model_support_handler(self) -> Any:
        return self.provider_bundle.handler

    @property
    def model_support_spec(self) -> Any:
        return self.provider_bundle.spec


class TrainStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reduced_loss: torch.Tensor
    probs_corr: float
    kl_policy_ref: float | None = None
    new_logprobs: list[torch.Tensor] | None = None
    update_successful: bool
    grad_norm: float
    num_zeros_in_grad: int | None
    workload: TrainingStepWorkload
    loss_metrics: dict[str, float] = Field(default_factory=dict)
    pipeline_metrics: dict[str, float] = Field(default_factory=dict)


class MegatronOptimizerStepResult(BaseModel):
    update_successful: bool
    grad_norm: float
    num_zeros_in_grad: int | None


class MegatronForwardBackwardStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reduced_loss: torch.Tensor
    probs_corr: float
    kl_policy_ref: float | None = None
    new_logprobs: list[torch.Tensor]
    workload: TrainingStepWorkload
    loss_metrics: dict[str, float] = Field(default_factory=dict)
    pipeline_metrics: dict[str, float] = Field(default_factory=dict)


class RLForwardBackwardState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_loss_sum: torch.Tensor
    probs_corr_total: torch.Tensor
    kl_values: tuple[float, ...]
    new_logprobs: tuple[torch.Tensor, ...]
    sample_indices: tuple[int | None, ...]
    token_count: torch.Tensor
    micro_count: int = Field(ge=1)
    schedule: MCoreScheduleAdapter[PreparedRLMicroInputs]
    loss_diagnostics: LossOffPolicyDiagnosticsAccumulator
    inter_schedule_metrics: dict[str, float] = Field(default_factory=dict)


class SFTForwardBackwardState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_loss_sum: torch.Tensor
    new_logprobs: tuple[torch.Tensor, ...]
    sample_indices: tuple[int | None, ...]
    prepared_micros: tuple[PreparedSFTMicroInputs, ...]
    device: torch.device
    schedule: MCoreScheduleAdapter[PreparedSFTMicroInputs]


class MegatronForwardBackwardJobResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    new_logprobs: tuple[torch.Tensor, ...]
    local_token_count: torch.Tensor
    completed_gradient_steps: int = Field(ge=0)
    logical_nonpadding_tokens: int = Field(ge=0)
    executed_token_equivalents: int = Field(ge=0)
    gpu_service_ns: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


class MegatronForwardJobResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    new_logprobs: tuple[torch.Tensor, ...]
    logical_nonpadding_tokens: int = Field(ge=0)
    executed_token_equivalents: int = Field(ge=0)
    gpu_service_ns: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)


def print0(rank: int, *values: Any) -> None:
    if rank == 0:
        print(*values)


def freeze_model(model_chunks: list[MegatronModule]) -> list[MegatronModule]:
    for module in model_chunks:
        for param in module.parameters():
            param.requires_grad = False
    return model_chunks


def _register_trainable_parameter_mode(
    provider: Any,
    *,
    trainable_parameter_mode: Literal["lora", "base_model"],
) -> None:
    if trainable_parameter_mode == "lora":
        provider.register_pre_wrap_hook(freeze_model)
        provider.register_pre_wrap_hook(
            lambda chunks: apply_lora_adapters(chunks, provider)
        )
        return
    if trainable_parameter_mode == "base_model":
        return
    raise ValueError(
        "trainable_parameter_mode must be 'lora' or 'base_model', got "
        f"{trainable_parameter_mode!r}"
    )


def _eager_initialize_optimizer_state(optimizer: Any) -> None:
    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if chained_optimizers is not None:
        for child_optimizer in chained_optimizers:
            _eager_initialize_optimizer_state(child_optimizer)
        return
    init_state_fn = getattr(optimizer, "init_state_fn", None)
    inner_optimizer = getattr(optimizer, "optimizer", None)
    if callable(init_state_fn) and inner_optimizer is not None:
        init_state_fn(inner_optimizer, getattr(optimizer, "config", None))


def _default_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        bf16=True,
        lr=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.1,
        adam_eps=1e-13,
    )


def _maybe_print_optimizer_stats(
    optimizer: Any,
    model: ModelChunks,
) -> None:
    global _optimizer_stats_printed
    if _optimizer_stats_printed:
        return
    if torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        if torch.distributed.get_rank() != 0:  # ty: ignore[possibly-missing-attribute]
            _optimizer_stats_printed = True
            return
    num_params = sum(
        p.numel()
        for group in optimizer.param_groups
        if not group["is_decoupled_lr"]
        for p in group["params"]
    )
    print(f"Number of parameters in optimizer: {num_params:,}")
    total_params = sum(p.numel() for module in model for p in module.parameters())
    percent = (num_params / total_params) * 100 if total_params > 0 else 0
    print(f"Optimizer parameters as percent of total: {percent:0.2f}%")
    _optimizer_stats_printed = True


def _build_optimizer(
    model: ModelChunks,
    optimizer_config: OptimizerConfig,
) -> Any:
    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=as_megatron_api_chunks(model),
    )
    _maybe_print_optimizer_stats(optimizer, model)
    return optimizer


def configure_moe_routing_replay(
    runtime: TrainingRuntime,
    *,
    replay_bundle_path: str | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
    strict: bool = True,
) -> None:
    if replay_bundle is not None and replay_bundle_path is not None:
        raise RuntimeError(
            "Provide either replay_bundle_path or replay_bundle, not both"
        )
    if replay_bundle is None and replay_bundle_path is None:
        if runtime.moe_routing_replay_controller is not None:
            runtime.moe_routing_replay_controller.remove_router_patches()
            runtime.moe_routing_replay_controller = None
        return

    if replay_bundle is None:
        if replay_bundle_path is None:
            raise RuntimeError(
                "replay_bundle_path is required when replay_bundle is None"
            )
        replay_bundle = MoeRoutingReplayBundle.from_dir(replay_bundle_path)

    if runtime.moe_routing_replay_controller is not None:
        runtime.moe_routing_replay_controller.update_bundle(
            bundle=replay_bundle,
            strict=strict,
        )
        return

    controller = MoeRoutingReplayController(
        bundle=replay_bundle,
        strict=strict,
    )
    controller.install_router_patches(runtime.model)
    runtime.moe_routing_replay_controller = controller


def _moe_routing_replay_requested(
    *,
    replay_bundle_path: str | None,
    replay_bundle: MoeRoutingReplayBundle | None,
) -> bool:
    if replay_bundle_path is not None or replay_bundle is not None:
        return True
    return os.environ.get("ART_MEGATRON_ENABLE_MOE_ROUTING_REPLAY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _enable_native_moe_routing_replay(provider: Any) -> None:
    if bool(getattr(provider, "moe_router_fusion", False)):
        raise RuntimeError(
            "MoE routing replay requires provider.moe_router_fusion=False because "
            "Megatron Core fused routing bypasses RouterReplay"
        )
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_router_replay_instances()
    provider.moe_enable_routing_replay = True


def _is_bridge_hf_load_hook(hook: Any) -> bool:
    function = hook
    seen: set[int] = set()
    while id(function) not in seen:
        seen.add(id(function))
        if getattr(function, "__name__", "") in {
            "load_weights_hf_to_megatron",
            "_optimized_load_weights_hf_to_megatron",
        } or getattr(function, "__qualname__", "").endswith(
            ".load_weights_hf_to_megatron"
        ):
            return True
        function = getattr(function, "func", None) or getattr(
            function, "__wrapped__", None
        )
        if function is None:
            return False
    return False


def build_training_runtime(
    *,
    model_identifier: str | None = None,
    model_initialization: Literal["pretrained", "random"] = "pretrained",
    provider_torch_dtype: torch.dtype = torch.bfloat16,
    provider_bundle_configure: Callable[[ProviderBundle], None] | None = None,
    provider_configure: Callable[[Any], None] | None = None,
    optimizer_config: OptimizerConfig | None = None,
    moe_routing_replay_path: str | None = None,
    moe_routing_replay_bundle: MoeRoutingReplayBundle | None = None,
    moe_routing_replay_strict: bool = True,
    print_env: bool = True,
    build_optimizer: bool = True,
    trainable_parameter_mode: Literal["lora", "base_model"] = "lora",
    allow_unvalidated_arch: bool | None = None,
    model_support_key: str | None = None,
    snapshot_pool_capacity: int = 2,
) -> TrainingRuntime:
    if random_state := os.environ.get("ART_MEGATRON_RANDOM_STATE"):
        seed = int(random_state)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    install_fast_frozen_output_backward()
    provider_bundle = prepare_provider_bundle(
        model_identifier
        or os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER),
        torch_dtype=provider_torch_dtype,
        load_weights=model_initialization == "pretrained",
        allow_unvalidated_arch=(
            os.environ.get("ART_MEGATRON_ALLOW_UNVALIDATED_ARCH", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if allow_unvalidated_arch is None
            else allow_unvalidated_arch
        ),
        model_support_key=model_support_key,
    )
    if model_initialization == "random":
        hooks = list(getattr(provider_bundle.provider, "_pre_wrap_hooks", ()))
        checkpoint_hooks = [hook for hook in hooks if _is_bridge_hf_load_hook(hook)]
        if len(checkpoint_hooks) != len(hooks):
            raise RuntimeError(
                "random model initialization requires only Bridge checkpoint loaders; "
                f"found {len(checkpoint_hooks)} loaders among {len(hooks)} hooks"
            )
        provider_bundle.provider._pre_wrap_hooks = []
        provider_bundle.provider.perform_initialization = True
    if provider_bundle_configure is not None:
        provider_bundle_configure(provider_bundle)
    provider = provider_bundle.provider
    if provider_configure is not None:
        provider_configure(provider)
    replay_requested = _moe_routing_replay_requested(
        replay_bundle_path=moe_routing_replay_path,
        replay_bundle=moe_routing_replay_bundle,
    )
    if replay_requested:
        _enable_native_moe_routing_replay(provider)
    finalize_provider_bundle(provider_bundle)
    _register_trainable_parameter_mode(
        provider,
        trainable_parameter_mode=trainable_parameter_mode,
    )

    model = cast(
        ModelChunks,
        provider.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(
                # memory and comm for this should be small anyways cause lora
                grad_reduce_in_fp32=True,
                average_in_collective=False,
            ),
            data_parallel_random_init=False,
            init_model_with_meta_device=model_initialization == "pretrained",
        ),
    )

    if not torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        raise RuntimeError(
            "torch.distributed must be initialized before building runtime"
        )
    rank = torch.distributed.get_rank()  # ty: ignore[possibly-missing-attribute]
    world_size = torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
    validate_pipeline_topology(
        world_size=world_size,
        tp=int(ps.get_tensor_model_parallel_world_size()),
        cp=int(ps.get_context_parallel_world_size()),
        pp=int(ps.get_pipeline_model_parallel_world_size()),
        ep=int(ps.get_expert_model_parallel_world_size()),
        etp=int(ps.get_expert_tensor_parallel_world_size()),
        vp=int(ps.get_virtual_pipeline_model_parallel_world_size() or 1),
        num_layers=(
            None
            if provider.pipeline_model_parallel_layout is not None
            else int(provider.num_layers)
        ),
    )

    if rank == 0 and print_env:
        print("TORCHINDUCTOR_CACHE_DIR:", os.environ["TORCHINDUCTOR_CACHE_DIR"])
        print("Resolved inductor cache_dir():", inductor_cache_dir())
        print("TRITON_CACHE_DIR:", os.environ["TRITON_CACHE_DIR"])

    provider_bundle.handler.install_preprocess_patch(model)
    if replay_requested:
        prepare_moe_routing_replay_boundaries(model)
    transformer_layers_compiled = configure_training_compile(
        model=model,
        provider=provider,
        provider_bundle=provider_bundle,
    )

    optimizer_config = optimizer_config or _default_optimizer_config()
    optimizer = _build_optimizer(model, optimizer_config) if build_optimizer else None
    metrics_group = (
        torch.distributed.new_group(backend="gloo")  # ty: ignore[possibly-missing-attribute]
        if world_size > 1
        else None
    )

    runtime = TrainingRuntime(
        provider_bundle=provider_bundle,
        provider=provider,
        model=model,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
        transformer_layers_compiled=transformer_layers_compiled,
        rank=rank,
        world_size=world_size,
        snapshot_pool_capacity=snapshot_pool_capacity,
        inter_forward_backward_timing=_InterForwardBackwardTiming(
            metrics_group=metrics_group
        ),
    )
    _model_runtime_sha256(runtime)
    configure_moe_routing_replay(
        runtime,
        replay_bundle_path=moe_routing_replay_path,
        replay_bundle=moe_routing_replay_bundle,
        strict=moe_routing_replay_strict,
    )
    return runtime


def _execute_megatron_rl_forward_backward_steps(
    runtime: TrainingRuntime,
    job: TrainJobSpec | ForwardBackwardJobSpec | ForwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    before_step: Callable[[int], None],
    after_step: Callable[[int, int, RLForwardBackwardState, float, float, float], None],
    cancelled: Event | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
    defer_grad_sync: bool = False,
    forward_only: bool = False,
    state_is_resident: bool = False,
) -> tuple[dict[str, torch.dtype], float]:
    job_prepare_started = time.perf_counter()
    template = None
    zero_template = None
    ref_logprobs_by_index = None
    cp_lookahead_state = None
    next_step_first_micro = None
    next_step_first_ref_logprobs = None
    try:
        global_grad_accumulation_sequences = resolve_global_grad_accumulation_sequences(
            job.config.grad_accumulation_sequences
        )
        replay_finalize_started = time.perf_counter()
        if (
            replay_bundle is None
            and packed_tensors.get("moe_routing_replay") is not None
        ):
            replay_bundle = build_moe_routing_replay_bundle_from_packed_tensors(
                packed_tensors=packed_tensors,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            )
        configure_moe_routing_replay(
            runtime,
            replay_bundle=replay_bundle,
            strict=_moe_replay_strict(job),
        )
        replay_finalize_s = time.perf_counter() - replay_finalize_started
        adapter_dtypes = (
            {} if state_is_resident else _prepare_rl_training_state(runtime, job)
        )
        template = _clone_packed_tensors(select_indexed_inputs(packed_tensors, 0))
        zero_template = _zero_contribution_inputs(template)
        num_sequences, packed_sequence_length = map(int, packed_tensors["tokens"].shape)
        num_steps = math.ceil(num_sequences / global_grad_accumulation_sequences)
        topology = _infer_parallel_topology(runtime.model)
        has_local_loss_stage = any(chunk_post_process(chunk) for chunk in runtime.model)
        hybridep_token_counts_by_step = (
            [
                build_rl_hybridep_token_counts(
                    packed_tensors=packed_tensors,
                    step_index=step_index,
                    num_sequences=num_sequences,
                    global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                    topology=topology,
                    provider=runtime.provider,
                    model_support_handler=runtime.model_support_handler,
                )
                for step_index in range(num_steps)
            ]
            if ps.get_expert_model_parallel_world_size() > 1
            else None
        )
        _ensure_hybridep_capacity(
            runtime,
            packed_sequence_length=packed_sequence_length,
            context_parallel_size=topology.cp,
            required_capacity=max(
                (
                    count
                    for step_counts in hybridep_token_counts_by_step or ()
                    for count in step_counts
                ),
                default=0,
            ),
        )
        ref_logprobs_by_index = _prepare_kl_reference_logprobs(
            runtime=runtime,
            job=job,
            packed_tensors=packed_tensors,
            num_sequences=num_sequences,
            num_steps=num_steps,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        cp_lookahead_state = CpBatchLookaheadState() if int(topology.cp) > 1 else None
        job_prepare_s = time.perf_counter() - job_prepare_started
        for step_index in range(num_steps):
            step_input_prepare_started = time.perf_counter()
            if cancelled is not None and cancelled.is_set():
                from art.megatron.runtime.trainer_run import TrainingCancelledError

                raise TrainingCancelledError("F/B command was cancelled")
            before_step(step_index)
            hybridep_token_counts = (
                None
                if hybridep_token_counts_by_step is None
                else hybridep_token_counts_by_step[step_index]
            )
            micro_indices = build_micro_sample_indices(
                step_index=step_index,
                num_sequences=num_sequences,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            )
            micro_inputs = select_micro_inputs(
                packed_tensors,
                micro_indices,
                zero_template,
            )
            ref_logprobs = (
                select_micro_ref_logprobs(
                    ref_logprobs_by_index,
                    micro_indices,
                    zero_template,
                )
                if ref_logprobs_by_index is not None and has_local_loss_stage
                else None
            )
            next_step_first_micro = (
                _select_next_step_first_micro(
                    packed_tensors=packed_tensors,
                    zero_template=zero_template,
                    step_index=step_index,
                    num_steps=num_steps,
                    num_sequences=num_sequences,
                    global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                )
                if cp_lookahead_state is not None
                else None
            )
            next_step_first_ref_logprobs = (
                _select_next_step_first_ref_logprobs(
                    ref_logprobs_by_index=ref_logprobs_by_index,
                    zero_template=zero_template,
                    step_index=step_index,
                    num_steps=num_steps,
                    num_sequences=num_sequences,
                    global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                )
                if (
                    cp_lookahead_state is not None
                    and ref_logprobs_by_index is not None
                    and has_local_loss_stage
                )
                else None
            )
            step_input_prepare_s = time.perf_counter() - step_input_prepare_started
            started = time.perf_counter()
            state = run_megatron_rl_forward_backward_step(
                model_chunks=runtime.model,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                inputs=micro_inputs,
                config=job.config,
                experimental_config=_experimental_train_config(job),
                ref_logprobs=ref_logprobs,
                step_index=step_index,
                sample_index=micro_indices,
                moe_routing_replay_controller=runtime.moe_routing_replay_controller,
                cp_lookahead_state=cp_lookahead_state,
                next_step_first_micro=next_step_first_micro,
                next_step_first_ref_logprobs=next_step_first_ref_logprobs,
                hybridep_token_counts=hybridep_token_counts,
                inter_forward_backward_timing=runtime.inter_forward_backward_timing,
                defer_grad_sync=defer_grad_sync,
                forward_only=forward_only,
                loss=(
                    job.loss
                    if isinstance(job, (ForwardBackwardJobSpec, ForwardJobSpec))
                    else None
                ),
            )
            after_step(
                step_index,
                num_steps,
                state,
                time.perf_counter() - started,
                replay_finalize_s,
                step_input_prepare_s,
            )
        return adapter_dtypes, job_prepare_s
    finally:
        if template is not None:
            del template
        if zero_template is not None:
            del zero_template
        if ref_logprobs_by_index is not None:
            del ref_logprobs_by_index
        if "micro_inputs" in locals():
            del micro_inputs
        if next_step_first_micro is not None:
            del next_step_first_micro
        if next_step_first_ref_logprobs is not None:
            del next_step_first_ref_logprobs
        if cp_lookahead_state is not None:
            cp_lookahead_state.pending_prepared_micro = None


def execute_megatron_rl_job(
    runtime: TrainingRuntime,
    job: TrainJobSpec,
    packed_tensors: PackedTensors,
    *,
    progress_sink: Callable[[int, int, dict[str, float]], None],
    adapter_ready_sink: Callable[[], None] | None,
    snapshot_sink: Callable[
        [TrainJobSpec, dict[str, torch.dtype], dict[str, Any], bool],
        dict[str, float],
    ]
    | None = None,
    cancelled: Event | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
) -> dict[str, float]:
    """Execute one current fused RL update from an in-memory packed batch."""
    adapter_dtypes = None
    inter_schedule_metrics: dict[str, float] = {}
    final_metrics: dict[str, float] = {}
    job_succeeded = False

    def finish_step(
        step_index: int,
        num_steps: int,
        state: RLForwardBackwardState,
        forward_backward_s: float,
        replay_finalize_s: float,
        step_input_prepare_s: float,
    ) -> None:
        nonlocal final_metrics, inter_schedule_metrics
        assert runtime.optimizer is not None
        optimizer_started = time.perf_counter()
        optimizer_result = run_megatron_optimizer_step(
            optimizer=runtime.optimizer,
            learning_rate=job.config.learning_rate,
            model_support_handler=runtime.model_support_handler,
            model_chunks=runtime.model,
            before_step=runtime.optimizer_snapshot_barrier.wait_before_mutation,
        )
        optimizer_s = time.perf_counter() - optimizer_started
        result_finalize_started = time.perf_counter()
        result_build_started = time.perf_counter()
        step_result = TrainStepResult(
            **_finish_megatron_rl_forward_backward_step(state).model_dump(),
            update_successful=optimizer_result.update_successful,
            grad_norm=optimizer_result.grad_norm,
            num_zeros_in_grad=optimizer_result.num_zeros_in_grad,
        )
        step_result.pipeline_metrics.update(
            {
                "time/post_schedule_optimizer_s": optimizer_s,
                "time/post_schedule_result_build_s": (
                    time.perf_counter() - result_build_started
                ),
            }
        )
        print0(
            runtime.rank,
            "Correlation between old and new probabilities:",
            step_result.probs_corr,
        )
        _validate_train_step_result_finite(runtime, step_result)
        final_metrics = _rl_step_metrics(
            step_result,
            num_gradient_steps=num_steps,
            train_step_s=forward_backward_s + optimizer_s,
        )
        if step_index == 0:
            inter_schedule_metrics = {
                name: value
                for name, value in step_result.pipeline_metrics.items()
                if name.startswith(
                    (
                        _INTER_FORWARD_BACKWARD_GAP_PREFIX,
                        _INTER_FORWARD_BACKWARD_GPU_GAP_PREFIX,
                    )
                )
            }
        final_metrics.update(inter_schedule_metrics)
        final_metrics["time/replay_finalize_s"] = replay_finalize_s
        final_metrics["time/step_input_prepare_s"] = step_input_prepare_s
        final_metrics["time/step_result_finalize_s"] = (
            time.perf_counter() - result_finalize_started
        )
        if runtime.rank == 0:
            progress_started = time.perf_counter()
            progress_sink(step_index, num_steps, final_metrics)
            final_metrics["time/step_progress_emit_s"] = (
                time.perf_counter() - progress_started
            )

    try:
        adapter_dtypes, job_prepare_s = _execute_megatron_rl_forward_backward_steps(
            runtime,
            job,
            packed_tensors,
            before_step=lambda _step_index: None,
            after_step=finish_step,
            cancelled=cancelled,
            replay_bundle=replay_bundle,
        )
        if cancelled is not None and cancelled.is_set():
            from art.megatron.runtime.trainer_run import TrainingCancelledError

            raise TrainingCancelledError("train job was cancelled")
        if snapshot_sink is None or adapter_ready_sink is None:
            raise RuntimeError("Typed training requires a snapshot publisher")
        if runtime.adapter_export_config is None:
            raise RuntimeError("Trainer has no resident adapter export config")
        final_metrics.update(
            snapshot_sink(
                job,
                adapter_dtypes,
                runtime.adapter_export_config,
                _should_snapshot_optimizer(
                    runtime,
                    step=job.step,
                    optimizer_save_interval=job.config.optimizer_save_interval,
                    final_training_step=job.config.final_training_step,
                ),
            )
        )
        final_metrics["time/job_prepare_s"] = job_prepare_s
        adapter_ready_started = time.perf_counter()
        adapter_ready_sink()
        final_metrics["time/step_adapter_ready_emit_s"] = (
            time.perf_counter() - adapter_ready_started
        )
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.step
        runtime.resident_generation_id = job.output_generation_id
        runtime.optimizer_state_loaded = True
        job_succeeded = True
        return final_metrics
    finally:
        if not job_succeeded:
            runtime.resident_training_session_id = None
            runtime.resident_policy_step = None
            runtime.resident_generation_id = None
            runtime.optimizer_state_loaded = False
        if adapter_dtypes is not None:
            del adapter_dtypes


def execute_megatron_rl_forward_backward_job(
    runtime: TrainingRuntime,
    job: ForwardBackwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    gradient_accumulator: Any,
    cancelled: Event | None = None,
) -> MegatronForwardBackwardJobResult:
    """Execute one independently admitted F/B contribution."""
    from art.megatron.training.gradient_accumulator import GradientAccumulator

    loss_mask = packed_tensors["assistant_mask"]
    if job.loss is None:
        loss_mask = loss_mask[:, 1:]
    observed_tokens = int(loss_mask.sum().item())
    if observed_tokens != job.expected_global_loss_bearing_tokens:
        raise RuntimeError(
            "packed F/B loss-bearing token count differs from command provenance: "
            f"observed={observed_tokens}, "
            f"expected={job.expected_global_loss_bearing_tokens}"
        )
    states: list[RLForwardBackwardState] = []
    durations: list[float] = []
    internal = GradientAccumulator(
        runtime.model,
        flush_gradients=flush_param_grads_to_main_grads,
    )
    reduction = "sum" if job.loss is not None else "token_mean"

    def before_step(step_index: int) -> None:
        if step_index:
            internal.before_forward_backward()

    def record_step(
        step_index: int,
        _num_steps: int,
        state: RLForwardBackwardState,
        duration_s: float,
        _replay_finalize_s: float,
        _step_input_prepare_s: float,
    ) -> None:
        internal.record(
            f"{job.operation_id}:{step_index}",
            state.token_count,
            reduction=reduction,
        )
        states.append(state)
        durations.append(duration_s)

    def execute() -> tuple[float, Any, tuple[MegatronForwardBackwardStepResult, ...]]:
        gradient_accumulator.before_forward_backward()
        _, job_prepare_s = _execute_megatron_rl_forward_backward_steps(
            runtime,
            job,
            packed_tensors,
            before_step=before_step,
            after_step=record_step,
            cancelled=cancelled,
            defer_grad_sync=True,
        )
        internal.seal(internal.contribution_ids)
        local_sums = internal.prepare_local_sums()
        gradient_accumulator.record(
            job.operation_id,
            local_sums.local_token_count,
            expected_global_token_count=job.expected_global_loss_bearing_tokens,
            reduction=reduction,
        )
        internal.consume()
        results = tuple(
            _finish_megatron_rl_forward_backward_step(state) for state in states
        )
        if any(not torch.isfinite(result.reduced_loss).item() for result in results):
            raise RuntimeError("Megatron F/B produced a non-finite loss")
        return job_prepare_s, local_sums, results

    (job_prepare_s, local_sums, results), gpu_service_ns = measure_cuda_call(execute)
    runtime.resident_run_id = job.run_id
    runtime.resident_training_session_id = job.training_session_id
    runtime.resident_policy_step = job.expected_learner_version
    runtime.resident_generation_id = job.source.generation_id
    return MegatronForwardBackwardJobResult(
        new_logprobs=tuple(
            values
            for _, values in sorted(
                (
                    (sample_index, values)
                    for state in states
                    for sample_index, values in zip(
                        state.sample_indices, state.new_logprobs, strict=True
                    )
                    if sample_index is not None
                ),
                key=lambda item: item[0],
            )
        ),
        local_token_count=local_sums.local_token_count,
        completed_gradient_steps=len(results),
        logical_nonpadding_tokens=sum(
            result.workload.logical_nonpadding_tokens for result in results
        ),
        executed_token_equivalents=sum(
            result.workload.executed_token_equivalents for result in results
        ),
        gpu_service_ns=gpu_service_ns,
        metrics={
            "time/forward_backward_s": sum(durations),
            "time/job_prepare_s": job_prepare_s,
            "data/gradient_steps": float(len(results)),
        },
    )


def execute_megatron_dynamic_lora_forward_backward_job(
    runtime: TrainingRuntime,
    job: ForwardBackwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    run_slots: Any,
    gradient_accumulator: Any,
    cancelled: Event | None = None,
) -> MegatronForwardBackwardJobResult:
    """Execute F/B against one exact-shape resident Megatron run slot."""
    from art.megatron.lora import LoRASlotRef, use_lora_slot
    from art.megatron.training.gradient_accumulator import (
        ParameterGradientAccumulator,
    )

    loss_mask = packed_tensors["assistant_mask"]
    if job.loss is None:
        loss_mask = loss_mask[:, 1:]
    observed_tokens = int(loss_mask.sum().item())
    if observed_tokens != job.expected_global_loss_bearing_tokens:
        raise RuntimeError(
            "packed F/B loss-bearing token count differs from command provenance: "
            f"observed={observed_tokens}, "
            f"expected={job.expected_global_loss_bearing_tokens}"
        )
    reference = _experimental_train_config(job).get("kl_ref_adapter_path")
    if (
        job.config.kl_penalty_coef > 0.0
        and reference is not None
        and os.path.abspath(reference) != os.path.abspath(job.source_adapter_path)
    ):
        raise NotImplementedError(
            "resident dynamic LoRA requires the KL reference to be the learner"
        )
    parameters = run_slots.checkpoint_slot_parameters(job.run_id)
    internal = ParameterGradientAccumulator(parameters)
    reduction = "sum" if job.loss is not None else "token_mean"
    states: list[RLForwardBackwardState] = []
    durations: list[float] = []

    def before_step(_step_index: int) -> None:
        run_slots.clear_checkpoint_slot_grads(job.run_id)

    def record_step(
        step_index: int,
        _num_steps: int,
        state: RLForwardBackwardState,
        duration_s: float,
        _replay_finalize_s: float,
        _step_input_prepare_s: float,
    ) -> None:
        gradients = tuple(
            (
                torch.zeros_like(parameter, dtype=torch.float32)
                if parameter.grad is None
                else parameter.grad.detach().float()
            )
            for parameter in parameters
        )
        internal.record(
            f"{job.operation_id}:{step_index}",
            state.token_count,
            gradients,
            step_flags=tuple(parameter.grad is not None for parameter in parameters),
            reduction=reduction,
        )
        states.append(state)
        durations.append(duration_s)

    def execute() -> tuple[float, Any, tuple[MegatronForwardBackwardStepResult, ...]]:
        with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
            _, job_prepare_s = _execute_megatron_rl_forward_backward_steps(
                runtime,
                job,
                packed_tensors,
                before_step=before_step,
                after_step=record_step,
                cancelled=cancelled,
                defer_grad_sync=True,
                state_is_resident=True,
            )
        internal.seal(internal.contribution_ids)
        local_sums, step_flags = internal.prepare_local_sums()
        gradient_accumulator.record(
            job.operation_id,
            local_sums.local_token_count,
            local_sums.gradients,
            expected_global_token_count=job.expected_global_loss_bearing_tokens,
            step_flags=step_flags,
            reduction=reduction,
        )
        internal.consume()
        run_slots.clear_checkpoint_slot_grads(job.run_id)
        results = tuple(
            _finish_megatron_rl_forward_backward_step(state) for state in states
        )
        if any(not torch.isfinite(result.reduced_loss).item() for result in results):
            raise RuntimeError("Megatron F/B produced a non-finite loss")
        return job_prepare_s, local_sums, results

    (job_prepare_s, local_sums, results), gpu_service_ns = measure_cuda_call(execute)
    return MegatronForwardBackwardJobResult(
        new_logprobs=tuple(
            values
            for _, values in sorted(
                (
                    (sample_index, values)
                    for state in states
                    for sample_index, values in zip(
                        state.sample_indices, state.new_logprobs, strict=True
                    )
                    if sample_index is not None
                ),
                key=lambda item: item[0],
            )
        ),
        local_token_count=local_sums.local_token_count,
        completed_gradient_steps=len(results),
        logical_nonpadding_tokens=sum(
            result.workload.logical_nonpadding_tokens for result in results
        ),
        executed_token_equivalents=sum(
            result.workload.executed_token_equivalents for result in results
        ),
        gpu_service_ns=gpu_service_ns,
        metrics={
            "time/forward_backward_s": sum(durations),
            "time/job_prepare_s": job_prepare_s,
            "data/gradient_steps": float(len(results)),
        },
    )


def _sft_command_step_inputs(
    runtime: TrainingRuntime,
    job: SftForwardBackwardJobSpec | SftForwardJobSpec,
    batch: SFTBatchData,
    step_index: int,
) -> tuple[list[dict[str, torch.Tensor]], list[int | None], list[int] | None]:
    trajectories = list(batch.trajectory_tensors)
    global_sequences = resolve_global_grad_accumulation_sequences(
        job.config.grad_accumulation_sequences
    )
    indices = build_micro_sample_indices(
        step_index=step_index,
        num_sequences=len(trajectories),
        global_grad_accumulation_sequences=global_sequences,
    )
    topology = _infer_parallel_topology(runtime.model)
    hybridep_token_counts = (
        build_sft_hybridep_token_counts(
            trajectory_tensors=trajectories,
            step_index=step_index,
            global_grad_accumulation_sequences=global_sequences,
            topology=topology,
            provider=runtime.provider,
            model_support_handler=runtime.model_support_handler,
        )
        if ps.get_expert_model_parallel_world_size() > 1
        else None
    )
    _ensure_hybridep_capacity(
        runtime,
        packed_sequence_length=max(
            int(inputs["input_ids"].numel()) for inputs in trajectories
        ),
        context_parallel_size=topology.cp,
        required_capacity=max(hybridep_token_counts or (), default=0),
    )
    return (
        select_sft_micro_inputs(
            trajectories,
            indices,
            _zero_contribution_sft_inputs(_clone_sft_tensors(trajectories[0])),
        ),
        indices,
        hybridep_token_counts,
    )


def _validate_sft_command_batch(
    job: SftForwardBackwardJobSpec | SftForwardJobSpec,
    batch: SFTBatchData,
) -> None:
    if batch.fingerprint != job.batch_fingerprint:
        raise ValueError("SFT payload fingerprint differs from its command")
    if (
        batch.num_tokens != job.expected_global_nonpadding_tokens
        or batch.num_trainable_tokens != job.expected_global_loss_bearing_tokens
    ):
        raise ValueError("SFT payload token counts differ from its command")


def _sft_command_result(
    states: list[SFTForwardBackwardState],
    batch: SFTBatchData,
    *,
    return_token_logprobs: bool,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int]:
    if not states:
        raise RuntimeError("SFT command executed no schedule steps")
    device = states[0].device
    local_tokens = sum(
        (
            _local_trainable_sft_token_count_tensor(
                state.prepared_micros, device=device
            )
            for state in states
        ),
        torch.zeros([], device=device, dtype=torch.float32),
    )
    raw_loss_sum = sum(
        (state.raw_loss_sum for state in states),
        torch.zeros([], device=device, dtype=torch.float32),
    )
    if torch.distributed.is_initialized():
        group = ps.get_data_parallel_group(with_context_parallel=True)
        torch.distributed.all_reduce(local_tokens, group=group)
        torch.distributed.all_reduce(raw_loss_sum, group=group)
    if int(local_tokens.item()) != batch.num_trainable_tokens:
        raise RuntimeError(
            "SFT command trainable-token provenance changed: "
            f"observed={int(local_tokens.item())}, "
            f"expected={batch.num_trainable_tokens}"
        )

    executed = torch.tensor(
        sum(
            state.schedule.training_workload().executed_token_equivalents
            for state in states
        ),
        device=device,
        dtype=torch.int64,
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            executed,
            op=torch.distributed.ReduceOp.MAX,
        )
    if not return_token_logprobs:
        return raw_loss_sum / local_tokens, (), int(executed.item())

    lengths = [
        int(tensors["attention_mask"].sum().item())
        for tensors in batch.trajectory_tensors
    ]
    values = states[0].new_logprobs[0].new_zeros((len(lengths), max(lengths)))
    present = torch.zeros(len(lengths), device=device, dtype=torch.int32)
    for state in states:
        if len(state.new_logprobs) != len(state.sample_indices):
            raise RuntimeError("SFT logprob/sample counts differ")
        for sample_index, logprobs in zip(
            state.sample_indices, state.new_logprobs, strict=True
        ):
            if sample_index is None:
                continue
            length = lengths[sample_index]
            if int(logprobs.numel()) != length:
                raise RuntimeError("SFT logprob length differs from its trajectory")
            values[sample_index, :length] = logprobs.reshape(-1)
            present[sample_index] = 1
    if ps.get_data_parallel_world_size() > 1:
        group = ps.get_data_parallel_group()
        torch.distributed.all_reduce(values, group=group)
        torch.distributed.all_reduce(present, group=group)
    if not torch.all(present == 1):
        raise RuntimeError("SFT command did not return every trajectory exactly once")
    host = values.cpu()
    return (
        raw_loss_sum / local_tokens,
        tuple(host[index : index + 1, :length] for index, length in enumerate(lengths)),
        int(executed.item()),
    )


def _sft_command_metrics(
    states: list[SFTForwardBackwardState],
    *,
    reduced_loss: torch.Tensor,
    elapsed_s: float,
    backward: bool,
    dropped_trajectories: int,
) -> dict[str, float]:
    metrics = {
        "loss/train": float(reduced_loss.item()),
        "data/gradient_steps": float(len(states) if backward else 0),
        "data/dropped_sft_trajectories": float(dropped_trajectories),
        "time/forward_backward_s" if backward else "time/forward_s": elapsed_s,
    }
    for state in states:
        for name, value in state.schedule.telemetry.metrics().items():
            metrics[name] = metrics.get(name, 0.0) + float(value)
    return metrics


def execute_megatron_dynamic_lora_sft_forward_backward_job(
    runtime: TrainingRuntime,
    job: SftForwardBackwardJobSpec,
    batch: SFTBatchData,
    *,
    run_slots: Any,
    gradient_accumulator: Any,
    cancelled: Event | None = None,
) -> MegatronForwardBackwardJobResult:
    """Execute one exact SFT contribution against a resident LoRA slot."""

    from art.megatron.lora import LoRASlotRef, use_lora_slot
    from art.megatron.training.gradient_accumulator import (
        ParameterGradientAccumulator,
    )

    _validate_sft_command_batch(job, batch)
    parameters = run_slots.checkpoint_slot_parameters(job.run_id)
    internal = ParameterGradientAccumulator(parameters)
    states: list[SFTForwardBackwardState] = []
    global_sequences = resolve_global_grad_accumulation_sequences(
        job.config.grad_accumulation_sequences
    )
    num_steps = math.ceil(batch.num_trajectories / global_sequences)

    def execute() -> tuple[Any, torch.Tensor, tuple[torch.Tensor, ...], int, float]:
        configure_moe_routing_replay(runtime)
        started = time.perf_counter()
        try:
            with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
                for step_index in range(num_steps):
                    if cancelled is not None and cancelled.is_set():
                        from art.megatron.runtime.trainer_run import (
                            TrainingCancelledError,
                        )

                        raise TrainingCancelledError("SFT F/B job was cancelled")
                    run_slots.clear_checkpoint_slot_grads(job.run_id)
                    inputs, indices, hybridep_counts = _sft_command_step_inputs(
                        runtime, job, batch, step_index
                    )
                    state = run_megatron_sft_forward_backward_step(
                        model_chunks=runtime.model,
                        provider=runtime.provider,
                        model_support_handler=runtime.model_support_handler,
                        inputs=inputs,
                        step_index=step_index,
                        sample_index=indices,
                        moe_routing_replay_controller=(
                            runtime.moe_routing_replay_controller
                        ),
                        hybridep_token_counts=hybridep_counts,
                        defer_grad_sync=True,
                        return_token_logprobs=job.return_token_logprobs,
                    )
                    gradients = tuple(
                        (
                            torch.zeros_like(parameter, dtype=torch.float32)
                            if parameter.grad is None
                            else parameter.grad.detach().float()
                        )
                        for parameter in parameters
                    )
                    internal.record(
                        f"{job.operation_id}:{step_index}",
                        _local_trainable_sft_token_count_tensor(
                            state.prepared_micros, device=state.device
                        ),
                        gradients,
                        step_flags=tuple(
                            parameter.grad is not None for parameter in parameters
                        ),
                        reduction="token_mean",
                    )
                    states.append(state)
            internal.seal(internal.contribution_ids)
            local_sums, step_flags = internal.prepare_local_sums()
            gradient_accumulator.record(
                job.operation_id,
                local_sums.local_token_count,
                local_sums.gradients,
                expected_global_token_count=batch.num_trainable_tokens,
                step_flags=step_flags,
                reduction="token_mean",
            )
            internal.consume()
            reduced_loss, logprobs, executed = _sft_command_result(
                states,
                batch,
                return_token_logprobs=job.return_token_logprobs,
            )
            return (
                local_sums,
                reduced_loss,
                logprobs,
                executed,
                time.perf_counter() - started,
            )
        finally:
            run_slots.clear_checkpoint_slot_grads(job.run_id)

    (local_sums, reduced_loss, logprobs, executed, elapsed), gpu_service_ns = (
        measure_cuda_call(execute)
    )
    return MegatronForwardBackwardJobResult(
        new_logprobs=logprobs,
        local_token_count=local_sums.local_token_count,
        completed_gradient_steps=len(states),
        logical_nonpadding_tokens=batch.num_tokens,
        executed_token_equivalents=executed,
        gpu_service_ns=gpu_service_ns,
        metrics=_sft_command_metrics(
            states,
            reduced_loss=reduced_loss,
            elapsed_s=elapsed,
            backward=True,
            dropped_trajectories=batch.num_dropped_trajectories,
        ),
    )


def execute_megatron_dynamic_lora_sft_forward_job(
    runtime: TrainingRuntime,
    job: SftForwardJobSpec,
    batch: SFTBatchData,
    *,
    cancelled: Event | None = None,
) -> MegatronForwardJobResult:
    """Execute one exact forward-only SFT command against a resident LoRA slot."""

    from art.megatron.lora import LoRASlotRef, use_lora_slot

    _validate_sft_command_batch(job, batch)
    states: list[SFTForwardBackwardState] = []
    global_sequences = resolve_global_grad_accumulation_sequences(
        job.config.grad_accumulation_sequences
    )
    num_steps = math.ceil(batch.num_trajectories / global_sequences)

    def execute() -> tuple[torch.Tensor, tuple[torch.Tensor, ...], int, float]:
        configure_moe_routing_replay(runtime)
        started = time.perf_counter()
        with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
            for step_index in range(num_steps):
                if cancelled is not None and cancelled.is_set():
                    from art.megatron.runtime.trainer_run import TrainingCancelledError

                    raise TrainingCancelledError("SFT forward job was cancelled")
                inputs, indices, hybridep_counts = _sft_command_step_inputs(
                    runtime, job, batch, step_index
                )
                states.append(
                    run_megatron_sft_forward_step(
                        model_chunks=runtime.model,
                        provider=runtime.provider,
                        model_support_handler=runtime.model_support_handler,
                        inputs=inputs,
                        step_index=step_index,
                        sample_index=indices,
                        moe_routing_replay_controller=(
                            runtime.moe_routing_replay_controller
                        ),
                        hybridep_token_counts=hybridep_counts,
                        return_token_logprobs=job.return_token_logprobs,
                    )
                )
        reduced_loss, logprobs, executed = _sft_command_result(
            states,
            batch,
            return_token_logprobs=job.return_token_logprobs,
        )
        return reduced_loss, logprobs, executed, time.perf_counter() - started

    (reduced_loss, logprobs, executed, elapsed), gpu_service_ns = measure_cuda_call(
        execute
    )
    return MegatronForwardJobResult(
        new_logprobs=logprobs,
        logical_nonpadding_tokens=batch.num_tokens,
        executed_token_equivalents=executed,
        gpu_service_ns=gpu_service_ns,
        metrics=_sft_command_metrics(
            states,
            reduced_loss=reduced_loss,
            elapsed_s=elapsed,
            backward=False,
            dropped_trajectories=batch.num_dropped_trajectories,
        ),
    )


def execute_megatron_rl_forward_job(
    runtime: TrainingRuntime,
    job: ForwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    cancelled: Event | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
    state_is_resident: bool = False,
) -> MegatronForwardJobResult:
    """Run the canonical packed forward without retaining gradients."""

    results: list[RLForwardBackwardState] = []
    durations: list[float] = []

    def record_step(
        _step_index: int,
        _num_steps: int,
        state: RLForwardBackwardState,
        duration_s: float,
        _replay_finalize_s: float,
        _step_input_prepare_s: float,
    ) -> None:
        results.append(state)
        durations.append(duration_s)

    def execute() -> tuple[float, tuple[MegatronForwardBackwardStepResult, ...]]:
        _, job_prepare_s = _execute_megatron_rl_forward_backward_steps(
            runtime,
            job,
            packed_tensors,
            before_step=lambda _step_index: None,
            after_step=record_step,
            cancelled=cancelled,
            replay_bundle=replay_bundle,
            forward_only=True,
            state_is_resident=state_is_resident,
        )
        finished = tuple(
            _finish_megatron_rl_forward_backward_step(state) for state in results
        )
        return job_prepare_s, finished

    (job_prepare_s, finished), gpu_service_ns = measure_cuda_call(execute)
    return MegatronForwardJobResult(
        new_logprobs=tuple(
            value for result in finished for value in result.new_logprobs
        ),
        logical_nonpadding_tokens=sum(
            result.workload.logical_nonpadding_tokens for result in finished
        ),
        executed_token_equivalents=sum(
            result.workload.executed_token_equivalents for result in finished
        ),
        gpu_service_ns=gpu_service_ns,
        metrics={
            "time/forward_s": sum(durations),
            "time/job_prepare_s": job_prepare_s,
        },
    )


def execute_megatron_sft_job(
    runtime: TrainingRuntime,
    job: SFTJobSpec,
    batches: tuple[SFTBatchData, ...],
    *,
    progress_sink: Callable[[int, int, dict[str, float]], None],
    adapter_ready_sink: Callable[[], None],
    snapshot_sink: Callable[
        [SFTJobSpec, dict[str, Any], dict[str, Any], bool], dict[str, float]
    ]
    | None = None,
    cancelled: Event | None = None,
) -> dict[str, float]:
    """Execute SFT from in-memory batches; callers own transport and events."""
    if len(batches) != job.num_batches:
        raise ValueError("SFT job batch count does not match its payload")
    adapter_dtypes = None
    succeeded = False
    final_metrics: dict[str, float] = {}
    try:
        configure_moe_routing_replay(runtime)
        adapter_dtypes = _prepare_rl_training_state(runtime, job)
        grad_accumulation_sequences = int(job.config.batch_size)
        grad_accumulation_sequences = resolve_global_grad_accumulation_sequences(
            grad_accumulation_sequences
        )
        assert runtime.optimizer is not None
        runtime.optimizer.config.clip_grad = job.max_grad_norm
        for param_group in runtime.optimizer.param_groups:
            param_group["weight_decay"] = job.weight_decay
        topology = _infer_parallel_topology(runtime.model)

        for batch_index, batch in enumerate(batches):
            if cancelled is not None and cancelled.is_set():
                from art.megatron.runtime.trainer_run import TrainingCancelledError

                raise TrainingCancelledError("SFT job was cancelled")
            started = time.perf_counter()
            trajectory_tensors = list(batch.trajectory_tensors)
            template = _clone_sft_tensors(trajectory_tensors[0])
            zero_template = _zero_contribution_sft_inputs(template)
            # Scheduling uses run-global sample IDs while each payload only owns one
            # batch. Prefix aliases place this window in global index space without
            # copying tensors, then selected IDs are rebased for local lookup.
            sample_offset = batch_index * grad_accumulation_sequences
            scheduled_tensors = [
                trajectory_tensors[0]
            ] * sample_offset + trajectory_tensors
            hybridep_token_counts = (
                build_sft_hybridep_token_counts(
                    trajectory_tensors=scheduled_tensors,
                    step_index=batch_index,
                    global_grad_accumulation_sequences=(grad_accumulation_sequences),
                    topology=topology,
                    provider=runtime.provider,
                    model_support_handler=runtime.model_support_handler,
                )
                if ps.get_expert_model_parallel_world_size() > 1
                else None
            )
            _ensure_hybridep_capacity(
                runtime,
                packed_sequence_length=max(
                    int(inputs["input_ids"].numel()) for inputs in trajectory_tensors
                ),
                context_parallel_size=topology.cp,
                required_capacity=max(hybridep_token_counts or (), default=0),
            )
            scheduled_indices = build_micro_sample_indices(
                step_index=batch_index,
                num_sequences=len(scheduled_tensors),
                global_grad_accumulation_sequences=grad_accumulation_sequences,
            )
            micro_indices = [
                None if index is None else index - sample_offset
                for index in scheduled_indices
            ]
            step_result = run_megatron_sft_step(
                model_chunks=runtime.model,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                optimizer=runtime.optimizer,
                learning_rate=batch.learning_rate,
                inputs=select_sft_micro_inputs(
                    trajectory_tensors, micro_indices, zero_template
                ),
                step_index=batch_index,
                sample_index=micro_indices,
                moe_routing_replay_controller=(runtime.moe_routing_replay_controller),
                hybridep_token_counts=hybridep_token_counts,
                before_optimizer_step=(
                    runtime.optimizer_snapshot_barrier.wait_before_mutation
                ),
            )
            elapsed = time.perf_counter() - started
            final_metrics = {
                "loss/train": float(step_result.reduced_loss.item()),
                "loss/learning_rate": batch.learning_rate,
                "loss/grad_norm": float(step_result.grad_norm),
                "throughput/train_executed_tok_equiv_per_s": (
                    batch.num_tokens / elapsed if elapsed else 0.0
                ),
                **step_result.pipeline_metrics,
            }
            if runtime.rank == 0:
                progress_sink(batch_index, len(batches), final_metrics)
            del step_result, template, zero_template

        if snapshot_sink is None or runtime.adapter_export_config is None:
            raise RuntimeError("typed SFT requires an immutable snapshot publisher")
        final_metrics.update(
            snapshot_sink(job, adapter_dtypes, runtime.adapter_export_config, True)
        )
        adapter_ready_sink()
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.learner_version
        runtime.resident_generation_id = job.output_generation_id
        runtime.optimizer_state_loaded = True
        succeeded = True
        return final_metrics
    finally:
        if not succeeded:
            runtime.resident_training_session_id = None
            runtime.resident_policy_step = None
            runtime.resident_generation_id = None
            runtime.optimizer_state_loaded = False
        if adapter_dtypes is not None:
            del adapter_dtypes


def _experimental_train_config(
    job: TrainJobSpec | ForwardBackwardJobSpec | ForwardJobSpec,
) -> dev.TrainConfig:
    return cast(
        dev.TrainConfig,
        job.experimental_config.model_dump(exclude_none=True),
    )


def _moe_replay_strict(
    job: TrainJobSpec | ForwardBackwardJobSpec | ForwardJobSpec,
) -> bool:
    return job.experimental_config.moe_routing_replay_strict


def _prepare_rl_training_state(
    runtime: TrainingRuntime,
    job: (
        TrainJobSpec
        | SFTJobSpec
        | ForwardBackwardJobSpec
        | ForwardJobSpec
        | OptimizerJobSpec
    ),
) -> dict[str, torch.dtype]:
    weights_are_resident = (
        runtime.resident_run_id == job.run_id
        and runtime.resident_training_session_id == job.training_session_id
        and runtime.resident_policy_step == job.source_policy_step
        and runtime.resident_generation_id == job.source.generation_id
    )
    if weights_are_resident:
        if (
            runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("Resident Megatron state has no LoRA export metadata")
    else:
        runtime.optimizer_snapshot_barrier.synchronize()
        replacing_resident_state = runtime.resident_run_id is not None
        runtime.resident_run_id = None
        runtime.resident_training_session_id = None
        runtime.resident_policy_step = None
        runtime.resident_generation_id = None
        runtime.optimizer_state_loaded = False
        runtime.adapter_export_config = None
        if replacing_resident_state or not runtime.optimizer_persistent:
            runtime.optimizer = None

        _load_adapter_into_model(
            runtime.model,
            job.source_adapter_path,
            runtime.rank,
            handler=runtime.model_support_handler,
            optimizer=runtime.optimizer,
        )
        # Serialize the live LoRA dtype instead of perpetuating a source
        # checkpoint's PEFT-upcast FP32 dtype.
        runtime.adapter_export_dtypes = {}
        runtime.adapter_export_config = load_adapter_config(job.source_adapter_path)
        runtime.resident_run_id = job.run_id
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.source_policy_step
        runtime.resident_generation_id = job.source.generation_id

    requires_optimizer = isinstance(job, (TrainJobSpec, SFTJobSpec, OptimizerJobSpec))
    if requires_optimizer and not runtime.optimizer_state_loaded:
        if runtime.optimizer is None:
            runtime.optimizer = _build_optimizer(
                runtime.model, runtime.optimizer_config
            )
        _load_optimizer(
            runtime,
            optimizer_state_path=job.optimizer_state_path,
            adapter_path=job.source_adapter_path,
            adapter_step=job.source_policy_step,
            allow_missing=(
                job.source_policy_step == 0
                or os.environ.get(ALLOW_UNPAIRED_MEGATRON_RESUME_ENV, "").lower()
                in {"1", "true", "yes"}
            ),
        )
        runtime.optimizer_state_loaded = True
    assert runtime.adapter_export_dtypes is not None
    return runtime.adapter_export_dtypes


def _load_optimizer(
    runtime: TrainingRuntime,
    *,
    optimizer_state_path: str,
    adapter_path: str,
    adapter_step: int,
    allow_missing: bool,
) -> None:
    assert runtime.optimizer is not None
    shard_path = load_optimizer_state(
        runtime,
        optimizer_state_path=optimizer_state_path,
        adapter_path=adapter_path,
        adapter_step=adapter_step,
        allow_missing=allow_missing,
        initialize=_eager_initialize_optimizer_state,
    )
    if shard_path is None:
        print0(
            runtime.rank,
            "No committed optimizer state found at",
            optimizer_state_path,
            "- resetting optimizer for a new lineage",
        )
        return
    print0(runtime.rank, "Loading optimizer state from", shard_path)


def _load_adapter_into_model(
    model_chunks: ModelChunks,
    lora_path: str,
    rank: int,
    *,
    handler: Any | None = None,
    optimizer: Any | None = None,
) -> dict[str, torch.Tensor]:
    print0(rank, "Loading adapter model from", lora_path)
    adapter_model = load_lora_tensors_for_megatron(lora_path, handler=handler)
    load_adapter_into_model(
        model_chunks,
        adapter_model,
        optimizer,
        model_support_handler=handler,
    )
    return adapter_model


def _validate_train_step_result_finite(
    runtime: TrainingRuntime,
    step_result: TrainStepResult,
) -> None:
    finite = torch.isfinite(step_result.reduced_loss.detach()).to(dtype=torch.float32)
    if not math.isfinite(float(step_result.grad_norm)):
        finite.zero_()
    if torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
            finite,
            op=torch.distributed.ReduceOp.MIN,  # ty: ignore[possibly-missing-attribute]
        )
    if bool(finite.item()):
        return
    raise RuntimeError(
        "Megatron training produced a non-finite result; refusing to save LoRA. "
        f"loss={float(step_result.reduced_loss.detach().float().item())}, "
        f"grad_norm={step_result.grad_norm}, rank={runtime.rank}"
    )


def _should_snapshot_optimizer(
    runtime: TrainingRuntime,
    *,
    step: int,
    optimizer_save_interval: int,
    final_training_step: int | None,
) -> bool:
    return (
        not runtime.optimizer_persistent
        or optimizer_save_interval == 1
        or step <= 1
        or step % optimizer_save_interval == 0
        or (final_training_step is not None and step >= final_training_step)
    )


def _rl_step_metrics(
    step_result: TrainStepResult,
    *,
    num_gradient_steps: int,
    train_step_s: float,
) -> dict[str, float]:
    workload = step_result.workload
    metrics = {
        "loss/train": step_result.reduced_loss.item(),
        "loss/grad_norm": step_result.grad_norm,
        "loss/probs_corr": step_result.probs_corr,
        TRAIN_GRADIENT_STEPS_KEY: float(num_gradient_steps),
        "data/gradient_step_nonpadding_logical_tokens": float(
            workload.logical_nonpadding_tokens
        ),
        "data/gradient_step_loss_bearing_tokens": float(workload.loss_bearing_tokens),
        "data/gradient_step_executed_token_equivalents": float(
            workload.executed_token_equivalents
        ),
        "data/gradient_step_nominal_schedule_capacity_tokens": float(
            workload.nominal_schedule_capacity_tokens
        ),
        "data/gradient_step_dummy_executed_token_equivalents": float(
            workload.dummy_executed_token_equivalents
        ),
        "data/gradient_step_dummy_schedule_capacity_tokens": float(
            workload.dummy_schedule_capacity_tokens
        ),
        "pipeline/gradient_step_real_microbatches": float(workload.real_microbatches),
        "pipeline/gradient_step_dummy_microbatches": float(workload.dummy_microbatches),
        "time/gradient_step_train_s": train_step_s,
    }
    if step_result.kl_policy_ref is not None:
        metrics["loss/kl_policy_ref"] = step_result.kl_policy_ref
    metrics.update(step_result.loss_metrics)
    metrics.update(step_result.pipeline_metrics)
    return metrics


def _placeholder_attention_mask(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)


def load_adapter_into_model(
    model_chunks: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
    optimizer: Any | None = None,
    *,
    model_support_handler: Any | None = None,
) -> None:
    with torch.no_grad():
        for chunk in model_chunks:
            for module in chunk.modules():
                load_lora = getattr(module, "load_lora", None)
                if callable(load_lora):
                    load_lora(adapter_model)
        if model_support_handler is not None:
            model_support_handler.zero_internal_padding_params(model_chunks)

    if optimizer is None:
        return
    optimizer.reload_model_params()


def _zero_grad_buffers(model_chunks: ModelChunks) -> None:
    for chunk in model_chunks:
        zero_grad_buffer = getattr(chunk, "zero_grad_buffer", None)
        if not callable(zero_grad_buffer):
            raise TypeError(f"{type(chunk).__name__} has no zero_grad_buffer method")
        zero_grad_buffer()


def _optimizer_step(
    optimizer: Any,
    learning_rate: float,
    *,
    model_support_handler: Any | None = None,
    model_chunks: ModelChunks | None = None,
    before_step: Callable[[], None] | None = None,
) -> tuple[bool, float, int | None]:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    if model_support_handler is not None and model_chunks is not None:
        model_support_handler.zero_internal_padding_grads(model_chunks)
    if before_step is not None:
        before_step()
    update_successful, grad_norm, num_zeros_in_grad = cast(
        tuple[bool, float, int | None], optimizer.step()
    )
    if model_support_handler is not None and model_chunks is not None:
        model_support_handler.zero_internal_padding_params(model_chunks)
    optimizer.zero_grad()
    return update_successful, grad_norm, num_zeros_in_grad


def run_megatron_optimizer_step(
    *,
    optimizer: Any,
    learning_rate: float,
    model_support_handler: Any | None = None,
    model_chunks: ModelChunks | None = None,
    before_step: Callable[[], None] | None = None,
) -> MegatronOptimizerStepResult:
    update_successful, grad_norm, num_zeros_in_grad = _optimizer_step(
        optimizer,
        learning_rate,
        model_support_handler=model_support_handler,
        model_chunks=model_chunks,
        before_step=before_step,
    )
    return MegatronOptimizerStepResult(
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
    )


def _reduce_loss_sum(
    loss_sum: torch.Tensor,
    token_count: torch.Tensor,
    group: Any | None = None,
) -> torch.Tensor:
    totals = torch.stack(
        (loss_sum.detach(), token_count.to(dtype=loss_sum.dtype)),
    )
    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        totals,
        op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        group=group,
    )
    return totals[0] / totals[1].clamp_min(1.0)


def _broadcast_from_pipeline_last(value: Any) -> Any:
    if ps.get_pipeline_model_parallel_world_size() <= 1:
        return value
    objects = [value]
    torch.distributed.broadcast_object_list(  # ty: ignore[possibly-missing-attribute]
        objects,
        src=ps.get_pipeline_model_parallel_last_rank(),
        group=ps.get_pipeline_model_parallel_group(),
    )
    return objects[0]


def _unwrap_model_config(model_chunks: ModelChunks) -> Any | None:
    module: Any = model_chunks[0]
    while hasattr(module, "module"):
        module = module.module
    return getattr(module, "config", None)


def _infer_parallel_topology(model_chunks: ModelChunks) -> ParallelTopology:
    model_config = _unwrap_model_config(model_chunks)
    return ParallelTopology(
        tp=ps.get_tensor_model_parallel_world_size(),
        cp=ps.get_context_parallel_world_size(),
        dp=ps.get_data_parallel_world_size(),
        pp=ps.get_pipeline_model_parallel_world_size(),
        sp=bool(getattr(model_config, "sequence_parallel", False)),
    )


def _hybridep_token_capacity(
    packed_sequence_length: int, context_parallel_size: int
) -> int:
    from art.megatron.context_parallel.types import ContextParallelConfig

    # Reserve the normal near-balanced extent once; cost-aware CP plans can
    # exceed it, so callers also provide their exact maximum planned extent.
    planner_chunk = ContextParallelConfig().planner_chunk_size
    mean_rank_load = (
        math.ceil(packed_sequence_length / (planner_chunk * context_parallel_size))
        * planner_chunk
    )
    return min(packed_sequence_length, 2 * mean_rank_load)


def _ensure_hybridep_capacity(
    runtime: TrainingRuntime,
    *,
    packed_sequence_length: int,
    context_parallel_size: int,
    required_capacity: int = 0,
) -> None:
    expert_parallel_size = ps.get_expert_model_parallel_world_size()
    if expert_parallel_size <= 1:
        return
    from megatron.core.transformer.moe import fused_a2a

    token_capacity = max(
        _hybridep_token_capacity(packed_sequence_length, context_parallel_size),
        int(required_capacity),
    )
    current = fused_a2a._hybrid_ep_buffer
    if (
        current is not None
        and current.configurer.buffer_config.max_num_of_tokens_per_rank
        >= token_capacity
    ):
        return
    num_experts = int(runtime.provider.num_moe_experts)
    if num_experts % expert_parallel_size:
        raise RuntimeError(
            f"num_moe_experts={num_experts} is not divisible by EP={expert_parallel_size}"
        )
    fused_a2a.reset_hybrid_ep_buffer()
    fused_a2a.init_hybrid_ep_buffer(
        group=ps.get_expert_tensor_and_model_parallel_group(),
        hidden_dim=int(runtime.provider.hidden_size),
        seq_len=token_capacity,
        num_local_experts=num_experts // expert_parallel_size,
        num_sms_dispatch_api=int(runtime.provider.moe_hybridep_num_sms),
        num_sms_combine_api=int(runtime.provider.moe_hybridep_num_sms),
        fp8_dispatch=False,
    )


def select_micro_ref_logprobs(
    ref_logprobs_by_index: dict[int, torch.Tensor],
    sample_indices: list[int | None],
    zero_template: PackedTensors,
) -> list[torch.Tensor]:
    zero_ref_logprobs = torch.zeros_like(zero_template["tokens"], dtype=torch.float32)
    return [
        zero_ref_logprobs.clone()
        if sample_index is None
        else ref_logprobs_by_index[sample_index]
        for sample_index in sample_indices
    ]


def _select_next_step_first_ref_logprobs(
    *,
    ref_logprobs_by_index: dict[int, torch.Tensor],
    zero_template: PackedTensors,
    step_index: int,
    num_steps: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int,
) -> torch.Tensor | None:
    next_step_index = step_index + 1
    if next_step_index >= num_steps:
        return None
    next_micro_indices = build_micro_sample_indices(
        step_index=next_step_index,
        num_sequences=num_sequences,
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
    )
    return select_micro_ref_logprobs(
        ref_logprobs_by_index,
        [next_micro_indices[0]],
        zero_template,
    )[0]


def _select_ref_logprobs(
    ref_logprobs: torch.Tensor | list[torch.Tensor] | None,
    micro_order: int,
) -> torch.Tensor | None:
    if isinstance(ref_logprobs, list):
        return ref_logprobs[micro_order]
    return ref_logprobs


def _select_next_ref_logprobs(
    ref_logprobs: torch.Tensor | list[torch.Tensor] | None,
    *,
    micro_order: int,
    micro_count: int,
    next_step_first_ref_logprobs: torch.Tensor | None,
) -> torch.Tensor | None:
    if isinstance(ref_logprobs, list):
        if micro_order + 1 < len(ref_logprobs):
            return ref_logprobs[micro_order + 1]
        return next_step_first_ref_logprobs
    if micro_order + 1 >= micro_count and next_step_first_ref_logprobs is not None:
        return next_step_first_ref_logprobs
    return ref_logprobs


def _forward_prepared_rl_micro(
    *,
    model_chunks: ModelChunks,
    model_chunk: MegatronModule | None = None,
    model_support_handler: Any,
    prepared_micro: PreparedRLMicroInputs,
    device: torch.device,
) -> TokenLossOutput:
    model = model_chunks[0] if model_chunk is None else model_chunk
    model_forward_kwargs = dict(
        input_ids=prepared_micro.model_tokens,
        position_ids=prepared_micro.model_input_pos,
        attention_mask=_placeholder_attention_mask(device),
        packed_seq_params=prepared_micro.packed_seq_params,
        **model_support_handler.get_forward_kwargs(
            model,
            attention_bias=prepared_micro.attention_state,
        ),
    )
    with attach_trace_token_uids(model_chunks, prepared_micro.local_token_uids):
        if chunk_post_process(model):
            return forward_token_losses(
                model,
                labels=prepared_micro.model_labels,
                selection=prepared_micro.lm_head_selection,
                forward_kwargs=model_forward_kwargs,
            )
        output = model(**model_forward_kwargs, labels=None)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"pipeline model chunk must return a tensor, got {type(output).__name__}"
            )
        return TokenLossOutput(token_losses=output)


def _defer_model_grad_finalization(
    model: list[MegatronModule],
    num_tokens: torch.Tensor | None = None,
    **kwargs: Any,
) -> None:
    del model, num_tokens, kwargs


def _install_schedule_finalize(
    model_chunks: ModelChunks, *, defer_grad_sync: bool = False
) -> None:
    seen: set[int] = set()
    for chunk in model_chunks:
        config = _unwrap_model_config([chunk])
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        config.finalize_model_grads_func = (
            _defer_model_grad_finalization
            if defer_grad_sync
            else finalize_model_grads_extended
        )


def _zero_logprob_graph_contribution(
    new_logprobs: torch.Tensor,
    loss_inputs: LossInputs | AlignedLossInputs,
) -> torch.Tensor:
    assistant_mask = loss_inputs.align_inputs().assistant_mask.to(dtype=torch.bool)
    return new_logprobs.masked_fill(~assistant_mask, 0.0).sum() * 0.0


def _globalize_context_parallel_logprob_batch(
    *,
    local_logprobs: list[torch.Tensor],
    attention_states: list[Any],
    sequence_lengths: list[int],
    empty_template: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    if len(attention_states) != len(sequence_lengths):
        raise ValueError("Context-parallel logprob/state/length counts differ")
    if local_logprobs and len(local_logprobs) != len(attention_states):
        raise ValueError("Context-parallel logprob/state/length counts differ")
    if not attention_states:
        return []
    if local_logprobs:
        template = local_logprobs[0]
        trailing_shape = tuple(template.shape[2:])
    else:
        if empty_template is None:
            raise ValueError("Empty context-parallel logprobs require a template")
        template = empty_template
        trailing_shape = ()
    rows = template.new_zeros(
        (len(local_logprobs), max(sequence_lengths), *trailing_shape)
        if local_logprobs
        else (len(attention_states), max(sequence_lengths))
    )
    cp_group = None
    for index, (attention_state, seq_len) in enumerate(
        zip(attention_states, sequence_lengths, strict=True)
    ):
        rank_plan = getattr(attention_state, "rank_plan", None)
        micro_cp_group = getattr(attention_state, "cp_group", None)
        if rank_plan is None or micro_cp_group is None:
            raise RuntimeError(
                "Context-parallel reference logprobs require a rank plan"
            )
        if cp_group is not None and micro_cp_group is not cp_group:
            raise RuntimeError(
                "Context-parallel microbatches use different process groups"
            )
        cp_group = micro_cp_group
        if not local_logprobs:
            continue
        values = local_logprobs[index]
        if tuple(values.shape[2:]) != trailing_shape:
            raise ValueError("Context-parallel logprob trailing shapes differ")
        local_values = values.reshape(-1, *trailing_shape)
        cursor = 0
        for range_ in rank_plan.local_row_ranges:
            if range_ is None:
                continue
            size = int(range_.size())
            if size <= 0:
                continue
            rows[index, int(range_.start) : int(range_.end)] = local_values[
                cursor : cursor + size
            ]
            cursor += size
        if cursor != int(local_values.shape[0]):
            raise RuntimeError(
                "Context-parallel reference-logprob layout did not consume all values: "
                f"consumed={cursor}, values={local_values.shape[0]}"
            )
    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        rows, group=cp_group
    )
    return [
        rows[index : index + 1, :length]
        for index, length in enumerate(sequence_lengths)
    ]


def _globalize_data_parallel_logprob_batch(
    *,
    local_logprobs: list[torch.Tensor],
    sample_indices: list[int | None],
    step_index: int,
    global_grad_accumulation_sequences: int,
) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    if len(local_logprobs) != len(sample_indices):
        raise ValueError("Data-parallel logprob/sample counts differ")
    shape = tuple(local_logprobs[0].shape[1:])
    values = local_logprobs[0].new_zeros((global_grad_accumulation_sequences, *shape))
    present = torch.zeros(
        global_grad_accumulation_sequences,
        device=values.device,
        dtype=torch.int32,
    )
    for sample_index, logprobs in zip(sample_indices, local_logprobs, strict=True):
        if sample_index is None:
            continue
        if tuple(logprobs.shape[1:]) != shape or int(logprobs.shape[0]) != 1:
            raise RuntimeError("tokenized DP logprob shapes differ")
        offset = sample_index % global_grad_accumulation_sequences
        values[offset] = logprobs[0]
        present[offset] = 1
    if ps.get_data_parallel_world_size() > 1:
        group = ps.get_data_parallel_group()
        torch.distributed.all_reduce(values, group=group)
        torch.distributed.all_reduce(present, group=group)
    if bool(torch.any(present > 1).item()):
        raise RuntimeError("multiple DP ranks returned the same packed sequence")
    offsets = torch.nonzero(present).flatten().cpu().tolist()
    if offsets != list(range(len(offsets))):
        raise RuntimeError("tokenized DP results are not a contiguous step prefix")
    host = values.cpu()
    return (
        [host[index : index + 1] for index in offsets],
        tuple(
            step_index * global_grad_accumulation_sequences + index for index in offsets
        ),
    )


@torch.no_grad()
def _calculate_megatron_logprob_batch(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: list[PackedTensors],
    sample_indices: list[int | None],
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    step_index: int | None = None,
    hybridep_token_counts: list[int] | None = None,
) -> list[torch.Tensor]:
    if not inputs or len(inputs) != len(sample_indices):
        raise ValueError("Reference input/sample counts must match and be nonzero")
    if moe_routing_replay_controller is not None:
        if step_index is None:
            raise ValueError("step_index is required for routing replay")
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=(
                sample_indices[0] if len(sample_indices) == 1 else sample_indices
            ),
        )

    device = next(model_chunks[0].parameters()).device
    topology = _infer_parallel_topology(model_chunks)
    trace_token_uids = context_parallel_trace_token_uids_enabled(
        topology,
        moe_routing_replay_controller,
    )
    previous_training_modes = [chunk.training for chunk in model_chunks]
    for chunk in model_chunks:
        chunk.eval()
    forward_succeeded = False
    try:
        pending_prepared_micro: PreparedMegatronBatch | None = None
        prepared_micros: list[PreparedRLMicroInputs] = []
        for order, micro in enumerate(inputs):
            prepared, pending_prepared_micro = _prepare_current_rl_micro(
                micro,
                device=device,
                topology=topology,
                provider=provider,
                model_support_handler=model_support_handler,
                ref_logprobs=None,
                trace_token_uids=trace_token_uids,
                pending_prepared_micro=pending_prepared_micro,
            )
            prepared_micros.append(prepared)
            pending_prepared_micro = _prepare_next_rl_cp_micro(
                _next_micro_lookahead(inputs, order),
                device=device,
                topology=topology,
                provider=provider,
                model_support_handler=model_support_handler,
                trace_token_uids=trace_token_uids,
                ref_logprobs=None,
            )
        microbatch_state = PipelineMicrobatchState(
            controller=moe_routing_replay_controller,
            hybridep_token_counts=hybridep_token_counts,
            microbatch_count=len(prepared_micros),
            model_activator=model_support_handler.build_pipeline_microbatch_activator(
                model_chunks
            ),
        )
        if not ps.model_parallel_is_initialized():
            # Unit/static callers do not have MCore process groups. Production
            # reference forwards always take the common schedule path below.
            if len(prepared_micros) != 1:
                raise RuntimeError("Static reference forward accepts one microbatch")
            prepared = prepared_micros[0]
            microbatch_state.activate(
                ScheduleMicrobatch(
                    0, sample_indices[0], prepared, prepared.attention_state
                ),
                chunk_index=0,
            )
            token_output = forward_token_losses(
                model_chunks[0],
                labels=prepared.model_labels,
                selection=prepared.lm_head_selection,
                forward_kwargs=dict(
                    input_ids=prepared.model_tokens,
                    position_ids=prepared.model_input_pos,
                    attention_mask=_placeholder_attention_mask(device),
                    packed_seq_params=prepared.packed_seq_params,
                    **model_support_handler.get_forward_kwargs(
                        model_chunks[0], attention_bias=prepared.attention_state
                    ),
                ),
                enabled=False,
            )
            forward_succeeded = True
            return [token_output.restore(-token_output.token_losses).detach().cpu()]
        schedule = MCoreScheduleAdapter(
            model_chunks=model_chunks,
            prepared_microbatches=prepared_micros,
            sample_indices=sample_indices,
            model_inputs=[prepared.model_tokens for prepared in prepared_micros],
            moe_routing_replay_controller=moe_routing_replay_controller,
            hybridep_token_counts=hybridep_token_counts,
            model_activator=model_support_handler.build_pipeline_microbatch_activator(
                model_chunks
            ),
        )

        def forward_step_func(data_iterator: Any, model: MegatronModule, *_args: Any):
            item = next(data_iterator)
            token_output = _forward_prepared_rl_micro(
                model_chunks=model_chunks,
                model_chunk=model,
                model_support_handler=model_support_handler,
                prepared_micro=item.payload,
                device=device,
            )

            def collect(output_tensor: torch.Tensor, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "order": item.order,
                    "logprobs": token_output.restore(-output_tensor).detach(),
                }

            return token_output.token_losses, collect

        forward_outputs = schedule.run(
            forward_step_func,
            forward_only=True,
            collect_non_loss_data=True,
        )
        if not any(chunk_post_process(chunk) for chunk in model_chunks):
            forward_succeeded = True
            return []
        outputs = cast(list[dict[str, Any]], forward_outputs)
        if len(outputs) != len(prepared_micros):
            raise RuntimeError(
                "Reference pipeline did not return one result per microbatch: "
                f"expected={len(prepared_micros)}, got={len(outputs)}"
            )
        outputs.sort(key=lambda output: int(output["order"]))
        logprobs = [cast(torch.Tensor, output["logprobs"]) for output in outputs]
        if int(topology.cp) > 1:
            logprobs = _globalize_context_parallel_logprob_batch(
                local_logprobs=logprobs,
                attention_states=[
                    prepared.attention_state for prepared in prepared_micros
                ],
                sequence_lengths=[
                    int(inputs[0]["tokens"].shape[1]) for _ in prepared_micros
                ],
            )
        host_logprobs = torch.cat(logprobs).detach().cpu()
        forward_succeeded = True
        return list(host_logprobs.split(1))
    finally:
        for chunk, was_training in zip(model_chunks, previous_training_modes):
            chunk.train(was_training)
        if moe_routing_replay_controller is not None and forward_succeeded:
            moe_routing_replay_controller.finalize_step()


def _update_fingerprint(digest: Any, value: str | bytes) -> None:
    payload = value.encode() if isinstance(value, str) else value
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_tensor_fingerprint(digest: Any, name: str, value: Any) -> None:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    _update_fingerprint(digest, name)
    _update_fingerprint(digest, str(tuple(tensor.shape)))
    _update_fingerprint(digest, str(tensor.dtype))
    _update_fingerprint(digest, tensor.numpy().tobytes())


def _packed_batch_fingerprint(packed_tensors: PackedTensors) -> str:
    digest = hashlib.sha256()
    for name in ("tokens", "group_ids", "parent_ids", "input_pos", "assistant_mask"):
        _update_tensor_fingerprint(digest, name, packed_tensors[name])
    replay = packed_tensors.get("moe_routing_replay")
    if replay is not None:
        _update_tensor_fingerprint(digest, "moe_routing_replay", replay.expert_indices)
    return digest.hexdigest()


@contextmanager
def _preserve_diagnostic_rng(device: torch.device) -> Iterator[None]:
    python_state = random.getstate()
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)


@contextmanager
def _temporary_resident_replay(
    runtime: TrainingRuntime,
    packed_tensors: PackedTensors,
    *,
    global_grad_accumulation_sequences: int,
) -> Iterator[tuple[MoeRoutingReplayController | None, int]]:
    packed_replay = packed_tensors.get("moe_routing_replay")
    if packed_replay is None:
        if bool(getattr(runtime.model_support_handler, "is_moe", False)):
            raise RuntimeError("resident MoE scoring requires packed routing replay")
        yield None, 0
        return

    bundle = build_moe_routing_replay_bundle_from_packed_tensors(
        packed_tensors=packed_tensors,
        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
    )
    packed_tokens = int(packed_replay.pack_stats.packed_tokens)
    previous = runtime.moe_routing_replay_controller
    if previous is None:
        configure_moe_routing_replay(runtime, replay_bundle=bundle, strict=True)
        controller = runtime.moe_routing_replay_controller
        assert controller is not None
        try:
            yield controller, packed_tokens
        finally:
            controller.remove_router_patches()
            runtime.moe_routing_replay_controller = None
        return

    if getattr(previous, "_active_step_index", None) is not None:
        raise RuntimeError("resident routing replay is active during score dispatch")
    previous_bundle = previous.bundle
    previous_strict = previous.strict
    previous.update_bundle(bundle=bundle, strict=True)
    try:
        yield previous, packed_tokens
    finally:
        previous.update_bundle(bundle=previous_bundle, strict=previous_strict)


def _vocab_parallel_token_scores(
    local_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if local_logits.ndim != 2 or labels.shape != local_logits.shape[:1]:
        raise ValueError(
            "resident score logits/labels do not align: "
            f"logits={tuple(local_logits.shape)} labels={tuple(labels.shape)}"
        )
    local_logits = local_logits.float()
    tp_size = int(ps.get_tensor_model_parallel_world_size())
    tp_rank = int(ps.get_tensor_model_parallel_rank())
    group = ps.get_tensor_model_parallel_group(check_initialized=False)

    local_max = local_logits.max(dim=-1).values
    global_max = local_max.clone()
    if tp_size > 1:
        torch.distributed.all_reduce(
            global_max,
            op=torch.distributed.ReduceOp.MAX,  # ty: ignore[possibly-missing-attribute]
            group=group,
        )
    local_exp_sum = torch.exp(local_logits - global_max.unsqueeze(1)).sum(dim=-1)
    global_exp_sum = local_exp_sum.clone()
    if tp_size > 1:
        torch.distributed.all_reduce(
            global_exp_sum,
            op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
            group=group,
        )
    log_z = global_max + torch.log(global_exp_sum)

    local_vocab = int(local_logits.shape[1])
    vocab_start = tp_rank * local_vocab
    local_labels = labels - vocab_start
    owns_target = (labels >= 0) & (local_labels >= 0) & (local_labels < local_vocab)
    rows = torch.arange(labels.numel(), device=labels.device)
    target_logits = local_logits[
        rows, local_labels.clamp(0, local_vocab - 1)
    ].masked_fill(~owns_target, 0.0)
    if tp_size > 1:
        torch.distributed.all_reduce(
            target_logits,
            op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
            group=group,
        )

    local_k = min(top_k, local_vocab)
    local_values, local_ids = torch.topk(local_logits, k=local_k, dim=-1)
    local_ids += vocab_start
    if tp_size > 1:
        gathered_values = [torch.empty_like(local_values) for _ in range(tp_size)]
        gathered_ids = [torch.empty_like(local_ids) for _ in range(tp_size)]
        torch.distributed.all_gather(gathered_values, local_values, group=group)
        torch.distributed.all_gather(gathered_ids, local_ids, group=group)
        candidate_values = torch.cat(gathered_values, dim=-1)
        candidate_ids = torch.cat(gathered_ids, dim=-1)
    else:
        candidate_values = local_values
        candidate_ids = local_ids
    if int(candidate_values.shape[1]) < top_k:
        raise ValueError("resident score top_k exceeds the padded vocabulary")
    top_values, top_offsets = torch.topk(candidate_values, k=top_k, dim=-1)
    top_ids = candidate_ids.gather(1, top_offsets)
    return target_logits - log_z, top_values - log_z.unsqueeze(1), top_ids


def _local_packed_token_scores(
    *,
    local_logits: torch.Tensor,
    prepared: PreparedRLMicroInputs,
    sample_index: int | None,
    top_k: int,
) -> tuple[PackedTokenScore, ...]:
    selection = prepared.lm_head_selection
    labels = selection.select(prepared.model_labels).reshape(-1)
    token_uids = prepared.local_token_uids
    if token_uids is None:
        raise RuntimeError("resident scoring requires packed token UIDs")
    uid_indices = selection.flat_indices.to(device=token_uids.device)
    selected_uids = token_uids.reshape(-1).index_select(0, uid_indices)
    target_logprobs, top_logprobs, top_ids = _vocab_parallel_token_scores(
        local_logits,
        labels,
        top_k=top_k,
    )
    if sample_index is None or int(ps.get_tensor_model_parallel_rank()) != 0:
        return ()

    labels_cpu = labels.detach().cpu()
    uids_cpu = selected_uids.detach().cpu()
    target_cpu = target_logprobs.detach().cpu()
    top_logprobs_cpu = top_logprobs.detach().cpu()
    top_ids_cpu = top_ids.detach().cpu()
    scores = []
    for index in range(int(labels_cpu.numel())):
        target_token_id = int(labels_cpu[index].item())
        logit_index = int(uids_cpu[index].item())
        if target_token_id < 0 or logit_index < 0:
            continue
        scores.append(
            PackedTokenScore(
                sample_index=sample_index,
                logit_index=logit_index,
                target_token_id=target_token_id,
                target_logprob=float(target_cpu[index].item()),
                top_token_ids=tuple(
                    int(value) for value in top_ids_cpu[index].tolist()
                ),
                top_logprobs=tuple(
                    float(value) for value in top_logprobs_cpu[index].tolist()
                ),
            )
        )
    return tuple(scores)


def _forward_prepared_score_micro(
    *,
    model_chunks: ModelChunks,
    model_chunk: MegatronModule,
    model_support_handler: Any,
    prepared: PreparedRLMicroInputs,
    device: torch.device,
) -> torch.Tensor:
    forward_kwargs = dict(
        input_ids=prepared.model_tokens,
        position_ids=prepared.model_input_pos,
        attention_mask=_placeholder_attention_mask(device),
        packed_seq_params=prepared.packed_seq_params,
        **model_support_handler.get_forward_kwargs(
            model_chunk,
            attention_bias=prepared.attention_state,
        ),
    )
    with attach_trace_token_uids(model_chunks, prepared.local_token_uids):
        if chunk_post_process(model_chunk):
            return forward_token_logits(
                model_chunk,
                selection=prepared.lm_head_selection,
                forward_kwargs=forward_kwargs,
            )
        output = model_chunk(**forward_kwargs, labels=None)
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"pipeline model chunk must return a tensor, got {type(output).__name__}"
        )
    return output


@torch.no_grad()
def _calculate_megatron_score_batch(
    *,
    runtime: TrainingRuntime,
    inputs: list[PackedTensors],
    sample_indices: list[int | None],
    step_index: int,
    top_k: int,
    controller: MoeRoutingReplayController | None,
    hybridep_token_counts: list[int] | None,
) -> tuple[PackedTokenScore, ...]:
    if not ps.model_parallel_is_initialized():
        raise RuntimeError("resident scoring requires initialized model parallelism")
    if not inputs or len(inputs) != len(sample_indices):
        raise ValueError("resident score input/sample counts must match and be nonzero")
    if controller is not None:
        controller.set_step(step_index=step_index, sample_index=sample_indices)

    model_chunks = runtime.model
    device = next(model_chunks[0].parameters()).device
    topology = _infer_parallel_topology(model_chunks)
    modules = {
        id(module): module for chunk in model_chunks for module in chunk.modules()
    }
    previous_training_modes = {
        module_id: module.training for module_id, module in modules.items()
    }
    for chunk in model_chunks:
        chunk.eval()
    forward_succeeded = False
    try:
        pending_prepared_micro: PreparedMegatronBatch | None = None
        prepared_micros: list[PreparedRLMicroInputs] = []
        for order, micro in enumerate(inputs):
            prepared, pending_prepared_micro = _prepare_current_rl_micro(
                micro,
                device=device,
                topology=topology,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                ref_logprobs=None,
                trace_token_uids=True,
                pending_prepared_micro=pending_prepared_micro,
            )
            prepared_micros.append(prepared)
            pending_prepared_micro = _prepare_next_rl_cp_micro(
                _next_micro_lookahead(inputs, order),
                device=device,
                topology=topology,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                trace_token_uids=True,
            )
        schedule = MCoreScheduleAdapter(
            model_chunks=model_chunks,
            prepared_microbatches=prepared_micros,
            sample_indices=sample_indices,
            model_inputs=[prepared.model_tokens for prepared in prepared_micros],
            moe_routing_replay_controller=controller,
            hybridep_token_counts=hybridep_token_counts,
            model_activator=runtime.model_support_handler.build_pipeline_microbatch_activator(
                model_chunks
            ),
        )

        def forward_step_func(data_iterator: Any, model: MegatronModule, *_args: Any):
            item = next(data_iterator)
            output = _forward_prepared_score_micro(
                model_chunks=model_chunks,
                model_chunk=model,
                model_support_handler=runtime.model_support_handler,
                prepared=item.payload,
                device=device,
            )

            def collect(output_tensor: torch.Tensor, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "order": item.order,
                    "scores": _local_packed_token_scores(
                        local_logits=output_tensor,
                        prepared=item.payload,
                        sample_index=item.sample_index,
                        top_k=top_k,
                    ),
                }

            return output, collect

        forward_outputs = schedule.run(
            forward_step_func,
            forward_only=True,
            collect_non_loss_data=True,
        )
        if not any(chunk_post_process(chunk) for chunk in model_chunks):
            forward_succeeded = True
            return ()
        outputs = cast(list[dict[str, Any]], forward_outputs)
        if len(outputs) != len(prepared_micros):
            raise RuntimeError(
                "resident score pipeline did not return every microbatch: "
                f"expected={len(prepared_micros)}, got={len(outputs)}"
            )
        outputs.sort(key=lambda output: int(output["order"]))
        forward_succeeded = True
        return tuple(
            score
            for output in outputs
            for score in cast(tuple[PackedTokenScore, ...], output["scores"])
        )
    finally:
        for module_id, module in modules.items():
            module.training = previous_training_modes[module_id]
        if controller is not None and forward_succeeded:
            controller.finalize_step()


def execute_megatron_score_job(
    runtime: TrainingRuntime,
    job: ResidentScoreJobSpec,
    packed_tensors: PackedTensors,
) -> ResidentScoreShard:
    """Score one packed batch against the exact resident learner without mutation."""
    global_accumulation = resolve_global_grad_accumulation_sequences(
        job.global_grad_accumulation_sequences
    )
    num_sequences, packed_sequence_length = map(int, packed_tensors["tokens"].shape)
    num_steps = math.ceil(num_sequences / global_accumulation)
    topology = _infer_parallel_topology(runtime.model)
    template = _clone_packed_tensors(select_indexed_inputs(packed_tensors, 0))
    zero_template = _zero_contribution_inputs(template)
    hybridep_token_counts_by_step = (
        [
            build_rl_hybridep_token_counts(
                packed_tensors=packed_tensors,
                step_index=step_index,
                num_sequences=num_sequences,
                global_grad_accumulation_sequences=global_accumulation,
                topology=topology,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
            )
            for step_index in range(num_steps)
        ]
        if ps.get_expert_model_parallel_world_size() > 1
        else None
    )
    _ensure_hybridep_capacity(
        runtime,
        packed_sequence_length=packed_sequence_length,
        context_parallel_size=topology.cp,
        required_capacity=max(
            (
                count
                for step_counts in hybridep_token_counts_by_step or ()
                for count in step_counts
            ),
            default=0,
        ),
    )
    device = next(runtime.model[0].parameters()).device
    scores: list[PackedTokenScore] = []
    with (
        runtime.model_support_handler.preserve_pipeline_microbatch_activation(
            runtime.model
        ),
        _preserve_diagnostic_rng(device),
        _temporary_resident_replay(
            runtime,
            packed_tensors,
            global_grad_accumulation_sequences=global_accumulation,
        ) as (controller, replay_tokens),
    ):
        for step_index in range(num_steps):
            micro_indices = build_micro_sample_indices(
                step_index=step_index,
                num_sequences=num_sequences,
                global_grad_accumulation_sequences=global_accumulation,
            )
            scores.extend(
                _calculate_megatron_score_batch(
                    runtime=runtime,
                    inputs=select_micro_inputs(
                        packed_tensors, micro_indices, zero_template
                    ),
                    sample_indices=micro_indices,
                    step_index=step_index,
                    top_k=job.top_k,
                    controller=controller,
                    hybridep_token_counts=(
                        None
                        if hybridep_token_counts_by_step is None
                        else hybridep_token_counts_by_step[step_index]
                    ),
                )
            )
    scores.sort(key=lambda score: (score.sample_index, score.logit_index))
    expected_score_count = int(packed_tensors["assistant_mask"][:, 1:].sum().item())
    if expected_score_count < 1:
        raise ValueError("resident scoring requires at least one assistant target")
    return ResidentScoreShard(
        rank=runtime.rank,
        job_id=job.job_id,
        run_id=job.run_id,
        learner=job.learner,
        batch_id=job.batch.batch_id,
        batch_fingerprint=_packed_batch_fingerprint(packed_tensors),
        top_k=job.top_k,
        expected_score_count=expected_score_count,
        routing_replay_packed_tokens=replay_tokens,
        scores=tuple(scores),
    )


@torch.no_grad()
def inspect_resident_lora(
    runtime: TrainingRuntime,
    request: ResidentLoraInspectionSpec,
) -> ResidentLoraInspectionShard:
    """Inspect resident LoRA wrappers and export coverage without changing state."""
    modules: dict[int, LoRA] = {}
    prefixes: set[str] = set()
    lora_parameter_ids: set[int] = set()
    for chunk in runtime.model:
        for module in chunk.modules():
            if not isinstance(module, LoRA) or id(module) in modules:
                continue
            modules[id(module)] = module
            prefixes.add(module.adapter_model_prefix)
            lora_parameter_ids.update(
                id(parameter) for parameter in module.parameters()
            )

    trainable_lora_names: set[str] = set()
    unexpected_trainable_names: set[str] = set()
    trainable_numel = 0
    seen_parameters: set[int] = set()
    for chunk_index, chunk in enumerate(runtime.model):
        for name, parameter in chunk.named_parameters():
            parameter_id = id(parameter)
            if not parameter.requires_grad or parameter_id in seen_parameters:
                continue
            seen_parameters.add(parameter_id)
            qualified_name = f"chunk_{chunk_index}.{name}"
            trainable_numel += int(parameter.numel())
            if parameter_id in lora_parameter_ids:
                trainable_lora_names.add(qualified_name)
            else:
                unexpected_trainable_names.add(qualified_name)

    device = next(runtime.model[0].parameters()).device
    with _preserve_diagnostic_rng(device):
        exported = runtime.model_support_handler.build_adapter_weights_by_base(
            runtime.model
        )
    exports = tuple(
        ResidentLoraExport(
            base_name=base_name,
            adapter_keys=tuple(
                sorted(
                    {getattr(weight, "adapter_key", None) for weight in weights},
                    key=lambda value: "" if value is None else value,
                )
            ),
        )
        for base_name, weights in sorted(exported.items())
    )
    return ResidentLoraInspectionShard(
        rank=runtime.rank,
        request_id=request.request_id,
        run_id=request.run_id,
        learner=request.learner,
        target_modules=request.target_modules,
        module_count=len(modules),
        wrapped_adapter_prefixes=tuple(sorted(prefixes)),
        exports=exports,
        trainable_lora_parameter_names=tuple(sorted(trainable_lora_names)),
        unexpected_trainable_parameter_names=tuple(sorted(unexpected_trainable_names)),
        trainable_numel=trainable_numel,
    )


def _calculate_megatron_logprobs(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: PackedTensors,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    step_index: int | None = None,
    sample_index: int | None = None,
    hybridep_token_count: int | None = None,
) -> torch.Tensor:
    results = _calculate_megatron_logprob_batch(
        model_chunks=model_chunks,
        provider=provider,
        model_support_handler=model_support_handler,
        inputs=[inputs],
        sample_indices=[sample_index],
        moe_routing_replay_controller=moe_routing_replay_controller,
        step_index=step_index,
        hybridep_token_counts=(
            None if hybridep_token_count is None else [hybridep_token_count]
        ),
    )
    if len(results) != 1:
        raise RuntimeError("Single reference forward did not run on the loss stage")
    return results[0]


def _precompute_reference_logprobs(
    *,
    runtime: TrainingRuntime,
    packed_tensors: PackedTensors,
    sample_step_indices: dict[int, int],
    global_grad_accumulation_sequences: int,
) -> dict[int, torch.Tensor]:
    print0(
        runtime.rank,
        "Precomputing KL reference logprobs for",
        len(sample_step_indices),
        "local sequences",
    )
    hybridep_enabled = ps.get_expert_model_parallel_world_size() > 1
    topology = _infer_parallel_topology(runtime.model) if hybridep_enabled else None
    results: dict[int, torch.Tensor] = {}
    if not ps.model_parallel_is_initialized():
        for sample_index, step_index in sorted(sample_step_indices.items()):
            results[sample_index] = _calculate_megatron_logprobs(
                model_chunks=runtime.model,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                inputs=select_indexed_inputs(packed_tensors, sample_index),
                moe_routing_replay_controller=runtime.moe_routing_replay_controller,
                step_index=step_index,
                sample_index=sample_index,
                hybridep_token_count=None,
            )
        return results

    num_sequences = int(packed_tensors["tokens"].shape[0])
    zero_template = _zero_contribution_inputs(
        _clone_packed_tensors(select_indexed_inputs(packed_tensors, 0))
    )
    for step_index in sorted(set(sample_step_indices.values())):
        micro_indices = build_micro_sample_indices(
            step_index=step_index,
            num_sequences=num_sequences,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        hybridep_token_counts = None
        if hybridep_enabled:
            assert topology is not None
            hybridep_token_counts = build_rl_hybridep_token_counts(
                packed_tensors=packed_tensors,
                step_index=step_index,
                num_sequences=num_sequences,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                topology=topology,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
            )
        outputs = _calculate_megatron_logprob_batch(
            model_chunks=runtime.model,
            provider=runtime.provider,
            model_support_handler=runtime.model_support_handler,
            inputs=select_micro_inputs(packed_tensors, micro_indices, zero_template),
            sample_indices=micro_indices,
            moe_routing_replay_controller=runtime.moe_routing_replay_controller,
            step_index=step_index,
            hybridep_token_counts=hybridep_token_counts,
        )
        if not outputs:
            continue
        for sample_index, output in zip(micro_indices, outputs, strict=True):
            if sample_index is not None:
                if sample_step_indices.get(sample_index) != step_index:
                    raise RuntimeError(
                        "Reference microbatch does not match its planned training step: "
                        f"sample={sample_index}, step={step_index}"
                    )
                results[sample_index] = output
    if any(chunk_post_process(chunk) for chunk in runtime.model) and set(
        results
    ) != set(sample_step_indices):
        raise RuntimeError("Reference forward did not materialize every local sample")
    return results


def _reference_sample_step_indices(
    *,
    num_sequences: int,
    num_steps: int,
    global_grad_accumulation_sequences: int,
) -> dict[int, int]:
    return {
        sample_index: step_index
        for step_index in range(num_steps)
        for sample_index in build_micro_sample_indices(
            step_index=step_index,
            num_sequences=num_sequences,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        if sample_index is not None
    }


def _prepare_kl_reference_logprobs(
    *,
    runtime: TrainingRuntime,
    job: TrainJobSpec | ForwardBackwardJobSpec | ForwardJobSpec,
    packed_tensors: PackedTensors,
    num_sequences: int,
    num_steps: int,
    global_grad_accumulation_sequences: int,
) -> dict[int, torch.Tensor] | None:
    if job.config.kl_penalty_coef <= 0.0:
        return None

    ref_adapter_path = _experimental_train_config(job).get("kl_ref_adapter_path")
    if ref_adapter_path is None:
        raise RuntimeError(
            "KL penalty is enabled but no kl_ref_adapter_path was provided. "
            "Megatron training requires an explicit reference LoRA path; pass "
            "kl_penalty_reference_step=0 for the identity/base reference or "
            "provide kl_ref_adapter_path."
        )

    current_adapter_path = job.source_adapter_path
    adapter_swapped = os.path.abspath(ref_adapter_path) != os.path.abspath(
        current_adapter_path
    )
    loaded_ref_adapter = False
    restore_parameters: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    try:
        if adapter_swapped:
            restore_parameters = _snapshot_trainable_parameters(runtime.model)
            _load_adapter_into_model(
                runtime.model,
                ref_adapter_path,
                runtime.rank,
                handler=runtime.model_support_handler,
            )
            loaded_ref_adapter = True
        return _precompute_reference_logprobs(
            runtime=runtime,
            packed_tensors=packed_tensors,
            sample_step_indices=_reference_sample_step_indices(
                num_sequences=num_sequences,
                num_steps=num_steps,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            ),
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
    finally:
        if loaded_ref_adapter:
            assert runtime.optimizer is not None
            assert restore_parameters is not None
            with torch.no_grad():
                for parameter, value in restore_parameters:
                    parameter.copy_(value)
                runtime.model_support_handler.zero_internal_padding_params(
                    runtime.model
                )
            runtime.optimizer_snapshot_barrier.synchronize()
            runtime.optimizer.reload_model_params()


def _snapshot_trainable_parameters(
    model_chunks: ModelChunks,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    seen: set[int] = set()
    snapshot: list[tuple[torch.Tensor, torch.Tensor]] = []
    for chunk in model_chunks:
        for parameter in chunk.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                snapshot.append((parameter, parameter.detach().clone()))
    return snapshot


def _run_megatron_sft_schedule(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    step_index: int,
    sample_index: int | list[int | None],
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    hybridep_token_counts: list[int] | None = None,
    defer_grad_sync: bool = False,
    forward_only: bool,
    return_token_logprobs: bool,
) -> SFTForwardBackwardState:
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("SFT forward/backward requires at least one trajectory")

    micro_sample_indices: list[int | None]
    if isinstance(sample_index, list):
        if len(sample_index) != len(micro_inputs):
            raise ValueError(
                "sample_index list length must match number of micro inputs: "
                f"{len(sample_index)} != {len(micro_inputs)}"
            )
        micro_sample_indices = [
            int(index) if index is not None else None for index in sample_index
        ]
    else:
        assert len(micro_inputs) == 1
        micro_sample_indices = [sample_index]

    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=micro_sample_indices,
        )

    topology = _infer_parallel_topology(model_chunks)
    device = next(model_chunks[0].parameters()).device
    trace_token_uids = context_parallel_trace_token_uids_enabled(
        topology,
        moe_routing_replay_controller,
    )

    if not forward_only:
        _zero_grad_buffers(model_chunks)
        _install_schedule_finalize(model_chunks, defer_grad_sync=defer_grad_sync)

    pending_prepared_micro: PreparedMegatronBatch | None = None
    prepared_micros: list[PreparedSFTMicroInputs] = []
    for micro_order, micro in enumerate(micro_inputs):
        prepared_micro, pending_prepared_micro = _prepare_current_sft_micro(
            micro,
            device=device,
            topology=topology,
            provider=provider,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
            pending_prepared_micro=pending_prepared_micro,
        )
        prepared_micros.append(prepared_micro)
        pending_prepared_micro = _prepare_next_sft_cp_micro(
            _next_micro_lookahead(micro_inputs, micro_order),
            device=device,
            topology=topology,
            provider=provider,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
        )
    schedule = MCoreScheduleAdapter(
        model_chunks=model_chunks,
        prepared_microbatches=prepared_micros,
        sample_indices=micro_sample_indices,
        model_inputs=[prepared.input_ids for prepared in prepared_micros],
        moe_routing_replay_controller=moe_routing_replay_controller,
        hybridep_token_counts=hybridep_token_counts,
        model_activator=model_support_handler.build_pipeline_microbatch_activator(
            model_chunks
        ),
    )

    def forward_step_func(data_iterator: Any, model: MegatronModule, *_args: Any):
        item = next(data_iterator)
        prepared = item.payload
        kwargs = dict(
            input_ids=prepared.input_ids,
            position_ids=prepared.position_ids,
            attention_mask=_placeholder_attention_mask(device),
            packed_seq_params=prepared.packed_seq_params,
            **model_support_handler.get_forward_kwargs(
                model, attention_bias=prepared.attention_state
            ),
        )
        with attach_trace_token_uids(model_chunks, prepared.local_token_uids):
            if chunk_post_process(model):
                token_output = forward_token_losses(
                    model,
                    labels=prepared.labels,
                    selection=prepared.lm_head_selection,
                    forward_kwargs=kwargs,
                )
                output = token_output.token_losses
            else:
                output = model(**kwargs, labels=None)
                token_output = None

        def collect(output_tensor: torch.Tensor, **_kwargs: Any):
            assert token_output is not None
            masked_loss = token_output.masked_sum(prepared.loss_mask)
            values = {
                "order": item.order,
                "raw_loss_sum": masked_loss.detach(),
                "logprobs": (
                    token_output.restore(-output_tensor).detach()
                    if return_token_logprobs
                    else None
                ),
            }
            if forward_only:
                return values
            return (
                masked_loss,
                _local_trainable_sft_token_count_tensor([prepared], device=device),
                values,
            )

        return output, collect

    forward_data_store = schedule.run(
        forward_step_func,
        forward_only=forward_only,
        collect_non_loss_data=forward_only,
    )
    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.finalize_step(expect_recompute=not forward_only)
    forward_data_store = cast(
        list[dict[str, Any]], _broadcast_from_pipeline_last(forward_data_store)
    )
    if len(forward_data_store) != len(prepared_micros):
        raise RuntimeError(
            "SFT pipeline did not return one result per local microbatch: "
            f"expected={len(prepared_micros)}, got={len(forward_data_store)}"
        )
    forward_data_store.sort(key=lambda data: int(data["order"]))
    raw_loss_sum = sum(
        (
            cast(torch.Tensor, data["raw_loss_sum"]).to(device)
            for data in forward_data_store
        ),
        torch.zeros([], device=device, dtype=torch.float32),
    )
    new_logprobs = [
        cast(torch.Tensor, data["logprobs"])
        for data in forward_data_store
        if data["logprobs"] is not None
    ]
    if return_token_logprobs and int(topology.cp) > 1:
        new_logprobs = _globalize_context_parallel_logprob_batch(
            local_logprobs=new_logprobs,
            attention_states=[prepared.attention_state for prepared in prepared_micros],
            sequence_lengths=[
                int(micro["attention_mask"].sum().item()) for micro in micro_inputs
            ],
            empty_template=raw_loss_sum,
        )
    return SFTForwardBackwardState(
        raw_loss_sum=raw_loss_sum,
        new_logprobs=tuple(new_logprobs),
        sample_indices=tuple(micro_sample_indices),
        prepared_micros=tuple(prepared_micros),
        device=device,
        schedule=schedule,
    )


def run_megatron_sft_forward_backward_step(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    step_index: int,
    sample_index: int | list[int | None],
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    hybridep_token_counts: list[int] | None = None,
    defer_grad_sync: bool = False,
    return_token_logprobs: bool = True,
) -> SFTForwardBackwardState:
    return _run_megatron_sft_schedule(
        model_chunks=model_chunks,
        provider=provider,
        model_support_handler=model_support_handler,
        inputs=inputs,
        step_index=step_index,
        sample_index=sample_index,
        moe_routing_replay_controller=moe_routing_replay_controller,
        hybridep_token_counts=hybridep_token_counts,
        defer_grad_sync=defer_grad_sync,
        forward_only=False,
        return_token_logprobs=return_token_logprobs,
    )


@torch.no_grad()
def run_megatron_sft_forward_step(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    step_index: int,
    sample_index: int | list[int | None],
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    hybridep_token_counts: list[int] | None = None,
    return_token_logprobs: bool = True,
) -> SFTForwardBackwardState:
    return _run_megatron_sft_schedule(
        model_chunks=model_chunks,
        provider=provider,
        model_support_handler=model_support_handler,
        inputs=inputs,
        step_index=step_index,
        sample_index=sample_index,
        moe_routing_replay_controller=moe_routing_replay_controller,
        hybridep_token_counts=hybridep_token_counts,
        forward_only=True,
        defer_grad_sync=False,
        return_token_logprobs=return_token_logprobs,
    )


def _finish_megatron_sft_step(
    state: SFTForwardBackwardState,
    optimizer_result: MegatronOptimizerStepResult,
) -> TrainStepResult:
    num_tokens = _local_trainable_sft_token_count_tensor(
        state.prepared_micros, device=state.device
    )
    reduced_loss = _reduce_loss_sum(
        state.raw_loss_sum,
        num_tokens,
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )
    return TrainStepResult(
        reduced_loss=reduced_loss,
        probs_corr=1.0,
        new_logprobs=None,
        update_successful=optimizer_result.update_successful,
        grad_norm=optimizer_result.grad_norm,
        num_zeros_in_grad=optimizer_result.num_zeros_in_grad,
        workload=state.schedule.training_workload(),
        pipeline_metrics=state.schedule.telemetry.metrics(),
    )


def run_megatron_sft_step(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    optimizer: Any,
    learning_rate: float,
    inputs: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    step_index: int,
    sample_index: int | list[int | None],
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    hybridep_token_counts: list[int] | None = None,
    before_optimizer_step: Callable[[], None] | None = None,
) -> TrainStepResult:
    state = run_megatron_sft_forward_backward_step(
        model_chunks=model_chunks,
        provider=provider,
        model_support_handler=model_support_handler,
        inputs=inputs,
        step_index=step_index,
        sample_index=sample_index,
        moe_routing_replay_controller=moe_routing_replay_controller,
        hybridep_token_counts=hybridep_token_counts,
    )
    optimizer_result = run_megatron_optimizer_step(
        optimizer=optimizer,
        learning_rate=learning_rate,
        model_support_handler=model_support_handler,
        model_chunks=model_chunks,
        before_step=before_optimizer_step,
    )
    return _finish_megatron_sft_step(state, optimizer_result)


def _run_training_schedule(
    schedule: MCoreScheduleAdapter[Any],
    forward_step_func: Callable[..., Any],
    timing: _InterForwardBackwardTiming | None,
    *,
    forward_only: bool = False,
) -> tuple[list[Any], Callable[[], dict[str, float]]]:
    started_s = time.monotonic()
    gap_s = (
        None
        if timing is None or timing.previous_schedule_end_s is None
        else started_s - timing.previous_schedule_end_s
    )
    phase_s = (
        None
        if timing is None
        or timing.previous_schedule_end_s is None
        or timing.previous_job_complete_s is None
        or timing.current_job_start_s is None
        or not (
            timing.previous_schedule_end_s
            <= timing.previous_job_complete_s
            <= timing.current_job_start_s
            <= started_s
        )
        else (
            timing.previous_job_complete_s - timing.previous_schedule_end_s,
            timing.current_job_start_s - timing.previous_job_complete_s,
            started_s - timing.current_job_start_s,
        )
    )
    previous_cuda_end = (
        timing.previous_schedule_cuda_end if timing is not None else None
    )
    outputs = schedule.run(
        forward_step_func,
        forward_only=forward_only,
        collect_non_loss_data=forward_only,
    )
    ended_s = time.monotonic()
    if timing is None:
        return outputs, lambda: {}
    cuda_span = schedule.telemetry.cuda_span()
    timing.previous_schedule_end_s = ended_s
    timing.previous_schedule_cuda_end = cuda_span[1] if cuda_span is not None else None
    if gap_s is None:
        return outputs, lambda: {}
    world_size = torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
    local_timing = torch.tensor(
        (gap_s, *(phase_s or (math.nan, math.nan, math.nan))), dtype=torch.float64
    )
    rank_timings = [torch.empty_like(local_timing) for _ in range(world_size)]
    work = None
    if world_size == 1:
        rank_timings[0].copy_(local_timing)
    else:
        if timing.metrics_group is None:
            raise RuntimeError("Multi-rank schedule timing requires a Gloo group")
        work = torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
            rank_timings,
            local_timing,
            group=timing.metrics_group,
            async_op=True,
        )

    def metrics() -> dict[str, float]:
        if work is not None:
            work.wait()
        values = {
            f"{_INTER_FORWARD_BACKWARD_GAP_PREFIX}{rank}_s": float(parts[0].item())
            for rank, parts in enumerate(rank_timings)
        }
        phase_names = ("previous_job_tail", "worker_idle", "current_job_prepare")
        values.update(
            {
                f"{_INTER_FORWARD_BACKWARD_PHASE_PREFIX}{name}_rank_{rank}_s": float(
                    parts[index].item()
                )
                for rank, parts in enumerate(rank_timings)
                for index, name in enumerate(phase_names, start=1)
                if not torch.isnan(parts[index])
            }
        )
        if previous_cuda_end is None or cuda_span is None:
            return values
        local_gpu_gap = torch.tensor(
            previous_cuda_end.elapsed_time(cuda_span[0]) / 1e3,
            dtype=torch.float64,
        )
        gpu_gaps = [torch.empty_like(local_gpu_gap) for _ in range(world_size)]
        if world_size == 1:
            gpu_gaps[0].copy_(local_gpu_gap)
        else:
            torch.distributed.all_gather(  # ty: ignore[possibly-missing-attribute]
                gpu_gaps,
                local_gpu_gap,
                group=timing.metrics_group,
            )
        values.update(
            {
                f"{_INTER_FORWARD_BACKWARD_GPU_GAP_PREFIX}{rank}_s": float(gap.item())
                for rank, gap in enumerate(gpu_gaps)
            }
        )
        return values

    return outputs, metrics


def run_megatron_rl_forward_backward_step(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: PackedTensors | list[PackedTensors],
    config: types.TrainConfig,
    experimental_config: dev.TrainConfig,
    step_index: int,
    sample_index: int | list[int | None],
    ref_logprobs: torch.Tensor | list[torch.Tensor] | None = None,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    cp_lookahead_state: CpBatchLookaheadState | None = None,
    next_step_first_micro: PackedTensors | None = None,
    next_step_first_ref_logprobs: torch.Tensor | None = None,
    hybridep_token_counts: list[int] | None = None,
    inter_forward_backward_timing: _InterForwardBackwardTiming | None = None,
    defer_grad_sync: bool = False,
    forward_only: bool = False,
    loss: LossConfig | None = None,
) -> RLForwardBackwardState:
    schedule_prepare_started = time.perf_counter()
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("RL forward/backward requires at least one packed sequence")

    micro_sample_indices: list[int | None]
    if isinstance(sample_index, list):
        if len(sample_index) != len(micro_inputs):
            raise ValueError(
                "sample_index list length must match number of micro inputs: "
                f"{len(sample_index)} != {len(micro_inputs)}"
            )
        micro_sample_indices = [
            int(index) if index is not None else None for index in sample_index
        ]
    else:
        assert len(micro_inputs) == 1
        micro_sample_indices = [sample_index]

    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=micro_sample_indices,
        )

    device = next(model_chunks[0].parameters()).device
    topology = _infer_parallel_topology(model_chunks)
    trace_token_uids = context_parallel_trace_token_uids_enabled(
        topology,
        moe_routing_replay_controller,
    )
    pending_prepared_micro = (
        cp_lookahead_state.pending_prepared_micro
        if cp_lookahead_state is not None and int(topology.cp) > 1
        else None
    )
    if cp_lookahead_state is not None and int(topology.cp) <= 1:
        cp_lookahead_state.pending_prepared_micro = None

    if not forward_only:
        _zero_grad_buffers(model_chunks)
        _install_schedule_finalize(model_chunks, defer_grad_sync=defer_grad_sync)

    micro_count = len(micro_inputs)
    prepared_micros: list[PreparedRLMicroInputs] = []
    for micro_order in range(micro_count):
        micro_ref_logprobs = _select_ref_logprobs(ref_logprobs, micro_order)
        if micro_ref_logprobs is not None and int(topology.cp) <= 1:
            micro_ref_logprobs = micro_ref_logprobs.to(device)
        prepared_micro, pending_prepared_micro = _prepare_current_rl_micro(
            micro_inputs[micro_order],
            device=device,
            topology=topology,
            provider=provider,
            model_support_handler=model_support_handler,
            ref_logprobs=micro_ref_logprobs,
            trace_token_uids=trace_token_uids,
            pending_prepared_micro=pending_prepared_micro,
        )
        prepared_micros.append(prepared_micro)
        pending_prepared_micro = _prepare_next_rl_cp_micro(
            _next_micro_lookahead(
                micro_inputs,
                micro_order,
                next_step_first_micro,
            ),
            device=device,
            topology=topology,
            provider=provider,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
            ref_logprobs=_select_next_ref_logprobs(
                ref_logprobs,
                micro_order=micro_order,
                micro_count=micro_count,
                next_step_first_ref_logprobs=next_step_first_ref_logprobs,
            ),
        )
    if cp_lookahead_state is not None:
        cp_lookahead_state.pending_prepared_micro = pending_prepared_micro

    schedule = MCoreScheduleAdapter(
        model_chunks=model_chunks,
        prepared_microbatches=prepared_micros,
        sample_indices=micro_sample_indices,
        model_inputs=[prepared.model_tokens for prepared in prepared_micros],
        moe_routing_replay_controller=moe_routing_replay_controller,
        hybridep_token_counts=hybridep_token_counts,
        model_activator=model_support_handler.build_pipeline_microbatch_activator(
            model_chunks
        ),
    )

    def forward_step_func(data_iterator: Any, model: MegatronModule, *_args: Any):
        item = next(data_iterator)
        prepared = item.payload
        token_output = (
            _forward_prepared_rl_micro(
                model_chunks=model_chunks,
                model_chunk=model,
                model_support_handler=model_support_handler,
                prepared_micro=prepared,
                device=device,
            )
            if loss is None
            else _forward_prepared_score_micro(
                model_chunks=model_chunks,
                model_chunk=model,
                model_support_handler=model_support_handler,
                prepared=prepared,
                device=device,
            )
        )

        def reduce_loss(output_tensor: torch.Tensor, **_kwargs: Any):
            if loss is not None:
                target_tokens = prepared.target_tokens
                loss_weights = prepared.loss_weights
                behavior_logprobs = prepared.behavior_logprobs
                token_advantages = prepared.token_advantages
                if any(
                    value is None
                    for value in (
                        target_tokens,
                        loss_weights,
                        behavior_logprobs,
                        token_advantages,
                    )
                ):
                    raise RuntimeError(
                        "tokenized F/B micro is missing named-loss tensors"
                    )
                assert target_tokens is not None
                assert loss_weights is not None
                assert behavior_logprobs is not None
                assert token_advantages is not None
                selected_targets = prepared.lm_head_selection.select_rows(target_tokens)
                selected_weights = prepared.lm_head_selection.select_rows(loss_weights)
                selected_behavior = prepared.lm_head_selection.select_rows(
                    behavior_logprobs
                )
                selected_advantages = prepared.lm_head_selection.select_rows(
                    token_advantages
                )
                selected_logprobs = vocab_parallel_selected_logprobs(
                    output_tensor, selected_targets
                )
                loss_output = tokenized_loss(
                    loss,
                    target_logprobs=selected_logprobs,
                    weights=selected_weights,
                    sampling_logprobs=selected_behavior,
                    advantages=selected_advantages,
                )
                micro_loss = loss_output.loss_sum
                if not forward_only and not micro_loss.requires_grad:
                    raise RuntimeError("tokenized micro loss is detached")
                ratio = loss_output.probability_ratio
                active = (
                    selected_weights != 0
                    if loss.name == "cross_entropy"
                    else selected_advantages != 0
                )
                diagnostics = None
                if ratio is not None and loss.name in {"ppo", "cispo"}:
                    loss_name = cast(Literal["ppo", "cispo"], loss.name)
                    low, high = tokenized_clip_bounds(loss_name, loss.values)
                    diagnostics = LossOffPolicyDiagnostics.from_tensors(
                        prob_ratio=ratio,
                        advantages=selected_advantages,
                        assistant_mask=active,
                        weights=torch.ones_like(selected_advantages),
                        ppo=loss.name == "ppo",
                        epsilon=1.0 - low,
                        epsilon_high=high - 1.0,
                    )
                values = {
                    "order": item.order,
                    "raw_loss_sum": micro_loss.detach(),
                    "probs_corr": (
                        compute_probs_corr(
                            selected_behavior.masked_fill(~active, float("nan")),
                            selected_logprobs.masked_fill(~active, float("nan")),
                        ).detach()
                        if ratio is not None
                        else micro_loss.detach().new_zeros(())
                    ),
                    "kl_policy_ref": None,
                    "offpolicy_diagnostics": diagnostics,
                    "new_logprobs": prepared.lm_head_selection.restore_rows(
                        selected_logprobs.detach()
                    ),
                }
                if forward_only:
                    return values
                return (
                    micro_loss,
                    _local_trainable_token_count_tensor(
                        [prepared.loss_inputs], device=device
                    ),
                    values,
                )
            assert isinstance(token_output, TokenLossOutput)
            new_logprobs = -output_tensor
            compact_loss_inputs = token_output.compact_loss_inputs(prepared.loss_inputs)
            loss_info = loss_fn(
                compact_loss_inputs,
                new_logprobs=new_logprobs,
                ref_logprobs=token_output.select_optional(prepared.ref_logprobs),
                entropies=None,
                experimental_config=experimental_config,
                reduction="sum",
            )
            micro_loss = loss_info.policy_loss + _zero_logprob_graph_contribution(
                new_logprobs, compact_loss_inputs
            )
            if not forward_only and not micro_loss.requires_grad:
                raise RuntimeError(
                    "RL micro_loss is detached before pipeline backward: "
                    f"micro={item.order}, sample={item.sample_index}"
                )
            num_tokens = _local_trainable_token_count_tensor(
                [prepared.loss_inputs], device=device
            )
            values = {
                "order": item.order,
                "raw_loss_sum": micro_loss.detach(),
                "probs_corr": loss_info.probs_corr.detach(),
                "kl_policy_ref": (
                    None
                    if loss_info.kl_policy_ref is None
                    else float(loss_info.kl_policy_ref.item())
                ),
                "offpolicy_diagnostics": loss_info.offpolicy_diagnostics,
                "new_logprobs": token_output.restore(new_logprobs.detach()).to("cpu"),
            }
            if forward_only:
                return values
            return (micro_loss, num_tokens, values)

        return (
            token_output.token_losses
            if isinstance(token_output, TokenLossOutput)
            else token_output,
            reduce_loss,
        )

    schedule_prepare_s = time.perf_counter() - schedule_prepare_started
    forward_data_store, collect_inter_schedule_metrics = _run_training_schedule(
        schedule,
        forward_step_func,
        inter_forward_backward_timing,
        forward_only=forward_only,
    )
    replay_finalize_started = time.perf_counter()
    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.finalize_step(expect_recompute=not forward_only)
    replay_finalize_s = time.perf_counter() - replay_finalize_started
    result_collect_started = time.perf_counter()
    pipeline_results = cast(
        list[dict[str, Any]],
        _broadcast_from_pipeline_last(forward_data_store),
    )
    if len(pipeline_results) != micro_count:
        raise RuntimeError(
            "MCore schedule did not return one final-stage result per microbatch: "
            f"expected={micro_count}, got={len(pipeline_results)}"
        )
    pipeline_results.sort(key=lambda data: int(data["order"]))
    raw_loss_sum = sum(
        (
            cast(torch.Tensor, data["raw_loss_sum"]).to(device)
            for data in pipeline_results
        ),
        torch.zeros([], device=device, dtype=torch.float32),
    )
    probs_corr_total = sum(
        (
            cast(torch.Tensor, data["probs_corr"]).to(device)
            for data in pipeline_results
        ),
        torch.zeros([], device=device, dtype=torch.float32),
    )
    kl_values = [
        float(value)
        for data in pipeline_results
        if (value := data["kl_policy_ref"]) is not None
    ]
    loss_diagnostics = LossOffPolicyDiagnosticsAccumulator()
    for data in pipeline_results:
        loss_diagnostics.add(data["offpolicy_diagnostics"])

    token_count = _local_trainable_token_count_tensor(
        [prepared.loss_inputs for prepared in prepared_micros],
        device=device,
    )
    result_collect_s = time.perf_counter() - result_collect_started
    inter_metrics_started = time.perf_counter()
    inter_metrics = collect_inter_schedule_metrics()
    inter_metrics_s = time.perf_counter() - inter_metrics_started
    inter_metrics.update(
        {
            "time/schedule_prepare_s": schedule_prepare_s,
            "time/post_schedule_replay_finalize_s": replay_finalize_s,
            "time/post_schedule_result_collect_s": result_collect_s,
            "time/post_schedule_inter_metrics_s": inter_metrics_s,
        }
    )
    new_logprobs = [
        cast(torch.Tensor, data["new_logprobs"]) for data in pipeline_results
    ]
    returned_sample_indices = tuple(micro_sample_indices)
    if loss is not None and int(topology.cp) > 1:
        new_logprobs = _globalize_context_parallel_logprob_batch(
            local_logprobs=new_logprobs,
            attention_states=[prepared.attention_state for prepared in prepared_micros],
            sequence_lengths=[
                int(micro_inputs[0]["tokens"].shape[1]) for _ in prepared_micros
            ],
        )
    if loss is not None:
        new_logprobs, returned_sample_indices = _globalize_data_parallel_logprob_batch(
            local_logprobs=new_logprobs,
            sample_indices=micro_sample_indices,
            step_index=step_index,
            global_grad_accumulation_sequences=(
                resolve_global_grad_accumulation_sequences(
                    config.grad_accumulation_sequences
                )
            ),
        )
    return RLForwardBackwardState(
        raw_loss_sum=raw_loss_sum,
        probs_corr_total=probs_corr_total,
        kl_values=tuple(kl_values),
        new_logprobs=tuple(new_logprobs),
        sample_indices=returned_sample_indices,
        token_count=token_count,
        micro_count=micro_count,
        schedule=schedule,
        loss_diagnostics=loss_diagnostics,
        inter_schedule_metrics=inter_metrics,
    )


def _finish_megatron_rl_forward_backward_step(
    state: RLForwardBackwardState,
) -> MegatronForwardBackwardStepResult:
    loss_reduce_started = time.perf_counter()
    reduced_loss = _reduce_loss_sum(
        state.raw_loss_sum,
        state.token_count,
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )
    loss_reduce_s = time.perf_counter() - loss_reduce_started
    loss_metrics_started = time.perf_counter()
    loss_metrics = state.loss_diagnostics.to_metrics(
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )
    loss_metrics_s = time.perf_counter() - loss_metrics_started
    return MegatronForwardBackwardStepResult(
        reduced_loss=reduced_loss,
        probs_corr=float((state.probs_corr_total / state.micro_count).item()),
        kl_policy_ref=(
            sum(state.kl_values) / len(state.kl_values) if state.kl_values else None
        ),
        new_logprobs=list(state.new_logprobs),
        workload=state.schedule.training_workload(),
        loss_metrics=loss_metrics,
        pipeline_metrics={
            **state.schedule.telemetry.metrics(),
            **state.inter_schedule_metrics,
            "time/post_schedule_loss_reduce_s": loss_reduce_s,
            "time/post_schedule_loss_metrics_s": loss_metrics_s,
        },
    )


def run_training_step(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    optimizer: Any,
    learning_rate: float,
    inputs: PackedTensors | list[PackedTensors],
    config: types.TrainConfig,
    experimental_config: dev.TrainConfig,
    step_index: int,
    sample_index: int | list[int | None],
    ref_logprobs: torch.Tensor | list[torch.Tensor] | None = None,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    cp_lookahead_state: CpBatchLookaheadState | None = None,
    next_step_first_micro: PackedTensors | None = None,
    next_step_first_ref_logprobs: torch.Tensor | None = None,
    hybridep_token_counts: list[int] | None = None,
    before_optimizer_step: Callable[[], None] | None = None,
    inter_forward_backward_timing: _InterForwardBackwardTiming | None = None,
) -> TrainStepResult:
    state = run_megatron_rl_forward_backward_step(
        model_chunks=model_chunks,
        provider=provider,
        model_support_handler=model_support_handler,
        inputs=inputs,
        config=config,
        experimental_config=experimental_config,
        step_index=step_index,
        sample_index=sample_index,
        ref_logprobs=ref_logprobs,
        moe_routing_replay_controller=moe_routing_replay_controller,
        cp_lookahead_state=cp_lookahead_state,
        next_step_first_micro=next_step_first_micro,
        next_step_first_ref_logprobs=next_step_first_ref_logprobs,
        hybridep_token_counts=hybridep_token_counts,
        inter_forward_backward_timing=inter_forward_backward_timing,
    )
    optimizer_started = time.perf_counter()
    optimizer_result = run_megatron_optimizer_step(
        optimizer=optimizer,
        learning_rate=learning_rate,
        model_support_handler=model_support_handler,
        model_chunks=model_chunks,
        before_step=before_optimizer_step,
    )
    optimizer_s = time.perf_counter() - optimizer_started
    result_build_started = time.perf_counter()
    forward_backward = _finish_megatron_rl_forward_backward_step(state)
    result = TrainStepResult(
        **forward_backward.model_dump(),
        update_successful=optimizer_result.update_successful,
        grad_norm=optimizer_result.grad_norm,
        num_zeros_in_grad=optimizer_result.num_zeros_in_grad,
    )
    result.pipeline_metrics["time/post_schedule_optimizer_s"] = optimizer_s
    result.pipeline_metrics["time/post_schedule_result_build_s"] = (
        time.perf_counter() - result_build_started
    )
    return result
