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
- execute_megatron_sft_forward_backward_job
"""

import math
import os
import random
from threading import Event
import time
from typing import Any, Callable, Literal, cast

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
    Loss,
    LossInputs,
    LossOffPolicyDiagnosticsAccumulator,
    loss_fn,
    shift_tensor,
)
from art.megatron.context_parallel.types import (
    DispatchedPackedTensors,
    ParallelTopology,
    PreparedMegatronBatch,
    TrainingStepWorkload,
)
from art.megatron.lora import apply_lora_adapters
from art.megatron.megatron_patches import install_fast_frozen_output_backward
from art.megatron.model_support.lora_disk import (
    load_adapter_config,
    load_lora_tensors_for_megatron,
)
from art.megatron.optimizer_state import (
    ALLOW_UNPAIRED_MEGATRON_RESUME_ENV,
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
)
from art.megatron.runtime.data_plane import SFTBatchData
from art.megatron.runtime.specs import (
    ForwardBackwardJobSpec,
    ForwardJobSpec,
    LoadStateJobSpec,
    RlForwardBackwardConfig,
    SftForwardBackwardJobSpec,
    SftForwardJobSpec,
    TrainJobSpec,
)
from art.megatron.runtime.weight_transfer import MergedWeightTransferInitInfo
from art.megatron.selective_lm_head import (
    TokenLossOutput,
    forward_token_losses,
)
from art.megatron.tensor_snapshot import SnapshotReadBarrier
from art.megatron.training.compile import (
    configure_training_compile,
)
from art.megatron.training.finalize_grads import (
    finalize_model_grads_extended,
    flush_param_grads_to_main_grads,
)
from art.megatron.training.microbatches import (
    CpBatchLookaheadState,
    PreparedRLMicroInputs,
    PreparedSFTMicroInputs,
    _causal_attention_state,
    _clone_packed_tensors,
    _clone_sft_tensors,
    _count_sft_trainable_tokens,
    _count_trainable_tokens,
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
    chunk_pre_process,
    validate_pipeline_topology,
)
from art.megatron.training.trace import (
    attach_trace_token_uids,
    context_parallel_trace_token_uids_enabled,
)
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.preprocessing.pack import PackedTensors

DEFAULT_MODEL_IDENTIFIER = "Qwen/Qwen3-30B-A3B-Instruct-2507"
_optimizer_stats_printed = False

__all__ = [
    "DEFAULT_MODEL_IDENTIFIER",
    "TrainingRuntime",
    "execute_megatron_rl_forward_backward_job",
    "execute_megatron_sft_forward_backward_job",
    "execute_megatron_sft_forward_job",
    "build_training_runtime",
    "execute_megatron_rl_job",
]


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_bundle: ProviderBundle
    provider: Any
    model: ModelChunks
    optimizer: Any | None
    optimizer_config: OptimizerConfig
    optimizer_persistent: bool = True
    resident_training_session_id: str | None = None
    resident_policy_step: int | None = None
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
    merged_weight_transfer_group: Any | None = None
    merged_weight_transfer_init_info: MergedWeightTransferInitInfo | None = None

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


class MegatronForwardBackwardJobResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result: MegatronForwardBackwardStepResult
    num_gradient_steps: int = Field(ge=1)
    forward_backward_s: float = Field(ge=0)
    replay_finalize_s: float = Field(ge=0)
    token_count: torch.Tensor

    def metrics(self) -> dict[str, float]:
        metrics = _rl_forward_backward_metrics(
            self.result,
            num_gradient_steps=self.num_gradient_steps,
            forward_backward_s=self.forward_backward_s,
        )
        metrics["time/replay_finalize_s"] = self.replay_finalize_s
        return metrics


class RLForwardBackwardState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_loss_sum: torch.Tensor
    probs_corr_total: torch.Tensor
    kl_values: tuple[float, ...]
    new_logprobs: tuple[torch.Tensor, ...]
    token_count: torch.Tensor
    micro_count: int = Field(ge=1)
    schedule: MCoreScheduleAdapter[PreparedRLMicroInputs]
    loss_diagnostics: LossOffPolicyDiagnosticsAccumulator


class SFTForwardBackwardState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_loss_sum: torch.Tensor
    new_logprobs: tuple[torch.Tensor, ...]
    sample_indices: tuple[int | None, ...]
    prepared_micros: tuple[PreparedSFTMicroInputs, ...]
    device: torch.device
    schedule: MCoreScheduleAdapter[PreparedSFTMicroInputs]


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
    if _moe_routing_replay_requested(
        replay_bundle_path=moe_routing_replay_path,
        replay_bundle=moe_routing_replay_bundle,
    ):
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
    transformer_layers_compiled = configure_training_compile(
        model=model,
        provider=provider,
        provider_bundle=provider_bundle,
    )

    optimizer_config = optimizer_config or _default_optimizer_config()
    optimizer = _build_optimizer(model, optimizer_config) if build_optimizer else None

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
    )
    configure_moe_routing_replay(
        runtime,
        replay_bundle_path=moe_routing_replay_path,
        replay_bundle=moe_routing_replay_bundle,
        strict=moe_routing_replay_strict,
    )
    return runtime


def _execute_megatron_rl_forward_backward_steps(
    runtime: TrainingRuntime,
    job: TrainJobSpec | ForwardBackwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    before_step: Callable[[int], None],
    after_step: Callable[[int, int, RLForwardBackwardState, float, float], None],
    cancelled: Event | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
    state_is_resident: bool = False,
) -> tuple[dict[str, torch.dtype], float, int]:
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
        for step_index in range(num_steps):
            if cancelled is not None and cancelled.is_set():
                from art.megatron.runtime.trainer_run import TrainingCancelledError

                raise TrainingCancelledError("train job was cancelled")
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
            train_step_started = time.perf_counter()
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
            )
            after_step(
                step_index,
                num_steps,
                state,
                time.perf_counter() - train_step_started,
                replay_finalize_s,
            )
        return adapter_dtypes, replay_finalize_s, num_steps
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
            del cp_lookahead_state


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
    job_succeeded = False
    final_metrics: dict[str, float] = {}

    def finish_step(
        step_index: int,
        num_steps: int,
        state: RLForwardBackwardState,
        forward_backward_s: float,
        replay_finalize_s: float,
    ) -> None:
        nonlocal final_metrics
        finish_started = time.perf_counter()
        assert runtime.optimizer is not None
        optimizer_result = run_megatron_optimizer_step(
            optimizer=runtime.optimizer,
            learning_rate=job.config.learning_rate,
            model_support_handler=runtime.model_support_handler,
            model_chunks=runtime.model,
            before_step=runtime.optimizer_snapshot_barrier.wait_before_mutation,
        )
        step_result = _finish_megatron_rl_step(state, optimizer_result)
        train_step_s = forward_backward_s + time.perf_counter() - finish_started
        print0(
            runtime.rank,
            "Correlation between old and new probabilities:",
            step_result.probs_corr,
        )
        _validate_train_step_result_finite(runtime, step_result)
        final_metrics = _rl_step_metrics(
            step_result,
            num_gradient_steps=num_steps,
            train_step_s=train_step_s,
        )
        final_metrics["time/replay_finalize_s"] = replay_finalize_s
        if runtime.rank == 0:
            progress_sink(step_index, num_steps, final_metrics)

    try:
        adapter_dtypes, replay_finalize_s, _ = (
            _execute_megatron_rl_forward_backward_steps(
                runtime,
                job,
                packed_tensors,
                before_step=lambda _step_index: None,
                after_step=finish_step,
                cancelled=cancelled,
                replay_bundle=replay_bundle,
            )
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
        adapter_ready_sink()
        runtime.resident_training_session_id = job.training_session_id
        runtime.resident_policy_step = job.step
        runtime.optimizer_state_loaded = True
        job_succeeded = True
        return final_metrics
    finally:
        if not job_succeeded:
            runtime.resident_training_session_id = None
            runtime.resident_policy_step = None
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
    replay_bundle: MoeRoutingReplayBundle | None = None,
) -> MegatronForwardBackwardJobResult:
    """Execute one independently admitted F/B and retain its gradients on GPU."""
    from art.megatron.training.gradient_accumulator import GradientAccumulator

    gradient_accumulator.before_forward_backward()
    internal = GradientAccumulator(model_chunks=runtime.model)
    states: list[RLForwardBackwardState] = []
    durations: list[float] = []
    global_token_counts: list[torch.Tensor] = []

    def before_step(step_index: int) -> None:
        if step_index:
            internal.before_forward_backward()

    def record_step(
        step_index: int,
        _num_steps: int,
        state: RLForwardBackwardState,
        duration_s: float,
        _replay_finalize_s: float,
    ) -> None:
        global_token_count = state.token_count.detach().clone()
        torch.distributed.all_reduce(
            global_token_count,
            group=ps.get_data_parallel_group(with_context_parallel=True),
        )
        internal.record(f"{job.operation_id}:{step_index}", global_token_count)
        states.append(state)
        durations.append(duration_s)
        global_token_counts.append(global_token_count)

    _, replay_finalize_s, _ = _execute_megatron_rl_forward_backward_steps(
        runtime,
        job,
        packed_tensors,
        before_step=before_step,
        after_step=record_step,
        cancelled=cancelled,
        replay_bundle=replay_bundle,
    )
    finish_started = time.perf_counter()
    internal.seal(internal.contribution_ids)
    internal.prepare_optimizer()
    internal.consume()
    token_count = sum(
        global_token_counts,
        torch.zeros_like(global_token_counts[0]),
    )
    gradient_accumulator.record(job.operation_id, token_count)
    result = _finish_megatron_rl_forward_backward_job(tuple(states))
    runtime.resident_training_session_id = job.training_session_id
    runtime.resident_policy_step = job.expected_learner_version
    runtime.optimizer_state_loaded = True
    return MegatronForwardBackwardJobResult(
        result=result,
        num_gradient_steps=len(states),
        forward_backward_s=sum(durations) + time.perf_counter() - finish_started,
        replay_finalize_s=replay_finalize_s,
        token_count=token_count,
    )


def execute_megatron_dynamic_lora_forward_backward_job(
    runtime: TrainingRuntime,
    job: ForwardBackwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    slot_trainer: Any,
    gradient_accumulator: Any,
    cancelled: Event | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
) -> MegatronForwardBackwardJobResult:
    """Run the MCore schedule against one exact-shape dynamic LoRA slot."""
    from art.megatron.lora import LoRASlotRef, use_lora_slot
    from art.megatron.training.gradient_accumulator import (
        ParameterGradientAccumulator,
    )

    ref_path = _experimental_train_config(job).get("kl_ref_adapter_path")
    if (
        job.config.kl_penalty_coef > 0.0
        and ref_path is not None
        and os.path.abspath(ref_path) != os.path.abspath(job.source_adapter_path)
    ):
        raise NotImplementedError(
            "dynamic MCore slots require the KL reference to be the current learner"
        )
    slot_name = job.run_id
    slot_ref = LoRASlotRef("checkpoint", slot_name)
    params = slot_trainer.checkpoint_slot_parameters(slot_name)
    internal = ParameterGradientAccumulator(parameters=params)
    states: list[RLForwardBackwardState] = []
    durations: list[float] = []
    global_token_counts: list[torch.Tensor] = []
    gradient_accumulator.before_forward_backward()

    def before_step(_step_index: int) -> None:
        internal.before_forward_backward()

    def record_step(
        step_index: int,
        _num_steps: int,
        state: RLForwardBackwardState,
        duration_s: float,
        _replay_finalize_s: float,
    ) -> None:
        global_token_count = state.token_count.detach().clone()
        torch.distributed.all_reduce(
            global_token_count,
            group=ps.get_data_parallel_group(with_context_parallel=True),
        )
        gradients = slot_trainer.reduce_checkpoint_slot_grads(
            slot_name,
            scale_grads=1.0 / float(global_token_count.item()),
        )
        internal.record(
            f"{job.operation_id}:{step_index}", global_token_count, gradients
        )
        states.append(state)
        durations.append(duration_s)
        global_token_counts.append(global_token_count)

    with use_lora_slot(slot_ref):
        _, replay_finalize_s, _ = _execute_megatron_rl_forward_backward_steps(
            runtime,
            job,
            packed_tensors,
            before_step=before_step,
            after_step=record_step,
            cancelled=cancelled,
            replay_bundle=replay_bundle,
            state_is_resident=True,
        )
    finish_started = time.perf_counter()
    internal.seal(internal.contribution_ids)
    gradients = internal.prepare_optimizer()
    token_count = sum(
        global_token_counts,
        torch.zeros_like(global_token_counts[0]),
    )
    gradient_accumulator.record(job.operation_id, token_count, gradients)
    internal.consume()
    slot_trainer.clear_checkpoint_slot_grads(slot_name)
    return MegatronForwardBackwardJobResult(
        result=_finish_megatron_rl_forward_backward_job(tuple(states)),
        num_gradient_steps=len(states),
        forward_backward_s=sum(durations) + time.perf_counter() - finish_started,
        replay_finalize_s=replay_finalize_s,
        token_count=token_count,
    )


def _sft_command_inputs(
    runtime: TrainingRuntime,
    job: SftForwardBackwardJobSpec | SftForwardJobSpec,
    batch: SFTBatchData,
) -> tuple[list[dict[str, torch.Tensor]], list[int | None], list[int] | None]:
    if batch.fingerprint != job.batch_fingerprint:
        raise ValueError("SFT command fingerprint differs from its tensor payload")
    if batch.num_trainable_tokens != job.trainable_token_count:
        raise ValueError("SFT command token count differs from its tensor payload")
    if batch.num_trajectories > job.global_grad_accumulation_sequences:
        raise ValueError("SFT command requires more than one MCore schedule step")
    configure_moe_routing_replay(runtime)
    trajectories = list(batch.trajectory_tensors)
    template = _clone_sft_tensors(trajectories[0])
    indices = build_micro_sample_indices(
        step_index=0,
        num_sequences=batch.num_trajectories,
        global_grad_accumulation_sequences=job.global_grad_accumulation_sequences,
    )
    topology = _infer_parallel_topology(runtime.model)
    hybridep_token_counts = (
        build_sft_hybridep_token_counts(
            trajectory_tensors=trajectories,
            step_index=0,
            global_grad_accumulation_sequences=(job.global_grad_accumulation_sequences),
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
            _zero_contribution_sft_inputs(template),
        ),
        indices,
        hybridep_token_counts,
    )


def _global_sft_loss_and_tokens(
    state: SFTForwardBackwardState,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = _local_trainable_sft_token_count_tensor(
        state.prepared_micros, device=state.device
    )
    raw_loss_sum = state.raw_loss_sum.detach().clone()
    if torch.distributed.is_initialized():
        group = ps.get_data_parallel_group(with_context_parallel=True)
        torch.distributed.all_reduce(token_count, group=group)
        torch.distributed.all_reduce(raw_loss_sum, group=group)
    return raw_loss_sum / token_count, token_count


def _global_sft_logprobs(
    state: SFTForwardBackwardState,
    batch: SFTBatchData,
) -> tuple[tuple[float, ...], ...]:
    lengths = [
        int(tensors["attention_mask"].sum().item())
        for tensors in batch.trajectory_tensors
    ]
    values = state.new_logprobs[0].new_zeros((len(lengths), max(lengths)))
    present = torch.zeros(len(lengths), device=state.device, dtype=torch.int32)
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
    if torch.distributed.is_initialized() and ps.get_data_parallel_world_size() > 1:
        group = ps.get_data_parallel_group()
        torch.distributed.all_reduce(values, group=group)
        torch.distributed.all_reduce(present, group=group)
    if not torch.all(present == 1):
        raise RuntimeError("SFT forward did not materialize every trajectory")
    host = values.detach().cpu()
    return tuple(
        tuple(float(value) for value in host[index, :length].tolist())
        for index, length in enumerate(lengths)
    )


def _sft_command_result(
    *,
    job: SftForwardBackwardJobSpec | SftForwardJobSpec,
    batch: SFTBatchData,
    state: SFTForwardBackwardState,
    reduced_loss: torch.Tensor,
    token_count: torch.Tensor,
    elapsed_s: float,
    backward: bool,
) -> dict[str, Any]:
    workload = state.schedule.training_workload()
    metrics = {
        "loss/train": float(reduced_loss.item()),
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
        "time/forward_backward_s" if backward else "time/forward_s": elapsed_s,
        **state.schedule.telemetry.metrics(),
    }
    if backward:
        metrics[TRAIN_GRADIENT_STEPS_KEY] = 1.0
    return {
        "operation_id": job.operation_id,
        "metrics": metrics,
        "token_count": int(token_count.item()),
        "token_logprobs": _global_sft_logprobs(state, batch),
    }


def execute_megatron_sft_forward_backward_job(
    runtime: TrainingRuntime,
    job: SftForwardBackwardJobSpec,
    batch: SFTBatchData,
    *,
    gradient_accumulator: Any,
    cancelled: Event | None = None,
) -> dict[str, Any]:
    """Run one supervised F/B against the resident single-run learner."""
    if cancelled is not None and cancelled.is_set():
        from art.megatron.runtime.trainer_run import TrainingCancelledError

        raise TrainingCancelledError("SFT F/B job was cancelled")
    started = time.perf_counter()
    gradient_accumulator.before_forward_backward()
    _prepare_rl_training_state(runtime, job)
    inputs, indices, hybridep_token_counts = _sft_command_inputs(runtime, job, batch)
    state = run_megatron_sft_forward_backward_step(
        model_chunks=runtime.model,
        provider=runtime.provider,
        model_support_handler=runtime.model_support_handler,
        inputs=inputs,
        step_index=0,
        sample_index=indices,
        hybridep_token_counts=hybridep_token_counts,
    )
    reduced_loss, token_count = _global_sft_loss_and_tokens(state)
    gradient_accumulator.record(job.operation_id, token_count)
    runtime.resident_training_session_id = job.training_session_id
    runtime.resident_policy_step = job.expected_learner_version
    runtime.optimizer_state_loaded = True
    return _sft_command_result(
        job=job,
        batch=batch,
        state=state,
        reduced_loss=reduced_loss,
        token_count=token_count,
        elapsed_s=time.perf_counter() - started,
        backward=True,
    )


def execute_megatron_sft_forward_job(
    runtime: TrainingRuntime,
    job: SftForwardJobSpec,
    batch: SFTBatchData,
    *,
    cancelled: Event | None = None,
) -> dict[str, Any]:
    """Run one supervised forward without mutating accumulated gradients."""
    if cancelled is not None and cancelled.is_set():
        from art.megatron.runtime.trainer_run import TrainingCancelledError

        raise TrainingCancelledError("SFT forward job was cancelled")
    started = time.perf_counter()
    _prepare_rl_training_state(runtime, job)
    inputs, indices, hybridep_token_counts = _sft_command_inputs(runtime, job, batch)
    state = run_megatron_sft_forward_step(
        model_chunks=runtime.model,
        provider=runtime.provider,
        model_support_handler=runtime.model_support_handler,
        inputs=inputs,
        sample_index=indices,
        hybridep_token_counts=hybridep_token_counts,
    )
    reduced_loss, token_count = _global_sft_loss_and_tokens(state)
    return _sft_command_result(
        job=job,
        batch=batch,
        state=state,
        reduced_loss=reduced_loss,
        token_count=token_count,
        elapsed_s=time.perf_counter() - started,
        backward=False,
    )


def execute_megatron_dynamic_lora_sft_forward_backward_job(
    runtime: TrainingRuntime,
    job: SftForwardBackwardJobSpec,
    batch: SFTBatchData,
    *,
    slot_trainer: Any,
    gradient_accumulator: Any,
    cancelled: Event | None = None,
) -> dict[str, Any]:
    from art.megatron.lora import LoRASlotRef, use_lora_slot

    if cancelled is not None and cancelled.is_set():
        from art.megatron.runtime.trainer_run import TrainingCancelledError

        raise TrainingCancelledError("SFT F/B job was cancelled")
    started = time.perf_counter()
    inputs, indices, hybridep_token_counts = _sft_command_inputs(runtime, job, batch)
    gradient_accumulator.before_forward_backward()
    try:
        with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
            state = run_megatron_sft_forward_backward_step(
                model_chunks=runtime.model,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                inputs=inputs,
                step_index=0,
                sample_index=indices,
                hybridep_token_counts=hybridep_token_counts,
            )
        reduced_loss, token_count = _global_sft_loss_and_tokens(state)
        gradients = slot_trainer.reduce_checkpoint_slot_grads(
            job.run_id,
            scale_grads=1.0 / float(token_count.item()),
        )
        gradient_accumulator.record(job.operation_id, token_count, gradients)
        return _sft_command_result(
            job=job,
            batch=batch,
            state=state,
            reduced_loss=reduced_loss,
            token_count=token_count,
            elapsed_s=time.perf_counter() - started,
            backward=True,
        )
    finally:
        slot_trainer.clear_checkpoint_slot_grads(job.run_id)


def execute_megatron_dynamic_lora_sft_forward_job(
    runtime: TrainingRuntime,
    job: SftForwardJobSpec,
    batch: SFTBatchData,
    *,
    cancelled: Event | None = None,
) -> dict[str, Any]:
    from art.megatron.lora import LoRASlotRef, use_lora_slot

    if cancelled is not None and cancelled.is_set():
        from art.megatron.runtime.trainer_run import TrainingCancelledError

        raise TrainingCancelledError("SFT forward job was cancelled")
    started = time.perf_counter()
    inputs, indices, hybridep_token_counts = _sft_command_inputs(runtime, job, batch)
    with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
        state = run_megatron_sft_forward_step(
            model_chunks=runtime.model,
            provider=runtime.provider,
            model_support_handler=runtime.model_support_handler,
            inputs=inputs,
            sample_index=indices,
            hybridep_token_counts=hybridep_token_counts,
        )
    reduced_loss, token_count = _global_sft_loss_and_tokens(state)
    return _sft_command_result(
        job=job,
        batch=batch,
        state=state,
        reduced_loss=reduced_loss,
        token_count=token_count,
        elapsed_s=time.perf_counter() - started,
        backward=False,
    )


def execute_megatron_rl_forward_job(
    runtime: TrainingRuntime,
    job: ForwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    cancelled: Event | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
) -> dict[str, Any]:
    """Run one packed forward without changing gradients or optimizer state."""
    started = time.perf_counter()
    global_sequences = resolve_global_grad_accumulation_sequences(
        job.config.grad_accumulation_sequences
    )
    if replay_bundle is None and packed_tensors.get("moe_routing_replay") is not None:
        replay_bundle = build_moe_routing_replay_bundle_from_packed_tensors(
            packed_tensors=packed_tensors,
            global_grad_accumulation_sequences=global_sequences,
        )
    configure_moe_routing_replay(
        runtime,
        replay_bundle=replay_bundle,
        strict=_moe_replay_strict(job),
    )
    replay_finalize_s = time.perf_counter() - started
    num_sequences = int(packed_tensors["tokens"].shape[0])
    num_steps = math.ceil(num_sequences / global_sequences)
    if ps.get_expert_model_parallel_world_size() > 1:
        topology = _infer_parallel_topology(runtime.model)
        required_capacity = max(
            (
                count
                for step_index in range(num_steps)
                for count in build_rl_hybridep_token_counts(
                    packed_tensors=packed_tensors,
                    step_index=step_index,
                    num_sequences=num_sequences,
                    global_grad_accumulation_sequences=global_sequences,
                    topology=topology,
                    provider=runtime.provider,
                    model_support_handler=runtime.model_support_handler,
                )
            ),
            default=0,
        )
        _ensure_hybridep_capacity(
            runtime,
            packed_sequence_length=int(packed_tensors["tokens"].shape[1]),
            context_parallel_size=topology.cp,
            required_capacity=required_capacity,
        )

    outputs = _precompute_reference_logprobs(
        runtime=runtime,
        packed_tensors=packed_tensors,
        sample_step_indices=_reference_sample_step_indices(
            num_sequences=num_sequences,
            num_steps=num_steps,
            global_grad_accumulation_sequences=global_sequences,
        ),
        global_grad_accumulation_sequences=global_sequences,
        cancelled=cancelled,
        purpose="forward-only",
    )
    return {
        "operation_id": job.operation_id,
        "token_logprobs": tuple(outputs[index] for index in sorted(outputs)),
        "metrics": {
            "time/forward_s": time.perf_counter() - started,
            "time/replay_finalize_s": replay_finalize_s,
        },
    }


def execute_megatron_dynamic_lora_forward_job(
    runtime: TrainingRuntime,
    job: ForwardJobSpec,
    packed_tensors: PackedTensors,
    *,
    cancelled: Event | None = None,
    replay_bundle: MoeRoutingReplayBundle | None = None,
) -> dict[str, Any]:
    from art.megatron.lora import LoRASlotRef, use_lora_slot

    with use_lora_slot(LoRASlotRef("checkpoint", job.run_id)):
        return execute_megatron_rl_forward_job(
            runtime,
            job,
            packed_tensors,
            cancelled=cancelled,
            replay_bundle=replay_bundle,
        )


def execute_megatron_load_state_job(
    runtime: TrainingRuntime,
    job: LoadStateJobSpec,
) -> None:
    """Replace resident single-run weights and reset or restore Adam exactly."""
    runtime.optimizer_snapshot_barrier.synchronize()
    runtime.resident_training_session_id = None
    runtime.resident_policy_step = None
    runtime.optimizer_state_loaded = False
    runtime.adapter_export_config = None
    runtime.optimizer = None
    _load_adapter_into_model(
        runtime.model,
        job.adapter_path,
        runtime.rank,
        handler=runtime.model_support_handler,
    )
    runtime.optimizer = _build_optimizer(runtime.model, runtime.optimizer_config)
    if job.restore_optimizer:
        assert job.optimizer_state_path is not None
        _load_optimizer(
            runtime,
            optimizer_state_path=job.optimizer_state_path,
            adapter_path=job.adapter_path,
            adapter_step=job.adapter_step,
            optimizer_generation_id=job.optimizer_generation_id,
            allow_missing=False,
        )
    else:
        _eager_initialize_optimizer_state(runtime.optimizer)
    runtime.adapter_export_dtypes = {}
    runtime.adapter_export_config = load_adapter_config(job.adapter_path)
    runtime.resident_training_session_id = job.training_session_id
    runtime.resident_policy_step = job.learner_version
    runtime.optimizer_state_loaded = True


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


def _load_lora_and_optimizer(
    runtime: TrainingRuntime,
    *,
    lora_path: str,
    optimizer_state_path: str,
    adapter_step: int,
) -> dict[str, torch.dtype]:
    runtime.optimizer_snapshot_barrier.synchronize()
    persistent_optimizer = runtime.optimizer if runtime.optimizer_persistent else None
    _load_adapter_into_model(
        runtime.model,
        lora_path,
        runtime.rank,
        handler=runtime.model_support_handler,
        optimizer=persistent_optimizer,
    )
    if persistent_optimizer is not None:
        return {}

    runtime.optimizer = _build_optimizer(
        runtime.model,
        runtime.optimizer_config,
    )
    assert runtime.optimizer is not None
    _load_optimizer(
        runtime,
        optimizer_state_path=optimizer_state_path,
        adapter_path=lora_path,
        adapter_step=adapter_step,
        allow_missing=True,
    )
    return {}


def _prepare_rl_training_state(
    runtime: TrainingRuntime,
    job: (
        TrainJobSpec
        | ForwardBackwardJobSpec
        | ForwardJobSpec
        | SftForwardBackwardJobSpec
        | SftForwardJobSpec
    ),
) -> dict[str, torch.dtype]:
    state_is_resident = (
        runtime.optimizer_persistent
        and runtime.resident_training_session_id == job.training_session_id
        and runtime.resident_policy_step == job.source_policy_step
        and runtime.optimizer_state_loaded
        and runtime.optimizer is not None
    )
    if state_is_resident:
        if (
            runtime.adapter_export_dtypes is None
            or runtime.adapter_export_config is None
        ):
            raise RuntimeError("Resident Megatron state has no LoRA export metadata")
        return runtime.adapter_export_dtypes

    runtime.optimizer_snapshot_barrier.synchronize()
    replacing_resident_state = runtime.resident_training_session_id is not None
    runtime.resident_training_session_id = None
    runtime.resident_policy_step = None
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
    if runtime.optimizer is None:
        runtime.optimizer = _build_optimizer(runtime.model, runtime.optimizer_config)
    assert runtime.optimizer is not None

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

    # Serialize the live LoRA dtype instead of perpetuating a source checkpoint's
    # PEFT-upcast FP32 dtype.
    runtime.adapter_export_dtypes = {}
    runtime.adapter_export_config = load_adapter_config(job.source_adapter_path)
    runtime.resident_training_session_id = job.training_session_id
    runtime.resident_policy_step = job.source_policy_step
    runtime.optimizer_state_loaded = True
    return runtime.adapter_export_dtypes


def _load_optimizer(
    runtime: TrainingRuntime,
    *,
    optimizer_state_path: str,
    adapter_path: str,
    adapter_step: int,
    allow_missing: bool,
    optimizer_generation_id: str | None = None,
) -> None:
    assert runtime.optimizer is not None
    shard_path = load_optimizer_state(
        runtime,
        optimizer_state_path=optimizer_state_path,
        adapter_path=adapter_path,
        adapter_step=adapter_step,
        optimizer_generation_id=optimizer_generation_id,
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


def _rl_forward_backward_metrics(
    step_result: MegatronForwardBackwardStepResult | TrainStepResult,
    *,
    num_gradient_steps: int,
    forward_backward_s: float,
) -> dict[str, float]:
    workload = step_result.workload
    metrics = {
        "loss/train": step_result.reduced_loss.item(),
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
        "time/forward_backward_s": forward_backward_s,
    }
    if step_result.kl_policy_ref is not None:
        metrics["loss/kl_policy_ref"] = step_result.kl_policy_ref
    metrics.update(step_result.loss_metrics)
    metrics.update(step_result.pipeline_metrics)
    return metrics


def _rl_step_metrics(
    step_result: TrainStepResult,
    *,
    num_gradient_steps: int,
    train_step_s: float,
) -> dict[str, float]:
    metrics = _rl_forward_backward_metrics(
        step_result,
        num_gradient_steps=num_gradient_steps,
        forward_backward_s=train_step_s,
    )
    metrics["time/gradient_step_train_s"] = metrics.pop("time/forward_backward_s")
    metrics["loss/grad_norm"] = step_result.grad_norm
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


def _install_schedule_finalize(model_chunks: ModelChunks) -> None:
    seen: set[int] = set()
    for chunk in model_chunks:
        config = _unwrap_model_config([chunk])
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        config.finalize_model_grads_func = finalize_model_grads_extended


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
) -> list[torch.Tensor]:
    if not (len(local_logprobs) == len(attention_states) == len(sequence_lengths)):
        raise ValueError("Context-parallel logprob/state/length counts differ")
    rows = local_logprobs[0].new_zeros((len(local_logprobs), max(sequence_lengths)))
    cp_group = None
    for index, (values, attention_state, seq_len) in enumerate(
        zip(local_logprobs, attention_states, sequence_lengths, strict=True)
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
        local_values = values.reshape(-1)
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
        if cursor != int(local_values.numel()):
            raise RuntimeError(
                "Context-parallel reference-logprob layout did not consume all values: "
                f"consumed={cursor}, values={local_values.numel()}"
            )
    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        rows, group=cp_group
    )
    return [
        rows[index : index + 1, :length]
        for index, length in enumerate(sequence_lengths)
    ]


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
    cancelled: Event | None = None,
    purpose: str = "KL reference",
) -> dict[int, torch.Tensor]:
    print0(
        runtime.rank,
        f"Precomputing {purpose} logprobs for",
        len(sample_step_indices),
        "local sequences",
    )
    hybridep_enabled = ps.get_expert_model_parallel_world_size() > 1
    topology = _infer_parallel_topology(runtime.model) if hybridep_enabled else None
    results: dict[int, torch.Tensor] = {}
    if not ps.model_parallel_is_initialized():
        for sample_index, step_index in sorted(sample_step_indices.items()):
            if cancelled is not None and cancelled.is_set():
                from art.megatron.runtime.trainer_run import TrainingCancelledError

                raise TrainingCancelledError("forward job was cancelled")
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
        if cancelled is not None and cancelled.is_set():
            from art.megatron.runtime.trainer_run import TrainingCancelledError

            raise TrainingCancelledError("forward job was cancelled")
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
    job: TrainJobSpec | ForwardBackwardJobSpec,
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
    forward_only: bool,
) -> SFTForwardBackwardState:
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("SFT schedule requires at least one trajectory")

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
        _install_schedule_finalize(model_chunks)

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
                "logprobs": token_output.restore(-output_tensor).detach(),
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
    logprobs = [cast(torch.Tensor, data["logprobs"]) for data in forward_data_store]
    if int(topology.cp) > 1:
        logprobs = _globalize_context_parallel_logprob_batch(
            local_logprobs=logprobs,
            attention_states=[prepared.attention_state for prepared in prepared_micros],
            sequence_lengths=[
                max(int(micro["attention_mask"].sum().item()), 1)
                for micro in micro_inputs
            ],
        )
    return SFTForwardBackwardState(
        raw_loss_sum=raw_loss_sum,
        new_logprobs=tuple(logprobs),
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
        forward_only=False,
    )


@torch.no_grad()
def run_megatron_sft_forward_step(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]],
    sample_index: int | list[int | None],
    hybridep_token_counts: list[int] | None = None,
) -> SFTForwardBackwardState:
    return _run_megatron_sft_schedule(
        model_chunks=model_chunks,
        provider=provider,
        model_support_handler=model_support_handler,
        inputs=inputs,
        step_index=0,
        sample_index=sample_index,
        hybridep_token_counts=hybridep_token_counts,
        forward_only=True,
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
    """Low-level fused oracle primitive retained for handler workflow parity."""
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


def run_megatron_rl_forward_backward_step(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: PackedTensors | list[PackedTensors],
    config: types.TrainConfig | RlForwardBackwardConfig,
    experimental_config: dev.TrainConfig,
    step_index: int,
    sample_index: int | list[int | None],
    ref_logprobs: torch.Tensor | list[torch.Tensor] | None = None,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    cp_lookahead_state: CpBatchLookaheadState | None = None,
    next_step_first_micro: PackedTensors | None = None,
    next_step_first_ref_logprobs: torch.Tensor | None = None,
    hybridep_token_counts: list[int] | None = None,
) -> RLForwardBackwardState:
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("run_training_step requires at least one packed sequence")

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

    _zero_grad_buffers(model_chunks)
    _install_schedule_finalize(model_chunks)

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
        token_output = _forward_prepared_rl_micro(
            model_chunks=model_chunks,
            model_chunk=model,
            model_support_handler=model_support_handler,
            prepared_micro=prepared,
            device=device,
        )

        def reduce_loss(output_tensor: torch.Tensor):
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
            if not micro_loss.requires_grad:
                raise RuntimeError(
                    "RL micro_loss is detached before pipeline backward: "
                    f"micro={item.order}, sample={item.sample_index}"
                )
            num_tokens = _local_trainable_token_count_tensor(
                [prepared.loss_inputs], device=device
            )
            return (
                micro_loss,
                num_tokens,
                {
                    "order": item.order,
                    "raw_loss_sum": micro_loss.detach(),
                    "probs_corr": loss_info.probs_corr.detach(),
                    "kl_policy_ref": (
                        None
                        if loss_info.kl_policy_ref is None
                        else float(loss_info.kl_policy_ref.item())
                    ),
                    "offpolicy_diagnostics": loss_info.offpolicy_diagnostics,
                    "new_logprobs": token_output.restore(new_logprobs.detach()).to(
                        "cpu"
                    ),
                },
            )

        return token_output.token_losses, reduce_loss

    forward_data_store = schedule.run(forward_step_func, forward_only=False)
    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.finalize_step(expect_recompute=True)
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
    return RLForwardBackwardState(
        raw_loss_sum=raw_loss_sum,
        probs_corr_total=probs_corr_total,
        kl_values=tuple(kl_values),
        new_logprobs=tuple(
            cast(torch.Tensor, data["new_logprobs"]) for data in pipeline_results
        ),
        token_count=token_count,
        micro_count=micro_count,
        schedule=schedule,
        loss_diagnostics=loss_diagnostics,
    )


def _finish_megatron_rl_forward_backward_step(
    state: RLForwardBackwardState,
) -> MegatronForwardBackwardStepResult:
    reduced_loss = _reduce_loss_sum(
        state.raw_loss_sum,
        state.token_count,
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )
    return MegatronForwardBackwardStepResult(
        reduced_loss=reduced_loss,
        probs_corr=float((state.probs_corr_total / state.micro_count).item()),
        kl_policy_ref=(
            sum(state.kl_values) / len(state.kl_values) if state.kl_values else None
        ),
        new_logprobs=list(state.new_logprobs),
        workload=state.schedule.training_workload(),
        loss_metrics=state.loss_diagnostics.to_metrics(
            group=ps.get_data_parallel_group(with_context_parallel=True),
        ),
        pipeline_metrics=state.schedule.telemetry.metrics(),
    )


def _finish_megatron_rl_forward_backward_job(
    states: tuple[RLForwardBackwardState, ...],
) -> MegatronForwardBackwardStepResult:
    if not states:
        raise ValueError("F/B job requires at least one gradient step")
    if len(states) == 1:
        return _finish_megatron_rl_forward_backward_step(states[0])

    raw_loss_sum = sum(
        (state.raw_loss_sum for state in states),
        torch.zeros_like(states[0].raw_loss_sum),
    )
    token_count = sum(
        (state.token_count for state in states),
        torch.zeros_like(states[0].token_count),
    )
    diagnostics = LossOffPolicyDiagnosticsAccumulator()
    for state in states:
        diagnostics.extend(state.loss_diagnostics)
    workloads = tuple(state.schedule.training_workload() for state in states)
    pipeline_metrics: dict[str, float] = {}
    for state in states:
        for key, value in state.schedule.telemetry.metrics().items():
            pipeline_metrics[key] = pipeline_metrics.get(key, 0.0) + value
    return MegatronForwardBackwardStepResult(
        reduced_loss=_reduce_loss_sum(
            raw_loss_sum,
            token_count,
            group=ps.get_data_parallel_group(with_context_parallel=True),
        ),
        probs_corr=float(
            (
                sum(
                    (state.probs_corr_total for state in states),
                    torch.zeros_like(states[0].probs_corr_total),
                )
                / sum(state.micro_count for state in states)
            ).item()
        ),
        kl_policy_ref=(
            sum(value for state in states for value in state.kl_values)
            / sum(len(state.kl_values) for state in states)
            if any(state.kl_values for state in states)
            else None
        ),
        new_logprobs=[value for state in states for value in state.new_logprobs],
        workload=TrainingStepWorkload(
            **{
                field: sum(getattr(workload, field) for workload in workloads)
                for field in TrainingStepWorkload.model_fields
            }
        ),
        loss_metrics=diagnostics.to_metrics(
            group=ps.get_data_parallel_group(with_context_parallel=True)
        ),
        pipeline_metrics=pipeline_metrics,
    )


def _finish_megatron_rl_step(
    state: RLForwardBackwardState,
    optimizer_result: MegatronOptimizerStepResult,
) -> TrainStepResult:
    forward_backward = _finish_megatron_rl_forward_backward_step(state)
    return TrainStepResult(
        **forward_backward.model_dump(),
        update_successful=optimizer_result.update_successful,
        grad_norm=optimizer_result.grad_norm,
        num_zeros_in_grad=optimizer_result.num_zeros_in_grad,
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
    )
    optimizer_result = run_megatron_optimizer_step(
        optimizer=optimizer,
        learning_rate=learning_rate,
        model_support_handler=model_support_handler,
        model_chunks=model_chunks,
        before_step=before_optimizer_step,
    )
    return _finish_megatron_rl_step(state, optimizer_result)


def _close_merged_weight_transfer_group(
    runtime: TrainingRuntime, *, abort: bool = False
) -> None:
    weight_transfer_group = runtime.merged_weight_transfer_group
    runtime.merged_weight_transfer_group = None
    runtime.merged_weight_transfer_init_info = None
    if weight_transfer_group is None:
        return
    shutdown = getattr(weight_transfer_group, "abort" if abort else "close", None)
    if shutdown is None and abort:
        shutdown = getattr(weight_transfer_group, "close", None)
    if shutdown is not None:
        shutdown()
