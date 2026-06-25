# isort: off
from art.megatron.runtime.runtime_env import configure_megatron_runtime_env

configure_megatron_runtime_env()
from art.megatron.runtime.bridge_runtime import install_art_bridge_runtime_patches

install_art_bridge_runtime_patches()
# isort: on

"""Megatron training runtime and public worker API.

Public cross-repo API consumed by serverless-training:
- build_training_runtime
- run_megatron_worker_loop
"""

import json
import math
import os
import random
import shutil
import time
from typing import Any, Callable, Literal, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import MegatronModule
from pydantic import BaseModel, ConfigDict, field_validator
import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir as inductor_cache_dir

from art import dev, types
from art.loss import Loss, LossInputs, loss_fn, shift_tensor
from art.megatron.context_parallel.types import (
    DispatchedPackedTensors,
    ParallelTopology,
    PreparedMegatronBatch,
)
from art.megatron.lora import apply_lora_adapters
from art.megatron.megatron_patches import install_fast_frozen_output_backward
from art.megatron.model_support.lora_disk import (
    load_adapter_config,
    load_lora_tensors_for_megatron,
)
from art.megatron.provider import (
    ProviderBundle,
    finalize_provider_bundle,
    prepare_provider_bundle,
)
from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    MoeRoutingReplayController,
)
from art.megatron.runtime.jobs import (
    DEFAULT_JOBS_DIR,
    DEFAULT_VLLM_WAKE_LOCK_PATH,
    MegatronJob,
    MegatronMergedTrainingJob,
    MegatronSFTTrainingJob,
    MegatronSyncJob,
    MegatronTrainingJob,
    MergedWeightTransferInitInfo,
    MergedWeightTransferSpec,
    load_megatron_job,
)
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
    _empty_new_logprobs_from_logits,
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
from art.megatron.training.sft_batches import load_sft_batch_from_disk
from art.megatron.training.trace import (
    attach_trace_token_uids,
    context_parallel_trace_token_uids_enabled,
    prepare_replay_local_input_token_uids,
)
from art.megatron.training.weight_offload import WeightOffloadManager
from art.megatron.weights.lora_publish import save_vllm_lora_from_model
from art.megatron.weights.merged_weight_export import (
    sync_merged_weights_to_vllm,
)
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.preprocessing.pack import (
    PackedTensors,
    packed_tensors_from_dir,
)

DEFAULT_MODEL_IDENTIFIER = "Qwen/Qwen3-30B-A3B-Instruct-2507"
_optimizer_stats_printed = False

__all__ = [
    "DEFAULT_MODEL_IDENTIFIER",
    "TrainingRuntime",
    "build_training_runtime",
    "run_megatron_worker_loop",
    "run_megatron_rl_job",
    "run_megatron_sft_job",
    "finalize_megatron_job",
]


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider_bundle: ProviderBundle
    provider: Any
    model: ModelChunks
    optimizer: Any | None
    optimizer_config: OptimizerConfig
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
    if runtime.moe_routing_replay_controller is not None:
        runtime.moe_routing_replay_controller.remove_router_patches()
        runtime.moe_routing_replay_controller = None

    if replay_bundle is not None and replay_bundle_path is not None:
        raise RuntimeError(
            "Provide either replay_bundle_path or replay_bundle, not both"
        )
    if replay_bundle is None and replay_bundle_path is None:
        return

    if replay_bundle is None:
        if replay_bundle_path is None:
            raise RuntimeError(
                "replay_bundle_path is required when replay_bundle is None"
            )
        replay_bundle = MoeRoutingReplayBundle.from_dir(replay_bundle_path)

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


def build_training_runtime(
    *,
    model_identifier: str | None = None,
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
        allow_unvalidated_arch=(
            os.environ.get("ART_MEGATRON_ALLOW_UNVALIDATED_ARCH", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if allow_unvalidated_arch is None
            else allow_unvalidated_arch
        ),
    )
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
            init_model_with_meta_device=True,
        ),
    )

    if not torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        raise RuntimeError(
            "torch.distributed must be initialized before building runtime"
        )
    rank = torch.distributed.get_rank()  # ty: ignore[possibly-missing-attribute]
    world_size = torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]

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
    )
    configure_moe_routing_replay(
        runtime,
        replay_bundle_path=moe_routing_replay_path,
        replay_bundle=moe_routing_replay_bundle,
        strict=moe_routing_replay_strict,
    )
    return runtime


def run_megatron_worker_loop(
    runtime: TrainingRuntime,
    *,
    supports_sft: bool,
    wait_until_ready: Callable[[], None] | None = None,
    before_job: Callable[[], None] | None = None,
    after_job: Callable[[], None] | None = None,
) -> None:
    jobs_dir = os.environ.get("ART_MEGATRON_JOBS_DIR", DEFAULT_JOBS_DIR)
    while True:
        torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
        os.makedirs(jobs_dir, exist_ok=True)
        job_names = sorted(
            job_name for job_name in os.listdir(jobs_dir) if job_name.endswith(".json")
        )
        if not job_names:
            time.sleep(1)
            continue

        if wait_until_ready is not None:
            wait_until_ready()
        if before_job is not None:
            before_job()

        job_path = os.path.join(jobs_dir, job_names[0])
        job = _load_megatron_job(job_path, supports_sft=supports_sft)
        print0(runtime.rank, "Loaded job from", job_path)
        print0(runtime.rank, "Job:", job)

        job_completed = False
        try:
            _run_megatron_job(runtime, job)
            job_completed = True
        finally:
            if job_completed and after_job is not None:
                after_job()

        finalize_megatron_job(
            runtime,
            job_path=job_path,
            log_path=job.log_path,
            cleanup_path=_job_cleanup_path(job),
        )


def run_megatron_rl_job(
    runtime: TrainingRuntime,
    job: MegatronTrainingJob | MegatronMergedTrainingJob,
) -> None:
    packed_tensors = None
    adapter_model = None
    template = None
    zero_template = None
    ref_logprobs_by_index = None
    cp_lookahead_state = None
    next_step_first_micro = None
    next_step_first_ref_logprobs = None
    step_result = None

    try:
        configure_moe_routing_replay(
            runtime,
            replay_bundle_path=job.moe_routing_replay_path,
            strict=job.moe_routing_replay_strict,
        )
        adapter_model = _load_lora_and_optimizer(
            runtime,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )

        print0(
            runtime.rank,
            "Loading packed tensors from",
            job.disk_packed_tensors["dir"],
        )
        packed_tensors = packed_tensors_from_dir(**job.disk_packed_tensors)
        template = _clone_packed_tensors(select_indexed_inputs(packed_tensors, 0))
        zero_template = _zero_contribution_inputs(template)
        num_sequences = job.disk_packed_tensors["num_sequences"]
        global_grad_accumulation_sequences = resolve_global_grad_accumulation_sequences(
            job.config.grad_accumulation_sequences
        )
        num_steps = math.ceil(num_sequences / global_grad_accumulation_sequences)
        ref_logprobs_by_index = _prepare_kl_reference_logprobs(
            runtime=runtime,
            job=job,
            adapter_model=adapter_model,
            packed_tensors=packed_tensors,
            num_sequences=num_sequences,
            num_steps=num_steps,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        topology = _infer_parallel_topology(runtime.model)
        cp_lookahead_state = CpBatchLookaheadState() if int(topology.cp) > 1 else None
        for step_index in range(num_steps):
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
                if ref_logprobs_by_index is not None
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
                if cp_lookahead_state is not None and ref_logprobs_by_index is not None
                else None
            )
            step_result = run_training_step(
                model_chunks=runtime.model,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                optimizer=runtime.optimizer,
                learning_rate=job.config.learning_rate,
                inputs=micro_inputs,
                config=job.config,
                experimental_config=cast(dev.TrainConfig, job.experimental_config),
                ref_logprobs=ref_logprobs,
                step_index=step_index,
                sample_index=micro_indices,
                moe_routing_replay_controller=runtime.moe_routing_replay_controller,
                cp_lookahead_state=cp_lookahead_state,
                next_step_first_micro=next_step_first_micro,
                next_step_first_ref_logprobs=next_step_first_ref_logprobs,
            )
            print0(
                runtime.rank,
                "Correlation between old and new probabilities:",
                step_result.probs_corr,
            )

            if runtime.rank == 0:
                with open(job.log_path, "a+", encoding="utf-8") as log_file:
                    metrics = {
                        "loss": step_result.reduced_loss.item(),
                        "grad_norm": step_result.grad_norm,
                        "probs_corr": step_result.probs_corr,
                        TRAIN_GRADIENT_STEPS_KEY: num_steps,
                    }
                    if step_result.kl_policy_ref is not None:
                        metrics["kl_policy_ref"] = step_result.kl_policy_ref
                    log_msg = json.dumps(metrics)
                    print("Logging", log_msg)
                    log_file.write(log_msg + "\n")

        _save_lora_and_optimizer(
            runtime,
            adapter_model=adapter_model,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )
    finally:
        configure_moe_routing_replay(runtime)
        if packed_tensors is not None:
            del packed_tensors
        if adapter_model is not None:
            del adapter_model
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
        if step_result is not None:
            del step_result
        if cp_lookahead_state is not None:
            cp_lookahead_state.pending_prepared_micro = None
            del cp_lookahead_state


def run_megatron_sft_job(
    runtime: TrainingRuntime,
    job: MegatronSFTTrainingJob,
) -> None:
    adapter_model = None

    try:
        configure_moe_routing_replay(runtime)
        adapter_model = _load_lora_and_optimizer(
            runtime,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )

        assert runtime.optimizer is not None
        runtime.optimizer.config.clip_grad = job.max_grad_norm
        for param_group in runtime.optimizer.param_groups:
            param_group["weight_decay"] = job.weight_decay

        grad_accumulation_sequences = resolve_global_grad_accumulation_sequences(
            job.grad_accumulation_sequences
        )
        checkpoint_interval = job.internal_checkpoint_interval

        for batch_idx in range(job.num_batches):
            batch_start_time = time.perf_counter()
            batch_dir = os.path.join(job.sft_data_dir, f"batch_{batch_idx:06d}")
            batch_metadata, trajectory_tensors = load_sft_batch_from_disk(batch_dir)
            num_trajectories = int(batch_metadata["num_trajectories"])
            if not trajectory_tensors:
                raise RuntimeError(f"SFT batch {batch_idx} is empty")
            if num_trajectories != len(trajectory_tensors):
                raise RuntimeError(
                    "SFT batch metadata does not match trajectory count: "
                    f"{num_trajectories} != {len(trajectory_tensors)}"
                )

            global_tokens = max(
                int(batch_metadata.get("num_tokens", 0)),
                1,
            )
            if "num_tokens" not in batch_metadata:
                global_tokens = max(
                    sum(
                        int(inputs["attention_mask"].sum().item())
                        for inputs in trajectory_tensors
                    ),
                    1,
                )
            global_trainable_tokens = max(
                int(batch_metadata["num_trainable_tokens"]),
                1,
            )
            template = _clone_sft_tensors(trajectory_tensors[0])
            zero_template = _zero_contribution_sft_inputs(template)
            micro_indices = build_micro_sample_indices(
                step_index=0,
                num_sequences=num_trajectories,
                global_grad_accumulation_sequences=grad_accumulation_sequences,
            )
            micro_inputs = select_sft_micro_inputs(
                trajectory_tensors,
                micro_indices,
                zero_template,
            )
            step_result = run_megatron_sft_step(
                model_chunks=runtime.model,
                provider=runtime.provider,
                model_support_handler=runtime.model_support_handler,
                optimizer=runtime.optimizer,
                learning_rate=job.learning_rates[batch_idx],
                inputs=micro_inputs,
                step_index=batch_idx,
                sample_index=micro_indices,
                global_grad_accumulation_sequences=grad_accumulation_sequences,
                moe_routing_replay_controller=runtime.moe_routing_replay_controller,
            )
            batch_time = time.perf_counter() - batch_start_time
            tokens_per_second = global_tokens / batch_time if batch_time > 0 else 0.0
            completed_batches = batch_idx + 1

            if (
                checkpoint_interval is not None
                and completed_batches < job.num_batches
                and completed_batches % checkpoint_interval == 0
            ):
                _save_lora_and_optimizer(
                    runtime,
                    adapter_model=adapter_model,
                    lora_path=job.lora_path,
                    optimizer_state_path=job.optimizer_state_path,
                )
                torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]

            if runtime.rank == 0:
                with open(job.log_path, "a+", encoding="utf-8") as log_file:
                    log_msg = json.dumps(
                        {
                            "loss": step_result.reduced_loss.item(),
                            "learning_rate": job.learning_rates[batch_idx],
                            "grad_norm": float(step_result.grad_norm),
                            "num_trajectories": float(num_trajectories),
                            "num_tokens": float(global_tokens),
                            "num_trainable_tokens": float(global_trainable_tokens),
                            "tokens_per_second": tokens_per_second,
                        }
                    )
                    print("Logging SFT", log_msg)
                    log_file.write(log_msg + "\n")

        _save_lora_and_optimizer(
            runtime,
            adapter_model=adapter_model,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )
    finally:
        if adapter_model is not None:
            del adapter_model


def _load_megatron_job(job_path: str, *, supports_sft: bool) -> MegatronJob:
    with open(job_path, "rb") as handle:
        job = load_megatron_job(handle.read())
    if isinstance(job, MegatronSFTTrainingJob) and not supports_sft:
        raise NotImplementedError("SFT jobs are not supported in this worker loop")
    return job


def _run_megatron_job(runtime: TrainingRuntime, job: MegatronJob) -> None:
    if isinstance(job, MegatronSyncJob):
        adapter_model = _load_adapter_into_model(
            runtime.model,
            job.lora_path,
            runtime.rank,
            handler=runtime.model_support_handler,
        )
        del adapter_model
        _sync_merged_weights_to_vllm(
            runtime,
            job.merged_weight_transfer,
            lora_path=job.lora_path,
            pause_generation=False,
        )
        return
    if isinstance(job, MegatronSFTTrainingJob):
        run_megatron_sft_job(runtime, job)
        return
    run_megatron_rl_job(runtime, job)
    if isinstance(job, MegatronMergedTrainingJob):
        _sync_merged_weights_to_vllm(
            runtime,
            job.merged_weight_transfer,
            lora_path=job.lora_path,
            pause_generation=True,
        )


def _job_cleanup_path(job: MegatronJob) -> str | None:
    if isinstance(job, MegatronSyncJob):
        return None
    if isinstance(job, MegatronSFTTrainingJob):
        return job.sft_data_dir
    return job.disk_packed_tensors["dir"]


def _load_lora_and_optimizer(
    runtime: TrainingRuntime,
    *,
    lora_path: str,
    optimizer_state_path: str,
) -> dict[str, torch.Tensor]:
    adapter_model = _load_adapter_into_model(
        runtime.model,
        lora_path,
        runtime.rank,
        handler=runtime.model_support_handler,
    )
    runtime.optimizer = _build_optimizer(
        runtime.model,
        runtime.optimizer_config,
    )
    assert runtime.optimizer is not None

    optimizer_shard_path = os.path.join(
        optimizer_state_path,
        f"{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.pt",
    )
    if os.path.exists(optimizer_shard_path):
        print0(runtime.rank, "Loading optimizer state from", optimizer_shard_path)
        runtime.optimizer.load_state_dict(torch.load(optimizer_shard_path))
    else:
        print0(
            runtime.rank,
            "No optimizer state found at",
            optimizer_shard_path,
            "- resetting optimizer for new run",
        )
        _eager_initialize_optimizer_state(runtime.optimizer)
    return adapter_model


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
    load_adapter_into_model(model_chunks, adapter_model, optimizer)
    return adapter_model


def _save_lora_and_optimizer(
    runtime: TrainingRuntime,
    *,
    adapter_model: dict[str, torch.Tensor],
    lora_path: str,
    optimizer_state_path: str,
) -> None:
    assert runtime.optimizer is not None
    save_vllm_lora_from_model(
        model=runtime.model,
        adapter_model=adapter_model,
        handler=runtime.model_support_handler,
        adapter_config=load_adapter_config(lora_path),
        output_dir=lora_path,
        rank=runtime.rank,
        world_size=runtime.world_size,
    )
    _save_optimizer(runtime, optimizer_state_path=optimizer_state_path)


def _save_optimizer(runtime: TrainingRuntime, *, optimizer_state_path: str) -> None:
    assert runtime.optimizer is not None
    optimizer_shard_path = os.path.join(
        optimizer_state_path,
        f"{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.pt",
    )
    print("Saving optimizer shard to", optimizer_shard_path)
    os.makedirs(optimizer_state_path, exist_ok=True)
    torch.save(runtime.optimizer.state_dict(), optimizer_shard_path)


def finalize_megatron_job(
    runtime: TrainingRuntime,
    *,
    job_path: str | None,
    log_path: str,
    cleanup_path: str | None,
) -> None:
    torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
    if runtime.rank != 0:
        return

    if job_path is not None and os.path.exists(job_path):
        os.remove(job_path)
    if cleanup_path is not None and os.path.exists(cleanup_path):
        shutil.rmtree(cleanup_path)
    with open(log_path, "a+", encoding="utf-8") as log_file:
        log_file.write("all done\n")


def _placeholder_attention_mask(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)


def load_adapter_into_model(
    model_chunks: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
    optimizer: Any | None = None,
) -> None:
    with torch.no_grad():
        for chunk in model_chunks:
            for module in chunk.modules():
                if hasattr(module, "load_lora"):
                    module.load_lora(adapter_model)  # type: ignore[attr-defined]

    if optimizer is None:
        return
    optimizer.reload_model_params()


def _optimizer_step(
    optimizer: Any,
    learning_rate: float,
) -> tuple[bool, float, int | None]:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    update_successful, grad_norm, num_zeros_in_grad = cast(
        tuple[bool, float, int | None], optimizer.step()
    )
    optimizer.zero_grad()
    return update_successful, grad_norm, num_zeros_in_grad


def _reduce_loss(
    loss: torch.Tensor,
    op: Any = torch.distributed.ReduceOp.AVG,  # ty: ignore[possibly-missing-attribute]
    group: Any | None = None,
) -> torch.Tensor:
    reduced_loss = loss.detach().clone()
    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        reduced_loss,
        op=op,
        group=group,
    )
    return reduced_loss


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
    model_support_handler: Any,
    prepared_micro: PreparedRLMicroInputs,
    device: torch.device,
) -> torch.Tensor:
    model_forward_kwargs = dict(
        input_ids=prepared_micro.model_tokens,
        position_ids=prepared_micro.model_input_pos,
        attention_mask=_placeholder_attention_mask(device),
        packed_seq_params=prepared_micro.packed_seq_params,
        **model_support_handler.get_forward_kwargs(
            model_chunks[0],
            attention_bias=prepared_micro.attention_state,
        ),
    )
    with attach_trace_token_uids(model_chunks, prepared_micro.local_token_uids):
        if int(prepared_micro.model_tokens.numel()) == 0:
            logits = model_chunks[0](**model_forward_kwargs, labels=None)
            return _empty_new_logprobs_from_logits(logits, prepared_micro.model_labels)
        return -model_chunks[0](
            **model_forward_kwargs,
            labels=prepared_micro.model_labels,
        )


def _globalize_context_parallel_logprobs(
    *,
    local_logprobs: torch.Tensor,
    attention_state: Any,
    seq_len: int,
) -> torch.Tensor:
    rank_plan = getattr(attention_state, "rank_plan", None)
    cp_group = getattr(attention_state, "cp_group", None)
    if rank_plan is None or cp_group is None:
        raise RuntimeError("Context-parallel reference logprobs require a rank plan")

    global_logprobs = local_logprobs.new_zeros((1, seq_len))
    local_values = local_logprobs.reshape(-1)
    cursor = 0
    for range_ in rank_plan.local_row_ranges:
        if range_ is None:
            continue
        size = int(range_.size())
        if size <= 0:
            continue
        global_logprobs[0, int(range_.start) : int(range_.end)] = local_values[
            cursor : cursor + size
        ]
        cursor += size

    torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
        global_logprobs,
        group=cp_group,
    )
    return global_logprobs


@torch.no_grad()
def _calculate_megatron_logprobs(
    *,
    model_chunks: ModelChunks,
    provider: Any,
    model_support_handler: Any,
    inputs: PackedTensors,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
    step_index: int | None = None,
    sample_index: int | None = None,
    global_grad_accumulation_sequences: int | None = None,
) -> torch.Tensor:
    if moe_routing_replay_controller is not None:
        if step_index is None or sample_index is None:
            raise ValueError(
                "step_index and sample_index are required for routing replay"
            )
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=sample_index,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        moe_routing_replay_controller.begin_micro(sample_index, 0)

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
        prepared_micro, _pending_prepared_micro = _prepare_current_rl_micro(
            inputs,
            device=device,
            topology=topology,
            provider=provider,
            model_support_handler=model_support_handler,
            ref_logprobs=None,
            trace_token_uids=trace_token_uids,
            pending_prepared_micro=None,
        )
        prepare_replay_local_input_token_uids(
            moe_routing_replay_controller,
            prepared_micro.local_token_uids,
            prepared_micro.attention_state,
        )
        logprobs = _forward_prepared_rl_micro(
            model_chunks=model_chunks,
            model_support_handler=model_support_handler,
            prepared_micro=prepared_micro,
            device=device,
        )
        if int(topology.cp) > 1:
            logprobs = _globalize_context_parallel_logprobs(
                local_logprobs=logprobs,
                attention_state=prepared_micro.attention_state,
                seq_len=int(inputs["tokens"].shape[1]),
            )
        forward_succeeded = True
    finally:
        for chunk, was_training in zip(model_chunks, previous_training_modes):
            chunk.train(was_training)
        if moe_routing_replay_controller is not None and forward_succeeded:
            moe_routing_replay_controller.finalize_step()
    return logprobs.detach().cpu()


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
    return {
        sample_index: _calculate_megatron_logprobs(
            model_chunks=runtime.model,
            provider=runtime.provider,
            model_support_handler=runtime.model_support_handler,
            inputs=select_indexed_inputs(packed_tensors, sample_index),
            moe_routing_replay_controller=runtime.moe_routing_replay_controller,
            step_index=step_index,
            sample_index=sample_index,
            global_grad_accumulation_sequences=global_grad_accumulation_sequences,
        )
        for sample_index, step_index in sorted(sample_step_indices.items())
    }


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
    job: MegatronTrainingJob | MegatronMergedTrainingJob,
    adapter_model: dict[str, torch.Tensor],
    packed_tensors: PackedTensors,
    num_sequences: int,
    num_steps: int,
    global_grad_accumulation_sequences: int,
) -> dict[int, torch.Tensor] | None:
    if job.config.kl_penalty_coef <= 0.0:
        return None

    ref_adapter_path = cast(dev.TrainConfig, job.experimental_config).get(
        "kl_ref_adapter_path"
    )
    if ref_adapter_path is None:
        raise RuntimeError(
            "KL penalty is enabled but no kl_ref_adapter_path was provided. "
            "Megatron training requires an explicit reference LoRA path; pass "
            "kl_penalty_reference_step=0 for the identity/base reference or "
            "provide kl_ref_adapter_path."
        )

    adapter_swapped = os.path.abspath(ref_adapter_path) != os.path.abspath(
        job.lora_path
    )
    loaded_ref_adapter = False
    try:
        if adapter_swapped:
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
            load_adapter_into_model(runtime.model, adapter_model, runtime.optimizer)


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
    global_grad_accumulation_sequences: int | None,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
) -> TrainStepResult:
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("run_megatron_sft_step requires at least one trajectory")

    if isinstance(sample_index, list):
        if len(sample_index) != len(micro_inputs):
            raise ValueError(
                "sample_index list length must match number of micro inputs: "
                f"{len(sample_index)} != {len(micro_inputs)}"
            )
        micro_sample_indices: list[int | None] = sample_index
    else:
        assert len(micro_inputs) == 1
        micro_sample_indices = [sample_index]

    if moe_routing_replay_controller is not None:
        resolved_global_grad_accumulation_sequences = (
            resolve_global_grad_accumulation_sequences(
                global_grad_accumulation_sequences
            )
        )
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=micro_sample_indices,
            global_grad_accumulation_sequences=resolved_global_grad_accumulation_sequences,
        )

    topology = _infer_parallel_topology(model_chunks)
    device = next(model_chunks[0].parameters()).device
    trace_token_uids = context_parallel_trace_token_uids_enabled(
        topology,
        moe_routing_replay_controller,
    )

    for chunk in model_chunks:
        chunk.zero_grad_buffer()  # type: ignore[call-non-callable]

    raw_loss_sum: torch.Tensor | None = None
    loss_inputs_for_count: list[dict[str, torch.Tensor] | PreparedSFTMicroInputs] = []
    pending_prepared_micro: PreparedMegatronBatch | None = None

    for micro_order, micro in enumerate(micro_inputs):
        if moe_routing_replay_controller is not None:
            moe_routing_replay_controller.begin_micro(
                micro_sample_indices[micro_order],
                micro_order,
            )
        prepared_micro, pending_prepared_micro = _prepare_current_sft_micro(
            micro,
            device=device,
            topology=topology,
            provider=provider,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
            pending_prepared_micro=pending_prepared_micro,
        )
        prepare_replay_local_input_token_uids(
            moe_routing_replay_controller,
            prepared_micro.local_token_uids,
            prepared_micro.attention_state,
        )
        with attach_trace_token_uids(model_chunks, prepared_micro.local_token_uids):
            per_token_loss: torch.Tensor = model_chunks[0](
                input_ids=prepared_micro.input_ids,
                position_ids=prepared_micro.position_ids,
                attention_mask=_placeholder_attention_mask(device),
                labels=prepared_micro.labels,
                packed_seq_params=prepared_micro.packed_seq_params,
                **model_support_handler.get_forward_kwargs(
                    model_chunks[0],
                    attention_bias=prepared_micro.attention_state,
                ),
            )
        masked_loss = (
            per_token_loss[prepared_micro.loss_mask].sum() + per_token_loss.sum() * 0.0
        )
        masked_loss.backward()
        pending_prepared_micro = _prepare_next_sft_cp_micro(
            _next_micro_lookahead(micro_inputs, micro_order),
            device=device,
            topology=topology,
            provider=provider,
            model_support_handler=model_support_handler,
            trace_token_uids=trace_token_uids,
        )
        detached_micro_loss = masked_loss.detach()
        if raw_loss_sum is None:
            raw_loss_sum = detached_micro_loss
        else:
            raw_loss_sum = raw_loss_sum + detached_micro_loss
        loss_inputs_for_count.append(prepared_micro)

    if raw_loss_sum is None:
        raise RuntimeError("run_megatron_sft_step did not produce outputs")

    num_tokens = _local_trainable_sft_token_count_tensor(
        loss_inputs_for_count,
        device=device,
    )
    flush_param_grads_to_main_grads(model_chunks)
    finalize_model_grads_extended(
        as_megatron_api_chunks(model_chunks), num_tokens=num_tokens
    )
    update_successful, grad_norm, num_zeros_in_grad = _optimizer_step(
        optimizer,
        learning_rate,
    )
    global_num_tokens = max(num_tokens.item(), 1.0)
    reduced_loss = _reduce_loss(
        raw_loss_sum / global_num_tokens,
        op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )

    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.finalize_step()

    return TrainStepResult(
        reduced_loss=reduced_loss,
        probs_corr=1.0,
        new_logprobs=None,
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
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
) -> TrainStepResult:
    micro_inputs = inputs if isinstance(inputs, list) else [inputs]
    if not micro_inputs:
        raise ValueError("run_training_step requires at least one packed sequence")

    if isinstance(sample_index, list):
        if len(sample_index) != len(micro_inputs):
            raise ValueError(
                "sample_index list length must match number of micro inputs: "
                f"{len(sample_index)} != {len(micro_inputs)}"
            )
        micro_sample_indices: list[int | None] = sample_index
    else:
        assert len(micro_inputs) == 1
        micro_sample_indices = [sample_index]

    if moe_routing_replay_controller is not None:
        resolved_global_grad_accumulation_sequences = (
            resolve_global_grad_accumulation_sequences(
                config.grad_accumulation_sequences
            )
        )
        moe_routing_replay_controller.set_step(
            step_index=step_index,
            sample_index=micro_sample_indices,
            global_grad_accumulation_sequences=resolved_global_grad_accumulation_sequences,
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

    for chunk in model_chunks:
        chunk.zero_grad_buffer()  # type: ignore[call-non-callable]

    micro_count = len(micro_inputs)
    raw_loss_sum: torch.Tensor | None = None
    loss_inputs_for_count: list[LossInputs | DispatchedPackedTensors] = []
    probs_corr_total: torch.Tensor | None = None
    kl_policy_ref_sum = 0.0
    kl_policy_ref_count = 0
    new_logprobs_gpu: list[torch.Tensor] = []

    def begin_micro(micro_order: int) -> None:
        if moe_routing_replay_controller is not None:
            moe_routing_replay_controller.begin_micro(
                micro_sample_indices[micro_order],
                micro_order,
            )

    for micro_order in range(micro_count):
        begin_micro(micro_order)
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
        prepare_replay_local_input_token_uids(
            moe_routing_replay_controller,
            prepared_micro.local_token_uids,
            prepared_micro.attention_state,
        )

        new_logprobs = _forward_prepared_rl_micro(
            model_chunks=model_chunks,
            model_support_handler=model_support_handler,
            prepared_micro=prepared_micro,
            device=device,
        )

        loss_info = loss_fn(
            prepared_micro.loss_inputs,
            new_logprobs=new_logprobs,
            ref_logprobs=prepared_micro.ref_logprobs,
            entropies=None,
            experimental_config=experimental_config,
            reduction="sum",
        )
        micro_loss = loss_info.policy_loss + new_logprobs.sum() * 0.0
        if not micro_loss.requires_grad:
            assistant_tokens = _count_trainable_tokens(prepared_micro.loss_inputs)
            nonzero_weights = int(
                torch.count_nonzero(
                    prepared_micro.loss_inputs.align_inputs().weights
                ).item()
            )
            nonzero_advantages = int(
                torch.count_nonzero(
                    prepared_micro.loss_inputs.align_inputs().advantages
                ).item()
            )
            raise RuntimeError(
                "RL micro_loss is detached before backward: "
                f"new_logprobs.requires_grad={new_logprobs.requires_grad}, "
                f"policy_loss_sum_requires_grad={loss_info.policy_loss_sum.requires_grad}, "
                f"assistant_tokens={assistant_tokens}, "
                f"nonzero_weights={nonzero_weights}, "
                f"nonzero_advantages={nonzero_advantages}"
            )
        micro_loss.backward()
        loss_inputs_for_count.append(prepared_micro.loss_inputs)
        del prepared_micro
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
        detached_probs_corr = loss_info.probs_corr.detach()
        if probs_corr_total is None:
            probs_corr_total = detached_probs_corr
        else:
            probs_corr_total = probs_corr_total + detached_probs_corr
        if loss_info.kl_policy_ref is not None:
            kl_policy_ref_sum += float(loss_info.kl_policy_ref.item())
            kl_policy_ref_count += 1
        detached_micro_loss = micro_loss.detach()
        if raw_loss_sum is None:
            raw_loss_sum = detached_micro_loss
        else:
            raw_loss_sum = raw_loss_sum + detached_micro_loss
        del loss_info
        del micro_loss
        new_logprobs_gpu.append(new_logprobs.detach())
        del new_logprobs

    if raw_loss_sum is None:
        raise RuntimeError("run_training_step did not produce outputs")
    if probs_corr_total is None:
        raise RuntimeError("run_training_step did not accumulate probs_corr")
    if cp_lookahead_state is not None:
        cp_lookahead_state.pending_prepared_micro = pending_prepared_micro

    token_count = _local_trainable_token_count_tensor(
        loss_inputs_for_count,
        device=device,
    )
    finalize_model_grads_extended(
        as_megatron_api_chunks(model_chunks),
        num_tokens=token_count,
    )
    update_successful, grad_norm, num_zeros_in_grad = _optimizer_step(
        optimizer,
        learning_rate,
    )
    global_num_tokens = max(token_count.item(), 1.0)
    reduced_loss = _reduce_loss(
        raw_loss_sum / global_num_tokens,
        op=torch.distributed.ReduceOp.SUM,  # ty: ignore[possibly-missing-attribute]
        group=ps.get_data_parallel_group(with_context_parallel=True),
    )

    if moe_routing_replay_controller is not None:
        moe_routing_replay_controller.finalize_step()

    return TrainStepResult(
        reduced_loss=reduced_loss,
        probs_corr=float((probs_corr_total / micro_count).item()),
        kl_policy_ref=(
            kl_policy_ref_sum / kl_policy_ref_count if kl_policy_ref_count > 0 else None
        ),
        new_logprobs=[
            tensor.to(device="cpu", non_blocking=True) for tensor in new_logprobs_gpu
        ],
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
    )


def _sync_merged_weights_to_vllm(
    runtime: TrainingRuntime,
    spec: MergedWeightTransferSpec,
    *,
    lora_path: str,
    pause_generation: bool,
) -> None:
    adapter_model = load_lora_tensors_for_megatron(
        lora_path,
        handler=runtime.model_support_handler,
    )
    (
        runtime.merged_weight_transfer_group,
        runtime.merged_weight_transfer_init_info,
    ) = sync_merged_weights_to_vllm(
        bridge=runtime.bridge,
        model=runtime.model,
        model_support_handler=runtime.model_support_handler,
        adapter_model=adapter_model,
        adapter_config=load_adapter_config(lora_path),
        rank=runtime.rank,
        world_size=runtime.world_size,
        merged_weight_transfer_group=runtime.merged_weight_transfer_group,
        merged_weight_transfer_init_info=runtime.merged_weight_transfer_init_info,
        spec=spec,
        pause_generation=pause_generation,
    )


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


def _run_service_loop(runtime: TrainingRuntime) -> None:
    weight_offload = WeightOffloadManager.from_env(
        model=runtime.model,
        rank=runtime.rank,
        compile_enabled=runtime.transformer_layers_compiled,
    )
    weight_offload.install()
    wake_lock_path = os.environ.get(
        "ART_MEGATRON_WAKE_LOCK_PATH", DEFAULT_VLLM_WAKE_LOCK_PATH
    )

    def wait_until_ready() -> None:
        while os.path.exists(wake_lock_path):
            time.sleep(0.2)

    def before_job() -> None:
        weight_offload.before_job()

    def after_job() -> None:
        runtime.optimizer = None
        weight_offload.after_job()

    worker_error = False
    try:
        after_job()
        run_megatron_worker_loop(
            runtime,
            supports_sft=True,
            wait_until_ready=wait_until_ready,
            before_job=before_job,
            after_job=after_job,
        )
    except BaseException:
        worker_error = True
        raise
    finally:
        _close_merged_weight_transfer_group(runtime, abort=worker_error)


def main() -> None:
    runtime = build_training_runtime(
        model_identifier=os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER),
        build_optimizer=False,
    )
    _run_service_loop(runtime)


if __name__ == "__main__":
    main()
