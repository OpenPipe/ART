# isort: off
from art.megatron.runtime_env import configure_megatron_runtime_env

configure_megatron_runtime_env()
# isort: on

"""Megatron training runtime and public worker API.

Public cross-repo API consumed by serverless-training:
- build_training_runtime
- run_megatron_worker_loop
- merge_lora_adapter
"""

import gc
import importlib
import json
import math
import os
import random
import shutil
import time
from typing import Any, Callable, cast

from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from pydantic import BaseModel, ConfigDict, field_validator
import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir as inductor_cache_dir

from art import dev, types
from art.loss import loss_fn, shift_tensor
from art.megatron.compile_workarounds import install_torch_compile_workarounds
from art.megatron.finalize_grads import finalize_model_grads_extended
from art.megatron.flex_attention import create_shared_prefix_attention_state
from art.megatron.jobs import (
    DEFAULT_JOBS_DIR,
    DEFAULT_VLLM_WAKE_LOCK_PATH,
    MegatronJob,
    MegatronSFTTrainingJob,
    MegatronTrainingJob,
)
from art.megatron.lora import apply_lora_adapters
from art.megatron.merge import load_lora_adapter_state_dict, merge_lora_adapter
from art.megatron.model_chunks import (
    ModelChunks,
    as_megatron_api_chunks,
    unwrap_megatron_chunk,
    validate_model_chunks,
)
from art.megatron.offload import (
    OffloadState,
    offload_to_cpu,
    reload_to_gpu,
)
from art.megatron.provider import _env_flag, get_provider
from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    MoeRoutingReplayController,
)
from art.megatron.sft_batches import load_sft_batch_from_disk
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY
from art.preprocessing.pack import (
    PackedTensors,
    packed_tensors_from_dir,
)

safetensors = importlib.import_module("safetensors")
safetensors_torch = importlib.import_module("safetensors.torch")
safe_open = safetensors.safe_open
save_file = safetensors_torch.save_file

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
    "merge_lora_adapter",
]


class TrainingRuntime(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: Any
    model: ModelChunks
    optimizer: Any | None
    optimizer_config: OptimizerConfig
    rank: int
    world_size: int
    moe_routing_replay_controller: MoeRoutingReplayController | None = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: ModelChunks) -> ModelChunks:
        validate_model_chunks(value)
        return value


class TrainStepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reduced_loss: torch.Tensor
    probs_corr: float
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


def _frozen_linear_grad_input(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if grad_output.dim() <= 2 or weight.dim() != 2:
        return grad_output.matmul(weight)
    grad_output_2d = grad_output.reshape(-1, int(grad_output.shape[-1]))
    grad_input_2d = grad_output_2d.matmul(weight)
    return grad_input_2d.reshape(*grad_output.shape[:-1], int(weight.shape[-1]))


def _install_fast_frozen_output_backward() -> None:
    from megatron.core.tensor_parallel.layers import LinearWithFrozenWeight

    if getattr(LinearWithFrozenWeight.backward, "__art_fast_output_backward__", False):
        return

    def _fast_backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        (weight,) = ctx.saved_tensors
        grad_input = _frozen_linear_grad_input(grad_output, weight)
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(  # ty: ignore[possibly-missing-attribute]
                grad_input,
                group=ctx.tp_group,
            )
        return grad_input, None, None, None, None

    setattr(_fast_backward, "__art_fast_output_backward__", True)
    LinearWithFrozenWeight.backward = staticmethod(_fast_backward)


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


def _compile_enabled() -> bool:
    disabled = _env_flag("ART_DISABLE_MEGATRON_COMPILE")
    return disabled is not True


def _install_gpt_preprocess_hook(model_chunks: ModelChunks) -> None:
    for chunk in model_chunks:
        module: Any = unwrap_megatron_chunk(chunk)
        while not isinstance(module, GPTModel) and hasattr(module, "module"):
            module = module.module
        if not isinstance(module, GPTModel):
            continue
        preprocess = module._preprocess

        def preprocess_hook(*args, _preprocess=preprocess, **kwargs):
            preproc_output = list(_preprocess(*args, **kwargs))
            preproc_output[0].requires_grad = True  # type: ignore[index]
            table = preproc_output[1]  # [S, B, 1, D]  # type: ignore[index]
            embedding_dim = table.size(-1)
            table_flat = table.view(table.size(0), embedding_dim)
            position_ids = kwargs["position_ids"]  # [B, S]
            batch_size, sequence_length = position_ids.shape
            gathered = table_flat.index_select(0, position_ids.reshape(-1))
            gathered = (
                gathered.view(batch_size, sequence_length, embedding_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
            preproc_output[1] = gathered.unsqueeze(2)  # [S, B, 1, D]
            return tuple(preproc_output)

        module._preprocess = preprocess_hook  # type: ignore[attr-defined]


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


def build_training_runtime(
    *,
    model_identifier: str | None = None,
    provider_torch_dtype: torch.dtype = torch.bfloat16,
    provider_configure: Callable[[Any], None] | None = None,
    optimizer_config: OptimizerConfig | None = None,
    moe_routing_replay_path: str | None = None,
    moe_routing_replay_bundle: MoeRoutingReplayBundle | None = None,
    moe_routing_replay_strict: bool = True,
    print_env: bool = True,
    build_optimizer: bool = True,
) -> TrainingRuntime:
    if random_state := os.environ.get("ART_MEGATRON_RANDOM_STATE"):
        seed = int(random_state)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    _install_fast_frozen_output_backward()
    provider = get_provider(
        model_identifier
        or os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER),
        torch_dtype=provider_torch_dtype,
    )
    if provider_configure is not None:
        provider_configure(provider)
    provider.register_pre_wrap_hook(freeze_model)
    provider.register_pre_wrap_hook(
        lambda chunks: apply_lora_adapters(chunks, provider)
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

    _install_gpt_preprocess_hook(model)
    if _compile_enabled():
        install_torch_compile_workarounds()
        for chunk in model:
            _compile_transformer_layers(chunk)

    optimizer_config = optimizer_config or _default_optimizer_config()
    optimizer = (
        _build_optimizer(
            model,
            optimizer_config,
        )
        if build_optimizer
        else None
    )

    runtime = TrainingRuntime(
        provider=provider,
        model=model,
        optimizer=optimizer,
        optimizer_config=optimizer_config,
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

        try:
            _run_megatron_job(runtime, job)
        finally:
            if after_job is not None:
                after_job()

        finalize_megatron_job(
            runtime,
            job_path=job_path,
            log_path=job.log_path,
            cleanup_path=_job_cleanup_path(job),
        )


def run_megatron_rl_job(
    runtime: TrainingRuntime,
    job: MegatronTrainingJob,
) -> None:
    packed_tensors = None
    adapter_model = None
    template = None
    zero_template = None

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
            step_result = run_training_step(
                model_chunks=runtime.model,
                optimizer=runtime.optimizer,
                learning_rate=job.config.learning_rate,
                inputs=micro_inputs,
                config=job.config,
                experimental_config=cast(dev.TrainConfig, job.experimental_config),
                ref_logprobs=None,
                step_index=step_index,
                sample_index=micro_indices,
                moe_routing_replay_controller=runtime.moe_routing_replay_controller,
            )
            print0(
                runtime.rank,
                "Correlation between old and new probabilities:",
                step_result.probs_corr,
            )

            if runtime.rank == 0:
                with open(job.log_path, "a+", encoding="utf-8") as log_file:
                    log_msg = json.dumps(
                        {
                            "loss": step_result.reduced_loss.item(),
                            "grad_norm": step_result.grad_norm,
                            "probs_corr": step_result.probs_corr,
                            TRAIN_GRADIENT_STEPS_KEY: num_steps,
                        }
                    )
                    print("Logging", log_msg)
                    log_file.write(log_msg + "\n")

        _save_lora_and_optimizer(
            runtime,
            adapter_model=adapter_model,
            lora_path=job.lora_path,
            optimizer_state_path=job.optimizer_state_path,
        )
    finally:
        if packed_tensors is not None:
            del packed_tensors
        if adapter_model is not None:
            del adapter_model
        if template is not None:
            del template
        if zero_template is not None:
            del zero_template
        if "micro_inputs" in locals():
            del micro_inputs
        gc.collect()
        torch.cuda.empty_cache()


def _flush_param_grads_to_main_grads(model_chunks: ModelChunks) -> None:
    """Fallback for direct SFT jobs when DDP post-hooks leave grads in param.grad.

    Megatron's distributed optimizer reads gradients from `main_grad`, which is
    normally populated by DDP backward post-hooks. Some direct ART runtimes can
    reach finalize/step with gradients still in `param.grad`, so copy them over
    using the same guard Megatron uses in its hook implementation.
    """
    for chunk in model_chunks:
        for param in chunk.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if not hasattr(param, "main_grad"):
                continue
            main_grad = cast(torch.Tensor, param.main_grad)
            if not getattr(param, "grad_added_to_main_grad", False) or getattr(
                param, "zero_out_wgrad", False
            ):
                main_grad.add_(param.grad.to(dtype=main_grad.dtype))
            param.grad = None


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
        gc.collect()
        torch.cuda.empty_cache()


def _load_megatron_job(job_path: str, *, supports_sft: bool) -> MegatronJob:
    with open(job_path, "rb") as handle:
        job_data = json.loads(handle.read())
    if job_data.get("job_type") == "sft":
        if not supports_sft:
            raise NotImplementedError("SFT jobs are not supported in this worker loop")
        return MegatronSFTTrainingJob.model_validate(job_data)
    return MegatronTrainingJob.model_validate(job_data)


def _run_megatron_job(runtime: TrainingRuntime, job: MegatronJob) -> None:
    if isinstance(job, MegatronSFTTrainingJob):
        run_megatron_sft_job(runtime, job)
        return
    run_megatron_rl_job(runtime, job)


def _job_cleanup_path(job: MegatronJob) -> str:
    if isinstance(job, MegatronSFTTrainingJob):
        return job.sft_data_dir
    return job.disk_packed_tensors["dir"]


def _load_lora_and_optimizer(
    runtime: TrainingRuntime,
    *,
    lora_path: str,
    optimizer_state_path: str,
) -> dict[str, torch.Tensor]:
    print0(runtime.rank, "Loading adapter model from", lora_path)
    adapter_model = load_lora_adapter_state_dict(lora_path)
    load_adapter_into_model(runtime.model, adapter_model)
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


def _save_lora_and_optimizer(
    runtime: TrainingRuntime,
    *,
    adapter_model: dict[str, torch.Tensor],
    lora_path: str,
    optimizer_state_path: str,
) -> None:
    assert runtime.optimizer is not None
    sharded_state_dict, sharded_state_manifest = collect_sharded_lora_state(
        runtime.model,
        adapter_model,
    )
    shard_path = os.path.join(
        lora_path,
        f"adapter_model-{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.safetensors",
    )
    manifest_path = os.path.join(
        lora_path,
        f"adapter_manifest-{runtime.rank + 1:02d}-of-{runtime.world_size:02d}.json",
    )
    print("Saving adapter shard to", shard_path)
    os.makedirs(lora_path, exist_ok=True)
    save_file(sharded_state_dict, shard_path)
    print("Saving adapter shard manifest to", manifest_path)
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(sharded_state_manifest, manifest_file, sort_keys=True)

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
    cleanup_path: str,
) -> None:
    torch.distributed.barrier()  # type: ignore[possibly-missing-attribute]
    if runtime.rank != 0:
        return

    if job_path is not None and os.path.exists(job_path):
        os.remove(job_path)
    if os.path.exists(cleanup_path):
        shutil.rmtree(cleanup_path)
    with open(log_path, "a+", encoding="utf-8") as log_file:
        log_file.write("all done\n")


def _placeholder_attention_mask(device: torch.device) -> torch.Tensor:
    return torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)


def _causal_attention_state(seq_len: int, device: torch.device) -> Any:
    group_ids = torch.zeros((1, seq_len), dtype=torch.int64, device=device)
    parent_ids = torch.zeros_like(group_ids)
    return create_shared_prefix_attention_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
    )


def _set_child_module(
    parent: torch.nn.Module,
    name: str,
    child: torch.nn.Module,
) -> None:
    if isinstance(parent, torch.nn.ModuleList | torch.nn.Sequential):
        parent[int(name)] = child
        return
    setattr(parent, name, child)


def _compile_transformer_layers(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, TransformerLayer):
            compiled_child = cast(torch.nn.Module, torch.compile(child))
            _set_child_module(parent=module, name=name, child=compiled_child)
            continue
        _compile_transformer_layers(child)


def iter_modules(model_chunks: ModelChunks) -> Any:
    for chunk in model_chunks:
        for module in chunk.modules():
            yield module


def load_adapter_into_model(
    model_chunks: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
    optimizer: Any | None = None,
) -> None:
    with torch.no_grad():
        for module in iter_modules(model_chunks):
            if hasattr(module, "load_lora"):
                module.load_lora(adapter_model)  # type: ignore[attr-defined]

    if optimizer is None:
        return
    optimizer.reload_model_params()


def collect_sharded_lora_state(
    model_chunks: ModelChunks,
    adapter_model: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    sharded_state_dict: dict[str, torch.Tensor] = {}
    sharded_state_manifest: dict[str, dict[str, Any]] = {}
    for module in iter_modules(model_chunks):
        if hasattr(module, "sharded_lora_state_dict"):
            module_sharded_lora_state_dict: dict[str, torch.Tensor] = (
                module.sharded_lora_state_dict()  # type: ignore[attr-defined]
            )
            for key, value in module_sharded_lora_state_dict.items():
                target_dtype = (
                    adapter_model[key].dtype if key in adapter_model else value.dtype
                )
                sharded_state_dict[key] = value.to(target_dtype).contiguous()
        if hasattr(module, "sharded_lora_manifest"):
            module_sharded_lora_manifest: dict[str, dict[str, Any]] = (
                module.sharded_lora_manifest()  # type: ignore[attr-defined]
            )
            sharded_state_manifest.update(module_sharded_lora_manifest)
    return sharded_state_dict, sharded_state_manifest


@torch.no_grad()
def select_indexed_inputs(packed_tensors: PackedTensors, index: int) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value[index : index + 1]
            for key, value in packed_tensors.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _clone_packed_tensors(inputs: PackedTensors) -> PackedTensors:
    return PackedTensors(  # type: ignore[call-arg]
        **{
            key: value.clone()
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        },
        pixel_values=[None],
        image_grid_thw=[None],
    )


@torch.no_grad()
def _zero_contribution_inputs(template: PackedTensors) -> PackedTensors:
    dummy = _clone_packed_tensors(template)
    dummy["assistant_mask"].zero_()
    return dummy


@torch.no_grad()
def _clone_sft_tensors(
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in inputs.items()}


@torch.no_grad()
def _zero_contribution_sft_inputs(
    template: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    dummy = _clone_sft_tensors(template)
    dummy["labels"].fill_(-100)
    return dummy


def resolve_global_grad_accumulation_sequences(
    global_grad_accumulation_sequences: int | None,
) -> int:
    dp_world_size = ps.get_data_parallel_world_size()
    if global_grad_accumulation_sequences is None:
        return dp_world_size
    return global_grad_accumulation_sequences


def resolve_local_grad_accumulation_sequences(
    global_grad_accumulation_sequences: int | None,
) -> int:
    resolved_global_grad_accumulation_sequences = (
        resolve_global_grad_accumulation_sequences(
            global_grad_accumulation_sequences=global_grad_accumulation_sequences
        )
    )
    dp_world_size = ps.get_data_parallel_world_size()
    if (
        resolved_global_grad_accumulation_sequences <= 0
        or resolved_global_grad_accumulation_sequences % dp_world_size != 0
    ):
        raise RuntimeError(
            "Invalid global grad accumulation / DP world size combination: "
            f"global_grad_accumulation_sequences={resolved_global_grad_accumulation_sequences}, "
            f"dp_world_size={dp_world_size}"
        )
    return resolved_global_grad_accumulation_sequences // dp_world_size


def build_micro_sample_indices(
    step_index: int,
    num_sequences: int,
    global_grad_accumulation_sequences: int | None,
) -> list[int | None]:
    dp_rank = ps.get_data_parallel_rank()
    resolved_global_grad_accumulation_sequences = (
        resolve_global_grad_accumulation_sequences(
            global_grad_accumulation_sequences=global_grad_accumulation_sequences
        )
    )
    dp_world_size = ps.get_data_parallel_world_size()
    local_grad_accumulation_sequences = resolve_local_grad_accumulation_sequences(
        global_grad_accumulation_sequences=resolved_global_grad_accumulation_sequences,
    )
    base_global_sample_index = step_index * resolved_global_grad_accumulation_sequences
    global_step_indices: list[int | None] = []
    for offset in range(resolved_global_grad_accumulation_sequences):
        global_sample_index = base_global_sample_index + offset
        global_step_indices.append(
            global_sample_index if global_sample_index < num_sequences else None
        )
    return [
        global_step_indices[offset * dp_world_size + dp_rank]
        for offset in range(local_grad_accumulation_sequences)
    ]


def select_micro_inputs(
    packed_tensors: PackedTensors,
    sample_indices: list[int | None],
    zero_template: PackedTensors,
) -> list[PackedTensors]:
    return [
        _clone_packed_tensors(zero_template)
        if sample_index is None
        else select_indexed_inputs(packed_tensors, sample_index)
        for sample_index in sample_indices
    ]


def select_sft_micro_inputs(
    trajectory_tensors: list[dict[str, torch.Tensor]],
    sample_indices: list[int | None],
    zero_template: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    return [
        _clone_sft_tensors(zero_template)
        if sample_index is None
        else _clone_sft_tensors(trajectory_tensors[sample_index])
        for sample_index in sample_indices
    ]


def _move_inputs_to_device(inputs: PackedTensors, device: torch.device) -> None:
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)  # type: ignore[index]


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


def _count_trainable_tokens(inputs: PackedTensors) -> float:
    assistant_mask = shift_tensor(inputs["assistant_mask"], False)
    return float(assistant_mask.sum().item())


def _local_trainable_token_count_tensor(
    micro_inputs: list[PackedTensors],
    device: torch.device,
) -> torch.Tensor:
    local_token_total = sum(_count_trainable_tokens(micro) for micro in micro_inputs)
    return torch.tensor([local_token_total], device=device, dtype=torch.float32)


def _count_sft_trainable_tokens(inputs: dict[str, torch.Tensor]) -> float:
    attention_mask = inputs["attention_mask"].reshape(-1)
    actual_len = int(attention_mask.sum().item())
    labels = inputs["labels"].reshape(-1)[:actual_len].unsqueeze(0)
    shifted_labels = shift_tensor(labels, -100)
    return float((shifted_labels != -100).sum().item())


def _local_trainable_sft_token_count_tensor(
    micro_inputs: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    local_token_total = sum(
        _count_sft_trainable_tokens(micro) for micro in micro_inputs
    )
    return torch.tensor([local_token_total], device=device, dtype=torch.float32)


def _prepare_sft_micro_inputs(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    attention_mask = inputs["attention_mask"].reshape(-1)
    actual_len = max(int(attention_mask.sum().item()), 1)
    input_ids = inputs["input_ids"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
    labels = inputs["labels"].reshape(-1)[:actual_len].unsqueeze(0).to(device)
    position_ids = torch.arange(actual_len, device=device).unsqueeze(0)
    shifted_labels = shift_tensor(labels, -100)
    mask = shifted_labels != -100
    return input_ids, position_ids, shifted_labels, mask, actual_len


def run_megatron_sft_step(
    *,
    model_chunks: ModelChunks,
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
        micro_sample_indices = sample_index
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

    device = next(model_chunks[0].parameters()).device

    for chunk in model_chunks:
        chunk.zero_grad_buffer()  # ty: ignore[call-non-callable]

    raw_loss_sum: torch.Tensor | None = None
    num_tokens = _local_trainable_sft_token_count_tensor(micro_inputs, device=device)

    for micro in micro_inputs:
        input_ids, position_ids, shifted_labels, mask, seq_len = (
            _prepare_sft_micro_inputs(micro, device)
        )
        per_token_loss: torch.Tensor = model_chunks[0](
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=_placeholder_attention_mask(device),
            labels=shifted_labels,
            extra_block_kwargs={
                "attention_bias": _causal_attention_state(seq_len, device),
            },
        )
        masked_loss = per_token_loss[mask].sum()
        masked_loss.backward()
        detached_micro_loss = masked_loss.detach()
        if raw_loss_sum is None:
            raw_loss_sum = detached_micro_loss
        else:
            raw_loss_sum = raw_loss_sum + detached_micro_loss

    if raw_loss_sum is None:
        raise RuntimeError("run_megatron_sft_step did not produce outputs")

    _flush_param_grads_to_main_grads(model_chunks)
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
    optimizer: Any,
    learning_rate: float,
    inputs: PackedTensors | list[PackedTensors],
    config: types.TrainConfig,
    experimental_config: dev.TrainConfig,
    step_index: int,
    sample_index: int | list[int | None],
    ref_logprobs: torch.Tensor | None = None,
    moe_routing_replay_controller: MoeRoutingReplayController | None = None,
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
        micro_sample_indices = sample_index
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

    for chunk in model_chunks:
        chunk.zero_grad_buffer()  # ty: ignore[call-non-callable]

    micro_count = len(micro_inputs)
    raw_loss_sum: torch.Tensor | None = None
    token_count = _local_trainable_token_count_tensor(micro_inputs, device=device)
    probs_corr_sum = 0.0
    new_logprobs_list: list[torch.Tensor] = []

    for micro in micro_inputs:
        _move_inputs_to_device(micro, device)
        attention_state = create_shared_prefix_attention_state(
            group_ids=micro["group_ids"],
            parent_ids=micro["parent_ids"],
        )
        attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=device)

        new_logprobs = -model_chunks[0](
            input_ids=micro["tokens"],
            position_ids=micro["input_pos"],
            attention_mask=attention_mask,
            labels=shift_tensor(micro["tokens"], 0),
            extra_block_kwargs={"attention_bias": attention_state},
        )

        loss_info = loss_fn(
            micro,  # ty: ignore[invalid-argument-type]
            new_logprobs,
            ref_logprobs,
            None,
            experimental_config,
            reduction="sum",
        )
        micro_loss = loss_info.policy_loss
        micro_loss.backward()
        probs_corr_sum += float(loss_info.probs_corr.item())
        detached_micro_loss = micro_loss.detach()
        if raw_loss_sum is None:
            raw_loss_sum = detached_micro_loss
        else:
            raw_loss_sum = raw_loss_sum + detached_micro_loss
        del loss_info
        del micro_loss
        del attention_mask
        del attention_state
        new_logprobs_list.append(
            new_logprobs.detach().to(device="cpu", non_blocking=True)
        )
        del new_logprobs

    if raw_loss_sum is None:
        raise RuntimeError("run_training_step did not produce outputs")

    torch.cuda.empty_cache()
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
        probs_corr=probs_corr_sum / micro_count,
        new_logprobs=new_logprobs_list,
        update_successful=update_successful,
        grad_norm=grad_norm,
        num_zeros_in_grad=num_zeros_in_grad,
    )


def _run_service_loop(runtime: TrainingRuntime) -> None:
    offload_state = OffloadState()
    wake_lock_path = os.environ.get(
        "ART_MEGATRON_WAKE_LOCK_PATH", DEFAULT_VLLM_WAKE_LOCK_PATH
    )

    def wait_until_ready() -> None:
        while os.path.exists(wake_lock_path):
            time.sleep(0.2)

    def before_job() -> None:
        reload_to_gpu(runtime.model, runtime.rank, offload_state)

    def after_job() -> None:
        runtime.optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        offload_to_cpu(runtime.model, runtime.rank, offload_state)

    after_job()
    run_megatron_worker_loop(
        runtime,
        supports_sft=True,
        wait_until_ready=wait_until_ready,
        before_job=before_job,
        after_job=after_job,
    )


def main() -> None:
    runtime = build_training_runtime(
        model_identifier=os.environ.get("MODEL_IDENTIFIER", DEFAULT_MODEL_IDENTIFIER),
        build_optimizer=False,
    )
    _run_service_loop(runtime)


if __name__ == "__main__":
    main()
