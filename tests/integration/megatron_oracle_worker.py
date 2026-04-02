from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
from types import MethodType
from typing import Any, Callable

import numpy as np
import torch

from art.megatron.routing_replay import (
    ParallelTopology as ReplayParallelTopology,
)
from art.megatron.routing_replay import (
    build_bundle_from_forward_trace_dir,
)

from .megatron_forward_trace import ForwardTraceCapture
from .megatron_oracle_harness import (
    SUPPORTED_SENSITIVITY_MUTATIONS,
    OracleCaseConfig,
    RunManifest,
    SensitivityMutation,
    StepTrace,
    Topology,
    WorkerRunRequest,
    _read_json,
    _require_not_none,
    _write_json,
)


def run_worker_subprocess(
    request: WorkerRunRequest,
    topology_dir: Path,
    *,
    repo_root: Path,
) -> None:
    """Runs one distributed worker subprocess and stores combined logs."""
    request_path = topology_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))
    worker_module = "integration.megatron_oracle_worker"
    worker_cwd = repo_root / "tests"

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(request.topology.world_size()),
        "-m",
        worker_module,
        "--worker-run",
        "--run-request",
        str(request_path),
    ]
    run = subprocess.run(
        command,
        cwd=str(worker_cwd),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = f"{run.stdout}\n{run.stderr}".strip()
    (topology_dir / "worker.log").write_text(combined_output + "\n", encoding="utf-8")
    if run.returncode != 0:
        tail = "\n".join(combined_output.splitlines()[-80:])
        raise RuntimeError(
            f"Topology run failed for {request.topology.slug()} with exit code "
            f"{run.returncode}.\n{tail}"
        )


def _set_deterministic_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _merge_sharded_dicts(shards_by_rank: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges rank-sharded LoRA tensors into a full state dict on rank 0."""
    import torch

    merged: dict[str, list[Any]] = {}
    for rank_shards in shards_by_rank:
        for key, tensor in rank_shards.items():
            merged.setdefault(key, []).append(tensor.detach().cpu())
    full_state: dict[str, Any] = {}
    for key, shards in merged.items():
        if len(shards) == 1:
            full_state[key] = shards[0].contiguous()
            continue
        concat_dim = 1 if ".lora_A." in key else 0
        full_state[key] = torch.cat(shards, dim=concat_dim).contiguous()
    return full_state


def _gather_full_state(
    local_state: dict[str, Any],
) -> dict[str, Any] | None:
    """Gathers local state dicts to rank 0 and merges them."""
    import torch

    rank = torch.distributed.get_rank()  # ty: ignore[possibly-missing-attribute]
    world_size = torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(  # ty: ignore[possibly-missing-attribute]
        local_state, gathered, dst=0
    )
    if rank != 0:
        return None
    assert gathered is not None
    entries = [entry for entry in gathered if entry is not None]
    return _merge_sharded_dicts(entries)


def _collect_lora_state(
    model_chunks: list[Any],
) -> dict[str, Any] | None:
    """Collects full LoRA adapter state for validation and delta computation."""
    local_state: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if not hasattr(module, "sharded_lora_state_dict"):
                continue
            module_state = module.sharded_lora_state_dict()
            for key, value in module_state.items():
                if key in local_state:
                    raise RuntimeError(
                        f"Duplicate LoRA key while collecting state: {key}"
                    )
                local_state[key] = value.detach().cpu()
    return _gather_full_state(local_state)


def _collect_lora_grads(
    model_chunks: list[Any],
) -> dict[str, Any] | None:
    """Collects full LoRA gradient tensors across all ranks."""
    from art.megatron.lora import LoRA

    local_grads: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            for key, param, expert in module._export_items():  # type: ignore[attr-defined]
                if not hasattr(param, "main_grad"):
                    raise RuntimeError(
                        f"LoRA param missing main_grad attribute for key '{key}'"
                    )
                grad = param.main_grad
                if grad is None:
                    raise RuntimeError(f"LoRA param main_grad is None for key '{key}'")
                if hasattr(grad, "_local_tensor"):
                    grad = grad._local_tensor
                captured_grad = grad[expert] if expert is not None else grad
                local_grads[key] = captured_grad.detach().cpu().T
    return _gather_full_state(local_grads)


def _apply_save_mutation_to_tensor_map(
    tensor_map: dict[str, Any],
    *,
    mutation: SensitivityMutation | None,
) -> dict[str, Any]:
    """Applies save-only mutation transforms to already-collected full tensor maps."""
    if mutation == "save_drop_nonzero_ranked_tp_shards":
        mutated: dict[str, Any] = {}
        for key, value in tensor_map.items():
            if not isinstance(value, torch.Tensor):
                mutated[key] = value
                continue
            if ".lora_A." in key and value.ndim >= 2 and value.shape[1] > 1:
                keep = max(1, value.shape[1] // 2)
                mutated[key] = value.narrow(1, 0, keep).contiguous()
                continue
            if ".lora_B." in key and value.ndim >= 2 and value.shape[0] > 1:
                keep = max(1, value.shape[0] // 2)
                mutated[key] = value.narrow(0, 0, keep).contiguous()
                continue
            mutated[key] = value
        return mutated

    if mutation == "save_duplicate_replicated_entries":
        mutated = dict(tensor_map)
        source_by_bucket: dict[tuple[tuple[int, ...], str], torch.Tensor] = {}
        for key in sorted(mutated.keys()):
            value = mutated[key]
            if not isinstance(value, torch.Tensor):
                continue
            if not key.endswith(".weight"):
                continue
            bucket = (tuple(value.shape), str(value.dtype))
            source = source_by_bucket.get(bucket)
            if source is None:
                source_by_bucket[bucket] = value
                continue
            mutated[key] = source.clone().contiguous()
        return mutated

    return tensor_map


def _validate_loaded_state_matches_adapter(
    loaded_state: dict[str, Any],
    adapter_model: dict[str, Any],
) -> None:
    """Checks loaded model LoRA state exactly matches adapter tensors and keys."""
    import torch

    for key in sorted(adapter_model.keys()):
        assert torch.equal(loaded_state[key].cpu(), adapter_model[key].cpu()), (
            f"Loaded LoRA state mismatch for key '{key}'"
        )


def _build_deterministic_shared_init(
    initial_state: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Builds deterministic nonzero LoRA init values for both A and B tensors."""
    initialized: dict[str, Any] = {}
    for key in sorted(initial_state.keys()):
        value = initial_state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor value for key '{key}', got {type(value)}")
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        key_seed = int.from_bytes(digest[:8], "little") % (2**31)
        generator = torch.Generator(device="cpu").manual_seed(key_seed)
        random_values = torch.randn(
            value.shape,
            generator=generator,
            dtype=torch.float32,
        )
        initialized[key] = (0.01 * random_values).to(dtype=value.dtype).contiguous()
    return initialized


def _configure_provider(
    provider: Any,
    topology: Topology,
    case_config: OracleCaseConfig,
) -> None:
    """Applies deterministic topology/model overrides to provider config."""
    provider.tensor_model_parallel_size = topology.tp
    provider.expert_model_parallel_size = topology.ep
    provider.expert_tensor_parallel_size = topology.etp
    # These are intentionally pinned to 1 for now
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = topology.sp
    provider.num_layers = case_config.num_layers
    if case_config.precision == "fp32":
        provider.bf16 = False
        provider.fp16 = False
        provider.params_dtype = torch.float32
        provider.pipeline_dtype = torch.float32
        provider.enable_autocast = False
        provider.autocast_dtype = None
        provider.attention_softmax_in_fp32 = True
        provider.fp32_residual_connection = True
    if hasattr(provider, "attention_dropout"):
        provider.attention_dropout = 0.0
    if hasattr(provider, "hidden_dropout"):
        provider.hidden_dropout = 0.0


def _build_optimizer_config(case_config: OracleCaseConfig):
    """Builds Megatron optimizer settings for deterministic harness runs."""
    from megatron.core.optimizer import OptimizerConfig

    if case_config.precision == "fp32":
        return OptimizerConfig(
            bf16=False,
            fp16=False,
            params_dtype=torch.float32,
            main_grads_dtype=torch.float32,
            main_params_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            lr=case_config.learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            clip_grad=0.1,
            weight_decay=0.0,
            adam_eps=1e-13,
        )

    return OptimizerConfig(
        bf16=True,
        fp16=False,
        lr=case_config.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        clip_grad=0.1,
        weight_decay=0.0,
        adam_eps=1e-13,
    )


def _configure_cuda_precision(case_config: OracleCaseConfig) -> None:
    if case_config.precision != "fp32":
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _assert_runtime_configuration(
    model_chunks: list[Any],
    case_config: OracleCaseConfig,
) -> None:
    """Validates runtime model depth equals requested oracle case config."""
    observed_num_layers: set[int] = set()

    for chunk in model_chunks:
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "num_layers"):
            observed_num_layers.add(int(config.num_layers))

    if observed_num_layers != {case_config.num_layers}:
        raise RuntimeError(
            "Runtime num_layers mismatch: "
            f"requested={case_config.num_layers}, observed={sorted(observed_num_layers)}"
        )


def _delta_state(
    initial_state: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    """Computes LoRA parameter deltas while enforcing stable key sets."""
    initial_keys = set(initial_state.keys())
    current_keys = set(current_state.keys())
    if initial_keys != current_keys:
        missing = sorted(initial_keys - current_keys)
        extra = sorted(current_keys - initial_keys)
        raise KeyError(
            f"LoRA state keys changed during training: missing={missing[:3]} extra={extra[:3]}"
        )
    return {
        key: current_state[key].detach().cpu() - initial_state[key].detach().cpu()
        for key in sorted(initial_keys)
    }


def _iter_named_unique_parameters(
    model_chunks: list[Any],
) -> list[tuple[str, torch.nn.Parameter]]:
    seen: set[int] = set()
    params: list[tuple[str, torch.nn.Parameter]] = []
    for chunk_index, chunk in enumerate(model_chunks):
        for name, param in chunk.named_parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append((f"chunk{chunk_index}.{name}", param))
    return params


def _matches_grad_sync_skip_mutation(
    param_name: str, mutation: SensitivityMutation
) -> bool:
    if mutation == "bwd_skip_sync_qkv_a":
        return any(
            token in param_name
            for token in (
                ".self_attention.linear_qkv.q_proj_lora.A_T",
                ".self_attention.linear_qkv.k_proj_lora.A_T",
                ".self_attention.linear_qkv.v_proj_lora.A_T",
            )
        )
    if mutation == "bwd_skip_sync_o_proj_b":
        return ".self_attention.linear_proj.lora.B_T" in param_name
    if mutation == "bwd_skip_sync_fc1_a":
        return (
            ".mlp.experts.linear_fc1.gate_lora.A_T" in param_name
            or ".mlp.experts.linear_fc1.up_lora.A_T" in param_name
        )
    return False


@contextmanager
def _apply_grad_sync_skip_mutation(
    model_chunks: list[Any],
    mutation: SensitivityMutation | None,
):
    if mutation not in {
        "bwd_skip_sync_qkv_a",
        "bwd_skip_sync_o_proj_b",
        "bwd_skip_sync_fc1_a",
    }:
        yield
        return

    saved_attrs: list[tuple[Any, str, Any]] = []
    for param_name, param in _iter_named_unique_parameters(model_chunks):
        # this only passes lora params atm, so we assume lora params below
        if not _matches_grad_sync_skip_mutation(param_name, mutation):
            continue
        if (
            mutation == "bwd_skip_sync_fc1_a" and param.grad_sync_domain != "expert_tp"  # ty: ignore[unresolved-attribute]
        ):
            continue

        # For fc1 A params, extended finalize handles expert-TP sync via grad_sync_op.
        saved_attrs.append((param, "grad_sync_op", param.grad_sync_op))  # ty: ignore[unresolved-attribute]
        param.grad_sync_op = "none"  # ty: ignore[unresolved-attribute]

        # Megatron native TP finalize uses this only for tp_default-domain params.
        average_gradients_across_tp_domain = param.average_gradients_across_tp_domain  # ty: ignore[unresolved-attribute]
        grad_sync_domain = param.grad_sync_domain  # ty: ignore[unresolved-attribute]
        if average_gradients_across_tp_domain and grad_sync_domain == "tp_default":
            saved_attrs.append(
                (
                    param,
                    "average_gradients_across_tp_domain",
                    average_gradients_across_tp_domain,
                )
            )
            param.average_gradients_across_tp_domain = False  # ty: ignore[unresolved-attribute]
    try:
        yield
    finally:
        for param, attr, value in reversed(saved_attrs):
            setattr(param, attr, value)


@contextmanager
def _apply_o_proj_forward_mutation(
    model_chunks: list[Any],
    mutation: SensitivityMutation | None,
):
    if mutation not in {
        "fwd_skip_o_proj_tp_reduce",
        "fwd_o_proj_tp_reduce_avg_not_sum",
    }:
        yield
        return

    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    from art.megatron.lora import SelfAttentionLinearProjLoRA

    original_forwards: list[tuple[Any, Any]] = []
    for chunk in model_chunks:
        for module in chunk.modules():
            if not isinstance(module, SelfAttentionLinearProjLoRA):
                continue
            original_forwards.append((module, module.forward))

            def _mutated_forward(self: Any, x: Any):
                base_output, bias_output = self.linear_proj(x)
                lora_output = self.lora(x)
                tp_size = self.provider.tensor_model_parallel_size
                if tp_size > 1:
                    if mutation == "fwd_o_proj_tp_reduce_avg_not_sum":
                        if self.provider.sequence_parallel:
                            lora_output = reduce_scatter_to_sequence_parallel_region(
                                lora_output
                            )
                        else:
                            lora_output = reduce_from_tensor_model_parallel_region(
                                lora_output
                            )
                        lora_output = lora_output / tp_size
                    elif mutation == "fwd_skip_o_proj_tp_reduce":
                        if self.provider.sequence_parallel:
                            seq_per_rank = lora_output.shape[0] // tp_size
                            tp_rank = ps.get_tensor_model_parallel_rank()
                            lora_output = lora_output.narrow(
                                0, tp_rank * seq_per_rank, seq_per_rank
                            )
                return base_output + lora_output, bias_output

            module.forward = MethodType(_mutated_forward, module)

    try:
        yield
    finally:
        for module, original_forward in reversed(original_forwards):
            module.forward = original_forward


@contextmanager
def _patch_lora_for_fp32(
    model_chunks: list[Any],
    optimizer: Any,
):
    """
    torch grouped_gemm is bf16 only, so we have a simple custom fp32 path
    to make the numbers match closely
    """
    from art.megatron.lora import LoRA, MLPExpertsLinearFC1LoRA

    del model_chunks
    del optimizer
    original_forward = LoRA.forward
    original_fc1_forward = MLPExpertsLinearFC1LoRA.forward

    def _reference_forward(
        self: Any,
        x: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        work_dtype = (
            torch.float32
            if torch.is_floating_point(x) and x.dtype != torch.float32
            else x.dtype
        )
        work_x = x.to(dtype=work_dtype)
        work_a = self.A_T.to(dtype=work_dtype)
        work_b = self.B_T.to(dtype=work_dtype)

        if tokens_per_expert is None or self.num_local_experts == 1:
            return (((work_x @ work_a) @ work_b) * self.scale).to(dtype=x.dtype)

        counts = (
            tokens_per_expert.tolist()
            if isinstance(tokens_per_expert, torch.Tensor)
            else list(tokens_per_expert)
        )
        out = work_x.new_zeros((work_x.shape[0], work_b.shape[-1]))

        cursor = 0
        for expert_index, count in enumerate(counts):
            count_int = int(count)
            if count_int <= 0:
                continue
            next_cursor = cursor + count_int
            x_chunk = work_x[cursor:next_cursor]
            out[cursor:next_cursor] = (x_chunk @ work_a[expert_index]) @ work_b[
                expert_index
            ]
            cursor = next_cursor

        if cursor != int(work_x.shape[0]):
            raise RuntimeError(
                "Expert LoRA reference path did not consume all grouped rows: "
                f"consumed={cursor}, rows={int(work_x.shape[0])}"
            )

        return (out * self.scale).to(dtype=x.dtype)

    def _reference_fc1_forward(self: Any, x: torch.Tensor, tokens_per_expert: Any):
        base_out, bias_out = self.linear_fc1(x, tokens_per_expert)
        adapter_out = torch.cat(
            (
                self.gate_lora(x, tokens_per_expert),
                self.up_lora(x, tokens_per_expert),
            ),
            dim=1,
        )
        return base_out + adapter_out, bias_out

    LoRA.forward = _reference_forward  # ty: ignore[invalid-assignment]
    MLPExpertsLinearFC1LoRA.forward = _reference_fc1_forward  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        LoRA.forward = original_forward
        MLPExpertsLinearFC1LoRA.forward = original_fc1_forward


@contextmanager
def _mutation_hook(
    megatron_train_module: Any,
    model_chunks: list[Any],
    mutation: SensitivityMutation | None,
    topology: Topology,
    pre_optimizer_step_hook: Callable[[], None] | None = None,
    loss_scale: float = 1.0,
):
    """Applies optional sensitivity mutation hooks around training steps."""
    original_finalize = megatron_train_module.finalize_model_grads_extended
    original_optimizer_step = megatron_train_module._optimizer_step
    original_loss_fn = megatron_train_module.loss_fn
    original_local_token_count_tensor = (
        megatron_train_module._local_trainable_token_count_tensor
    )
    original_build_micro_sample_indices = (
        megatron_train_module.build_micro_sample_indices
    )

    known_mutations = {None, *SUPPORTED_SENSITIVITY_MUTATIONS}
    if mutation not in known_mutations:
        raise ValueError(f"Unsupported mutation: {mutation}")

    if mutation == "skip_finalize":
        megatron_train_module.finalize_model_grads_extended = (
            lambda _model, **_kwargs: (None)
        )

    if mutation == "dp_local_token_normalization":

        def _wrong_local_trainable_token_count_tensor(
            micro_inputs: list[Any],
            device: torch.device,
        ) -> torch.Tensor:
            local_token_total = sum(
                megatron_train_module._count_trainable_tokens(micro)
                for micro in micro_inputs
            )
            dp_world_size = int(
                megatron_train_module.ps.get_data_parallel_world_size(
                    with_context_parallel=True
                )
            )
            wrong_local_token_total = local_token_total / max(dp_world_size, 1)
            return torch.tensor(
                [wrong_local_token_total],
                device=device,
                dtype=torch.float32,
            )

        megatron_train_module._local_trainable_token_count_tensor = (
            _wrong_local_trainable_token_count_tensor
        )

    if mutation == "dp_grad_accumulation_seqs":

        def _wrong_build_micro_sample_indices(
            *,
            step_index: int,
            num_sequences: int,
            global_grad_accumulation_sequences: int,
        ) -> list[int | None]:
            base_global_sample_index = step_index * global_grad_accumulation_sequences
            return [
                (global_sample_index if global_sample_index < num_sequences else None)
                for global_sample_index in range(
                    base_global_sample_index,
                    base_global_sample_index + global_grad_accumulation_sequences,
                )
            ]

        megatron_train_module.build_micro_sample_indices = (
            _wrong_build_micro_sample_indices
        )

    if pre_optimizer_step_hook is not None:

        def _patched_optimizer_step(optimizer: Any, learning_rate: float):
            if pre_optimizer_step_hook is not None:
                pre_optimizer_step_hook()
            return original_optimizer_step(optimizer, learning_rate)

        megatron_train_module._optimizer_step = _patched_optimizer_step

    effective_loss_scale = loss_scale
    if effective_loss_scale <= 0:
        raise ValueError(
            f"effective_loss_scale must be > 0, got {effective_loss_scale}"
        )
    if effective_loss_scale != 1.0:

        def _scaled_loss_fn(*args: Any, **kwargs: Any):
            loss = original_loss_fn(*args, **kwargs)
            return loss.model_copy(
                update={
                    "policy_loss": loss.policy_loss * effective_loss_scale,
                    "policy_loss_sum": loss.policy_loss_sum * effective_loss_scale,
                }
            )

        megatron_train_module.loss_fn = _scaled_loss_fn

    if mutation is None:
        if pre_optimizer_step_hook is None and effective_loss_scale == 1.0:
            yield
            return
    with ExitStack() as stack:
        stack.enter_context(_apply_o_proj_forward_mutation(model_chunks, mutation))
        stack.enter_context(_apply_grad_sync_skip_mutation(model_chunks, mutation))
        try:
            yield
        finally:
            megatron_train_module.finalize_model_grads_extended = original_finalize
            megatron_train_module._optimizer_step = original_optimizer_step
            megatron_train_module.loss_fn = original_loss_fn
            megatron_train_module._local_trainable_token_count_tensor = (
                original_local_token_count_tensor
            )
            megatron_train_module.build_micro_sample_indices = (
                original_build_micro_sample_indices
            )


def _worker_run(request: WorkerRunRequest) -> None:
    """Executes one full distributed training trace generation worker run."""
    from safetensors.torch import load_file, save_file
    import torch

    from art import dev, types
    from art.megatron import train as megatron_train
    from art.preprocessing.pack import packed_tensors_from_dir

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")  # ty: ignore[possibly-missing-attribute]
    _set_deterministic_seed(request.case_config.seed)
    _configure_cuda_precision(request.case_config)

    runtime = megatron_train.build_training_runtime(
        model_identifier=request.case_config.base_model,
        provider_torch_dtype=(
            torch.float32 if request.case_config.precision == "fp32" else torch.bfloat16
        ),
        provider_configure=lambda provider: _configure_provider(
            provider, request.topology, request.case_config
        ),
        optimizer_config=_build_optimizer_config(request.case_config),
        print_env=False,
        print_optimizer_stats=False,
    )
    model_chunks = runtime.model
    optimizer = runtime.optimizer
    megatron_train.configure_moe_routing_replay(
        runtime,
        replay_bundle_path=request.moe_routing_replay_path,
        strict=request.moe_routing_replay_strict,
    )
    _assert_runtime_configuration(model_chunks, request.case_config)

    topology_dir = Path(request.topology_dir)
    traces_dir = topology_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # setup the shared initial lora
    shared_init_path = Path(request.shared_init_adapter_path)
    if not shared_init_path.exists():
        initial_state = _collect_lora_state(model_chunks)
        if torch.distributed.get_rank() == 0:  # ty: ignore[possibly-missing-attribute]
            shared_init_path.parent.mkdir(parents=True, exist_ok=True)
            deterministic_init = _build_deterministic_shared_init(
                _require_not_none(initial_state, "initial_state"),
                seed=request.case_config.seed,
            )
            save_file(
                deterministic_init,
                str(shared_init_path),
            )
    torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]

    # load the shared initial lora into the model and validate we can collect it from the model
    adapter_model = load_file(str(shared_init_path))
    megatron_train.load_adapter_into_model(model_chunks, adapter_model, optimizer)
    loaded_state = _collect_lora_state(model_chunks)
    if torch.distributed.get_rank() == 0:  # ty: ignore[possibly-missing-attribute]
        _validate_loaded_state_matches_adapter(
            _require_not_none(loaded_state, "loaded_state"), adapter_model
        )
    torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]

    # load the inputs
    packed_tensors = packed_tensors_from_dir(
        **request.packed_tensors.model_dump(exclude_none=True)
    )
    template = megatron_train.select_indexed_inputs(packed_tensors, 0)
    zero_template = megatron_train._zero_contribution_inputs(template)
    initial_lora_state = loaded_state
    global_grad_accumulation_sequences = request.case_config.grad_accumulation_sequences

    train_config = types.TrainConfig(
        learning_rate=request.case_config.learning_rate,
        kl_penalty_coef=0.0,
        grad_accumulation_sequences=global_grad_accumulation_sequences,
    )
    experimental_config: dev.TrainConfig = {}
    step_traces: list[StepTrace] = []
    captured_grads: dict[str, Any] | None = None
    routing_replay_controller = runtime.moe_routing_replay_controller
    micro_start_callback = (
        routing_replay_controller.begin_micro
        if routing_replay_controller is not None
        else None
    )
    forward_trace_capture = ForwardTraceCapture(
        model_chunks,
        enabled=True,
        micro_start_callback=micro_start_callback,
    )

    def _capture_lora_grads() -> None:
        nonlocal captured_grads
        captured_grads = _collect_lora_grads(model_chunks)

    with (
        _mutation_hook(
            megatron_train,
            model_chunks,
            request.mutation,
            request.topology,
            pre_optimizer_step_hook=_capture_lora_grads,
            loss_scale=request.case_config.loss_scale,
        ),
        _patch_lora_for_fp32(model_chunks, optimizer),
    ):
        for step_index in range(request.case_config.num_steps):
            micro_sample_indices = megatron_train.build_micro_sample_indices(
                step_index=step_index,
                num_sequences=request.packed_tensors.num_sequences,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            )
            forward_trace_capture.set_step(step_index, micro_sample_indices)
            micro_inputs = megatron_train.select_micro_inputs(
                packed_tensors, micro_sample_indices, zero_template
            )
            captured_grads = None

            step_result = megatron_train.run_training_step(
                model_chunks=model_chunks,
                optimizer=optimizer,
                learning_rate=train_config.learning_rate,
                inputs=micro_inputs,
                config=train_config,
                experimental_config=experimental_config,
                ref_logprobs=None,
                step_index=step_index,
                sample_index=micro_sample_indices,
                moe_routing_replay_controller=runtime.moe_routing_replay_controller,
            )
            ordered_micro_outputs = forward_trace_capture.ordered_step_outputs()
            forward_trace_capture.save_current_step(traces_dir)
            torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
            current_lora_state = _collect_lora_state(model_chunks)

            if torch.distributed.get_rank() == 0:  # ty: ignore[possibly-missing-attribute]
                grads = _require_not_none(captured_grads, "captured_grads")
                initial_state = _require_not_none(
                    initial_lora_state, "initial_lora_state"
                )
                current_state = _require_not_none(
                    current_lora_state, "current_lora_state"
                )
                deltas = _delta_state(initial_state, current_state)
                saved_deltas = _apply_save_mutation_to_tensor_map(
                    deltas,
                    mutation=request.mutation,
                )
                saved_current_state = _apply_save_mutation_to_tensor_map(
                    current_state,
                    mutation=request.mutation,
                )

                output_rel = Path("traces") / f"output_step_{step_index:03d}.pt"
                grads_rel = Path("traces") / f"grads_step_{step_index:03d}.safetensors"
                deltas_rel = (
                    Path("traces") / f"deltas_step_{step_index:03d}.safetensors"
                )
                lora_rel = Path(f"lora_step_{step_index:03d}.safetensors")
                ordered_outputs = _require_not_none(
                    ordered_micro_outputs, "ordered_micro_outputs"
                )
                if not ordered_outputs:
                    raise RuntimeError("Expected at least one captured micro output")

                torch.save(
                    torch.stack(ordered_outputs, dim=0),
                    topology_dir / output_rel,
                )
                save_file(grads, str(topology_dir / grads_rel))
                save_file(saved_deltas, str(topology_dir / deltas_rel))
                save_file(saved_current_state, str(topology_dir / lora_rel))

                step_traces.append(
                    StepTrace(
                        step_index=step_index,
                        loss=float(
                            step_result.reduced_loss.item()
                            / request.case_config.loss_scale
                        ),
                        probs_corr=step_result.probs_corr,
                        output_file=str(output_rel),
                        grads_file=str(grads_rel),
                        deltas_file=str(deltas_rel),
                        lora_file=str(lora_rel),
                    )
                )
            torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]

    forward_trace_capture.close()

    if torch.distributed.get_rank() == 0:  # ty: ignore[possibly-missing-attribute]
        # build and save the moe routing replay bundle
        if request.capture_moe_routing_bundle_path is not None:
            replay_bundle = build_bundle_from_forward_trace_dir(
                traces_dir=traces_dir,
                num_steps=request.case_config.num_steps,
                topology=ReplayParallelTopology.model_validate(
                    request.topology.model_dump(
                        include={"tp", "ep", "etp", "dp", "sp", "cp", "pp", "vpp"},
                        mode="python",
                    )
                ),
            )
            replay_bundle.to_dir(request.capture_moe_routing_bundle_path)

        # build and save the run manifest
        manifest = RunManifest(
            case_id=request.case_id,
            base_model=request.case_config.base_model,
            num_layers=request.case_config.num_layers,
            topology=request.topology.slug(),
            world_size=request.topology.world_size(),
            seed=request.case_config.seed,
            num_steps=request.case_config.num_steps,
            packed_tensors=request.packed_tensors,
            steps=step_traces,
        )
        _write_json(topology_dir / "manifest.json", manifest.model_dump(mode="json"))
    torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
    torch.distributed.destroy_process_group()  # ty: ignore[possibly-missing-attribute]


def run_worker_cli(run_request_path: Path) -> None:
    """Loads a worker request and dispatches worker execution."""
    request = WorkerRunRequest.model_validate(_read_json(run_request_path))
    _worker_run(request)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parses worker CLI arguments."""
    parser = argparse.ArgumentParser(description="Megatron oracle harness worker")
    parser.add_argument("--worker-run", action="store_true")
    parser.add_argument("--run-request", type=Path)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    """CLI entry for worker-only execution mode."""
    args = _parse_args(argv)
    if not args.worker_run:
        raise SystemExit("This module is intended for test imports or --worker-run")
    if args.run_request is None:
        raise SystemExit("--run-request is required with --worker-run")
    run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
