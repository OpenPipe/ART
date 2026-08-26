from __future__ import annotations

import argparse
import atexit
from collections import deque
from contextlib import ExitStack, contextmanager
import faulthandler
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import MethodType
from typing import Any, Callable, cast

import numpy as np
import torch

from art.megatron.routing_replay import (
    ROUTER_NAME_TOKEN,
)
from art.megatron.routing_replay import (
    ParallelTopology as ReplayParallelTopology,
)
from art.preprocessing.pack import PackedTensors
from art.utils.lifecycle import terminate_popen_process_group

from ..routing_replay.bundle import build_bundle_from_forward_trace_dir
from ..routing_replay.trace import install_moe_routing_trace_hooks
from .forward_trace import CAPTURE_NAME_TOKENS, ForwardTraceCapture
from .fp32_grouped_gemm import (
    allow_fp32_grouped_gemm_fallback_for_model_support_tests,
)
from .gdn_fp32_reference import install_megatron_qwen35_gdn_fp32_reference
from .gdn_trace_uids import install_gdn_trace_token_uid_hooks
from .oracle_harness import (
    SUPPORTED_SENSITIVITY_MUTATIONS,
    OracleCaseConfig,
    RunManifest,
    SensitivityMutation,
    StepTrace,
    Topology,
    WorkerRunRequest,
    _comparison_dir,
    _read_json,
    _remove_comparison_dir,
    _require_not_none,
    _sample_valid_lengths,
    _trim_trace_padding,
    _write_comparison_sink,
    _write_json,
)
from .test_inputs import build_sft_trajectory_tensors_from_packed_tensors

_TOPOLOGY_ENV_VARS = {
    "tp": "ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE",
    "cp": "ART_MEGATRON_CONTEXT_PARALLEL_SIZE",
    "ep": "ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE",
    "etp": "ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE",
    "pp": "ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE",
    "vpp": "ART_MEGATRON_VIRTUAL_PIPELINE_MODEL_PARALLEL_SIZE",
}
_ORACLE_DEBUG_ENV = "ART_ORACLE_DEBUG"
_ATTACH_TOKEN_UIDS_ENV = "ART_MEGATRON_ATTACH_TOKEN_UIDS"
_ORACLE_DEBUG_START_TIME = time.perf_counter()


def _oracle_debug_enabled() -> bool:
    return os.environ.get(_ORACLE_DEBUG_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug(message: str) -> None:
    if not _oracle_debug_enabled():
        return
    elapsed = time.perf_counter() - _ORACLE_DEBUG_START_TIME
    print(f"[oracle-debug +{elapsed:.2f}s] {message}", flush=True)


def _enable_debug_traceback_dump() -> None:
    if not _oracle_debug_enabled():
        return
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)


def run_worker_subprocess(
    request: WorkerRunRequest,
    topology_dir: Path,
    *,
    repo_root: Path,
) -> None:
    """Runs one distributed worker subprocess and stores combined logs."""
    run_worker_subprocesses([request], [topology_dir], repo_root=repo_root)


def run_worker_subprocesses(
    requests: list[WorkerRunRequest],
    topology_dirs: list[Path],
    *,
    repo_root: Path,
) -> None:
    """Runs compatible requests in one distributed rank-process lifetime."""
    if not requests or len(requests) != len(topology_dirs):
        raise ValueError(
            "Worker requests and topology directories must be non-empty and aligned"
        )
    topology = requests[0].topology
    if any(request.topology != topology for request in requests[1:]):
        raise ValueError("One worker process lifetime requires one parallel topology")

    request_paths: list[Path] = []
    for request, topology_dir in zip(requests, topology_dirs, strict=True):
        request_path = topology_dir / "run_request.json"
        _write_json(request_path, request.model_dump(mode="json"))
        request_paths.append(request_path)
    worker_module = "integration.megatron.model_support.oracle_worker"
    worker_cwd = repo_root / "tests"

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(topology.world_size()),
        "-m",
        worker_module,
        "--worker-run",
    ]
    for request_path in request_paths:
        command.extend(("--run-request", str(request_path)))
    output_tail: deque[str] = deque(maxlen=80)
    live_log_raw = os.environ.get("ART_ORACLE_LIVE_TRAINING_LOG")
    live_log_path = None if not live_log_raw else Path(live_log_raw)
    run: subprocess.Popen[str] | None = None
    for topology_dir in topology_dirs:
        topology_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as logs:
        worker_logs = [
            logs.enter_context(
                (topology_dir / "worker.log").open("w", encoding="utf-8")
            )
            for topology_dir in topology_dirs
        ]
        live_log = None
        try:
            if live_log_path is not None:
                live_log_path.parent.mkdir(parents=True, exist_ok=True)
                live_log = live_log_path.open("a", encoding="utf-8")
                live_log.write(f"\n=== {requests[0].objective} {topology.slug()} ===\n")
                live_log.flush()
            env = {
                **os.environ,
                "ART_MEGATRON_ATTACH_TOKEN_UIDS": "1",
                "PYTHONUNBUFFERED": "1",
            }
            if requests[0].case_config.precision == "fp32":
                env["NVIDIA_TF32_OVERRIDE"] = "0"
            run = subprocess.Popen(
                command,
                cwd=str(worker_cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert run.stdout is not None
            active_log_index = 0
            request_markers = {
                f"=== oracle request {index} ===": index
                for index in range(len(requests))
            }
            for line in run.stdout:
                output_tail.append(line.rstrip())
                marker_index = request_markers.get(line.strip())
                if marker_index is not None:
                    active_log_index = marker_index
                worker_logs[active_log_index].write(line)
                worker_logs[active_log_index].flush()
                if live_log is not None:
                    live_log.write(line)
                    live_log.flush()
            run.returncode = run.wait()
        finally:
            if run is not None and run.poll() is None:
                terminate_popen_process_group(run)
            if live_log is not None:
                live_log.close()
    if run.returncode != 0:
        raise RuntimeError(
            f"Topology run failed for {topology.slug()} with exit code "
            f"{run.returncode}.\n" + "\n".join(output_tail)
        )


def _set_deterministic_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def provider_topology_env_vars(topology: Topology) -> dict[str, str]:
    return {
        env_var: str(getattr(topology, field))
        for field, env_var in _TOPOLOGY_ENV_VARS.items()
        if field != "vpp" or topology.vpp > 1
    }


@contextmanager
def provider_topology_env(topology: Topology):
    previous = {name: os.environ.get(name) for name in _TOPOLOGY_ENV_VARS.values()}
    os.environ.update(provider_topology_env_vars(topology))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
                continue
            os.environ[name] = value


def _merge_sharded_dicts(shards_by_rank: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges rank-sharded LoRA tensors into a full state dict on rank 0."""
    from art.megatron.weights.lora_publish import merge_sharded_adapter_entries

    entries_by_key: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {}
    for rank_entry in shards_by_rank:
        rank_state = rank_entry["state"]
        rank_manifest = rank_entry["manifest"]
        for key, tensor in rank_state.items():
            if key not in rank_manifest:
                raise RuntimeError(f"Missing manifest entry for sharded key '{key}'")
            entries_by_key.setdefault(key, []).append(
                (rank_manifest[key], tensor.detach().cpu())
            )
    return merge_sharded_adapter_entries(entries_by_key)


def _gather_full_state(
    local_state: dict[str, Any],
    local_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Gathers local state dicts to rank 0 and merges them."""
    import torch

    rank = torch.distributed.get_rank()  # ty: ignore[possibly-missing-attribute]
    world_size = torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(  # ty: ignore[possibly-missing-attribute]
        {"state": local_state, "manifest": local_manifest},
        gathered,
        dst=0,
    )
    if rank != 0:
        return None
    assert gathered is not None
    entries = [entry for entry in gathered if entry is not None]
    return _merge_sharded_dicts(entries)


def _collect_lora_state(
    model_chunks: list[Any],
    *,
    optimizer_master: bool = False,
) -> dict[str, Any] | None:
    """Collects full LoRA adapter state for validation and delta computation."""
    local_state: dict[str, Any] = {}
    local_manifest: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if hasattr(module, "sharded_lora_manifest"):
                module_manifest = module.sharded_lora_manifest()
                for key, value in module_manifest.items():
                    if key in local_manifest and local_manifest[key] != value:
                        raise RuntimeError(
                            f"Duplicate manifest key while collecting state: {key}"
                        )
                    local_manifest[key] = value
            if optimizer_master:
                export_items = getattr(module, "_export_items", None)
                if not callable(export_items):
                    continue
                module_state = {}
                for key, param, expert in export_items():
                    main_param = getattr(param, "main_param", None)
                    if main_param is None and param.dtype == torch.float32:
                        main_param = param
                    if main_param is None or bool(
                        getattr(param, "main_param_sharded", False)
                    ):
                        raise RuntimeError(
                            f"Oracle requires a full FP32 optimizer master parameter for '{key}'"
                        )
                    value = main_param[expert] if expert is not None else main_param
                    module_state[key] = value.T
            elif hasattr(module, "sharded_lora_state_dict"):
                module_state = module.sharded_lora_state_dict()
            else:
                continue
            for key, value in module_state.items():
                if key in local_state:
                    raise RuntimeError(
                        f"Duplicate LoRA key while collecting state: {key}"
                    )
                local_state[key] = value.detach().cpu().contiguous()
    return _gather_full_state(local_state, local_manifest)


def _collect_lora_grads(
    model_chunks: list[Any],
) -> dict[str, Any] | None:
    """Collects full LoRA gradient tensors across all ranks."""
    local_grads: dict[str, Any] = {}
    local_manifest: dict[str, Any] = {}
    for chunk in model_chunks:
        for module in chunk.modules():
            if hasattr(module, "sharded_lora_manifest"):
                module_manifest = module.sharded_lora_manifest()
                for key, value in module_manifest.items():
                    if key in local_manifest and local_manifest[key] != value:
                        raise RuntimeError(
                            f"Duplicate manifest key while collecting grads: {key}"
                        )
                    local_manifest[key] = value
            if not hasattr(module, "sharded_lora_grad_dict"):
                continue
            module_grads = module.sharded_lora_grad_dict()
            for key, value in module_grads.items():
                if key in local_grads:
                    raise RuntimeError(
                        f"Duplicate LoRA grad key while collecting grads: {key}"
                    )
                local_grads[key] = value.detach().cpu().contiguous()
    return _gather_full_state(local_grads, local_manifest)


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
    *,
    model_chunks: list[Any],
    model_support_handler: Any,
) -> None:
    """Checks loaded model LoRA state exactly matches adapter tensors and keys."""
    import torch

    expected_state = model_support_handler.canonicalize_loaded_lora_state(
        adapter_model,
        model_chunks,
    )
    for key in sorted(expected_state.keys()):
        assert torch.equal(loaded_state[key].cpu(), expected_state[key].cpu()), (
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


def _output_tensor_map(
    outputs: list[torch.Tensor],
    sample_indices: list[int | None],
    valid_lengths: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Materializes the exact per-micro tensors consumed by comparison."""
    if not outputs:
        raise RuntimeError("Expected at least one captured micro output")
    first = outputs[0]
    if any(tensor.ndim != first.ndim for tensor in outputs[1:]) or any(
        tensor.shape[:-1] != first.shape[:-1] for tensor in outputs[1:]
    ):
        raise RuntimeError("Unable to stack output tensors with incompatible shapes")
    max_last_dim = max(int(tensor.shape[-1]) for tensor in outputs) if first.ndim else 0
    result: dict[str, torch.Tensor] = {}
    for index, output in enumerate(outputs):
        output = (-output).contiguous()
        sample_index = sample_indices[index] if index < len(sample_indices) else None
        target_length = max_last_dim
        if isinstance(sample_index, int):
            target_length = min(target_length, max(valid_lengths[sample_index] - 1, 0))
        if output.ndim and int(output.shape[-1]) < target_length:
            padded = output.new_full(
                (*output.shape[:-1], target_length),
                float("nan") if output.dtype.is_floating_point else 0,
            )
            padded[..., : output.shape[-1]] = output
            output = padded
        elif output.ndim and int(output.shape[-1]) > target_length:
            output = output[..., :target_length].contiguous()
        result[f"logprobs.micro_{index:03d}"] = output
    return result


def _configure_provider(
    provider: Any,
    topology: Topology,
    case_config: OracleCaseConfig,
    prepare_moe_routing_replay: bool = False,
) -> None:
    """Applies deterministic topology/model overrides to provider config.

    Handler-specific oracle hooks are validation-only. They keep large model
    families such as DSV4 fit-sized while preserving the layer families and
    kernel-facing invariants under test.
    """
    del topology
    provider.num_layers = case_config.num_layers
    for name in ("moe_layer_freq", "glm52_indexer_types", "hybrid_layer_pattern"):
        pattern = getattr(provider, name, None)
        if isinstance(pattern, (list, tuple, str)):
            setattr(provider, name, type(pattern)(pattern[: case_config.num_layers]))
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
    handler = provider._art_model_support_handler
    configure_oracle_provider = getattr(handler, "configure_oracle_provider", None)
    if configure_oracle_provider is not None:
        configure_oracle_provider(provider, case_config=case_config)
    if prepare_moe_routing_replay:
        from art.megatron.train import _enable_native_moe_routing_replay

        _enable_native_moe_routing_replay(provider)


@contextmanager
def _patch_finalize_provider_bundle_for_oracle(
    megatron_train_module: Any,
    case_config: OracleCaseConfig,
):
    original_finalize_provider_bundle = megatron_train_module.finalize_provider_bundle

    def _oracle_finalize_provider_bundle(provider_bundle: Any) -> Any:
        provider = provider_bundle.provider
        from art.megatron.provider import _finalize_provider_with_art_overrides

        if case_config.precision == "fp32":
            provider.moe_token_dispatcher_type = "alltoall"
            provider.moe_flex_dispatcher_backend = None
            provider.moe_enable_deepep = False
            provider.overlap_moe_expert_parallel_comm = False
            provider.delay_wgrad_compute = False
            provider.ep_overlap_early_attn_memory_release = False
        _finalize_provider_with_art_overrides(provider)
        return provider_bundle

    megatron_train_module.finalize_provider_bundle = _oracle_finalize_provider_bundle
    try:
        yield
    finally:
        megatron_train_module.finalize_provider_bundle = (
            original_finalize_provider_bundle
        )


def _build_optimizer_config(case_config: OracleCaseConfig):
    """Builds a linear one-step optimizer for deterministic delta comparisons."""
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
            optimizer="sgd",
            sgd_momentum=0.0,
            lr=case_config.learning_rate,
            clip_grad=0.0,
            weight_decay=0.0,
        )
    return OptimizerConfig(
        bf16=True,
        fp16=False,
        optimizer="sgd",
        sgd_momentum=0.0,
        lr=case_config.learning_rate,
        clip_grad=0.0,
        weight_decay=0.0,
    )


def _configure_cuda_precision(case_config: OracleCaseConfig) -> None:
    if case_config.precision != "fp32":
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _sparse_flex_test_call(attention_call: Callable[..., Any]) -> Callable[..., Any]:
    from torch.nn.attention.flex_attention import AuxRequest

    def sparse_call(q, k, v, *, block_mask, scale, enable_gqa):
        out, aux = attention_call(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_aux=AuxRequest(lse=True),
        )
        return out, aux.lse

    return sparse_call


@contextmanager
def _apply_requested_flex_backend_patch(flex_backend: str | None):
    if flex_backend is None:
        yield
        return

    import art.megatron.flex_attn.compiled as compiled_flex_attention

    original_dense = compiled_flex_attention.dense_compiled_flex_attention
    original_sparse = compiled_flex_attention.sparse_compiled_flex_attention
    original_backend = compiled_flex_attention._FORCED_FLEX_BACKEND
    original_kernel_options = compiled_flex_attention._FORCED_FLEX_KERNEL_OPTIONS
    if flex_backend == "FLASH":
        patched_backend = "FLASH"
        patched_kernel_options = cast(Any, {"BACKEND": "FLASH"})
    elif flex_backend == "TRITON":
        patched_backend = "TRITON"
        patched_kernel_options = cast(Any, {"BACKEND": "TRITON"})
    elif flex_backend in {
        "TRITON_LEGACY",
        "TRITON_LEGACY_INNER_FP32",
        "TRITON_LEGACY_FULL_FP32",
    }:
        patched_backend = "TRITON"
        patched_kernel_options = cast(Any, {"FORCE_USE_FLEX_ATTENTION": True})
    else:
        raise RuntimeError(f"Unsupported flex backend request: {flex_backend}")

    setattr(compiled_flex_attention, "_FORCED_FLEX_BACKEND", patched_backend)
    compiled_flex_attention._FORCED_FLEX_KERNEL_OPTIONS = patched_kernel_options
    compiled_flex_attention.dense_compiled_flex_attention = torch.compile(
        compiled_flex_attention._forced_flex_attention_dense
    )
    compiled_flex_attention.sparse_compiled_flex_attention = torch.compile(
        compiled_flex_attention._sparse_flex_attention_with_options(
            patched_kernel_options
        )
    )
    try:
        yield
    finally:
        compiled_flex_attention._FORCED_FLEX_BACKEND = original_backend
        compiled_flex_attention._FORCED_FLEX_KERNEL_OPTIONS = original_kernel_options
        compiled_flex_attention.dense_compiled_flex_attention = original_dense
        compiled_flex_attention.sparse_compiled_flex_attention = original_sparse


@contextmanager
def _apply_test_flex_inner_fp32_patch(flex_backend: str | None):
    if flex_backend != "TRITON_LEGACY_INNER_FP32":
        yield
        return

    from torch.nn.attention.flex_attention import AuxRequest, flex_attention

    import art.megatron.flex_attn.compiled as compiled_flex_attention

    original_dense = compiled_flex_attention.dense_compiled_flex_attention
    original_sparse = compiled_flex_attention.sparse_compiled_flex_attention
    legacy_kernel_options = cast(Any, {"FORCE_USE_FLEX_ATTENTION": True})

    def _fp32_inner_call(
        q,
        k,
        v,
        *,
        block_mask,
        scale,
        enable_gqa,
        return_aux: AuxRequest | None = None,
    ):
        out = flex_attention(
            q.float(),
            k.float(),
            v.float(),
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            kernel_options=legacy_kernel_options,
            return_aux=return_aux,
        )
        if return_aux is None:
            assert torch.is_tensor(out)
            return out.to(dtype=q.dtype)
        attn_out, aux = out
        return attn_out.to(dtype=q.dtype), aux

    compiled_flex_attention.dense_compiled_flex_attention = torch.compile(
        _fp32_inner_call
    )
    compiled_flex_attention.sparse_compiled_flex_attention = torch.compile(
        _sparse_flex_test_call(_fp32_inner_call)
    )
    try:
        yield
    finally:
        compiled_flex_attention.dense_compiled_flex_attention = original_dense
        compiled_flex_attention.sparse_compiled_flex_attention = original_sparse


@contextmanager
def _apply_test_attention_full_fp32_patch(flex_backend: str | None):
    if flex_backend != "TRITON_LEGACY_FULL_FP32":
        yield
        return

    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.transformer.attention import Attention
    from torch.nn.attention.flex_attention import AuxRequest, flex_attention

    import art.megatron.flex_attn.compiled as compiled_flex_attention

    original_dense = compiled_flex_attention.dense_compiled_flex_attention
    original_sparse = compiled_flex_attention.sparse_compiled_flex_attention
    original_column_forward_impl = ColumnParallelLinear._forward_impl
    original_row_forward_impl = RowParallelLinear._forward_impl
    original_attention_forward = Attention.forward
    legacy_kernel_options = cast(Any, {"FORCE_USE_FLEX_ATTENTION": True})

    def _fp32_inner_call(
        q,
        k,
        v,
        *,
        block_mask,
        scale,
        enable_gqa,
        return_aux: AuxRequest | None = None,
    ):
        out = flex_attention(
            q.float(),
            k.float(),
            v.float(),
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            kernel_options=legacy_kernel_options,
            return_aux=return_aux,
        )
        if return_aux is None:
            return out
        return out

    def _column_forward_impl_fp32(self, input, weight, *args, **kwargs):
        fp32_kwargs = dict(kwargs)
        if fp32_kwargs.get("bias") is not None:
            fp32_kwargs["bias"] = fp32_kwargs["bias"].float()
        return original_column_forward_impl(
            self, input.float(), weight.float(), *args, **fp32_kwargs
        )

    def _row_forward_impl_fp32(self, input, weight, *args, **kwargs):
        fp32_kwargs = dict(kwargs)
        if fp32_kwargs.get("bias") is not None:
            fp32_kwargs["bias"] = fp32_kwargs["bias"].float()
        return original_row_forward_impl(
            self, input.float(), weight.float(), *args, **fp32_kwargs
        )

    def _attention_forward_fp32(self, hidden_states, *args, **kwargs):
        output, bias = original_attention_forward(self, hidden_states, *args, **kwargs)
        target_dtype = hidden_states.dtype
        if torch.is_tensor(output):
            output = output.to(dtype=target_dtype)
        if torch.is_tensor(bias):
            bias = bias.to(dtype=target_dtype)
        return output, bias

    compiled_flex_attention.dense_compiled_flex_attention = torch.compile(
        _fp32_inner_call
    )
    compiled_flex_attention.sparse_compiled_flex_attention = torch.compile(
        _sparse_flex_test_call(_fp32_inner_call)
    )
    setattr(ColumnParallelLinear, "_forward_impl", _column_forward_impl_fp32)
    setattr(RowParallelLinear, "_forward_impl", _row_forward_impl_fp32)
    setattr(Attention, "forward", _attention_forward_fp32)
    try:
        yield
    finally:
        compiled_flex_attention.dense_compiled_flex_attention = original_dense
        compiled_flex_attention.sparse_compiled_flex_attention = original_sparse
        setattr(ColumnParallelLinear, "_forward_impl", original_column_forward_impl)
        setattr(RowParallelLinear, "_forward_impl", original_row_forward_impl)
        setattr(Attention, "forward", original_attention_forward)


def _assert_runtime_configuration(
    model_chunks: list[Any],
    case_config: OracleCaseConfig,
    topology: Topology,
) -> None:
    """Validates runtime model depth/topology equals requested oracle config."""
    observed_num_layers: set[int] = set()
    observed_context_parallel_sizes: set[int] = set()
    gdn_layers = 0
    standard_attention_layers = 0

    try:
        import megatron.core.ssm.gated_delta_net as gated_delta_net
    except ImportError:  # pragma: no cover - optional dependency guard.
        gated_delta_net_type = None
    else:
        gated_delta_net_type = getattr(gated_delta_net, "GatedDeltaNet")
    from megatron.core.transformer.attention import SelfAttention

    for chunk in model_chunks:
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "num_layers"):
            observed_num_layers.add(int(config.num_layers))
        if config is not None and hasattr(config, "context_parallel_size"):
            observed_context_parallel_sizes.add(int(config.context_parallel_size))
        for child in module.modules():
            if gated_delta_net_type is not None and isinstance(
                child, gated_delta_net_type
            ):
                gdn_layers += 1
            if isinstance(child, SelfAttention):
                standard_attention_layers += 1

    if observed_num_layers != {case_config.num_layers}:
        raise RuntimeError(
            "Runtime num_layers mismatch: "
            f"requested={case_config.num_layers}, observed={sorted(observed_num_layers)}"
        )
    if observed_context_parallel_sizes != {topology.cp}:
        raise RuntimeError(
            "Runtime context_parallel_size mismatch: "
            f"requested={topology.cp}, observed={sorted(observed_context_parallel_sizes)}"
        )
    if "qwen3.5" not in case_config.base_model.lower():
        return
    if gdn_layers <= 0:
        raise RuntimeError("Expected Qwen3.5 runtime to include GatedDeltaNet layers.")
    if topology.cp > 1 and case_config.num_layers == 1 and standard_attention_layers:
        raise RuntimeError(
            "Expected one-layer Qwen3.5 CP oracle to skip standard self-attention, "
            f"found {standard_attention_layers} SelfAttention layer(s)."
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
            ".mlp.experts.linear_fc1.lora.A_T" in param_name
            or ".mlp.experts.linear_fc1.gate_lora.A_T" in param_name
            or ".mlp.experts.linear_fc1.up_lora.A_T" in param_name
            or ".mlp.linear_fc1.gate_lora.A_T" in param_name
            or ".mlp.linear_fc1.up_lora.A_T" in param_name
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
        if mutation == "bwd_skip_sync_fc1_a" and (
            ".mlp.experts." in param_name and param.grad_sync_domain != "expert_tp"  # ty: ignore[unresolved-attribute]
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
            if not module.reduce_output:
                continue
            adapter_prefix = module.lora.adapter_model_prefix
            if not adapter_prefix.endswith((".o_proj", ".out_proj")):
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
def _apply_attention_async_comm_mutation(mutation: SensitivityMutation | None):
    if mutation != "attn_kv_fetch_pack_on_comm_stream":
        yield
        return

    from art.megatron.context_parallel import comm

    original = comm.A2AVCommunicator._launch_exchange
    comm_delay_cycles = 80_000_000

    def _mutated_launch_exchange(
        self: Any,
        *,
        tensor: torch.Tensor,
        recv_buffer: torch.Tensor,
        total_send_rows: int,
        make_send_buffer: Callable[[], torch.Tensor],
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        group: Any,
        async_op: bool,
        input_layout: str,
        row_factor: int = 2,
    ):
        stream = self._get_stream(tensor) if async_op else None
        if stream is None:
            return original(
                self,
                tensor=tensor,
                recv_buffer=recv_buffer,
                total_send_rows=total_send_rows,
                make_send_buffer=make_send_buffer,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=async_op,
                input_layout=input_layout,
                row_factor=row_factor,
            )
        current_stream = torch.cuda.current_stream(tensor.device)
        if total_send_rows > 0:
            stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            if total_send_rows <= 0:
                send_buffer = tensor.new_empty(
                    comm._packed_peer_tensor_shape(
                        tensor=tensor,
                        total_rows=0,
                        input_layout=input_layout,
                        row_factor=row_factor,
                    )
                )
            else:
                send_buffer = make_send_buffer()
            if total_send_rows > 0:
                torch.cuda._sleep(comm_delay_cycles)
            handle = comm._launch_peer_exchange(
                recv_buffer=recv_buffer,
                send_buffer=send_buffer,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
        if total_send_rows > 0 and send_buffer.numel() > 0:
            send_buffer.zero_()
        return handle, send_buffer, stream

    comm.A2AVCommunicator._launch_exchange = _mutated_launch_exchange  # type: ignore[invalid-assignment]
    try:
        yield
    finally:
        comm.A2AVCommunicator._launch_exchange = original


@contextmanager
def _apply_attention_nested_grad_mutation(mutation: SensitivityMutation | None):
    if mutation != "attn_skip_nested_grad_sanitize":
        yield
        return

    from art.megatron.context_parallel import executor

    original = executor._sanitize_nested_stage_input_grad
    shared_scratch: dict[tuple[int | None, torch.dtype], torch.Tensor] = {}

    def _mutated_sanitize(grad: torch.Tensor | None) -> torch.Tensor | None:
        if grad is None:
            return None
        key = (grad.device.index, grad.dtype)
        flat = shared_scratch.get(key)
        needed = int(grad.numel())
        if flat is None or flat.numel() < needed:
            flat = torch.empty(needed, device=grad.device, dtype=grad.dtype)
            shared_scratch[key] = flat
        view = flat[:needed].view_as(grad)
        view.copy_(grad)
        return view

    setattr(executor, "_sanitize_nested_stage_input_grad", _mutated_sanitize)
    try:
        yield
    finally:
        executor._sanitize_nested_stage_input_grad = original


@contextmanager
def _apply_attention_lse_normalize_mutation(mutation: SensitivityMutation | None):
    if mutation != "attn_skip_flash_lse_normalize":
        yield
        return

    from art.megatron.context_parallel import executor
    import art.megatron.flex_attn.compiled as compiled_flex_attention

    original_compiled = compiled_flex_attention.normalize_flex_lse
    original_executor = executor.normalize_flex_lse

    def _identity(lse: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        return lse

    setattr(compiled_flex_attention, "normalize_flex_lse", _identity)
    setattr(executor, "normalize_flex_lse", _identity)
    try:
        yield
    finally:
        compiled_flex_attention.normalize_flex_lse = original_compiled
        executor.normalize_flex_lse = original_executor


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

        if tokens_per_expert is None or not self.is_expert:
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
        adapter_out = (
            self.up_lora(x, tokens_per_expert)
            if self.non_gated
            else self.lora(x, tokens_per_expert)
            if self.fused_gate_up
            else torch.cat(
                (
                    self.gate_lora(x, tokens_per_expert),
                    self.up_lora(x, tokens_per_expert),
                ),
                dim=1,
            )
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
    original_local_sft_token_count_tensor = (
        megatron_train_module._local_trainable_sft_token_count_tensor
    )
    original_build_micro_sample_indices = (
        megatron_train_module.build_micro_sample_indices
    )

    known_mutations = {None, *SUPPORTED_SENSITIVITY_MUTATIONS}
    known_mutations |= {
        "attn_kv_fetch_pack_on_comm_stream",
        "attn_skip_nested_grad_sanitize",
        "attn_skip_flash_lse_normalize",
    }
    if mutation not in known_mutations:
        raise ValueError(f"Unsupported mutation: {mutation}")

    if mutation == "skip_finalize":
        megatron_train_module.finalize_model_grads_extended = (
            lambda _model, _num_tokens=None, **_kwargs: None
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

    if mutation == "sft_local_token_normalization":

        def _wrong_local_trainable_sft_token_count_tensor(
            micro_inputs: list[Any],
            device: torch.device,
        ) -> torch.Tensor:
            local_token_total = sum(
                megatron_train_module._count_sft_trainable_tokens(micro)
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

        megatron_train_module._local_trainable_sft_token_count_tensor = (
            _wrong_local_trainable_sft_token_count_tensor
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

        def _patched_optimizer_step(
            optimizer: Any,
            learning_rate: float,
            *,
            model_support_handler: Any | None = None,
            model_chunks: Any | None = None,
            before_step: Callable[[], None] | None = None,
        ):
            if pre_optimizer_step_hook is not None:
                pre_optimizer_step_hook()
            return original_optimizer_step(
                optimizer,
                learning_rate,
                model_support_handler=model_support_handler,
                model_chunks=model_chunks,
                before_step=before_step,
            )

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
        stack.enter_context(_apply_attention_async_comm_mutation(mutation))
        stack.enter_context(_apply_attention_nested_grad_mutation(mutation))
        stack.enter_context(_apply_attention_lse_normalize_mutation(mutation))
        try:
            yield
        finally:
            megatron_train_module.finalize_model_grads_extended = original_finalize
            megatron_train_module._optimizer_step = original_optimizer_step
            megatron_train_module.loss_fn = original_loss_fn
            megatron_train_module._local_trainable_token_count_tensor = (
                original_local_token_count_tensor
            )
            megatron_train_module._local_trainable_sft_token_count_tensor = (
                original_local_sft_token_count_tensor
            )
            megatron_train_module.build_micro_sample_indices = (
                original_build_micro_sample_indices
            )


class _WorkerSession:
    """Owns reusable distributed model state for one parallel topology."""

    def __init__(
        self,
        *,
        request: WorkerRunRequest,
        runtime: Any,
        weight_offload: Any,
        flex_patch_stack: ExitStack,
    ) -> None:
        self.request = request
        self.runtime = runtime
        self.weight_offload = weight_offload
        self.flex_patch_stack = flex_patch_stack
        self.rng_state: tuple[Any, Any, torch.Tensor, list[torch.Tensor]] | None = None

    def begin_request(self) -> None:
        self.weight_offload.before_job()
        if self.rng_state is None:
            self.rng_state = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all(),
            )
            return
        python_state, numpy_state, torch_state, cuda_states = self.rng_state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_states)

    def end_request(self) -> None:
        self.weight_offload.after_job()

    def close(self) -> None:
        _debug("starting worker session close")
        torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
        self.flex_patch_stack.close()
        torch.distributed.destroy_process_group()  # ty: ignore[possibly-missing-attribute]
        _debug("finished worker session close")


def _validate_session_request(
    session_request: WorkerRunRequest,
    request: WorkerRunRequest,
) -> None:
    """Rejects request differences that require rebuilding distributed state."""
    per_run_fields = {
        "objective",
        "topology_dir",
        "comparison_dir",
        "mutation",
        "moe_routing_replay_path",
        "moe_routing_replay_strict",
        "capture_moe_routing_bundle_path",
    }
    if session_request.model_dump(exclude=per_run_fields) != request.model_dump(
        exclude=per_run_fields
    ):
        raise ValueError("Worker requests require different distributed runtimes")


def _clear_optimizer_state(optimizer: Any) -> None:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for child in chained:
            _clear_optimizer_state(child)
        return
    inner = getattr(optimizer, "optimizer", None)
    state = getattr(inner, "state", None)
    if state is None:
        raise TypeError(f"{type(optimizer).__name__} has no mutable optimizer state")
    state.clear()


def _reset_optimizer_state(optimizer: Any) -> None:
    from art.megatron import train as megatron_train

    _clear_optimizer_state(optimizer)
    megatron_train._eager_initialize_optimizer_state(optimizer)


def _start_worker_session(request: WorkerRunRequest) -> _WorkerSession:
    """Builds distributed model state once for compatible oracle requests."""
    _debug("starting worker session setup")
    os.environ.setdefault(_ATTACH_TOKEN_UIDS_ENV, "1")
    from art.megatron import train as megatron_train
    from art.megatron.training.weight_offload import WeightOffloadManager

    if request.case_config.precision == "fp32":
        allow_fp32_grouped_gemm_fallback_for_model_support_tests()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")  # ty: ignore[possibly-missing-attribute]
    _enable_debug_traceback_dump()
    _set_deterministic_seed(request.case_config.seed)
    _configure_cuda_precision(request.case_config)
    flex_patch_stack = ExitStack()
    flex_patch_stack.enter_context(
        _apply_requested_flex_backend_patch(request.flex_backend)
    )
    flex_patch_stack.enter_context(
        _apply_test_flex_inner_fp32_patch(request.flex_backend)
    )
    flex_patch_stack.enter_context(
        _apply_test_attention_full_fp32_patch(request.flex_backend)
    )
    if request.case_config.precision == "fp32":
        install_megatron_qwen35_gdn_fp32_reference(
            flex_patch_stack,
            base_model=request.case_config.base_model,
        )

    with provider_topology_env(request.topology):
        _debug(
            f"starting build_training_runtime objective={request.objective} "
            f"topology={request.topology.slug()} local_rank={local_rank}"
        )
        with _patch_finalize_provider_bundle_for_oracle(
            megatron_train, request.case_config
        ):
            provider_torch_dtype = (
                torch.float32
                if request.case_config.precision == "fp32"
                else torch.bfloat16
            )
            runtime = megatron_train.build_training_runtime(
                model_identifier=(
                    request.case_config.provider_model or request.case_config.base_model
                ),
                provider_torch_dtype=provider_torch_dtype,
                provider_configure=lambda provider: _configure_provider(
                    provider,
                    request.topology,
                    request.case_config,
                    request.prepare_moe_routing_replay,
                ),
                optimizer_config=_build_optimizer_config(request.case_config),
                moe_routing_replay_path=request.moe_routing_replay_path,
                moe_routing_replay_strict=request.moe_routing_replay_strict,
                print_env=False,
                allow_unvalidated_arch=request.case_config.allow_unvalidated_arch,
                model_support_key=request.case_config.model_support_key,
            )
        _debug("finished build_training_runtime")
    model_chunks = runtime.model
    optimizer = runtime.optimizer
    _assert_runtime_configuration(model_chunks, request.case_config, request.topology)
    weight_offload = WeightOffloadManager.from_config(
        model=model_chunks,
        rank=torch.distributed.get_rank(),  # ty: ignore[possibly-missing-attribute]
        compile_enabled=runtime.transformer_layers_compiled,
        offload_between_jobs=request.offload_between_jobs,
        streaming_config=request.streaming_weight_offload,
    )
    weight_offload.install()
    weight_offload.after_job()
    _debug("finished worker session setup")
    return _WorkerSession(
        request=request,
        runtime=runtime,
        weight_offload=weight_offload,
        flex_patch_stack=flex_patch_stack,
    )


def _worker_run(
    request: WorkerRunRequest,
    session: _WorkerSession | None = None,
) -> _WorkerSession:
    """Executes one trace while retaining compatible distributed model state."""
    from safetensors.torch import load_file, save_file  # ty: ignore[unresolved-import]

    from art import dev, types
    from art.megatron import train as megatron_train
    from art.preprocessing.pack import packed_tensors_from_dir

    reused_runtime = session is not None
    if session is None:
        session = _start_worker_session(request)
    else:
        _validate_session_request(session.request, request)
    runtime = session.runtime
    model_chunks = runtime.model
    optimizer = runtime.optimizer
    capture_routes = request.capture_moe_routing_bundle_path is not None
    _debug(f"starting request objective={request.objective} capture={capture_routes}")
    session.begin_request()
    # Reloading LoRA masters does not clear moments from a prior paired objective.
    _reset_optimizer_state(optimizer)
    if reused_runtime:
        had_replay = runtime.moe_routing_replay_controller is not None
        megatron_train.configure_moe_routing_replay(
            runtime,
            replay_bundle_path=request.moe_routing_replay_path,
            strict=request.moe_routing_replay_strict,
        )
        if not had_replay and request.moe_routing_replay_path is not None:
            # Recompile with the full replay comparison hooks.
            torch.compiler.reset()

    topology_dir = Path(request.topology_dir)
    comparison_dir = _comparison_dir(request.comparison_dir)
    rank0 = torch.distributed.get_rank() == 0  # ty: ignore[possibly-missing-attribute]
    if rank0:
        atexit.register(_remove_comparison_dir, comparison_dir)
    routing_traces_dir = comparison_dir / "routing_traces"
    if rank0 and capture_routes:
        routing_traces_dir.mkdir(parents=True)

    # setup the shared initial lora
    shared_init_path = Path(request.shared_init_adapter_path)
    if not shared_init_path.exists():
        _debug("collecting initial lora state")
        initial_state = _collect_lora_state(model_chunks)
        if rank0:
            _debug("building deterministic initial lora state")
            shared_init_path.parent.mkdir(parents=True, exist_ok=True)
            deterministic_init = _build_deterministic_shared_init(
                _require_not_none(initial_state, "initial_state"),
                seed=request.case_config.seed,
            )
            _debug("saving deterministic initial lora state")
            save_file(
                deterministic_init,
                str(shared_init_path),
            )
    _debug("waiting for shared initial lora state")
    torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]

    # load the shared initial lora into the model and validate we can collect it from the model
    _debug("loading shared initial lora state")
    adapter_model = load_file(str(shared_init_path))
    megatron_train.load_adapter_into_model(
        model_chunks,
        adapter_model,
        optimizer,
        model_support_handler=runtime.model_support_handler,
    )
    optimizer.zero_grad()
    megatron_train._zero_grad_buffers(model_chunks)
    _debug("collecting loaded lora state")
    loaded_state = _collect_lora_state(model_chunks)
    if rank0:
        _debug("validating loaded lora state")
        _validate_loaded_state_matches_adapter(
            _require_not_none(loaded_state, "loaded_state"),
            adapter_model,
            model_chunks=model_chunks,
            model_support_handler=runtime.model_support_handler,
        )
    _debug("waiting after loaded lora validation")
    torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]

    # load the inputs
    packed_tensors = packed_tensors_from_dir(
        **request.packed_tensors.model_dump(exclude_none=True)
    )
    valid_lengths = (
        _sample_valid_lengths(cast(dict[str, torch.Tensor], packed_tensors))
        if not capture_routes and rank0
        else None
    )
    sft_trajectory_tensors: list[dict[str, torch.Tensor]] | None = None
    rl_zero_template: PackedTensors | None = None
    sft_zero_template: dict[str, torch.Tensor] | None = None
    if request.objective == "rl":
        template = megatron_train.select_indexed_inputs(packed_tensors, 0)
        rl_zero_template = megatron_train._zero_contribution_inputs(template)
    else:
        sft_trajectory_tensors = build_sft_trajectory_tensors_from_packed_tensors(
            packed_tensors
        )
        sft_zero_template = megatron_train._zero_contribution_sft_inputs(
            sft_trajectory_tensors[0]
        )
    initial_optimizer_state = (
        None
        if capture_routes
        else _collect_lora_state(model_chunks, optimizer_master=True)
    )
    global_grad_accumulation_sequences = request.case_config.grad_accumulation_sequences

    train_config = types.TrainConfig(
        learning_rate=request.case_config.learning_rate,
        kl_penalty_coef=0.0,
        grad_accumulation_sequences=global_grad_accumulation_sequences,
    )
    experimental_config: dev.TrainConfig = {}
    step_traces: list[StepTrace] = []
    captured_grads: dict[str, Any] | None = None
    forward_trace_capture = ForwardTraceCapture(
        model_chunks,
        enabled=True,
        capture_name_tokens=(
            (ROUTER_NAME_TOKEN,) if capture_routes else CAPTURE_NAME_TOKENS
        ),
        capture_layer_outputs=not capture_routes,
    )
    install_moe_routing_trace_hooks(lambda: runtime.moe_routing_replay_controller)
    from megatron.core import parallel_state as ps

    topology = megatron_train._infer_parallel_topology(model_chunks)
    if ps.get_expert_model_parallel_world_size() > 1:
        sequence_length = (
            int(packed_tensors["tokens"].shape[1])
            if request.objective == "rl"
            else max(
                int(inputs["input_ids"].numel())
                for inputs in _require_not_none(
                    sft_trajectory_tensors, "sft_trajectory_tensors"
                )
            )
        )
        megatron_train._ensure_hybridep_capacity(
            runtime,
            packed_sequence_length=sequence_length,
            context_parallel_size=topology.cp,
        )

    def _capture_lora_grads() -> None:
        nonlocal captured_grads
        captured_grads = _collect_lora_grads(model_chunks)

    with ExitStack() as training_stack:
        training_stack.enter_context(install_gdn_trace_token_uid_hooks())
        training_stack.enter_context(
            _mutation_hook(
                megatron_train,
                model_chunks,
                request.mutation,
                request.topology,
                pre_optimizer_step_hook=(
                    None if capture_routes else _capture_lora_grads
                ),
                loss_scale=request.case_config.loss_scale,
            )
        )
        if request.use_fp32_lora_reference:
            training_stack.enter_context(_patch_lora_for_fp32(model_chunks, optimizer))

        _debug("starting training loop")
        for step_index in range(request.case_config.num_steps):
            hybridep_token_counts = None
            if ps.get_expert_model_parallel_world_size() > 1:
                if request.objective == "rl":
                    hybridep_token_counts = megatron_train.build_rl_hybridep_token_counts(
                        packed_tensors=packed_tensors,
                        step_index=step_index,
                        num_sequences=request.packed_tensors.num_sequences,
                        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                        topology=topology,
                        provider=runtime.provider,
                        model_support_handler=runtime.model_support_handler,
                    )
                else:
                    hybridep_token_counts = megatron_train.build_sft_hybridep_token_counts(
                        trajectory_tensors=_require_not_none(
                            sft_trajectory_tensors, "sft_trajectory_tensors"
                        ),
                        step_index=step_index,
                        global_grad_accumulation_sequences=global_grad_accumulation_sequences,
                        topology=topology,
                        provider=runtime.provider,
                        model_support_handler=runtime.model_support_handler,
                    )
            micro_sample_indices = megatron_train.build_micro_sample_indices(
                step_index=step_index,
                num_sequences=request.packed_tensors.num_sequences,
                global_grad_accumulation_sequences=global_grad_accumulation_sequences,
            )
            trace_sample_indices = micro_sample_indices
            if request.mutation == "dp_grad_accumulation_seqs":
                trace_offset = (
                    ps.get_data_parallel_rank() * request.packed_tensors.num_sequences
                )
                trace_sample_indices = [
                    None if index is None else index + trace_offset
                    for index in micro_sample_indices
                ]
            forward_trace_capture.set_step(step_index, trace_sample_indices)
            captured_grads = None
            _debug(f"starting step_index={step_index}")
            if request.objective == "rl":
                micro_inputs = megatron_train.select_micro_inputs(
                    packed_tensors,
                    micro_sample_indices,
                    _require_not_none(rl_zero_template, "rl_zero_template"),
                )
                step_result = megatron_train.run_training_step(
                    model_chunks=model_chunks,
                    provider=runtime.provider,
                    model_support_handler=runtime.model_support_handler,
                    optimizer=optimizer,
                    learning_rate=train_config.learning_rate,
                    inputs=micro_inputs,
                    config=train_config,
                    experimental_config=experimental_config,
                    ref_logprobs=None,
                    step_index=step_index,
                    sample_index=micro_sample_indices,
                    moe_routing_replay_controller=runtime.moe_routing_replay_controller,
                    hybridep_token_counts=hybridep_token_counts,
                )
            else:
                micro_inputs = megatron_train.select_sft_micro_inputs(
                    _require_not_none(sft_trajectory_tensors, "sft_trajectory_tensors"),
                    micro_sample_indices,
                    _require_not_none(sft_zero_template, "sft_zero_template"),
                )
                step_result = megatron_train.run_megatron_sft_step(
                    model_chunks=model_chunks,
                    provider=runtime.provider,
                    model_support_handler=runtime.model_support_handler,
                    optimizer=optimizer,
                    learning_rate=train_config.learning_rate,
                    inputs=micro_inputs,
                    step_index=step_index,
                    sample_index=micro_sample_indices,
                    moe_routing_replay_controller=runtime.moe_routing_replay_controller,
                    hybridep_token_counts=hybridep_token_counts,
                )
            _debug(f"finished step_index={step_index}")
            print(f"finished step_index={step_index}", flush=True)
            ordered_micro_sample_indices = micro_sample_indices
            if capture_routes:
                forward_trace_capture.save_current_step(routing_traces_dir)
            else:
                ordered_step_outputs = (
                    forward_trace_capture.ordered_step_outputs_with_sample_indices()
                )
                ordered_micro_sample_indices, ordered_micro_outputs = (
                    (micro_sample_indices, None)
                    if ordered_step_outputs is None
                    else ordered_step_outputs
                )
                gathered_traces: list[Any] | None = (
                    [None] * torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
                    if rank0
                    else None
                )
                torch.distributed.gather_object(  # ty: ignore[possibly-missing-attribute]
                    forward_trace_capture.current_step_trace, gathered_traces, dst=0
                )
                merged_trace = (
                    None
                    if gathered_traces is None
                    else forward_trace_capture.canonicalize_trace(
                        forward_trace_capture._merge_rank_traces(
                            cast(Any, gathered_traces)
                        )
                    )
                )
                torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
                current_optimizer_state = _collect_lora_state(
                    model_chunks, optimizer_master=True
                )
                if rank0:
                    if request.mutation == "dp_grad_accumulation_seqs":
                        num_sequences = request.packed_tensors.num_sequences
                        ordered_micro_sample_indices = [
                            None if index is None else index % num_sequences
                            for index in ordered_micro_sample_indices
                        ]
                        for calls in _require_not_none(
                            merged_trace, "merged_trace"
                        ).values():
                            for call in calls:
                                sample_index = call.get("micro_sample_index")
                                if isinstance(sample_index, int):
                                    call["micro_sample_index"] = (
                                        sample_index % num_sequences
                                    )
                    trace = _trim_trace_padding(
                        _require_not_none(merged_trace, "merged_trace"),
                        valid_lengths=_require_not_none(valid_lengths, "valid_lengths"),
                        sequence_length=request.case_config.packed_tensors.sequence_length,
                    )
                    ordered_outputs = _require_not_none(
                        ordered_micro_outputs, "ordered_micro_outputs"
                    )
                    current_state = _require_not_none(
                        current_optimizer_state, "current_optimizer_state"
                    )
                    _write_comparison_sink(
                        comparison_dir / f"step_{step_index:03d}.safetensors",
                        {
                            "outputs": _output_tensor_map(
                                ordered_outputs,
                                ordered_micro_sample_indices,
                                _require_not_none(valid_lengths, "valid_lengths"),
                            ),
                            "grads": _require_not_none(
                                captured_grads, "captured_grads"
                            ),
                            "deltas": _apply_save_mutation_to_tensor_map(
                                _delta_state(
                                    _require_not_none(
                                        initial_optimizer_state,
                                        "initial_optimizer_state",
                                    ),
                                    current_state,
                                ),
                                mutation=request.mutation,
                            ),
                            "forward": ForwardTraceCapture.flatten_trace_tensors(
                                trace, value_key="primary_output"
                            ),
                            "router_scores": ForwardTraceCapture.flatten_trace_tensors(
                                trace, value_key="router_topk_scores"
                            ),
                            "router_topk_ids": ForwardTraceCapture.flatten_trace_tensors(
                                trace, value_key="router_topk_ids"
                            ),
                        },
                    )

            if rank0:
                step_traces.append(
                    StepTrace(
                        step_index=step_index,
                        loss=float(
                            step_result.reduced_loss.item()
                            / request.case_config.loss_scale
                        ),
                        probs_corr=step_result.probs_corr,
                        micro_sample_indices=list(ordered_micro_sample_indices),
                    )
                )
            torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]

    forward_trace_capture.close()

    if rank0:
        # build and save the moe routing replay bundle
        if capture_routes:
            replay_bundle = build_bundle_from_forward_trace_dir(
                traces_dir=routing_traces_dir,
                num_steps=request.case_config.num_steps,
                topology=ReplayParallelTopology.model_validate(
                    request.topology.model_dump(
                        include={"tp", "ep", "etp", "dp", "sp", "cp", "pp", "vpp"},
                        mode="python",
                    )
                ),
            )
            replay_bundle.to_dir(
                _require_not_none(
                    request.capture_moe_routing_bundle_path,
                    "capture_moe_routing_bundle_path",
                )
            )
            _remove_comparison_dir(comparison_dir)

        # build and save the run manifest
        manifest = RunManifest(
            git=request.git,
            case_id=request.case_id,
            objective=request.objective,
            base_model=request.case_config.base_model,
            num_layers=request.case_config.num_layers,
            topology=request.topology.slug(),
            world_size=request.topology.world_size(),
            seed=request.case_config.seed,
            num_steps=request.case_config.num_steps,
            comparison_dir=None if capture_routes else str(comparison_dir),
            packed_tensors=request.packed_tensors,
            offload_between_jobs=request.offload_between_jobs,
            streaming_weight_offload=request.streaming_weight_offload,
            use_fp32_lora_reference=request.use_fp32_lora_reference,
            steps=step_traces,
        )
        _write_json(topology_dir / "manifest.json", manifest.model_dump(mode="json"))
    session.end_request()
    _debug(f"finished request objective={request.objective}")
    if rank0:
        atexit.unregister(_remove_comparison_dir)
    return session


def run_worker_cli(run_request_paths: list[Path]) -> None:
    """Loads compatible worker requests and dispatches them in one process lifetime."""
    requests = [
        WorkerRunRequest.model_validate(_read_json(run_request_path))
        for run_request_path in run_request_paths
    ]
    session: _WorkerSession | None = None
    try:
        for index, request in enumerate(requests):
            if index > 0:
                torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
                if torch.distributed.get_rank() == 0:  # ty: ignore[possibly-missing-attribute]
                    print(
                        f"=== oracle request {index} ===",
                        flush=True,
                    )
                torch.distributed.barrier()  # ty: ignore[possibly-missing-attribute]
            session = _worker_run(request, session)
    finally:
        if session is not None:
            session.close()
        if _oracle_debug_enabled():
            faulthandler.cancel_dump_traceback_later()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parses worker CLI arguments."""
    parser = argparse.ArgumentParser(description="Megatron oracle harness worker")
    parser.add_argument("--worker-run", action="store_true")
    parser.add_argument("--run-request", type=Path, action="append")
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    """CLI entry for worker-only execution mode."""
    args = _parse_args(argv)
    if not args.worker_run:
        raise SystemExit("This module is intended for test imports or --worker-run")
    if not args.run_request:
        raise SystemExit("--run-request is required with --worker-run")
    run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
