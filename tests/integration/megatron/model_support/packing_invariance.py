from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast
from unittest.mock import patch

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_model import GPTModel
from pydantic import BaseModel, Field
import torch

from art.megatron import train as megatron_train
from art.megatron.model_support.discovery import inspect_architecture
from art.megatron.model_support.registry import (
    get_model_support_handler_for_spec,
    get_model_support_spec,
)
from art.megatron.model_support.spec import PrefixTreeModelStateContext
from art.megatron.prefix_tree import parse_prefix_tree_row
from art.megatron.prefix_tree_state import create_prefix_tree_state

from ..artifacts import GitRepoState, pinned_git_state
from .base_megatron_session import (
    BaseMegatronResetReport,
    BaseMegatronSessionKey,
    active_base_megatron_session,
    initialize_single_rank_process_group,
)
from .fp32_grouped_gemm import (
    allow_fp32_grouped_gemm_fallback_for_model_support_tests,
)
from .oracle_harness import (
    ORACLE_TOPOLOGY,
    TEST_DEFAULT_FLEX_BACKEND,
    OracleCaseConfig,
    PackedTensorConfig,
    _read_json,
    _write_json,
)
from .oracle_worker import (
    _apply_requested_flex_backend_patch,
    _apply_test_attention_full_fp32_patch,
    _apply_test_flex_inner_fp32_patch,
    _configure_provider,
    provider_topology_env,
)
from .prefix_tree_workloads import build_complex_prefix_tree_packed_tensors

allow_fp32_grouped_gemm_fallback_for_model_support_tests()

_LOGITS_MEAN_ABS_PCT_LIMITS = {"fp32": 0.5, "bf16": 3.0}
_DEBUG_ENV = "ART_PACKING_INVARIANCE_DEBUG"
PACKING_INVARIANCE_REPORT_FILENAME = "report.json"
PACKING_INVARIANCE_ARTIFACT_SUITE_NAME = "Megatron packing-invariance artifacts"
REPO_ROOT = Path(__file__).resolve().parents[4]
_SINGLE_ROTARY_OUTPUT_HANDLER_KEYS = frozenset(
    {
        "default_dense",
        "default_moe",
        "llama3_dense",
        "qwen3_dense",
        "qwen3_moe",
        "qwen3_5_dense",
        "qwen3_5_moe",
        "dsv4",
        "gpt_oss_moe",
    }
)
_TUPLE_ROTARY_OUTPUT_HANDLER_KEYS = frozenset({"gemma4_dense", "gemma4_moe"})


def _slugify(value: str) -> str:
    return value.lower().replace("/", "_").replace(".", "_").replace("-", "_")


def _artifact_dir(base_model: str) -> Path:
    root = Path(__file__).resolve().parents[4] / ".local" / "model_support_validation"
    path = root / _slugify(base_model) / "packing_invariance"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _debug_enabled() -> bool:
    value = os.environ.get(_DEBUG_ENV, "")
    return value not in ("", "0", "false", "False")


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[packing_invariance] {message}", flush=True)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _reset_vllm_compile_overrides() -> None:
    """Undo vLLM's global Inductor compile-thread override for this test worker."""
    os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
    torch._inductor.config.compile_threads = (
        torch._inductor.config.decide_compile_threads()
    )
    _debug_log(
        f"reset inductor compile_threads={torch._inductor.config.compile_threads}"
    )


def _cuda_synchronize(device: torch.device | None = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
        return
    torch.cuda.synchronize(device)


def _time_block(
    label: str,
    fn: Any,
    *,
    device: torch.device | None = None,
) -> Any:
    _cuda_synchronize(device)
    start = time.perf_counter()
    result = fn()
    _cuda_synchronize(device)
    elapsed = time.perf_counter() - start
    _debug_log(f"{label}: {elapsed:.3f}s")
    return result


def _cleanup_distributed_state() -> None:
    if getattr(ps, "model_parallel_is_initialized", lambda: False)():
        ps.destroy_model_parallel()
    if torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
        torch.distributed.destroy_process_group()  # type: ignore[possibly-missing-attribute]


def _locate_gpt_module(model_chunks: list[Any]) -> GPTModel:
    for chunk in model_chunks:
        module: Any = chunk
        while hasattr(module, "module"):
            module = module.module
        if isinstance(module, GPTModel):
            return module
        language_model = getattr(module, "language_model", None)
        if isinstance(language_model, GPTModel):
            return language_model
    raise RuntimeError("Failed to locate GPTModel for packing invariance validation")


class PackingInvarianceScenario(BaseModel):
    name: str
    num_sequences: int
    sequence_length: int
    checked_token_count: int
    prompt_family_count: int
    max_tree_depth: int
    repeated_position_key_count: int
    rotary_grouping_checked: bool
    rotary_grouping_respected: bool
    completion_pair_count: int
    logits_equivalent: bool
    logits_mean_abs_pct: float
    logits_mean_abs_pct_limit: float
    logits_max_abs_diff: float
    matched: bool


class PackingInvarianceReport(BaseModel):
    git: GitRepoState
    base_model: str
    output_dir: str
    num_layers: int
    precision: str
    base_megatron_session_reused: bool = False
    base_megatron_reset: BaseMegatronResetReport | None = None
    scenarios: list[PackingInvarianceScenario] = Field(default_factory=list)


class PackingInvarianceRunRequest(BaseModel):
    git: GitRepoState
    base_model: str
    num_layers: int
    output_dir: str
    allow_unvalidated_arch: bool = False


def _prompt_family_count(group_ids: torch.Tensor, parent_ids: torch.Tensor) -> int:
    families = 0
    for row_index in range(int(group_ids.shape[0])):
        valid_tokens = int((group_ids[row_index] != -1).sum().item())
        cursor = 0
        while cursor < valid_tokens:
            group_id = int(group_ids[row_index, cursor].item())
            parent_id = int(parent_ids[row_index, cursor].item())
            if group_id == parent_id:
                families += 1
            while (
                cursor < valid_tokens
                and int(group_ids[row_index, cursor].item()) == group_id
            ):
                cursor += 1
    return families


def _max_tree_depth(group_ids: torch.Tensor, parent_ids: torch.Tensor) -> int:
    return max(
        (
            segment.depth
            for row_index in range(int(group_ids.shape[0]))
            for segment in parse_prefix_tree_row(
                group_ids=group_ids[row_index],
                parent_ids=parent_ids[row_index],
            ).segments
        ),
        default=0,
    )


def _position_keys(position_ids: torch.Tensor) -> list[tuple[int, ...]]:
    if position_ids.ndim == 1:
        return [(int(value),) for value in position_ids.tolist()]
    if position_ids.ndim == 2:
        return [
            (int(position_ids[batch_index, token_index].item()),)
            for batch_index in range(int(position_ids.shape[0]))
            for token_index in range(int(position_ids.shape[1]))
        ]
    if position_ids.ndim == 3:
        channel_first = position_ids.permute(1, 2, 0).contiguous()
        return [
            tuple(
                int(value) for value in channel_first[batch_index, token_index].tolist()
            )
            for batch_index in range(int(channel_first.shape[0]))
            for token_index in range(int(channel_first.shape[1]))
        ]
    raise ValueError(
        f"Unsupported position_ids rank for packed position validation: {position_ids.ndim}"
    )


def _flatten_rotary_vectors(
    rotary_output: torch.Tensor,
    *,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    sequence_length = int(position_ids.shape[-1])
    batch_size = int(position_ids.shape[-2]) if position_ids.ndim >= 2 else 1
    if rotary_output.ndim < 2 or rotary_output.shape[0] != sequence_length:
        raise ValueError(
            "Unexpected rotary output shape for packed position validation: "
            f"{tuple(rotary_output.shape)} with position_ids shape {tuple(position_ids.shape)}"
        )
    embedding_dim = int(rotary_output.shape[-1])
    vectors = rotary_output.reshape(sequence_length, -1, embedding_dim)
    if vectors.shape[1] != batch_size:
        raise ValueError(
            "Rotary output batch/slot mismatch for packed position validation: "
            f"got {vectors.shape[1]} slots for batch_size={batch_size}"
        )
    return vectors.permute(1, 0, 2).reshape(batch_size * sequence_length, embedding_dim)


def _rotary_grouping_check(
    rotary_output: torch.Tensor | None,
    *,
    position_ids: torch.Tensor,
) -> tuple[bool, bool, int]:
    keys = _position_keys(position_ids)
    key_counts: dict[tuple[int, ...], int] = {}
    for key in keys:
        key_counts[key] = key_counts.get(key, 0) + 1
    repeated_position_key_count = sum(1 for count in key_counts.values() if count > 1)
    if rotary_output is None:
        return False, True, repeated_position_key_count
    vectors = _flatten_rotary_vectors(rotary_output, position_ids=position_ids)
    first_vector_by_key: dict[tuple[int, ...], torch.Tensor] = {}
    for key, vector in zip(keys, vectors, strict=True):
        reference = first_vector_by_key.get(key)
        if reference is None:
            first_vector_by_key[key] = vector
            continue
        if not torch.equal(reference, vector):
            return True, False, repeated_position_key_count
    return True, True, repeated_position_key_count


def _rotary_outputs_for_validation(
    *,
    handler: Any,
    preprocess_output: Any,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    handler_key = handler.key
    if handler_key == "glm52":
        model_state = handler.build_prefix_tree_model_state(
            PrefixTreeModelStateContext(
                input_pos=position_ids.detach().cpu(),
                group_ids=group_ids.detach().cpu(),
                parent_ids=parent_ids.detach().cpu(),
                device=position_ids.device,
            )
        )
        state = model_state.get("glm52")
        if (
            state is None
            or not torch.is_tensor(state.rope_cos)
            or not torch.is_tensor(state.rope_sin)
        ):
            raise RuntimeError("GLM-5.2 packed-position validation requires RoPE state")
        return (torch.cat((state.rope_cos, state.rope_sin), dim=-1).permute(1, 0, 2),)
    rotary_output = preprocess_output[1]
    if handler_key in _SINGLE_ROTARY_OUTPUT_HANDLER_KEYS:
        return (
            cast(torch.Tensor | None, rotary_output)
            if torch.is_tensor(rotary_output)
            else None,
        )
    if handler_key in _TUPLE_ROTARY_OUTPUT_HANDLER_KEYS:
        local_rotary, global_rotary = rotary_output
        return local_rotary, global_rotary
    raise RuntimeError(
        f"Packed position validation has no rotary output mapping for {handler_key!r}"
    )


def _build_art_realistic_packed_tensors(
    config: PackedTensorConfig,
    seed: int,
    *,
    deep: bool,
) -> dict[str, Any]:
    return build_complex_prefix_tree_packed_tensors(config, seed, deep=deep)


def _dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported packed-position precision: {precision}")


def _prefix_tree_leaf_paths(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    required_leaf_count: int = 2,
) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    tree = parse_prefix_tree_row(group_ids=group_ids, parent_ids=parent_ids)
    segment_by_group = {segment.group_id: segment for segment in tree.segments}
    child_count_by_group: dict[int, int] = {}
    for segment in tree.segments:
        if segment.group_id == segment.parent_id:
            continue
        child_count_by_group[segment.parent_id] = (
            child_count_by_group.get(segment.parent_id, 0) + 1
        )
    paths = [
        (
            tuple(
                (segment_by_group[group_id].start, segment_by_group[group_id].end)
                for group_id in leaf.ancestors
            ),
            (leaf.start, leaf.end),
        )
        for leaf in tree.segments
        if leaf.group_id != leaf.parent_id
        and child_count_by_group.get(leaf.group_id, 0) == 0
        and leaf.end - leaf.start >= 2
    ]
    return paths if len(paths) >= required_leaf_count else []


def _run_logits(
    *,
    model: Any,
    handler: Any,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_bias: Any,
) -> torch.Tensor:
    forward_kwargs = handler.get_forward_kwargs(
        model,
        attention_bias=attention_bias,
    )
    with torch.no_grad():
        return cast(
            torch.Tensor,
            model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=torch.zeros(
                    (1, 1, 1, 1),
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
                labels=None,
                **forward_kwargs,
            ),
        )


def _logits_equivalence_check(
    *,
    model: Any,
    handler: Any,
    provider: Any,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    mean_abs_pct_limit: float,
) -> tuple[int, bool, float, float]:
    _debug_log(
        "logits_check start "
        f"batch={int(input_ids.shape[0])} seq={int(input_ids.shape[1])}"
    )
    completion_pair_count = 0
    logits_max_abs_diff = 0.0
    logits_abs_sum = 0.0
    logits_ref_abs_sum = 0.0
    logits_numel = 0
    sliding_windows = tuple(
        dict.fromkeys(
            int(window) for window in getattr(provider, "art_flex_sliding_windows", ())
        )
    )
    recurrent_contract = handler.linear_recurrent_contract(provider)
    recurrent_planner_config = handler.linear_recurrent_planner_config(provider)
    for row_index in range(int(input_ids.shape[0])):
        row_group_ids = group_ids[row_index : row_index + 1]
        row_parent_ids = parent_ids[row_index : row_index + 1]
        leaf_paths = _prefix_tree_leaf_paths(row_group_ids[0], row_parent_ids[0])
        if not leaf_paths:
            _debug_log(f"logits_check row={row_index} skipped no prefix-tree leaves")
            continue
        row_input_ids = input_ids[row_index : row_index + 1]
        row_position_ids = position_ids[row_index : row_index + 1]
        packed_bias = create_prefix_tree_state(
            group_ids=row_group_ids,
            parent_ids=row_parent_ids,
            input_pos=row_position_ids,
            sliding_windows=sliding_windows,
            linear_recurrent_contract=recurrent_contract,
            recurrent_planner_config=recurrent_planner_config,
            model_support_handler=handler,
            attention_head_dim=getattr(provider, "kv_channels", None),
            attention_value_head_dim=getattr(provider, "kv_channels", None),
        )
        _debug_log(f"logits_check row={row_index} leaves={len(leaf_paths)}")
        packed_logits = _time_block(
            f"logits_check row={row_index} packed_forward",
            lambda: _run_logits(
                model=model,
                handler=handler,
                input_ids=row_input_ids,
                position_ids=row_position_ids,
                attention_bias=packed_bias,
            ),
            device=row_input_ids.device,
        )
        for leaf_index, (ancestor_segments, leaf_segment) in enumerate(leaf_paths):
            leaf_start, leaf_end = leaf_segment
            prompt_len = sum(end - start for start, end in ancestor_segments)
            reference_segments = (*ancestor_segments, leaf_segment)
            _debug_log(
                "logits_check row="
                f"{row_index} leaf={leaf_index} "
                f"ancestors={ancestor_segments} leaf={leaf_segment}"
            )
            reference_input_ids = torch.cat(
                tuple(row_input_ids[:, start:end] for start, end in reference_segments),
                dim=1,
            )
            reference_position_ids = torch.cat(
                tuple(
                    row_position_ids[:, start:end] for start, end in reference_segments
                ),
                dim=1,
            )
            reference_group_ids = torch.zeros_like(reference_input_ids)
            reference_parent_ids = torch.zeros_like(reference_input_ids)
            reference_bias = create_prefix_tree_state(
                group_ids=reference_group_ids,
                parent_ids=reference_parent_ids,
                input_pos=reference_position_ids,
                sliding_windows=sliding_windows,
                linear_recurrent_contract=recurrent_contract,
                recurrent_planner_config=recurrent_planner_config,
                model_support_handler=handler,
                attention_head_dim=getattr(provider, "kv_channels", None),
                attention_value_head_dim=getattr(provider, "kv_channels", None),
            )
            reference_logits = _time_block(
                f"logits_check row={row_index} leaf={leaf_index} reference_forward",
                lambda: _run_logits(
                    model=model,
                    handler=handler,
                    input_ids=reference_input_ids,
                    position_ids=reference_position_ids,
                    attention_bias=reference_bias,
                ),
                device=reference_input_ids.device,
            )
            packed_completion_logits = packed_logits[:, leaf_start : leaf_end - 1, :]
            reference_completion_logits = reference_logits[:, prompt_len:-1, :]
            diff = (packed_completion_logits - reference_completion_logits).abs()
            logits_abs_sum += float(diff.sum().item())
            logits_ref_abs_sum += float(reference_completion_logits.abs().sum().item())
            logits_numel += int(diff.numel())
            logits_max_abs_diff = max(logits_max_abs_diff, float(diff.max().item()))
            completion_pair_count += 1
            _debug_log(
                "logits_check row="
                f"{row_index} leaf={leaf_index} "
                f"max_abs_diff={float(diff.max().item()):.6f}"
            )
    if completion_pair_count > 0:
        mean_abs = logits_abs_sum / max(logits_numel, 1)
        typical_abs = logits_ref_abs_sum / max(logits_numel, 1)
        logits_mean_abs_pct = (mean_abs / (typical_abs + 1e-12)) * 100.0
        logits_equivalent = logits_mean_abs_pct <= mean_abs_pct_limit
        _debug_log(
            "logits_check done "
            f"pairs={completion_pair_count} "
            f"equivalent={logits_equivalent} "
            f"mean_abs_pct={logits_mean_abs_pct:.6f} "
            f"limit={mean_abs_pct_limit:.6f} "
            f"max_abs_diff={logits_max_abs_diff:.6f}"
        )
        return (
            completion_pair_count,
            logits_equivalent,
            logits_mean_abs_pct,
            logits_max_abs_diff,
        )
    _debug_log("logits_check finished without any prompt family")
    return 0, False, float("inf"), float("inf")


def _run_packing_invariance_subprocess(
    request: PackingInvarianceRunRequest,
    output_dir: Path,
) -> None:
    request_path = output_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))
    worker_cwd = REPO_ROOT / "tests"
    command = [
        sys.executable,
        "-m",
        "integration.megatron.model_support.packing_invariance",
        "--run-request",
        str(request_path),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    run = subprocess.run(
        command,
        cwd=str(worker_cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = f"{run.stdout}\n{run.stderr}".strip()
    (output_dir / "worker.log").write_text(combined_output + "\n", encoding="utf-8")
    if run.returncode != 0:
        tail = "\n".join(combined_output.splitlines()[-80:])
        raise RuntimeError(
            f"Packing invariance worker failed with exit code {run.returncode}.\n{tail}"
        )


def _run_packing_invariance_in_process(
    request: PackingInvarianceRunRequest,
    output_dir: Path,
) -> None:
    request_path = output_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))
    with (output_dir / "worker.log").open("w", encoding="utf-8") as worker_log:
        with redirect_stdout(worker_log), redirect_stderr(worker_log):
            run_worker_cli(request_path)


def _run_packing_invariance_worker(
    *,
    git: GitRepoState,
    base_model: str,
    num_layers: int,
    output_dir: Path,
    allow_unvalidated_arch: bool = False,
) -> PackingInvarianceReport:
    _debug_log(f"run start base_model={base_model} num_layers={num_layers}")
    _reset_vllm_compile_overrides()
    scenarios = [
        (
            "stop_early",
            PackedTensorConfig(
                num_sequences=4,
                sequence_length=_env_int(
                    "ART_PACKING_INVARIANCE_STOP_EARLY_SEQUENCE_LENGTH", 2048
                ),
                prefill_tokens=_env_int(
                    "ART_PACKING_INVARIANCE_STOP_EARLY_PREFILL_TOKENS", 256
                ),
                completion_branches_per_prefix=2,
                decode_tokens=_env_int(
                    "ART_PACKING_INVARIANCE_STOP_EARLY_DECODE_TOKENS", 128
                ),
                decode_tokens_jitter=_env_int(
                    "ART_PACKING_INVARIANCE_STOP_EARLY_DECODE_TOKENS_JITTER", 32
                ),
                packing_mode="stop_early",
            ),
            False,
        ),
        (
            "truncate",
            PackedTensorConfig(
                num_sequences=4,
                sequence_length=_env_int(
                    "ART_PACKING_INVARIANCE_TRUNCATE_SEQUENCE_LENGTH", 2048
                ),
                prefill_tokens=_env_int(
                    "ART_PACKING_INVARIANCE_TRUNCATE_PREFILL_TOKENS", 256
                ),
                completion_branches_per_prefix=2,
                decode_tokens=_env_int(
                    "ART_PACKING_INVARIANCE_TRUNCATE_DECODE_TOKENS", 128
                ),
                decode_tokens_jitter=_env_int(
                    "ART_PACKING_INVARIANCE_TRUNCATE_DECODE_TOKENS_JITTER", 32
                ),
                packing_mode="truncate",
            ),
            False,
        ),
        (
            "deep_nested",
            PackedTensorConfig(
                num_sequences=2,
                sequence_length=1024,
                prefill_tokens=384,
                completion_branches_per_prefix=2,
                decode_tokens=128,
                decode_tokens_jitter=64,
                packing_mode="stop_early",
            ),
            True,
        ),
        (
            "repeated_short",
            PackedTensorConfig(
                num_sequences=2,
                sequence_length=1024,
                prefill_tokens=96,
                completion_branches_per_prefix=2,
                decode_tokens=48,
                decode_tokens_jitter=16,
                packing_mode="stop_early",
            ),
            False,
        ),
    ]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for packing invariance validation")

    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    precision = handler.correctness_precision()
    mean_abs_pct_limit = _LOGITS_MEAN_ABS_PCT_LIMITS[precision]
    report = PackingInvarianceReport(
        git=git,
        base_model=base_model,
        output_dir=str(output_dir),
        num_layers=num_layers,
        precision=precision,
    )
    case_config = OracleCaseConfig(
        base_model=base_model,
        precision=precision,
        num_layers=num_layers,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    runtime: megatron_train.TrainingRuntime | None = None
    session = active_base_megatron_session()
    reused_runtime = session is not None
    runtime_stack = ExitStack()
    if not torch.distributed.is_initialized():
        initialize_single_rank_process_group()
        runtime_stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                    "LOCAL_RANK": "0",
                    "LOCAL_WORLD_SIZE": "1",
                },
            )
        )
    runtime_stack.enter_context(
        _apply_requested_flex_backend_patch(TEST_DEFAULT_FLEX_BACKEND)
    )
    if precision == "fp32":
        runtime_stack.enter_context(
            _apply_test_flex_inner_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        runtime_stack.enter_context(
            _apply_test_attention_full_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
    try:
        if session is not None:
            runtime = session.reset_for_packing(
                key=BaseMegatronSessionKey(
                    base_model=base_model,
                    model_key=spec.key,
                    num_layers=num_layers,
                    precision=precision,
                    allow_unvalidated_arch=allow_unvalidated_arch,
                )
            )
            report.base_megatron_session_reused = True
            report.base_megatron_reset = session.reset_report
        else:
            with provider_topology_env(ORACLE_TOPOLOGY):
                runtime = _time_block(
                    "build_training_runtime",
                    lambda: megatron_train.build_training_runtime(
                        model_identifier=base_model,
                        provider_torch_dtype=_dtype_for_precision(precision),
                        provider_configure=lambda provider: _configure_provider(
                            provider,
                            ORACLE_TOPOLOGY,
                            case_config,
                        ),
                        print_env=False,
                        build_optimizer=False,
                        trainable_parameter_mode="base_model",
                        allow_unvalidated_arch=allow_unvalidated_arch,
                    ),
                )
        if runtime is None:
            raise RuntimeError("packing invariance did not acquire a Megatron runtime")
        model_chunks = cast(list[Any], runtime.model)
        gpt_module = _locate_gpt_module(model_chunks)
        for chunk in model_chunks:
            chunk.eval()
        hooked_preprocess = gpt_module._preprocess

        for scenario_name, packed_config, deep in scenarios:
            _debug_log(
                f"scenario {scenario_name} start seq_len={packed_config.sequence_length}"
            )
            packed_tensors = _time_block(
                f"scenario {scenario_name} build_packed_tensors",
                lambda: _build_art_realistic_packed_tensors(
                    packed_config,
                    case_config.seed,
                    deep=deep,
                ),
            )
            position_ids = cast(torch.Tensor, packed_tensors["input_pos"]).cuda()
            input_ids = cast(torch.Tensor, packed_tensors["tokens"]).cuda()
            group_ids = cast(torch.Tensor, packed_tensors["group_ids"]).cuda()
            parent_ids = cast(torch.Tensor, packed_tensors["parent_ids"]).cuda()
            rotary_grouping_checked = False
            rotary_grouping_respected = True
            repeated_position_key_count = 0
            for row_index in range(int(position_ids.shape[0])):
                row_position_ids = position_ids[row_index : row_index + 1]
                row_input_ids = input_ids[row_index : row_index + 1]
                hooked_output = _time_block(
                    f"scenario {scenario_name} row={row_index} hooked_preprocess",
                    lambda: hooked_preprocess(
                        input_ids=row_input_ids,
                        position_ids=row_position_ids,
                    ),
                    device=row_input_ids.device,
                )
                row_checked = False
                row_respected = True
                row_repeated_count = 0
                rotary_outputs = _rotary_outputs_for_validation(
                    handler=runtime.model_support_handler,
                    preprocess_output=hooked_output,
                    position_ids=row_position_ids,
                    group_ids=group_ids[row_index : row_index + 1],
                    parent_ids=parent_ids[row_index : row_index + 1],
                )
                for rotary_output in rotary_outputs:
                    checked, respected, repeated_count = _rotary_grouping_check(
                        rotary_output,
                        position_ids=row_position_ids,
                    )
                    row_checked = row_checked or checked
                    row_respected = row_respected and respected
                    row_repeated_count = repeated_count
                rotary_grouping_checked = rotary_grouping_checked or row_checked
                rotary_grouping_respected = rotary_grouping_respected and row_respected
                repeated_position_key_count += row_repeated_count
                _debug_log(
                    f"scenario {scenario_name} row={row_index} "
                    f"checked={row_checked} respected={row_respected} "
                    f"repeated_keys={row_repeated_count}"
                )
            (
                completion_pair_count,
                logits_equivalent,
                logits_mean_abs_pct,
                logits_max_abs_diff,
            ) = _time_block(
                f"scenario {scenario_name} logits_equivalence_check",
                lambda: _logits_equivalence_check(
                    model=model_chunks[0],
                    handler=runtime.model_support_handler,
                    provider=runtime.provider,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    group_ids=group_ids,
                    parent_ids=parent_ids,
                    mean_abs_pct_limit=mean_abs_pct_limit,
                ),
                device=input_ids.device,
            )
            matched = (
                repeated_position_key_count > 0
                and completion_pair_count > 0
                and rotary_grouping_checked
                and rotary_grouping_respected
                and logits_equivalent
            )
            _debug_log(
                f"scenario {scenario_name} done matched={matched} "
                f"pairs={completion_pair_count} logits_equivalent={logits_equivalent} "
                f"logits_mean_abs_pct={logits_mean_abs_pct:.6f} "
                f"logits_max_abs_diff={logits_max_abs_diff:.6f}"
            )
            report.scenarios.append(
                PackingInvarianceScenario(
                    name=scenario_name,
                    num_sequences=int(position_ids.shape[0]),
                    sequence_length=int(position_ids.shape[1]),
                    checked_token_count=int((group_ids != -1).sum().item()),
                    prompt_family_count=_prompt_family_count(
                        group_ids.cpu(),
                        parent_ids.cpu(),
                    ),
                    max_tree_depth=_max_tree_depth(
                        group_ids.cpu(),
                        parent_ids.cpu(),
                    ),
                    repeated_position_key_count=repeated_position_key_count,
                    rotary_grouping_checked=rotary_grouping_checked,
                    rotary_grouping_respected=rotary_grouping_respected,
                    completion_pair_count=completion_pair_count,
                    logits_equivalent=logits_equivalent,
                    logits_mean_abs_pct=logits_mean_abs_pct,
                    logits_mean_abs_pct_limit=mean_abs_pct_limit,
                    logits_max_abs_diff=logits_max_abs_diff,
                    matched=matched,
                )
            )
        del model_chunks
        torch.cuda.empty_cache()
        _debug_log("run complete; model deleted and cuda cache emptied")
    finally:
        runtime_stack.close()
        if not reused_runtime:
            del runtime
            torch.cuda.empty_cache()
            _cleanup_distributed_state()

    (output_dir / PACKING_INVARIANCE_REPORT_FILENAME).write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report


def run_packing_invariance(
    *,
    base_model: str,
    num_layers: int | None = None,
    allow_unvalidated_arch: bool = False,
    in_process: bool = False,
) -> PackingInvarianceReport:
    _debug_log(f"run start base_model={base_model} requested_num_layers={num_layers}")
    spec = get_model_support_spec(
        base_model,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    handler = get_model_support_handler_for_spec(spec)
    dtype = _dtype_for_precision(handler.correctness_precision())
    resolved_num_layers = (
        max(
            1,
            inspect_architecture(
                base_model,
                torch_dtype=dtype,
                allow_unvalidated_arch=allow_unvalidated_arch,
            ).recommended_min_layers,
        )
        if num_layers is None
        else num_layers
    )
    _debug_log(f"run resolved_num_layers={resolved_num_layers}")
    output_dir = _artifact_dir(base_model)
    report_path = output_dir / PACKING_INVARIANCE_REPORT_FILENAME
    if report_path.exists():
        report_path.unlink()
    request = PackingInvarianceRunRequest(
        git=pinned_git_state(PACKING_INVARIANCE_ARTIFACT_SUITE_NAME),
        base_model=base_model,
        num_layers=resolved_num_layers,
        output_dir=str(output_dir),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    with provider_topology_env(ORACLE_TOPOLOGY):
        runner = (
            _run_packing_invariance_in_process
            if in_process
            else _run_packing_invariance_subprocess
        )
        runner(request, output_dir)
    return PackingInvarianceReport.model_validate(_read_json(report_path))


def run_worker_cli(run_request_path: Path) -> None:
    request = PackingInvarianceRunRequest.model_validate(_read_json(run_request_path))
    _run_packing_invariance_worker(
        git=request.git,
        base_model=request.base_model,
        num_layers=request.num_layers,
        output_dir=Path(request.output_dir),
        allow_unvalidated_arch=request.allow_unvalidated_arch,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron packing invariance worker")
    parser.add_argument("--run-request", type=Path, required=True)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
