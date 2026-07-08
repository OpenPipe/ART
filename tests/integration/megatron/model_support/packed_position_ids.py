from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, cast

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_model import GPTModel
from pydantic import BaseModel, Field
import torch

from art.megatron import train as megatron_train
from art.megatron.model_support.discovery import inspect_architecture
from art.megatron.shared_prefix_state import create_shared_prefix_state

from ..artifacts import GitRepoState, pinned_git_state
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

# Qwen3.5/3.6 hybrid MoE runs show small shape-dependent logit drift between
# the single packed forward and many shorter reference forwards, even when the
# rotary grouping and shared-prefix semantics are correct. Keep the bound tight,
# but above the observed ~0.13% truncate-case jitter.
_LOGITS_MEAN_ABS_PCT_LIMIT = 0.2
_DEBUG_ENV = "ART_PACKED_POSITION_IDS_DEBUG"
PACKED_POSITION_IDS_REPORT_FILENAME = "report.json"
PACKED_POSITION_IDS_ARTIFACT_SUITE_NAME = "Megatron packed-position-id artifacts"
REPO_ROOT = Path(__file__).resolve().parents[4]
_SINGLE_ROTARY_OUTPUT_HANDLER_KEYS = frozenset(
    {
        "default_dense",
        "default_moe",
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
    path = root / _slugify(base_model) / "packed_position_ids"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _debug_enabled() -> bool:
    value = os.environ.get(_DEBUG_ENV, "")
    return value not in ("", "0", "false", "False")


def _debug_log(message: str) -> None:
    if _debug_enabled():
        print(f"[packed_position_ids] {message}", flush=True)


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
    raise RuntimeError("Failed to locate GPTModel for packed position id validation")


class PackedPositionIdScenario(BaseModel):
    name: str
    num_sequences: int
    sequence_length: int
    checked_token_count: int
    prompt_family_count: int
    repeated_position_key_count: int
    rotary_grouping_checked: bool
    rotary_grouping_respected: bool
    completion_pair_count: int
    logits_equivalent: bool
    logits_mean_abs_pct: float
    logits_max_abs_diff: float
    matched: bool


class PackedPositionIdsReport(BaseModel):
    git: GitRepoState
    base_model: str
    output_dir: str
    num_layers: int
    scenarios: list[PackedPositionIdScenario] = Field(default_factory=list)


class PackedPositionIdsRunRequest(BaseModel):
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
    handler_key: str,
    preprocess_output: Any,
) -> tuple[torch.Tensor | None, ...]:
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
) -> dict[str, Any]:
    if config.num_sequences <= 1:
        raise ValueError("num_sequences must be greater than 1")
    if config.prefill_tokens < 2:
        raise ValueError(
            "prefill_tokens must be at least 2 to build ART-style branch context"
        )
    if config.sequence_length < 3:
        raise ValueError(
            "sequence_length must leave room for shared prompt, branch context, "
            "and at least one trainable token"
        )

    shape = (config.num_sequences, config.sequence_length)
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(shape, dtype=torch.long)
    group_ids = torch.full(shape, -1, dtype=torch.long)
    parent_ids = torch.full(shape, -1, dtype=torch.long)
    input_pos = torch.zeros(shape, dtype=torch.long)
    assistant_mask = torch.zeros(shape, dtype=torch.bool)
    logprobs = torch.full(shape, float("nan"), dtype=torch.float32)
    advantages = torch.zeros(shape, dtype=torch.float32)
    weights = torch.zeros(shape, dtype=torch.float32)

    first_trainable_pos = max(2, min(config.sequence_length - 1, config.prefill_tokens))
    shared_prompt_length = first_trainable_pos - 1
    max_completion_tokens = max(1, config.sequence_length - first_trainable_pos)
    base_completion_tokens = max(1, min(config.decode_tokens, max_completion_tokens))
    jitter_width = min(config.decode_tokens_jitter, max_completion_tokens - 1)
    token_low = 10
    token_span = max(1, config.vocab_high - token_low)

    def _sample_completion_length() -> int:
        if jitter_width > 0:
            jitter = int(
                torch.randint(
                    low=-jitter_width,
                    high=jitter_width + 1,
                    size=(1,),
                    generator=generator,
                    dtype=torch.long,
                ).item()
            )
        else:
            jitter = 0
        return max(1, min(max_completion_tokens, base_completion_tokens + jitter))

    def _sample_token_block(length: int) -> torch.Tensor:
        return torch.randint(
            low=token_low,
            high=config.vocab_high,
            size=(length,),
            dtype=torch.long,
            generator=generator,
        )

    def _sample_logprob_block(length: int) -> torch.Tensor:
        return (
            torch.randn((length,), generator=generator, dtype=torch.float32) * 0.25
            - 1.75
        )

    def _sample_advantage_value() -> float:
        return float(
            (torch.randn((1,), generator=generator, dtype=torch.float32) * 0.5).item()
        )

    def _write_prompt(
        sequence_index: int,
        cursor: int,
        prompt_group_id: int,
    ) -> tuple[int, int]:
        prompt_tokens = _sample_token_block(first_trainable_pos)
        prompt_end = cursor + shared_prompt_length
        tokens[sequence_index, cursor:prompt_end] = prompt_tokens[:shared_prompt_length]
        group_ids[sequence_index, cursor:prompt_end] = prompt_group_id
        parent_ids[sequence_index, cursor:prompt_end] = prompt_group_id
        input_pos[sequence_index, cursor:prompt_end] = torch.arange(
            shared_prompt_length,
            dtype=torch.long,
        )
        return prompt_end, int(prompt_tokens[shared_prompt_length].item())

    def _write_branch(
        sequence_index: int,
        cursor: int,
        completion_group_id: int,
        prompt_group_id: int,
        context_token: int,
        completion_length: int,
    ) -> int:
        branch_end = cursor + 1 + completion_length
        tokens[sequence_index, cursor] = context_token
        tokens[sequence_index, cursor + 1 : branch_end] = _sample_token_block(
            completion_length
        )
        group_ids[sequence_index, cursor:branch_end] = completion_group_id
        parent_ids[sequence_index, cursor:branch_end] = prompt_group_id
        input_pos[sequence_index, cursor:branch_end] = torch.arange(
            shared_prompt_length,
            shared_prompt_length + 1 + completion_length,
            dtype=torch.long,
        )
        trainable_start = cursor + 1
        assistant_mask[sequence_index, trainable_start:branch_end] = True
        logprobs[sequence_index, trainable_start:branch_end] = _sample_logprob_block(
            completion_length
        )
        advantages[sequence_index, trainable_start:branch_end] = (
            _sample_advantage_value()
        )
        weights[sequence_index, trainable_start:branch_end] = 1.0 / completion_length
        return branch_end

    for sequence_index in range(config.num_sequences):
        cursor = 0
        next_group_id = 0
        while cursor < config.sequence_length:
            prompt_group_id = next_group_id
            next_group_id += 1
            completion_lengths = [
                _sample_completion_length()
                for _ in range(config.completion_branches_per_prefix)
            ]
            remaining = config.sequence_length - cursor
            if remaining <= shared_prompt_length + 1:
                break

            if config.packing_mode == "stop_early":
                included_completion_lengths = list(completion_lengths)
                while included_completion_lengths and (
                    shared_prompt_length
                    + sum(1 + length for length in included_completion_lengths)
                    > remaining
                ):
                    included_completion_lengths.pop()
                if not included_completion_lengths:
                    break

                cursor, context_token = _write_prompt(
                    sequence_index,
                    cursor,
                    prompt_group_id,
                )
                for completion_length in included_completion_lengths:
                    completion_group_id = next_group_id
                    next_group_id += 1
                    cursor = _write_branch(
                        sequence_index,
                        cursor,
                        completion_group_id,
                        prompt_group_id,
                        context_token,
                        completion_length,
                    )
                continue

            cursor, context_token = _write_prompt(
                sequence_index,
                cursor,
                prompt_group_id,
            )
            for completion_length in completion_lengths:
                remaining = config.sequence_length - cursor
                if remaining <= 1:
                    break
                completion_take = min(completion_length, remaining - 1)
                completion_group_id = next_group_id
                next_group_id += 1
                cursor = _write_branch(
                    sequence_index,
                    cursor,
                    completion_group_id,
                    prompt_group_id,
                    context_token,
                    completion_take,
                )

    half = config.num_sequences // 2
    if half > 0 and config.num_sequences % 2 == 0:
        valid_lengths = (group_ids != -1).sum(dim=1)
        for pair_index in range(half):
            left_index = pair_index
            right_index = pair_index + half
            left_valid = int(valid_lengths[left_index].item())
            right_valid = int(valid_lengths[right_index].item())
            if left_valid != right_valid or left_valid == 0:
                continue
            if torch.equal(
                tokens[left_index, :left_valid],
                tokens[right_index, :right_valid],
            ):
                tokens[right_index, 0] = (
                    (tokens[right_index, 0] - token_low + 1) % token_span
                ) + token_low

    weights = torch.where(assistant_mask, weights, torch.zeros_like(weights))
    if bool(assistant_mask.any().item()):
        weights[assistant_mask] /= weights[assistant_mask].mean()
        advantages = torch.where(
            assistant_mask,
            advantages,
            torch.zeros_like(advantages),
        )
        advantage_scale = (
            advantages[assistant_mask].abs() * weights[assistant_mask]
        ).mean()
        if float(advantage_scale.item()) > 0.0:
            advantages[assistant_mask] /= advantage_scale

    return {
        "tokens": tokens,
        "group_ids": group_ids,
        "parent_ids": parent_ids,
        "input_pos": input_pos,
        "assistant_mask": assistant_mask,
        "logprobs": logprobs,
        "advantages": advantages,
        "weights": weights,
        "pixel_values": [None] * config.num_sequences,
        "image_grid_thw": [None] * config.num_sequences,
    }


def _prompt_family_segments(
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    required_completion_count: int = 2,
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    families: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    valid_tokens = int((group_ids != -1).sum().item())
    cursor = 0
    while cursor < valid_tokens:
        group_id = int(group_ids[cursor].item())
        parent_id = int(parent_ids[cursor].item())
        prompt_start = cursor
        while cursor < valid_tokens and int(group_ids[cursor].item()) == group_id:
            cursor += 1
        prompt_end = cursor
        if group_id != parent_id:
            continue
        completions: list[tuple[int, int]] = []
        while cursor < valid_tokens:
            completion_group_id = int(group_ids[cursor].item())
            completion_parent_id = int(parent_ids[cursor].item())
            if completion_parent_id != group_id or completion_group_id == group_id:
                break
            completion_start = cursor
            while (
                cursor < valid_tokens
                and int(group_ids[cursor].item()) == completion_group_id
            ):
                cursor += 1
            completions.append((completion_start, cursor))
        if len(completions) >= required_completion_count:
            families.append(((prompt_start, prompt_end), completions))
    return families


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
    for row_index in range(int(input_ids.shape[0])):
        row_group_ids = group_ids[row_index : row_index + 1]
        row_parent_ids = parent_ids[row_index : row_index + 1]
        families = _prompt_family_segments(row_group_ids[0], row_parent_ids[0])
        if not families:
            _debug_log(f"logits_check row={row_index} skipped no prompt family")
            continue
        row_input_ids = input_ids[row_index : row_index + 1]
        row_position_ids = position_ids[row_index : row_index + 1]
        packed_bias = create_shared_prefix_state(
            group_ids=row_group_ids,
            parent_ids=row_parent_ids,
            input_pos=row_position_ids,
            sliding_windows=sliding_windows,
            build_gdn_execution_spec=bool(
                getattr(handler, "build_gdn_execution_spec", False)
            ),
            model_support_handler=handler,
            attention_head_dim=getattr(provider, "kv_channels", None),
            attention_value_head_dim=getattr(provider, "kv_channels", None),
        )
        _debug_log(f"logits_check row={row_index} families={len(families)}")
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
        for family_index, (prompt_segment, completion_segments) in enumerate(families):
            prompt_start, prompt_end = prompt_segment
            _debug_log(
                "logits_check row="
                f"{row_index} family={family_index} "
                f"prompt=({prompt_start},{prompt_end}) "
                f"completions={completion_segments}"
            )
            for completion_index, (completion_start, completion_end) in enumerate(
                completion_segments
            ):
                reference_input_ids = torch.cat(
                    (
                        row_input_ids[:, prompt_start:prompt_end],
                        row_input_ids[:, completion_start:completion_end],
                    ),
                    dim=1,
                )
                reference_position_ids = torch.cat(
                    (
                        row_position_ids[:, prompt_start:prompt_end],
                        row_position_ids[:, completion_start:completion_end],
                    ),
                    dim=1,
                )
                reference_group_ids = torch.zeros_like(reference_input_ids)
                reference_parent_ids = torch.zeros_like(reference_input_ids)
                reference_bias = create_shared_prefix_state(
                    group_ids=reference_group_ids,
                    parent_ids=reference_parent_ids,
                    input_pos=reference_position_ids,
                    sliding_windows=sliding_windows,
                    build_gdn_execution_spec=bool(
                        getattr(handler, "build_gdn_execution_spec", False)
                    ),
                    model_support_handler=handler,
                    attention_head_dim=getattr(provider, "kv_channels", None),
                    attention_value_head_dim=getattr(provider, "kv_channels", None),
                )
                _debug_log(
                    "logits_check row="
                    f"{row_index} family={family_index} "
                    f"completion={completion_index} "
                    f"segment=({completion_start},{completion_end}) "
                    f"reference_seq={int(reference_input_ids.shape[1])}"
                )
                reference_logits = _time_block(
                    (
                        f"logits_check row={row_index} "
                        f"family={family_index} "
                        f"completion={completion_index} reference_forward"
                    ),
                    lambda: _run_logits(
                        model=model,
                        handler=handler,
                        input_ids=reference_input_ids,
                        position_ids=reference_position_ids,
                        attention_bias=reference_bias,
                    ),
                    device=reference_input_ids.device,
                )
                if completion_end - completion_start < 2:
                    continue
                packed_completion_logits = packed_logits[
                    :,
                    completion_start : completion_end - 1,
                    :,
                ]
                reference_completion_logits = reference_logits[
                    :,
                    prompt_end - prompt_start : -1,
                    :,
                ]
                diff = (packed_completion_logits - reference_completion_logits).abs()
                logits_abs_sum += float(diff.sum().item())
                logits_ref_abs_sum += float(
                    reference_completion_logits.abs().sum().item()
                )
                logits_numel += int(diff.numel())
                logits_max_abs_diff = max(
                    logits_max_abs_diff,
                    float(diff.max().item()),
                )
                completion_pair_count += 1
                _debug_log(
                    "logits_check row="
                    f"{row_index} family={family_index} "
                    f"completion={completion_index} "
                    f"max_abs_diff={float(diff.max().item()):.6f}"
                )
    if completion_pair_count > 0:
        mean_abs = logits_abs_sum / max(logits_numel, 1)
        typical_abs = logits_ref_abs_sum / max(logits_numel, 1)
        logits_mean_abs_pct = (mean_abs / (typical_abs + 1e-12)) * 100.0
        logits_equivalent = logits_mean_abs_pct <= _LOGITS_MEAN_ABS_PCT_LIMIT
        _debug_log(
            "logits_check done "
            f"pairs={completion_pair_count} "
            f"equivalent={logits_equivalent} "
            f"mean_abs_pct={logits_mean_abs_pct:.6f} "
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


def _run_packed_position_ids_subprocess(
    request: PackedPositionIdsRunRequest,
    output_dir: Path,
) -> None:
    request_path = output_dir / "run_request.json"
    _write_json(request_path, request.model_dump(mode="json"))
    worker_cwd = REPO_ROOT / "tests"
    command = [
        sys.executable,
        "-m",
        "integration.megatron.model_support.packed_position_ids",
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
            "Packed position ids worker failed with exit code "
            f"{run.returncode}.\n{tail}"
        )


def _run_packed_position_ids_worker(
    *,
    git: GitRepoState,
    base_model: str,
    num_layers: int,
    output_dir: Path,
    allow_unvalidated_arch: bool = False,
) -> PackedPositionIdsReport:
    _debug_log(f"run start base_model={base_model} num_layers={num_layers}")
    _reset_vllm_compile_overrides()
    scenarios = [
        (
            "stop_early",
            PackedTensorConfig(
                num_sequences=4,
                sequence_length=_env_int(
                    "ART_PACKED_POSITION_IDS_STOP_EARLY_SEQUENCE_LENGTH", 2048
                ),
                prefill_tokens=_env_int(
                    "ART_PACKED_POSITION_IDS_STOP_EARLY_PREFILL_TOKENS", 256
                ),
                completion_branches_per_prefix=2,
                decode_tokens=_env_int(
                    "ART_PACKED_POSITION_IDS_STOP_EARLY_DECODE_TOKENS", 128
                ),
                decode_tokens_jitter=_env_int(
                    "ART_PACKED_POSITION_IDS_STOP_EARLY_DECODE_TOKENS_JITTER", 32
                ),
                packing_mode="stop_early",
            ),
        ),
        (
            "truncate",
            PackedTensorConfig(
                num_sequences=4,
                sequence_length=_env_int(
                    "ART_PACKED_POSITION_IDS_TRUNCATE_SEQUENCE_LENGTH", 2048
                ),
                prefill_tokens=_env_int(
                    "ART_PACKED_POSITION_IDS_TRUNCATE_PREFILL_TOKENS", 256
                ),
                completion_branches_per_prefix=2,
                decode_tokens=_env_int(
                    "ART_PACKED_POSITION_IDS_TRUNCATE_DECODE_TOKENS", 128
                ),
                decode_tokens_jitter=_env_int(
                    "ART_PACKED_POSITION_IDS_TRUNCATE_DECODE_TOKENS_JITTER", 32
                ),
                packing_mode="truncate",
            ),
        ),
    ]
    report = PackedPositionIdsReport(
        git=git,
        base_model=base_model,
        output_dir=str(output_dir),
        num_layers=num_layers,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for packed position id validation")

    case_config = OracleCaseConfig(
        base_model=base_model,
        precision="fp32",
        num_layers=num_layers,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    runtime: megatron_train.TrainingRuntime | None = None
    flex_patch_stack = ExitStack()
    flex_patch_stack.enter_context(
        _apply_requested_flex_backend_patch(TEST_DEFAULT_FLEX_BACKEND)
    )
    flex_patch_stack.enter_context(
        _apply_test_flex_inner_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
    )
    flex_patch_stack.enter_context(
        _apply_test_attention_full_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
    )
    try:
        with provider_topology_env(ORACLE_TOPOLOGY):
            runtime = _time_block(
                "build_training_runtime",
                lambda: megatron_train.build_training_runtime(
                    model_identifier=base_model,
                    provider_torch_dtype=torch.float32,
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
        model_chunks = cast(list[Any], runtime.model)
        gpt_module = _locate_gpt_module(model_chunks)
        for chunk in model_chunks:
            chunk.eval()
        hooked_preprocess = gpt_module._preprocess

        for scenario_name, packed_config in scenarios:
            _debug_log(
                f"scenario {scenario_name} start seq_len={packed_config.sequence_length}"
            )
            packed_tensors = _time_block(
                f"scenario {scenario_name} build_packed_tensors",
                lambda: _build_art_realistic_packed_tensors(
                    packed_config,
                    case_config.seed,
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
                    handler_key=runtime.model_support_handler.key,
                    preprocess_output=hooked_output,
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
                PackedPositionIdScenario(
                    name=scenario_name,
                    num_sequences=int(position_ids.shape[0]),
                    sequence_length=int(position_ids.shape[1]),
                    checked_token_count=int((group_ids != -1).sum().item()),
                    prompt_family_count=_prompt_family_count(
                        group_ids.cpu(),
                        parent_ids.cpu(),
                    ),
                    repeated_position_key_count=repeated_position_key_count,
                    rotary_grouping_checked=rotary_grouping_checked,
                    rotary_grouping_respected=rotary_grouping_respected,
                    completion_pair_count=completion_pair_count,
                    logits_equivalent=logits_equivalent,
                    logits_mean_abs_pct=logits_mean_abs_pct,
                    logits_max_abs_diff=logits_max_abs_diff,
                    matched=matched,
                )
            )
        del model_chunks
        torch.cuda.empty_cache()
        _debug_log("run complete; model deleted and cuda cache emptied")
    finally:
        flex_patch_stack.close()
        del runtime
        torch.cuda.empty_cache()
        _cleanup_distributed_state()

    (output_dir / PACKED_POSITION_IDS_REPORT_FILENAME).write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report


def run_packed_position_ids(
    *,
    base_model: str,
    num_layers: int | None = None,
    allow_unvalidated_arch: bool = False,
) -> PackedPositionIdsReport:
    _debug_log(f"run start base_model={base_model} requested_num_layers={num_layers}")
    resolved_num_layers = (
        max(
            1,
            inspect_architecture(
                base_model,
                torch_dtype=torch.float32,
                allow_unvalidated_arch=allow_unvalidated_arch,
            ).recommended_min_layers,
        )
        if num_layers is None
        else num_layers
    )
    _debug_log(f"run resolved_num_layers={resolved_num_layers}")
    output_dir = _artifact_dir(base_model)
    report_path = output_dir / PACKED_POSITION_IDS_REPORT_FILENAME
    if report_path.exists():
        report_path.unlink()
    request = PackedPositionIdsRunRequest(
        git=pinned_git_state(PACKED_POSITION_IDS_ARTIFACT_SUITE_NAME),
        base_model=base_model,
        num_layers=resolved_num_layers,
        output_dir=str(output_dir),
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    with provider_topology_env(ORACLE_TOPOLOGY):
        _run_packed_position_ids_subprocess(request, output_dir)
    return PackedPositionIdsReport.model_validate(_read_json(report_path))


def run_worker_cli(run_request_path: Path) -> None:
    request = PackedPositionIdsRunRequest.model_validate(_read_json(run_request_path))
    _run_packed_position_ids_worker(
        git=request.git,
        base_model=request.base_model,
        num_layers=request.num_layers,
        output_dir=Path(request.output_dir),
        allow_unvalidated_arch=request.allow_unvalidated_arch,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron packed position ids worker")
    parser.add_argument("--run-request", type=Path, required=True)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    run_worker_cli(args.run_request)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
