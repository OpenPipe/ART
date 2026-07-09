from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn.attention.flex_attention import AuxRequest, BlockMask
from torch.nn.attention.flex_attention import create_block_mask as torch_block_mask
import typer

from art.megatron.context_parallel.block_mask import (
    build_block_mask_from_context,
    prepare_block_mask_context,
)
from art.megatron.context_parallel.builder import build_shared_prefix_attention_spec
from art.megatron.context_parallel.executor import _build_stage_execution_spec
from art.megatron.context_parallel.runtime import (
    _RUNTIME_PLAN_CACHE,
    get_or_build_runtime_plan,
)
from art.megatron.context_parallel.types import (
    ContextParallelConfig,
    FlexMaskSpec,
    ParallelTopology,
    StageExecutionSpec,
    StagePlan,
)
from art.megatron.flex_attn.compiled import (
    normalize_sparse_block_size,
    sparse_compiled_flex_attention,
)
from art.megatron.shared_prefix_packing import SharedPrefixPack, pack_shared_prefixes


def main(
    workload: str = "austin_198k",
    max_depth: int = 1,
    cp_size: int = 4,
    block_size: int = 128,
    prefix_families: int = 4,
    prefix_len: int = 1024,
    mid_prefixes_per_family: int = 1,
    mid_prefix_len: int = 0,
    branches_per_prefix: int = 8,
    completion_len: int = 128,
    warmup: int = 3,
    repeat: int = 10,
    shape_variants: int = 4,
    validate_torch: bool = True,
    validate_torch_token_cap: int = 32768,
    run_flex: bool = True,
    flex_token_cap: int = 8192,
    flex_heads: int = 2,
    flex_head_dim: int = 128,
    flex_mask_variants: str = "current,causal_abs_only",
    max_block_mask_build_ms: float | None = None,
    max_cp_planning_cold_ms: float | None = None,
    output_jsonl: Path = Path(".local/trainer_rank_review/block_mask_flex.jsonl"),
) -> None:
    if warmup < 0 or repeat < 1:
        raise ValueError("warmup must be >= 0 and repeat must be >= 1")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pack = _pack_workload(
        workload=workload,
        max_depth=max_depth,
        prefix_families=prefix_families,
        prefix_len=prefix_len,
        mid_prefixes_per_family=mid_prefixes_per_family,
        mid_prefix_len=mid_prefix_len,
        branches_per_prefix=branches_per_prefix,
        completion_len=completion_len,
    )
    spec = build_shared_prefix_attention_spec(
        group_ids=pack.group_ids,
        parent_ids=pack.parent_ids,
    )
    config = ContextParallelConfig(block_size=block_size)
    topology = ParallelTopology(cp=cp_size)
    base = {
        "workload": workload,
        "max_depth": max_depth,
        "cp_size": cp_size,
        "block_size": block_size,
        "packed_tokens": int(pack.tokens.numel()),
        "logical_tokens": _logical_tokens(pack),
        "warmup": warmup,
        "repeat": repeat,
        "validate_torch": validate_torch,
        "validate_torch_token_cap": validate_torch_token_cap,
    }

    plan, plan_ms = _bench_cpu(
        lambda: _build_cp_plan(pack, spec, topology, config),
        warmup=warmup,
        repeat=repeat,
        before_each=_RUNTIME_PLAN_CACHE.clear,
    )
    _write(
        output_jsonl,
        {
            **base,
            "case": "cp_planning_cold",
            "ms": plan_ms,
            **_plan_stats(plan),
        },
    )

    cached_plan, cached_plan_ms = _bench_cpu(
        lambda: _build_cp_plan(pack, spec, topology, config),
        warmup=warmup,
        repeat=repeat,
    )
    _write(
        output_jsonl,
        {
            **base,
            "case": "cp_planning_cached",
            "ms": cached_plan_ms,
            **_plan_stats(cached_plan),
        },
    )

    stage_masks, mask_ms = _bench_cpu(
        lambda: _build_stage_masks(pack, plan, config),
        warmup=warmup,
        repeat=repeat,
    )
    masks = tuple(mask for mask, _ in stage_masks)
    torch_validation_skipped = _torch_validation_skip_reason(
        validate_torch=validate_torch,
        packed_tokens=int(pack.tokens.numel()),
        token_cap=validate_torch_token_cap,
    )
    if torch_validation_skipped is None:
        for mask, slices in stage_masks:
            _assert_matches_torch_block_mask(mask, slices=slices)
    _write(
        output_jsonl,
        {
            **base,
            "case": "block_mask_build",
            "ms": mask_ms,
            "torch_validation_skipped": torch_validation_skipped,
            **_mask_stats(masks),
        },
    )
    _check_threshold("block_mask_build", mask_ms, max_block_mask_build_ms)
    _check_threshold("cp_planning_cold", plan_ms, max_cp_planning_cold_ms)

    if run_flex:
        for record in _flex_records(
            pack,
            plan,
            config,
            warmup=warmup,
            repeat=repeat,
            token_cap=flex_token_cap,
            heads=flex_heads,
            head_dim=flex_head_dim,
            variants=_csv_values(flex_mask_variants),
        ):
            _write(output_jsonl, {**base, **record})

    for variant in range(shape_variants):
        variant_pack = _pack_workload(
            workload="regular",
            max_depth=max_depth,
            prefix_families=prefix_families,
            prefix_len=prefix_len + variant * 17,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len + variant * 3,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len + variant * 11,
        )
        variant_spec = build_shared_prefix_attention_spec(
            group_ids=variant_pack.group_ids,
            parent_ids=variant_pack.parent_ids,
        )
        variant_plan, variant_plan_ms = _bench_cpu(
            lambda pack=variant_pack, spec=variant_spec: _build_cp_plan(
                pack,
                spec,
                topology,
                config,
            ),
            warmup=0,
            repeat=1,
            before_each=_RUNTIME_PLAN_CACHE.clear,
        )
        variant_stage_masks, variant_mask_ms = _bench_cpu(
            lambda pack=variant_pack, plan=variant_plan: _build_stage_masks(
                pack,
                plan,
                config,
            ),
            warmup=0,
            repeat=1,
        )
        variant_masks = tuple(mask for mask, _ in variant_stage_masks)
        variant_torch_validation_skipped = _torch_validation_skip_reason(
            validate_torch=validate_torch,
            packed_tokens=int(variant_pack.tokens.numel()),
            token_cap=validate_torch_token_cap,
        )
        if variant_torch_validation_skipped is None:
            for mask, slices in variant_stage_masks:
                _assert_matches_torch_block_mask(mask, slices=slices)
        _write(
            output_jsonl,
            {
                **base,
                "case": "shape_variant",
                "variant": variant,
                "variant_packed_tokens": int(variant_pack.tokens.numel()),
                "variant_logical_tokens": _logical_tokens(variant_pack),
                "cp_planning_ms": variant_plan_ms,
                "block_mask_build_ms": variant_mask_ms,
                "torch_validation_skipped": variant_torch_validation_skipped,
                **_plan_stats(variant_plan),
                **_mask_stats(variant_masks),
            },
        )

    print(f"wrote review perf records to {output_jsonl}", flush=True)


def _pack_workload(
    *,
    workload: str,
    max_depth: int,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> SharedPrefixPack:
    sequences = (
        _austin_sequences()
        if workload == "austin_198k"
        else _austin_varied_sequences()
        if workload == "austin_varied"
        else _regular_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    )
    return pack_shared_prefixes(sequences, max_depth=max_depth)


def _austin_sequences() -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.cat(
            (
                _tokens(family * 10_000_019, 5000),
                _tokens(family * 10_000_019 + branch * 1009 + 17, 100),
            )
        )
        for family in range(30)
        for branch in range(16)
    )


def _austin_varied_sequences() -> tuple[torch.Tensor, ...]:
    sequences: list[torch.Tensor] = []
    for family in range(30):
        family_base = family * 10_000_019
        prefix_len = 4500 + ((family * 137) % 1001)
        root = _tokens(family_base, prefix_len)
        branch_count = 10 + ((family * 7) % 13)
        for branch in range(branch_count):
            completion_len = 32 + ((family * 19 + branch * 23) % 145)
            sequences.append(
                torch.cat(
                    (
                        root,
                        _tokens(
                            family_base + branch * 1009 + 17,
                            completion_len,
                        ),
                    )
                )
            )
    return tuple(sequences)


def _regular_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> tuple[torch.Tensor, ...]:
    sequences = []
    for family in range(max(1, prefix_families)):
        family_base = family * 10_000_019
        root = _tokens(family_base, max(1, prefix_len))
        for mid in range(max(1, mid_prefixes_per_family)):
            mid_prefix = _tokens(
                family_base + 1_000_003 + mid * 100_003,
                max(0, mid_prefix_len),
            )
            prefix = torch.cat((root, mid_prefix))
            for branch in range(max(1, branches_per_prefix)):
                sequences.append(
                    torch.cat(
                        (
                            prefix,
                            _tokens(
                                family_base + mid * 100_003 + branch * 1009 + 17,
                                max(1, completion_len),
                            ),
                        )
                    )
                )
    return tuple(sequences)


def _tokens(offset: int, length: int) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) + offset) % 32_000 + 100


def _build_cp_plan(
    pack: SharedPrefixPack,
    spec: object,
    topology: ParallelTopology,
    config: ContextParallelConfig,
) -> object:
    return get_or_build_runtime_plan(
        spec,
        topology=topology,
        config=config,
        original_seq_len=int(pack.tokens.numel()),
    )


def _build_stage_masks(
    pack: SharedPrefixPack,
    plan: object,
    config: ContextParallelConfig,
) -> tuple[tuple[BlockMask, tuple[object, ...]], ...]:
    masks = []
    context = prepare_block_mask_context(
        group_ids=pack.group_ids[0],
        parent_ids=pack.parent_ids[0],
    )
    for rank_plan in plan:
        for stage in rank_plan.stage_plans:
            if stage.mask_metadata is None:
                continue
            execution_spec = _stage_execution_spec(stage, config)
            mask_metadata = execution_spec.mask_metadata or stage.mask_metadata
            if mask_metadata is None:
                continue
            mask = build_block_mask_from_context(
                FlexMaskSpec(
                    q_len=execution_spec.q_len,
                    k_len=execution_spec.k_len,
                    block_size=_sparse_block_size(config),
                    slices=stage.slices,
                    exact_mask=mask_metadata,
                ),
                context=context,
                device=torch.device("cpu"),
                validate=False,
            )
            if mask is not None:
                masks.append((mask, tuple(stage.slices)))
    return tuple(masks)


def _flex_records(
    pack: SharedPrefixPack,
    plan: object,
    config: ContextParallelConfig,
    *,
    warmup: int,
    repeat: int,
    token_cap: int,
    heads: int,
    head_dim: int,
    variants: Sequence[str],
) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return [{"case": "flex_attention_fwd_bwd", "skipped": "cuda_unavailable"}]
    device = torch.device("cuda")
    stage_cases = _build_stage_flex_cases(
        pack,
        plan,
        config,
        device=device,
    )
    if not stage_cases:
        return [{"case": "flex_attention_fwd_bwd", "skipped": "no_stage_masks"}]
    largest_stage = max(max(case.q_len, case.k_len) for case in stage_cases)
    if int(largest_stage) > int(token_cap):
        return [
            {
                "case": "flex_attention_fwd_bwd",
                "skipped": "stage_tokens_exceed_flex_token_cap",
                "flex_token_cap": int(token_cap),
                "largest_stage_tokens": int(largest_stage),
            }
        ]
    records: list[dict[str, object]] = []
    base_tensors = _stage_tensors(
        stage_cases,
        heads=heads,
        head_dim=head_dim,
        device=device,
    )
    for variant in variants:
        block_masks = []
        try:
            block_masks = [
                _stage_variant_block_mask(case, variant, device=device)
                for case in stage_cases
            ]
        except Exception as exc:
            records.append(
                {
                    "case": "flex_attention_fwd_bwd",
                    "flex_mask_variant": variant,
                    "compile_error": type(exc).__name__,
                    "compile_error_message": str(exc).splitlines()[0][:500],
                    "flex_heads": heads,
                    "flex_head_dim": head_dim,
                }
            )
            continue
        qkv = [
            (
                q.detach().clone().requires_grad_(True),
                k.detach().clone().requires_grad_(True),
                v.detach().clone().requires_grad_(True),
            )
            for q, k, v in base_tensors
        ]

        def step() -> None:
            loss = torch.zeros((), device=device, dtype=torch.float32)
            for (q, k, v), block_mask in zip(qkv, block_masks, strict=True):
                q.grad = None
                k.grad = None
                v.grad = None
                out, _aux = sparse_compiled_flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                    scale=float(head_dim) ** -0.5,
                    enable_gqa=False,
                    return_aux=AuxRequest(lse=True),
                )
                loss = loss + out.float().sum()
            loss.backward()

        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            first_started = time.perf_counter()
            step()
            torch.cuda.synchronize()
            first_call_ms = round((time.perf_counter() - first_started) * 1000.0, 3)
            ms = _bench_cuda(step, warmup=warmup, repeat=repeat)
        except Exception as exc:
            torch.cuda.empty_cache()
            records.append(
                {
                    "case": "flex_attention_fwd_bwd",
                    "flex_mask_variant": variant,
                    "compile_error": type(exc).__name__,
                    "compile_error_message": str(exc).splitlines()[0][:500],
                    "flex_heads": heads,
                    "flex_head_dim": head_dim,
                    **_stage_flex_stats(stage_cases),
                }
            )
            continue
        records.append(
            {
                "case": "flex_attention_fwd_bwd",
                "flex_mask_variant": variant,
                "first_call_ms": first_call_ms,
                "ms": ms,
                "packed_tok_s": round(int(pack.tokens.numel()) * 1000.0 / ms, 3),
                "flex_heads": heads,
                "flex_head_dim": head_dim,
                **_stage_flex_stats(stage_cases),
                "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
        )
    return records


@dataclass(frozen=True)
class _StageFlexCase:
    rank: int
    stage_index: int
    q_len: int
    k_len: int
    logical_q_len: int
    logical_k_len: int
    block_mask: BlockMask
    q_abs: np.ndarray
    k_abs: np.ndarray


def _build_stage_flex_cases(
    pack: SharedPrefixPack,
    plan: object,
    config: ContextParallelConfig,
    *,
    device: torch.device,
) -> tuple[_StageFlexCase, ...]:
    cases: list[_StageFlexCase] = []
    context = prepare_block_mask_context(
        group_ids=pack.group_ids[0],
        parent_ids=pack.parent_ids[0],
    )
    for rank_plan in plan:
        for stage in rank_plan.stage_plans:
            if stage.mask_metadata is None:
                continue
            execution_spec = _stage_execution_spec(stage, config)
            mask_metadata = execution_spec.mask_metadata or stage.mask_metadata
            if mask_metadata is None:
                continue
            mask = build_block_mask_from_context(
                FlexMaskSpec(
                    q_len=execution_spec.q_len,
                    k_len=execution_spec.k_len,
                    block_size=_sparse_block_size(config),
                    slices=stage.slices,
                    exact_mask=mask_metadata,
                ),
                context=context,
                device=device,
                validate=False,
            )
            if mask is None:
                continue
            q_abs = (
                mask_metadata.q_token_indices.detach()
                .to(device="cpu", dtype=torch.int64)
                .reshape(-1)
                .numpy()
            )
            k_abs = (
                mask_metadata.k_token_indices.detach()
                .to(device="cpu", dtype=torch.int64)
                .reshape(-1)
                .numpy()
            )
            cases.append(
                _StageFlexCase(
                    rank=int(rank_plan.rank),
                    stage_index=int(stage.stage_index),
                    q_len=int(execution_spec.q_len),
                    k_len=int(execution_spec.k_len),
                    logical_q_len=int(stage.q_len),
                    logical_k_len=int(stage.k_len),
                    block_mask=mask,
                    q_abs=q_abs,
                    k_abs=k_abs,
                )
            )
    return tuple(cases)


def _stage_tensors(
    cases: Sequence[_StageFlexCase],
    *,
    heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator(device=device).manual_seed(17)
    tensors = []
    for case in cases:
        q_shape = (1, int(heads), int(case.q_len), int(head_dim))
        k_shape = (1, int(heads), int(case.k_len), int(head_dim))
        tensors.append(
            (
                torch.randn(
                    q_shape, device=device, dtype=torch.bfloat16, generator=generator
                ),
                torch.randn(
                    k_shape, device=device, dtype=torch.bfloat16, generator=generator
                ),
                torch.randn(
                    k_shape, device=device, dtype=torch.bfloat16, generator=generator
                ),
            )
        )
    return tuple(tensors)


def _stage_variant_block_mask(
    case: _StageFlexCase,
    variant: str,
    *,
    device: torch.device,
) -> BlockMask:
    if variant == "current":
        return case.block_mask
    q_abs = torch.as_tensor(case.q_abs, device=device, dtype=torch.int64)
    k_abs = torch.as_tensor(case.k_abs, device=device, dtype=torch.int64)
    if variant == "causal_abs_only":

        def mask_mod(batch_idx, head_idx, query_idx, kv_idx):
            del batch_idx, head_idx
            return q_abs[query_idx] >= k_abs[kv_idx]

        return _replace_block_mask_mod(case.block_mask, mask_mod)
    raise ValueError(f"unknown flex_mask_variant {variant!r}")


def _stage_flex_stats(cases: Sequence[_StageFlexCase]) -> dict[str, object]:
    return {
        "flex_stage_count": len(cases),
        "flex_stage_q_tokens": sum(case.q_len for case in cases),
        "flex_stage_k_tokens": sum(case.k_len for case in cases),
        "flex_stage_logical_q_tokens": sum(case.logical_q_len for case in cases),
        "flex_stage_logical_k_tokens": sum(case.logical_k_len for case in cases),
        "flex_stage_max_q_tokens": max(case.q_len for case in cases),
        "flex_stage_max_k_tokens": max(case.k_len for case in cases),
        "flex_stage_max_logical_q_tokens": max(case.logical_q_len for case in cases),
        "flex_stage_max_logical_k_tokens": max(case.logical_k_len for case in cases),
    }


def _sparse_block_size(config: ContextParallelConfig) -> tuple[int, int]:
    return normalize_sparse_block_size(
        config.attention_sparse_block_size or config.block_size
    )


def _stage_execution_spec(
    stage: StagePlan,
    config: ContextParallelConfig,
) -> StageExecutionSpec:
    return _build_stage_execution_spec(
        stage_plan=stage,
        block_size=_sparse_block_size(config),
    )


def _replace_block_mask_mod(block_mask: BlockMask, mask_mod: object) -> BlockMask:
    return BlockMask(
        seq_lengths=block_mask.seq_lengths,
        kv_num_blocks=block_mask.kv_num_blocks,
        kv_indices=block_mask.kv_indices,
        full_kv_num_blocks=block_mask.full_kv_num_blocks,
        full_kv_indices=block_mask.full_kv_indices,
        q_num_blocks=block_mask.q_num_blocks,
        q_indices=block_mask.q_indices,
        full_q_num_blocks=block_mask.full_q_num_blocks,
        full_q_indices=block_mask.full_q_indices,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
        mask_mod=mask_mod,
    )


def _bench_cpu(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    before_each: Callable[[], object] | None = None,
) -> tuple[object, float]:
    result = None
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        result = fn()
    elapsed = []
    for _ in range(repeat):
        if before_each is not None:
            before_each()
        start = time.perf_counter()
        result = fn()
        elapsed.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return result, round(sum(elapsed) / len(elapsed), 3)


def _bench_cuda(fn: Callable[[], object], *, warmup: int, repeat: int) -> float:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return round(float(start.elapsed_time(stop)) / repeat, 3)


def _plan_stats(plan: object) -> dict[str, int]:
    stage_count = 0
    remote_stage_count = 0
    mask_stage_count = 0
    for rank_plan in plan:
        for stage in rank_plan.stage_plans:
            stage_count += 1
            remote_stage_count += int(not stage.is_local_stage)
            mask_stage_count += int(stage.mask_metadata is not None)
    return {
        "rank_count": len(plan),
        "stage_count": stage_count,
        "remote_stage_count": remote_stage_count,
        "mask_stage_count": mask_stage_count,
    }


def _mask_stats(masks: Sequence[BlockMask]) -> dict[str, int]:
    return {
        "mask_count": len(masks),
        "partial_kv_blocks": sum(_block_count(mask, "kv_num_blocks") for mask in masks),
        "full_kv_blocks": sum(
            _block_count(mask, "full_kv_num_blocks") for mask in masks
        ),
        "partial_q_blocks": sum(_block_count(mask, "q_num_blocks") for mask in masks),
        "full_q_blocks": sum(_block_count(mask, "full_q_num_blocks") for mask in masks),
    }


def _block_count(block_mask: BlockMask, name: str) -> int:
    counts = getattr(block_mask, name)
    return 0 if counts is None else int(counts.sum().item())


def _assert_matches_torch_block_mask(
    block_mask: BlockMask,
    *,
    slices: Sequence[object] = (),
) -> None:
    q_len, k_len = block_mask.seq_lengths
    reference = torch_block_mask(
        _slice_mask_mod(block_mask.mask_mod, slices),
        B=int(block_mask.kv_num_blocks.shape[0]),
        H=1,
        Q_LEN=q_len,
        KV_LEN=k_len,
        device="cpu",
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
    )
    for counts_name, indices_name in (
        ("kv_num_blocks", "kv_indices"),
        ("full_kv_num_blocks", "full_kv_indices"),
        ("q_num_blocks", "q_indices"),
        ("full_q_num_blocks", "full_q_indices"),
    ):
        actual = _block_entries(block_mask, counts_name, indices_name)
        expected = _block_entries(reference, counts_name, indices_name)
        if actual != expected:
            raise AssertionError(f"{counts_name}/{indices_name} mismatch")


def _slice_mask_mod(mask_mod: object, slices: Sequence[object]) -> object:
    if not slices:
        return mask_mod

    def sliced_mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        in_slice = (query_idx < 0) & (kv_idx < 0)
        for slice_ in slices:
            in_slice |= (
                (query_idx >= int(slice_.q_range.start))
                & (query_idx < int(slice_.q_range.end))
                & (kv_idx >= int(slice_.k_range.start))
                & (kv_idx < int(slice_.k_range.end))
            )
        return in_slice & mask_mod(batch_idx, head_idx, query_idx, kv_idx)

    return sliced_mask_mod


def _block_entries(
    block_mask: BlockMask,
    counts_name: str,
    indices_name: str,
) -> set[tuple[int, int, int, int]]:
    counts = getattr(block_mask, counts_name)
    indices = getattr(block_mask, indices_name)
    if counts is None or indices is None:
        return set()
    entries = set()
    for batch_index in range(int(counts.shape[0])):
        for head_index in range(int(counts.shape[1])):
            for block_index in range(int(counts.shape[2])):
                block_count = int(counts[batch_index, head_index, block_index])
                for other_block in indices[
                    batch_index,
                    head_index,
                    block_index,
                    :block_count,
                ].tolist():
                    entries.add(
                        (
                            batch_index,
                            head_index,
                            block_index,
                            int(other_block),
                        )
                    )
    return entries


def _logical_tokens(pack: SharedPrefixPack) -> int:
    return sum(int(positions.numel()) for positions in pack.positions_by_sequence)


def _torch_validation_skip_reason(
    *,
    validate_torch: bool,
    packed_tokens: int,
    token_cap: int,
) -> str | None:
    if not validate_torch:
        return "disabled"
    if token_cap > 0 and packed_tokens > token_cap:
        return f"packed_tokens>{token_cap}"
    return None


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("CSV option must contain at least one value")
    return values


def _write(path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(line + "\n")
    print(line, flush=True)


def _check_threshold(name: str, value_ms: float, limit_ms: float | None) -> None:
    if limit_ms is not None and float(value_ms) > float(limit_ms):
        raise RuntimeError(
            f"{name} took {float(value_ms):.3f}ms, exceeding {float(limit_ms):.3f}ms"
        )


if __name__ == "__main__":
    typer.run(main)
