from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager, suppress
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import torch
import torch.distributed as dist
import typer

from art.megatron.prefix_tree_packing import PrefixTreePack, prefix_tree_pack
import art.trainer_rank as trainer_rank_module
from art.trainer_rank import (
    AdamParams,
    ForwardInput,
    TopK,
    TrainerRank,
    _batch_seq_logits,
    _language_model,
    _unflatten,
)


def _pack_forward_items(items: Sequence[Any], *, max_depth: int) -> PrefixTreePack:
    return prefix_tree_pack(
        (item.input_ids for item in items),
        max_depth=max_depth,
    )


def main(
    model: str = "Qwen/Qwen3-0.6B",
    layers: int = 1,
    seq_len: int = 2048,
    prefix_families: int = 0,
    prefix_len: int = 5000,
    mid_prefixes_per_family: int = 1,
    mid_prefix_len: int = 0,
    branches_per_prefix: int = 16,
    completion_len: int = 100,
    warmup: int = 2,
    repeat: int = 5,
    head_chunk_tokens: int = 512,
    shared_prefix_max_depth: int = 1,
    benchmark: str = "target_builtin_fwd",
    target_count: int = 4,
    top_k: int = 5,
    top_k_values: str = "1,2,5,10,20,50",
    max_unpacked_output_gb: float = 0.5,
    mask_prefix_targets: bool = True,
    workload: str = "regular",
    tree_depth: int = 3,
    tree_seed: int = 1,
    tree_duplicate_factor: int = 1,
    adapter_slots: int = 0,
    adapter_slot_mode: str = "family",
    adapter_slot_rank: int = 1,
    learning_rate: float = 1e-5,
    full_step_offload_reload: bool = False,
    memory_safety_factor: float = 1.10,
    memory_reserve_fraction: float = 0.03,
    memory_sample_interval_s: float = 0.05,
    compare_target_correctness: bool = False,
    run_adapter_sanity: bool = False,
    progress_jsonl: str = "",
    output_jsonl: str = "",
) -> None:
    if progress_jsonl:
        os.environ["ART_TRAINER_RANK_PROGRESS_JSONL"] = progress_jsonl

    os.environ.setdefault("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        from megatron.core import parallel_state as ps

        from art.megatron import train as megatron_train

        provider_configure = (
            (lambda provider: setattr(provider, "num_layers", layers))
            if layers > 0
            else None
        )
        runtime = megatron_train.build_training_runtime(
            model_identifier=model,
            provider_configure=provider_configure,
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.eval()
        rank = TrainerRank(
            runtime,
            head_chunk_tokens=head_chunk_tokens,
            shared_prefix_max_depth=shared_prefix_max_depth,
            memory_safety_factor=memory_safety_factor,
            memory_reserve_fraction=memory_reserve_fraction,
        )
        if adapter_slots < 0:
            raise ValueError("adapter_slots must be >= 0")
        if adapter_slot_rank < 1:
            raise ValueError("adapter_slot_rank must be >= 1")
        if adapter_slots:
            loaded_sites = _load_adapter_slots(
                rank,
                count=adapter_slots,
                slot_rank=adapter_slot_rank,
            )
        else:
            loaded_sites = 0
        hidden_size, vocab_size, dtype_size = _runtime_output_shape(runtime)
        model_config = getattr(_language_model(runtime.model[0]), "config", None)

        benchmarks = {
            name.strip().replace("-", "_")
            for name in benchmark.split(",")
            if name.strip()
        }
        if "all" in benchmarks:
            benchmarks = {
                "target_builtin_fwd",
                "target_trainer_fwd",
                "target_hidden_fwd",
                "logits_builtin_fwd",
                "logits_hidden_fwd",
                "target_builtin_fwd_bwd",
                "target_builtin_masked_fwd_bwd",
                "target_trainer_fwd_bwd",
                "target_hidden_fwd_bwd",
                "target_builtin_train_step",
                "target_trainer_train_step",
                "target_trainer_fixed_train_step",
                "target_trainer_adaptive_train_step",
                "target_trainer_adaptive_profile_train_step",
                "target_hidden_train_step",
                "trainer_multi_target_fwd_bwd",
                "trainer_multi_target_train_step",
                "trainer_multi_target_fixed_train_step",
                "trainer_multi_target_adaptive_train_step",
                "trainer_target",
                "trainer_multi_target",
                "trainer_topk",
                "trainer_topk_head",
                "trainer_topk_fwd_bwd",
                "trainer_topk_train_step",
                "trainer_topk_fixed_train_step",
                "trainer_topk_adaptive_train_step",
                "trainer_topk_sweep",
                "trainer_target_topk",
                "trainer_hidden",
                "trainer_all_no_logits",
                "trainer_logits",
            }
        if "trainer_all" in benchmarks:
            benchmarks.update(
                {
                    "trainer_target",
                    "trainer_multi_target",
                    "trainer_multi_target_fwd_bwd",
                    "trainer_multi_target_train_step",
                    "trainer_multi_target_fixed_train_step",
                    "trainer_multi_target_adaptive_train_step",
                    "trainer_topk",
                    "trainer_topk_head",
                    "trainer_topk_fwd_bwd",
                    "trainer_topk_train_step",
                    "trainer_topk_fixed_train_step",
                    "trainer_topk_adaptive_train_step",
                    "trainer_topk_sweep",
                    "trainer_target_topk",
                    "trainer_hidden",
                    "trainer_all_no_logits",
                    "trainer_logits",
                }
            )

        if target_count < 1:
            raise ValueError("target_count must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if memory_sample_interval_s < 0:
            raise ValueError("memory_sample_interval_s must be >= 0")
        requests, multi_target_requests, request_metadata = _requests(
            seq_len=seq_len,
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
            target_count=target_count,
            mask_prefix_targets=mask_prefix_targets,
            workload=workload,
            tree_depth=tree_depth,
            tree_seed=tree_seed,
            tree_duplicate_factor=tree_duplicate_factor,
        )
        requests = _route_adapter_slots(
            requests,
            adapter_slots=adapter_slots,
            mode=adapter_slot_mode,
        )
        multi_target_requests = _route_adapter_slots(
            multi_target_requests,
            adapter_slots=adapter_slots,
            mode=adapter_slot_mode,
        )
        stats_items = [rank._forward_item(request) for request in requests]
        stats_batch = _pack_forward_items(
            stats_items,
            max_depth=rank.shared_prefix_max_depth,
        )
        stats_prepared = rank._prepare_packed_forward(stats_batch)
        request_stats = _packed_request_stats(
            requests,
            stats_items,
            stats_batch,
            request_metadata=request_metadata,
        )
        planner_metadata = _gather_planner_metadata(stats_prepared)
        target_items = None
        target_prepared = None
        if any(name.startswith("target_") for name in benchmarks):
            target_items = stats_items
            target_prepared = stats_prepared
        logits_items = None
        logits_prepared = None
        if any(name.startswith("logits_") for name in benchmarks):
            logits_items = [
                rank._forward_item(_with_outputs(request, logits=True))
                for request in requests
            ]
            logits_prepared = rank._prepare_packed_forward(
                _pack_forward_items(
                    logits_items,
                    max_depth=rank.shared_prefix_max_depth,
                )
            )
        results: dict[str, float] = {}
        metadata: dict[str, object] = {}
        rate_units: dict[str, dict[str, int]] = {}

        def register_case(
            name: str,
            case_requests: Sequence[
                ForwardInput[
                    torch.Tensor | None,
                    TopK | None,
                    torch.Tensor | None,
                    torch.Tensor | None,
                ]
            ],
            case_stats: dict[str, int | str],
        ) -> None:
            units = _rate_units(
                case_requests,
                case_stats,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dtype_size=dtype_size,
            )
            rate_units[name] = units
            for key, value in units.items():
                metadata[f"{name}_{key}"] = value

        for name in (
            "target_builtin_fwd",
            "target_hidden_fwd",
            "target_trainer_fwd",
            "target_builtin_fwd_bwd",
            "target_builtin_masked_fwd_bwd",
            "target_trainer_fwd_bwd",
            "target_hidden_fwd_bwd",
            "target_builtin_train_step",
            "target_trainer_train_step",
            "target_trainer_fixed_train_step",
            "target_trainer_adaptive_train_step",
            "target_trainer_adaptive_profile_train_step",
            "target_hidden_train_step",
        ):
            register_case(name, requests, request_stats)

        memory_tracker = _CudaMemoryTracker(
            device_index=int(os.environ["LOCAL_RANK"]),
            sample_interval_s=memory_sample_interval_s,
        )
        memory_tracker.start()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if "target_builtin_fwd" in benchmarks:
                assert target_items is not None and target_prepared is not None
                results["target_builtin_fwd_ms"] = _bench(
                    lambda: _builtin(
                        rank,
                        target_prepared,
                        _packed_labels(target_items, target_prepared),
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "target_hidden_fwd" in benchmarks:
                assert target_items is not None and target_prepared is not None
                results["target_hidden_fwd_ms"] = _bench(
                    lambda: rank._project_head(
                        target_items,
                        target_prepared,
                        rank._gather_sequence_parallel_hidden(
                            rank._decoder_hidden(target_prepared)
                        ),
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "target_trainer_fwd" in benchmarks:
                assert target_items is not None and target_prepared is not None
                results["target_trainer_fwd_ms"] = _bench(
                    lambda: rank._forward_packed(target_items, target_prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "logits_builtin_fwd" in benchmarks:
                assert logits_prepared is not None
                register_case(
                    "logits_builtin_fwd", _logits_requests(requests), request_stats
                )
                results["logits_builtin_fwd_ms"] = _bench(
                    lambda: _full_logits(rank, logits_prepared),
                    warmup=warmup,
                    repeat=repeat,
                )
            if "logits_hidden_fwd" in benchmarks:
                assert logits_items is not None and logits_prepared is not None
                register_case(
                    "logits_hidden_fwd", _logits_requests(requests), request_stats
                )
                results["logits_hidden_fwd_ms"] = _bench(
                    lambda: rank._project_head(
                        logits_items,
                        logits_prepared,
                        rank._gather_sequence_parallel_hidden(
                            rank._decoder_hidden(logits_prepared)
                        ),
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
            trainer_cases = {
                "trainer_target": requests,
                "trainer_multi_target": multi_target_requests,
                "trainer_topk": [
                    _with_outputs(request, top_k=top_k) for request in requests
                ],
                "trainer_target_topk": [
                    _with_outputs(
                        request,
                        target_tokens=request.target_tokens,
                        top_k=top_k,
                    )
                    for request in requests
                ],
                "trainer_hidden": [
                    _with_outputs(request, hidden_states=True) for request in requests
                ],
                "trainer_all_no_logits": [
                    _with_outputs(
                        request,
                        target_tokens=multi_request.target_tokens,
                        top_k=top_k,
                        hidden_states=True,
                    )
                    for request, multi_request in zip(
                        requests, multi_target_requests, strict=True
                    )
                ],
                "trainer_logits": [
                    ForwardInput(input_tokens=request.input_tokens, logits=True)
                    for request in requests
                ],
            }
            if "trainer_topk_sweep" in benchmarks:
                for k in _int_values(top_k_values):
                    trainer_cases[f"trainer_topk_{k}"] = [
                        _with_outputs(request, top_k=k) for request in requests
                    ]
            for name, case_requests in trainer_cases.items():
                if name not in benchmarks and not (
                    "trainer_topk_sweep" in benchmarks
                    and name.startswith("trainer_topk_")
                ):
                    continue
                output_gb = _request_output_gb(
                    case_requests,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    dtype_size=dtype_size,
                )
                metadata[f"{name}_output_gb"] = round(output_gb, 3)
                if max_unpacked_output_gb > 0 and output_gb > max_unpacked_output_gb:
                    metadata[f"{name}_skipped"] = "unpacked_output_cap"
                    continue
                items = [rank._forward_item(request) for request in case_requests]
                batch = _pack_forward_items(
                    items,
                    max_depth=rank.shared_prefix_max_depth,
                )
                register_case(
                    name,
                    case_requests,
                    _packed_request_stats(
                        case_requests, items, batch, request_metadata={}
                    ),
                )
                prepared = rank._prepare_packed_forward(batch)
                if adapter_slots:
                    results[f"{name}_ms"] = _bench(
                        lambda case_requests=case_requests: rank.dp_rank_forward(
                            case_requests
                        ),
                        warmup=warmup,
                        repeat=repeat,
                    )
                else:
                    results[f"{name}_ms"] = _bench(
                        lambda items=items, prepared=prepared: rank._forward_packed(
                            items,
                            prepared,
                        ),
                        warmup=warmup,
                        repeat=repeat,
                    )
            if "trainer_topk_head" in benchmarks:
                case_requests = [
                    _with_outputs(request, top_k=top_k) for request in requests
                ]
                output_gb = _request_output_gb(
                    case_requests,
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    dtype_size=dtype_size,
                )
                metadata["trainer_topk_head_output_gb"] = round(output_gb, 3)
                items = [rank._forward_item(request) for request in case_requests]
                batch = _pack_forward_items(
                    items,
                    max_depth=rank.shared_prefix_max_depth,
                )
                register_case(
                    "trainer_topk_head",
                    case_requests,
                    _packed_request_stats(
                        case_requests, items, batch, request_metadata={}
                    ),
                )
                prepared = rank._prepare_packed_forward(batch)
                hidden = rank._gather_sequence_parallel_hidden(
                    rank._decoder_hidden(prepared)
                )
                results["trainer_topk_head_ms"] = _bench(
                    lambda: rank._project_head(items, prepared, hidden),
                    warmup=warmup,
                    repeat=repeat,
                )

        if "target_builtin_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_builtin_fwd_bwd_ms"] = _bench(
                lambda: _target_builtin_loss(
                    rank,
                    target_items,
                    target_prepared,
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "target_builtin_masked_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_builtin_masked_fwd_bwd_ms"] = _bench(
                lambda: _target_builtin_masked_loss(
                    rank,
                    target_items,
                    target_prepared,
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "target_trainer_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_trainer_fwd_bwd_ms"] = _bench(
                lambda: (
                    _target_requests_loss(rank, requests)
                    if adapter_slots
                    else _target_trainer_loss(
                        rank,
                        target_items,
                        target_prepared,
                    )
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "target_hidden_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_hidden_fwd_bwd_ms"] = _bench(
                lambda: _target_hidden_loss(
                    rank,
                    target_items,
                    target_prepared,
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        train_step_params = AdamParams(learning_rate=learning_rate)
        offload_manager = (
            _make_offload_manager(runtime) if full_step_offload_reload else None
        )
        if "target_builtin_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_builtin_train_step_ms"] = _bench(
                lambda: _training_step(
                    rank,
                    lambda: _target_builtin_loss(rank, target_items, target_prepared),
                    params=train_step_params,
                    offload_manager=offload_manager,
                ),
                warmup=warmup,
                repeat=repeat,
            )
        if "target_trainer_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_trainer_train_step_ms"] = _bench(
                lambda: _training_step(
                    rank,
                    lambda: (
                        _target_requests_loss(rank, requests)
                        if adapter_slots
                        else _target_trainer_loss(rank, target_items, target_prepared)
                    ),
                    params=train_step_params,
                    offload_manager=offload_manager,
                ),
                warmup=warmup,
                repeat=repeat,
            )
        if "target_trainer_fixed_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            fixed_stats: list[dict[str, int | bool]] = []
            results["target_trainer_fixed_train_step_ms"] = _bench(
                lambda: _fixed_micro_batch_training_step(
                    rank,
                    requests,
                    params=train_step_params,
                    offload_manager=offload_manager,
                    loss_kind="target",
                    stats_sink=fixed_stats,
                ),
                warmup=warmup,
                repeat=repeat,
            )
            _record_micro_batch_stats(
                metadata, "target_trainer_fixed_train_step", fixed_stats
            )
        if "target_trainer_adaptive_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            adaptive_stats: list[dict[str, int | bool]] = []
            results["target_trainer_adaptive_train_step_ms"] = _bench(
                lambda: _adaptive_micro_batch_training_step(
                    rank,
                    requests,
                    params=train_step_params,
                    offload_manager=offload_manager,
                    loss_kind="target",
                    stats_sink=adaptive_stats,
                ),
                warmup=warmup,
                repeat=repeat,
            )
            _record_micro_batch_stats(
                metadata, "target_trainer_adaptive_train_step", adaptive_stats
            )
        if "target_trainer_adaptive_profile_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            adaptive_stats: list[dict[str, int | bool | float]] = []
            results["target_trainer_adaptive_profile_train_step_ms"] = _bench(
                lambda: _profiled_adaptive_micro_batch_training_step(
                    rank,
                    requests,
                    params=train_step_params,
                    offload_manager=offload_manager,
                    loss_kind="target",
                    stats_sink=adaptive_stats,
                ),
                warmup=warmup,
                repeat=repeat,
            )
            _record_micro_batch_stats(
                metadata,
                "target_trainer_adaptive_profile_train_step",
                adaptive_stats,
            )
            _record_profile_stats(
                metadata,
                "target_trainer_adaptive_profile_train_step",
                adaptive_stats,
            )
        if "target_hidden_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            assert target_items is not None and target_prepared is not None
            results["target_hidden_train_step_ms"] = _bench(
                lambda: _training_step(
                    rank,
                    lambda: _target_hidden_loss(rank, target_items, target_prepared),
                    params=train_step_params,
                    offload_manager=offload_manager,
                ),
                warmup=warmup,
                repeat=repeat,
            )
        if "trainer_multi_target_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            items = [rank._forward_item(request) for request in multi_target_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            register_case(
                "trainer_multi_target_fwd_bwd",
                multi_target_requests,
                _packed_request_stats(
                    multi_target_requests,
                    items,
                    batch,
                    request_metadata={},
                ),
            )
            prepared = rank._prepare_packed_forward(batch)
            results["trainer_multi_target_fwd_bwd_ms"] = _bench(
                lambda: (
                    _target_requests_loss(rank, multi_target_requests)
                    if adapter_slots
                    else _target_trainer_loss(rank, items, prepared)
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "trainer_multi_target_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            items = [rank._forward_item(request) for request in multi_target_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            register_case(
                "trainer_multi_target_train_step",
                multi_target_requests,
                _packed_request_stats(
                    multi_target_requests,
                    items,
                    batch,
                    request_metadata={},
                ),
            )
            prepared = rank._prepare_packed_forward(batch)
            results["trainer_multi_target_train_step_ms"] = _bench(
                lambda: _training_step(
                    rank,
                    lambda: (
                        _target_requests_loss(rank, multi_target_requests)
                        if adapter_slots
                        else _target_trainer_loss(rank, items, prepared)
                    ),
                    params=train_step_params,
                    offload_manager=offload_manager,
                ),
                warmup=warmup,
                repeat=repeat,
            )
        if (
            "trainer_multi_target_fixed_train_step" in benchmarks
            or "trainer_multi_target_adaptive_train_step" in benchmarks
        ):
            items = [rank._forward_item(request) for request in multi_target_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            multi_target_stats = _packed_request_stats(
                multi_target_requests,
                items,
                batch,
                request_metadata={},
            )
            if "trainer_multi_target_fixed_train_step" in benchmarks:
                register_case(
                    "trainer_multi_target_fixed_train_step",
                    multi_target_requests,
                    multi_target_stats,
                )
                for chunk in runtime.model:
                    chunk.train()
                fixed_stats = []
                results["trainer_multi_target_fixed_train_step_ms"] = _bench(
                    lambda: _fixed_micro_batch_training_step(
                        rank,
                        multi_target_requests,
                        params=train_step_params,
                        offload_manager=offload_manager,
                        loss_kind="target",
                        stats_sink=fixed_stats,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
                _record_micro_batch_stats(
                    metadata,
                    "trainer_multi_target_fixed_train_step",
                    fixed_stats,
                )
            if "trainer_multi_target_adaptive_train_step" in benchmarks:
                register_case(
                    "trainer_multi_target_adaptive_train_step",
                    multi_target_requests,
                    multi_target_stats,
                )
                for chunk in runtime.model:
                    chunk.train()
                adaptive_stats = []
                results["trainer_multi_target_adaptive_train_step_ms"] = _bench(
                    lambda: _adaptive_micro_batch_training_step(
                        rank,
                        multi_target_requests,
                        params=train_step_params,
                        offload_manager=offload_manager,
                        loss_kind="target",
                        stats_sink=adaptive_stats,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
                _record_micro_batch_stats(
                    metadata,
                    "trainer_multi_target_adaptive_train_step",
                    adaptive_stats,
                )
        if "trainer_topk_fwd_bwd" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            topk_requests = [
                _with_outputs(request, top_k=top_k) for request in requests
            ]
            items = [rank._forward_item(request) for request in topk_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            register_case(
                "trainer_topk_fwd_bwd",
                topk_requests,
                _packed_request_stats(topk_requests, items, batch, request_metadata={}),
            )
            prepared = rank._prepare_packed_forward(batch)
            results["trainer_topk_fwd_bwd_ms"] = _bench(
                lambda: (
                    _topk_requests_loss(rank, topk_requests)
                    if adapter_slots
                    else _trainer_topk_loss(rank, items, prepared)
                ).backward(),
                warmup=warmup,
                repeat=repeat,
                after=rank.zero_grad,
            )
        if "trainer_topk_train_step" in benchmarks:
            for chunk in runtime.model:
                chunk.train()
            topk_requests = [
                _with_outputs(request, top_k=top_k) for request in requests
            ]
            items = [rank._forward_item(request) for request in topk_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            register_case(
                "trainer_topk_train_step",
                topk_requests,
                _packed_request_stats(topk_requests, items, batch, request_metadata={}),
            )
            prepared = rank._prepare_packed_forward(batch)
            results["trainer_topk_train_step_ms"] = _bench(
                lambda: _training_step(
                    rank,
                    lambda: (
                        _topk_requests_loss(rank, topk_requests)
                        if adapter_slots
                        else _trainer_topk_loss(rank, items, prepared)
                    ),
                    params=train_step_params,
                    offload_manager=offload_manager,
                ),
                warmup=warmup,
                repeat=repeat,
            )
        if (
            "trainer_topk_fixed_train_step" in benchmarks
            or "trainer_topk_adaptive_train_step" in benchmarks
        ):
            topk_requests = [
                _with_outputs(request, top_k=top_k) for request in requests
            ]
            items = [rank._forward_item(request) for request in topk_requests]
            batch = _pack_forward_items(
                items,
                max_depth=rank.shared_prefix_max_depth,
            )
            topk_stats = _packed_request_stats(
                topk_requests,
                items,
                batch,
                request_metadata={},
            )
            if "trainer_topk_fixed_train_step" in benchmarks:
                register_case(
                    "trainer_topk_fixed_train_step",
                    topk_requests,
                    topk_stats,
                )
                for chunk in runtime.model:
                    chunk.train()
                fixed_stats = []
                results["trainer_topk_fixed_train_step_ms"] = _bench(
                    lambda: _fixed_micro_batch_training_step(
                        rank,
                        topk_requests,
                        params=train_step_params,
                        offload_manager=offload_manager,
                        loss_kind="topk",
                        stats_sink=fixed_stats,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
                _record_micro_batch_stats(
                    metadata, "trainer_topk_fixed_train_step", fixed_stats
                )
            if "trainer_topk_adaptive_train_step" in benchmarks:
                register_case(
                    "trainer_topk_adaptive_train_step",
                    topk_requests,
                    topk_stats,
                )
                for chunk in runtime.model:
                    chunk.train()
                adaptive_stats = []
                results["trainer_topk_adaptive_train_step_ms"] = _bench(
                    lambda: _adaptive_micro_batch_training_step(
                        rank,
                        topk_requests,
                        params=train_step_params,
                        offload_manager=offload_manager,
                        loss_kind="topk",
                        stats_sink=adaptive_stats,
                    ),
                    warmup=warmup,
                    repeat=repeat,
                )
                _record_micro_batch_stats(
                    metadata, "trainer_topk_adaptive_train_step", adaptive_stats
                )

        if compare_target_correctness and adapter_slots:
            metadata["target_correctness_skipped"] = "adapter_slots"
        elif compare_target_correctness:
            assert target_items is not None and target_prepared is not None
            metadata.update(
                _target_correctness_metrics(rank, target_items, target_prepared)
            )
        if run_adapter_sanity and adapter_slots > 0:
            metadata.update(
                _adapter_sanity_metrics(
                    rank,
                    requests,
                    params=train_step_params,
                    adapter_slots=adapter_slots,
                )
            )

        memory_tracker.stop()
        memory_metadata = _distributed_memory_metadata(memory_tracker)
        model_metadata = _model_metadata(runtime, model, layers=layers)

        if dist.get_rank() == 0:
            token_rates = _rate_metrics(results, rate_units)
            payload = {
                "world": dist.get_world_size(),
                "tp": int(ps.get_tensor_model_parallel_world_size()),
                "cp": int(ps.get_context_parallel_world_size()),
                "seq_len": seq_len,
                "prefix_families": prefix_families,
                "prefix_len": prefix_len,
                "mid_prefixes_per_family": mid_prefixes_per_family,
                "mid_prefix_len": mid_prefix_len,
                "branches_per_prefix": branches_per_prefix,
                "completion_len": completion_len,
                "head_chunk_tokens": head_chunk_tokens,
                "shared_prefix_max_depth": shared_prefix_max_depth,
                "warmup": warmup,
                "repeat": repeat,
                "target_count": target_count,
                "top_k": top_k,
                "top_k_values": top_k_values,
                "max_unpacked_output_gb": max_unpacked_output_gb,
                "mask_prefix_targets": mask_prefix_targets,
                "workload": workload,
                "tree_depth": tree_depth,
                "tree_seed": tree_seed,
                "tree_duplicate_factor": tree_duplicate_factor,
                "adapter_slots": adapter_slots,
                "adapter_slot_mode": adapter_slot_mode,
                "adapter_slot_rank": adapter_slot_rank,
                "adapter_loaded_sites": loaded_sites,
                "learning_rate": learning_rate,
                "full_step_offload_reload": full_step_offload_reload,
                "memory_safety_factor": memory_safety_factor,
                "memory_reserve_fraction": memory_reserve_fraction,
                "mtp_num_layers": getattr(model_config, "mtp_num_layers", None),
                "cross_entropy_loss_fusion": getattr(
                    model_config, "cross_entropy_loss_fusion", None
                ),
                "cross_entropy_fusion_impl": getattr(
                    model_config, "cross_entropy_fusion_impl", None
                ),
                **model_metadata,
                **request_stats,
                **memory_metadata,
                **results,
                **token_rates,
                **metadata,
                **planner_metadata,
            }
            line = json.dumps(payload, sort_keys=True)
            print(line, flush=True)
            if output_jsonl:
                output_path = Path(output_jsonl)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("a", encoding="utf-8") as output_file:
                    output_file.write(line + "\n")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _requests(
    *,
    seq_len: int,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
    target_count: int,
    mask_prefix_targets: bool,
    workload: str,
    tree_depth: int,
    tree_seed: int,
    tree_duplicate_factor: int,
) -> tuple[
    list[ForwardInput[torch.Tensor, None, None, None]],
    list[ForwardInput[torch.Tensor, None, None, None]],
    dict[str, int | str],
]:
    if workload == "regular" and prefix_families <= 0:
        tokens = torch.arange(seq_len, dtype=torch.long) % 32_000 + 100
        labels = _labels(tokens, target_count=1)
        return (
            [ForwardInput(input_tokens=tokens, target_tokens=labels)],
            [
                ForwardInput(
                    input_tokens=tokens,
                    target_tokens=_labels(tokens, target_count=target_count),
                )
            ],
            {
                "request_count": 1,
                "workload_shape": "single",
            },
        )

    if prefix_len < 1 or branches_per_prefix < 1 or completion_len < 1:
        raise ValueError(
            "prefix_len, branches_per_prefix, and completion_len must be >= 1"
        )
    if mid_prefixes_per_family < 1 or mid_prefix_len < 0:
        raise ValueError("mid_prefixes_per_family must be >= 1 and mid_prefix_len >= 0")

    sequences, prefix_lengths, workload_shape = _workload_sequences(
        workload=workload,
        seq_len=seq_len,
        prefix_families=max(prefix_families, 1),
        prefix_len=prefix_len,
        mid_prefixes_per_family=mid_prefixes_per_family,
        mid_prefix_len=mid_prefix_len,
        branches_per_prefix=branches_per_prefix,
        completion_len=completion_len,
        tree_depth=tree_depth,
        tree_seed=tree_seed,
        tree_duplicate_factor=tree_duplicate_factor,
    )
    requests = []
    multi_requests = []
    for tokens, shared_length in zip(sequences, prefix_lengths, strict=True):
        labels = _labels(tokens, target_count=1)
        multi_labels = _labels(tokens, target_count=target_count)
        if mask_prefix_targets and shared_length:
            labels[:shared_length] = -100
            multi_labels[:shared_length] = -100
        requests.append(ForwardInput(input_tokens=tokens, target_tokens=labels))
        multi_requests.append(
            ForwardInput(input_tokens=tokens, target_tokens=multi_labels)
        )

    return (
        requests,
        multi_requests,
        {
            "request_count": len(requests),
            "workload_shape": workload_shape,
        },
    )


def _load_adapter_slots(
    rank: TrainerRank,
    *,
    count: int,
    slot_rank: int,
) -> int:
    loaded_sites = 0
    for slot_index in range(count):
        loaded_sites += rank.load_checkpoint_slot(
            f"S{slot_index}",
            _synthetic_adapter(
                rank.runtime.model, slot_rank=slot_rank, seed=slot_index
            ),
        )
    return loaded_sites


def _synthetic_adapter(
    model: Sequence[torch.nn.Module],
    *,
    slot_rank: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    from art.megatron.lora import LoRA

    adapter: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cuda").manual_seed(10_000 + seed)
    for chunk in model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            a_keys = module._expected_weight_keys("lora_A")
            b_keys = module._expected_weight_keys("lora_B")
            for a_key, b_key in zip(a_keys, b_keys, strict=True):
                adapter[a_key] = (
                    torch.randn(
                        slot_rank,
                        module.in_features,
                        dtype=module.A_T.dtype,
                        device=module.A_T.device,
                        generator=generator,
                    )
                    * 0.01
                )
                adapter[b_key] = (
                    torch.randn(
                        module.out_features,
                        slot_rank,
                        dtype=module.B_T.dtype,
                        device=module.B_T.device,
                        generator=generator,
                    )
                    * 0.01
                )
    if not adapter:
        raise RuntimeError("adapter slot stress requested, but model has no LoRA sites")
    return adapter


def _route_adapter_slots(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    adapter_slots: int,
    mode: str,
) -> list[
    ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ]
]:
    if adapter_slots == 0:
        return list(requests)
    if mode not in {"family", "round_robin", "single", "skewed_random"}:
        raise ValueError(
            "adapter_slot_mode must be one of: family, round_robin, single, "
            "skewed_random"
        )
    return [
        ForwardInput(
            input_tokens=request.input_tokens,
            target_tokens=request.target_tokens,
            top_k=request.top_k,
            logits=request.logits,
            hidden_states=request.hidden_states,
            checkpoint=f"S{_adapter_slot_index(index, request, adapter_slots, mode)}",
        )
        for index, request in enumerate(requests)
    ]


def _adapter_slot_index(
    index: int,
    request: ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ],
    adapter_slots: int,
    mode: str,
) -> int:
    if mode == "single":
        return 0
    if mode == "round_robin":
        return index % adapter_slots
    if mode == "skewed_random":
        bucket = (index * 1103515245 + 12345) & 0x7FFFFFFF
        skew = bucket % 100
        if skew < 50:
            return 0
        if skew < 75:
            return min(1, adapter_slots - 1)
        if skew < 90:
            return min(2, adapter_slots - 1)
        return min(3 + (bucket % max(1, adapter_slots - 3)), adapter_slots - 1)
    first_token = (
        int(request.input_tokens[0].item()) if request.input_tokens.numel() else 0
    )
    return (first_token // 10_000_019) % adapter_slots


def _with_outputs(
    request: ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ],
    *,
    target_tokens: torch.Tensor | None = None,
    top_k: int | None = None,
    logits: bool = False,
    hidden_states: bool = False,
) -> ForwardInput[
    torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
]:
    return ForwardInput(
        input_tokens=request.input_tokens,
        target_tokens=target_tokens,
        top_k=top_k,
        logits=logits,
        hidden_states=hidden_states,
        checkpoint=request.checkpoint,
        lora=request.lora,
    )


def _workload_sequences(
    *,
    workload: str,
    seq_len: int,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
    tree_depth: int,
    tree_seed: int,
    tree_duplicate_factor: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    if workload in {"austin_198k", "austin_5k_16x100"}:
        return _regular_tree_sequences(
            prefix_families=30,
            prefix_len=5000,
            mid_prefixes_per_family=1,
            mid_prefix_len=0,
            branches_per_prefix=16,
            completion_len=100,
        )
    if workload == "austin_varied":
        return _austin_varied_sequences()
    if workload == "regular":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=mid_prefixes_per_family,
            mid_prefix_len=mid_prefix_len,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "single":
        tokens = torch.arange(seq_len, dtype=torch.long) % 32_000 + 100
        return (tokens,), (0,), "single"
    if workload == "long_root":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=1,
            mid_prefix_len=0,
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "long_mid":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(2, mid_prefixes_per_family),
            mid_prefix_len=max(1, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "many_tiny_leaves":
        return _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(1, mid_prefixes_per_family),
            mid_prefix_len=max(0, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=max(1, completion_len),
        )
    if workload == "uneven":
        return _uneven_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(2, mid_prefixes_per_family),
            mid_prefix_len=max(1, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
    if workload == "duplicates":
        sequences, shared, shape = _regular_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            mid_prefixes_per_family=max(2, mid_prefixes_per_family),
            mid_prefix_len=max(1, mid_prefix_len),
            branches_per_prefix=branches_per_prefix,
            completion_len=completion_len,
        )
        factor = max(1, tree_duplicate_factor)
        return (
            tuple(sequence for sequence in sequences for _ in range(factor)),
            tuple(length for length in shared for _ in range(factor)),
            f"{shape}:duplicates={factor}",
        )
    if workload == "random":
        return _random_tree_sequences(
            prefix_families=prefix_families,
            prefix_len=prefix_len,
            branches_per_prefix=max(2, min(branches_per_prefix, 4)),
            completion_len=completion_len,
            tree_depth=max(1, tree_depth),
            seed=tree_seed,
        )
    raise ValueError(
        "workload must be one of: regular, single, long_root, long_mid, "
        "many_tiny_leaves, uneven, duplicates, random, austin_198k, austin_varied"
    )


def _regular_tree_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    nested = mid_prefixes_per_family > 1 and mid_prefix_len > 0
    sequences: list[torch.Tensor] = []
    shared_lengths: list[int] = []
    for family in range(prefix_families):
        family_base = family * 10_000_019
        root = _tokens(family_base, prefix_len)
        mid_count = mid_prefixes_per_family if nested else 1
        for mid in range(mid_count):
            mid_prefix = (
                _tokens(family_base + 1_000_003 + mid * 100_003, mid_prefix_len)
                if nested
                else torch.empty(0, dtype=torch.long)
            )
            shared = torch.cat((root, mid_prefix))
            for branch in range(branches_per_prefix):
                sequences.append(
                    torch.cat(
                        (
                            shared,
                            _tokens(
                                family_base + mid * 100_003 + branch * 1009 + 17,
                                completion_len,
                            ),
                        )
                    )
                )
                shared_lengths.append(int(shared.numel()))
    shape = (
        f"families={prefix_families}:mid={mid_prefixes_per_family}:"
        f"branches={branches_per_prefix}:nested={int(nested)}"
    )
    return tuple(sequences), tuple(shared_lengths), shape


def _austin_varied_sequences() -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    sequences: list[torch.Tensor] = []
    shared_lengths: list[int] = []
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
            shared_lengths.append(int(root.numel()))
    return tuple(sequences), tuple(shared_lengths), "austin_varied"


def _uneven_tree_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    mid_prefixes_per_family: int,
    mid_prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    sequences: list[torch.Tensor] = []
    shared_lengths: list[int] = []
    for family in range(prefix_families):
        family_base = family * 10_000_019
        root_len = max(1, prefix_len // (family + 1))
        root = _tokens(family_base, root_len)
        for mid in range(mid_prefixes_per_family):
            mid_len = max(1, mid_prefix_len // (mid + 1))
            mid_prefix = _tokens(family_base + 1_000_003 + mid * 100_003, mid_len)
            branch_count = max(1, branches_per_prefix - mid)
            for branch in range(branch_count):
                leaf_len = max(1, completion_len * (branch + 1) // branch_count)
                shared = torch.cat((root, mid_prefix))
                sequences.append(
                    torch.cat(
                        (
                            shared,
                            _tokens(
                                family_base + mid * 100_003 + branch * 1009 + 17,
                                leaf_len,
                            ),
                        )
                    )
                )
                shared_lengths.append(int(shared.numel()))
    return tuple(sequences), tuple(shared_lengths), "uneven"


def _random_tree_sequences(
    *,
    prefix_families: int,
    prefix_len: int,
    branches_per_prefix: int,
    completion_len: int,
    tree_depth: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...], str]:
    generator = torch.Generator().manual_seed(seed)
    next_offset = 1
    sequences: list[torch.Tensor] = []
    shared_lengths: list[int] = []

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (), generator=generator).item())

    def segment(length: int) -> torch.Tensor:
        nonlocal next_offset
        out = _tokens(next_offset, max(1, length))
        next_offset += max(1, length) + 10_000
        return out

    def length_for_depth(depth: int) -> int:
        if depth == 0:
            return max(1, prefix_len)
        choices = (1, 8, 64, max(1, completion_len), max(1, prefix_len // 2))
        return choices[randint(0, len(choices) - 1)]

    def walk(prefix: torch.Tensor, depth: int) -> None:
        shared = torch.cat((prefix, segment(length_for_depth(depth))))
        if depth + 1 >= tree_depth:
            leaf_count = randint(2, branches_per_prefix)
            for _ in range(leaf_count):
                leaf = segment(randint(1, max(1, completion_len)))
                sequences.append(torch.cat((shared, leaf)))
                shared_lengths.append(int(shared.numel()))
            return
        for _ in range(randint(2, branches_per_prefix)):
            walk(shared, depth + 1)

    for _ in range(prefix_families):
        walk(torch.empty(0, dtype=torch.long), 0)
    return tuple(sequences), tuple(shared_lengths), f"random:depth={tree_depth}"


def _packed_request_stats(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    items: Sequence[object],
    batch: object,
    *,
    request_metadata: dict[str, int | str],
) -> dict[str, int | str]:
    from art.megatron.prefix_tree import parse_prefix_tree

    trainable_mask = torch.zeros(int(batch.tokens.numel()), dtype=torch.bool)
    trainable_tokens = 0
    for item, positions in zip(items, batch.positions_by_sequence, strict=True):
        labels = getattr(item, "labels", None)
        if labels is None:
            continue
        mask = labels != -100
        row_mask = mask.reshape(int(mask.shape[0]), -1).any(dim=1)
        trainable_tokens += int(mask.sum().item())
        trainable_mask[positions.reshape(-1).cpu()] |= row_mask.cpu()
    group_ids = batch.group_ids
    parent_ids = batch.parent_ids
    return {
        **request_metadata,
        "request_count": len(requests),
        "packed_tokens": int(batch.tokens.numel()),
        "logical_tokens": sum(
            int(request.input_tokens.numel()) for request in requests
        ),
        "trainable_tokens": trainable_tokens,
        "packed_trainable_tokens": int(trainable_mask.sum().item()),
        "packed_group_count": int(group_ids.max().item())
        if int(group_ids.numel())
        else 0,
        "nested_prefix_depth": max(
            (
                segment.depth
                for row in parse_prefix_tree(
                    group_ids=group_ids,
                    parent_ids=parent_ids,
                )
                for segment in row.segments
            ),
            default=0,
        ),
    }


def _gather_planner_metadata(prepared: object) -> dict[str, object]:
    local = _local_planner_metadata(prepared)
    gathered: list[dict[str, object] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if dist.get_rank() != 0:
        return {}
    ranks = [metrics or {} for metrics in gathered]
    gdn_tokens = [int(metrics.get("gdn_tokens", 0)) for metrics in ranks]
    attention_tokens = [int(metrics.get("attention_tokens", 0)) for metrics in ranks]
    keys = (
        "tree_local_bucket_count",
        "tree_chain_bucket_count",
        "tree_local_segment_count",
        "tree_chain_segment_count",
        "tree_local_real_tokens",
        "tree_chain_real_tokens",
        "tree_state_transfer_count",
        "tree_state_transfer_rows",
        "tree_max_padding_ratio",
    )
    merged: dict[str, object] = {
        "planner_rank_gdn_tokens": gdn_tokens,
        "planner_rank_attention_tokens": attention_tokens,
        "planner_gdn_token_imbalance": max(gdn_tokens, default=0)
        - min(gdn_tokens, default=0),
    }
    for key in keys:
        values = [metrics[key] for metrics in ranks if key in metrics]
        if not values:
            continue
        if key.endswith("_ratio"):
            merged[f"planner_{key}_max"] = round(
                max(float(value) for value in values), 3
            )
        else:
            merged[f"planner_{key}_sum"] = int(sum(int(value) for value in values))
            merged[f"planner_{key}_max"] = int(max(int(value) for value in values))
    rank0 = ranks[0] if ranks else {}
    if "tree_depth_count" in rank0:
        merged["planner_tree_depth_count"] = rank0["tree_depth_count"]
    return merged


def _local_planner_metadata(prepared: object) -> dict[str, object]:
    plan = getattr(
        getattr(prepared, "attention_state", None), "gdn_execution_plan", None
    )
    if plan is None:
        return {}
    local_buckets = tuple(
        bucket
        for depth in getattr(plan, "tree_segment_buckets_by_depth", ())
        for bucket in depth
    )
    chain_buckets = tuple(
        bucket
        for depth in getattr(plan, "tree_chain_buckets_by_depth", ())
        for bucket in depth
    )
    all_buckets = (*local_buckets, *chain_buckets)
    padding_ratios = [
        bucket.length * bucket.segment_count / max(1, bucket.real_token_count)
        for bucket in all_buckets
    ]
    transfers_by_depth = getattr(plan, "tree_state_transfers_by_depth", ())
    return {
        "attention_tokens": int(getattr(plan, "attention_token_count", 0)),
        "gdn_tokens": int(getattr(plan, "gdn_token_count", 0)),
        "tree_depth_count": len(getattr(plan, "tree_segment_buckets_by_depth", ())),
        "tree_local_bucket_count": len(local_buckets),
        "tree_chain_bucket_count": len(chain_buckets),
        "tree_local_segment_count": sum(
            bucket.segment_count for bucket in local_buckets
        ),
        "tree_chain_segment_count": sum(
            bucket.segment_count for bucket in chain_buckets
        ),
        "tree_local_real_tokens": sum(
            bucket.real_token_count for bucket in local_buckets
        ),
        "tree_chain_real_tokens": sum(
            bucket.real_token_count for bucket in chain_buckets
        ),
        "tree_state_transfer_count": sum(
            len(transfers) for transfers in transfers_by_depth
        ),
        "tree_state_transfer_rows": sum(
            len(transfer.family_indices)
            for transfers in transfers_by_depth
            for transfer in transfers
        ),
        "tree_max_padding_ratio": max(padding_ratios, default=1.0),
    }


def _tokens(offset: int, length: int) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) + offset) % 32_000 + 100


def _int_values(value: str) -> list[int]:
    values = [int(part) for part in value.split(",") if part.strip()]
    if not values or any(item < 1 for item in values):
        raise ValueError("top_k_values must contain positive integers")
    return values


def _labels(tokens: torch.Tensor, *, target_count: int) -> torch.Tensor:
    labels = torch.stack(
        [((tokens * 7 + 3 + index) % 32_000) for index in range(target_count)],
        dim=1,
    )
    if target_count > 1:
        labels[::17, -1] = -100
        return labels
    return labels[:, 0]


class _CudaMemoryTracker:
    def __init__(self, *, device_index: int, sample_interval_s: float) -> None:
        self.device_index = device_index
        self.sample_interval_s = sample_interval_s
        self.process_peak_bytes = 0
        self.allocated_peak_bytes = 0
        self.reserved_peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        self._sample()
        if self.sample_interval_s <= 0:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not torch.cuda.is_available():
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        torch.cuda.synchronize()
        self._sample()
        self.allocated_peak_bytes = max(
            self.allocated_peak_bytes,
            int(torch.cuda.max_memory_allocated()),
        )
        self.reserved_peak_bytes = max(
            self.reserved_peak_bytes,
            int(torch.cuda.max_memory_reserved()),
        )

    def _poll(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            self._sample()

    def _sample(self) -> None:
        self.process_peak_bytes = max(
            self.process_peak_bytes,
            _current_process_gpu_memory_bytes(self.device_index),
        )
        self.allocated_peak_bytes = max(
            self.allocated_peak_bytes,
            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        )
        self.reserved_peak_bytes = max(
            self.reserved_peak_bytes,
            int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
        )


def _current_process_gpu_memory_bytes(device_index: int) -> int:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        pid = os.getpid()
        processes = list(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
        with suppress(Exception):
            processes.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
        for process in processes:
            if int(process.pid) == pid:
                return int(process.usedGpuMemory)
    except Exception:
        return 0
    return 0


def _distributed_memory_metadata(tracker: _CudaMemoryTracker) -> dict[str, float]:
    values = torch.tensor(
        [
            tracker.allocated_peak_bytes,
            tracker.reserved_peak_bytes,
            tracker.process_peak_bytes,
        ],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return {
        "peak_memory_allocated_gb": round(float(values[0].item()) / 1024**3, 3),
        "peak_memory_reserved_gb": round(float(values[1].item()) / 1024**3, 3),
        "peak_memory_process_gb": round(float(values[2].item()) / 1024**3, 3),
        "peak_memory_gb": round(float(values[0].item()) / 1024**3, 3),
    }


def _mean_abs_pct(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_fp32 = reference.detach().float()
    candidate_fp32 = candidate.detach().float()
    return float(
        (candidate_fp32 - reference_fp32).abs().mean().item()
        / (reference_fp32.abs().mean().item() + 1e-18)
    )


def _model_metadata(runtime: object, model_name: str, *, layers: int) -> dict[str, Any]:
    from art.megatron.lora import LoRA

    provider = getattr(runtime, "provider")
    model = _language_model(getattr(runtime, "model")[0])
    config = getattr(model, "config", None)
    total_params = sum(
        int(param.numel()) for chunk in runtime.model for param in chunk.parameters()
    )
    trainable_params = sum(
        int(param.numel())
        for chunk in runtime.model
        for param in chunk.parameters()
        if param.requires_grad
    )
    lora_sites = sum(
        1
        for chunk in runtime.model
        for module in chunk.modules()
        if isinstance(module, LoRA)
    )
    local = torch.tensor(
        [total_params, trainable_params, lora_sites],
        device="cuda",
        dtype=torch.float64,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return {
        "model": model_name,
        "layers_arg": layers,
        "provider_num_layers": getattr(provider, "num_layers", None),
        "config_num_layers": getattr(config, "num_layers", None),
        "rank_local_param_count": int(local[0].item()),
        "rank_local_trainable_param_count": int(local[1].item()),
        "rank_local_lora_site_count": int(local[2].item()),
    }


def _bench(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    after: Callable[[], object] | None = None,
) -> float:
    for _ in range(warmup):
        fn()
        if after is not None:
            after()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
        if after is not None:
            after()
    stop.record()
    torch.cuda.synchronize()
    elapsed = torch.tensor(start.elapsed_time(stop) / repeat, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return round(float(elapsed.item()), 3)


def _builtin(
    rank: TrainerRank,
    prepared: object,
    labels: torch.Tensor | None,
) -> torch.Tensor:
    from art.megatron.train import _placeholder_attention_mask

    return rank.runtime.model[0](
        input_ids=prepared.tokens,
        position_ids=prepared.position_ids,
        attention_mask=_placeholder_attention_mask(rank.device),
        labels=labels,
        packed_seq_params=prepared.packed_seq_params,
        **rank.runtime.model_support_handler.get_forward_kwargs(
            rank.runtime.model[0],
            attention_bias=prepared.attention_state,
        ),
    )


def _full_logits(rank: TrainerRank, prepared: object) -> torch.Tensor:
    logits = rank._gather_tensor_parallel_logits(_builtin(rank, prepared, None))
    return _batch_seq_logits(logits, seq_len=int(prepared.tokens.shape[1]))


def _target_builtin_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    return _builtin(rank, prepared, _packed_labels(items, prepared)).float().sum()


def _target_builtin_masked_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    labels = _packed_labels(items, prepared)
    per_token_loss = _builtin(rank, prepared, labels).float().reshape(-1)
    valid = labels.reshape(-1) != -100
    return per_token_loss[valid].sum() + per_token_loss.sum() * 0.0


def _target_hidden_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    hidden = rank._gather_sequence_parallel_hidden(rank._decoder_hidden(prepared))
    outputs = rank._project_head(items, prepared, hidden)
    losses = [
        -output.target_logprobs.sum()
        for output in outputs
        if output.target_logprobs is not None
    ]
    if not losses:
        raise RuntimeError("target logprobs were not produced")
    return torch.stack(losses).sum()


def _target_trainer_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    outputs = rank._forward_packed(items, prepared)
    losses = [
        -output.target_logprobs.sum()
        for output in outputs
        if output.target_logprobs is not None
    ]
    if not losses:
        raise RuntimeError("target logprobs were not produced")
    return torch.stack(losses).sum()


def _target_requests_loss(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> torch.Tensor:
    outputs = rank.dp_rank_forward(requests)
    losses = [
        -output.target_logprobs.sum()
        for output in outputs
        if output.target_logprobs is not None
    ]
    if not losses:
        raise RuntimeError("target logprobs were not produced")
    return torch.stack(losses).sum()


def _trainer_topk_loss(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> torch.Tensor:
    outputs = rank._forward_packed(items, prepared)
    losses = [
        -output.top_k.logprobs.sum() for output in outputs if output.top_k is not None
    ]
    if not losses:
        raise RuntimeError("top_k logprobs were not produced")
    return torch.stack(losses).sum()


def _topk_requests_loss(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> torch.Tensor:
    outputs = rank.dp_rank_forward(requests)
    losses = [
        -output.top_k.logprobs.sum() for output in outputs if output.top_k is not None
    ]
    if not losses:
        raise RuntimeError("top_k logprobs were not produced")
    return torch.stack(losses).sum()


def _fixed_micro_batch_training_step(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    params: AdamParams,
    offload_manager: object | None,
    loss_kind: str,
    stats_sink: list[dict[str, int | bool]],
) -> dict[str, float]:
    def body() -> dict[str, float]:
        return _fixed_micro_batch_training_step_body(
            rank,
            requests,
            params=params,
            loss_kind=loss_kind,
            stats_sink=stats_sink,
        )

    if offload_manager is None:
        return body()
    with offload_manager.job():  # type: ignore[attr-defined]
        return body()


def _fixed_micro_batch_training_step_body(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    params: AdamParams,
    loss_kind: str,
    stats_sink: list[dict[str, int | bool]],
) -> dict[str, float]:
    rank.zero_grad()
    dp_rank, dp_size = rank._dp_rank_and_size()
    stats: list[dict[str, int | bool]] = []
    for start in range(0, len(requests), dp_size):
        stop = min(start + dp_size, len(requests))
        indices = tuple(range(start + dp_rank, stop, dp_size))
        local_requests = [requests[index] for index in indices]
        outputs = rank.dp_rank_forward(local_requests)
        loss = _micro_batch_loss(rank, outputs, loss_kind=loss_kind)
        if loss.requires_grad:
            loss.backward()
        stats.append(
            {
                "global_count": stop - start,
                "local_count": len(local_requests),
                "packed_tokens": _logical_input_tokens(local_requests),
                "logical_tokens": _logical_input_tokens(local_requests),
                "rejected_candidates": 0,
                "cold_start": False,
            }
        )
    stats_sink[:] = stats
    return rank.optim_step(params=params, scale_grads=1.0)


def _adaptive_micro_batch_training_step(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    params: AdamParams,
    offload_manager: object | None,
    loss_kind: str,
    stats_sink: list[dict[str, int | bool]],
) -> dict[str, float]:
    def body() -> dict[str, float]:
        return _adaptive_micro_batch_training_step_body(
            rank,
            requests,
            params=params,
            loss_kind=loss_kind,
            stats_sink=stats_sink,
        )

    if offload_manager is None:
        return body()
    with offload_manager.job():  # type: ignore[attr-defined]
        return body()


def _adaptive_micro_batch_training_step_body(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    params: AdamParams,
    loss_kind: str,
    stats_sink: list[dict[str, int | bool]],
) -> dict[str, float]:
    rank.zero_grad()
    stats: list[dict[str, int | bool]] = []
    step_start = time.perf_counter()
    for micro_batch in rank.forward_micro_batches(requests):
        loss = _micro_batch_loss(rank, micro_batch.outputs, loss_kind=loss_kind)
        if loss.requires_grad:
            loss.backward()
        row = {
            "global_count": int(micro_batch.stats.global_count),
            "local_count": int(micro_batch.stats.local_count),
            "packed_tokens": int(micro_batch.stats.packed_tokens),
            "logical_tokens": int(micro_batch.stats.logical_tokens),
            "estimated_required_bytes": int(micro_batch.stats.estimated_required_bytes),
            "available_bytes": int(micro_batch.stats.available_bytes),
            "rejected_candidates": int(micro_batch.stats.rejected_candidates),
            "cold_start": bool(micro_batch.stats.cold_start),
        }
        stats.append(row)
        _emit_adaptive_progress(
            "target_trainer_adaptive_train_step_window",
            {
                **row,
                "window_index": len(stats) - 1,
                "elapsed_ms": (time.perf_counter() - step_start) * 1000.0,
            },
        )
    stats_sink[:] = stats
    return rank.optim_step(params=params, scale_grads=1.0)


def _profiled_adaptive_micro_batch_training_step(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    params: AdamParams,
    offload_manager: object | None,
    loss_kind: str,
    stats_sink: list[dict[str, int | bool | float]],
) -> dict[str, float]:
    def body() -> dict[str, float]:
        return _profiled_adaptive_micro_batch_training_step_body(
            rank,
            requests,
            params=params,
            loss_kind=loss_kind,
            stats_sink=stats_sink,
        )

    if offload_manager is None:
        return body()
    with offload_manager.job():  # type: ignore[attr-defined]
        return body()


def _profiled_adaptive_micro_batch_training_step_body(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    params: AdamParams,
    loss_kind: str,
    stats_sink: list[dict[str, int | bool | float]],
) -> dict[str, float]:
    rank.zero_grad()
    items = list(requests)
    rank._validate_replicated_top_level_count(len(items))
    start = 0
    stats: list[dict[str, int | bool | float]] = []
    step_start = time.perf_counter()
    while start < len(items):
        with _profile_adaptive_selection(rank) as select_profile:
            candidate, select_ms = _timed_cuda(
                rank, lambda: rank._select_next_micro_batch(items, start)
            )
        select_profile["select_plan_residual_ms"] = max(
            0.0,
            select_profile["select_plan_ms"]
            - select_profile["select_forward_item_ms"]
            - select_profile["select_pack_ms"]
            - select_profile["select_output_estimate_ms"]
            - select_profile["select_signature_ms"],
        )
        select_profile["select_memory_check_residual_ms"] = max(
            0.0,
            select_profile["select_memory_check_ms"]
            - select_profile["select_memory_estimate_ms"]
            - select_profile["select_available_memory_ms"],
        )
        select_profile["select_residual_ms"] = max(
            0.0,
            select_ms
            - select_profile["select_estimate_ms"]
            - select_profile["select_plan_ms"]
            - select_profile["select_memory_check_ms"]
            - select_profile["select_profile_check_ms"],
        )
        flat_outputs, execute_ms = _timed_cuda(
            rank,
            lambda: rank._run_flat_plan_with_memory_tracking(
                candidate.plan,
                context="target_trainer_adaptive_profile_train_step",
            ),
        )

        def unflatten_outputs() -> list[object]:
            flat_iter = iter(flat_outputs)
            return [_unflatten(item, flat_iter) for item in candidate.inputs]

        outputs, unflatten_ms = _timed_cuda(
            rank,
            unflatten_outputs,
        )
        loss, loss_ms = _timed_cuda(
            rank, lambda: _micro_batch_loss(rank, outputs, loss_kind=loss_kind)
        )
        if loss.requires_grad:
            _, backward_ms = _timed_cuda(rank, loss.backward)
        else:
            backward_ms = 0.0
        row = {
            "global_count": int(candidate.stats_global_count),
            "local_count": int(len(candidate.inputs)),
            "packed_tokens": int(candidate.plan.packed_tokens),
            "logical_tokens": int(candidate.plan.logical_tokens),
            "estimated_required_bytes": int(candidate.check.estimated_required_bytes),
            "available_bytes": int(candidate.check.available_bytes),
            "rejected_candidates": int(candidate.rejected_candidates),
            "cold_start": bool(candidate.cold_start),
            "select_ms": select_ms,
            "execute_ms": execute_ms,
            "unflatten_ms": unflatten_ms,
            "loss_ms": loss_ms,
            "backward_ms": backward_ms,
            "optim_ms": 0.0,
            **select_profile,
        }
        stats.append(row)
        stop = start + candidate.stats_global_count
        if stop < len(items):
            rank._last_global_micro_batch_size = max(
                rank._last_global_micro_batch_size or 0,
                candidate.stats_global_count,
            )
        _emit_adaptive_progress(
            "target_trainer_adaptive_profile_train_step_window",
            {
                **row,
                "window_index": len(stats) - 1,
                "global_start": int(start),
                "global_stop": int(stop),
                "remembered_window": int(rank._last_global_micro_batch_size or 0),
                "elapsed_ms": (time.perf_counter() - step_start) * 1000.0,
            },
        )
        start = stop
    metrics, optim_ms = _timed_cuda(
        rank, lambda: rank.optim_step(params=params, scale_grads=1.0)
    )
    if stats:
        stats[-1]["optim_ms"] = optim_ms
    stats_sink[:] = stats
    return metrics


def _emit_adaptive_progress(event: str, row: dict[str, object]) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    path = os.environ.get("ART_TRAINER_RANK_PROGRESS_JSONL")
    if not path:
        return
    payload = {"event": event, **row}
    line = json.dumps(payload, sort_keys=True)
    print(line, flush=True)
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a") as handle:
        handle.write(line + "\n")


@contextmanager
def _profile_adaptive_selection(rank: TrainerRank) -> Any:
    stats = {
        "select_plan_ms": 0.0,
        "select_plan_calls": 0,
        "select_forward_item_ms": 0.0,
        "select_forward_item_calls": 0,
        "select_pack_ms": 0.0,
        "select_pack_calls": 0,
        "select_estimate_ms": 0.0,
        "select_estimate_calls": 0,
        "select_plan_lookup_calls": 0,
        "select_plan_cache_hit_calls": 0,
        "select_plan_cache_miss_calls": 0,
        "select_estimate_lookup_calls": 0,
        "select_estimate_cache_hit_calls": 0,
        "select_estimate_cache_miss_calls": 0,
        "select_output_estimate_ms": 0.0,
        "select_output_estimate_calls": 0,
        "select_signature_ms": 0.0,
        "select_signature_calls": 0,
        "select_memory_check_ms": 0.0,
        "select_memory_check_calls": 0,
        "select_memory_estimate_ms": 0.0,
        "select_memory_estimate_calls": 0,
        "select_available_memory_ms": 0.0,
        "select_available_memory_calls": 0,
        "select_profile_check_ms": 0.0,
        "select_profile_check_calls": 0,
    }

    def timed(
        key: str,
        calls_key: str,
        fn: Callable[..., object],
        *args: object,
        **kwargs: object,
    ) -> object:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            stats[key] += (time.perf_counter() - start) * 1000.0
            stats[calls_key] += 1

    original_plan = rank._plan_flat_forward
    original_cached_plan = rank._cached_adaptive_plan
    original_estimate = rank._estimate_flat_forward
    original_cached_estimate = rank._cached_adaptive_estimate
    original_forward_item = rank._forward_item
    original_pack = trainer_rank_module.prefix_tree_pack
    original_output_estimate = rank._estimate_group_request_output_bytes
    original_signature = rank._memory_signature_from_requests
    original_memory_check = rank._memory_check
    original_memory_estimate = rank._estimate_required_memory_bytes_from_values
    original_available = rank._available_memory_bytes
    original_profile_check = rank._all_ranks_have_memory_profile

    def plan_wrapper(requests: object) -> object:
        return timed("select_plan_ms", "select_plan_calls", original_plan, requests)

    def cached_plan_wrapper(*args: object, **kwargs: object) -> object:
        stats["select_plan_lookup_calls"] += 1
        before = stats["select_plan_calls"]
        result = original_cached_plan(*args, **kwargs)
        if stats["select_plan_calls"] == before:
            stats["select_plan_cache_hit_calls"] += 1
        else:
            stats["select_plan_cache_miss_calls"] += 1
        return result

    def estimate_wrapper(requests: object) -> object:
        return timed(
            "select_estimate_ms",
            "select_estimate_calls",
            original_estimate,
            requests,
        )

    def cached_estimate_wrapper(*args: object, **kwargs: object) -> object:
        stats["select_estimate_lookup_calls"] += 1
        before = stats["select_estimate_calls"]
        result = original_cached_estimate(*args, **kwargs)
        if stats["select_estimate_calls"] == before:
            stats["select_estimate_cache_hit_calls"] += 1
        else:
            stats["select_estimate_cache_miss_calls"] += 1
        return result

    def forward_item_wrapper(request: object) -> object:
        return timed(
            "select_forward_item_ms",
            "select_forward_item_calls",
            original_forward_item,
            request,
        )

    def pack_wrapper(*args: object, **kwargs: object) -> object:
        start = time.perf_counter()
        try:
            return original_pack(*args, **kwargs)
        finally:
            stats["select_pack_ms"] += (time.perf_counter() - start) * 1000.0
            stats["select_pack_calls"] += 1

    def output_estimate_wrapper(items: object) -> object:
        return timed(
            "select_output_estimate_ms",
            "select_output_estimate_calls",
            original_output_estimate,
            items,
        )

    def signature_wrapper(*args: object, **kwargs: object) -> object:
        return timed(
            "select_signature_ms",
            "select_signature_calls",
            original_signature,
            *args,
            **kwargs,
        )

    def memory_check_wrapper(plan: object) -> object:
        return timed(
            "select_memory_check_ms",
            "select_memory_check_calls",
            original_memory_check,
            plan,
        )

    def memory_estimate_wrapper(*args: object, **kwargs: object) -> object:
        return timed(
            "select_memory_estimate_ms",
            "select_memory_estimate_calls",
            original_memory_estimate,
            *args,
            **kwargs,
        )

    def available_wrapper() -> object:
        return timed(
            "select_available_memory_ms",
            "select_available_memory_calls",
            original_available,
        )

    def profile_check_wrapper(*args: object, **kwargs: object) -> object:
        return timed(
            "select_profile_check_ms",
            "select_profile_check_calls",
            original_profile_check,
            *args,
            **kwargs,
        )

    rank._plan_flat_forward = plan_wrapper  # type: ignore[method-assign]
    rank._cached_adaptive_plan = cached_plan_wrapper  # type: ignore[method-assign]
    rank._estimate_flat_forward = estimate_wrapper  # type: ignore[method-assign]
    rank._cached_adaptive_estimate = cached_estimate_wrapper  # type: ignore[method-assign]
    rank._forward_item = forward_item_wrapper  # type: ignore[method-assign]
    trainer_rank_module.prefix_tree_pack = pack_wrapper  # type: ignore[assignment]
    rank._estimate_group_request_output_bytes = output_estimate_wrapper  # type: ignore[method-assign]
    rank._memory_signature_from_requests = signature_wrapper  # type: ignore[method-assign]
    rank._memory_check = memory_check_wrapper  # type: ignore[method-assign]
    rank._estimate_required_memory_bytes_from_values = memory_estimate_wrapper  # type: ignore[method-assign]
    rank._available_memory_bytes = available_wrapper  # type: ignore[method-assign]
    rank._all_ranks_have_memory_profile = profile_check_wrapper  # type: ignore[method-assign]
    try:
        yield stats
    finally:
        rank._plan_flat_forward = original_plan  # type: ignore[method-assign]
        rank._cached_adaptive_plan = original_cached_plan  # type: ignore[method-assign]
        rank._estimate_flat_forward = original_estimate  # type: ignore[method-assign]
        rank._cached_adaptive_estimate = original_cached_estimate  # type: ignore[method-assign]
        rank._forward_item = original_forward_item  # type: ignore[method-assign]
        trainer_rank_module.prefix_tree_pack = original_pack  # type: ignore[assignment]
        rank._estimate_group_request_output_bytes = original_output_estimate  # type: ignore[method-assign]
        rank._memory_signature_from_requests = original_signature  # type: ignore[method-assign]
        rank._memory_check = original_memory_check  # type: ignore[method-assign]
        rank._estimate_required_memory_bytes_from_values = original_memory_estimate  # type: ignore[method-assign]
        rank._available_memory_bytes = original_available  # type: ignore[method-assign]
        rank._all_ranks_have_memory_profile = original_profile_check  # type: ignore[method-assign]


def _timed_cuda(
    rank: TrainerRank,
    fn: Callable[[], object],
) -> tuple[object, float]:
    _sync_cuda(rank)
    start = time.perf_counter()
    result = fn()
    _sync_cuda(rank)
    return result, (time.perf_counter() - start) * 1000.0


def _sync_cuda(rank: TrainerRank) -> None:
    if torch.cuda.is_available() and rank.device.type == "cuda":
        torch.cuda.synchronize(rank.device)


def _micro_batch_loss(
    rank: TrainerRank,
    outputs: object,
    *,
    loss_kind: str,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for output in _iter_outputs(outputs):
        if loss_kind == "target":
            target_logprobs = getattr(output, "target_logprobs", None)
            if target_logprobs is not None:
                losses.append(-target_logprobs.sum())
        elif loss_kind == "topk":
            top_k = getattr(output, "top_k", None)
            if top_k is not None:
                losses.append(-top_k.logprobs.sum())
        else:
            raise ValueError(f"unknown loss_kind: {loss_kind}")
    if not losses:
        return torch.tensor(0.0, device=rank.device)
    return torch.stack(losses).sum()


def _iter_outputs(value: object) -> Sequence[object]:
    if hasattr(value, "target_logprobs") and hasattr(value, "top_k"):
        return (value,)
    if isinstance(value, Sequence):
        outputs: list[object] = []
        for item in value:
            outputs.extend(_iter_outputs(item))
        return outputs
    raise TypeError(f"unexpected TrainerRank output value: {type(value)!r}")


def _logical_input_tokens(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> int:
    return sum(
        int(request.input_tokens.numel())
        for request in requests
        if request.input_tokens is not None
    )


def _record_micro_batch_stats(
    metadata: dict[str, object],
    name: str,
    stats: Sequence[dict[str, int | bool | float]],
) -> None:
    if not stats:
        metadata[f"{name}_micro_window_count"] = 0
        return
    global_counts = [int(stat["global_count"]) for stat in stats]
    local_counts = [int(stat["local_count"]) for stat in stats]
    packed_tokens = [int(stat["packed_tokens"]) for stat in stats]
    rejected = [int(stat["rejected_candidates"]) for stat in stats]
    estimated_required = [
        int(stat.get("estimated_required_bytes", 0)) for stat in stats
    ]
    available = [int(stat.get("available_bytes", 0)) for stat in stats]
    metadata[f"{name}_micro_window_count"] = len(stats)
    metadata[f"{name}_micro_global_count_first"] = global_counts[0]
    metadata[f"{name}_micro_global_count_last"] = global_counts[-1]
    metadata[f"{name}_micro_global_count_min"] = min(global_counts)
    metadata[f"{name}_micro_global_count_max"] = max(global_counts)
    metadata[f"{name}_micro_local_count_min"] = min(local_counts)
    metadata[f"{name}_micro_local_count_max"] = max(local_counts)
    metadata[f"{name}_micro_packed_tokens_min"] = min(packed_tokens)
    metadata[f"{name}_micro_packed_tokens_max"] = max(packed_tokens)
    metadata[f"{name}_micro_rejected_candidates_total"] = sum(rejected)
    metadata[f"{name}_micro_estimated_required_gb_max"] = round(
        max(estimated_required) / 1024**3, 3
    )
    metadata[f"{name}_micro_available_gb_min"] = round(min(available) / 1024**3, 3)
    metadata[f"{name}_micro_cold_start_count"] = sum(
        int(bool(stat["cold_start"])) for stat in stats
    )
    metadata[f"{name}_micro_global_counts_head"] = ",".join(
        str(count) for count in global_counts[:8]
    )


def _record_profile_stats(
    metadata: dict[str, object],
    name: str,
    stats: Sequence[dict[str, int | bool | float]],
) -> None:
    fields = sorted(
        {
            key
            for stat in stats
            for key, value in stat.items()
            if key.endswith("_ms") and isinstance(value, int | float)
        }
    )
    for field in fields:
        total = sum(float(stat.get(field, 0.0)) for stat in stats)
        metadata[f"{name}_{field}_sum"] = round(total, 3)
        metadata[f"{name}_{field}_max"] = round(
            max((float(stat.get(field, 0.0)) for stat in stats), default=0.0),
            3,
        )
    call_fields = sorted(
        {
            key
            for stat in stats
            for key, value in stat.items()
            if key.endswith("_calls") and isinstance(value, int | float)
        }
    )
    for field in call_fields:
        metadata[f"{name}_{field}_sum"] = int(
            sum(int(stat.get(field, 0)) for stat in stats)
        )
        metadata[f"{name}_{field}_max"] = int(
            max((int(stat.get(field, 0)) for stat in stats), default=0)
        )


def _training_step(
    rank: TrainerRank,
    loss_fn: Callable[[], torch.Tensor],
    *,
    params: AdamParams,
    offload_manager: object | None,
) -> dict[str, float]:
    if offload_manager is None:
        return _training_step_body(rank, loss_fn, params=params)
    with offload_manager.job():  # type: ignore[attr-defined]
        return _training_step_body(rank, loss_fn, params=params)


def _training_step_body(
    rank: TrainerRank,
    loss_fn: Callable[[], torch.Tensor],
    *,
    params: AdamParams,
) -> dict[str, float]:
    rank.zero_grad()
    loss = loss_fn()
    loss.backward()
    return rank.optim_step(params=params, scale_grads=1.0)


def _make_offload_manager(runtime: object) -> object:
    from art.megatron.training.streaming_weight_offload import (
        StreamingWeightOffloadConfig,
    )
    from art.megatron.training.weight_offload import WeightOffloadManager

    manager = WeightOffloadManager.from_config(
        model=getattr(runtime, "model"),
        rank=dist.get_rank(),
        compile_enabled=bool(getattr(runtime, "transformer_layers_compiled", False)),
        offload_between_jobs=True,
        streaming_config=StreamingWeightOffloadConfig(enabled=False),
    )
    manager.install()
    manager.after_job()
    return manager


def _target_correctness_metrics(
    rank: TrainerRank,
    items: object,
    prepared: object,
) -> dict[str, float]:
    for chunk in rank.runtime.model:
        chunk.eval()
    with torch.no_grad():
        labels = _packed_labels(items, prepared)
        native_logprobs = _native_target_logprobs(rank, items, prepared, labels)
        hidden = rank._gather_sequence_parallel_hidden(rank._decoder_hidden(prepared))
        head_outputs = rank._project_head(items, prepared, hidden)
        abs_diff_sum = torch.tensor(0.0, device=rank.device)
        reference_abs_sum = torch.tensor(0.0, device=rank.device)
        value_count = torch.tensor(0.0, device=rank.device)
        max_abs_diff = torch.tensor(0.0, device=rank.device)
        for native, candidate in zip(
            native_logprobs,
            (output.target_logprobs for output in head_outputs),
            strict=True,
        ):
            if candidate is None:
                continue
            diff = (candidate.float() - native.float()).abs()
            if int(diff.numel()) == 0:
                continue
            abs_diff_sum += diff.sum()
            reference_abs_sum += native.float().abs().sum()
            value_count += float(diff.numel())
            max_abs_diff = torch.maximum(max_abs_diff, diff.max())
        sums = torch.stack((abs_diff_sum, reference_abs_sum, value_count))
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
        mean_abs_pct = float((sums[0] / torch.clamp(sums[1], min=1e-18)).item())
        max_abs = float(max_abs_diff.item())
    return {
        "target_hidden_vs_native_mean_abs_pct": mean_abs_pct,
        "target_hidden_vs_native_max_abs_diff": max_abs,
        "target_hidden_vs_native_value_count": float(sums[2].item()),
    }


def _native_target_logprobs(
    rank: TrainerRank,
    items: object,
    prepared: object,
    labels: torch.Tensor,
) -> list[torch.Tensor]:
    from art.megatron.train import _placeholder_attention_mask

    per_token_loss = rank.runtime.model[0](
        input_ids=prepared.tokens,
        position_ids=prepared.position_ids,
        attention_mask=_placeholder_attention_mask(rank.device),
        labels=labels,
        packed_seq_params=prepared.packed_seq_params,
        **rank.runtime.model_support_handler.get_forward_kwargs(
            rank.runtime.model[0],
            attention_bias=prepared.attention_state,
        ),
    )
    flat_logprobs = -per_token_loss.reshape(-1)
    outputs: list[torch.Tensor] = []
    for item, positions, source_positions in zip(
        items,
        prepared.positions_by_item,
        prepared.source_positions_by_item,
        strict=True,
    ):
        if item.labels is None:
            raise RuntimeError("native target oracle requires labels")
        item_labels = item.labels.to(device=rank.device).index_select(
            0,
            source_positions.to(device=rank.device),
        )
        outputs.append(
            flat_logprobs.index_select(0, positions.to(device=rank.device)).masked_fill(
                item_labels == -100,
                0.0,
            )
        )
    return outputs


def _adapter_sanity_metrics(
    rank: TrainerRank,
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    params: AdamParams,
    adapter_slots: int,
) -> dict[str, float]:
    target_request = next(
        (request for request in requests if request.target_tokens is not None),
        None,
    )
    if target_request is None:
        return {"adapter_sanity_skipped": 1.0}
    base_request = ForwardInput(
        input_tokens=target_request.input_tokens,
        target_tokens=target_request.target_tokens,
        checkpoint=None,
    )
    slot_request = ForwardInput(
        input_tokens=target_request.input_tokens,
        target_tokens=target_request.target_tokens,
        checkpoint="S0",
    )
    for chunk in rank.runtime.model:
        chunk.eval()
    with torch.no_grad():
        base_output = rank.dp_rank_forward([base_request])[0]
        slot_output = rank.dp_rank_forward([slot_request])[0]
        if base_output.target_logprobs is None or slot_output.target_logprobs is None:
            raise RuntimeError("adapter sanity target outputs were not produced")
        output_diff = _mean_abs_pct(
            base_output.target_logprobs,
            slot_output.target_logprobs,
        )
        output_max = float(
            (slot_output.target_logprobs.float() - base_output.target_logprobs.float())
            .abs()
            .max()
            .item()
        )

    slot_params = list(rank._checkpoint_slot_params_by_name["S0"])
    other_params = (
        list(rank._checkpoint_slot_params_by_name["S1"]) if adapter_slots > 1 else []
    )
    before = [param.detach().clone() for param in slot_params]
    other_before = [param.detach().clone() for param in other_params]
    for chunk in rank.runtime.model:
        chunk.train()
    rank.zero_grad()
    loss = _target_requests_loss(rank, [slot_request])
    loss.backward()
    grad_sq = torch.tensor(0.0, device=rank.device)
    for param in slot_params:
        if param.grad is not None:
            grad_sq = grad_sq + param.grad.detach().float().square().sum()
    grad_norm = torch.sqrt(grad_sq)
    rank.optim_step(params=params, checkpoints=["S0"])
    slot_delta = sum(
        float((param.detach().float() - old.float()).abs().sum().item())
        for param, old in zip(slot_params, before, strict=True)
    )
    other_delta = sum(
        float((param.detach().float() - old.float()).abs().sum().item())
        for param, old in zip(other_params, other_before, strict=True)
    )
    values = torch.tensor(
        [output_diff, output_max, float(grad_norm.item()), slot_delta, other_delta],
        device=rank.device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return {
        "adapter_sanity_output_mean_abs_pct": float(values[0].item()),
        "adapter_sanity_output_max_abs_diff": float(values[1].item()),
        "adapter_sanity_grad_norm": float(values[2].item()),
        "adapter_sanity_stepped_slot_delta": float(values[3].item()),
        "adapter_sanity_unselected_slot_delta": float(values[4].item()),
    }


def _runtime_output_shape(runtime: object) -> tuple[int, int, int]:
    provider = getattr(runtime, "provider")
    model = _language_model(getattr(runtime, "model")[0])
    hidden_size = int(
        getattr(provider, "hidden_size", None)
        or getattr(getattr(model, "config", None), "hidden_size", 0)
    )
    vocab_size = int(
        getattr(getattr(model, "config", None), "padded_vocab_size", None)
        or getattr(model, "vocab_size", 0)
    )
    dtype_size = next(getattr(runtime, "model")[0].parameters()).element_size()
    if hidden_size <= 0 or vocab_size <= 0:
        raise RuntimeError(
            f"could not infer output shape: hidden_size={hidden_size}, "
            f"vocab_size={vocab_size}"
        )
    return hidden_size, vocab_size, dtype_size


def _request_output_gb(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    *,
    hidden_size: int,
    vocab_size: int,
    dtype_size: int,
) -> float:
    return (
        sum(
            _request_output_bytes(
                request,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dtype_size=dtype_size,
            )
            for request in requests
        )
        / 1024**3
    )


def _request_output_bytes(
    request: ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ],
    *,
    hidden_size: int,
    vocab_size: int,
    dtype_size: int,
) -> int:
    seq_len = int(request.input_tokens.numel())
    bytes_total = 0
    if request.target_tokens is not None:
        bytes_total += int(request.target_tokens.numel()) * 4
    if request.top_k is not None:
        bytes_total += seq_len * int(request.top_k) * (4 + 8)
    if request.logits:
        bytes_total += seq_len * vocab_size * dtype_size
    if request.hidden_states:
        bytes_total += seq_len * hidden_size * dtype_size
    return bytes_total


def _logits_requests(
    requests: Sequence[ForwardInput[torch.Tensor, None, None, None]],
) -> list[ForwardInput[None, None, torch.Tensor, None]]:
    return [
        ForwardInput(input_tokens=request.input_tokens, logits=True)
        for request in requests
    ]


def _rate_units(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    stats: dict[str, int | str],
    *,
    hidden_size: int,
    vocab_size: int,
    dtype_size: int,
) -> dict[str, int]:
    return {
        "packed_tokens": int(stats.get("packed_tokens", 0)),
        "logical_tokens": int(stats.get("logical_tokens", 0)),
        "target_values": _target_value_count(requests),
        "output_bytes": sum(
            _request_output_bytes(
                request,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dtype_size=dtype_size,
            )
            for request in requests
        ),
    }


def _target_value_count(
    requests: Sequence[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> int:
    count = 0
    for request in requests:
        if request.target_tokens is not None:
            count += int((request.target_tokens != -100).sum().item())
    return count


def _rate_metrics(
    results: dict[str, float],
    units_by_name: dict[str, dict[str, int]],
) -> dict[str, float]:
    suffixes = {
        "packed_tokens": "packed_tok_s",
        "logical_tokens": "logical_tok_s",
        "target_values": "target_logprob_s",
        "output_bytes": "output_gb_s",
    }
    metrics: dict[str, float] = {}
    for key, ms in results.items():
        if ms <= 0:
            continue
        name = key.removesuffix("_ms")
        units = units_by_name.get(name, {})
        for unit_key, suffix in suffixes.items():
            value = int(units.get(unit_key, 0))
            if value <= 0:
                continue
            scale = 1024**3 if unit_key == "output_bytes" else 1
            metrics[f"{name}_{suffix}"] = round(value * 1000.0 / ms / scale, 3)
    return metrics


def _packed_labels(items: object, prepared: object) -> torch.Tensor:
    labels = torch.full_like(prepared.tokens, -100)
    for item, positions, source_positions in zip(
        items,
        prepared.positions_by_item,
        prepared.source_positions_by_item,
        strict=True,
    ):
        if item.labels is None:
            continue
        labels.reshape(-1)[positions.to(device=labels.device)] = item.labels.to(
            device=labels.device
        ).index_select(0, source_positions.to(device=labels.device))
    return labels


if __name__ == "__main__":
    typer.run(main)
