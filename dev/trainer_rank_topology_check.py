from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time

import torch
import torch.distributed as dist
import typer

from art.megatron.shared_prefix_packing import SharedPrefixPack, pack_shared_prefixes
from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    TopK,
    TrainerRank,
    _batch_seq_logits,
    _language_model,
    _select_positions,
)


@dataclass
class CheckOutput:
    source_positions: torch.Tensor
    target_logprobs: torch.Tensor | None
    top_k: TopK | None
    logits: torch.Tensor | None
    hidden_states: torch.Tensor | None


@dataclass(frozen=True)
class DiffStats:
    max_abs_diff: float = 0.0
    mean_abs_pct: float = 0.0

    def merge(self, other: DiffStats) -> DiffStats:
        return DiffStats(
            max_abs_diff=max(self.max_abs_diff, other.max_abs_diff),
            mean_abs_pct=max(self.mean_abs_pct, other.mean_abs_pct),
        )


def _gather_target_logprobs(
    logprobs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if int(labels.shape[0]) == 0:
        return torch.empty(labels.shape, device=logprobs.device, dtype=logprobs.dtype)
    flat_labels = labels.clamp_min(0).reshape(int(labels.shape[0]), -1)
    selected = logprobs.gather(1, flat_labels).reshape(labels.shape)
    return selected.masked_fill(labels == -100, 0.0)


def _empty_logits_like_positions(
    positions: torch.Tensor,
    model: object,
    like: torch.Tensor,
) -> torch.Tensor:
    vocab_size = getattr(
        getattr(model, "config", None),
        "padded_vocab_size",
        None,
    ) or getattr(model, "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("could not determine full padded vocabulary size")
    return torch.empty(
        (int(positions.numel()), int(vocab_size)),
        device=like.device,
        dtype=like.dtype,
    )


def main(
    model: str = "Qwen/Qwen3-0.6B",
    layers: int = 1,
    head_chunk_a: int = 17,
    head_chunk_b: int = 512,
    max_prefix_depth: int = 1,
    request_case: str = "shared",
    stress_tokens: int = 0,
    max_unpacked_output_gb: float = 0.25,
    debug_output: str = "none",
    compare_independent: bool = False,
    compare_same_layout: bool = False,
) -> None:
    os.environ.setdefault("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        from megatron.core import parallel_state as ps

        from art.megatron import train as megatron_train

        torch.manual_seed(1234)
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

        requests = (
            _stress_requests(stress_tokens)
            if stress_tokens > 0
            else _requests(request_case)
        )
        requests = _debug_output_requests(requests, debug_output)
        unpacked_output_gb = _estimate_unpacked_output_gb(requests, runtime)
        if max_unpacked_output_gb > 0 and unpacked_output_gb > max_unpacked_output_gb:
            if dist.get_rank() == 0:
                print(
                    json.dumps(
                        {
                            "world": dist.get_world_size(),
                            "dp": int(ps.get_data_parallel_world_size()),
                            "tp": int(ps.get_tensor_model_parallel_world_size()),
                            "cp": int(ps.get_context_parallel_world_size()),
                            "stress_tokens": stress_tokens,
                            "estimated_unpacked_output_gb": round(
                                unpacked_output_gb, 3
                            ),
                            "max_unpacked_output_gb": max_unpacked_output_gb,
                            "skipped": "unpacked_output_cap",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            dist.barrier()
            return
        dp_rank = int(ps.get_data_parallel_rank())
        dp_size = int(ps.get_data_parallel_world_size())
        local_pairs = [
            (index, request)
            for index, request in enumerate(requests)
            if index % dp_size == dp_rank
        ]
        local_requests = [request for _, request in local_pairs]

        rank_a = TrainerRank(
            runtime,
            head_chunk_tokens=head_chunk_a,
            shared_prefix_max_depth=max_prefix_depth,
        )
        rank_b = TrainerRank(
            runtime,
            head_chunk_tokens=head_chunk_b,
            shared_prefix_max_depth=max_prefix_depth,
        )
        independent_outputs: list[CheckOutput] | None = None
        same_layout_outputs: list[CheckOutput] | None = None

        torch.cuda.reset_peak_memory_stats()
        diff_stats = DiffStats()
        with torch.no_grad():
            started_at = time.perf_counter()
            if request_case == "target_only":
                _debug("forward-target-only")
                outputs_a = list(rank_a.dp_rank_forward(local_requests))
                outputs_b = list(rank_b.dp_rank_forward(local_requests))
                oracle_outputs, actual_source_positions = _packed_oracle(
                    rank_a, local_requests
                )
            elif stress_tokens > 0:
                _debug("forward-a")
                outputs_a = list(rank_a.dp_rank_forward(local_requests))
                outputs_b = outputs_a
                actual_source_positions = _source_positions(rank_a, local_requests)
                oracle_outputs = [
                    _as_check_output(source_positions, output)
                    for source_positions, output in zip(
                        actual_source_positions,
                        outputs_a,
                        strict=True,
                    )
                ]
            else:
                _debug("forward-shared")
                (
                    outputs_a,
                    outputs_b,
                    oracle_outputs,
                    actual_source_positions,
                ) = _shared_hidden_check(rank_a, rank_b, local_requests)
                if compare_independent and request_case in {"shared", "unique", "deep"}:
                    independent_outputs = _independent_check_outputs(
                        rank_a, local_requests
                    )
                    if int(ps.get_context_parallel_world_size()) <= 1:
                        for index, (actual, independent) in enumerate(
                            zip(outputs_a, independent_outputs, strict=True)
                        ):
                            diff_stats = diff_stats.merge(
                                _assert_close(
                                    actual,
                                    independent,
                                    f"independent[{index}]",
                                ),
                            )
                if compare_same_layout and request_case in {"shared", "unique", "deep"}:
                    same_layout_outputs = _same_layout_check_outputs(
                        rank_a,
                        local_requests,
                    )
                    for index, (actual, same_layout) in enumerate(
                        zip(outputs_a, same_layout_outputs, strict=True)
                    ):
                        diff_stats = diff_stats.merge(
                            _assert_close(
                                actual,
                                same_layout,
                                f"same_layout[{index}]",
                            ),
                        )
            _debug("compare")
            elapsed_s = time.perf_counter() - started_at

        peak_memory_gb = torch.tensor(
            torch.cuda.max_memory_allocated() / 1024**3,
            device=rank_a.device,
        )
        for index, (actual, chunked, oracle) in enumerate(
            zip(outputs_a, outputs_b, oracle_outputs, strict=True)
        ):
            if int(oracle.source_positions.numel()) == 0:
                continue
            diff_stats = diff_stats.merge(
                _assert_close(actual, chunked, f"chunk[{index}]"),
            )
            diff_stats = diff_stats.merge(
                _assert_close(actual, oracle, f"oracle[{index}]"),
            )

        diff_tensor = torch.tensor(
            [diff_stats.max_abs_diff, diff_stats.mean_abs_pct],
            device=rank_a.device,
        )
        dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_memory_gb, op=dist.ReduceOp.MAX)
        max_diff_value = float(diff_tensor[0].item())
        mean_abs_pct_value = float(diff_tensor[1].item())
        records = _records(
            local_pairs=local_pairs,
            actual_outputs=outputs_a,
            actual_source_positions=actual_source_positions,
            oracle_outputs=oracle_outputs,
            independent_outputs=independent_outputs,
            rank=int(dist.get_rank()),
            dp=dp_rank,
            tp=int(ps.get_tensor_model_parallel_rank()),
            cp=int(ps.get_context_parallel_rank()),
        )
        gathered: list[list[dict[str, object]] | None] = [None] * dist.get_world_size()
        _debug("all-gather")
        dist.all_gather_object(gathered, records)
        _debug("reconstruct")
        reconstruction_error: str | None = None
        if dist.get_rank() == 0:
            seen = {
                record["input_index"]
                for rank_records in gathered
                for record in rank_records or []
            }
            if seen != set(range(len(requests))):
                reconstruction_error = f"DP reconstruction missed inputs: {seen}"
            else:
                try:
                    reconstructed_stats = _assert_reconstructed(gathered, requests)
                    max_diff_value = max(
                        max_diff_value,
                        reconstructed_stats.max_abs_diff,
                    )
                    mean_abs_pct_value = max(
                        mean_abs_pct_value,
                        reconstructed_stats.mean_abs_pct,
                    )
                except AssertionError as exc:
                    reconstruction_error = str(exc)
            if reconstruction_error is None:
                print(
                    json.dumps(
                        {
                            "world": dist.get_world_size(),
                            "dp": dp_size,
                            "tp": int(ps.get_tensor_model_parallel_world_size()),
                            "cp": int(ps.get_context_parallel_world_size()),
                            "mean_abs_pct": mean_abs_pct_value,
                            "max_abs_diff": max_diff_value,
                            "records": sum(
                                len(rank_records or []) for rank_records in gathered
                            ),
                            "same_layout": compare_same_layout,
                            "stress_tokens": stress_tokens,
                            "estimated_unpacked_output_gb": round(
                                unpacked_output_gb, 3
                            ),
                            "elapsed_s": round(elapsed_s, 3),
                            "peak_memory_gb": round(float(peak_memory_gb.item()), 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        errors = [reconstruction_error]
        dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            raise AssertionError(errors[0])
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _requests(
    request_case: str = "shared",
) -> list[
    ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ]
]:
    if request_case not in {"shared", "target_only", "unique", "deep"}:
        raise ValueError(
            "request_case must be 'shared', 'target_only', 'unique', or 'deep'"
        )
    rows = [
        torch.tensor([11, 12, 13, 14, 15, 16, 17]),
        torch.tensor([11, 12, 13, 14, 24, 25]),
        torch.tensor([11, 12, 13, 14, 24, 26]),
        torch.tensor([11, 12, 13, 27]),
        torch.tensor([31, 32, 33, 34]),
        torch.tensor([31, 32, 33, 35]),
        torch.tensor([11, 12, 13, 14, 15, 16, 17]),
        torch.tensor([41, 42, 43]),
        torch.tensor([41, 42, 44, 45]),
        torch.tensor([51, 52, 53, 54, 55]),
        torch.tensor([61, 62, 63]),
        torch.tensor([61, 62, 64, 65]),
        torch.tensor([71, 72]),
        torch.tensor([81, 82, 83, 84]),
        torch.tensor([91, 92, 93]),
        torch.tensor([101, 102, 103, 104, 105]),
    ]
    if request_case == "deep":
        rows = _deep_rows()
    if request_case == "unique":
        rows = [row + 1000 * index for index, row in enumerate(rows)]
    if request_case == "target_only":
        target_only_labels = [_labels(row, 0) for row in rows]
        target_only_labels[0][2] = -100
        target_only_labels[3][1] = -100
        target_only_labels[10][0] = -100
        return [
            ForwardInput(input_tokens=row, target_tokens=label)
            for row, label in zip(rows, target_only_labels, strict=True)
        ]

    labels = [_labels(row, offset) for offset, row in enumerate(rows)]
    labels[0][2] = -100
    labels[3][1] = -100
    labels[10][0] = -100
    multi_labels = torch.stack((labels[1], (labels[1] + 17) % 1000), dim=1)
    multi_labels[2, 1] = -100
    requests = []
    for mask, row in enumerate(rows):
        target_tokens = None
        if mask & 1:
            target_tokens = multi_labels if mask == 1 else labels[mask]
        requests.append(
            ForwardInput(
                input_tokens=row,
                target_tokens=target_tokens,
                top_k=3 if mask & 2 else None,
                logits=bool(mask & 4),
                hidden_states=bool(mask & 8),
            )
        )
    return requests


def _debug_output_requests(
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    debug_output: str,
) -> list[
    ForwardInput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ]
]:
    if debug_output == "none":
        return requests
    if debug_output == "hidden":
        return [
            ForwardInput(input_tokens=request.input_tokens, hidden_states=True)
            for request in requests
        ]
    if debug_output == "logits":
        return [
            ForwardInput(input_tokens=request.input_tokens, logits=True)
            for request in requests
        ]
    if debug_output == "target":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                target_tokens=_labels(request.input_tokens, 0),
            )
            for request in requests
        ]
    if debug_output == "topk":
        return [
            ForwardInput(input_tokens=request.input_tokens, top_k=3)
            for request in requests
        ]
    if debug_output == "target_topk":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                target_tokens=_labels(request.input_tokens, 0),
                top_k=3,
            )
            for request in requests
        ]
    if debug_output == "mixed_no_topk":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                target_tokens=request.target_tokens,
                logits=request.logits,
                hidden_states=request.hidden_states,
            )
            for request in requests
        ]
    if debug_output == "mixed_no_logits":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                target_tokens=request.target_tokens,
                top_k=request.top_k,
                hidden_states=request.hidden_states,
            )
            for request in requests
        ]
    if debug_output == "mixed_no_targets":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                top_k=request.top_k,
                logits=request.logits,
                hidden_states=request.hidden_states,
            )
            for request in requests
        ]
    if debug_output == "mixed_targets_only":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                target_tokens=request.target_tokens,
            )
            for request in requests
        ]
    if debug_output == "mixed_targets_hidden":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                target_tokens=request.target_tokens,
                hidden_states=request.hidden_states,
            )
            for request in requests
        ]
    if debug_output == "mixed_targets_logits":
        return [
            ForwardInput(
                input_tokens=request.input_tokens,
                target_tokens=request.target_tokens,
                logits=request.logits,
            )
            for request in requests
        ]
    raise ValueError(
        "debug_output must be 'none', 'hidden', 'logits', 'target', 'topk', "
        "'target_topk', 'mixed_no_topk', 'mixed_no_logits', 'mixed_no_targets', "
        "'mixed_targets_only', 'mixed_targets_hidden', or 'mixed_targets_logits'"
    )


def _deep_rows() -> list[torch.Tensor]:
    return [
        torch.tensor([11, 12, 13, 14, 15, 16, 17]),
        torch.tensor([11, 12, 13, 14, 15, 16, 18]),
        torch.tensor([11, 12, 13, 14, 15, 19]),
        torch.tensor([11, 12, 13, 14, 20]),
        torch.tensor([11, 12, 21]),
        torch.tensor([31, 32, 33, 34, 35]),
        torch.tensor([31, 32, 33, 34, 36]),
        torch.tensor([31, 32, 33, 37]),
        torch.tensor([41, 42, 43]),
        torch.tensor([41, 42, 44]),
        torch.tensor([51, 52, 53, 54]),
        torch.tensor([61, 62]),
        torch.tensor([71, 72, 73, 74, 75]),
        torch.tensor([71, 72, 73, 76]),
        torch.tensor([81]),
        torch.tensor([91, 92, 93]),
    ]


def _stress_requests(
    token_count: int,
) -> list[ForwardInput[None, None, None, torch.Tensor]]:
    if token_count < 8:
        raise ValueError("stress_tokens must be >= 8")
    prefix_len = token_count // 2
    tail_len = max(1, token_count // 4)
    prefix = _stress_tokens(0, prefix_len)
    return [
        ForwardInput(
            input_tokens=torch.cat((prefix, _stress_tokens(10_000, tail_len))),
            hidden_states=True,
        ),
        ForwardInput(
            input_tokens=torch.cat((prefix, _stress_tokens(20_000, tail_len))),
            hidden_states=True,
        ),
        ForwardInput(input_tokens=_stress_tokens(30_000, tail_len), hidden_states=True),
    ]


def _stress_tokens(offset: int, length: int) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) + offset) % 32_000 + 100


def _estimate_unpacked_output_gb(
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    runtime: object,
) -> float:
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
    bytes_total = sum(
        _request_output_bytes(
            request,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dtype_size=dtype_size,
        )
        for request in requests
    )
    return bytes_total / 1024**3


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


def _debug(label: str) -> None:
    if os.environ.get("TRAINER_RANK_CHECK_DEBUG") != "1":
        return
    print(f"[rank{dist.get_rank()}] {label}", flush=True)


def _labels(tokens: torch.Tensor, offset: int) -> torch.Tensor:
    return ((tokens * 7 + 3 + offset) % 1000).to(dtype=torch.long)


def _packed_oracle(
    rank: TrainerRank,
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> tuple[list[CheckOutput], tuple[torch.Tensor, ...]]:
    items = [rank._forward_item(request) for request in requests]
    prepared = rank._prepare_packed_forward(
        pack_shared_prefixes(
            (item.input_ids for item in items),
            max_depth=rank.shared_prefix_max_depth,
        )
    )
    hidden = rank._gather_sequence_parallel_hidden(rank._decoder_hidden(prepared))
    return (
        _packed_oracle_from_hidden(rank, items, prepared, hidden),
        prepared.source_positions_by_item,
    )


def _shared_hidden_check(
    rank_a: TrainerRank,
    rank_b: TrainerRank,
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> tuple[
    list[
        ForwardOutput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    list[
        ForwardOutput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    list[CheckOutput],
    tuple[torch.Tensor, ...],
]:
    items = [rank_a._forward_item(request) for request in requests]
    prepared = rank_a._prepare_packed_forward(
        pack_shared_prefixes(
            (item.input_ids for item in items),
            max_depth=rank_a.shared_prefix_max_depth,
        )
    )
    hidden = rank_a._gather_sequence_parallel_hidden(rank_a._decoder_hidden(prepared))
    outputs_a = _outputs_from_hidden(rank_a, items, prepared, hidden)
    outputs_b = _outputs_from_hidden(rank_b, items, prepared, hidden)
    oracle = _packed_oracle_from_hidden(rank_a, items, prepared, hidden)
    return (
        outputs_a,
        outputs_b,
        oracle,
        prepared.source_positions_by_item,
    )


def _independent_check_outputs(
    rank: TrainerRank,
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> list[CheckOutput]:
    outputs: list[CheckOutput] = []
    for request in requests:
        source_positions = _source_positions(rank, [request])[0]
        outputs.append(
            _as_check_output(source_positions, rank.dp_rank_forward([request])[0])
        )
    return outputs


def _same_layout_check_outputs(
    rank: TrainerRank,
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> list[CheckOutput]:
    items = [rank._forward_item(request) for request in requests]
    batch = pack_shared_prefixes(
        (item.input_ids for item in items),
        max_depth=rank.shared_prefix_max_depth,
    )
    outputs = []
    for index, positions in enumerate(batch.positions_by_sequence):
        mutated = _mutated_batch(batch, keep_positions=positions)
        prepared = rank._prepare_packed_forward(mutated)
        hidden = rank._gather_sequence_parallel_hidden(rank._decoder_hidden(prepared))
        mutated_outputs = _outputs_from_hidden(rank, items, prepared, hidden)
        outputs.append(
            _as_check_output(
                prepared.source_positions_by_item[index],
                mutated_outputs[index],
            )
        )
    return outputs


def _mutated_batch(
    batch: SharedPrefixPack,
    *,
    keep_positions: torch.Tensor,
) -> SharedPrefixPack:
    tokens = batch.tokens.clone()
    mutate = torch.ones(int(tokens.shape[1]), dtype=torch.bool, device=tokens.device)
    mutate[keep_positions.to(device=tokens.device)] = False
    replacement = (
        torch.arange(int(tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
        + 50_000
    )
    tokens[0, mutate] = replacement[mutate] % 100_000
    return SharedPrefixPack(
        tokens=tokens,
        group_ids=batch.group_ids,
        parent_ids=batch.parent_ids,
        position_ids=batch.position_ids,
        positions_by_sequence=batch.positions_by_sequence,
    )


def _outputs_from_hidden(
    rank: TrainerRank,
    items: list[object],
    prepared: object,
    hidden: torch.Tensor,
) -> list[
    ForwardOutput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ]
]:
    return rank._project_head(items, prepared, hidden)


def _packed_oracle_from_hidden(
    rank: TrainerRank,
    items: list[object],
    prepared: object,
    hidden: torch.Tensor,
) -> list[CheckOutput]:
    model = _language_model(rank.runtime.model[0])
    output_weight = (
        model.shared_embedding_or_output_weight()
        if bool(model.share_embeddings_and_output_weights)
        else None
    )

    outputs: list[CheckOutput] = []
    for item, positions, source_positions in zip(
        items,
        prepared.positions_by_item,
        prepared.source_positions_by_item,
        strict=True,
    ):
        needs_projection = (
            item.labels is not None or item.request.logits or item.request.top_k
        )
        all_logits = None
        if needs_projection:
            if int(positions.numel()):
                local_logits = rank._local_logits_from_hidden_rows(
                    model,
                    _select_positions(hidden, positions),
                    output_weight=output_weight,
                )
                all_logits = _batch_seq_logits(
                    rank._gather_tensor_parallel_logits(local_logits.unsqueeze(1)),
                    seq_len=int(positions.numel()),
                ).squeeze(0)
            else:
                all_logits = _empty_logits_like_positions(positions, model, hidden)
        logprobs = (
            None
            if all_logits is None
            else torch.log_softmax(all_logits.float(), dim=-1)
        )

        target_logprobs = None
        if item.labels is not None:
            if logprobs is None:
                raise RuntimeError("target_logprobs oracle requires logprobs")
            labels = item.labels.to(device=logprobs.device).index_select(
                0, source_positions.to(device=logprobs.device)
            )
            target_logprobs = _gather_target_logprobs(logprobs, labels)

        top_k = None
        if item.request.top_k is not None:
            if all_logits is None:
                raise RuntimeError("top_k oracle requires logits")
            log_z = torch.logsumexp(all_logits.float(), dim=-1)
            values, tokens = torch.topk(
                all_logits.float(), k=item.request.top_k, dim=-1
            )
            top_k = TopK(logprobs=values - log_z.unsqueeze(1), tokens=tokens)

        hidden_states = None
        if item.request.hidden_states:
            hidden_states = _select_positions(hidden, positions)

        outputs.append(
            CheckOutput(
                source_positions=source_positions,
                target_logprobs=target_logprobs,
                top_k=top_k,
                logits=all_logits if item.request.logits else None,
                hidden_states=hidden_states,
            )
        )
    return outputs


def _source_positions(
    rank: TrainerRank,
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> tuple[torch.Tensor, ...]:
    items = [rank._forward_item(request) for request in requests]
    prepared = rank._prepare_packed_forward(
        pack_shared_prefixes(
            (item.input_ids for item in items),
            max_depth=rank.shared_prefix_max_depth,
        )
    )
    return prepared.source_positions_by_item


def _as_check_output(
    source_positions: torch.Tensor,
    output: ForwardOutput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ],
) -> CheckOutput:
    return CheckOutput(
        source_positions=source_positions,
        target_logprobs=output.target_logprobs,
        top_k=output.top_k,
        logits=output.logits,
        hidden_states=output.hidden_states,
    )


def _records(
    *,
    local_pairs: list[
        tuple[
            int,
            ForwardInput[
                torch.Tensor | None,
                TopK | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ],
        ]
    ],
    actual_outputs: list[
        ForwardOutput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
    actual_source_positions: tuple[torch.Tensor, ...],
    oracle_outputs: list[CheckOutput],
    independent_outputs: list[CheckOutput] | None,
    rank: int,
    dp: int,
    tp: int,
    cp: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    independent_records: list[CheckOutput | None] = (
        independent_outputs
        if independent_outputs is not None
        else [None] * len(local_pairs)
    )
    for local_index, (
        (input_index, _),
        actual,
        actual_sources,
        oracle,
        independent,
    ) in enumerate(
        zip(
            local_pairs,
            actual_outputs,
            actual_source_positions,
            oracle_outputs,
            independent_records,
            strict=True,
        )
    ):
        records.append(
            {
                "input_index": input_index,
                "local_index": local_index,
                "rank": rank,
                "dp": dp,
                "tp": tp,
                "cp": cp,
                "actual": _cpu_record(actual_sources, actual),
                "oracle": _cpu_record(oracle.source_positions, oracle),
                "independent": (
                    None
                    if independent is None
                    else _cpu_record(independent.source_positions, independent)
                ),
            }
        )
    return records


def _cpu_record(
    source_positions: torch.Tensor,
    output: ForwardOutput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ]
    | CheckOutput,
) -> dict[str, torch.Tensor | None]:
    return {
        "source_positions": source_positions.cpu(),
        "target_logprobs": _cpu(output.target_logprobs),
        "logits": _cpu(output.logits),
        "hidden_states": _cpu(output.hidden_states),
        "top_k_logprobs": None if output.top_k is None else _cpu(output.top_k.logprobs),
        "top_k_tokens": None if output.top_k is None else _cpu(output.top_k.tokens),
    }


def _cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return None if tensor is None else tensor.detach().cpu()


def _assert_reconstructed(
    gathered: list[list[dict[str, object]] | None],
    requests: list[
        ForwardInput[
            torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
        ]
    ],
) -> DiffStats:
    diff_stats = DiffStats()
    records = [
        record
        for rank_records in gathered
        for record in rank_records or []
        if record["tp"] == 0
    ]
    for input_index, request in enumerate(requests):
        _debug(f"reconstruct-input-{input_index}")
        actual = [
            record["actual"]
            for record in records
            if record["input_index"] == input_index
        ]
        oracle = [
            record["oracle"]
            for record in records
            if record["input_index"] == input_index
        ]
        independent = [
            record["independent"]
            for record in records
            if record["input_index"] == input_index
            and record.get("independent") is not None
        ]
        length = int(request.input_tokens.numel())
        for key in ("target_logprobs", "logits", "hidden_states", "top_k_logprobs"):
            _debug(f"reconstruct-input-{input_index}-{key}")
            _debug(f"reconstruct-input-{input_index}-{key}-assemble-actual")
            actual_value = _assemble(actual, key, length)
            _debug(
                f"reconstruct-input-{input_index}-{key}-actual-"
                f"{_tensor_summary(actual_value)}"
            )
            _debug(f"reconstruct-input-{input_index}-{key}-assemble-oracle")
            oracle_value = _assemble(oracle, key, length)
            _debug(
                f"reconstruct-input-{input_index}-{key}-oracle-"
                f"{_tensor_summary(oracle_value)}"
            )
            _debug(f"reconstruct-input-{input_index}-{key}-diff-oracle")
            diff_stats = diff_stats.merge(
                _tensor_diff_value(
                    actual_value,
                    oracle_value,
                    f"reconstructed[{input_index}].{key}",
                ),
            )
            _debug(f"reconstruct-input-{input_index}-{key}-diff-oracle-done")
            if independent:
                _debug(f"reconstruct-input-{input_index}-{key}-assemble-independent")
                independent_value = _assemble(independent, key, length)
                _debug(
                    f"reconstruct-input-{input_index}-{key}-independent-"
                    f"{_tensor_summary(independent_value)}"
                )
                _debug(f"reconstruct-input-{input_index}-{key}-diff-independent")
                diff_stats = diff_stats.merge(
                    _tensor_diff_value(
                        actual_value,
                        independent_value,
                        f"independent[{input_index}].{key}",
                    ),
                )
                _debug(f"reconstruct-input-{input_index}-{key}-diff-independent-done")
            _debug(f"reconstruct-input-{input_index}-{key}-done")
        actual_tokens = _assemble(actual, "top_k_tokens", length)
        oracle_tokens = _assemble(oracle, "top_k_tokens", length)
        if actual_tokens is None or oracle_tokens is None:
            if actual_tokens is not oracle_tokens:
                raise AssertionError(
                    f"reconstructed[{input_index}].top_k None mismatch"
                )
        elif not torch.equal(actual_tokens, oracle_tokens):
            actual_logprobs = _assemble(actual, "top_k_logprobs", length)
            oracle_logprobs = _assemble(oracle, "top_k_logprobs", length)
            if (
                actual_logprobs is None
                or oracle_logprobs is None
                or _tensor_diff_value(
                    actual_logprobs,
                    oracle_logprobs,
                    f"reconstructed[{input_index}].top_k.logprobs",
                ).max_abs_diff
                > 5e-6
            ):
                raise AssertionError(
                    f"reconstructed[{input_index}].top_k.tokens mismatch"
                )
        if independent:
            independent_tokens = _assemble(independent, "top_k_tokens", length)
            if actual_tokens is None or independent_tokens is None:
                if actual_tokens is not independent_tokens:
                    raise AssertionError(
                        f"independent[{input_index}].top_k None mismatch"
                    )
            elif not torch.equal(actual_tokens, independent_tokens):
                actual_logprobs = _assemble(actual, "top_k_logprobs", length)
                independent_logprobs = _assemble(
                    independent,
                    "top_k_logprobs",
                    length,
                )
                if (
                    actual_logprobs is None
                    or independent_logprobs is None
                    or _tensor_diff_value(
                        actual_logprobs,
                        independent_logprobs,
                        f"independent[{input_index}].top_k.logprobs",
                    ).max_abs_diff
                    > 5e-6
                ):
                    raise AssertionError(
                        f"independent[{input_index}].top_k.tokens mismatch"
                    )
    return diff_stats


def _assemble(
    records: list[object],
    key: str,
    length: int,
) -> torch.Tensor | None:
    typed_records = [record for record in records if isinstance(record, dict)]
    values = [record[key] for record in typed_records if record[key] is not None]
    if not values:
        return None
    first = values[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError(key)
    output = torch.empty((length, *first.shape[1:]), dtype=first.dtype)
    filled = torch.zeros(length, dtype=torch.bool)
    for record in typed_records:
        value = record[key]
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(key)
        positions = record["source_positions"]
        if not isinstance(positions, torch.Tensor):
            raise TypeError("source_positions")
        output[positions] = value
        filled[positions] = True
    if not bool(filled.all().item()):
        raise AssertionError(f"{key} reconstruction missed positions")
    return output


def _tensor_summary(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "None"
    return f"shape={tuple(tensor.shape)} device={tensor.device} dtype={tensor.dtype}"


def _assert_close(
    actual: ForwardOutput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ],
    expected: ForwardOutput[
        torch.Tensor | None, TopK | None, torch.Tensor | None, torch.Tensor | None
    ]
    | CheckOutput,
    label: str,
) -> DiffStats:
    diffs = [
        _tensor_diff(
            actual.target_logprobs, expected.target_logprobs, f"{label}.target_logprobs"
        )
    ]
    diffs.append(_tensor_diff(actual.logits, expected.logits, f"{label}.logits"))
    diffs.append(
        _tensor_diff(
            actual.hidden_states, expected.hidden_states, f"{label}.hidden_states"
        )
    )
    if actual.top_k is None or expected.top_k is None:
        if actual.top_k is not expected.top_k:
            raise AssertionError(f"{label}.top_k None mismatch")
    else:
        try:
            top_k_diff = _tensor_diff(
                actual.top_k.logprobs,
                expected.top_k.logprobs,
                f"{label}.top_k.logprobs",
            )
        except AssertionError as exc:
            flat_offset = int(
                (actual.top_k.logprobs.float() - expected.top_k.logprobs.float())
                .abs()
                .flatten()
                .argmax()
            )
            row, _ = divmod(flat_offset, int(actual.top_k.logprobs.shape[1]))
            raise AssertionError(
                f"{exc}; actual_row={actual.top_k.logprobs[row].tolist()} "
                f"expected_row={expected.top_k.logprobs[row].tolist()} "
                f"actual_tokens={actual.top_k.tokens[row].tolist()} "
                f"expected_tokens={expected.top_k.tokens[row].tolist()}"
            ) from exc
        diffs.append(top_k_diff)
        if (
            not torch.equal(actual.top_k.tokens, expected.top_k.tokens)
            and top_k_diff.max_abs_diff > 5e-6
        ):
            mismatch = torch.nonzero(
                actual.top_k.tokens != expected.top_k.tokens,
                as_tuple=False,
            )[0]
            row = int(mismatch[0].item())
            col = int(mismatch[1].item())
            raise AssertionError(
                f"{label}.top_k.tokens mismatch at ({row}, {col}): "
                f"actual={int(actual.top_k.tokens[row, col].item())} "
                f"expected={int(expected.top_k.tokens[row, col].item())} "
                f"actual_logprob={float(actual.top_k.logprobs[row, col].item())} "
                f"expected_logprob={float(expected.top_k.logprobs[row, col].item())}"
            )
    return _merge_diff_stats(diffs)


def _tensor_diff(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    label: str,
) -> DiffStats:
    return _tensor_diff_value(actual, expected, label)


def _tensor_diff_value(
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    label: str,
) -> DiffStats:
    if actual is None or expected is None:
        if actual is not expected:
            raise AssertionError(f"{label} None mismatch")
        return DiffStats()
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shape mismatch: {actual.shape} != {expected.shape}"
        )
    actual_for_diff = actual
    expected_for_diff = expected
    if torch.cuda.is_available():
        actual_for_diff = actual_for_diff.to(device="cuda")
        expected_for_diff = expected_for_diff.to(device="cuda")
    if actual_for_diff.numel():
        abs_diff = (actual_for_diff.float() - expected_for_diff.float()).abs()
        max_abs_diff = float(abs_diff.max().item())
        denominator = float(expected_for_diff.float().abs().mean().item())
        mean_abs_pct = float(abs_diff.mean().item()) / (denominator + 1e-18)
    else:
        max_abs_diff = 0.0
        mean_abs_pct = 0.0
    mean_abs_pct_tolerance = 5e-3 if label.startswith("independent[") else 2e-5
    max_abs_tolerance = 0.0
    _debug(
        f"{label} max_abs_diff={max_abs_diff} "
        f"mean_abs_pct={mean_abs_pct} tolerance={mean_abs_pct_tolerance}"
    )
    if mean_abs_pct > mean_abs_pct_tolerance:
        raise AssertionError(
            f"{label} mean_abs_pct {mean_abs_pct} max_abs_diff {max_abs_diff}"
        )
    if max_abs_diff > max_abs_tolerance and not actual_for_diff.is_floating_point():
        raise AssertionError(f"{label} max diff {max_abs_diff}")
    return DiffStats(max_abs_diff=max_abs_diff, mean_abs_pct=mean_abs_pct)


def _merge_diff_stats(stats: list[DiffStats]) -> DiffStats:
    merged = DiffStats()
    for stat in stats:
        merged = merged.merge(stat)
    return merged


if __name__ == "__main__":
    typer.run(main)
