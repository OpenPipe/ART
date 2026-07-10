from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
import os
import statistics
import time
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from trainer_rank_support import load_random_checkpoint_slots
import typer

from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import (
    AdamParams,
    ForwardInput,
    ForwardOutput,
    MicroBatchStats,
    TopK,
    TrainerRank,
    Unset,
)


@dataclass(frozen=True)
class Diff:
    mean_abs_pct: float = 0.0
    max_abs_diff: float = 0.0

    def merge(self, other: Diff) -> Diff:
        return Diff(
            max(self.mean_abs_pct, other.mean_abs_pct),
            max(self.max_abs_diff, other.max_abs_diff),
        )


def main(
    mode: Literal["correctness", "performance"] = "correctness",
    model: str = "Qwen/Qwen3-0.6B",
    layers: int = 1,
    depths: str = "0,1,2,3,4",
    performance_depth: int = 1,
    chunks: str = "17,512,8192",
    workload: Literal["regular", "austin", "varied"] = "regular",
    request: Literal["target", "multi", "topk", "logits", "hidden", "mixed"] = "target",
    families: int = 8,
    prefix_tokens: int = 128,
    branches: int = 4,
    completion_tokens: int = 32,
    slots: int = 0,
    adaptive: bool = False,
    optimizer_step: bool = False,
    warmup: int = 3,
    repeat: int = 10,
    output_jsonl: str = "",
) -> None:
    os.environ.setdefault("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("dev/trainer_rank_check.py requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    try:
        from art.megatron import train as megatron_train

        torch.manual_seed(1234)
        runtime = megatron_train.build_training_runtime(
            model_identifier=model,
            provider_configure=(
                (lambda provider: setattr(provider, "num_layers", layers))
                if layers > 0
                else None
            ),
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.eval()
        if mode == "correctness":
            payload = _correctness(
                runtime,
                depths=_ints(depths),
                chunks=_ints(chunks),
                slots=slots,
            )
        else:
            payload = _performance(
                runtime,
                depth=performance_depth,
                workload=workload,
                request=request,
                families=families,
                prefix_tokens=prefix_tokens,
                branches=branches,
                completion_tokens=completion_tokens,
                slots=slots,
                adaptive=adaptive,
                optimizer_step=optimizer_step,
                warmup=warmup,
                repeat=repeat,
            )
        payload.update(_topology(), model=model, layers=layers, mode=mode)
        if dist.get_rank() == 0:
            line = json.dumps(payload, sort_keys=True)
            print(line, flush=True)
            if output_jsonl:
                with open(output_jsonl, "a", encoding="utf-8") as output:
                    output.write(line + "\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _correctness(
    runtime: Any,
    *,
    depths: tuple[int, ...],
    chunks: tuple[int, ...],
    slots: int,
) -> dict[str, object]:
    assert depths and chunks, "depths and chunks must not be empty"
    slot_names = load_random_checkpoint_slots(runtime, TrainerRank(runtime), slots)
    requests = _correctness_requests(slot_names)
    reference = _global_outputs(
        TrainerRank(
            runtime,
            shared_prefix_max_depth=0,
            head_chunk_tokens=max(chunks),
        ),
        requests,
    )
    worst = Diff()
    grad_worst = Diff()
    rows: list[dict[str, object]] = []
    for depth in depths:
        depth_reference: list[dict[str, object]] | None = None
        for chunk_tokens in chunks:
            rank = TrainerRank(
                runtime,
                shared_prefix_max_depth=depth,
                head_chunk_tokens=chunk_tokens,
            )
            outputs = _global_outputs(rank, requests)
            if dist.get_rank() == 0:
                assert reference is not None and outputs is not None
                independent_diff = _compare_outputs(
                    outputs,
                    reference,
                    tolerance=5e-3,
                )
                chunk_diff = (
                    Diff()
                    if depth_reference is None
                    else _compare_outputs(outputs, depth_reference, tolerance=2e-5)
                )
                depth_reference = outputs
                worst = worst.merge(independent_diff).merge(chunk_diff)
                rows.append(
                    {
                        "depth": depth,
                        "head_chunk_tokens": chunk_tokens,
                        "independent_mean_abs_pct": independent_diff.mean_abs_pct,
                        "chunk_mean_abs_pct": chunk_diff.mean_abs_pct,
                    }
                )
                print(rows[-1], flush=True)
        grad_diff = _head_backward_chunk_parity(
            runtime, requests, depth=depth, chunks=chunks
        )
        grad_worst = grad_worst.merge(grad_diff)
    return {
        "request_combinations": 16,
        "slots": slots,
        "rows": rows,
        "mean_abs_pct": worst.mean_abs_pct,
        "max_abs_diff": worst.max_abs_diff,
        "head_backward_mean_abs_pct": grad_worst.mean_abs_pct,
        "head_backward_max_abs_diff": grad_worst.max_abs_diff,
    }


def _local_outputs(
    rank: TrainerRank,
    indexed_requests: Sequence[tuple[int, ForwardInput]],
) -> list[dict[str, object]]:
    from art.megatron.lora import use_lora_slot

    requests = [request for _, request in indexed_requests]
    plan = rank._plan_flat_forward(requests)
    outputs: list[ForwardOutput] = [
        ForwardOutput(None, None, None, None) for _ in requests
    ]
    sources: list[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in requests]
    for group in plan.groups:
        prepared = rank._prepare_packed_forward(group.packed)
        with use_lora_slot(group.slot_ref):
            group_outputs = rank._forward_packed(group.items, prepared)
        for index, source, output in zip(
            group.request_indices,
            prepared.source_positions_by_item,
            group_outputs,
            strict=True,
        ):
            sources[index] = source
            outputs[index] = output
    return [
        _output_record(global_index, source, output)
        for (global_index, _), source, output in zip(
            indexed_requests,
            sources,
            outputs,
            strict=True,
        )
    ]


def _global_outputs(
    rank: TrainerRank,
    requests: Sequence[ForwardInput],
) -> list[dict[str, object]] | None:
    from megatron.core import parallel_state as ps

    dp_rank = int(ps.get_data_parallel_rank())
    dp_size = int(ps.get_data_parallel_world_size())
    indexed = list(enumerate(requests))[dp_rank::dp_size]
    local = _local_outputs(rank, indexed)
    gathered: list[list[dict[str, object]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if dist.get_rank() != 0:
        return None
    records = [
        record for rank_records in gathered if rank_records for record in rank_records
    ]
    return [
        _reconstruct(index, request, records) for index, request in enumerate(requests)
    ]


def _output_record(
    index: int,
    source_positions: torch.Tensor,
    output: ForwardOutput,
) -> dict[str, object]:
    return {
        "index": index,
        "source": source_positions.detach().cpu(),
        "target": _cpu(output.target_logprobs),
        "topk_logprobs": _cpu(None if output.top_k is None else output.top_k.logprobs),
        "topk_tokens": _cpu(None if output.top_k is None else output.top_k.tokens),
        "logits": _cpu(output.logits),
        "hidden": _cpu(output.hidden_states),
    }


def _reconstruct(
    index: int,
    request: ForwardInput,
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    selected = [record for record in records if record["index"] == index]
    return {
        key: _reconstruct_tensor(
            selected,
            key,
            length=int(request.input_tokens.numel()),
        )
        for key in ("target", "topk_logprobs", "topk_tokens", "logits", "hidden")
    }


def _reconstruct_tensor(
    records: Sequence[dict[str, object]],
    key: str,
    *,
    length: int,
) -> torch.Tensor | None:
    values = [
        record[key] for record in records if isinstance(record[key], torch.Tensor)
    ]
    if not values:
        return None
    first = cast(torch.Tensor, values[0])
    output = torch.empty((length, *first.shape[1:]), dtype=first.dtype)
    filled = torch.zeros(length, dtype=torch.bool)
    for record in records:
        value = record[key]
        if not isinstance(value, torch.Tensor):
            continue
        source = cast(torch.Tensor, record["source"])
        duplicate = filled.index_select(0, source)
        if bool(duplicate.any()):
            torch.testing.assert_close(
                output.index_select(0, source[duplicate]),
                value[duplicate],
                atol=2e-5,
                rtol=2e-5,
            )
        output[source] = value
        filled[source] = True
    assert bool(filled.all()), f"{key} reconstruction missed positions"
    return output


def _compare_outputs(
    actual: Sequence[dict[str, object]],
    expected: Sequence[dict[str, object]],
    *,
    tolerance: float,
) -> Diff:
    worst = Diff()
    for actual_output, expected_output in zip(actual, expected, strict=True):
        for key in actual_output:
            actual_tensor = actual_output[key]
            expected_tensor = expected_output[key]
            if actual_tensor is None or expected_tensor is None:
                if actual_tensor is not expected_tensor:
                    raise AssertionError(f"{key} None mismatch")
                continue
            assert isinstance(actual_tensor, torch.Tensor)
            assert isinstance(expected_tensor, torch.Tensor)
            if key == "topk_tokens":
                if not torch.equal(
                    actual_tensor.sort(dim=-1).values,
                    expected_tensor.sort(dim=-1).values,
                ):
                    raise AssertionError("top-k token sets differ")
                continue
            diff = _diff(actual_tensor, expected_tensor)
            if diff.mean_abs_pct > tolerance:
                raise AssertionError(
                    f"{key} mean_abs_pct={diff.mean_abs_pct} exceeds {tolerance}"
                )
            worst = worst.merge(diff)
    return worst


def _head_backward_chunk_parity(
    runtime: Any,
    requests: Sequence[ForwardInput],
    *,
    depth: int,
    chunks: Sequence[int],
) -> Diff:
    active = [
        request
        for request in requests
        if request.target_tokens is not None or request.top_k is not None
    ]
    rank = TrainerRank(
        runtime,
        shared_prefix_max_depth=depth,
        head_chunk_tokens=chunks[0],
    )
    items = [rank._forward_item(request) for request in active]
    prepared = rank._prepare_packed_forward(
        prefix_tree_pack(
            (item.input_ids for item in items),
            max_depth=depth,
        )
    )
    with torch.no_grad():
        hidden = rank._gather_sequence_parallel_hidden(rank._decoder_hidden(prepared))
    gradients: list[torch.Tensor] = []
    for chunk_tokens in (chunks[0], chunks[-1]):
        rank.head_chunk_tokens = chunk_tokens
        candidate = hidden.detach().requires_grad_(True)
        outputs = rank._project_head(items, prepared, candidate)
        _output_loss(outputs).backward()
        assert candidate.grad is not None
        gradients.append(candidate.grad)
    diff = _diff(gradients[0], gradients[1])
    if diff.mean_abs_pct > 2e-3:
        raise AssertionError(f"head gradient mean_abs_pct={diff.mean_abs_pct}")
    return diff


def _performance(
    runtime: Any,
    *,
    depth: int,
    workload: str,
    request: str,
    families: int,
    prefix_tokens: int,
    branches: int,
    completion_tokens: int,
    slots: int,
    adaptive: bool,
    optimizer_step: bool,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    if workload == "austin":
        families, prefix_tokens, branches, completion_tokens = 30, 5000, 16, 100
    rank = TrainerRank(runtime, shared_prefix_max_depth=depth, head_chunk_tokens=8192)
    slot_names = load_random_checkpoint_slots(runtime, rank, slots)
    requests = _performance_requests(
        request=request,
        families=families,
        prefix_tokens=prefix_tokens,
        branches=branches,
        completion_tokens=completion_tokens,
        varied=workload == "varied",
        slots=slot_names,
    )
    dp_rank, dp_size = rank._dp_rank_and_size()
    plan = rank._plan_flat_forward(requests)
    assert workload != "austin" or plan.packed_tokens == 198_000

    def step() -> list[MicroBatchStats]:
        rank.zero_grad()
        stats: list[MicroBatchStats] = []
        if adaptive:
            for micro in rank.forward_micro_batches(requests):
                _output_loss(cast(Sequence[ForwardOutput], micro.outputs)).backward()
                stats.append(micro.stats)
        else:
            outputs = rank.dp_rank_forward(requests[dp_rank::dp_size])
            _output_loss(outputs).backward()
        if optimizer_step:
            if not slot_names:
                raise ValueError("--optimizer-step requires --slots >= 1")
            rank.optim_step(params=AdamParams(learning_rate=1e-5))
        return stats

    for _ in range(warmup):
        step()
    times: list[float] = []
    all_stats: list[MicroBatchStats] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeat):
        torch.cuda.synchronize()
        started = time.perf_counter()
        all_stats.extend(step())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    median = statistics.median(times)
    free, total = torch.cuda.mem_get_info()
    return {
        "depth": depth,
        "workload": workload,
        "request": request,
        "adaptive": adaptive,
        "optimizer_step": optimizer_step,
        "slots": slots,
        "warmup": warmup,
        "repeat": repeat,
        "packed_tokens": plan.packed_tokens,
        "logical_tokens": plan.logical_tokens,
        "median_s": median,
        "packed_tok_s": plan.packed_tokens / median,
        "logical_tok_s": plan.logical_tokens / median,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "device_used_gb": (total - free) / 1024**3,
        "windows": [stat.global_count for stat in all_stats],
        "rejected_candidates": sum(stat.rejected_candidates for stat in all_stats),
    }


def _output_loss(outputs: Iterable[ForwardOutput]) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for output in outputs:
        if output.target_logprobs is not None:
            terms.append(-output.target_logprobs.float().sum())
        if output.top_k is not None:
            terms.append(-output.top_k.logprobs.float().sum())
        if output.logits is not None:
            terms.append(output.logits.float().square().mean())
        if output.hidden_states is not None:
            terms.append(output.hidden_states.float().square().mean())
    if not terms:
        raise RuntimeError("request produced no differentiable outputs")
    return torch.stack(terms).sum()


def _correctness_requests(slots: Sequence[str] = ()) -> list[ForwardInput]:
    requests: list[ForwardInput] = []
    for mask in range(16):
        tokens = torch.tensor(
            [11, 12, 20 + mask // 8, 30 + mask // 4 % 2, 40 + mask // 2 % 2, 50 + mask]
        )
        tokens = tokens.reshape(2, 3) if mask == 15 else tokens
        labels: torch.Tensor | None = None
        if mask & 1:
            labels = (tokens * 7 + mask) % 1000
            if mask == 1:
                labels = torch.stack((labels, (labels + 17) % 1000), dim=1)
                labels[2, 1] = -100
        requests.append(
            ForwardInput(
                input_tokens=tokens,
                target_tokens=labels,
                top_k=3 if mask & 2 else None,
                logits=bool(mask & 4),
                hidden_states=bool(mask & 8),
                checkpoint=slots[mask % len(slots)] if slots else Unset,
            )
        )
    return requests


def _performance_requests(
    *,
    request: str,
    families: int,
    prefix_tokens: int,
    branches: int,
    completion_tokens: int,
    varied: bool,
    slots: Sequence[str],
) -> list[ForwardInput]:
    requests: list[ForwardInput] = []
    for family in range(families):
        family_base = family * 10_000_019
        prefix_len = prefix_tokens + ((family * 97) % 257 - 128 if varied else 0)
        prefix = _tokens(family_base, max(1, prefix_len))
        family_branches = max(1, branches + ((family % 5) - 2 if varied else 0))
        for branch in range(family_branches):
            completion_len = completion_tokens + (
                (branch * 17) % 33 - 16 if varied else 0
            )
            tokens = torch.cat(
                (
                    prefix,
                    _tokens(family_base + branch * 1009 + 17, max(1, completion_len)),
                )
            )
            labels = (tokens * 7 + 3) % 32_000
            labels[: int(prefix.numel())] = -100
            if request == "multi":
                labels = torch.stack(
                    tuple((labels + offset) % 32_000 for offset in range(4)), dim=1
                )
                labels[: int(prefix.numel())] = -100
            requests.append(
                ForwardInput(
                    input_tokens=tokens,
                    target_tokens=labels
                    if request in {"target", "multi", "mixed"}
                    else None,
                    top_k=10 if request in {"topk", "mixed"} else None,
                    logits=request == "logits"
                    or request == "mixed"
                    and branch % 16 == 0,
                    hidden_states=request == "hidden"
                    or request == "mixed"
                    and branch % 8 == 0,
                    checkpoint=slots[family % len(slots)] if slots else Unset,
                )
            )
    return requests


def _tokens(offset: int, length: int) -> torch.Tensor:
    return (torch.arange(length, dtype=torch.long) + offset) % 32_000 + 100


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> Diff:
    assert actual.shape == expected.shape, (
        f"shape mismatch: {actual.shape} != {expected.shape}"
    )
    if not actual.numel():
        return Diff()
    delta = (actual.float() - expected.float()).abs()
    return Diff(
        float(delta.mean() / expected.float().abs().mean().clamp_min(1e-18)),
        float(delta.max()),
    )


def _cpu(tensor: object) -> torch.Tensor | None:
    return tensor.detach().cpu() if isinstance(tensor, torch.Tensor) else None


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _topology() -> dict[str, int]:
    from megatron.core import parallel_state as ps

    return {
        "world": dist.get_world_size(),
        "dp": int(ps.get_data_parallel_world_size()),
        "tp": int(ps.get_tensor_model_parallel_world_size()),
        "cp": int(ps.get_context_parallel_world_size()),
        "ep": int(ps.get_expert_model_parallel_world_size()),
    }


if __name__ == "__main__":
    typer.run(main)
