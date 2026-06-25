from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
import re
from typing import Any, cast

import torch
import torch.distributed as dist
import typer

from art.megatron.shared_prefix_packing import SharedPrefixPack, pack_shared_prefixes
from art.trainer_rank import (
    AnyForwardInput,
    TrainerRank,
    _batch_seq_logits,
    _language_model,
)


@dataclass(frozen=True)
class _Capture:
    values: dict[str, torch.Tensor]
    positions_by_item: tuple[torch.Tensor, ...]
    source_positions_by_item: tuple[torch.Tensor, ...]


def main(
    model: str = "Qwen/Qwen3-0.6B",
    layers: int = 1,
    sequences: int = 6,
    sequence_length: int = 7,
    compare_requests: int = 6,
    request_shape: str = "varied",
    oracle: str = "independent",
    max_depth: int = 1,
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
        runtime = megatron_train.build_training_runtime(
            model_identifier=model,
            provider_configure=lambda provider: setattr(
                provider,
                "num_layers",
                layers,
            ),
            print_env=dist.get_rank() == 0,
        )
        if int(ps.get_tensor_model_parallel_world_size()) != 1:
            raise RuntimeError("trainer_rank_parity_probe currently expects TP=1")
        for chunk in runtime.model:
            chunk.eval()

        rank = TrainerRank(runtime, shared_prefix_max_depth=max_depth)
        requests = _unique_requests(
            sequences=sequences,
            sequence_length=sequence_length,
            request_shape=request_shape,
        )
        request_count = min(compare_requests, len(requests))

        with torch.no_grad():
            packed = _run_capture(rank, requests)
            records = _records_from_capture(
                kind="packed",
                capture=packed,
                request_indices=range(len(requests)),
                cp_rank=int(ps.get_context_parallel_rank()),
                dp_rank=int(ps.get_data_parallel_rank()),
            )
            for request_index, request in enumerate(requests):
                if oracle == "independent":
                    oracle_capture = _run_capture(rank, [request])
                    oracle_request_indices = (request_index,)
                    oracle_local_indices = None
                elif oracle == "same-layout":
                    oracle_capture = _run_capture(
                        rank,
                        requests,
                        mutate_except=request_index,
                    )
                    oracle_request_indices = range(len(requests))
                    oracle_local_indices = (request_index,)
                else:
                    raise ValueError("oracle must be 'independent' or 'same-layout'")
                records.extend(
                    _records_from_capture(
                        kind="independent",
                        capture=oracle_capture,
                        request_indices=oracle_request_indices,
                        cp_rank=int(ps.get_context_parallel_rank()),
                        dp_rank=int(ps.get_data_parallel_rank()),
                        local_indices=oracle_local_indices,
                    )
                )

        gathered: list[list[dict[str, object]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, records)
        if dist.get_rank() == 0:
            flat_records = [
                record for rank_records in gathered for record in rank_records or []
            ]
            report = _build_report(
                records=flat_records,
                requests=requests[:request_count],
                topology={
                    "world": dist.get_world_size(),
                    "dp": int(ps.get_data_parallel_world_size()),
                    "tp": int(ps.get_tensor_model_parallel_world_size()),
                    "cp": int(ps.get_context_parallel_world_size()),
                },
                oracle=oracle,
            )
            print(json.dumps(report, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _unique_requests(
    *,
    sequences: int,
    sequence_length: int,
    request_shape: str,
) -> list[AnyForwardInput]:
    from art.trainer_rank import ForwardInput

    if sequences < 1 or sequence_length < 2:
        raise ValueError("sequences must be >= 1 and sequence_length must be >= 2")
    if request_shape == "varied":
        base_rows = (
            (11, 12, 13, 14, 15, 16, 17),
            (11, 12, 13, 14, 24, 25),
            (11, 12, 13, 14, 24, 26),
            (11, 12, 13, 27),
            (31, 32, 33, 34),
            (31, 32, 33, 35),
            (11, 12, 13, 14, 15, 16, 17),
            (41, 42, 43),
            (41, 42, 44, 45),
            (51, 52, 53, 54, 55),
            (61, 62, 63),
            (61, 62, 64, 65),
            (71, 72),
            (81, 82, 83, 84),
            (91, 92, 93),
            (101, 102, 103, 104, 105),
        )
        return [
            ForwardInput(
                input_tokens=torch.tensor(row, dtype=torch.long) + 1000 * index
            )
            for index, row in enumerate(base_rows[:sequences])
        ]
    if request_shape == "deep":
        base_rows = (
            (11, 12, 13, 14, 15, 16, 17),
            (11, 12, 13, 14, 15, 16, 18),
            (11, 12, 13, 14, 15, 19),
            (11, 12, 13, 14, 20),
            (11, 12, 21),
            (31, 32, 33, 34, 35),
            (31, 32, 33, 34, 36),
            (31, 32, 33, 37),
            (41, 42, 43),
            (41, 42, 44),
            (51, 52, 53, 54),
            (61, 62),
            (71, 72, 73, 74, 75),
            (71, 72, 73, 76),
            (81,),
            (91, 92, 93),
        )
        return [
            ForwardInput(input_tokens=torch.tensor(row, dtype=torch.long))
            for row in base_rows[:sequences]
        ]
    if request_shape != "equal":
        raise ValueError("request_shape must be 'equal', 'varied', or 'deep'")
    return [
        ForwardInput(
            input_tokens=torch.arange(
                1000 * index + 11,
                1000 * index + 11 + sequence_length,
                dtype=torch.long,
            )
        )
        for index in range(sequences)
    ]


def _run_capture(
    rank: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    mutate_except: int | None = None,
) -> _Capture:
    from art.megatron.train import _placeholder_attention_mask

    model = _language_model(rank.runtime.model[0])
    items = [rank._forward_item(request) for request in requests]
    batch = pack_shared_prefixes(
        (item.input_ids for item in items),
        max_depth=rank.shared_prefix_max_depth,
    )
    if mutate_except is not None:
        batch = _mutated_batch(
            batch, keep_positions=batch.positions_by_sequence[mutate_except]
        )
    prepared = rank._prepare_packed_forward(batch)
    local_seq_len = int(prepared.tokens.shape[1])
    values: dict[str, torch.Tensor] = {}
    handles = _register_hooks(model, values, seq_len=local_seq_len)
    try:
        handler = rank.runtime.model_support_handler
        forward_kwargs = handler.get_forward_kwargs(
            rank.runtime.model[0],
            attention_bias=prepared.attention_state,
        )
        extra_block_kwargs = cast(
            dict[str, object] | None,
            forward_kwargs.pop("extra_block_kwargs", None),
        )
        preprocessed = model._preprocess(
            input_ids=prepared.tokens,
            position_ids=prepared.position_ids,
            packed_seq_params=prepared.packed_seq_params,
        )
        values["00.preprocess.decoder_input"] = _rows(
            cast(torch.Tensor, preprocessed[0]).detach(),
            seq_len=local_seq_len,
        )
        hidden = cast(
            torch.Tensor,
            model.decoder(
                hidden_states=preprocessed[0],
                attention_mask=_placeholder_attention_mask(rank.device),
                rotary_pos_emb=preprocessed[1],
                rotary_pos_cos=preprocessed[2],
                rotary_pos_sin=preprocessed[3],
                rotary_pos_cos_sin=preprocessed[6] if len(preprocessed) == 7 else None,
                packed_seq_params=prepared.packed_seq_params,
                sequence_len_offset=preprocessed[4],
                padding_mask=preprocessed[5],
                **(extra_block_kwargs or {}),
            ),
        )
        gathered_hidden = rank._gather_sequence_parallel_hidden(hidden)
        values["90.decoder.output"] = gathered_hidden.detach()
        values["99.lm_head.logits"] = _logits(rank, gathered_hidden).detach()
        return _Capture(
            values=values,
            positions_by_item=prepared.positions_by_item,
            source_positions_by_item=prepared.source_positions_by_item,
        )
    finally:
        for handle in handles:
            handle.remove()


def _mutated_batch(
    batch: SharedPrefixPack,
    *,
    keep_positions: torch.Tensor,
) -> SharedPrefixPack:
    tokens = batch.tokens.clone()
    mask = torch.ones(int(tokens.shape[1]), dtype=torch.bool, device=tokens.device)
    mask[keep_positions.to(device=tokens.device)] = False
    replacement = (
        torch.arange(int(tokens.shape[1]), dtype=tokens.dtype, device=tokens.device)
        + 50_000
    )
    tokens[0, mask] = replacement[mask] % 100_000
    return SharedPrefixPack(
        tokens=tokens,
        group_ids=batch.group_ids,
        parent_ids=batch.parent_ids,
        position_ids=batch.position_ids,
        positions_by_sequence=batch.positions_by_sequence,
    )


def _register_hooks(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    *,
    seq_len: int,
) -> list[Any]:
    handles: list[Any] = []
    for module_name, module in model.named_modules():
        label = _capture_label(module_name)
        if label is None:
            continue

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            label: str = label,
        ) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                try:
                    values[label] = _rows(tensor.detach(), seq_len=seq_len)
                except RuntimeError:
                    pass

        handles.append(module.register_forward_hook(hook))
    return handles


def _capture_label(module_name: str) -> str | None:
    layer_prefix = r"decoder\.layers\.(\d+)(?:\._orig_mod)?"
    if re.fullmatch(r"decoder\.layers\.(\d+)\._orig_mod", module_name):
        return None
    layer_match = re.fullmatch(r"decoder\.layers\.(\d+)", module_name)
    if layer_match:
        return f"30.layer.{int(layer_match.group(1)):03d}.output"
    input_norm_match = re.fullmatch(rf"{layer_prefix}\.input_layernorm", module_name)
    if input_norm_match:
        return f"05.layer.{int(input_norm_match.group(1)):03d}.input_layernorm"
    qkv_match = re.fullmatch(
        rf"{layer_prefix}\.self_attention\.linear_qkv", module_name
    )
    if qkv_match:
        return f"08.layer.{int(qkv_match.group(1)):03d}.self_attention.linear_qkv"
    core_attention_match = re.fullmatch(
        rf"{layer_prefix}\.self_attention\.core_attention",
        module_name,
    )
    if core_attention_match:
        return f"10.layer.{int(core_attention_match.group(1)):03d}.self_attention.core_attention"
    attention_proj_match = re.fullmatch(
        rf"{layer_prefix}\.self_attention\.linear_proj",
        module_name,
    )
    if attention_proj_match:
        return f"12.layer.{int(attention_proj_match.group(1)):03d}.self_attention.linear_proj"
    attention_match = re.fullmatch(
        rf"{layer_prefix}\.self_attention",
        module_name,
    )
    if attention_match:
        return f"15.layer.{int(attention_match.group(1)):03d}.self_attention"
    pre_mlp_norm_match = re.fullmatch(
        rf"{layer_prefix}\.pre_mlp_layernorm",
        module_name,
    )
    if pre_mlp_norm_match:
        return f"18.layer.{int(pre_mlp_norm_match.group(1)):03d}.pre_mlp_layernorm"
    fc1_match = re.fullmatch(rf"{layer_prefix}\.mlp\.linear_fc1", module_name)
    if fc1_match:
        return f"20.layer.{int(fc1_match.group(1)):03d}.mlp.linear_fc1"
    fc2_match = re.fullmatch(rf"{layer_prefix}\.mlp\.linear_fc2", module_name)
    if fc2_match:
        return f"22.layer.{int(fc2_match.group(1)):03d}.mlp.linear_fc2"
    mlp_match = re.fullmatch(rf"{layer_prefix}\.mlp", module_name)
    if mlp_match:
        return f"25.layer.{int(mlp_match.group(1)):03d}.mlp"
    if module_name == "decoder.final_layernorm":
        return "80.decoder.final_layernorm"
    return None


def _first_tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _rows(tensor: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    if tensor.ndim >= 2 and int(tensor.shape[0]) == seq_len:
        rows = tensor
        if rows.ndim >= 3 and int(rows.shape[1]) == 1:
            return rows[:, 0].contiguous()
        return rows.contiguous()
    if tensor.ndim >= 2 and int(tensor.shape[1]) == seq_len:
        rows = (
            tensor[:, :, 0]
            if tensor.ndim == 4 and int(tensor.shape[2]) == 1
            else tensor
        )
        if int(rows.shape[0]) == 1:
            return rows[0].contiguous()
    raise RuntimeError(
        f"Cannot identify sequence axis for tensor shape={tuple(tensor.shape)} "
        f"seq_len={seq_len}"
    )


def _logits(rank: TrainerRank, hidden_rows: torch.Tensor) -> torch.Tensor:
    model = _language_model(rank.runtime.model[0])
    output_weight = (
        model.shared_embedding_or_output_weight()
        if bool(model.share_embeddings_and_output_weights)
        else None
    )
    if int(hidden_rows.shape[0]) == 0:
        return hidden_rows.new_empty((0, int(model.vocab_size)))
    local_logits = rank._local_logits_from_hidden_rows(
        model,
        hidden_rows,
        output_weight=output_weight,
    )
    return _batch_seq_logits(
        rank._gather_tensor_parallel_logits(local_logits.unsqueeze(1)),
        seq_len=int(hidden_rows.shape[0]),
    ).squeeze(0)


def _records_from_capture(
    *,
    kind: str,
    capture: _Capture,
    request_indices: Sequence[int],
    cp_rank: int,
    dp_rank: int,
    local_indices: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    local_index_set = None if local_indices is None else frozenset(local_indices)
    for local_index, request_index in enumerate(request_indices):
        if local_index_set is not None and local_index not in local_index_set:
            continue
        positions = capture.positions_by_item[local_index]
        source_positions = capture.source_positions_by_item[local_index]
        if int(positions.numel()) == 0:
            continue
        for name, rows in capture.values.items():
            records.append(
                {
                    "kind": kind,
                    "name": name,
                    "request_index": int(request_index),
                    "source_positions": source_positions.cpu(),
                    "value": rows.index_select(0, positions.to(rows.device)).cpu(),
                    "cp": int(cp_rank),
                    "dp": int(dp_rank),
                }
            )
    return records


def _build_report(
    *,
    records: list[dict[str, object]],
    requests: Sequence[AnyForwardInput],
    topology: dict[str, int],
    oracle: str,
) -> dict[str, object]:
    results = []
    names = sorted(
        {
            cast(str, record["name"])
            for record in records
            if record.get("kind") == "packed"
        }
    )
    for request_index, request in enumerate(requests):
        length = int(request.input_tokens.numel())
        for name in names:
            packed = _assemble(records, "packed", name, request_index, length)
            independent = _assemble(records, "independent", name, request_index, length)
            if packed is None or independent is None:
                continue
            diff = (packed.float() - independent.float()).abs()
            denom = independent.float().abs().max().clamp_min(1e-12)
            results.append(
                {
                    "request": request_index,
                    "site": name,
                    "shape": list(packed.shape),
                    "max_abs": float(diff.max().item()) if int(diff.numel()) else 0.0,
                    "mean_abs": float(diff.mean().item()) if int(diff.numel()) else 0.0,
                    "rel_max": float((diff.max() / denom).item())
                    if int(diff.numel())
                    else 0.0,
                }
            )
    return {
        "topology": topology,
        "oracle": oracle,
        "requests": len(requests),
        "results": results,
    }


def _assemble(
    records: list[dict[str, object]],
    kind: str,
    name: str,
    request_index: int,
    length: int,
) -> torch.Tensor | None:
    matching = [
        record
        for record in records
        if record["kind"] == kind
        and record["name"] == name
        and record["request_index"] == request_index
    ]
    if not matching:
        return None
    first = cast(torch.Tensor, matching[0]["value"])
    output = torch.empty((length, *first.shape[1:]), dtype=first.dtype)
    filled = torch.zeros(length, dtype=torch.bool)
    for record in matching:
        positions = cast(torch.Tensor, record["source_positions"])
        value = cast(torch.Tensor, record["value"])
        output[positions] = value
        filled[positions] = True
    if not bool(filled.all().item()):
        raise RuntimeError(
            f"Missing positions for {kind} {name} request={request_index}"
        )
    return output


if __name__ == "__main__":
    typer.run(main)
