from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Literal

import pytest
import torch
import torch.multiprocessing as mp

# This CPU-only probe does not install Megatron Bridge; lora.py imports this
# provider solely for runtime annotations used outside the publication path.
_bridge = ModuleType("megatron.bridge")
_bridge_models = ModuleType("megatron.bridge.models")
_bridge_gpt = ModuleType("megatron.bridge.models.gpt_provider")
_bridge_conversion = ModuleType("megatron.bridge.models.conversion")
_bridge_mapping = ModuleType("megatron.bridge.models.conversion.param_mapping")
_bridge_gpt.GPTModelProvider = type("GPTModelProvider", (), {})
_bridge_mapping.AutoMapping = type(
    "AutoMapping",
    (),
    {"register_module_type": staticmethod(lambda *_args, **_kwargs: None)},
)
sys.modules.setdefault("megatron.bridge", _bridge)
sys.modules.setdefault("megatron.bridge.models", _bridge_models)
sys.modules.setdefault("megatron.bridge.models.gpt_provider", _bridge_gpt)
sys.modules.setdefault("megatron.bridge.models.conversion", _bridge_conversion)
sys.modules.setdefault(
    "megatron.bridge.models.conversion.param_mapping", _bridge_mapping
)

from art.distributed.object_store import OrderedBinaryObjectTarget, S3ObjectStoreConfig
from art.megatron.lora import LoraShardMeta, _block_for_key
from art.megatron.model_support.handlers.qwen3_5 import QWEN3_5_MOE_HANDLER
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_VLLM,
    encode_adapter_config,
)
from art.megatron.tensor_snapshot import PinnedCpuSnapshotStager
from art.megatron.weights.lora_publish import (
    PackedExpertShardMeta,
    _canonical_global_metadata,
    _exchange_batched_tensors,
    _rank0_merged_lora_tensors,
)
from art.megatron.weights.rank_distributed_lora_publish import (
    prepare_rank_distributed_vllm_lora,
)
from art.utils.safetensors import prepare_safetensors

Case = Literal["regular", "packed", "shared_outer"]


def _tensor(shape: tuple[int, ...], seed: int, dtype: torch.dtype) -> torch.Tensor:
    return (torch.arange(torch.tensor(shape).prod().item()).reshape(shape) + seed).to(
        dtype
    )


def _regular_meta(
    key: str,
    tensor: torch.Tensor,
    rank: int,
    manifest: dict[str, Any],
) -> LoraShardMeta:
    return LoraShardMeta(
        key=key,
        owner_rank=rank,
        shape=tuple(tensor.shape),
        dtype_name=str(tensor.dtype).removeprefix("torch."),
        manifest=manifest,
        block=_block_for_key(key),
    )


def _packed_meta(
    key: str,
    tensor: torch.Tensor,
    rank: int,
    *,
    expert_start: int,
    layout: str,
) -> PackedExpertShardMeta:
    return PackedExpertShardMeta(
        key=key,
        owner_rank=rank,
        shape=tuple(tensor.shape),
        dtype_name=str(tensor.dtype).removeprefix("torch."),
        manifest={"sharded": False, "shard_world_size": 1, "shard_rank": 0},
        expert_start=expert_start,
        expert_count=tensor.shape[0],
        pack_layout=layout,
    )


def _fixture(
    case: Case,
    rank: int,
    world_size: int,
) -> tuple[
    dict[str, torch.Tensor],
    list[LoraShardMeta],
    dict[str, torch.Tensor],
    list[PackedExpertShardMeta],
    dict[str, Any],
]:
    regular: dict[str, torch.Tensor] = {}
    regular_meta: list[LoraShardMeta] = []
    packed: dict[str, torch.Tensor] = {}
    packed_meta: list[PackedExpertShardMeta] = []
    layers = max(4, world_size * 2)
    if case == "regular":
        for layer in range(layers):
            prefix = f"base_model.model.model.layers.{layer}.self_attn"
            replicated_key = f"{prefix}.q_proj.lora_A.weight"
            replicated = _tensor((2, 3), layer * 100, torch.bfloat16)
            regular[replicated_key] = replicated
            regular_meta.append(
                _regular_meta(
                    replicated_key,
                    replicated,
                    rank,
                    {"sharded": False, "shard_world_size": 1, "shard_rank": 0},
                )
            )
            uniform_key = f"{prefix}.o_proj.lora_B.weight"
            uniform = _tensor((2, 3), layer * 1000 + rank * 10, torch.float32)
            regular[uniform_key] = uniform
            regular_meta.append(
                _regular_meta(
                    uniform_key,
                    uniform,
                    rank,
                    {
                        "sharded": True,
                        "shard_world_size": world_size,
                        "shard_rank": rank,
                        "export_shard_dim": 0,
                        "export_shard_strategy": "uniform",
                    },
                )
            )
            component_key = f"{prefix}.k_proj.lora_B.weight"
            component = _tensor((4, 3), layer * 10000 + rank * 100, torch.float16)
            regular[component_key] = component
            regular_meta.append(
                _regular_meta(
                    component_key,
                    component,
                    rank,
                    {
                        "sharded": True,
                        "shard_world_size": world_size,
                        "shard_rank": rank,
                        "export_shard_dim": 0,
                        "export_shard_strategy": "componentwise",
                        "component_sizes": [2 * world_size, 2 * world_size],
                    },
                )
            )
        config = {
            "r": 2,
            "lora_alpha": 4,
            "target_modules": ["q_proj", "o_proj", "k_proj"],
        }
        return regular, regular_meta, packed, packed_meta, config

    for layer in range(layers):
        prefix = f"base_model.model.model.layers.{layer}.mlp.experts"
        slots = (
            ("base_layer.lora_B.weight", (2, 6, 2), "rank_major_expert_cols"),
            ("lora_A.weight", (2, 2, 5), "expert_rows"),
        )
        if case == "packed":
            slots += (
                ("base_layer.lora_A.weight", (2, 2, 5), "expert_rows"),
                ("lora_B.weight", (2, 5, 2), "rank_major_expert_cols"),
            )
        for slot, shape, layout in slots:
            key = f"{prefix}.{slot}"
            value = _tensor(shape, layer * 10000 + rank * 1000, torch.bfloat16)
            packed[key] = value
            packed_meta.append(
                _packed_meta(
                    key,
                    value,
                    rank,
                    expert_start=rank * 2,
                    layout=layout,
                )
            )
        if case == "shared_outer":
            for module, lora, shape in (
                ("gate_up_proj", "lora_A", (2, 5)),
                ("down_proj", "lora_B", (5, 2)),
            ):
                key = f"{prefix}.shared.{module}.{lora}.weight"
                value = _tensor(shape, layer * 100, torch.bfloat16)
                regular[key] = value
                regular_meta.append(
                    _regular_meta(
                        key,
                        value,
                        rank,
                        {
                            "sharded": False,
                            "shard_world_size": 1,
                            "shard_rank": 0,
                        },
                    )
                )
    config = {
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["gate_up_proj", "down_proj"],
        **({"moe_parameterization": "shared_outer"} if case == "shared_outer" else {}),
    }
    return regular, regular_meta, packed, packed_meta, config


def _target(case: Case, world_size: int) -> OrderedBinaryObjectTarget:
    store = S3ObjectStoreConfig(
        endpoint_url="https://objects.invalid",
        region="test",
        bucket="bucket",
        prefix="rank-distributed-test",
        multipart_concurrency=16,
    )
    return OrderedBinaryObjectTarget(
        store=store,
        object_id=f"{world_size}{'123'[('regular', 'packed', 'shared_outer').index(case)]}".ljust(
            64, "0"
        ),
        format="vllm_lora",
        shard_bytes=37,
        max_concurrent_shards=16,
        max_shards=10_000,
    )


def _bytes(tensor_chunks: tuple[torch.Tensor, ...]) -> bytes:
    return b"".join(
        memoryview(chunk.reshape(-1).view(torch.uint8).numpy()).cast("B")
        for chunk in tensor_chunks
    )


def _old_rank0_bytes(
    regular: dict[str, torch.Tensor],
    regular_meta: list[LoraShardMeta],
    packed: dict[str, torch.Tensor],
    packed_meta: list[PackedExpertShardMeta],
    config: dict[str, Any],
    rank: int,
) -> tuple[dict[str, bytes] | None, int]:
    canonical_regular = _canonical_global_metadata(regular_meta)
    canonical_packed = _canonical_global_metadata(packed_meta)
    old_regular = _exchange_batched_tensors(
        canonical_regular,
        local_tensors=regular,
        rank=rank,
        device=torch.device("cpu"),
    )
    old_packed = _exchange_batched_tensors(
        canonical_packed,
        local_tensors=packed,
        rank=rank,
        device=torch.device("cpu"),
    )
    source_bytes = sum(
        meta.numel
        * torch.empty((), dtype=getattr(torch, meta.dtype_name)).element_size()
        for meta in (*canonical_regular, *canonical_packed)
    )
    if rank != 0:
        return None, source_bytes
    merged = _rank0_merged_lora_tensors(
        metadata=canonical_regular,
        tensors_by_owner_key=old_regular,
        packed_expert_metadata=canonical_packed,
        packed_expert_tensors_by_owner_key=old_packed,
    )
    converted, published_config = QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        merged,
        adapter_config=dict(config),
    )
    return {
        "adapter_config.json": encode_adapter_config(
            {
                **published_config,
                ART_LORA_FORMAT_CONFIG_KEY: ART_LORA_FORMAT_VLLM,
            }
        ),
        "adapter_model.safetensors": _bytes(prepare_safetensors(converted).chunks),
    }, source_bytes


def _assert_new_bytes(
    expected: dict[str, bytes],
    prepared: Any,
    gathered_payloads: list[dict[int, bytes]],
) -> None:
    plan = prepared.layout.plan
    payloads = {
        index: payload
        for rank_payloads in gathered_payloads
        for index, payload in rank_payloads.items()
    }
    assert len(payloads) == sum(len(value) for value in gathered_payloads)
    assert set(payloads) == set(range(len(plan.shards)))
    actual = {file.relative_path: bytearray(file.byte_count) for file in plan.files}
    for shard in plan.shards:
        payload = payloads[shard.index]
        assert len(payload) == shard.byte_count
        target = actual[shard.relative_path]
        target[shard.file_offset : shard.file_offset + shard.byte_count] = payload
    assert {key: bytes(value) for key, value in actual.items()} == expected


def _run_case(case: Case, rank: int, world_size: int) -> dict[str, Any] | None:
    regular, regular_meta, packed, packed_meta, config = _fixture(
        case, rank, world_size
    )
    torch.distributed.barrier()
    old_started = time.perf_counter()
    expected, old_source_bytes = _old_rank0_bytes(
        regular,
        regular_meta,
        packed,
        packed_meta,
        config,
        rank,
    )
    torch.distributed.barrier()
    old_elapsed_s = time.perf_counter() - old_started
    new_started = time.perf_counter()
    prepared = prepare_rank_distributed_vllm_lora(
        target=_target(case, world_size),
        local_tensors=regular,
        local_metadata=regular_meta,
        local_packed_tensors=packed,
        local_packed_metadata=packed_meta,
        handler=QWEN3_5_MOE_HANDLER,
        adapter_config=config,
        exchange_device=torch.device("cpu"),
        stager=PinnedCpuSnapshotStager(),
    ).resolve()
    torch.distributed.barrier()
    new_elapsed_s = time.perf_counter() - new_started
    layouts: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(layouts, prepared.layout)
    assert all(layout == layouts[0] for layout in layouts)
    local_payloads = {
        index: b"".join(bytes(chunk) for chunk in chunks)
        for index, chunks in prepared.shard_payloads().items()
    }
    gathered_payloads: list[dict[int, bytes] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered_payloads, local_payloads)
    stats: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(stats, prepared.stats)
    timings: list[tuple[float, float] | None] = [None] * world_size
    torch.distributed.all_gather_object(timings, (old_elapsed_s, new_elapsed_s))
    if rank != 0:
        return None
    assert expected is not None
    assert all(value is not None for value in gathered_payloads)
    assert all(value is not None for value in stats)
    assert all(value is not None for value in timings)
    _assert_new_bytes(expected, prepared, [value or {} for value in gathered_payloads])
    typed_stats = [value for value in stats if value is not None]
    typed_timings = [value for value in timings if value is not None]
    global_output_bytes = sum(tensor.byte_count for tensor in prepared.layout.tensors)
    assert sum(value.owned_tensor_bytes for value in typed_stats) == global_output_bytes
    block_owners: dict[str, set[int]] = defaultdict(set)
    block_bytes: dict[str, int] = defaultdict(int)
    for tensor in prepared.layout.tensors:
        block = _block_for_key(tensor.name)
        block_owners[block].add(tensor.owner_rank)
        block_bytes[block] += tensor.byte_count
    assert all(len(owners) == 1 for owners in block_owners.values())
    if world_size > 1:
        rank0 = typed_stats[0]
        assert rank0.owned_tensor_bytes < global_output_bytes
        assert rank0.peak_accounted_owner_bytes < old_source_bytes + global_output_bytes
        assert max(value.owned_tensor_bytes for value in typed_stats) <= (
            global_output_bytes // world_size + max(block_bytes.values())
        )
    return {
        "case": case,
        "world_size": world_size,
        "global_output_bytes": global_output_bytes,
        "old_rank0_source_bytes": old_source_bytes,
        "rank0_owned_bytes": typed_stats[0].owned_tensor_bytes,
        "rank0_peak_accounted_bytes": typed_stats[0].peak_accounted_owner_bytes,
        "sent_bytes": sum(value.sent_bytes for value in typed_stats),
        "received_bytes": sum(value.received_bytes for value in typed_stats),
        "old_rank0_wall_s": max(value[0] for value in typed_timings),
        "distributed_wall_s": max(value[1] for value in typed_timings),
        "shards": len(prepared.layout.plan.shards),
    }


def _worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    report_path: str,
) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        reports = [
            report
            for case in ("regular", "packed", "shared_outer")
            if (report := _run_case(case, rank, world_size)) is not None
        ]
        if rank == 0:
            Path(report_path).write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_rank_distributed_lora_matches_current_rank0_bytes(
    tmp_path: Path,
    world_size: int,
) -> None:
    mp.spawn(
        _worker,
        args=(
            world_size,
            str(tmp_path / f"gloo-{world_size}"),
            str(tmp_path / f"report-{world_size}.json"),
        ),
        nprocs=world_size,
        join=True,
    )
    reports = json.loads((tmp_path / f"report-{world_size}.json").read_text())
    assert {report["case"] for report in reports} == {
        "regular",
        "packed",
        "shared_outer",
    }
