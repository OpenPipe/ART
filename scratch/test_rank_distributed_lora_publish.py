from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
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
from art.megatron.model_support.handlers.dsv4 import DSV4_HANDLER
from art.megatron.model_support.handlers.gemma4 import GEMMA4_MOE_HANDLER
from art.megatron.model_support.handlers.gpt_oss import GPT_OSS_MOE_HANDLER
from art.megatron.model_support.handlers.qwen3_5 import QWEN3_5_MOE_HANDLER
from art.megatron.model_support.lora_disk import (
    ART_LORA_FORMAT_CONFIG_KEY,
    ART_LORA_FORMAT_VLLM,
    encode_adapter_config,
)
from art.megatron.runtime.executor import _GenerationPublisher
from art.megatron.runtime.specs import TrainerGeneration
from art.megatron.tensor_snapshot import (
    PinnedCpuSnapshotBuilder,
    PinnedCpuSnapshotStager,
)
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

Case = Literal[
    "qwen36_per_expert",
    "qwen36_packed",
    "qwen36_shared_outer",
    "gpt_oss_per_expert",
    "gpt_oss_packed",
    "gemma4_per_expert",
    "gemma4_packed",
    "dsv4_per_expert",
    "dsv4_packed",
]
CASES: tuple[Case, ...] = (
    "qwen36_per_expert",
    "qwen36_packed",
    "qwen36_shared_outer",
    "gpt_oss_per_expert",
    "gpt_oss_packed",
    "gemma4_per_expert",
    "gemma4_packed",
    "dsv4_per_expert",
    "dsv4_packed",
)


class Fixture(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    handler: Any
    regular: dict[str, torch.Tensor]
    regular_metadata: list[LoraShardMeta]
    packed: dict[str, torch.Tensor]
    packed_metadata: list[PackedExpertShardMeta]
    adapter_config: dict[str, Any]


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
    fixture_root: Path,
) -> Fixture:
    regular: dict[str, torch.Tensor] = {}
    regular_meta: list[LoraShardMeta] = []
    packed: dict[str, torch.Tensor] = {}
    packed_meta: list[PackedExpertShardMeta] = []
    layers = max(4, world_size * 2)
    if case.endswith("_per_expert"):
        family, mode = case.removesuffix("_per_expert"), "per_expert"
    elif case.endswith("_shared_outer"):
        family, mode = case.removesuffix("_shared_outer"), "shared_outer"
    else:
        family, mode = case.removesuffix("_packed"), "packed"
    if family == "qwen36":
        handler = QWEN3_5_MOE_HANDLER
        hidden, intermediate = 3, 4
        attention = ("q_proj", "o_proj", "k_proj")
        base_model = "Qwen/Qwen3.6-35B-A3B"
        extra_config = {
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 3,
        }
        gate_layout = "rank_major_expert_cols"
    elif family == "gpt_oss":
        handler = GPT_OSS_MOE_HANDLER
        hidden, intermediate = 128, 1024
        attention = ("q_proj", "o_proj", "k_proj")
        base_model = str(fixture_root / "gpt_oss")
        extra_config = {}
        gate_layout = "interleaved_gate_up_rank_major_expert_cols"
    elif family == "gemma4":
        handler = GEMMA4_MOE_HANDLER
        hidden, intermediate = 6, 128
        attention = ("k_proj", "o_proj", "q_proj")
        base_model = str(fixture_root / "gemma4")
        extra_config = {}
        gate_layout = "rank_major_expert_cols"
    else:
        handler = DSV4_HANDLER
        hidden, intermediate = 3, 4
        attention = ("q_a_proj", "o_a_proj", "kv_proj")
        base_model = "deepseek-ai/DeepSeek-V4-Flash"
        extra_config = {}
        gate_layout = "rank_major_expert_cols"

    replicated_manifest = {
        "sharded": False,
        "shard_world_size": 1,
        "shard_rank": 0,
    }
    for layer in range(layers):
        layer_prefix = f"base_model.model.model.layers.{layer}"
        attention_prefix = f"{layer_prefix}.self_attn"
        q_key = f"{attention_prefix}.{attention[0]}.lora_A.weight"
        q = _tensor((2, hidden), layer * 100, torch.bfloat16)
        regular[q_key] = q
        regular_meta.append(_regular_meta(q_key, q, rank, replicated_manifest))
        q_b_rows = 12 if family == "qwen36" else hidden
        q_b_key = f"{attention_prefix}.{attention[0]}.lora_B.weight"
        q_b = _tensor((q_b_rows, 2), layer * 100 + 20, torch.float32)
        regular[q_b_key] = q_b
        regular_meta.append(_regular_meta(q_b_key, q_b, rank, replicated_manifest))

        uniform_key = f"{attention_prefix}.{attention[1]}.lora_B.weight"
        uniform = _tensor((2, 2), layer * 1000 + rank * 10, torch.float16)
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
        component_key = f"{attention_prefix}.{attention[2]}.lora_B.weight"
        component = _tensor((4, 2), layer * 10000 + rank * 100, torch.float16)
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

        expert_prefix = f"{layer_prefix}.mlp.experts"
        if mode == "per_expert":
            for expert in range(rank * 2, rank * 2 + 2):
                for module, lora, shape in (
                    ("gate_up_proj", "lora_A", (2, hidden)),
                    ("gate_up_proj", "lora_B", (2 * intermediate, 2)),
                    ("down_proj", "lora_A", (2, intermediate)),
                    ("down_proj", "lora_B", (hidden, 2)),
                ):
                    key = f"{expert_prefix}.{expert}.{module}.{lora}.weight"
                    value = _tensor(
                        shape,
                        layer * 100_000 + expert * 1_000 + len(regular),
                        torch.bfloat16,
                    )
                    regular[key] = value
                    regular_meta.append(
                        _regular_meta(key, value, rank, replicated_manifest)
                    )
        else:
            slots = (
                ("base_layer.lora_B.weight", (2, 2 * intermediate, 2), gate_layout),
                ("lora_A.weight", (2, 2, intermediate), "expert_rows"),
            )
            if mode == "packed":
                slots += (
                    ("base_layer.lora_A.weight", (2, 2, hidden), "expert_rows"),
                    (
                        "lora_B.weight",
                        (2, hidden, 2),
                        "rank_major_expert_cols",
                    ),
                )
            for slot, shape, layout in slots:
                key = f"{expert_prefix}.{slot}"
                value = _tensor(
                    shape,
                    layer * 100_000 + rank * 1_000 + len(packed),
                    torch.bfloat16,
                )
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
            if mode == "shared_outer":
                for module, lora, shape in (
                    ("gate_up_proj", "lora_A", (2, hidden)),
                    ("down_proj", "lora_B", (hidden, 2)),
                ):
                    key = f"{expert_prefix}.shared.{module}.{lora}.weight"
                    value = _tensor(shape, layer * 100 + len(regular), torch.bfloat16)
                    regular[key] = value
                    regular_meta.append(
                        _regular_meta(key, value, rank, replicated_manifest)
                    )

    target_modules = [*attention, "gate_proj", "up_proj", "down_proj"]
    config = {
        "base_model_name_or_path": base_model,
        "r": 2,
        "lora_alpha": 4,
        "target_modules": target_modules,
        "bias": "none",
        "moe_parameterization": (
            "shared_outer" if mode == "shared_outer" else "per_expert"
        ),
        **extra_config,
    }
    return Fixture(
        handler=handler,
        regular=regular,
        regular_metadata=regular_meta,
        packed=packed,
        packed_metadata=packed_meta,
        adapter_config=config,
    )


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
        object_id=f"{world_size:02x}{CASES.index(case):02x}".ljust(64, "0"),
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
    handler: Any,
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
    converted, published_config = handler.to_vllm_lora_tensors(
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


def _run_case(
    case: Case,
    rank: int,
    world_size: int,
    fixture_root: Path,
) -> dict[str, Any] | None:
    fixture = _fixture(case, rank, world_size, fixture_root)
    torch.distributed.barrier()
    old_started = time.perf_counter()
    expected, old_source_bytes = _old_rank0_bytes(
        fixture.regular,
        fixture.regular_metadata,
        fixture.packed,
        fixture.packed_metadata,
        fixture.handler,
        fixture.adapter_config,
        rank,
    )
    torch.distributed.barrier()
    old_elapsed_s = time.perf_counter() - old_started
    new_started = time.perf_counter()
    object_collectives = 0
    original_all_gather_object = torch.distributed.all_gather_object

    def counted_all_gather_object(*args: Any, **kwargs: Any) -> Any:
        nonlocal object_collectives
        object_collectives += 1
        return original_all_gather_object(*args, **kwargs)

    torch.distributed.all_gather_object = counted_all_gather_object
    try:
        prepared = prepare_rank_distributed_vllm_lora(
            target=_target(case, world_size),
            local_tensors=fixture.regular,
            local_metadata=fixture.regular_metadata,
            local_packed_tensors=fixture.packed,
            local_packed_metadata=fixture.packed_metadata,
            handler=fixture.handler,
            adapter_config=fixture.adapter_config,
            conversion_group_for_key=_block_for_key,
            exchange_device=torch.device("cpu"),
            stager=PinnedCpuSnapshotStager(),
        ).resolve()
    finally:
        torch.distributed.all_gather_object = original_all_gather_object
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
    timings: list[tuple[float, float, int] | None] = [None] * world_size
    torch.distributed.all_gather_object(
        timings, (old_elapsed_s, new_elapsed_s, object_collectives)
    )
    if rank != 0:
        return None
    assert expected is not None
    assert all(value is not None for value in gathered_payloads)
    assert all(value is not None for value in stats)
    assert all(value is not None for value in timings)
    _assert_new_bytes(expected, prepared, [value or {} for value in gathered_payloads])
    published_config = json.loads(expected["adapter_config.json"])
    expected_targets = {
        "qwen36": ["q_proj", "o_proj", "k_proj", "experts"],
        "gpt_oss": ["q_proj", "o_proj", "k_proj", "experts"],
        "gemma4": [
            "k_proj",
            "o_proj",
            "q_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "experts",
        ],
        "dsv4": [
            "fused_wqa_wkv",
            "wo_a",
            "gate_up_proj",
            "experts",
            "down_proj",
        ],
    }
    family = next(name for name in expected_targets if case.startswith(name))
    assert published_config["target_modules"] == expected_targets[family]
    assert published_config["moe_parameterization"] == (
        "shared_outer" if case.endswith("shared_outer") else "per_expert"
    )
    assert published_config[ART_LORA_FORMAT_CONFIG_KEY] == ART_LORA_FORMAT_VLLM
    if family == "gemma4":
        names = {tensor.name for tensor in prepared.layout.tensors}
        assert any(".v_proj." in name for name in names)
    if case == "qwen36_shared_outer":
        assert not any(
            ".shared." in tensor.name for tensor in prepared.layout.tensors
        )
    typed_stats = [value for value in stats if value is not None]
    typed_timings = [value for value in timings if value is not None]
    assert {value[2] for value in typed_timings} == {0 if world_size == 1 else 2}
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
        "prepare_object_collectives": typed_timings[0][2],
        "shards": len(prepared.layout.plan.shards),
    }


def _worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    report_path: str,
    fixture_root: str,
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
            for case in CASES
            if (
                report := _run_case(case, rank, world_size, Path(fixture_root))
            )
            is not None
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
    fixture_root = tmp_path / "models"
    (fixture_root / "gpt_oss").mkdir(parents=True)
    (fixture_root / "gpt_oss" / "config.json").write_text(
        json.dumps({"hidden_size": 128, "intermediate_size": 128})
    )
    (fixture_root / "gemma4").mkdir()
    (fixture_root / "gemma4" / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "enable_moe_block": True,
                    "hidden_size": 6,
                    "moe_intermediate_size": 4,
                    "attention_k_eq_v": True,
                    "layer_types": ["full_attention"] * max(4, world_size * 2),
                }
            }
        )
    )
    mp.spawn(
        _worker,
        args=(
            world_size,
            str(tmp_path / f"gloo-{world_size}"),
            str(tmp_path / f"report-{world_size}.json"),
            str(fixture_root),
        ),
        nprocs=world_size,
        join=True,
    )
    reports = json.loads((tmp_path / f"report-{world_size}.json").read_text())
    assert {report["case"] for report in reports} == set(CASES)


def _contract_worker(
    rank: int,
    rendezvous: str,
    report_path: str,
) -> None:
    world_size = 2
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        fixture = _fixture(
            "qwen36_per_expert",
            rank,
            world_size,
            Path(report_path).parent,
        )
        target = _target("qwen36_per_expert", world_size)
        if rank:
            target = target.model_copy(update={"object_id": "f" * 64})
        with pytest.raises(
            RuntimeError,
            match="publication calls are inconsistent",
        ):
            prepare_rank_distributed_vllm_lora(
                target=target,
                local_tensors=fixture.regular,
                local_metadata=fixture.regular_metadata,
                local_packed_tensors=fixture.packed,
                local_packed_metadata=fixture.packed_metadata,
                handler=fixture.handler,
                adapter_config=fixture.adapter_config,
                conversion_group_for_key=_block_for_key,
                exchange_device=torch.device("cpu"),
                stager=PinnedCpuSnapshotStager(),
            )
        with pytest.raises(
            RuntimeError,
            match="dependency group changed across ranks",
        ):
            prepare_rank_distributed_vllm_lora(
                target=_target("qwen36_per_expert", world_size),
                local_tensors=fixture.regular,
                local_metadata=fixture.regular_metadata,
                local_packed_tensors=fixture.packed,
                local_packed_metadata=fixture.packed_metadata,
                handler=fixture.handler,
                adapter_config=fixture.adapter_config,
                conversion_group_for_key=(
                    _block_for_key
                    if rank == 0
                    else lambda key: f"rank-1:{_block_for_key(key)}"
                ),
                exchange_device=torch.device("cpu"),
                stager=PinnedCpuSnapshotStager(),
            )

        class RankFailingHandler:
            key = fixture.handler.key

            def to_vllm_lora_tensors(self, *args: Any, **kwargs: Any) -> Any:
                if rank == 1:
                    raise ValueError("rank-local conversion failure")
                return fixture.handler.to_vllm_lora_tensors(*args, **kwargs)

            def to_vllm_lora_config(self, *args: Any, **kwargs: Any) -> Any:
                return fixture.handler.to_vllm_lora_config(*args, **kwargs)

        with pytest.raises(
            RuntimeError,
            match="LoRA conversion failed: rank 1: ValueError",
        ):
            prepare_rank_distributed_vllm_lora(
                target=_target("qwen36_per_expert", world_size),
                local_tensors=fixture.regular,
                local_metadata=fixture.regular_metadata,
                local_packed_tensors=fixture.packed,
                local_packed_metadata=fixture.packed_metadata,
                handler=RankFailingHandler(),
                adapter_config=fixture.adapter_config,
                conversion_group_for_key=_block_for_key,
                exchange_device=torch.device("cpu"),
                stager=PinnedCpuSnapshotStager(),
            )
        if rank == 0:
            Path(report_path).write_text("ok\n")
    finally:
        torch.distributed.destroy_process_group()


def test_rank_distributed_lora_rejects_collective_ordering_mismatch(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "contract-ok"
    mp.spawn(
        _contract_worker,
        args=(str(tmp_path / "gloo-contract"), str(report_path)),
        nprocs=2,
        join=True,
    )
    assert report_path.read_text() == "ok\n"


def test_snapshot_fence_waits_for_caller_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    caller_stream = object()

    class Stream:
        waited_for: object | None = None

        def wait_stream(self, stream: object) -> None:
            self.waited_for = stream

    class Stager:
        stream_value = Stream()

        def stream(self, device: int) -> Stream:
            assert device == 3
            return self.stream_value

    stager = Stager()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: caller_stream)
    builder = PinnedCpuSnapshotBuilder(stager)  # type: ignore[arg-type]
    builder.fence_current_stream(3)

    assert stager.stream_value.waited_for is caller_stream
    assert builder._devices == {3}


def test_generation_publisher_uses_independent_ordered_control_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import art.megatron.weights.lora_publish as lora_publish
    import art.megatron.weights.rank_distributed_lora_publish as distributed_publish

    fixture = _fixture("qwen36_per_expert", 0, 1, tmp_path)
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=7,
        generation_id="step-00000007-00000000000000000000000000000000",
        adapter_path="adapter",
    )
    target = _target("qwen36_per_expert", 1).model_copy(
        update={
            "metadata": {
                "run_id": "run",
                "training_session_id": generation.training_session_id,
                "generation_id": generation.generation_id,
                "policy_step": str(generation.policy_step),
            }
        }
    )
    monkeypatch.setattr(
        lora_publish,
        "collect_local_lora_entries",
        lambda *_args, **_kwargs: (fixture.regular, fixture.regular_metadata),
    )
    monkeypatch.setattr(
        lora_publish,
        "collect_local_packed_expert_entries",
        lambda *_args, **_kwargs: (fixture.packed, fixture.packed_metadata),
    )

    class Barrier:
        registered = []

        def register(self, pending, *, key: str) -> None:
            self.registered.append((pending, key))

    class Sink:
        events = []

        def publication(self, event) -> None:
            self.events.append(event)

    control_group = object()
    runtime = SimpleNamespace(
        rank=0,
        world_size=1,
        model=(),
        model_support_handler=fixture.handler,
        optimizer_snapshot_barrier=Barrier(),
        publication_group=control_group,
        publication_metadata_group=None,
    )
    publisher = _GenerationPublisher(runtime, capacity=2)
    plan, _metrics = publisher.prepare_ordered_sampler(
        operation_id="operation",
        run_id="run",
        generation=generation,
        optimizer_state_path=str(tmp_path / "optimizer"),
        target=target,
        adapter_dtypes={},
        adapter_config=fixture.adapter_config,
        slot_ref=None,
        sink=Sink(),
    )
    prepared = publisher._prepared["operation"]
    calls = []
    store = object()

    def publish(value, received_store, *, group):
        calls.append((value, received_store, group))
        return value.layout.ref

    monkeypatch.setattr(
        distributed_publish,
        "publish_rank_distributed_vllm_lora",
        publish,
    )
    monkeypatch.setattr(publisher, "_object_store_for", lambda _target: store)
    transport = publisher._transfer_prepared_snapshot(prepared, time.perf_counter())

    assert runtime.optimizer_snapshot_barrier.registered[0][1] == "run"
    assert plan.transport_adapter is not None
    assert transport.adapter == plan.transport_adapter
    assert calls == [(prepared.distributed_adapter, store, control_group)]
    publisher.discard("operation")
    publisher.close()
