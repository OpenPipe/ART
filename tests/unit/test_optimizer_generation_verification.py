import asyncio
import hashlib
import json
from pathlib import Path
import threading

import pytest
import torch

from art.megatron.optimizer_state import (
    CheckpointFile,
    OptimizerAdapter,
    OptimizerGenerationManifest,
    OptimizerLogicalTensor,
    OptimizerShard,
    OptimizerTopology,
    acknowledge_materialized_adapter,
    authenticated_optimizer_generation_lease,
    build_optimizer_manifest,
    delete_optimizer_generation,
    load_trainer_rank_optimizer_state,
    optimizer_generation_path,
    optimizer_shard_name,
    verify_optimizer_generation,
)
from art.megatron.portable_optimizer_archive import (
    LoadedPortableOptimizerArchive,
    PortableOptimizerArchiveMetadata,
    PreparedPortableOptimizerArchive,
    write_portable_optimizer_archive,
)
from art.megatron.weights.rank_distributed_types import RankDistributedLoraStats

_DIGEST = "0" * 64
_GENERATION = f"step-00000001-{'1' * 32}"


def _generation(
    root: Path, *, optimizer_semantic_sha256: str = "1" * 64
) -> tuple[Path, OptimizerAdapter]:
    adapter_path = root.parent / f"{root.name}-adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_bytes(b"c")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"m")
    files = (
        CheckpointFile(name="adapter_config.json", size_bytes=1),
        CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
    )
    adapter = acknowledge_materialized_adapter(
        adapter_path,
        step=1,
        training_session_id="session",
        generation_id=_GENERATION,
        files=files,
    )
    generation = optimizer_generation_path(str(root), _GENERATION)
    generation.mkdir(parents=True)
    shard_path = generation / optimizer_shard_name(
        0, 1, "art_logical_safetensors_v1"
    )
    shard_path.write_bytes(b"logical optimizer bytes")
    manifest = build_optimizer_manifest(
        generation=_GENERATION,
        step=1,
        adapter=adapter,
        runtime_sha256=_DIGEST,
        optimizer_semantic_sha256=optimizer_semantic_sha256,
        world_size=1,
        topology=OptimizerTopology(
            world_size=1,
            tp=1,
            cp=1,
            ep=1,
            etp=1,
            pp=1,
            vpp=1,
        ),
        shards=[
            OptimizerShard(
                rank=0,
                size_bytes=shard_path.stat().st_size,
                layout_sha256=_DIGEST,
                sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                serialization="art_logical_safetensors_v1",
                logical_tensors=(
                    OptimizerLogicalTensor(key="weight", shape=(2, 4)),
                ),
            )
        ],
    )
    (generation / "manifest.json").write_text(manifest.model_dump_json())
    return shard_path, adapter


def _real_logical_generation(
    root: Path,
) -> tuple[OptimizerAdapter, dict[str, tuple[int, ...]]]:
    adapter_path = root.parent / f"{root.name}-real-adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_bytes(b"c")
    (adapter_path / "adapter_model.safetensors").write_bytes(b"m")
    files = (
        CheckpointFile(name="adapter_config.json", size_bytes=1),
        CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
    )
    adapter = acknowledge_materialized_adapter(
        adapter_path,
        step=1,
        training_session_id="session",
        generation_id=_GENERATION,
        files=files,
    )
    generation = optimizer_generation_path(str(root), _GENERATION)
    generation.mkdir(parents=True)
    rank_values = (
        {},
        {
            "layer.shared.lora_A.weight": 1.0,
            "layer.4.lora_B.weight": 2.0,
        },
        {"layer.7.lora_B.weight": 3.0},
    )
    logical_shapes = {key: (2, 3) for values in rank_values for key in values}
    shards: list[OptimizerShard] = []
    for rank, values in enumerate(rank_values):
        keys = tuple(sorted(values))
        tensors = {
            f"{component}/{key}": torch.full(
                logical_shapes[key], value + offset, dtype=torch.float32
            )
            for component, offset in (
                ("master", 0.0),
                ("exp_avg", 10.0),
                ("exp_avg_sq", 20.0),
            )
            for key, value in values.items()
        }
        prepared = PreparedPortableOptimizerArchive(
            metadata=PortableOptimizerArchiveMetadata(
                source_rank=rank,
                source_world_size=len(rank_values),
                logical_keys=keys,
                steps=dict.fromkeys(keys, 31.0),
                param_group={
                    "lr": 3e-5,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "step": 31.0,
                    "weight_decay": 0.1,
                },
            ),
            tensors=tensors,
            exchange_stats=RankDistributedLoraStats(
                rank=rank,
                world_size=len(rank_values),
                source_bytes=0,
                sent_bytes=0,
                received_bytes=0,
                owned_tensor_bytes=0,
                peak_accounted_owner_bytes=0,
                owned_upload_bytes=0,
                owned_tensor_count=len(tensors),
                owned_block_count=0,
            ),
        )
        path = generation / optimizer_shard_name(
            rank, len(rank_values), "art_logical_safetensors_v1"
        )
        identity = write_portable_optimizer_archive(prepared, path)
        shards.append(
            OptimizerShard(
                rank=rank,
                size_bytes=identity.size_bytes,
                layout_sha256=_DIGEST,
                sha256=identity.sha256,
                serialization="art_logical_safetensors_v1",
                logical_tensors=tuple(
                    OptimizerLogicalTensor(key=key, shape=logical_shapes[key])
                    for key in keys
                ),
            )
        )
    manifest = build_optimizer_manifest(
        generation=_GENERATION,
        step=1,
        adapter=adapter,
        runtime_sha256=_DIGEST,
        optimizer_semantic_sha256="1" * 64,
        world_size=len(rank_values),
        topology=OptimizerTopology(
            world_size=len(rank_values),
            tp=1,
            cp=1,
            ep=len(rank_values),
            etp=1,
            pp=1,
            vpp=1,
        ),
        shards=shards,
    )
    (generation / "manifest.json").write_text(manifest.model_dump_json())
    return adapter, logical_shapes


class _SharedOuterPaddedDestination:
    moe_parameterization = "shared_outer"

    def __init__(self, expert_ids: tuple[int | None, ...]) -> None:
        self.expert_ids = expert_ids

    def _expected_weight_keys_for_param(
        self, suffix: str, _parameter: torch.Tensor
    ) -> tuple[str, ...]:
        if suffix == "lora_A":
            return ("layer.shared.lora_A.weight",)
        return tuple(
            f"layer.{expert}.lora_B.weight"
            for expert in self.expert_ids
            if expert is not None
        )

    def _adapter_weight(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        suffix: str,
        moe_parameterization: object,
    ) -> torch.Tensor:
        assert moe_parameterization == "shared_outer"
        if suffix == "lora_A":
            return tensors["layer.shared.lora_A.weight"]
        real = iter(
            tensors[f"layer.{expert}.lora_B.weight"]
            for expert in self.expert_ids
            if expert is not None
        )
        first = next(iter(tensors.values()))
        return torch.stack(
            [
                torch.zeros_like(first) if expert is None else next(real)
                for expert in self.expert_ids
            ]
        )

    @staticmethod
    def _localized_weight(
        weight: torch.Tensor, *, into: torch.Tensor
    ) -> torch.Tensor:
        assert weight.shape == into.shape
        return weight


def test_generation_verification_rejects_same_size_shard_corruption(
    tmp_path: Path,
) -> None:
    shard, _adapter = _generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)
    assert receipt.generation == _GENERATION

    encoded = bytearray(shard.read_bytes())
    encoded[-1] ^= 1
    shard.write_bytes(encoded)

    with pytest.raises(RuntimeError, match="Optimizer shard digest mismatch"):
        verify_optimizer_generation(str(tmp_path), _GENERATION)


@pytest.mark.parametrize("old_format", [3, 4])
def test_logical_optimizer_manifest_rejects_pre_semantic_format(
    old_format: int,
) -> None:
    manifest = build_optimizer_manifest(
        generation=_GENERATION,
        step=1,
        adapter=OptimizerAdapter(
            identity="adapter",
            training_session_id="session",
            step=1,
            generation_id=_GENERATION,
            files=(
                CheckpointFile(name="adapter_config.json", size_bytes=1),
                CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
            ),
        ),
        runtime_sha256=_DIGEST,
        optimizer_semantic_sha256="1" * 64,
        world_size=1,
        topology=OptimizerTopology(
            world_size=1,
            tp=1,
            cp=1,
            ep=1,
            etp=1,
            pp=1,
            vpp=1,
        ),
        shards=[
            OptimizerShard(
                rank=0,
                size_bytes=1,
                layout_sha256=_DIGEST,
                sha256=_DIGEST,
                serialization="art_logical_safetensors_v1",
                logical_tensors=(
                    OptimizerLogicalTensor(key="weight", shape=(2, 4)),
                ),
            )
        ],
    )
    payload = manifest.model_dump(mode="json")
    payload["format_version"] = old_format
    with pytest.raises(ValueError, match="format_version"):
        OptimizerGenerationManifest.model_validate(payload)


def test_logical_state_identity_ignores_source_topology_and_rank_ownership() -> None:
    adapter = OptimizerAdapter(
        identity="adapter",
        training_session_id="session",
        step=1,
        generation_id=_GENERATION,
        files=(
            CheckpointFile(name="adapter_config.json", size_bytes=1),
            CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
        ),
    )
    tensors = (
        OptimizerLogicalTensor(key="a", shape=(2, 4)),
        OptimizerLogicalTensor(key="b", shape=(3, 5)),
    )
    single = build_optimizer_manifest(
        generation=_GENERATION,
        step=1,
        adapter=adapter,
        runtime_sha256=_DIGEST,
        optimizer_semantic_sha256="1" * 64,
        world_size=1,
        topology=OptimizerTopology(
            world_size=1, tp=1, cp=1, ep=1, etp=1, pp=1, vpp=1
        ),
        shards=[
            OptimizerShard(
                rank=0,
                size_bytes=1,
                layout_sha256="2" * 64,
                sha256="3" * 64,
                serialization="art_logical_safetensors_v1",
                logical_tensors=tensors,
            )
        ],
    )
    repartitioned = build_optimizer_manifest(
        generation=_GENERATION,
        step=1,
        adapter=adapter,
        runtime_sha256="4" * 64,
        optimizer_semantic_sha256="1" * 64,
        world_size=2,
        topology=OptimizerTopology(
            world_size=2, tp=1, cp=2, ep=1, etp=1, pp=1, vpp=1
        ),
        shards=[
            OptimizerShard(
                rank=0,
                size_bytes=2,
                layout_sha256="5" * 64,
                sha256="6" * 64,
                serialization="art_logical_safetensors_v1",
                logical_tensors=(tensors[0],),
            ),
            OptimizerShard(
                rank=1,
                size_bytes=3,
                layout_sha256="7" * 64,
                sha256="8" * 64,
                serialization="art_logical_safetensors_v1",
                logical_tensors=(tensors[1],),
            ),
        ],
    )

    assert single.logical_state_sha256 == repartitioned.logical_state_sha256


@pytest.mark.parametrize("mutation", ["key", "shape", "lineage"])
def test_logical_state_manifest_rejects_semantic_mutation(mutation: str) -> None:
    adapter = OptimizerAdapter(
        identity="adapter",
        training_session_id="session",
        step=1,
        generation_id=_GENERATION,
        files=(
            CheckpointFile(name="adapter_config.json", size_bytes=1),
            CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
        ),
    )
    manifest = build_optimizer_manifest(
        generation=_GENERATION,
        step=1,
        adapter=adapter,
        runtime_sha256=_DIGEST,
        optimizer_semantic_sha256="1" * 64,
        world_size=1,
        shards=[
            OptimizerShard(
                rank=0,
                size_bytes=1,
                layout_sha256=_DIGEST,
                sha256=_DIGEST,
                serialization="art_logical_safetensors_v1",
                logical_tensors=(
                    OptimizerLogicalTensor(key="weight", shape=(2, 4)),
                ),
            )
        ],
        topology=OptimizerTopology(
            world_size=1, tp=1, cp=1, ep=1, etp=1, pp=1, vpp=1
        ),
    )
    payload = manifest.model_dump(mode="json")
    if mutation == "key":
        payload["shards"][0]["logical_tensors"][0]["key"] = "other"
    elif mutation == "shape":
        payload["shards"][0]["logical_tensors"][0]["shape"] = [2, 5]
    else:
        payload["adapter"]["training_session_id"] = "other-session"

    with pytest.raises(ValueError, match="logical-state fingerprint mismatch"):
        OptimizerGenerationManifest.model_validate(payload)


def test_rank_loader_authenticates_the_verified_manifest(tmp_path: Path) -> None:
    _shard, adapter = _generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)
    manifest_path = optimizer_generation_path(str(tmp_path), _GENERATION) / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")

    with pytest.raises(
        RuntimeError, match="Optimizer generation manifest verification mismatch"
    ):
        load_trainer_rank_optimizer_state(
            optimizer_state_path=str(tmp_path),
            adapter_path=adapter.identity,
            adapter_step=adapter.step,
            optimizer_generation_id=_GENERATION,
            verification=receipt,
            expected_optimizer_semantic_sha256="1" * 64,
            layout={},
            sites=(),
            expected_shapes={},
        )


def _runtime_semantic_sha256(*, optimizer_implementation: str) -> str:
    payload = {
        "model": "same-model",
        "lora": {
            "rank": 1,
            "targets": ["experts"],
            "parameterization": "shared_outer",
        },
        "optimizer": {
            "implementation": optimizer_implementation,
            "algorithm": "adamw",
            "state_schema": ["master", "exp_avg", "exp_avg_sq", "step"],
            "parameter_group_construction": "one_group_residency_order_v1",
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_rank_loader_rejects_same_model_lora_with_incompatible_optimizer_semantics(
    tmp_path: Path,
) -> None:
    saved_semantics = _runtime_semantic_sha256(
        optimizer_implementation="transformer_engine.FusedAdam"
    )
    incompatible_semantics = _runtime_semantic_sha256(
        optimizer_implementation="torch.optim.AdamW"
    )
    _shard, adapter = _generation(
        tmp_path, optimizer_semantic_sha256=saved_semantics
    )
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)

    with pytest.raises(RuntimeError, match="logical runtime mismatch"):
        load_trainer_rank_optimizer_state(
            optimizer_state_path=str(tmp_path),
            adapter_path=adapter.identity,
            adapter_step=adapter.step,
            optimizer_generation_id=_GENERATION,
            verification=receipt,
            expected_optimizer_semantic_sha256=incompatible_semantics,
            layout={},
            sites=(),
            expected_shapes={},
        )


def test_rank_loader_rejects_destination_logical_geometry_change(
    tmp_path: Path,
) -> None:
    _shard, adapter = _generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)

    with pytest.raises(RuntimeError, match="logical geometry differs"):
        load_trainer_rank_optimizer_state(
            optimizer_state_path=str(tmp_path),
            adapter_path=adapter.identity,
            adapter_step=adapter.step,
            optimizer_generation_id=_GENERATION,
            verification=receipt,
            expected_optimizer_semantic_sha256="1" * 64,
            layout={},
            sites=(),
            expected_shapes={"weight": (2, 5)},
        )


def test_real_archive_loader_reconstructs_empty_destination_rank(
    tmp_path: Path,
) -> None:
    adapter, logical_shapes = _real_logical_generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)

    state = load_trainer_rank_optimizer_state(
        optimizer_state_path=str(tmp_path),
        adapter_path=adapter.identity,
        adapter_step=adapter.step,
        optimizer_generation_id=_GENERATION,
        verification=receipt,
        expected_optimizer_semantic_sha256="1" * 64,
        layout={"parameters": ()},
        sites=(),
        expected_shapes=logical_shapes,
    )

    assert state["master_params"] == ()
    assert state["optimizer"]["state"] == {}
    assert state["optimizer"]["param_groups"] == [
        {
            "lr": 3e-5,
            "betas": [0.9, 0.95],
                "eps": 1e-8,
                "step": 31.0,
                "weight_decay": 0.1,
                "params": [],
        }
    ]


def test_real_archive_loader_reconstructs_shared_outer_moe_with_padding(
    tmp_path: Path,
) -> None:
    adapter, logical_shapes = _real_logical_generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)
    module = _SharedOuterPaddedDestination((4, None, 7))
    slot = type(
        "Slot",
        (),
        {
            "A_T": torch.nn.Parameter(torch.empty(2, 3)),
            "B_T": torch.nn.Parameter(torch.empty(3, 2, 3)),
        },
    )()
    layout = {
        "parameters": (
            (("A",), (2, 3), "torch.float32", "cpu", True, None, "", ()),
            (("B",), (3, 2, 3), "torch.float32", "cpu", True, None, "", ()),
        )
    }

    state = load_trainer_rank_optimizer_state(
        optimizer_state_path=str(tmp_path),
        adapter_path=adapter.identity,
        adapter_step=adapter.step,
        optimizer_generation_id=_GENERATION,
        verification=receipt,
        expected_optimizer_semantic_sha256="1" * 64,
        layout=layout,
        sites=((module, slot),),
        expected_shapes=logical_shapes,
    )

    torch.testing.assert_close(state["master_params"][0], torch.full((2, 3), 1.0))
    torch.testing.assert_close(
        state["master_params"][1],
        torch.stack(
            (
                torch.full((2, 3), 2.0),
                torch.zeros((2, 3)),
                torch.full((2, 3), 3.0),
            )
        ),
    )
    torch.testing.assert_close(
        state["optimizer"]["state"][1]["exp_avg"],
        torch.stack(
            (
                torch.full((2, 3), 12.0),
                torch.zeros((2, 3)),
                torch.full((2, 3), 13.0),
            )
        ),
    )
    assert "step" not in state["optimizer"]["state"][0]
    assert "step" not in state["optimizer"]["state"][1]
    assert state["optimizer"]["param_groups"][0]["step"] == 31.0


@pytest.mark.asyncio
async def test_authenticated_lease_blocks_generation_deletion_through_rank_read(
    tmp_path: Path,
) -> None:
    shard, _adapter = _generation(tmp_path)
    deletion_started = threading.Event()

    def delete() -> None:
        deletion_started.set()
        delete_optimizer_generation(
            str(tmp_path), _GENERATION, replacement_generation=None
        )

    async with authenticated_optimizer_generation_lease(
        str(tmp_path), _GENERATION
    ) as receipt:
        deletion = asyncio.create_task(asyncio.to_thread(delete))
        assert await asyncio.to_thread(deletion_started.wait, 1)
        await asyncio.sleep(0.05)
        assert receipt.generation == _GENERATION
        assert not deletion.done()
        assert shard.read_bytes() == b"logical optimizer bytes"

    await asyncio.wait_for(deletion, timeout=1)
    assert not optimizer_generation_path(str(tmp_path), _GENERATION).exists()


def test_rank_loader_does_not_repeat_full_shard_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shard, adapter = _generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)
    metadata = PortableOptimizerArchiveMetadata(
        source_rank=0,
        source_world_size=1,
        logical_keys=("weight",),
        steps={"weight": 1.0},
        param_group={"lr": 3e-5, "step": 1.0},
    )
    monkeypatch.setattr(
        "art.megatron.optimizer_state._file_sha256",
        lambda _path: pytest.fail("rank-local load repeated full-shard hashing"),
    )
    monkeypatch.setattr(
        "art.megatron.optimizer_state._loaded_adapter", lambda *_args: adapter
    )
    monkeypatch.setattr(
        "art.megatron.optimizer_state._validate_adapter_publication",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "art.megatron.portable_optimizer_archive.read_portable_optimizer_archive",
        lambda _path, *, logical_keys: LoadedPortableOptimizerArchive(
            metadata=metadata,
            loaded_logical_keys=tuple(sorted(logical_keys)),
            tensors={},
        ),
    )

    state = load_trainer_rank_optimizer_state(
        optimizer_state_path=str(tmp_path),
        adapter_path=adapter.identity,
        adapter_step=adapter.step,
        optimizer_generation_id=_GENERATION,
        verification=receipt,
        expected_optimizer_semantic_sha256="1" * 64,
        layout={"parameters": ()},
        sites=(),
        expected_shapes={"weight": (2, 4)},
    )

    assert state["master_params"] == ()
