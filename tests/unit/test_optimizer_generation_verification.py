import asyncio
import hashlib
import json
from pathlib import Path
import threading

import pytest
import torch

from art.distributed.rollout import RolloutModelSpec
from art.megatron import optimizer_state as optimizer_state_module
from art.megatron.optimizer_state import (
    CheckpointFile,
    OptimizerAdapter,
    OptimizerGenerationManifest,
    OptimizerLogicalTensor,
    OptimizerShard,
    OptimizerTopology,
    VerifiedOptimizerGeneration,
    acknowledge_materialized_adapter,
    authenticated_optimizer_generation_lease,
    build_optimizer_manifest,
    delete_optimizer_generation,
    load_trainer_rank_optimizer_state,
    optimizer_generation_path,
    optimizer_shard_name,
    validate_verified_optimizer_generation,
    verify_optimizer_generation,
)
from art.megatron.portable_optimizer_archive import (
    LoadedPortableOptimizerArchive,
    PortableOptimizerArchiveMetadata,
    PreparedPortableOptimizerArchive,
    write_portable_optimizer_archive,
)
from art.megatron.runtime.specs import (
    RankLocalOptimizerWorkSummary,
    ResolvedCheckpointState,
    RunOptimizerWorkSummary,
    RunSlotRegistration,
)
from art.megatron.training import slot as training_slot_module
from art.megatron.training.slot import MegatronTrainingSlot
from art.megatron.weights.rank_distributed_types import RankDistributedLoraStats
from art.training.contracts import LoadStateRequest, OperationRef

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


def test_generation_verification_returns_serializable_semantic_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _generation(tmp_path)
    manifest = optimizer_state_module.read_optimizer_generation_manifest(
        str(tmp_path), _GENERATION
    )
    verification_calls = 0
    verify_manifest = optimizer_state_module.verify_optimizer_generation_manifest

    def count_verification(
        optimizer_state_path: str,
        candidate: OptimizerGenerationManifest,
    ) -> None:
        nonlocal verification_calls
        verification_calls += 1
        verify_manifest(optimizer_state_path, candidate)

    monkeypatch.setattr(
        optimizer_state_module,
        "verify_optimizer_generation_manifest",
        count_verification,
    )

    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)

    assert verification_calls == 1
    assert receipt.receipt_format_version == 1
    assert receipt.manifest_format_version == manifest.format_version
    assert receipt.optimizer_semantic_sha256 == manifest.optimizer_semantic_sha256
    assert receipt.logical_state_sha256 == manifest.logical_state_sha256
    assert (
        VerifiedOptimizerGeneration.model_validate_json(receipt.model_dump_json())
        == receipt
    )
    incomplete = receipt.model_dump()
    incomplete["logical_state_sha256"] = None
    with pytest.raises(ValueError, match="both topology-portable identities"):
        VerifiedOptimizerGeneration.model_validate(incomplete)


def test_verified_generation_receipt_reuse_does_not_rehash_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard, _adapter = _generation(tmp_path)
    hashed_paths: list[Path] = []
    file_sha256 = optimizer_state_module._file_sha256

    def count_hashes(path: Path) -> str:
        hashed_paths.append(path)
        return file_sha256(path)

    monkeypatch.setattr(optimizer_state_module, "_file_sha256", count_hashes)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)

    assert hashed_paths == [shard]
    hashed_paths.clear()
    path_open = Path.open

    def reject_shard_reads(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ) -> object:
        if path == shard and "r" in mode:
            pytest.fail("receipt reuse reread optimizer shard content")
        return path_open(path, mode, *args, **kwargs)

    def reject_full_verification(*_args: object, **_kwargs: object) -> None:
        pytest.fail("receipt reuse entered full optimizer shard verification")

    monkeypatch.setattr(Path, "open", reject_shard_reads)
    monkeypatch.setattr(
        optimizer_state_module,
        "verify_optimizer_generation_manifest",
        reject_full_verification,
    )
    restored = VerifiedOptimizerGeneration.model_validate_json(
        receipt.model_dump_json()
    )

    assert (
        validate_verified_optimizer_generation(
            str(tmp_path), _GENERATION, restored
        )
        == receipt
    )
    assert hashed_paths == []


@pytest.mark.parametrize("mutation", ["generation", "semantic", "manifest"])
def test_verified_generation_receipt_reuse_rejects_identity_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    _generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)
    generation = _GENERATION
    if mutation == "generation":
        generation = f"step-00000002-{'2' * 32}"
    elif mutation == "semantic":
        receipt = VerifiedOptimizerGeneration.model_validate(
            {
                **receipt.model_dump(),
                "optimizer_semantic_sha256": "2" * 64,
            }
        )
    else:
        manifest_path = (
            optimizer_generation_path(str(tmp_path), _GENERATION) / "manifest.json"
        )
        manifest_path.write_text(manifest_path.read_text() + "\n")

    expected = (
        "names another generation"
        if mutation == "generation"
        else "receipt identity mismatch"
    )
    with pytest.raises(RuntimeError, match=expected):
        validate_verified_optimizer_generation(str(tmp_path), generation, receipt)


@pytest.mark.asyncio
async def test_run_registration_reuses_receipt_and_retains_generation_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer_state_path = tmp_path / "optimizer"
    shard, adapter = _generation(optimizer_state_path)
    receipt = VerifiedOptimizerGeneration.model_validate_json(
        verify_optimizer_generation(
            str(optimizer_state_path), _GENERATION
        ).model_dump_json()
    )
    deletion_started = threading.Event()

    def delete() -> None:
        deletion_started.set()
        delete_optimizer_generation(
            str(optimizer_state_path), _GENERATION, replacement_generation=None
        )

    optimizer_work = RunOptimizerWorkSummary(
        run_id="run",
        ranks=(
            RankLocalOptimizerWorkSummary(
                rank=0,
                adapter_rank=1,
                target_modules=("q_proj",),
                trainable_lora_numel=1,
                optimizer_passes=1,
                parameter_count=1,
                layout_fingerprint=_DIGEST,
            ),
        ),
    )

    class _RankReadingTrainer:
        valid = True

        def __init__(self) -> None:
            self.deletion: asyncio.Task[None] | None = None

        async def register_run(
            self, registration: RunSlotRegistration
        ) -> RunOptimizerWorkSummary:
            assert registration.initial_optimizer_verification == receipt
            self.deletion = asyncio.create_task(asyncio.to_thread(delete))
            assert await asyncio.to_thread(deletion_started.wait, 1)
            await asyncio.sleep(0.05)
            assert not self.deletion.done()
            assert shard.read_bytes() == b"logical optimizer bytes"
            return optimizer_work

    trainer = _RankReadingTrainer()
    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot._closed = False
    slot._batch_release_failures = []
    slot._runs = {}
    slot.artifact_root = str(tmp_path.resolve())
    slot.trainer = trainer
    monkeypatch.setattr(
        optimizer_state_module,
        "_file_sha256",
        lambda _path: pytest.fail("supplied receipt repeated optimizer shard hashing"),
    )
    monkeypatch.setattr(
        training_slot_module, "validate_adapter_manifest", lambda _adapter: None
    )

    registration = RunSlotRegistration(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=1,
        generation_id=_GENERATION,
        adapter=adapter,
        optimizer_state_path=str(optimizer_state_path),
        initial_optimizer_state_path=str(optimizer_state_path),
        initial_optimizer_generation_id=_GENERATION,
        initial_optimizer_verification=receipt,
    )
    result = await slot.register_run(
        registration,
        model=RolloutModelSpec(payload={}),
        output_dir=str(tmp_path / "output"),
    )

    assert result == optimizer_work
    assert trainer.deletion is not None
    await asyncio.wait_for(trainer.deletion, timeout=1)
    assert not optimizer_generation_path(
        str(optimizer_state_path), _GENERATION
    ).exists()


@pytest.mark.asyncio
async def test_prepare_load_state_reuses_serialized_receipt_through_rank_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer_state_path = tmp_path / "optimizer"
    shard, adapter = _generation(optimizer_state_path)
    receipt = verify_optimizer_generation(str(optimizer_state_path), _GENERATION)
    source = ResolvedCheckpointState.model_validate_json(
        ResolvedCheckpointState(
            adapter=adapter,
            optimizer_state_path=str(optimizer_state_path),
            optimizer_generation_id=_GENERATION,
            optimizer_verification=receipt,
        ).model_dump_json()
    )
    deletion_started = threading.Event()

    def delete() -> None:
        deletion_started.set()
        delete_optimizer_generation(
            str(optimizer_state_path), _GENERATION, replacement_generation=None
        )

    optimizer_work = RunOptimizerWorkSummary(
        run_id="run",
        ranks=(
            RankLocalOptimizerWorkSummary(
                rank=0,
                adapter_rank=1,
                target_modules=("q_proj",),
                trainable_lora_numel=1,
                optimizer_passes=1,
                parameter_count=1,
                layout_fingerprint=_DIGEST,
            ),
        ),
    )

    class _RankReadingTrainer:
        valid = True

        def __init__(self) -> None:
            self.deletion: asyncio.Task[None] | None = None

        async def register_run(
            self, _registration: RunSlotRegistration
        ) -> RunOptimizerWorkSummary:
            return optimizer_work

        async def prepare_load_state(self, job: object) -> None:
            assert getattr(job, "optimizer_verification") == receipt
            self.deletion = asyncio.create_task(asyncio.to_thread(delete))
            assert await asyncio.to_thread(deletion_started.wait, 1)
            await asyncio.sleep(0.05)
            assert not self.deletion.done()
            assert shard.read_bytes() == b"logical optimizer bytes"

    trainer = _RankReadingTrainer()
    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot._closed = False
    slot._batch_release_failures = []
    slot._runs = {}
    slot._results = {}
    slot.artifact_root = str(tmp_path.resolve())
    slot.trainer = trainer
    monkeypatch.setattr(
        optimizer_state_module,
        "_file_sha256",
        lambda _path: pytest.fail("load-state receipt repeated optimizer shard hashing"),
    )
    monkeypatch.setattr(
        training_slot_module, "validate_adapter_manifest", lambda _adapter: None
    )
    monkeypatch.setattr(
        training_slot_module,
        "link_adapter_generation",
        lambda *_args, **_kwargs: adapter,
    )
    registration = RunSlotRegistration(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=1,
        generation_id=_GENERATION,
        adapter=adapter,
        optimizer_state_path=str(tmp_path / "resident-optimizer"),
    )
    await slot.register_run(
        registration,
        model=RolloutModelSpec(payload={}),
        output_dir=str(tmp_path / "output"),
    )

    prepared = await slot.prepare_load_state(
        OperationRef(
            run_id="run",
            operation_id="load-operation",
            sequence_id=1,
            learner_parent_version=1,
            reserved_output_learner_version=2,
            kind="load_state",
        ),
        LoadStateRequest(
            run_id="run",
            request_id="load-request",
            sequence_id=1,
            checkpoint="checkpoint",
            restore_optimizer=True,
        ),
        source,
    )

    assert prepared.job.optimizer_verification == receipt
    assert trainer.deletion is not None
    await asyncio.wait_for(trainer.deletion, timeout=1)
    assert not optimizer_generation_path(
        str(optimizer_state_path), _GENERATION
    ).exists()


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
    assert state["optimizer"]["state"][0]["step"].item() == 31.0
    assert state["optimizer"]["state"][1]["step"].item() == 31.0


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
        param_group={"lr": 3e-5},
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
