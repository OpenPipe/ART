import hashlib
from pathlib import Path

import pytest

from art.megatron.optimizer_state import (
    CheckpointFile,
    OptimizerAdapter,
    OptimizerGenerationManifest,
    OptimizerShard,
    OptimizerTopology,
    build_optimizer_manifest,
    load_trainer_rank_optimizer_state,
    optimizer_generation_path,
    optimizer_shard_name,
    verify_optimizer_generation,
)

_DIGEST = "0" * 64
_GENERATION = f"step-00000001-{'1' * 32}"


def _generation(root: Path) -> tuple[Path, OptimizerAdapter]:
    adapter = OptimizerAdapter(
        identity=str(root / "adapter"),
        training_session_id="session",
        step=1,
        generation_id=_GENERATION,
        files=(
            CheckpointFile(name="adapter_config.json", size_bytes=1),
            CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
        ),
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
                size_bytes=shard_path.stat().st_size,
                layout_sha256=_DIGEST,
                sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                serialization="art_logical_safetensors_v1",
                logical_keys=("weight",),
            )
        ],
    )
    (generation / "manifest.json").write_text(manifest.model_dump_json())
    return shard_path, adapter


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


def test_logical_optimizer_manifest_rejects_pre_semantic_format() -> None:
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
                logical_keys=("weight",),
            )
        ],
    )
    payload = manifest.model_dump(mode="json")
    payload["format_version"] = 3
    with pytest.raises(ValueError, match="format_version"):
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
            expected_keys=(),
        )


def test_rank_loader_rejects_a_different_logical_runtime(tmp_path: Path) -> None:
    _shard, adapter = _generation(tmp_path)
    receipt = verify_optimizer_generation(str(tmp_path), _GENERATION)

    with pytest.raises(RuntimeError, match="logical runtime mismatch"):
        load_trainer_rank_optimizer_state(
            optimizer_state_path=str(tmp_path),
            adapter_path=adapter.identity,
            adapter_step=adapter.step,
            optimizer_generation_id=_GENERATION,
            verification=receipt,
            expected_optimizer_semantic_sha256="2" * 64,
            layout={},
            sites=(),
            expected_keys=(),
        )
