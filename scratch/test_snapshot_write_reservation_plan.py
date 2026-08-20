from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import ValidationError
import pytest

from art.distributed.object_store import (
    BinaryObjectTarget,
    S3ObjectStoreConfig,
    binary_object_manifest_uri,
    vllm_lora_ordered_target,
)
from art.megatron.optimizer_state import (
    CheckpointFile,
    OptimizerAdapter,
    OptimizerShard,
    OptimizerTopology,
    canonical_adapter_path,
)
from art.megatron.runtime.publication import (
    PreparedSave,
    SnapshotRankWritePlan,
    SnapshotWriteReservationPlan,
    SnapshotWriteTargets,
    build_snapshot_write_plan,
    build_snapshot_write_reservation_plan,
)
from art.megatron.runtime.specs import TrainerGeneration


def _reservation_plan(tmp_path: Path) -> SnapshotWriteReservationPlan:
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=1,
        generation_id="step-00000001-0123456789abcdef0123456789abcdef",
        adapter_path=str(tmp_path / "checkpoints" / "0001"),
    )
    staging = tmp_path / "megatron_runtime" / "staging" / generation.generation_id
    files = (
        CheckpointFile(name="adapter_config.json", size_bytes=11),
        CheckpointFile(name="adapter_model.safetensors", size_bytes=29),
    )
    local = OptimizerAdapter(
        identity=str(canonical_adapter_path(staging, generation.policy_step)),
        training_session_id=generation.training_session_id,
        step=generation.policy_step,
        generation_id=generation.generation_id,
        files=files,
    )
    target = BinaryObjectTarget(
        store=S3ObjectStoreConfig(
            endpoint_url="https://objects.invalid",
            region="test",
            bucket="bucket",
            prefix="training",
        ),
        object_id=hashlib.sha256(b"run\0generation").hexdigest(),
        format="art_vllm_lora_v1",
        metadata={
            "run_id": "run",
            "training_session_id": generation.training_session_id,
            "generation_id": generation.generation_id,
            "policy_step": str(generation.policy_step),
        },
    )
    transport = local.model_copy(
        update={"identity": binary_object_manifest_uri(target)}
    )
    rank = SnapshotRankWritePlan(
        rank=0,
        generation=generation,
        adapter=local,
        transport_adapter=transport,
        optimizer_shard=OptimizerShard(
            rank=0,
            size_bytes=101,
            layout_sha256="1" * 64,
            sha256="2" * 64,
            serialization="art_safetensors_v1",
        ),
        runtime_sha256="3" * 64,
        topology=OptimizerTopology(world_size=1, tp=1, cp=1, ep=1, etp=1, pp=1, vpp=1),
        saves_optimizer=True,
    )
    snapshot = build_snapshot_write_plan(
        operation_id="save", generation=generation, ranks=(rank,)
    )
    return build_snapshot_write_reservation_plan(
        snapshot,
        local_adapter_staging_path=str(staging),
        optimizer_state_path=str(tmp_path / "optimizer_states"),
        writes_optimizer=True,
        adapter_object_target=target,
    )


def test_reservation_plan_round_trips_every_authorized_target(tmp_path: Path) -> None:
    plan = _reservation_plan(tmp_path)
    restored = SnapshotWriteReservationPlan.model_validate_json(plan.model_dump_json())

    assert restored == plan
    assert restored.digest == plan.digest
    assert restored.targets.local_adapter_target == plan.snapshot.ranks[0].adapter
    assert restored.targets.optimizer_state_path == str(tmp_path / "optimizer_states")
    assert restored.targets.adapter_object_target is not None
    assert restored.local_write_bytes == (
        plan.snapshot.adapter_bytes + plan.snapshot.optimizer_bytes
    )
    assert restored.local_write_paths == (
        str(
            tmp_path
            / "megatron_runtime"
            / "staging"
            / plan.snapshot.generation.generation_id
        ),
        plan.snapshot.ranks[0].adapter.identity,
        str(
            tmp_path
            / "optimizer_states"
            / "generations"
            / f".pending-{plan.snapshot.generation.generation_id}"
        ),
        str(
            tmp_path
            / "optimizer_states"
            / "generations"
            / plan.snapshot.generation.generation_id
        ),
    )

    prepared = PreparedSave(
        operation_id="save",
        kind="state",
        generation=plan.snapshot.generation,
        plan=plan.snapshot,
        plan_digest=plan.snapshot.digest,
        reservation_plan=plan,
        reservation_plan_digest=plan.digest,
    )
    assert PreparedSave.model_validate_json(prepared.model_dump_json()) == prepared


def test_reservation_plan_round_trips_ordered_sampler_target(tmp_path: Path) -> None:
    plan = _reservation_plan(tmp_path)
    generation = plan.snapshot.generation
    current = plan.targets.adapter_object_target
    assert current is not None
    target = vllm_lora_ordered_target(
        current.store,
        run_id=current.metadata["run_id"],
        training_session_id=generation.training_session_id,
        generation_id=generation.generation_id,
        policy_step=generation.policy_step,
    )
    rank = plan.snapshot.ranks[0]
    assert rank.transport_adapter is not None
    snapshot = plan.snapshot.model_copy(
        update={
            "ranks": (
                rank.model_copy(
                    update={
                        "transport_adapter": rank.transport_adapter.model_copy(
                            update={"identity": binary_object_manifest_uri(target)}
                        )
                    }
                ),
            )
        }
    )
    ordered = SnapshotWriteReservationPlan(
        snapshot=snapshot,
        targets=plan.targets.model_copy(update={"adapter_object_target": target}),
    )

    assert SnapshotWriteReservationPlan.model_validate_json(
        ordered.model_dump_json()
    ) == ordered


def test_reservation_digest_binds_targets_beyond_snapshot_content(
    tmp_path: Path,
) -> None:
    plan = _reservation_plan(tmp_path)
    moved = SnapshotWriteReservationPlan(
        snapshot=plan.snapshot,
        targets=plan.targets.model_copy(
            update={"optimizer_state_path": str(tmp_path / "other_optimizer")}
        ),
    )

    assert moved.snapshot.digest == plan.snapshot.digest
    assert moved.digest != plan.digest


def test_reservation_plan_rejects_target_identity_changes(tmp_path: Path) -> None:
    plan = _reservation_plan(tmp_path)
    wrong_staging = tmp_path / "megatron_runtime" / "staging" / "step-00000001-ffffffff"
    with pytest.raises(ValidationError, match="another generation"):
        SnapshotWriteReservationPlan(
            snapshot=plan.snapshot,
            targets=plan.targets.model_copy(
                update={"local_adapter_staging_path": str(wrong_staging)}
            ),
        )

    target = plan.targets.adapter_object_target
    assert target is not None
    with pytest.raises(ValidationError, match="another generation"):
        SnapshotWriteReservationPlan(
            snapshot=plan.snapshot,
            targets=plan.targets.model_copy(
                update={
                    "adapter_object_target": target.model_copy(
                        update={
                            "metadata": {**target.metadata, "generation_id": "other"}
                        }
                    )
                }
            ),
        )


def test_existing_snapshot_retains_exact_physical_sources(tmp_path: Path) -> None:
    plan = _reservation_plan(tmp_path)
    optimizer = str(tmp_path / "optimizer")
    existing = build_snapshot_write_reservation_plan(
        plan.snapshot, optimizer_state_path=optimizer, writes_optimizer=False
    )

    assert existing.targets == SnapshotWriteTargets(
        local_adapter_target=plan.snapshot.ranks[0].adapter,
        optimizer_state_path=optimizer,
        writes_optimizer=False,
    )
    assert existing.local_write_bytes == 0
    assert existing.local_write_paths == ()
    assert existing.digest != plan.digest
