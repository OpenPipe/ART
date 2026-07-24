import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pydantic import ValidationError
import pytest

from art.megatron import optimizer_state
from art.megatron.migrations import apply_megatron_migrations, optimizer_state_path
from art.megatron.optimizer_state import (
    OPTIMIZER_MANIFEST,
    PREPARED_CHECKPOINT_MANIFEST,
    OptimizerCommit,
    PreparedCheckpointCommit,
    commit_optimizer_generation,
    optimizer_generation_files,
    read_optimizer_commit,
    read_prepared_checkpoint_commit,
    resolve_optimizer_shard_path,
    write_prepared_checkpoint_commit,
)


def _write_files(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (root / name).write_bytes(name.encode())


def _operation(name: str = "update-4") -> dict[str, Any]:
    return {
        "route": "prepared",
        "identity": {
            "receipt_binding": {
                "idempotency_key": name,
                "payload_sha256": "a" * 64,
            }
        },
        "source_revision": 4,
        "target_revision": 5,
    }


def test_schema_one_optimizer_manifest_serializes_exactly_as_before(
    tmp_path: Path,
) -> None:
    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    files = optimizer_generation_files(4, 1)
    _write_files(optimizer, files)

    commit_optimizer_generation(
        str(optimizer),
        step=4,
        world_size=1,
        files=files,
    )

    encoded = (optimizer / OPTIMIZER_MANIFEST).read_text()
    assert encoded == (
        f'{{"schema_version":1,"step":4,"world_size":1,"files":["{files[0]}"]}}'
    )
    commit = read_optimizer_commit(str(optimizer))
    assert commit is not None
    assert commit == OptimizerCommit(
        schema_version=1,
        step=4,
        world_size=1,
        files=files,
    )
    assert commit.operation_identity is None


def test_schema_two_optimizer_manifest_round_trips_canonical_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    files = optimizer_generation_files(5, 1)
    _write_files(optimizer, files)
    operation = _operation()
    reversed_operation = dict(reversed(tuple(operation.items())))

    commit_optimizer_generation(
        str(optimizer),
        step=5,
        world_size=1,
        files=files,
        operation_identity=reversed_operation,
    )

    encoded = (optimizer / OPTIMIZER_MANIFEST).read_text()
    commit = read_optimizer_commit(str(optimizer))
    assert commit is not None
    assert commit.schema_version == 2
    assert commit.operation_identity == operation
    assert json.loads(encoded)["operation_identity"] == operation
    operation_start = encoded.index('"operation_identity":')
    assert encoded.index('"identity":', operation_start) < encoded.index(
        '"route":', operation_start
    )
    assert encoded.index('"route":', operation_start) < encoded.index(
        '"source_revision":', operation_start
    )
    monkeypatch.setattr(
        optimizer_state,
        "_atomic_write",
        lambda *_args, **_kwargs: pytest.fail("same-step replay rewrote CURRENT"),
    )
    commit_optimizer_generation(
        str(optimizer),
        step=5,
        world_size=1,
        files=files,
        operation_identity=operation,
    )
    with pytest.raises(RuntimeError, match="different operation"):
        commit_optimizer_generation(
            str(optimizer),
            step=5,
            world_size=1,
            files=files,
            operation_identity=_operation("other"),
        )


def test_optimizer_commit_rejects_missing_shards_without_changing_current(
    tmp_path: Path,
) -> None:
    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    files_4 = optimizer_generation_files(4, 1)
    _write_files(optimizer, files_4)
    commit_optimizer_generation(
        str(optimizer),
        step=4,
        world_size=1,
        files=files_4,
    )
    before = (optimizer / OPTIMIZER_MANIFEST).read_bytes()
    files_5 = optimizer_generation_files(5, 2)
    (optimizer / files_5[0]).write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="missing optimizer shard"):
        commit_optimizer_generation(
            str(optimizer),
            step=5,
            world_size=2,
            files=files_5,
            operation_identity=_operation(),
        )

    assert (optimizer / OPTIMIZER_MANIFEST).read_bytes() == before
    assert read_optimizer_commit(str(optimizer)) is not None


def test_prepared_checkpoint_marker_round_trip_and_state_transition(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    operation = _operation()
    submitted = PreparedCheckpointCommit(
        state="submitted",
        step=5,
        operation_identity=dict(reversed(tuple(operation.items()))),
    )
    cast(dict[str, Any], operation["identity"])["changed_after_validation"] = True
    identity_view = cast(dict[str, Any], submitted.operation_identity["identity"])
    identity_view["changed_through_view"] = True
    write_prepared_checkpoint_commit(checkpoint, submitted)

    submitted_bytes = (checkpoint / PREPARED_CHECKPOINT_MANIFEST).read_bytes()
    assert read_prepared_checkpoint_commit(checkpoint) == submitted
    persisted = json.loads(submitted_bytes)
    persisted_identity = persisted["operation_identity"]["identity"]
    assert "changed_after_validation" not in persisted_identity
    assert "changed_through_view" not in persisted_identity
    write_prepared_checkpoint_commit(checkpoint, submitted)
    assert (checkpoint / PREPARED_CHECKPOINT_MANIFEST).read_bytes() == submitted_bytes

    files = optimizer_generation_files(5, 2)
    ready = PreparedCheckpointCommit(
        state="outputs_ready",
        step=5,
        operation_identity=submitted.operation_identity,
        world_size=2,
        files=files,
    )
    write_prepared_checkpoint_commit(checkpoint, ready)

    assert read_prepared_checkpoint_commit(checkpoint) == ready
    encoded = (checkpoint / PREPARED_CHECKPOINT_MANIFEST).read_text()
    assert '"operation_identity":{"identity":' in encoded
    assert not list(checkpoint.glob(f".{PREPARED_CHECKPOINT_MANIFEST}.*.tmp"))


def test_prepared_checkpoint_marker_rejects_invalid_or_foreign_transition(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    submitted = PreparedCheckpointCommit(
        state="submitted",
        step=5,
        operation_identity=_operation(),
    )
    write_prepared_checkpoint_commit(checkpoint, submitted)
    before = (checkpoint / PREPARED_CHECKPOINT_MANIFEST).read_bytes()

    with pytest.raises(ValidationError, match="requires optimizer generation"):
        PreparedCheckpointCommit(
            state="outputs_ready",
            step=5,
            operation_identity=_operation(),
        )
    with pytest.raises(RuntimeError, match="different operation"):
        write_prepared_checkpoint_commit(
            checkpoint,
            PreparedCheckpointCommit(
                state="outputs_ready",
                step=5,
                operation_identity=_operation("other"),
                world_size=1,
                files=optimizer_generation_files(5, 1),
            ),
        )

    assert (checkpoint / PREPARED_CHECKPOINT_MANIFEST).read_bytes() == before


def test_marker_atomic_replace_failure_preserves_submitted_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    operation = _operation()
    submitted = PreparedCheckpointCommit(
        state="submitted",
        step=5,
        operation_identity=operation,
    )
    write_prepared_checkpoint_commit(checkpoint, submitted)
    before = (checkpoint / PREPARED_CHECKPOINT_MANIFEST).read_bytes()
    real_replace = os.replace

    def fail_marker_replace(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        if Path(target).name == PREPARED_CHECKPOINT_MANIFEST:
            raise OSError("injected marker replace failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_marker_replace)
    with pytest.raises(OSError, match="injected"):
        write_prepared_checkpoint_commit(
            checkpoint,
            PreparedCheckpointCommit(
                state="outputs_ready",
                step=5,
                operation_identity=operation,
                world_size=1,
                files=optimizer_generation_files(5, 1),
            ),
        )

    assert (checkpoint / PREPARED_CHECKPOINT_MANIFEST).read_bytes() == before
    assert not list(checkpoint.glob(f".{PREPARED_CHECKPOINT_MANIFEST}.*.tmp"))


def test_optimizer_commit_preserves_previous_generation_until_manifest_advance(
    tmp_path: Path,
) -> None:
    optimizer = tmp_path / "optimizer"
    optimizer.mkdir()
    files_8 = optimizer_generation_files(8, 2)
    _write_files(optimizer, files_8)
    commit_optimizer_generation(
        str(optimizer),
        step=8,
        world_size=2,
        files=files_8,
    )

    files_9 = optimizer_generation_files(9, 2)
    (optimizer / files_9[0]).write_bytes(b"interrupted")
    commit = read_optimizer_commit(str(optimizer))
    assert commit is not None and commit.step == 8
    assert all((optimizer / name).exists() for name in files_8)

    (optimizer / files_9[1]).write_bytes(b"complete")
    commit_optimizer_generation(
        str(optimizer),
        step=9,
        world_size=2,
        files=files_9,
    )
    commit = read_optimizer_commit(str(optimizer))
    assert commit is not None and commit.step == 9
    assert not any((optimizer / name).exists() for name in files_8)
    assert all((optimizer / name).exists() for name in files_9)
    with pytest.raises(RuntimeError, match="source policy"):
        resolve_optimizer_shard_path(
            str(optimizer), rank=0, world_size=2, expected_step=8
        )


def test_complete_legacy_optimizer_without_marker_resumes_latest_lora(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model"
    optimizer = output / "optimizer_states_rl"
    (output / "checkpoints" / "0007").mkdir(parents=True)
    optimizer.mkdir()
    _write_files(optimizer, ("01-of-02.pt", "02-of-02.pt"))

    with pytest.warns(UserWarning, match="Migrated legacy RL optimizer"):
        migrated = apply_megatron_migrations(str(output))
    commit = read_optimizer_commit(migrated)
    assert migrated == optimizer_state_path(str(output))
    assert commit is not None and commit.step == 7


def test_ambiguous_legacy_optimizer_requires_explicit_selection(
    tmp_path: Path,
) -> None:
    for mode in ("rl", "sft"):
        path = tmp_path / f"optimizer_states_{mode}"
        path.mkdir()
        _write_files(path, ("01-of-01.pt",))

    with pytest.raises(RuntimeError, match="Both legacy RL and SFT"):
        apply_megatron_migrations(str(tmp_path))


def test_resident_optimizer_is_reused_across_objectives_in_one_run(
    tmp_path: Path,
) -> None:
    from art.megatron import train

    old_optimizer = object()
    runtime = cast(
        train.TrainingRuntime,
        SimpleNamespace(
            optimizer_persistent=True,
            optimizer=old_optimizer,
            optimizer_config=object(),
            model=object(),
            rank=0,
            world_size=1,
            model_support_handler=object(),
            resident_training_session_id="session",
            resident_optimizer_state_path=str(tmp_path / "optimizer"),
            resident_policy_step=4,
            resident_optimizer_dirty=False,
            optimizer_state_loaded=True,
            adapter_export_dtypes={"lora": "old"},
        ),
    )
    adapter_dtypes = train._prepare_training_state(
        runtime,
        training_session_id="session",
        source_policy_step=4,
        lora_path=str(tmp_path / "adapter"),
        optimizer_state_path=str(tmp_path / "optimizer"),
    )

    assert runtime.optimizer is old_optimizer
    assert adapter_dtypes == {"lora": "old"}
