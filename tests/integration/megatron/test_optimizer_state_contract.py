from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from art.megatron.migrations import apply_megatron_migrations, optimizer_state_path
from art.megatron.optimizer_state import (
    commit_optimizer_generation,
    optimizer_generation_files,
    read_optimizer_commit,
    resolve_optimizer_shard_path,
)


def _write_files(root: Path, names: tuple[str, ...]) -> None:
    for name in names:
        (root / name).write_bytes(name.encode())


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
