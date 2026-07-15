from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from art.megatron import train
from art.megatron.optimizer_state import (
    commit_optimizer_generation,
    optimizer_generation_files,
    read_optimizer_commit,
    resolve_megatron_resume_step,
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
        training_mode="rl",
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
        training_mode="rl",
        files=files_9,
    )
    commit = read_optimizer_commit(str(optimizer))
    assert commit is not None and commit.step == 9
    assert not any((optimizer / name).exists() for name in files_8)
    assert all((optimizer / name).exists() for name in files_9)
    with pytest.raises(RuntimeError, match="training mode"):
        resolve_optimizer_shard_path(
            str(optimizer), rank=0, world_size=2, training_mode="sft"
        )


def test_complete_legacy_optimizer_without_marker_resumes_latest_lora(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model"
    optimizer = tmp_path / "optimizer"
    (output / "checkpoints" / "0007").mkdir(parents=True)
    optimizer.mkdir()
    _write_files(optimizer, ("01-of-02.pt", "02-of-02.pt"))

    resume = resolve_megatron_resume_step(
        output_dir=str(output), optimizer_state_path=str(optimizer)
    )
    assert resume.step == resume.optimizer_step == 7


def test_resident_optimizer_is_not_reused_across_training_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_optimizer = object()
    new_optimizer = SimpleNamespace()
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
            resident_training_mode="rl",
            resident_optimizer_state_path=str(tmp_path / "rl"),
            resident_policy_step=4,
            resident_optimizer_dirty=False,
            optimizer_state_loaded=True,
            adapter_export_dtypes={"lora": "old"},
        ),
    )
    monkeypatch.setattr(
        train,
        "_load_adapter_into_model",
        lambda *_args, **_kwargs: {"lora": SimpleNamespace(dtype="bf16")},
    )
    monkeypatch.setattr(train, "_build_optimizer", lambda *_args: new_optimizer)

    train._prepare_training_state(
        runtime,
        training_session_id="session",
        training_mode="sft",
        source_policy_step=4,
        lora_path=str(tmp_path / "adapter"),
        optimizer_state_path=str(tmp_path / "sft"),
    )

    assert runtime.optimizer is new_optimizer
    assert runtime.resident_training_mode == "sft"
    assert runtime.resident_optimizer_state_path == str(tmp_path / "sft")
