from pathlib import Path

import pytest

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
