from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from art.megatron.distributed_service import DistributedMegatronService
from art.megatron.migrations import apply_megatron_migrations, optimizer_state_path


def test_split_optimizer_root_moves_to_unified_path(tmp_path: Path) -> None:
    output = tmp_path / "model"
    optimizer = output / "optimizer_states_rl"
    generation = optimizer / "generations" / "interrupted"
    generation.mkdir(parents=True)
    (generation / "shard").write_bytes(b"state")

    with pytest.warns(UserWarning, match="Migrated split Megatron optimizer"):
        migrated = apply_megatron_migrations(str(output))

    assert migrated == optimizer_state_path(str(output))
    assert not optimizer.exists()
    assert (Path(migrated) / "generations" / "interrupted" / "shard").is_file()


def test_ambiguous_legacy_optimizer_requires_explicit_selection(
    tmp_path: Path,
) -> None:
    for mode in ("rl", "sft"):
        path = tmp_path / f"optimizer_states_{mode}"
        (path / "generations").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Both legacy RL and SFT"):
        apply_megatron_migrations(str(tmp_path))


def test_loose_optimizer_shards_are_not_silently_upgraded(tmp_path: Path) -> None:
    path = tmp_path / "optimizer_states_rl"
    path.mkdir()
    (path / "01-of-01.pt").write_bytes(b"state")

    with pytest.raises(RuntimeError, match="Legacy optimizer checkpoint format"):
        apply_megatron_migrations(str(tmp_path))


def test_service_uses_one_optimizer_root_for_all_objectives(tmp_path: Path) -> None:
    service = cast(
        DistributedMegatronService, SimpleNamespace(output_dir=str(tmp_path))
    )

    assert DistributedMegatronService._optimizer_state_path.__get__(
        service, DistributedMegatronService
    ) == optimizer_state_path(str(tmp_path))
