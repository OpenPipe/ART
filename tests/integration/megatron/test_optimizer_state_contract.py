from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from art.megatron.distributed_service import DistributedMegatronService
from art.megatron.migrations import apply_megatron_migrations, optimizer_state_path
from art.megatron.tensor_snapshot import SnapshotReadBarrier


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


def test_resident_optimizer_is_reused_across_objectives_in_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron import train

    old_optimizer = object()
    runtime = cast(
        train.TrainingRuntime,
        SimpleNamespace(
            optimizer_persistent=True,
            optimizer=old_optimizer,
            model=object(),
            rank=0,
            model_support_handler=object(),
            optimizer_snapshot_barrier=SnapshotReadBarrier(),
        ),
    )
    monkeypatch.setattr(train, "_load_adapter_into_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        train,
        "_build_optimizer",
        lambda *_args, **_kwargs: pytest.fail("resident optimizer was rebuilt"),
    )
    monkeypatch.setattr(
        train,
        "_load_optimizer",
        lambda *_args, **_kwargs: pytest.fail("resident optimizer was reloaded"),
    )

    adapter_dtypes = train._load_lora_and_optimizer(
        runtime,
        lora_path=str(tmp_path / "adapter"),
        optimizer_state_path=str(tmp_path / "optimizer"),
        adapter_step=4,
    )

    assert runtime.optimizer is old_optimizer
    assert adapter_dtypes == {}
