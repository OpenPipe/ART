from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from art.megatron.distributed_service import DistributedMegatronService
from art.megatron.migrations import apply_megatron_migrations, optimizer_state_path
from art.megatron.tensor_snapshot import PinnedCpuSnapshotStager, SnapshotReadBarrier
from art.trainer_rank import TrainerRankOptimizerState


class _ResolvedSnapshot:
    def __init__(self, value: object) -> None:
        self.value = value

    def resolve(self) -> object:
        return self.value


class _RecordingBarrier:
    def __init__(self) -> None:
        self.snapshots: list[object] = []

    def register(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)


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


def test_lora_only_generation_can_add_optimizer_without_restaging_lora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.lora import LoRASlotRef
    from art.megatron.runtime.executor import _GenerationPublisher
    from art.megatron.runtime.specs import TrainerGeneration

    lora_stages: list[bool] = []
    optimizer = object()
    barrier = _RecordingBarrier()
    runtime = SimpleNamespace(
        rank=1,
        world_size=2,
        model=object(),
        model_support_handler=object(),
        optimizer_snapshot_barrier=barrier,
    )
    monkeypatch.setattr(
        "art.megatron.weights.lora_publish.stage_vllm_lora_snapshot_from_model",
        lambda **_kwargs: lora_stages.append(True),
    )
    monkeypatch.setattr(
        "art.megatron.optimizer_state.stage_trainer_rank_optimizer_state_snapshot",
        lambda *_args, **_kwargs: _ResolvedSnapshot(optimizer),
    )
    publisher = _GenerationPublisher(
        runtime, stager=PinnedCpuSnapshotStager(), capacity=1
    )
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=1,
        generation_id="step-00000001-0123456789abcdef0123456789abcdef",
        adapter_path="/tmp/step-1",
    )
    publisher.stage(
        run_id="run",
        generation=generation,
        adapter_dtypes={},
        adapter_config={},
        slot_ref=LoRASlotRef("checkpoint", "run"),
        snapshot_optimizer=False,
    )
    publisher.ensure_generation(
        run_id="run",
        generation=generation,
        adapter_dtypes={},
        adapter_config={},
        slot_ref=LoRASlotRef("checkpoint", "run"),
        trainer_rank_optimizer_state=cast(TrainerRankOptimizerState, {}),
        snapshot_optimizer=True,
    )

    entry = publisher._cache[generation.generation_id]
    assert lora_stages == [True]
    assert len(barrier.snapshots) == 1
    assert entry.optimizer_upgrade is not None
    assert entry.optimizer_upgrade.result(timeout=1) is optimizer
    assert publisher.has_generation(generation, require_optimizer=True)
    publisher.close()
