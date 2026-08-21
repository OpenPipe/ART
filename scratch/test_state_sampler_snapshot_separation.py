from types import SimpleNamespace

import pytest

from art.distributed.object_store import S3ObjectStoreConfig
from art.megatron.optimizer_state import CheckpointFile, OptimizerAdapter
from art.megatron.runtime.publication import (
    SnapshotRankWritePlan,
    build_snapshot_write_plan,
    build_snapshot_write_reservation_plan,
)
from art.megatron.runtime.specs import GenerationSnapshotJobSpec, TrainerGeneration
from art.megatron.training.slot import (
    MegatronTrainingSlot,
    _ExistingSnapshot,
    _SnapshotSource,
)
from art.training.contracts import (
    OperationRef,
    SamplerPublication,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
)


def _adapter() -> OptimizerAdapter:
    return OptimizerAdapter(
        identity="/tmp/adapter",
        training_session_id="session",
        step=3,
        generation_id="step-00000003-0123456789abcdef0123456789abcdef",
        files=(
            CheckpointFile(name="adapter_config.json", size_bytes=1),
            CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
        ),
    )


def _ref(kind: str, operation_id: str) -> OperationRef:
    return OperationRef(
        run_id="run",
        operation_id=operation_id,
        sequence_id=0,
        learner_parent_version=3,
        kind=kind,
    )


def _slot(calls: list[tuple[bool, bool]]) -> MegatronTrainingSlot:
    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    adapter = _adapter()
    slot._closed = False
    slot._batch_release_failures = []
    slot._results = {}
    slot._pending_results = {}
    slot._prepared_saves = {}
    slot._runs = {
        "run": SimpleNamespace(
            registration=SimpleNamespace(optimizer_state_path="/tmp/optimizer"),
            generation=TrainerGeneration(
                training_session_id=adapter.training_session_id,
                policy_step=adapter.step,
                generation_id=adapter.generation_id,
                adapter_path=adapter.identity,
            ),
            output_dir="/tmp/output",
        )
    }

    async def start_snapshot(_ref, *, save_optimizer, publish_sampler, **_kwargs):
        calls.append((save_optimizer, publish_sampler))
        adapter = _adapter()
        generation = TrainerGeneration(
            training_session_id=adapter.training_session_id,
            policy_step=adapter.step,
            generation_id=adapter.generation_id,
            adapter_path=adapter.identity,
        )
        plan = build_snapshot_write_plan(
            operation_id=_ref.operation_id,
            generation=generation,
            ranks=(
                SnapshotRankWritePlan(
                    rank=0,
                    generation=generation,
                    adapter=adapter,
                    saves_optimizer=False,
                ),
            ),
        )
        return _ExistingSnapshot(
            adapter=adapter,
            optimizer_bytes=1,
            plan=plan,
            reservation_plan=build_snapshot_write_reservation_plan(
                plan, writes_optimizer=False
            ),
        )

    slot._start_snapshot = start_snapshot
    return slot


@pytest.mark.asyncio
async def test_save_state_and_sampler_publication_are_independent() -> None:
    calls: list[tuple[bool, bool]] = []
    slot = _slot(calls)
    state = await slot.start_save_state(
        _ref("save_state", "save-state"),
        SaveStateRequest(
            run_id="run",
            request_id="save-state",
            sequence_id=0,
            checkpoint_name="state",
        ),
    )
    sampler = await slot.start_save_weights_for_sampler(
        _ref("save_sampler", "save-sampler"),
        SaveWeightsForSamplerRequest(
            run_id="run",
            request_id="save-sampler",
            sequence_id=0,
            checkpoint_name="sampler",
            publication=SamplerPublication(mode="none"),
        ),
    )
    await state
    await sampler
    assert calls == [(True, False), (False, True)]


@pytest.mark.asyncio
async def test_ordered_sampler_snapshot_does_not_add_existing_local_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    adapter = _adapter()
    generation = TrainerGeneration(
        training_session_id=adapter.training_session_id,
        policy_step=adapter.step,
        generation_id=adapter.generation_id,
        adapter_path=adapter.identity,
    )
    jobs: list[GenerationSnapshotJobSpec] = []

    class StopAfterCapture(Exception):
        pass

    class Trainer:
        async def prepare_snapshot(self, job: GenerationSnapshotJobSpec):
            jobs.append(job)
            raise StopAfterCapture

    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot.trainer = Trainer()
    slot.sampler_store = S3ObjectStoreConfig(
        endpoint_url="https://objects.invalid",
        region="test",
        bucket="bucket",
        prefix="training",
    )
    monkeypatch.setattr(
        "art.megatron.training.slot.read_adapter_publication",
        lambda *_args, **_kwargs: adapter,
    )
    monkeypatch.setattr(
        "art.megatron.training.slot.read_committed_optimizer_pointer",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(StopAfterCapture):
        await slot._start_snapshot(
            _ref("save_sampler", "save-sampler"),
            source=_SnapshotSource(
                run_id="run",
                generation=generation,
                output_dir=str(tmp_path),
                optimizer_state_path=str(tmp_path / "optimizer"),
            ),
            save_optimizer=False,
            publish_sampler=True,
        )

    assert len(jobs) == 1
    assert jobs[0].adapter_object_target is not None
    assert jobs[0].existing_adapter is None
    assert jobs[0].staging_adapter_path is None
    assert not jobs[0].save_optimizer
