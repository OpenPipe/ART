from pathlib import Path

import pytest

from art.megatron.local_checkpoint import (
    LOCAL_CHECKPOINT_ARCHIVE_FILENAME,
    MegatronLocalCheckpointOperations,
)
from art.megatron.runtime.portable_snapshot import (
    PortableSnapshotExportReceipt,
    PortableSnapshotFile,
    PortableSnapshotGeneration,
    PortableSnapshotInstallReceipt,
    PortableSnapshotLoadReceipt,
    PortableSnapshotRankReceipt,
    PortableSnapshotReadFile,
    PortableSnapshotReadReceipt,
    PortableSnapshotTensorOwner,
    build_portable_snapshot_archive,
)
from art.megatron.runtime.specs import TrainerGeneration
from art.training import (
    LoadStateRequest,
    OperationRef,
    SamplerPublication,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
)


def _archive(generation: TrainerGeneration, *, include_optimizer: bool = False):
    portable = PortableSnapshotGeneration(
        training_session_id=generation.training_session_id,
        policy_step=generation.policy_step,
        generation_id=generation.generation_id,
    )
    return build_portable_snapshot_archive(
        generation=portable,
        checkpoint_digest="d" * 64,
        ranks=(
            PortableSnapshotRankReceipt(
                rank=0,
                checkpoint_digest="d" * 64,
                files=tuple(
                    PortableSnapshotFile(
                        object_id=f"object/{path}",
                        relative_path=path,
                        component=(
                            "optimizer"
                            if path.startswith("optimizer/")
                            else (
                                "metadata"
                                if path in {"adapter_config.json", "checkpoint.json"}
                                else "adapter"
                            )
                        ),
                        byte_count=1,
                        sha256="a" * 64,
                        source_ref=f"local://{path}",
                    )
                    for path in (
                        "adapter_config.json",
                        "adapter_model.safetensors",
                        "checkpoint.json",
                        *(("optimizer/state.pt",) if include_optimizer else ()),
                    )
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_local_checkpoint_saves_and_selectively_loads_portable_state(
    tmp_path: Path,
) -> None:
    source = TrainerGeneration(
        training_session_id="session",
        policy_step=1,
        generation_id=f"step-00000001-{'a' * 32}",
        adapter_path=str(tmp_path / "checkpoints" / "0001"),
    )
    archive = _archive(source)

    class _Coordinator:
        async def export_run_checkpoint(self, operation):
            return PortableSnapshotExportReceipt(
                export_id=operation.operation_id,
                generation=archive.generation,
                archive=archive,
                tensor_owners=(
                    PortableSnapshotTensorOwner(
                        tensor_name="adapter.weight", shard_rank=0, rank=0
                    ),
                ),
            )

        async def install_run_checkpoint(
            self, operation, generation, archive, *, restore_optimizer
        ):
            assert archive == expected_archive
            assert not restore_optimizer
            return PortableSnapshotLoadReceipt(
                operation_id=operation.operation_id,
                generation=PortableSnapshotGeneration(
                    training_session_id=generation.training_session_id,
                    policy_step=generation.policy_step,
                    generation_id=generation.generation_id,
                ),
                install=PortableSnapshotInstallReceipt(
                    archive_sha256=archive.archive_sha256,
                    runtime_fingerprint="f" * 64,
                    restore_optimizer=False,
                    ranks=(
                        PortableSnapshotReadReceipt(
                            archive_sha256=archive.archive_sha256,
                            destination_rank=0,
                            files=tuple(
                                PortableSnapshotReadFile(
                                    source_rank=0,
                                    relative_path=file.relative_path,
                                    byte_count=file.byte_count,
                                    sha256=file.sha256,
                                )
                                for file in archive.ranks[0].files
                            ),
                        ),
                    ),
                ),
            )

    expected_archive = archive
    checkpoint_dir = Path(source.adapter_path)
    checkpoint_dir.mkdir(parents=True)
    operations = MegatronLocalCheckpointOperations(
        _Coordinator(),  # type: ignore[arg-type]
        checkpoint_dir.parent,
        run_id="run",
        training_session_id="session",
        output_adapter_root=checkpoint_dir.parent,
        optimizer_state_path=tmp_path / "optimizer",
    )
    save_operation = OperationRef(
        run_id="run",
        operation_id="save",
        sequence_id=7,
        learner_parent_version=1,
        kind="save_state",
    )
    saved = await operations.save_state(
        SaveStateRequest(
            run_id="run",
            request_id="save",
            sequence_id=7,
            checkpoint_name="0001",
        ),
        save_operation,
        source,
    )
    assert Path(saved.checkpoint.checkpoint_id) == checkpoint_dir
    assert (checkpoint_dir / LOCAL_CHECKPOINT_ARCHIVE_FILENAME).is_file()

    load_operation = OperationRef(
        run_id="run",
        operation_id="load",
        sequence_id=8,
        learner_parent_version=1,
        reserved_output_learner_version=2,
        kind="load_state",
    )
    loaded = await operations.load_state(
        LoadStateRequest(
            run_id="run",
            request_id="load",
            sequence_id=8,
            checkpoint=saved.checkpoint.checkpoint_id,
            restore_optimizer=False,
        ),
        load_operation,
    )
    assert loaded.result.checkpoint.learner_version == 1
    assert loaded.result.optimizer_restored is False
    assert loaded.generation.policy_step == 2


@pytest.mark.asyncio
async def test_local_oracle_accepts_nonpublishing_sampler_and_restores_optimizer(
    tmp_path: Path,
) -> None:
    source = TrainerGeneration(
        training_session_id="session",
        policy_step=1,
        generation_id=f"step-00000001-{'a' * 32}",
        adapter_path=str(tmp_path / "checkpoints" / "0001"),
    )
    archive = _archive(source, include_optimizer=True)
    installs = []

    class _Coordinator:
        async def export_run_checkpoint(self, operation):
            return PortableSnapshotExportReceipt(
                export_id=operation.operation_id,
                generation=archive.generation,
                archive=archive,
                tensor_owners=(
                    PortableSnapshotTensorOwner(
                        tensor_name="adapter.weight", shard_rank=0, rank=0
                    ),
                ),
            )

        async def install_run_checkpoint(
            self, operation, generation, archive, *, restore_optimizer
        ):
            installs.append((operation, generation, archive, restore_optimizer))
            return PortableSnapshotLoadReceipt(
                operation_id=operation.operation_id,
                generation=PortableSnapshotGeneration(
                    training_session_id=generation.training_session_id,
                    policy_step=generation.policy_step,
                    generation_id=generation.generation_id,
                ),
                install=PortableSnapshotInstallReceipt(
                    archive_sha256=archive.archive_sha256,
                    runtime_fingerprint="f" * 64,
                    restore_optimizer=True,
                    ranks=(
                        PortableSnapshotReadReceipt(
                            archive_sha256=archive.archive_sha256,
                            destination_rank=0,
                            files=tuple(
                                PortableSnapshotReadFile(
                                    source_rank=0,
                                    relative_path=file.relative_path,
                                    byte_count=file.byte_count,
                                    sha256=file.sha256,
                                )
                                for file in archive.ranks[0].files
                            ),
                        ),
                    ),
                ),
            )

    operations = MegatronLocalCheckpointOperations(
        _Coordinator(),  # type: ignore[arg-type]
        tmp_path / "checkpoints",
        run_id="run",
        training_session_id="session",
        output_adapter_root=tmp_path / "checkpoints",
        optimizer_state_path=tmp_path / "optimizer",
    )
    sampler_request = SaveWeightsForSamplerRequest(
        run_id="run",
        request_id="sampler",
        sequence_id=0,
        checkpoint_name="initial",
        publication=SamplerPublication(mode="none"),
    )
    sampler_operation = OperationRef(
        run_id="run",
        operation_id="sampler",
        sequence_id=0,
        learner_parent_version=1,
        kind="save_sampler",
    )
    sampler = await operations.save_weights_for_sampler(
        sampler_request, sampler_operation, source
    )
    assert sampler.lora == source.adapter_path
    assert (
        await operations.plan_artifacts(sampler_request, source)
    ).transfer_bytes == 0

    save_request = SaveStateRequest(
        run_id="run",
        request_id="save",
        sequence_id=1,
        checkpoint_name="oracle",
    )
    save_operation = OperationRef(
        run_id="run",
        operation_id="save",
        sequence_id=1,
        learner_parent_version=1,
        kind="save_state",
    )
    saved = await operations.save_state(save_request, save_operation, source)
    load_request = LoadStateRequest(
        run_id="run",
        request_id="load",
        sequence_id=2,
        checkpoint=saved.checkpoint.checkpoint_id,
        restore_optimizer=True,
    )
    load_operation = OperationRef(
        run_id="run",
        operation_id="load",
        sequence_id=2,
        learner_parent_version=1,
        reserved_output_learner_version=2,
        kind="load_state",
    )

    loaded = await operations.load_state(load_request, load_operation)

    assert loaded.result.optimizer_restored is True
    assert installs == [(load_operation, loaded.generation, archive, True)]
