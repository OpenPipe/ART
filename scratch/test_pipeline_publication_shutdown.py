import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from art import TrainableModel
from art.pipeline_trainer import PipelineRuntimeConfig, PipelineTrainer
from art.pipeline_trainer import trainer as trainer_module


class _Backend:
    def __init__(
        self,
        finalize: Callable[[], Awaitable[dict[str, float]]] | None = None,
    ) -> None:
        self._finalize = finalize
        self.finalize_calls = 0

    async def register(self, _model) -> None:
        return None

    async def _get_step(self, _model) -> int:
        return 0

    async def finalize_training_session(self, _model) -> dict[str, float]:
        self.finalize_calls += 1
        if self._finalize is None:
            return {}
        return await self._finalize()


def _trainer(tmp_path: Path, backend: _Backend) -> PipelineTrainer:
    model = TrainableModel(
        name="pipeline-shutdown",
        run_name="pipeline-shutdown",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    model._backend = backend  # type: ignore[assignment]
    trainer = PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=MagicMock(),
        scenarios=[],
        config={},
        pipeline=PipelineRuntimeConfig(
            num_rollout_workers=1,
            min_batch_size=1,
            max_batch_size=1,
        ),
        eval_fn=None,
        resume=False,
    )
    trainer._status = MagicMock()
    return trainer


def _replace_stages(trainer: PipelineTrainer, training_stage) -> None:
    async def wait_for_stop() -> None:
        await trainer._stop_event.wait()

    trainer._rollout_stage = wait_for_stop  # type: ignore[method-assign]
    trainer._training_stage = training_stage  # type: ignore[method-assign]
    trainer._eval_stage = wait_for_stop  # type: ignore[method-assign]
    trainer._status_loop = wait_for_stop  # type: ignore[method-assign]


def _pipeline_tasks() -> set[asyncio.Task]:
    stage_names = {
        "rollout_stage",
        "packing_stage",
        "training_stage",
        "eval_stage",
        "status_loop",
    }
    task_prefixes = (
        "pipeline_",
        "post_train_",
        "publication_metrics_",
        "checkpoint_log_",
    )
    return {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and (
            task.get_name() in stage_names or task.get_name().startswith(task_prefixes)
        )
    }


@pytest.mark.asyncio
async def test_clean_stop_finalizes_owner_without_failure_mode(tmp_path: Path) -> None:
    backend = _Backend()
    trainer = _trainer(tmp_path, backend)
    attachment_modes: list[bool] = []

    async def training_stage() -> None:
        trainer._backend_training_started = True
        trainer.request_stop()

    async def stop_attachments(*, training_failed: bool = False) -> None:
        attachment_modes.append(training_failed)

    trainer._stop_attachments = stop_attachments  # type: ignore[method-assign]
    _replace_stages(trainer, training_stage)

    await trainer.train(handle_signals=False)
    assert backend.finalize_calls == 1
    assert attachment_modes == [False]
    assert not _pipeline_tasks()


@pytest.mark.asyncio
async def test_failed_pipeline_closes_backend_owned_publication_and_joins_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer_module, "_PIPELINE_SHUTDOWN_TIMEOUT_SECONDS", 0.08)
    publication_started = asyncio.Event()
    publication_closed = asyncio.Event()
    publication: asyncio.Task[dict[str, float]] | None = None

    async def finalize() -> dict[str, float]:
        assert publication is not None
        publication.cancel()
        await asyncio.gather(publication, return_exceptions=True)
        return {}

    backend = _Backend(finalize)
    trainer = _trainer(tmp_path, backend)

    async def owned_publication() -> dict[str, float]:
        publication_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            publication_closed.set()

    async def fail_stage() -> None:
        nonlocal publication
        publication = asyncio.create_task(
            owned_publication(), name="backend_owned_publication"
        )
        result = SimpleNamespace(
            step=1,
            checkpoint_id="checkpoint-1",
            checkpoint_ready=asyncio.shield(publication),
            publication_metrics_ready=asyncio.shield(publication),
        )
        trainer._backend_training_started = True
        await trainer._log_checkpoint_saved(result)
        trainer._schedule_publication_metrics(result)
        await publication_started.wait()
        raise RuntimeError("training failed")

    _replace_stages(trainer, fail_stage)
    with pytest.raises(RuntimeError, match="training failed"):
        await trainer.train(handle_signals=False)

    assert backend.finalize_calls == 1
    assert publication_closed.is_set()
    assert publication is not None and publication.done()
    assert not trainer._publication_metric_tasks
    assert not trainer._checkpoint_log_tasks
    assert not _pipeline_tasks()


@pytest.mark.asyncio
async def test_stubborn_stage_is_cancelled_and_joined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer_module, "_PIPELINE_SHUTDOWN_TIMEOUT_SECONDS", 0.08)
    trainer = _trainer(tmp_path, _Backend())
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def stubborn_stage() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    _replace_stages(trainer, stubborn_stage)
    operation = asyncio.create_task(trainer.train(handle_signals=False))
    await started.wait()
    trainer.request_stop()

    with pytest.raises(TimeoutError, match="stage shutdown cutoff"):
        await operation
    assert cancelled.is_set()
    assert not _pipeline_tasks()


@pytest.mark.asyncio
async def test_cleanup_cancellation_is_primary_and_joins_optional_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer_module, "_PIPELINE_SHUTDOWN_TIMEOUT_SECONDS", 0.08)
    trainer = _trainer(tmp_path, _Backend())
    cleanup_started = asyncio.Event()
    cleanup_cancelled: asyncio.Event | None = None
    release_gate = asyncio.Event()
    optional_cancelled = asyncio.Event()
    attachment_modes: list[bool] = []
    stage_failure = RuntimeError("training failed before cleanup cancellation")

    async def optional_work() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            optional_cancelled.set()

    async def training_stage() -> None:
        task = asyncio.create_task(optional_work(), name="post_train_cleanup_probe")
        trainer._post_train_tasks.add(task)
        raise stage_failure

    original_shutdown = trainer._shutdown_pipeline

    async def observed_shutdown(**kwargs):
        nonlocal cleanup_cancelled
        cleanup_cancelled = kwargs["cleanup_cancelled"]
        cleanup_started.set()
        return await original_shutdown(**kwargs)

    async def release_data() -> None:
        await release_gate.wait()

    async def stop_attachments(*, training_failed: bool = False) -> None:
        attachment_modes.append(training_failed)

    trainer._shutdown_pipeline = observed_shutdown  # type: ignore[method-assign]
    trainer._release_pipeline_resources = release_data  # type: ignore[method-assign]
    trainer._stop_attachments = stop_attachments  # type: ignore[method-assign]
    _replace_stages(trainer, training_stage)
    operation = asyncio.create_task(trainer.train(handle_signals=False))
    await cleanup_started.wait()
    operation.cancel()
    assert cleanup_cancelled is not None
    await cleanup_cancelled.wait()
    release_gate.set()

    with pytest.raises(BaseExceptionGroup) as caught:
        await operation
    leaves = trainer_module.unique_exception_leaves((caught.value,))
    assert isinstance(leaves[0], asyncio.CancelledError)
    assert leaves[1] is stage_failure
    assert optional_cancelled.is_set()
    assert attachment_modes == [True]
    assert not trainer._post_train_tasks
    assert not _pipeline_tasks()


@pytest.mark.asyncio
async def test_mandatory_releases_run_while_backend_owner_exhausts_optional_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer_module, "_PIPELINE_SHUTDOWN_TIMEOUT_SECONDS", 0.08)
    owner_started = asyncio.Event()
    owner_cancelled = asyncio.Event()
    data_released = asyncio.Event()
    leases_released = asyncio.Event()
    attachment_modes: list[bool] = []
    ordering: list[str] = []

    async def finalize() -> dict[str, float]:
        owner_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            owner_cancelled.set()

    trainer = _trainer(tmp_path, _Backend(finalize))

    async def training_stage() -> None:
        async def optional_work() -> None:
            ordering.append("optional_start")
            try:
                await asyncio.Event().wait()
            finally:
                ordering.append("optional_end")

        trainer._post_train_tasks.add(
            asyncio.create_task(optional_work(), name="post_train_drain_probe")
        )
        trainer._backend_training_started = True
        trainer.request_stop()

    async def release_data() -> None:
        ordering.append("data_release")
        data_released.set()

    async def release_leases() -> None:
        ordering.append("lease_release")
        leases_released.set()

    async def stop_attachments(*, training_failed: bool = False) -> None:
        attachment_modes.append(training_failed)

    trainer._release_pipeline_resources = release_data  # type: ignore[method-assign]
    trainer._release_all_scheduled_eval_leases = release_leases  # type: ignore[method-assign]
    trainer._stop_attachments = stop_attachments  # type: ignore[method-assign]
    _replace_stages(trainer, training_stage)

    with pytest.raises(BaseExceptionGroup) as caught:
        await trainer.train(handle_signals=False)
    leaves = trainer_module.unique_exception_leaves((caught.value,))
    assert any("post-training tasks" in str(error) for error in leaves)
    assert any("backend publication owner" in str(error) for error in leaves)
    assert owner_started.is_set() and owner_cancelled.is_set()
    assert data_released.is_set() and leases_released.is_set()
    assert ordering.index("data_release") < ordering.index("optional_end")
    assert ordering.index("lease_release") < ordering.index("optional_end")
    assert attachment_modes == [True]
    assert not _pipeline_tasks()


@pytest.mark.parametrize("readiness", ["publication", "checkpoint"])
@pytest.mark.asyncio
async def test_readiness_failure_is_primary_and_recursively_deduplicated(
    tmp_path: Path, readiness: str
) -> None:
    shared = RuntimeError(f"{readiness} failed")

    async def finalize() -> dict[str, float]:
        raise BaseExceptionGroup(
            "backend readiness failed",
            [BaseExceptionGroup("nested readiness failure", [shared, shared])],
        )

    backend = _Backend(finalize)
    trainer = _trainer(tmp_path, backend)
    attachment_modes: list[bool] = []

    async def failed_publication() -> dict[str, float]:
        raise shared

    async def failed_checkpoint() -> None:
        raise shared

    async def training_stage() -> None:
        trainer._backend_training_started = True
        if readiness == "publication":
            trainer._schedule_publication_metrics(
                SimpleNamespace(step=1, publication_metrics_ready=failed_publication())
            )
            tasks = tuple(trainer._publication_metric_tasks)
        else:
            await trainer._log_checkpoint_saved(
                SimpleNamespace(
                    step=1,
                    checkpoint_id=None,
                    checkpoint_ready=failed_checkpoint(),
                )
            )
            tasks = tuple(trainer._checkpoint_log_tasks)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_attachments(*, training_failed: bool = False) -> None:
        attachment_modes.append(training_failed)

    trainer._stop_attachments = stop_attachments  # type: ignore[method-assign]
    _replace_stages(trainer, training_stage)

    with pytest.raises(RuntimeError, match=f"{readiness} failed") as caught:
        await trainer.train(handle_signals=False)
    assert caught.value is shared
    assert backend.finalize_calls == 1
    assert attachment_modes == [True]
    assert not _pipeline_tasks()
