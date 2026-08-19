import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from art import TrainableModel
from art.pipeline_trainer import PipelineRuntimeConfig, PipelineTrainer
from art.pipeline_trainer import trainer as trainer_module


def _leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in _leaves(child)]
    return [error]


@pytest.mark.asyncio
async def test_failed_pipeline_bounds_stuck_publication_and_reaches_backend_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trainer_module, "_PIPELINE_SHUTDOWN_TIMEOUT_SECONDS", 0.02)

    class Backend:
        def __init__(self) -> None:
            self.closed = asyncio.Event()

        async def register(self, _model) -> None:
            return None

        async def _get_step(self, _model) -> int:
            return 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_error) -> None:
            self.closed.set()

    backend = Backend()
    model = TrainableModel(
        name="publication-shutdown",
        run_name="publication-shutdown",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    model._backend = backend  # type: ignore[assignment]

    async def rollout(*_args):
        raise AssertionError("rollout stage was replaced")

    trainer = PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=rollout,
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
    publication_started = asyncio.Event()
    readiness_cancelled = asyncio.Event()

    async def stuck_publication() -> dict[str, float]:
        publication_started.set()
        while not backend.closed.is_set():
            try:
                await backend.closed.wait()
            except asyncio.CancelledError:
                continue
        return {}

    async def checkpoint_readiness() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            readiness_cancelled.set()

    async def fail_stage() -> None:
        async def failed_publication() -> dict[str, float]:
            raise RuntimeError("publication failed")

        trainer._schedule_publication_metrics(
            SimpleNamespace(step=1, publication_metrics_ready=failed_publication())
        )
        failed_task = next(iter(trainer._publication_metric_tasks))
        await asyncio.gather(failed_task, return_exceptions=True)
        await asyncio.sleep(0)

        result = SimpleNamespace(
            step=2,
            checkpoint_id=None,
            checkpoint_ready=checkpoint_readiness(),
            publication_metrics_ready=stuck_publication(),
        )
        await trainer._log_checkpoint_saved(result)
        trainer._schedule_publication_metrics(result)
        await publication_started.wait()
        raise RuntimeError("training failed")

    async def wait_stage() -> None:
        await asyncio.Event().wait()

    trainer._rollout_stage = wait_stage  # type: ignore[method-assign]
    trainer._packing_stage = wait_stage  # type: ignore[method-assign]
    trainer._training_stage = fail_stage  # type: ignore[method-assign]
    trainer._eval_stage = wait_stage  # type: ignore[method-assign]
    trainer._status_loop = wait_stage  # type: ignore[method-assign]

    async def run() -> None:
        async with backend:
            await trainer.train(handle_signals=False)

    operation = asyncio.create_task(run())
    try:
        await asyncio.wait_for(backend.closed.wait(), timeout=0.25)
    except BaseException:
        backend.closed.set()
        await asyncio.gather(operation, return_exceptions=True)
        raise

    with pytest.raises(BaseExceptionGroup) as caught:
        await operation
    leaves = _leaves(caught.value)
    assert any(str(error) == "training failed" for error in leaves)
    assert any(str(error) == "publication failed" for error in leaves)
    assert any(
        isinstance(error, TimeoutError)
        and "publication metric/readiness tasks" in str(error)
        for error in leaves
    )
    assert readiness_cancelled.is_set()

    for _ in range(10):
        if not trainer._publication_metric_tasks:
            break
        await asyncio.sleep(0)
    assert not trainer._publication_metric_tasks
    assert not trainer._checkpoint_log_tasks
