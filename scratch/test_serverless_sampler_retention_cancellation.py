from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from art import TrainableModel
from art.pipeline_trainer.checkpoint_retention import CheckpointRetentionPlan
from art.serverless.backend import ServerlessBackend
from art.training.contracts import CheckpointRef, SamplerWeightsResult


class _Sampler:
    async def publish(self, *args, **kwargs):
        raise AssertionError("sampler publication is not expected")

    async def remove(self, *args, **kwargs) -> None:
        raise AssertionError("sampler removal is not expected")

    async def close(self) -> None:
        pass


class _Client:
    def __init__(self) -> None:
        self.run_id = "run"
        self.shutdown_calls = 0

    async def shutdown(self) -> None:
        self.shutdown_calls += 1

    async def abort_result_waiters(self) -> None:
        pass


class _Service:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _RetentionService(_Service):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.apply_started = asyncio.Event()
        self.apply_gate = asyncio.Event()

    async def iter_checkpoint_pages(self, run_id: str):
        assert run_id == "run"
        yield SimpleNamespace(
            checkpoints=(
                SimpleNamespace(
                    learner_version=1,
                    checkpoint_id="step-1",
                    revision=1,
                    state="ready",
                ),
            ),
            current_checkpoint_id=None,
            protected_checkpoint_ids=(),
        )

    async def apply_checkpoint_retention(self, run_id: str, request: Any) -> None:
        assert run_id == "run"
        assert request.retain_checkpoint_ids == ()
        self.apply_started.set()
        await self.apply_gate.wait()
        if self.fail:
            raise RuntimeError("retention RPC failed")


def _backend() -> tuple[ServerlessBackend, TrainableModel, _Client, _Service]:
    backend = ServerlessBackend(
        training_base_url="http://training.invalid/v1",
        inference_base_url="http://inference.invalid/v1",
        sampler_manager=_Sampler(),
        api_key="test",
        enable_expert_replay=False,
        close_timeout_s=1,
    )
    model = TrainableModel(
        name="model",
        run_name="run",
        project="scratch",
        base_model="Qwen/Qwen3.5-35B-A3B",
    )
    client = _Client()
    service = _Service()
    backend._clients[backend._model_key(model)] = cast(Any, client)
    backend._service = cast(Any, service)
    return backend, model, client, service


def _weights(step: int, *, generation: str) -> SamplerWeightsResult:
    return SamplerWeightsResult(
        operation_id=f"sampler-{generation}",
        checkpoint=CheckpointRef(
            run_id="run",
            learner_version=step,
            checkpoint_id=f"step-{step}",
        ),
        lora="lora",
        training_session_id="session",
        generation_id=generation,
        lora_bytes=1,
    )


async def _wait_until(predicate) -> None:
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancelled_begin_cannot_strand_chained_retention() -> None:
    backend, model, client, service = _backend()
    model_key = backend._model_key(model)
    first = await backend._begin_sampler_retention(model)

    second_begin = asyncio.create_task(backend._begin_sampler_retention(model))
    await _wait_until(lambda: backend._sampler_retention_tails[model_key] is not first)
    second = backend._sampler_retention_tails[model_key]
    second_begin.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_begin

    assert second.finish_requested.done()
    assert not second.settled.done()
    await backend._finish_sampler_retention(first, set())
    await asyncio.wait_for(second.settled, timeout=1)

    third = await backend._begin_sampler_retention(model)
    await backend._finish_sampler_retention(third, set())
    await backend.close()
    assert client.shutdown_calls == 1
    assert service.close_calls == 1
    assert model_key not in backend._sampler_retention_tails


@pytest.mark.asyncio
async def test_cancelled_finish_cannot_interrupt_locked_settlement() -> None:
    backend, model, _, _ = _backend()
    retention = await backend._begin_sampler_retention(model)
    key = backend._sampler_key(model, 1)
    backend._sampler_results[key] = _weights(1, generation="old")
    await backend._reserve_sampler_forgetting(
        retention, observed_steps={1}, retain_steps=set()
    )

    await backend._sampler_state_lock.acquire()
    finish = asyncio.create_task(backend._finish_sampler_retention(retention, {key}))
    await _wait_until(lambda: retention.finish_requested.done())
    finish.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finish
    assert not retention.settled.done()

    backend._sampler_state_lock.release()
    await backend.close()
    assert retention.settled.done()
    assert key not in backend._sampler_results
    assert key not in backend._sampler_retention_reservations


@pytest.mark.asyncio
async def test_cancelled_retention_rpc_releases_reservation() -> None:
    backend, model, _, _ = _backend()
    service = _RetentionService()
    backend._service = cast(Any, service)
    key = backend._sampler_key(model, 1)
    result = _weights(1, generation="old")
    backend._sampler_results[key] = result

    apply = asyncio.create_task(
        backend._apply_checkpoint_retention(
            model, CheckpointRetentionPlan(observed_steps={1})
        )
    )
    await asyncio.wait_for(service.apply_started.wait(), timeout=1)
    assert key in backend._sampler_retention_reservations
    apply.cancel()
    with pytest.raises(asyncio.CancelledError):
        await apply

    assert key not in backend._sampler_retention_reservations
    assert backend._sampler_results[key] is result
    successor = await backend._begin_sampler_retention(model)
    await backend._finish_sampler_retention(successor, set())
    await backend.close()


@pytest.mark.asyncio
async def test_failed_retention_rpc_releases_reservation() -> None:
    backend, model, _, _ = _backend()
    service = _RetentionService(fail=True)
    service.apply_gate.set()
    backend._service = cast(Any, service)
    key = backend._sampler_key(model, 1)
    result = _weights(1, generation="old")
    backend._sampler_results[key] = result

    with pytest.raises(RuntimeError, match="retention RPC failed"):
        await backend._apply_checkpoint_retention(
            model, CheckpointRetentionPlan(observed_steps={1})
        )

    assert key not in backend._sampler_retention_reservations
    assert backend._sampler_results[key] is result
    await backend.close()


@pytest.mark.asyncio
async def test_retention_failure_releases_reservations_and_is_reported_on_close() -> (
    None
):
    backend, model, _, service = _backend()
    model_key = backend._model_key(model)
    retention = await backend._begin_sampler_retention(model)
    key = backend._sampler_key(model, 1)
    original = _weights(1, generation="old")
    replacement = _weights(1, generation="new")
    backend._sampler_results[key] = original
    await backend._reserve_sampler_forgetting(
        retention, observed_steps={1}, retain_steps=set()
    )
    backend._sampler_results[key] = replacement

    with pytest.raises(BaseExceptionGroup, match="settlement failed"):
        await backend._finish_sampler_retention(retention, {key})
    await asyncio.sleep(0)
    assert retention.settled.done()
    assert backend._sampler_results[key] is replacement
    assert key not in backend._sampler_retention_reservations
    assert model_key not in backend._sampler_retention_tails

    successor = await backend._begin_sampler_retention(model)
    await backend._finish_sampler_retention(successor, set())
    with pytest.raises(BaseExceptionGroup, match="shutdown failed"):
        await backend.close()
    assert service.close_calls == 1
