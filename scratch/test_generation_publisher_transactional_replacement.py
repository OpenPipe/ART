from collections.abc import Callable
from concurrent.futures import Future
from threading import Event
import time
from types import SimpleNamespace

import pytest
import torch

from art.megatron.runtime.executor import _GenerationPublisher
from art.megatron.runtime.residency import (
    ResidencyCapacityUnavailable,
    ResidencyKey,
)
from art.megatron.runtime.specs import TrainerGeneration
from art.megatron.weights.lora_publish import LoraSnapshot


class _Pending:
    fences = ()

    def __init__(self, ready: Event, value: object, error: BaseException | None = None):
        self.ready = ready
        self.value = value
        self.error = error

    def resolve(self) -> object:
        assert self.ready.wait(2)
        if self.error is not None:
            raise self.error
        return self.value


class _Barrier:
    def register(self, _snapshot: object, *, key: str) -> None:
        assert key == "run"


class _Residency:
    def __init__(self) -> None:
        self.config = SimpleNamespace(shutdown_timeout_s=1.0)
        self.registered: list[ResidencyKey] = []
        self.retired: list[ResidencyKey] = []

    def register_l2(
        self, key: ResidencyKey, _tensors: tuple[torch.Tensor, ...]
    ) -> None:
        self.registered.append(key)

    def retire_async(self, key: ResidencyKey) -> Future[None]:
        self.retired.append(key)
        retired: Future[None] = Future()
        retired.set_result(None)
        return retired


def _generation(step: int) -> TrainerGeneration:
    return TrainerGeneration(
        training_session_id="session",
        policy_step=step,
        generation_id=f"step-{step:08d}-{step:032x}",
        adapter_path=f"/tmp/step-{step}",
    )


def _residency_key(generation: TrainerGeneration) -> ResidencyKey:
    return ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id=generation.generation_id,
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        rank=0,
        world_size=1,
        model=object(),
        model_support_handler=object(),
        optimizer_snapshot_barrier=_Barrier(),
    )


def _snapshot(value: float) -> LoraSnapshot:
    return LoraSnapshot(
        tensors={"lora": torch.tensor([value])},
        adapter_config={"r": 1},
    )


def _stage(
    publisher: _GenerationPublisher,
    generation: TrainerGeneration,
    *,
    resident: bool = False,
) -> None:
    publisher.stage(
        run_id="run",
        generation=generation,
        adapter_dtypes={},
        adapter_config={"r": 1},
        snapshot_optimizer=False,
        residency_key=_residency_key(generation) if resident else None,
    )


def _wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 2
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert predicate()


def test_synchronous_stage_failure_preserves_latest(monkeypatch) -> None:
    ready = Event()
    ready.set()
    stages: list[_Pending | BaseException] = [
        _Pending(ready, _snapshot(1)),
        RuntimeError("stage failed"),
    ]

    def stage_lora(**_kwargs: object) -> _Pending:
        staged = stages.pop(0)
        if isinstance(staged, BaseException):
            raise staged
        return staged

    monkeypatch.setattr(
        "art.megatron.weights.lora_publish.stage_vllm_lora_snapshot_from_model",
        stage_lora,
    )
    publisher = _GenerationPublisher(_runtime(), capacity=2)
    previous = _generation(1)
    try:
        _stage(publisher, previous)
        _wait_until(
            lambda: publisher._latest_by_run.get("run") == previous.generation_id
        )

        with pytest.raises(RuntimeError, match="stage failed"):
            _stage(publisher, _generation(2))

        assert publisher.has_generation(previous)
        assert publisher._latest_by_run["run"] == previous.generation_id
    finally:
        publisher.close()


def test_asynchronous_stage_failure_preserves_latest(monkeypatch) -> None:
    previous_ready = Event()
    replacement_ready = Event()
    previous_ready.set()
    stages = iter(
        (
            _Pending(previous_ready, _snapshot(1)),
            _Pending(replacement_ready, _snapshot(2), RuntimeError("resolve failed")),
        )
    )
    monkeypatch.setattr(
        "art.megatron.weights.lora_publish.stage_vllm_lora_snapshot_from_model",
        lambda **_kwargs: next(stages),
    )
    publisher = _GenerationPublisher(_runtime(), capacity=2)
    previous = _generation(1)
    replacement = _generation(2)
    _stage(publisher, previous)
    _wait_until(lambda: publisher._latest_by_run.get("run") == previous.generation_id)
    _stage(publisher, replacement)

    replacement_ready.set()
    _wait_until(lambda: replacement.generation_id not in publisher._cache)
    assert publisher.has_generation(previous)
    assert publisher._latest_by_run["run"] == previous.generation_id
    with pytest.raises(BaseExceptionGroup) as raised:
        publisher.close()
    assert any(str(error) == "resolve failed" for error in raised.value.exceptions)


@pytest.mark.parametrize("resident", [False, True])
def test_out_of_order_replacements_retire_predecessor_chain(
    monkeypatch, resident: bool
) -> None:
    ready = [Event() for _ in range(3)]
    ready[0].set()
    stages = iter(_Pending(event, _snapshot(step)) for step, event in enumerate(ready))
    monkeypatch.setattr(
        "art.megatron.weights.lora_publish.stage_vllm_lora_snapshot_from_model",
        lambda **_kwargs: next(stages),
    )
    residency = _Residency() if resident else None
    publisher = _GenerationPublisher(_runtime(), capacity=3, residency=residency)
    generations = tuple(_generation(step) for step in range(1, 4))
    try:
        _stage(publisher, generations[0], resident=resident)
        _wait_until(
            lambda: publisher._latest_by_run.get("run") == generations[0].generation_id
        )
        _stage(publisher, generations[1], resident=resident)
        _stage(publisher, generations[2], resident=resident)
        assert publisher._latest_by_run["run"] == generations[0].generation_id

        ready[2].set()
        _wait_until(
            lambda: publisher._latest_by_run.get("run") == generations[2].generation_id
        )
        ready[1].set()
        _wait_until(lambda: generations[1].generation_id not in publisher._cache)

        assert set(publisher._cache) == {generations[2].generation_id}
        assert publisher.has_generation(generations[2])
        if residency is not None:
            assert {key.generation_id for key in residency.retired} == {
                generations[0].generation_id,
                generations[1].generation_id,
            }
    finally:
        publisher.close()


def test_capacity_cannot_evict_same_run_latest_for_replacement(monkeypatch) -> None:
    ready = Event()
    ready.set()
    stage_calls = 0

    def stage_lora(**_kwargs: object) -> _Pending:
        nonlocal stage_calls
        stage_calls += 1
        return _Pending(ready, _snapshot(stage_calls))

    monkeypatch.setattr(
        "art.megatron.weights.lora_publish.stage_vllm_lora_snapshot_from_model",
        stage_lora,
    )
    publisher = _GenerationPublisher(_runtime(), capacity=1)
    previous = _generation(1)
    try:
        _stage(publisher, previous)
        _wait_until(
            lambda: publisher._latest_by_run.get("run") == previous.generation_id
        )

        with pytest.raises(ResidencyCapacityUnavailable):
            _stage(publisher, _generation(2))

        assert stage_calls == 1
        assert publisher.has_generation(previous)
        assert publisher._latest_by_run["run"] == previous.generation_id
    finally:
        publisher.close()
