from concurrent.futures import Future
import sys
from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest

from art.megatron.runtime.executor import (
    GenerationResidency,
    MCoreRunSlotExecutor,
    MegatronTrainJobExecutor,
    _GenerationPublisher,
)
from art.megatron.runtime.residency import ResidencyKey
from art.megatron.runtime.specs import TrainerGeneration


class _Gradients:
    def __init__(self, contributions: tuple[str, ...], prepared: Any = None) -> None:
        self.contributions = contributions
        self.prepared = prepared
        self.sealed: tuple[str, ...] | None = None

    def seal(self, contributions: tuple[str, ...]) -> None:
        self.sealed = contributions

    def prepare_optimizer(self) -> Any:
        return self.prepared

    def consume(self) -> tuple[str, ...]:
        return self.contributions


class _OptimizerOnlyPublisher:
    def __init__(self) -> None:
        self.health_checks = 0

    def raise_if_failed(self) -> None:
        self.health_checks += 1

    def stage(self, **_kwargs: Any) -> None:
        raise AssertionError("optimizer staged sampler publication")

    def attach_resident_optimizer(self, **_kwargs: Any) -> None:
        raise AssertionError("optimizer attached publication state")


class _Residency:
    def __init__(self) -> None:
        self.acquired: list[tuple[ResidencyKey, ...]] = []
        self.advanced: list[tuple[ResidencyKey, ResidencyKey, bool]] = []
        self.retired: list[ResidencyKey] = []

    def acquire_l1_working_set(self, keys: Any) -> None:
        self.acquired.append(tuple(keys))

    def release_l1_working_set(self, _keys: Any) -> None:
        return None

    def wait_before_mutation_working_set(self, _keys: Any) -> None:
        return None

    def advance_l1(
        self,
        source: ResidencyKey,
        target: ResidencyKey,
        _tensors: Any,
        *,
        retire_source: bool,
    ) -> Future[Any]:
        self.advanced.append((source, target, retire_source))
        future: Future[Any] = Future()
        future.set_result(None)
        return future

    def retire_async(self, key: ResidencyKey) -> Future[None]:
        self.retired.append(key)
        future: Future[None] = Future()
        future.set_result(None)
        return future


class _SnapshotPublisher:
    def __init__(self) -> None:
        self.staged = False
        self.has_optimizer = False
        self.ensure_calls: list[dict[str, Any]] = []
        self.attach_calls: list[dict[str, Any]] = []
        self.prepare_calls: list[dict[str, Any]] = []

    def has_generation(
        self, _generation: TrainerGeneration, *, require_optimizer: bool = False
    ) -> bool:
        return self.staged and (not require_optimizer or self.has_optimizer)

    def ensure_generation(self, **kwargs: Any) -> dict[str, float]:
        self.ensure_calls.append(kwargs)
        self.staged = True
        return {"snapshot_launch_s": 1.0}

    def attach_resident_optimizer(self, **kwargs: Any) -> dict[str, float]:
        self.attach_calls.append(kwargs)
        self.has_optimizer = True
        return {"snapshot_optimizer_attach_s": 2.0}

    def prepare(self, **kwargs: Any) -> tuple[Any, dict[str, float]]:
        self.prepare_calls.append(kwargs)
        plan = SimpleNamespace(model_dump=lambda **_kwargs: {"rank": 0})
        return plan, {"snapshot_prepare_s": 3.0}


def _generation(step: int) -> TrainerGeneration:
    return TrainerGeneration(
        training_session_id="session",
        policy_step=step,
        generation_id=_generation_id(step),
        adapter_path=f"/adapter/{step}",
    )


def _generation_id(step: int) -> str:
    return f"step-{step:08d}-{step:032x}"


def _key(step: int, representation: str = "weights") -> ResidencyKey:
    return ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id=_generation_id(step),
        representation=representation,
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )


def _optimizer_job(step: int) -> SimpleNamespace:
    return SimpleNamespace(
        operation_id=f"optimizer-{step}",
        run_id="run",
        training_session_id="session",
        expected_learner_version=step - 1,
        learner_version=step,
        generation=_generation(step),
        contributing_forward_backward_operation_ids=(f"forward-{step}",),
        optimizer=SimpleNamespace(
            learning_rate=1e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
            grad_clip_norm=1.0,
        ),
    )


def test_static_optimizer_does_not_stage_without_save_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.train",
        SimpleNamespace(
            run_megatron_optimizer_step=lambda **_kwargs: SimpleNamespace(
                update_successful=True, grad_norm=1.0, num_zeros_in_grad=0
            )
        ),
    )
    publisher = _OptimizerOnlyPublisher()
    runtime = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{}], config=SimpleNamespace()),
        model_support_handler=object(),
        model=(),
        optimizer_snapshot_barrier=SimpleNamespace(wait_before_mutation=lambda: None),
        resident_training_session_id=None,
        resident_policy_step=0,
        resident_generation_id=None,
        optimizer_state_loaded=True,
        adapter_export_dtypes=None,
        adapter_export_config=None,
    )
    executor = MegatronTrainJobExecutor.__new__(MegatronTrainJobExecutor)
    executor.runtime = runtime
    executor._publisher = publisher
    executor._gradients = _Gradients(("forward-1",))
    executor._gradient_parent_version = 0
    executor._closed = False

    result = executor.execute_optimizer(_optimizer_job(1))

    assert publisher.health_checks == 1
    assert runtime.resident_generation_id == _generation_id(1)
    assert not any(key.startswith("snapshot_") for key in result["metrics"])


def test_run_slot_optimizer_advances_immutable_state_without_publication() -> None:
    publisher = _OptimizerOnlyPublisher()
    residency = _Residency()
    parent_weights = _key(0)
    parent_optimizer = _key(0, "optimizer")
    accumulator = parent_weights.model_copy(
        update={"representation": "accumulator", "accumulator_revision": 1}
    )
    gradients = _Gradients(("forward-1",), prepared=("gradient",))
    state = SimpleNamespace(
        training_session_id="session",
        learner_version=0,
        gradients=gradients,
        desired=GenerationResidency(
            weights=parent_weights,
            optimizer=parent_optimizer,
            accumulator=accumulator,
        ),
        installed_weights=parent_weights,
        installed_optimizer=parent_optimizer,
        pending_load=None,
        adapter_config={"r": 1},
        residency_revision=0,
        registration_complete=True,
        unregistering=False,
    )
    slot = SimpleNamespace(
        optim_step_reduced=lambda *_args, **_kwargs: {
            "update_successful": True,
            "grad_norm": 1.0,
            "num_zeros_in_grad": 0.0,
        },
        checkpoint_slot_residency_tensors=lambda _run_id: SimpleNamespace(
            weights=(object(),), optimizer=(object(),)
        ),
        checkpoint_slot_optimizer_residency_source=lambda _run_id: (
            _ for _ in ()
        ).throw(AssertionError("optimizer inspected publication state")),
    )
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    executor.runtime = SimpleNamespace(
        optimizer_snapshot_barrier=SimpleNamespace(
            wait_before_mutation=lambda *, key: None
        )
    )
    executor._slot_trainer = slot
    executor._residency = residency
    executor._publisher = publisher
    executor._runs = {"run": state}
    executor._closed = False

    result = executor.execute_optimizer(_optimizer_job(1))

    output_weights = _key(1)
    output_optimizer = _key(1, "optimizer")
    assert residency.advanced == [
        (parent_weights, output_weights, True),
        (parent_optimizer, output_optimizer, True),
    ]
    assert residency.retired == [accumulator]
    assert state.desired == GenerationResidency(
        weights=output_weights, optimizer=output_optimizer
    )
    assert publisher.health_checks == 1
    assert not any(key.startswith("snapshot_") for key in result["metrics"])


def _snapshot_executor(*, optimizer_source: Any) -> tuple[Any, ...]:
    weights = _key(1)
    optimizer = _key(1, "optimizer")
    state = SimpleNamespace(
        training_session_id="session",
        learner_version=1,
        desired=GenerationResidency(weights=weights, optimizer=optimizer),
        installed_weights=weights,
        installed_optimizer=optimizer,
        pending_load=None,
        adapter_config={"r": 1},
        registration_complete=True,
        unregistering=False,
    )
    publisher = _SnapshotPublisher()
    residency = _Residency()
    source_calls: list[str] = []

    def source(run_id: str) -> Any:
        source_calls.append(run_id)
        return optimizer_source

    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    executor.runtime = SimpleNamespace()
    executor._slot_trainer = SimpleNamespace(
        checkpoint_slot_optimizer_residency_source=source
    )
    executor._residency = residency
    executor._publisher = publisher
    executor._runs = {"run": state}
    executor._closed = False
    return executor, publisher, residency, weights, optimizer, source_calls


def _snapshot_job(*, save_optimizer: bool) -> SimpleNamespace:
    return SimpleNamespace(
        operation_id="save",
        run_id="run",
        training_session_id="session",
        learner_version=1,
        generation=_generation(1),
        optimizer_state_path="/optimizer",
        staging_adapter_path=None,
        existing_adapter=None,
        publication_targets=(),
        adapter_object_target=object(),
        save_optimizer=save_optimizer,
    )


def test_sampler_successor_stages_selected_weights_without_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.lora",
        SimpleNamespace(LoRASlotRef=lambda kind, name: (kind, name)),
    )
    executor, publisher, residency, weights, _optimizer, source_calls = (
        _snapshot_executor(optimizer_source=object())
    )

    result = executor.execute_snapshot(_snapshot_job(save_optimizer=False), object())

    assert residency.acquired == [(weights,)]
    assert len(publisher.ensure_calls) == 1
    assert publisher.ensure_calls[0]["residency_key"] == weights
    assert publisher.ensure_calls[0]["snapshot_optimizer"] is False
    assert publisher.attach_calls == []
    assert source_calls == []
    assert result["metrics"] == {
        "snapshot_launch_s": 1.0,
        "snapshot_prepare_s": 3.0,
    }


def test_state_successor_attaches_existing_optimizer_l2_only_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.lora",
        SimpleNamespace(LoRASlotRef=lambda kind, name: (kind, name)),
    )
    optimizer_source = object()
    executor, publisher, residency, weights, optimizer, source_calls = (
        _snapshot_executor(optimizer_source=optimizer_source)
    )

    result = executor.execute_snapshot(_snapshot_job(save_optimizer=True), object())

    assert residency.acquired == [(weights,)]
    assert publisher.ensure_calls[0]["snapshot_optimizer"] is False
    assert publisher.attach_calls == [
        {
            "generation": _generation(1),
            "source": optimizer_source,
            "residency_key": optimizer,
        }
    ]
    assert source_calls == ["run"]
    assert result["metrics"] == {
        "snapshot_launch_s": 1.0,
        "snapshot_optimizer_attach_s": 2.0,
        "snapshot_prepare_s": 3.0,
    }


def test_lazy_generation_forwards_exact_residency_identity() -> None:
    publisher = _GenerationPublisher.__new__(_GenerationPublisher)
    publisher._lock = Lock()
    publisher._cache = {}
    calls: list[dict[str, Any]] = []
    publisher.stage = lambda **kwargs: calls.append(kwargs) or {"staged": 1.0}
    weights = _key(1)

    metrics = publisher.ensure_generation(
        run_id="run",
        generation=_generation(1),
        adapter_dtypes={},
        adapter_config={"r": 1},
        snapshot_optimizer=False,
        residency_key=weights,
    )

    assert metrics == {"staged": 1.0}
    assert calls[0]["residency_key"] == weights
