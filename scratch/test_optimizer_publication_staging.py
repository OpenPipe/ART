from concurrent.futures import Future
from contextlib import contextmanager
import sys
from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest

from art.megatron.runtime.executor import (
    GenerationResidency,
    MCoreRunSlotExecutor,
    _GenerationPublisher,
    _ResidentRunState,
)
from art.megatron.runtime.residency import ResidencyKey
from art.megatron.runtime.specs import TrainerGeneration


def _generation_id(step: int) -> str:
    return f"step-{step:08d}-{step:032x}"


def _generation(step: int) -> TrainerGeneration:
    return TrainerGeneration(
        training_session_id="session",
        policy_step=step,
        generation_id=_generation_id(step),
        adapter_path=f"/adapter/{step}",
    )


def _key(step: int, representation: str = "weights") -> ResidencyKey:
    return ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id=_generation_id(step),
        representation=representation,
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )


def _resolved(value: Any = None) -> Future[Any]:
    future: Future[Any] = Future()
    future.set_result(value)
    return future


class _Gradients:
    def __init__(self) -> None:
        self.contributions = ("forward-1",)
        self.sealed: tuple[str, ...] | None = None

    def seal(self, contributions: tuple[str, ...]) -> None:
        self.sealed = contributions

    def prepare_optimizer(self) -> Any:
        return SimpleNamespace(
            expected_global_token_count=1,
            local_token_count=1,
            gradients=(),
            reduction="sum",
        )

    def consume(self) -> tuple[str, ...]:
        return self.contributions


class _CommitResidency:
    def __init__(self) -> None:
        self.acquired: list[tuple[ResidencyKey, ...]] = []
        self.advanced: list[tuple[ResidencyKey, ResidencyKey, bool]] = []
        self.retired: list[ResidencyKey] = []

    def acquire_l1_working_set(self, keys: Any) -> None:
        self.acquired.append(tuple(keys))

    acquire_prepared_l1_working_set = acquire_l1_working_set

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
        return _resolved()

    def retire_async(self, key: ResidencyKey) -> Future[Any]:
        self.retired.append(key)
        return _resolved()


class _SlotTrainer:
    def __init__(self) -> None:
        self.optimizer_source = object()
        self.optimizer_source_calls = 0
        self.stepped = False

    def reduce_checkpoint_slot_gradient_sums(
        self, _run_id: str, gradients: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return gradients

    def optim_step_reduced(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        self.stepped = True
        return {
            "update_successful": True,
            "grad_norm": 1.0,
            "num_zeros_in_grad": 0.0,
        }

    def checkpoint_slot_residency_tensors(self, _run_id: str) -> Any:
        return SimpleNamespace(weights=(object(),), optimizer=(object(),))

    def checkpoint_slot_optimizer_residency_source(self, _run_id: str) -> object:
        assert self.stepped
        self.optimizer_source_calls += 1
        return self.optimizer_source


class _CommittedPublisher:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.events: list[str] = []
        self.archives: list[dict[str, Any]] = []

    def raise_if_failed(self) -> None:
        return None

    def register_resident_generation(self, **kwargs: Any) -> dict[str, float]:
        self.events.append("register_resident")
        generation = kwargs["generation"]
        self.entries[generation.generation_id] = {
            "generation": generation,
            "weights": kwargs["weights_key"],
            "optimizer": kwargs["optimizer_key"],
            "optimizer_source": kwargs["optimizer_source"],
        }
        return {"snapshot_resident_attach_s": 1.0}

    def has_generation(
        self, generation: TrainerGeneration, *, require_optimizer: bool = False
    ) -> bool:
        entry = self.entries.get(generation.generation_id)
        return bool(
            entry is not None
            and entry["generation"] == generation
            and (not require_optimizer or entry["optimizer"] is not None)
        )

    def prepare(self, **kwargs: Any) -> tuple[Any, dict[str, float]]:
        self.events.append("prepare")
        entry = self.entries[kwargs["generation"].generation_id]
        assert not kwargs["save_optimizer"] or entry["optimizer"] is not None
        self.archives.append(entry.copy())
        plan = SimpleNamespace(model_dump=lambda **_kwargs: {"rank": 0})
        return plan, {"snapshot_prepare_s": 3.0}


def _optimizer_job() -> SimpleNamespace:
    return SimpleNamespace(
        operation_id="optimizer-1",
        run_id="run",
        training_session_id="session",
        expected_learner_version=0,
        learner_version=1,
        generation=_generation(1),
        contributing_forward_backward_operation_ids=("forward-1",),
        optimizer=SimpleNamespace(
            learning_rate=1e-4,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.1,
            grad_clip_norm=1.0,
        ),
    )


def _snapshot_job(*, save_optimizer: bool = True) -> SimpleNamespace:
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


def _state(
    *,
    learner_version: int,
    desired: GenerationResidency,
    installed: bool = True,
    gradients: Any | None = None,
) -> _ResidentRunState:
    return _ResidentRunState(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=learner_version,
        adapter_config={"r": 1},
        gradients=gradients,
        desired=desired,
        installed_weights=desired.weights if installed else None,
        installed_optimizer=desired.optimizer if installed else None,
        registration_complete=True,
        lora_export_plan="export-plan",
    )


def test_post_optimizer_save_uses_registered_generation_without_l1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_data_parallel_group",
        lambda **_kwargs: object(),
    )
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.training.finalize_grads",
        SimpleNamespace(
            reduce_accumulated_token_count=lambda *_args, **_kwargs: 1,
            finalize_model_grads_extended=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.lora",
        SimpleNamespace(LoRASlotRef=lambda kind, name: (kind, name)),
    )
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.weights.lora_publish",
        SimpleNamespace(
            build_local_lora_export_plan=lambda *_args, **_kwargs: object()
        ),
    )
    parent_weights = _key(0)
    parent_optimizer = _key(0, "optimizer")
    accumulator = parent_weights.model_copy(
        update={"representation": "accumulator", "accumulator_revision": 1}
    )
    state = _state(
        learner_version=0,
        desired=GenerationResidency(
            weights=parent_weights,
            optimizer=parent_optimizer,
            accumulator=accumulator,
        ),
        gradients=_Gradients(),
    )
    residency = _CommitResidency()
    slot = _SlotTrainer()
    publisher = _CommittedPublisher()
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    executor.runtime = SimpleNamespace(
        model=(),
        optimizer_snapshot_barrier=SimpleNamespace(
            wait_before_mutation=lambda *, key: None
        ),
        model_support_handler=SimpleNamespace(expert_packed_lora_groups=lambda: ()),
        inter_forward_backward_timing=SimpleNamespace(previous_job_complete_s=None),
    )
    executor._slot_trainer = slot
    executor._residency = residency
    executor._publisher = publisher
    executor._runs = {"run": state}
    executor._residency_admission_lock = Lock()
    executor._residency_admissions = {
        "optimizer-1": (parent_weights, parent_optimizer, accumulator)
    }
    executor._closing = False
    executor._closed = False

    optimizer_result = executor.execute_optimizer(_optimizer_job())
    optimizer_acquisitions = tuple(residency.acquired)

    def forbid_resident(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("save reacquired the trainer working set")

    executor._resident = forbid_resident
    save_result = executor.execute_snapshot(_snapshot_job(), object())

    output_weights = _key(1)
    output_optimizer = _key(1, "optimizer")
    assert residency.advanced == [
        (parent_weights, output_weights, True),
        (parent_optimizer, output_optimizer, True),
    ]
    assert tuple(residency.acquired) == optimizer_acquisitions
    assert slot.optimizer_source_calls == 1
    assert publisher.events == ["register_resident", "prepare"]
    assert publisher.archives == [
        {
            "generation": _generation(1),
            "weights": output_weights,
            "optimizer": output_optimizer,
            "optimizer_source": slot.optimizer_source,
        }
    ]
    assert optimizer_result["metrics"]["snapshot_resident_attach_s"] == 1.0
    assert save_result["metrics"] == {"snapshot_prepare_s": 3.0}


class _Lora:
    def __init__(self, tensors: dict[str, Any]) -> None:
        self.tensors = tensors

    def model_copy(self, *, update: dict[str, Any]) -> "_Lora":
        return _Lora(update.get("tensors", self.tensors))


class _OptimizerSource:
    def __init__(self) -> None:
        self.bound: tuple[Any, ...] | None = None

    def bind(self, tensors: tuple[Any, ...]) -> Any:
        self.bound = tensors
        return SimpleNamespace(source=self, tensors=tensors)


class _LowerTierResidency:
    def __init__(
        self,
        origins: dict[ResidencyKey, str],
        tensors: dict[ResidencyKey, tuple[Any, ...]],
    ) -> None:
        self.origins = origins
        self.tensor_images = tensors
        self.borrowed: list[tuple[ResidencyKey, str]] = []

    @contextmanager
    def borrow_l2(self, key: ResidencyKey) -> Any:
        self.borrowed.append((key, self.origins[key]))
        yield SimpleNamespace(tensors=lambda: self.tensor_images[key])


class _LowerTierPublisher:
    def __init__(
        self,
        generation: TrainerGeneration,
        sampler_key: ResidencyKey,
        optimizer_key: ResidencyKey,
        residency: _LowerTierResidency,
        optimizer_source: _OptimizerSource,
    ) -> None:
        resolved = _resolved(
            SimpleNamespace(
                lora=_Lora({"adapter.weight": object()}),
                optimizer=None,
                prepared_tensors=None,
                lora_residency_key=sampler_key,
            )
        )
        self.entry = SimpleNamespace(
            generation=generation,
            resolved=resolved,
            optimizer_upgrade=None,
            resident_lora=None,
            resident_optimizer=SimpleNamespace(
                key=optimizer_key,
                source=optimizer_source,
            ),
        )
        self.delegate = _GenerationPublisher.__new__(_GenerationPublisher)
        self.delegate.runtime = SimpleNamespace(rank=0)
        self.delegate._residency = residency
        self.delegate._prepare_lora_tensors = lambda lora: tuple(lora.tensors.values())
        self.archive: tuple[Any, Any] | None = None

    def has_generation(
        self, generation: TrainerGeneration, *, require_optimizer: bool = False
    ) -> bool:
        return generation == self.entry.generation

    def prepare(self, **kwargs: Any) -> tuple[Any, dict[str, float]]:
        with self.delegate._lora_snapshot(self.entry, None) as (_lora, weights):
            with self.delegate._optimizer_snapshot(
                self.entry,
                kwargs["generation"],
                required=kwargs["save_optimizer"],
            ) as optimizer:
                self.archive = weights, optimizer
        plan = SimpleNamespace(model_dump=lambda **_kwargs: {"rank": 0})
        return plan, {"snapshot_prepare_s": 1.0}


def test_nonresident_committed_save_borrows_l2_and_restores_l3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _key(1)
    sampler = weights.model_copy(update={"representation": "sampler"})
    optimizer = _key(1, "optimizer")
    weight_tensors = (object(),)
    optimizer_tensors = (object(), object())
    residency = _LowerTierResidency(
        origins={sampler: "l2_cpu", optimizer: "l3_nvme"},
        tensors={sampler: weight_tensors, optimizer: optimizer_tensors},
    )
    optimizer_source = _OptimizerSource()
    publisher = _LowerTierPublisher(
        _generation(1), sampler, optimizer, residency, optimizer_source
    )
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.optimizer_state",
        SimpleNamespace(
            trainer_rank_optimizer_snapshot_from_cpu=lambda _runtime, state, **kwargs: (
                state,
                kwargs["generation_id"],
                kwargs["step"],
            )
        ),
    )
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    executor.runtime = SimpleNamespace()
    executor._slot_trainer = SimpleNamespace()
    executor._residency = residency
    executor._publisher = publisher
    executor._runs = {
        "run": _state(
            learner_version=1,
            desired=GenerationResidency(weights=weights, optimizer=optimizer),
            installed=False,
        )
    }
    executor._closing = False
    executor._closed = False
    executor._resident = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("save reacquired the trainer working set")
    )

    result = executor.execute_snapshot(_snapshot_job(), object())

    assert residency.borrowed == [(sampler, "l2_cpu"), (optimizer, "l3_nvme")]
    assert optimizer_source.bound == optimizer_tensors
    assert publisher.archive is not None
    assert publisher.archive[0] == weight_tensors
    assert publisher.archive[1][1:] == (_generation_id(1), 1)
    assert result["metrics"] == {"snapshot_prepare_s": 1.0}


class _IncompletePublisher:
    def __init__(self, *, has_weights: bool, has_optimizer: bool) -> None:
        self.has_weights = has_weights
        self.has_optimizer = has_optimizer
        self.prepare_calls = 0

    def has_generation(
        self, _generation: TrainerGeneration, *, require_optimizer: bool = False
    ) -> bool:
        return self.has_optimizer if require_optimizer else self.has_weights

    def prepare(self, **_kwargs: Any) -> Any:
        self.prepare_calls += 1
        raise AssertionError("incomplete generation reached snapshot preparation")


@pytest.mark.parametrize(
    ("has_weights", "has_optimizer", "save_optimizer", "message"),
    (
        (False, False, False, "no immutable weights snapshot"),
        (True, False, True, "no immutable optimizer snapshot"),
    ),
)
def test_save_fails_closed_for_missing_or_incomplete_generation(
    has_weights: bool,
    has_optimizer: bool,
    save_optimizer: bool,
    message: str,
) -> None:
    weights = _key(1)
    optimizer = _key(1, "optimizer")
    publisher = _IncompletePublisher(
        has_weights=has_weights, has_optimizer=has_optimizer
    )
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    executor.runtime = SimpleNamespace()
    executor._slot_trainer = SimpleNamespace(
        checkpoint_slot_optimizer_residency_source=lambda _run_id: (
            _ for _ in ()
        ).throw(AssertionError("save inspected mutable optimizer state"))
    )
    executor._residency = SimpleNamespace(
        acquire_l1_working_set=lambda _keys: (_ for _ in ()).throw(
            AssertionError("save acquired L1")
        )
    )
    executor._publisher = publisher
    executor._runs = {
        "run": _state(
            learner_version=1,
            desired=GenerationResidency(weights=weights, optimizer=optimizer),
        )
    }
    executor._closing = False
    executor._closed = False

    with pytest.raises(RuntimeError, match=message):
        executor.execute_snapshot(
            _snapshot_job(save_optimizer=save_optimizer), object()
        )

    assert publisher.prepare_calls == 0
