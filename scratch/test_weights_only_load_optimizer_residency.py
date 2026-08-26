from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from art.megatron.model_support import lora_disk
from art.megatron.runtime import executor as executor_module
from art.megatron.runtime.executor import (
    GenerationResidency,
    MCoreRunSlotExecutor,
    _ResidentRunState,
)
from art.megatron.runtime.residency import ResidencyCapacityUnavailable, ResidencyKey
from art.megatron.runtime.specs import LoadStateJobSpec, TrainerGeneration

_CONFIG = {
    "base_model_name_or_path": "test/model",
    "r": 2,
    "lora_alpha": 2,
    "target_modules": ["q_proj"],
}


class _PreparedOptimizer:
    def __init__(self, tensors: tuple[torch.Tensor, ...]) -> None:
        self.tensors = tensors
        self.source = object()

    def snapshot_source(self) -> object:
        return self.source


class _SlotTrainer:
    def __init__(self) -> None:
        self.fresh_calls = 0
        self.weight_installs = 0
        self.optimizer_installs = 0

    def prepare_checkpoint_slot_load_sync(
        self, materialized: Any, *, device: str
    ) -> Any:
        assert materialized.path == "run" and device == "cpu"
        return SimpleNamespace(
            parameters=(torch.nn.Parameter(torch.arange(6.0).reshape(2, 3)),)
        )

    def prepare_checkpoint_slot_optimizer_for_residency(self, *_args: Any) -> None:
        raise AssertionError("weights-only load restored optimizer state")

    def prepare_fresh_checkpoint_slot_optimizer_for_residency(
        self, checkpoint: Any
    ) -> _PreparedOptimizer:
        self.fresh_calls += 1
        return _PreparedOptimizer(
            tuple(torch.zeros_like(parameter) for parameter in checkpoint.parameters)
        )

    def install_prepared_checkpoint_slot_load_sync(self, _checkpoint: Any) -> None:
        self.weight_installs += 1

    def install_prepared_checkpoint_slot_optimizer(
        self, _run_id: str, _optimizer: _PreparedOptimizer
    ) -> None:
        self.optimizer_installs += 1


class _Residency:
    def __init__(self) -> None:
        self.registered: tuple[tuple[ResidencyKey, tuple[torch.Tensor, ...]], ...] = ()
        self.acquisitions: list[tuple[ResidencyKey, ...]] = []
        self.acquire_error: BaseException | None = None

    def register_l2_working_set(self, working_set: Any) -> tuple[Any, ...]:
        self.registered = tuple(working_set)
        return tuple(
            SimpleNamespace(
                stats=SimpleNamespace(
                    byte_count=sum(
                        tensor.numel() * tensor.element_size() for tensor in tensors
                    )
                )
            )
            for _key, tensors in self.registered
        )

    def acquire_l1_working_set(self, keys: Any) -> None:
        self.acquisitions.append(tuple(keys))
        if self.acquire_error is not None:
            raise self.acquire_error

    acquire_prepared_l1_working_set = acquire_l1_working_set

    def release_l1_working_set(self, _keys: Any) -> None:
        return None

    def retire(self, _key: ResidencyKey) -> None:
        return None


class _Publisher:
    def __init__(self) -> None:
        self.registrations: list[dict[str, Any]] = []

    def register_resident_generation(self, **kwargs: Any) -> dict[str, float]:
        self.registrations.append(kwargs)
        return {}

    def raise_if_failed(self) -> None:
        return None


class _Gradients:
    contribution_ids = ("forward",)

    def __init__(self) -> None:
        self.seal_calls = 0

    def seal(self, _operation_ids: tuple[str, ...]) -> None:
        self.seal_calls += 1


def _key(generation_id: str) -> ResidencyKey:
    return ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id=generation_id,
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )


def _executor(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    MCoreRunSlotExecutor,
    LoadStateJobSpec,
    _SlotTrainer,
    _Residency,
    _Publisher,
]:
    generation = TrainerGeneration(
        training_session_id="session",
        policy_step=1,
        generation_id=f"step-00000001-{'1' * 32}",
        adapter_path="/replacement",
    )
    job = LoadStateJobSpec(
        operation_id="load",
        run_id="run",
        sequence_id=1,
        training_session_id="session",
        expected_learner_version=0,
        learner_version=1,
        generation=generation,
        adapter_path=generation.adapter_path,
        adapter_step=generation.policy_step,
        restore_optimizer=False,
    )
    current_key = _key(f"step-00000000-{'0' * 32}")
    state = _ResidentRunState(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=0,
        adapter_config=_CONFIG,
        gradients=None,
        desired=GenerationResidency(weights=current_key),
        installed_weights=current_key,
        registration_complete=True,
    )
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    slot = _SlotTrainer()
    residency = _Residency()
    publisher = _Publisher()
    executor.runtime = SimpleNamespace(optimizer_layout_fingerprint="topology")
    executor._slot_trainer = slot
    executor._residency = residency
    executor._publisher = publisher
    executor._runs = {"run": state}
    executor._residency_admission_lock = Lock()
    executor._residency_admissions = {}
    executor._closing = False
    executor._closed = False
    monkeypatch.setattr(
        MCoreRunSlotExecutor,
        "_build_prepared_lora_export_plan",
        lambda _self, _checkpoint: "replacement-plan",
    )
    monkeypatch.setattr(lora_disk, "load_adapter_config", lambda _path: _CONFIG)
    monkeypatch.setattr(
        executor_module,
        "read_adapter_publication",
        lambda *_args, **_kwargs: SimpleNamespace(
            training_session_id=generation.training_session_id,
            generation_id=generation.generation_id,
        ),
    )
    return executor, job, slot, residency, publisher


def _prepare_and_commit(
    executor: MCoreRunSlotExecutor, job: LoadStateJobSpec
) -> tuple[Any, dict[str, Any]]:
    prepared = executor.prepare_load_state(job)
    return prepared, executor.commit_load_state(job, prepared)


def test_weights_only_load_registers_fresh_cpu_optimizer_in_l2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, job, slot, residency, publisher = _executor(monkeypatch)

    prepared, result = _prepare_and_commit(executor, job)

    assert slot.fresh_calls == 1
    assert prepared.optimizer is not None and prepared.optimizer_key is not None
    assert not prepared.optimizer_restored and not result["optimizer_restored"]
    assert tuple(key.representation for key, _tensors in residency.registered) == (
        "weights",
        "optimizer",
    )
    assert all(
        tensor.device.type == "cpu"
        for _key, tensors in residency.registered
        for tensor in tensors
    )
    state = executor._runs["run"]
    assert state.desired.optimizer == prepared.optimizer_key
    assert state.pending_load is prepared
    assert publisher.registrations == [
        {
            "run_id": "run",
            "generation": job.generation,
            "weights_key": prepared.weights_key,
            "export_plan": "replacement-plan",
            "adapter_config": _CONFIG,
            "optimizer_source": prepared.optimizer.source,
            "optimizer_key": prepared.optimizer_key,
        }
    ]


def test_optimizer_budget_rejection_precedes_gradient_seal_and_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, job, slot, residency, _publisher = _executor(monkeypatch)
    _prepare_and_commit(executor, job)
    state = executor._runs["run"]
    gradients = _Gradients()
    state.gradients = gradients
    residency.acquire_error = ResidencyCapacityUnavailable("forced L1 capacity")
    optimizer_job = SimpleNamespace(
        operation_id="optimizer",
        run_id="run",
        training_session_id="session",
        expected_learner_version=1,
        contributing_forward_backward_operation_ids=("forward",),
    )
    state = executor._runs["run"]
    executor._residency_admissions["optimizer"] = tuple(
        key
        for key in (
            state.desired.weights,
            state.desired.optimizer,
            state.desired.accumulator,
        )
        if key is not None
    )

    with pytest.raises(ResidencyCapacityUnavailable, match="forced L1 capacity"):
        executor.execute_optimizer(optimizer_job)

    assert tuple(key.representation for key in residency.acquisitions[-1]) == (
        "weights",
        "optimizer",
    )
    assert gradients.seal_calls == 0
    assert slot.weight_installs == slot.optimizer_installs == 0


@pytest.mark.parametrize(
    ("current", "replacement"),
    (
        ({}, {"moe_parameterization": "per_expert"}),
        ({"moe_parameterization": "per_expert"}, {}),
        (
            {"moe_parameterization": "shared_outer"},
            {"moe_parameterization": "shared_outer"},
        ),
    ),
)
def test_adapter_layout_normalizes_absent_moe_parameterization(
    current: dict[str, str], replacement: dict[str, str]
) -> None:
    MCoreRunSlotExecutor._validate_adapter_layout(current, replacement)


@pytest.mark.parametrize(
    ("current", "replacement"),
    (
        ({}, {"moe_parameterization": "shared_outer"}),
        ({"moe_parameterization": "shared_outer"}, {}),
        (
            {"moe_parameterization": "per_expert"},
            {"moe_parameterization": "shared_outer"},
        ),
    ),
)
def test_adapter_layout_rejects_moe_parameterization_transition(
    current: dict[str, str], replacement: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="immutable moe_parameterization"):
        MCoreRunSlotExecutor._validate_adapter_layout(current, replacement)
