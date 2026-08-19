from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.megatron import optimizer_state as optimizer_state_module
from art.megatron.model_support import lora_disk
from art.megatron.runtime.executor import MCoreRunSlotExecutor, _GenerationPublisher
from art.megatron.runtime.residency import ResidencyCapacityUnavailable
from art.megatron.training import gradient_accumulator as accumulator_module
from art.trainer_rank import TrainerRank


class _PreparedOptimizer:
    def __init__(self, tensors: tuple[torch.Tensor, ...], layout: Any) -> None:
        self.tensors = tensors
        self.layout = layout
        self.source = SimpleNamespace(
            bind=lambda resident_tensors: {"bound": tuple(resident_tensors)}
        )

    def snapshot_source(self) -> object:
        return self.source


class _SlotTrainer:
    def __init__(self, shapes: tuple[tuple[int, ...], ...], targets: tuple[str, ...]):
        self.shapes = shapes
        self.targets = targets
        self.prepare_devices: list[str] = []
        self.fresh_calls = 0
        self.exact_calls: list[tuple[str, Any, Any]] = []
        self.weight_installs = 0
        self.optimizer_installs = 0
        self.optimizer_steps = 0
        self.clear_calls = 0
        self.unload_calls = 0
        self.unload_error: BaseException | None = None
        self.checkpoint: Any | None = None
        self.optimizer: _PreparedOptimizer | None = None
        self.installed_checkpoint: Any | None = None
        self.installed_optimizer: _PreparedOptimizer | None = None

    def prepare_checkpoint_slot_load_sync(
        self, materialized: Any, *, device: str
    ) -> Any:
        assert materialized.path == "run"
        self.prepare_devices.append(device)
        self.checkpoint = SimpleNamespace(
            parameters=tuple(
                torch.nn.Parameter(
                    torch.arange(
                        torch.Size(shape).numel(), dtype=torch.float32
                    ).reshape(shape)
                )
                for shape in self.shapes
            )
        )
        return self.checkpoint

    def prepared_checkpoint_slot_optimizer_layout(self, checkpoint: Any) -> Any:
        return {
            "shapes": [list(parameter.shape) for parameter in checkpoint.parameters],
            "targets": list(self.targets),
        }

    def prepare_fresh_checkpoint_slot_optimizer_for_residency(
        self, checkpoint: Any
    ) -> _PreparedOptimizer:
        self.fresh_calls += 1
        self.optimizer = _PreparedOptimizer(
            tuple(
                torch.zeros_like(parameter, dtype=torch.float32)
                for parameter in checkpoint.parameters
            ),
            self.prepared_checkpoint_slot_optimizer_layout(checkpoint),
        )
        return self.optimizer

    def prepare_checkpoint_slot_optimizer_for_residency(
        self, name: str, checkpoint: Any, state: Any
    ) -> _PreparedOptimizer:
        self.exact_calls.append((name, checkpoint, state))
        self.optimizer = _PreparedOptimizer(
            tuple(state["tensors"]),
            self.prepared_checkpoint_slot_optimizer_layout(checkpoint),
        )
        return self.optimizer

    def load_checkpoint_sync(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("cold registration called load_checkpoint_sync")

    def restore_checkpoint_slot_optimizer_state(
        self, *_args: Any, **_kwargs: Any
    ) -> None:
        raise AssertionError("cold registration restored a live optimizer")

    def install_prepared_checkpoint_slot_load_sync(self, checkpoint: Any) -> None:
        self.weight_installs += 1
        self.installed_checkpoint = checkpoint

    def install_prepared_checkpoint_slot_optimizer(
        self, name: str, optimizer: _PreparedOptimizer
    ) -> None:
        assert name == "run"
        self.optimizer_installs += 1
        self.installed_optimizer = optimizer

    def clear_checkpoint_slot_optimizer(self, name: str) -> None:
        assert name == "run"
        self.clear_calls += 1

    def optim_step_reduced(self, name: str, *, params: Any, grads: Any) -> None:
        assert name == "run" and params is not None and grads == ()
        assert self.installed_optimizer is self.optimizer
        self.optimizer_steps += 1

    def checkpoint_slot_residency_tensors(self, name: str) -> Any:
        assert name == "run" and self.installed_checkpoint is not None
        return SimpleNamespace(
            weights=self.installed_checkpoint.parameters,
            optimizer=(
                ()
                if self.installed_optimizer is None
                else self.installed_optimizer.tensors
            ),
        )

    def checkpoint_slot_parameters(self, name: str) -> tuple[torch.Tensor, ...]:
        assert name == "run" and self.installed_checkpoint is not None
        return self.installed_checkpoint.parameters

    def unload_checkpoint_slot(self, name: str) -> None:
        assert name == "run"
        self.unload_calls += 1
        if self.unload_error is not None:
            raise self.unload_error
        self.installed_checkpoint = None
        self.installed_optimizer = None


class _Residency:
    def __init__(self) -> None:
        self.config = SimpleNamespace(shutdown_timeout_s=1.0)
        self.components: dict[Any, tuple[torch.Tensor, ...]] = {}
        self.admissions: list[tuple[tuple[Any, tuple[torch.Tensor, ...]], ...]] = []
        self.acquisitions: list[tuple[Any, ...]] = []
        self.acquire_error: BaseException | None = None
        self.admission_error: BaseException | None = None

    def register_l2_working_set(self, working_set: Any) -> tuple[Any, ...]:
        admission = tuple(working_set)
        self.admissions.append(admission)
        if self.admission_error is not None:
            raise self.admission_error
        self.components.update(admission)
        return tuple(SimpleNamespace() for _component in admission)

    def acquire_l1_working_set(self, keys: Any) -> None:
        keys = tuple(keys)
        self.acquisitions.append(keys)
        if self.acquire_error is not None:
            raise self.acquire_error

    def release_l1_working_set(self, _keys: Any) -> None:
        return None

    def retire(self, key: Any) -> None:
        self.components.pop(key, None)

    def retire_async(self, key: Any) -> Future[None]:
        self.retire(key)
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def keys(self, run_id: str) -> tuple[Any, ...]:
        return tuple(key for key in self.components if key.run_id == run_id)

    def ensure_l2(self, key: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result(SimpleNamespace(tensors=lambda: self.components[key]))
        return future

    @contextmanager
    def borrow_l2(self, key: Any) -> Any:
        yield SimpleNamespace(tensors=lambda: self.components[key])


class _Publisher:
    def __init__(self) -> None:
        self.registrations: list[dict[str, Any]] = []
        self.fail = False

    def register_existing(self, **kwargs: Any) -> dict[str, float]:
        if self.fail:
            raise RuntimeError("forced publication failure")
        self.registrations.append(kwargs)
        return {}

    def retire_run(self, run_id: str) -> None:
        self.registrations = [
            registration
            for registration in self.registrations
            if registration["run_id"] != run_id
        ]


class _Accumulator:
    def __init__(self, *, parameters: tuple[torch.Tensor, ...]) -> None:
        self.parameters = parameters


def _executor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int = 2,
    targets: tuple[str, ...] = ("q_proj",),
    shapes: tuple[tuple[int, ...], ...] = ((2, 3),),
) -> tuple[MCoreRunSlotExecutor, _SlotTrainer, _Residency, _Publisher]:
    config = {
        "base_model_name_or_path": "test/model",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": list(targets),
    }
    monkeypatch.setattr(lora_disk, "load_adapter_config", lambda _path: config)
    monkeypatch.setattr(
        accumulator_module, "ParameterGradientAccumulator", _Accumulator
    )
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    slot = _SlotTrainer(shapes, targets)
    residency = _Residency()
    publisher = _Publisher()
    executor.runtime = SimpleNamespace(optimizer_layout_fingerprint="topology")
    executor._slot_trainer = slot
    executor._residency = residency
    executor._publisher = publisher
    executor._load_preparations = {}
    executor._runs = {}
    executor._closed = False
    return executor, slot, residency, publisher


def _register(executor: MCoreRunSlotExecutor, **kwargs: Any) -> None:
    executor.register_run(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=0,
        generation_id="step-00000000-0123456789abcdef0123456789abcdef",
        adapter_path="/adapter",
        adapter_step=0,
        **kwargs,
    )


def test_exact_optimizer_preparation_uses_no_installed_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = object.__new__(TrainerRank)
    parameter = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    checkpoint = SimpleNamespace(parameters=(parameter,))
    layout = {
        "parallel": (0, 0, 0, 0, 0, 0, 0, 0),
        "parameters": (
            (("parameter",), (2, 3), "torch.float32", "tp", False, None, "uniform", ()),
        ),
    }
    archive = {
        "format_version": 1,
        "layout": layout,
        "master_params": (parameter.detach().clone(),),
        "optimizer": {
            "param_groups": [{"params": [0]}],
            "state": {0: {}},
        },
    }
    monkeypatch.setattr(
        trainer,
        "prepared_checkpoint_slot_optimizer_layout",
        lambda candidate: layout if candidate is checkpoint else None,
        raising=False,
    )
    monkeypatch.setattr(
        trainer,
        "_dynamic_optimizer_layout",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("exact CPU preparation inspected an installed slot")
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_prepared_optimizer_padding_masks",
        lambda candidate: (
            (torch.zeros_like(parameter, dtype=torch.bool),)
            if candidate is checkpoint
            else ()
        ),
    )

    prepared = trainer.prepare_checkpoint_slot_optimizer_for_residency(
        "run", checkpoint, archive
    )

    assert prepared.layout == layout
    assert all(tensor.device.type == "cpu" for tensor in prepared.tensors)


def test_fresh_registration_first_fb_then_optim_reuses_prepared_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, residency, publisher = _executor(monkeypatch)
    cuda_initialized = torch.cuda.is_initialized()

    _register(executor)

    state = executor._runs["run"]
    pending = state.pending_load
    assert pending is not None and pending.optimizer is not None
    assert torch.cuda.is_initialized() == cuda_initialized
    assert slot.prepare_devices == ["cpu"]
    assert slot.fresh_calls == 1 and slot.exact_calls == []
    assert len(residency.admissions) == 1
    assert tuple(key.representation for key, _tensors in residency.admissions[0]) == (
        "weights",
        "optimizer",
    )
    assert all(
        tensor.device.type == "cpu"
        for _key, tensors in residency.admissions[0]
        for tensor in tensors
    )
    assert state.installed_weights is state.installed_optimizer is None
    assert state.gradients is None
    assert state.desired.optimizer == pending.optimizer_key
    assert slot.weight_installs == slot.optimizer_installs == 0
    optimizer_tensor_ids = tuple(map(id, pending.optimizer_tensors))
    optimizer_storage_ptrs = tuple(
        tensor.untyped_storage().data_ptr() for tensor in pending.optimizer_tensors
    )

    executor.complete_run_registration("run")

    assert publisher.registrations[0]["optimizer_source"] is pending.optimizer.source
    assert (
        publisher.registrations[0]["optimizer_residency_key"] == pending.optimizer_key
    )
    assert slot.weight_installs == slot.optimizer_installs == 0

    residency.acquire_error = ResidencyCapacityUnavailable("forced L1 capacity")
    with pytest.raises(ResidencyCapacityUnavailable, match="forced L1 capacity"):
        with executor._resident(state):
            pass
    assert slot.weight_installs == slot.optimizer_installs == 0
    assert state.installed_weights is state.installed_optimizer is None

    residency.acquire_error = None
    with executor._resident(state):
        assert state.installed_weights == state.desired.weights
        assert state.installed_optimizer is None
        assert state.gradients is not None
    assert tuple(key.representation for key in residency.acquisitions[-1]) == (
        "weights",
    )
    assert slot.weight_installs == 1 and slot.optimizer_installs == 0
    assert state.pending_load is pending

    residency.acquire_error = ResidencyCapacityUnavailable(
        "forced optimizer L1 capacity"
    )
    with pytest.raises(ResidencyCapacityUnavailable, match="optimizer L1 capacity"):
        with executor._resident(state, include_optimizer=True):
            slot.optim_step_reduced("run", params=object(), grads=())
    assert slot.optimizer_installs == slot.optimizer_steps == 0
    assert state.pending_load is pending

    residency.acquire_error = None
    with executor._resident(state, include_optimizer=True):
        assert state.installed_optimizer == state.desired.optimizer
        slot.optim_step_reduced("run", params=object(), grads=())
    assert tuple(key.representation for key in residency.acquisitions[-1]) == (
        "weights",
        "optimizer",
    )
    assert slot.optimizer_installs == 1
    assert slot.optimizer_steps == 1
    assert slot.installed_optimizer is pending.optimizer
    assert tuple(map(id, slot.installed_optimizer.tensors)) == optimizer_tensor_ids
    assert (
        tuple(
            tensor.untyped_storage().data_ptr()
            for tensor in slot.installed_optimizer.tensors
        )
        == optimizer_storage_ptrs
    )
    assert state.pending_load is None
    assert slot.clear_calls == 0


def test_fresh_initial_optimizer_snapshots_from_l2_before_l1_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, residency, _publisher = _executor(monkeypatch)
    _register(executor)
    state = executor._runs["run"]
    generation = state.initial_generation
    pending = state.pending_load
    assert generation is not None and pending is not None
    publisher = _GenerationPublisher(
        SimpleNamespace(rank=0, world_size=1),
        capacity=2,
        residency=cast(Any, residency),
    )
    executor._publisher = publisher
    optimizer_snapshot = object()

    def snapshot_from_cpu(_runtime: Any, optimizer_state: Any, **identity: Any) -> Any:
        assert set(optimizer_state) == {"bound"}
        assert tuple(map(id, optimizer_state["bound"])) == tuple(
            map(id, pending.optimizer_tensors)
        )
        assert identity == {
            "generation_id": generation.generation_id,
            "step": generation.policy_step,
        }
        return optimizer_snapshot

    monkeypatch.setattr(
        optimizer_state_module,
        "trainer_rank_optimizer_snapshot_from_cpu",
        snapshot_from_cpu,
    )

    executor.complete_run_registration("run")
    entry = publisher._cache[generation.generation_id]
    with publisher._optimizer_snapshot(entry, generation, required=True) as snapshot:
        assert snapshot is optimizer_snapshot

    assert slot.weight_installs == slot.optimizer_installs == 0
    assert state.installed_weights is state.installed_optimizer is None
    publisher.retire_run("run")
    publisher.close()


@pytest.mark.parametrize("failure", ["identity", "accumulator"])
def test_cold_first_install_rolls_back_post_install_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    executor, slot, _residency, _publisher = _executor(monkeypatch)
    _register(executor)
    state = executor._runs["run"]

    if failure == "identity":
        monkeypatch.setattr(
            slot,
            "checkpoint_slot_residency_tensors",
            lambda _name: SimpleNamespace(weights=(torch.zeros(1),), optimizer=()),
        )
        expected = "changed prepared weight tensors"
    else:
        monkeypatch.setattr(
            accumulator_module,
            "ParameterGradientAccumulator",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("forced accumulator failure")
            ),
        )
        expected = "forced accumulator failure"

    with pytest.raises(RuntimeError, match=expected):
        with executor._resident(state):
            pass

    assert slot.weight_installs == slot.unload_calls == 1
    assert slot.installed_checkpoint is None
    assert not state.checkpoint_slot_installed
    assert state.installed_weights is state.gradients is None


def test_unregister_retries_failed_cold_install_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, _residency, _publisher = _executor(monkeypatch)
    _register(executor)
    state = executor._runs["run"]
    monkeypatch.setattr(
        accumulator_module,
        "ParameterGradientAccumulator",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced accumulator failure")
        ),
    )
    slot.unload_error = RuntimeError("forced rollback failure")

    with pytest.raises(BaseExceptionGroup, match="install and rollback failed"):
        with executor._resident(state):
            pass

    assert state.checkpoint_slot_installed
    assert state.installed_weights is None
    assert slot.installed_checkpoint is not None
    slot.unload_error = None

    executor.unregister_run("run")

    assert slot.unload_calls == 2
    assert slot.installed_checkpoint is None
    assert "run" not in executor._runs


@pytest.mark.parametrize(
    ("rank", "targets", "shapes"),
    (
        (1, ("q_proj",), ((1, 4),)),
        (4, ("gate_proj", "down_proj"), ((4, 8), (8, 4))),
    ),
)
def test_exact_resume_uses_prepared_layout_and_is_published_exactly(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    targets: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> None:
    executor, slot, _residency, publisher = _executor(
        monkeypatch, rank=rank, targets=targets, shapes=shapes
    )
    loaded: dict[str, Any] = {}
    archive = {
        "tensors": tuple(torch.zeros(shape, dtype=torch.float32) for shape in shapes)
    }

    def load_optimizer(_runtime: Any, **kwargs: Any) -> Any:
        loaded.update(kwargs)
        return archive

    monkeypatch.setattr(
        optimizer_state_module, "load_trainer_rank_optimizer_state", load_optimizer
    )

    _register(
        executor,
        initial_optimizer_state_path="/optimizer",
        initial_optimizer_generation_id="optimizer-generation",
    )
    state = executor._runs["run"]
    pending = state.pending_load
    assert pending is not None and pending.optimizer is not None
    assert loaded["layout"] == {
        "shapes": [list(shape) for shape in shapes],
        "targets": list(targets),
    }
    assert slot.fresh_calls == 0
    assert slot.exact_calls == [("run", pending.checkpoint, archive)]

    executor.complete_run_registration("run")

    registration = publisher.registrations[0]
    assert registration["optimizer_source"] is pending.optimizer.source
    assert registration["optimizer_residency_key"] == pending.optimizer_key
    assert slot.weight_installs == slot.optimizer_installs == 0


def test_admission_and_publication_failures_leave_no_registered_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, residency, publisher = _executor(monkeypatch)
    prepare = slot.prepare_checkpoint_slot_load_sync
    monkeypatch.setattr(
        slot,
        "prepare_checkpoint_slot_load_sync",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced CPU preparation failure")
        ),
    )
    with pytest.raises(RuntimeError, match="forced CPU preparation failure"):
        _register(executor)

    assert executor._runs == {}
    assert residency.components == {}
    assert publisher.registrations == []

    monkeypatch.setattr(slot, "prepare_checkpoint_slot_load_sync", prepare)
    prepare_optimizer = slot.prepare_fresh_checkpoint_slot_optimizer_for_residency
    monkeypatch.setattr(
        slot,
        "prepare_fresh_checkpoint_slot_optimizer_for_residency",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced CPU optimizer preparation failure")
        ),
    )
    with pytest.raises(RuntimeError, match="forced CPU optimizer preparation failure"):
        _register(executor)

    assert executor._runs == {}
    assert residency.components == {}
    assert publisher.registrations == []

    monkeypatch.setattr(
        slot,
        "prepare_fresh_checkpoint_slot_optimizer_for_residency",
        prepare_optimizer,
    )
    residency.admission_error = RuntimeError("forced atomic admission failure")

    with pytest.raises(RuntimeError, match="forced atomic admission failure"):
        _register(executor)

    assert executor._runs == {}
    assert residency.components == {}
    assert publisher.registrations == []
    assert slot.weight_installs == slot.optimizer_installs == 0

    residency.admission_error = None
    _register(executor)
    publisher.fail = True
    with pytest.raises(RuntimeError, match="forced publication failure"):
        executor.complete_run_registration("run")

    assert executor._runs == {}
    assert residency.components == {}
    assert publisher.registrations == []
    assert slot.unload_calls == 0
