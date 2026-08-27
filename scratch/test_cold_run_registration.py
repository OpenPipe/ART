from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.megatron import optimizer_state as optimizer_state_module
from art.megatron.model_support import lora_disk
from art.megatron.optimizer_state import (
    CheckpointFile,
    OptimizerAdapter,
    VerifiedOptimizerGeneration,
)
from art.megatron.runtime.executor import (
    GenerationResidency,
    MCoreRunSlotExecutor,
    _GenerationPublisher,
    _PreparedRunLoad,
)
from art.megatron.runtime.residency import ResidencyCapacityUnavailable
from art.megatron.runtime.specs import (
    RankLocalOptimizerWorkSummary,
    RunSlotRegistration,
)
from art.megatron.training import gradient_accumulator as accumulator_module
from art.trainer_rank import TrainerRank


class _PreparedOptimizer:
    def __init__(self, tensors: tuple[torch.Tensor, ...], layout: Any) -> None:
        self.master_params = tuple(
            torch.nn.Parameter(tensor, requires_grad=True) for tensor in tensors
        )
        self.state = tuple(
            {
                "step": torch.zeros((), dtype=torch.float32),
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
            }
            for master in self.master_params
        )
        self.layout = layout
        self.source = SimpleNamespace(
            bind=lambda resident_tensors: {"bound": tuple(resident_tensors)}
        )

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            *self.master_params,
            *(
                tensor
                for state in self.state
                for tensor in state.values()
                if isinstance(tensor, torch.Tensor)
            ),
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
            ),
            sites=(),
            expected_keys=frozenset(),
            expected_shapes={},
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
        self.retirements: list[Any] = []
        self.retirement_errors: dict[str, BaseException] = {}
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

    def acquire_prepared_l1_working_set(self, keys: Any) -> None:
        self.acquire_l1_working_set(keys)

    def retain_l1_working_set(self, keys: Any) -> None:
        self.acquire_l1_working_set(keys)

    def prefetch_l1_working_set(self, keys: Any) -> None:
        self.acquire_l1_working_set(keys)

    def release_l1_working_set(self, _keys: Any) -> None:
        return None

    def acquire_l1(self, key: Any) -> None:
        self.acquire_l1_working_set((key,))

    def release_l1(self, key: Any) -> None:
        self.release_l1_working_set((key,))

    def retire(self, key: Any) -> None:
        self.components.pop(key, None)

    def retire_async(self, key: Any) -> Future[None]:
        self.retirements.append(key)
        future: Future[None] = Future()
        if error := self.retirement_errors.get(key.representation):
            future.set_exception(error)
        else:
            self.retire(key)
            future.set_result(None)
        return future

    def keys(self, run_id: str) -> tuple[Any, ...]:
        return tuple(key for key in self.components if key.run_id == run_id)

    def ensure_l2(self, key: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result(SimpleNamespace(tensors=lambda: self.components[key]))
        return future

    def retain_l2(self, key: Any) -> Future[Any]:
        return self.ensure_l2(key)

    def release_l2(self, _key: Any) -> None:
        return None

    @contextmanager
    def borrow_l2(self, key: Any) -> Any:
        yield SimpleNamespace(tensors=lambda: self.components[key])


class _Publisher:
    def __init__(self) -> None:
        self.registrations: list[dict[str, Any]] = []
        self.fail = False

    def register_resident_generation(self, **kwargs: Any) -> dict[str, float]:
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

    def discard(self) -> None:
        return None


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
    executor.runtime = SimpleNamespace(
        rank=0,
        optimizer_layout_fingerprint="topology",
        optimizer_semantic_sha256="0" * 64,
    )
    executor._slot_trainer = slot
    executor._residency = residency
    executor._publisher = publisher
    executor._load_preparations = {}
    executor._registration_preparations = {}
    executor._kl_reference_preparations = {}
    executor._kl_reference_cache_capacity = 1
    executor._cleanup_pool = ThreadPoolExecutor(max_workers=1)
    executor._run_cleanups = {}
    executor._runs = {}
    executor._residency_admission_lock = Lock()
    executor._residency_admissions = {}
    executor._lifecycle_lock = Lock()
    executor._transition_futures = set()
    executor._closing = False
    executor._closed = False
    monkeypatch.setattr(
        MCoreRunSlotExecutor,
        "_build_prepared_lora_export_plan",
        lambda _self, _checkpoint, **_kwargs: "export-plan",
    )
    return executor, slot, residency, publisher


@contextmanager
def _resident(
    executor: MCoreRunSlotExecutor,
    state: Any,
    *,
    include_optimizer: bool = False,
) -> Any:
    operation_id = "test-residency"
    keys = tuple(
        key
        for key in (
            state.desired.weights,
            state.desired.optimizer if include_optimizer else None,
        )
        if key is not None
    )
    executor._residency_admissions[operation_id] = keys
    with executor._resident(
        state,
        operation_id=operation_id,
        include_optimizer=include_optimizer,
    ):
        yield


def _register(
    executor: MCoreRunSlotExecutor, **kwargs: Any
) -> RankLocalOptimizerWorkSummary:
    generation_id = "step-00000000-0123456789abcdef0123456789abcdef"
    optimizer_generation = kwargs.get("initial_optimizer_generation_id")
    if (
        optimizer_generation is not None
        and "initial_optimizer_verification" not in kwargs
    ):
        kwargs["initial_optimizer_verification"] = VerifiedOptimizerGeneration(
            generation=optimizer_generation,
            manifest_sha256="0" * 64,
        )
    return executor.register_run(
        RunSlotRegistration(
            tenant_id="tenant",
            run_id="run",
            training_session_id="session",
            learner_version=0,
            generation_id=generation_id,
            adapter=OptimizerAdapter(
                identity="/adapter",
                training_session_id="session",
                step=0,
                generation_id=generation_id,
                files=(
                    CheckpointFile(name="adapter_config.json", size_bytes=1),
                    CheckpointFile(name="adapter_model.safetensors", size_bytes=1),
                ),
            ),
            optimizer_state_path="/optimizer",
            **kwargs,
        )
    )


def test_registration_returns_exact_rank_local_optimizer_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, _slot, _residency, _publisher = _executor(
        monkeypatch,
        rank=4,
        targets=("q_proj", "o_proj"),
        shapes=((4, 8), (8, 2)),
    )

    work = _register(executor)

    assert work.rank == 0
    assert work.adapter_rank == 4
    assert work.target_modules == ("q_proj", "o_proj")
    assert work.trainable_lora_numel == 48
    assert work.parameter_count == 2
    assert work.optimizer_passes == 3
    assert work.cost == 144


def test_registration_rejects_inexact_optimizer_plane_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, _residency, _publisher = _executor(monkeypatch)
    prepare = slot.prepare_fresh_checkpoint_slot_optimizer_for_residency

    def invalid(checkpoint: Any) -> _PreparedOptimizer:
        optimizer = prepare(checkpoint)
        optimizer.state[0]["exp_avg"] = torch.zeros(7)
        return optimizer

    monkeypatch.setattr(
        slot, "prepare_fresh_checkpoint_slot_optimizer_for_residency", invalid
    )

    with pytest.raises(RuntimeError, match="neither parameter nor scalar shape"):
        _register(executor)

    assert executor._runs == {}


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
        "_prepared_optimizer_valid_ranges",
        lambda candidate: (None,) if candidate is checkpoint else (),
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
    assert publisher.registrations[0]["weights_key"] == pending.weights_key
    assert publisher.registrations[0]["optimizer_key"] == pending.optimizer_key
    assert publisher.registrations[0]["export_plan"] == "export-plan"
    assert slot.weight_installs == slot.optimizer_installs == 0

    residency.acquire_error = ResidencyCapacityUnavailable("forced L1 capacity")
    with pytest.raises(ResidencyCapacityUnavailable, match="forced L1 capacity"):
        with _resident(executor, state):
            pass
    assert slot.weight_installs == slot.optimizer_installs == 0
    assert state.installed_weights is state.installed_optimizer is None

    residency.acquire_error = None
    with _resident(executor, state):
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
        with _resident(executor, state, include_optimizer=True):
            slot.optim_step_reduced("run", params=object(), grads=())
    assert slot.optimizer_installs == slot.optimizer_steps == 0
    assert state.pending_load is pending

    residency.acquire_error = None
    with _resident(executor, state, include_optimizer=True):
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
    monkeypatch.setattr(
        publisher,
        "_prepare_resident_lora",
        lambda *_args, **_kwargs: SimpleNamespace(),
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


def test_cold_first_install_setup_failure_precedes_slot_mutation(
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

    with pytest.raises(RuntimeError, match="forced accumulator failure"):
        with _resident(executor, state):
            pass

    assert slot.weight_installs == slot.unload_calls == 0
    assert slot.installed_checkpoint is None
    assert not state.checkpoint_slot_installed
    assert state.installed_weights is state.gradients is None


def test_existing_install_setup_failure_keeps_previous_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, residency, _publisher = _executor(monkeypatch)
    _register(executor)
    state = executor._runs["run"]
    with _resident(executor, state):
        pass
    previous_key = state.installed_weights
    previous_checkpoint = slot.installed_checkpoint
    previous_gradients = state.gradients
    assert previous_key is not None and previous_checkpoint is not None

    replacement = slot.prepare_checkpoint_slot_load_sync(
        SimpleNamespace(path="run"), device="cpu"
    )
    replacement_key = previous_key.model_copy(
        update={"generation_id": "replacement-generation"}
    )
    prepared = _PreparedRunLoad(
        operation_id="load:replacement-generation",
        weights_key=replacement_key,
        optimizer_key=None,
            checkpoint=replacement,
            optimizer=None,
            lora_export_plan="replacement-plan",
            optimizer_export_plan="replacement-optimizer-plan",
            adapter_config=state.adapter_config,
    )
    residency.register_l2_working_set(((replacement_key, prepared.weights),))
    state.desired = GenerationResidency(weights=replacement_key)
    state.pending_load = prepared

    def fail_before_install(**_kwargs: Any) -> None:
        assert slot.installed_checkpoint is previous_checkpoint
        assert slot.weight_installs == 1
        raise RuntimeError("forced replacement setup failure")

    monkeypatch.setattr(
        accumulator_module,
        "ParameterGradientAccumulator",
        fail_before_install,
    )

    with pytest.raises(RuntimeError, match="forced replacement setup failure"):
        with _resident(executor, state):
            pass

    assert slot.weight_installs == 1
    assert slot.installed_checkpoint is previous_checkpoint
    assert state.checkpoint_slot_installed
    assert state.installed_weights == previous_key
    assert state.gradients is previous_gradients
    assert state.pending_load is prepared
    assert residency.acquisitions[-1] == (previous_key,)


def test_unregister_retries_repeated_checkpoint_unload_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, residency, _publisher = _executor(monkeypatch)
    _register(executor)
    executor.complete_run_registration("run")
    state = executor._runs["run"]
    with _resident(executor, state):
        pass
    resident_keys = set(residency.components)
    slot.unload_error = RuntimeError("forced unload failure")

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="forced unload failure"):
            executor.unregister_run("run")
        assert executor._runs["run"] is state
        assert state.unregistering and state.checkpoint_slot_installed
        assert slot.installed_checkpoint is not None
        assert set(residency.components) == resident_keys
        assert residency.retirements == []
        with pytest.raises(RuntimeError, match="being unregistered"):
            executor.prefetch_residency("run", "forward", 0)

    slot.unload_error = None

    executor.unregister_run("run")

    assert slot.unload_calls == 3
    assert slot.installed_checkpoint is None
    assert set(residency.retirements) == resident_keys
    assert residency.components == {}
    assert "run" not in executor._runs


def test_unregister_retries_residency_retirement_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, slot, residency, _publisher = _executor(monkeypatch)
    _register(executor)
    executor.complete_run_registration("run")
    state = executor._runs["run"]
    with _resident(executor, state):
        pass
    residency.retirement_errors["weights"] = RuntimeError(
        "forced residency retirement failure"
    )
    retire = residency.retire_async

    def retire_after_unload(key: Any) -> Future[None]:
        assert slot.installed_checkpoint is None
        return retire(key)

    monkeypatch.setattr(residency, "retire_async", retire_after_unload)

    with pytest.raises(RuntimeError, match="forced residency retirement failure"):
        executor.unregister_run("run")

    assert executor._runs["run"] is state
    assert state.unregistering and not state.checkpoint_slot_installed
    assert state.installed_weights is state.installed_optimizer is None
    assert slot.unload_calls == 1 and slot.installed_checkpoint is None
    assert tuple(key.representation for key in residency.keys("run")) == ("weights",)
    with pytest.raises(RuntimeError, match="being unregistered"):
        executor.prefetch_residency("run", "forward", 0)

    residency.retirement_errors.clear()
    executor.unregister_run("run")

    assert slot.unload_calls == 1
    assert residency.components == {}
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

    def load_optimizer(**kwargs: Any) -> Any:
        loaded.update(kwargs)
        return archive

    monkeypatch.setattr(
        optimizer_state_module, "load_trainer_rank_optimizer_state", load_optimizer
    )

    _register(
        executor,
        initial_optimizer_state_path="/optimizer",
        initial_optimizer_generation_id=(
            "step-00000001-11111111111111111111111111111111"
        ),
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
    assert registration["optimizer_key"] == pending.optimizer_key
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
