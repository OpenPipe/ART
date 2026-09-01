from __future__ import annotations

from concurrent.futures import Future
import hashlib
import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator, Literal, cast

import pytest
import torch

from art.megatron import (
    MegatronRunBootstrapConfig,
    MegatronSlotRuntime,
    MegatronSlotRuntimeDescriptor,
)
from art.megatron.runtime import portable_snapshot
from art.megatron.runtime.portable_snapshot import (
    PortableSnapshotFile,
    PortableSnapshotGeneration,
    PortableSnapshotRankReceipt,
    PortableSnapshotReadFile,
    PortableSnapshotReadReceipt,
    build_portable_snapshot_archive,
)
from art.megatron.runtime.residency import ResidencyKey
from art.megatron.runtime.specs import TrainerGeneration
from art.runtime_attestation import RuntimeArchitectureAttestation
from art.training import AdapterSpec, TrainingRunSpec


@pytest.fixture
def executor_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    existing = sys.modules.get("art.megatron.runtime.executor")
    if existing is not None:
        yield existing
        return

    prior_modules = set(sys.modules)
    modules = {
        name: ModuleType(name)
        for name in (
            "megatron",
            "megatron.core",
            "megatron.core.parallel_state",
            "megatron.core.distributed",
            "megatron.core.distributed.finalize_model_grads",
            "megatron.core.transformer",
            "megatron.core.transformer.module",
        )
    }
    setattr(
        modules["megatron.core.distributed.finalize_model_grads"],
        "finalize_model_grads",
        lambda *_args, **_kwargs: None,
    )
    setattr(
        modules["megatron.core.transformer.module"],
        "MegatronModule",
        torch.nn.Module,
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    try:
        yield importlib.import_module("art.megatron.runtime.executor")
    finally:
        for name in set(sys.modules).difference(prior_modules):
            if name.startswith("art.megatron."):
                sys.modules.pop(name, None)


def test_production_run_slots_do_not_depend_on_trainer_rank(
    executor_module: Any,
) -> None:
    from art.megatron.runtime import run_slots

    assert "art.trainer_rank" not in inspect.getsource(executor_module)
    assert "art.trainer_rank" not in inspect.getsource(run_slots)


def test_rank_residency_evidence_retains_bounded_copy_facts(
    executor_module: Any,
) -> None:
    key = ResidencyKey(
        training_session_id="session",
        run_id="run",
        generation_id="generation",
        representation="weights",
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )
    state = executor_module._ResidentCommandRun(
        spec=SimpleNamespace(run_id="run"),
        learner_version=0,
        gradients=SimpleNamespace(contribution_ids=()),
        adapter_config={},
        weights_key=key,
    )
    entry = SimpleNamespace(
        copies=(
            SimpleNamespace(tier="l1_gpu", byte_count=1024, ready_at=0.0),
            SimpleNamespace(tier="l2_cpu", byte_count=1024, ready_at=0.0),
        )
    )
    executor = object.__new__(executor_module.MCoreRunSlotExecutor)
    executor._runs = {"run": state}
    executor._residency = SimpleNamespace(
        ledger=SimpleNamespace(
            entry=lambda observed: entry if observed == key else None
        )
    )
    executor.runtime = SimpleNamespace(rank=0)

    evidence = executor._residency_evidence("run", "operation", ("weights",), (key,))

    component = evidence["components"][0]
    assert component["byte_count"] == 1024
    assert component["tiers"] == ("l1_gpu", "l2_cpu")
    assert component["l1_ready"] is True
    assert component["copies"] == (
        {"tier": "l1_gpu", "byte_count": 1024, "ready": True},
        {"tier": "l2_cpu", "byte_count": 1024, "ready": True},
    )


def test_fresh_optimizer_is_prepared_entirely_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    executor_module: Any,
) -> None:
    from art.megatron import checkpoint
    from art.megatron.runtime.run_slots import MegatronRunSlots, OptimizerConfig

    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    slots = object.__new__(MegatronRunSlots)
    slots._checkpoint_slots = {
        "run": checkpoint.CheckpointSlot(params=(parameter,), config={})
    }
    monkeypatch.setattr(
        MegatronRunSlots,
        "_zero_dynamic_optimizer_padding",
        lambda *_args, **_kwargs: None,
    )
    cuda_initialized = torch.cuda.is_initialized()

    tensors = slots.prepare_fresh_checkpoint_slot_optimizer_for_residency(
        "run", OptimizerConfig(learning_rate=0.0)
    )

    assert torch.cuda.is_initialized() == cuda_initialized
    dynamic = slots._checkpoint_slots["run"].optimizer
    assert dynamic is not None
    assert len(dynamic.master_params) == 1
    master = dynamic.master_params[0]
    assert master.device.type == "cpu"
    assert master.dtype == torch.float32
    assert master.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()
    torch.testing.assert_close(master, parameter.float())
    state = dynamic.optimizer.state[master]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert tuple(map(id, tensors)) == tuple(
        map(id, (master, state["exp_avg"], state["exp_avg_sq"]))
    )


def test_cold_registration_atomically_adopts_cpu_working_set(
    monkeypatch: pytest.MonkeyPatch,
    executor_module: Any,
) -> None:
    from art.megatron.model_support import lora_disk

    weights = (torch.nn.Parameter(torch.tensor([1.0])),)
    optimizer = (torch.tensor([2.0]), torch.tensor([3.0]))

    class Slots:
        def __init__(self) -> None:
            self.loads: list[tuple[str, str]] = []

        def load_checkpoint_for_residency(
            self, run_id: str, path: str
        ) -> tuple[tuple[torch.nn.Parameter, ...], tuple[torch.Tensor, ...]]:
            self.loads.append((run_id, path))
            return weights, optimizer

        def release_checkpoint_slot(self, _run_id: str) -> None:
            raise AssertionError("successful cold registration released its slot")

    monkeypatch.setattr(
        lora_disk,
        "load_adapter_config",
        lambda _path: {"r": 4, "target_modules": ["q_proj"]},
    )
    monkeypatch.setattr(
        lora_disk,
        "training_target_modules",
        lambda _config: ("q_proj",),
    )
    slots = Slots()
    residency = _Residency(())
    executor = object.__new__(executor_module.MCoreRunSlotExecutor)
    executor._closed = False
    executor._runs = {}
    executor._slots = slots
    executor._residency = residency
    executor._topology_fingerprint = "topology"
    spec = SimpleNamespace(
        run_id="run",
        training_session_id="session",
        lora_rank=4,
        lora_target_modules=("q_proj",),
        initial_adapter_path="/adapter",
        initial_portable_snapshot=None,
        initial_learner_version=0,
        initial_generation_id="generation",
    )

    assert executor.register_run(spec) is None

    assert slots.loads == [("run", "/adapter")]
    assert [event for event, _value in residency.events] == ["register_l2"]
    working_set = residency.events[0][1]
    assert tuple(key.representation for key, _tensors in working_set) == (
        "weights",
        "optimizer",
    )
    assert working_set[0][1] is weights
    assert working_set[1][1] is optimizer


def _archive(*, step: int = 7) -> Any:
    checkpoint_digest = "d" * 64
    files = tuple(
        PortableSnapshotFile(
            object_id=f"object-{index}",
            relative_path=path,
            component=cast(Literal["metadata", "adapter", "optimizer"], component),
            byte_count=1,
            sha256=hashlib.sha256(path.encode()).hexdigest(),
            source_ref=f"wandb://artifact/{path}",
        )
        for index, (path, component) in enumerate(
            (
                ("adapter_config.json", "metadata"),
                ("adapter_model.safetensors", "adapter"),
                ("checkpoint.json", "metadata"),
            )
        )
    )
    generation = PortableSnapshotGeneration(
        training_session_id="session",
        policy_step=step,
        generation_id=f"step-{step:08d}-{'a' * 32}",
    )
    return build_portable_snapshot_archive(
        generation=generation,
        checkpoint_digest=checkpoint_digest,
        ranks=(
            PortableSnapshotRankReceipt(
                rank=0,
                checkpoint_digest=checkpoint_digest,
                files=files,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_cold_bind_uses_portable_archive_generation(tmp_path: Any) -> None:
    archive = _archive(step=37)
    output_dir = tmp_path / "run"
    registered: dict[str, Any] = {}

    class Coordinator:
        trainer = SimpleNamespace(
            runtime_spec=SimpleNamespace(
                model_identifier="model",
                dtype="bfloat16",
                lora_rank=8,
                lora_target_modules=("q_proj",),
                lora_alpha=32.0,
                random_state=0,
                allow_unvalidated_arch=False,
            )
        )

        async def register_run(self, config: Any, **kwargs: Any) -> object:
            registered.update(config=config, **kwargs)
            return "run"

    slot = MegatronSlotRuntime(
        runtime=SimpleNamespace(),
        coordinator=Coordinator(),  # type: ignore[arg-type]
        descriptor=MegatronSlotRuntimeDescriptor(
            runtime_source_id="slot",
            runtime_source_epoch=1,
            runtime_fingerprint="f" * 64,
            trainer_architecture=RuntimeArchitectureAttestation.create(
                runtime_kind="trainer",
                base_model="model",
                model_source="model",
                model_revision="default",
                model_support_key="test",
                handler_name="test",
                canonical_config_sha256="a" * 64,
                loaded_layer_count=1,
                tensor_parallel_size=1,
                context_parallel_size=1,
                pipeline_parallel_size=1,
                expert_parallel_size=1,
                data_parallel_size=1,
                world_size=1,
                runtime_identity="f" * 64,
            ),
        ),
    )
    binding = await slot.bind_run(
        MegatronRunBootstrapConfig(
            run_id="run",
            training_session_id="session",
            run=TrainingRunSpec(
                base_model="model",
                adapter=AdapterSpec(rank=4, target_modules=("q_proj",)),
            ),
            output_dir=str(output_dir),
        ),
        portable_archive=archive,
    )

    assert binding.run == "run"
    assert binding.config.source.policy_step == 37
    assert binding.config.source.generation_id == archive.generation.generation_id
    assert binding.config.source.adapter_path == str(
        output_dir / "checkpoints" / archive.generation.generation_id
    )
    assert registered["portable_archive"] is archive
    assert not output_dir.exists()


class _Residency:
    def __init__(self, keys: tuple[ResidencyKey, ...]) -> None:
        self.config = SimpleNamespace(shutdown_timeout_s=1.0)
        self.current = set(keys)
        self.events: list[tuple[str, Any]] = []

    def register_l1(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("cold run registration eagerly registered L1 state")

    def register_l2_working_set(self, working_set: Any) -> None:
        working_set = tuple(working_set)
        assert all(
            tensor.device.type == "cpu"
            for _key, tensors in working_set
            for tensor in tensors
        )
        self.current.update(key for key, _tensors in working_set)
        self.events.append(("register_l2", working_set))

    def retire_async(self, key: ResidencyKey) -> Future[None]:
        self.current.discard(key)
        self.events.append(("retire", key))
        future: Future[None] = Future()
        future.set_result(None)
        return future


def _executor(executor_module: Any) -> tuple[Any, Any, _Residency]:
    spec = SimpleNamespace(
        run_id="run",
        training_session_id="session",
        lora_rank=4,
        lora_target_modules=("q_proj",),
    )
    state = executor_module._ResidentCommandRun(
        spec=spec,
        learner_version=2,
        gradients=SimpleNamespace(contribution_ids=()),
        adapter_config={"r": 4, "target_modules": ["q_proj"]},
    )
    old_keys = tuple(
        ResidencyKey(
            training_session_id="session",
            run_id="run",
            generation_id="step-00000002-" + "b" * 32,
            representation=representation,
            topology_fingerprint="topology",
            adapter_layout_fingerprint="old-layout",
        )
        for representation in ("weights", "optimizer")
    )
    state.weights_key, state.optimizer_key = old_keys
    residency = _Residency(old_keys)
    executor = object.__new__(executor_module.MCoreRunSlotExecutor)
    executor._runs = {"run": state}
    executor._residency = residency
    executor._checkpoint_hydrations = {}
    executor._portable_snapshot_source = object()
    executor._topology_fingerprint = "topology"
    executor.runtime = SimpleNamespace(rank=0)
    executor._slots = SimpleNamespace(release_checkpoint_slot=lambda _name: None)
    return executor, state, residency


def _generation() -> TrainerGeneration:
    return TrainerGeneration(
        training_session_id="session",
        policy_step=8,
        generation_id="step-00000008-" + "c" * 32,
        adapter_path="/adopted/adapter",
    )


def test_restore_preparation_failure_preserves_existing_run(
    monkeypatch: pytest.MonkeyPatch,
    executor_module: Any,
) -> None:
    executor, state, residency = _executor(executor_module)
    original = (
        state.learner_version,
        state.gradients,
        state.weights_key,
        state.optimizer_key,
    )

    def fail_prepare(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("archive read failed")

    monkeypatch.setattr(portable_snapshot, "prepare_portable_checkpoint", fail_prepare)

    with pytest.raises(RuntimeError, match="archive read failed"):
        executor.prepare_run_checkpoint(
            "load-operation",
            "run",
            _generation(),
            _archive(),
            restore_optimizer=False,
        )

    assert residency.events == []
    assert (
        state.learner_version,
        state.gradients,
        state.weights_key,
        state.optimizer_key,
    ) == original


def test_restore_adopts_prepared_cpu_working_set_before_commit(
    monkeypatch: pytest.MonkeyPatch,
    executor_module: Any,
) -> None:
    executor, state, residency = _executor(executor_module)
    old_keys = (state.weights_key, state.optimizer_key)
    archive = _archive()
    generation = _generation()
    weights = (torch.nn.Parameter(torch.tensor([1.0])),)
    optimizer = (torch.tensor([2.0]),)
    receipt = PortableSnapshotReadReceipt(
        archive_sha256=archive.archive_sha256,
        destination_rank=0,
        files=(
            PortableSnapshotReadFile(
                source_rank=0,
                relative_path="adapter_model.safetensors",
                byte_count=1,
                sha256="e" * 64,
            ),
        ),
    )
    prepared = executor_module._PreparedPortableRun(
        receipt=receipt,
        staging_name="staged",
        adapter_config={"r": 4, "target_modules": ["q_proj"]},
        weights=weights,
        optimizer_tensors=optimizer,
    )
    prepared_call: dict[str, Any] = {}

    def prepare(_self: Any, **kwargs: Any) -> Any:
        prepared_call.update(kwargs)
        return prepared

    def commit(_trainer: Any, *, staging_name: str, name: str) -> None:
        residency.events.append(("commit", (staging_name, name)))

    monkeypatch.setattr(
        executor_module.MCoreRunSlotExecutor, "_prepare_portable_run", prepare
    )
    monkeypatch.setattr(
        portable_snapshot, "commit_prepared_portable_checkpoint", commit
    )

    observed = executor.prepare_run_checkpoint(
        "load-operation",
        "run",
        generation,
        archive,
        restore_optimizer=False,
    )

    assert observed is receipt
    assert prepared_call["restore_optimizer"] is False
    assert prepared_call["generation_id"] == generation.generation_id
    assert prepared_call["archive"] is archive
    assert [event for event, _value in residency.events] == ["register_l2"]
    assert state.learner_version == 2
    assert (state.weights_key, state.optimizer_key) == old_keys

    committed = executor.commit_prepared_run_checkpoint("load-operation", "run")

    assert committed is receipt
    assert [event for event, _value in residency.events] == [
        "register_l2",
        "commit",
        "retire",
        "retire",
    ]
    assert (
        tuple(value for event, value in residency.events if event == "retire")
        == old_keys
    )
    assert state.learner_version == 8
    assert state.gradients.parameters == weights
    assert state.portable_read is receipt
    assert state.weights_key.generation_id == generation.generation_id
    assert state.optimizer_key.generation_id == generation.generation_id
    assert state.weights_key in residency.current
    assert state.optimizer_key in residency.current
