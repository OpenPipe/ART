from threading import Lock
from types import SimpleNamespace

import torch

from art.megatron.runtime.executor import (
    GenerationResidency,
    MCoreRunSlotExecutor,
    _ResidentRunState,
)
from art.megatron.runtime.residency import ResidencyKey


class _NoWait:
    def result(self, *_args, **_kwargs):
        raise AssertionError(
            "executor synchronized an asynchronous residency operation"
        )


class _ResidencyProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int | None]] = []
        self.current: tuple[ResidencyKey, tuple[torch.Tensor, ...]] | None = None
        self.tensor_ids: tuple[int, ...] | None = None
        self.acquired: list[ResidencyKey] = []
        self.working_sets: list[tuple[ResidencyKey, ...]] = []

    def register_mutable_l1(self, key, tensors):
        assert self.current is None
        self._install(key, tensors)
        self.calls.append(("register", key.accumulator_revision, None))

    def acquire_l1(self, key):
        if key.representation == "accumulator":
            assert self.current is not None and self.current[0] == key
        self.acquired.append(key)

    def acquire_l1_working_set(self, keys):
        self.working_sets.append(tuple(keys))
        for key in keys:
            self.acquire_l1(key)

    acquire_prepared_l1_working_set = acquire_l1_working_set

    def wait_before_mutation_working_set(self, keys):
        (key,) = tuple(keys)
        assert self.current is not None and self.current[0] == key
        self.calls.append(("wait", key.accumulator_revision, None))

    def begin_l1_mutation(self, key):
        assert self.current is not None and self.current[0] == key
        self.calls.append(("mutate", key.accumulator_revision, None))

    def release_l1(self, key):
        assert self.acquired.pop() == key

    def release_l1_working_set(self, keys):
        for key in reversed(tuple(keys)):
            self.release_l1(key)

    def touch(self, key):
        assert self.current is not None and self.current[0] == key
        self.calls.append(("touch", key.accumulator_revision, None))

    def retire_async(self, key):
        self.calls.append(("retire", key.accumulator_revision, None))
        return _NoWait()

    def _install(self, key, tensors):
        tensor_ids = tuple(map(id, tensors))
        self.tensor_ids = self.tensor_ids or tensor_ids
        assert tensor_ids == self.tensor_ids
        self.current = key, tensors


def test_accumulation_window_reuses_one_mutable_residency_revision() -> None:
    weights = ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id="generation",
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )
    gradient = torch.zeros(4)
    residency_reads = 0

    def residency_tensors():
        nonlocal residency_reads
        residency_reads += 1
        return (gradient,)

    state = _ResidentRunState(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=0,
        adapter_config={},
        gradients=SimpleNamespace(residency_tensors=residency_tensors),
        desired=GenerationResidency(weights=weights),
        installed_weights=weights,
    )
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    executor.runtime = SimpleNamespace(rank=0)
    residency = _ResidencyProbe()
    executor._residency = residency

    for operation_id, contribution in (("fb-1", 1.0), ("fb-2", 2.0)):
        with executor._accumulator_resident(state):
            gradient.add_(contribution)
        executor._register_gradient_contribution(state, operation_id)

    current = state.desired.accumulator
    assert current is not None
    torch.testing.assert_close(gradient, torch.full_like(gradient, 3.0))
    assert current.accumulator_revision == 1
    executor._residency_admission_lock = Lock()
    residency.acquire_l1_working_set((weights, current))
    executor._residency_admissions = {"forward": (weights, current)}
    with executor._resident(state, operation_id="forward", include_accumulator=True):
        assert residency.acquired == [weights, current, weights, current]
    assert residency.acquired == []
    assert residency.working_sets == [
        (current,),
        (weights, current),
        (weights, current),
    ]
    executor._retire_accumulator(state)
    assert state.desired.accumulator is None
    assert state.next_accumulator_revision == 2
    assert residency_reads == 1
    assert residency.calls == [
        ("register", 1, None),
        ("wait", 1, None),
        ("mutate", 1, None),
        ("touch", 1, None),
        ("retire", 1, None),
    ]
