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
        self.snapshots: dict[ResidencyKey, tuple[torch.Tensor, ...]] = {}
        self.tensor_ids: tuple[int, ...] | None = None
        self.acquired: list[ResidencyKey] = []

    def register_l1(self, key, tensors):
        assert self.current is None
        self._install(key, tensors)
        self.calls.append(("register", key.accumulator_revision, None))
        return _NoWait()

    def acquire_l1(self, key):
        if key.representation == "accumulator":
            assert self.current is not None and self.current[0] == key
        self.acquired.append(key)

    def wait_before_mutation(self, key):
        assert self.current is not None and self.current[0] == key
        self.calls.append(("wait", key.accumulator_revision, None))

    def release_l1(self, key):
        assert self.acquired.pop() == key

    def touch(self, key):
        assert self.current is not None and self.current[0] == key
        self.calls.append(("touch", key.accumulator_revision, None))

    def advance_l1(self, source, target, tensors):
        assert self.current is not None and self.current[0] == source
        assert tuple(map(id, tensors)) == self.tensor_ids
        self._install(target, tensors)
        self.calls.append(
            ("advance", source.accumulator_revision, target.accumulator_revision)
        )
        return _NoWait()

    def retire_async(self, key):
        self.calls.append(("retire", key.accumulator_revision, None))
        self.snapshots.pop(key)
        return _NoWait()

    def evict_and_recover(self, key):
        assert self.current is not None and self.current[0] == key
        tensors = self.current[1]
        for tensor in tensors:
            tensor.fill_(-1)
        for tensor, snapshot in zip(tensors, self.snapshots[key], strict=True):
            tensor.copy_(snapshot)

    def _install(self, key, tensors):
        tensor_ids = tuple(map(id, tensors))
        self.tensor_ids = self.tensor_ids or tensor_ids
        assert tensor_ids == self.tensor_ids
        self.current = key, tensors
        self.snapshots[key] = tuple(tensor.clone() for tensor in tensors)


def test_each_contribution_creates_a_recoverable_accumulator_revision() -> None:
    weights = ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id="generation",
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )
    gradient = torch.zeros(4)
    state = _ResidentRunState(
        tenant_id="tenant",
        run_id="run",
        training_session_id="session",
        learner_version=0,
        adapter_config={},
        gradients=SimpleNamespace(residency_tensors=lambda: (gradient,)),
        desired=GenerationResidency(weights=weights),
        installed_weights=weights,
    )
    executor = MCoreRunSlotExecutor.__new__(MCoreRunSlotExecutor)
    residency = _ResidencyProbe()
    executor._residency = residency

    for operation_id, contribution in (("fb-1", 1.0), ("fb-2", 2.0)):
        with executor._accumulator_resident(state):
            gradient.add_(contribution)
        executor._register_gradient_contribution(state, operation_id)

    current = state.desired.accumulator
    assert current is not None
    residency.evict_and_recover(current)

    torch.testing.assert_close(gradient, torch.full_like(gradient, 3.0))
    assert current.accumulator_revision == 2
    with executor._resident(state, include_accumulator=True):
        assert residency.acquired == [weights, current]
    assert residency.acquired == []
    executor._retire_accumulator(state)
    assert state.desired.accumulator is None
    assert state.next_accumulator_revision == 3
    assert residency.calls == [
        ("register", 1, None),
        ("wait", 1, None),
        ("advance", 1, 2),
        ("retire", 1, None),
        ("retire", 2, None),
    ]
