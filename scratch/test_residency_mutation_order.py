from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
import torch

from art.megatron.runtime.nvme_residency import NvmeResidencyStoreConfig
from art.megatron.runtime.residency import ResidencyKey, ResidencyLimits, TierCapacity
from art.megatron.runtime.run_residency import RunResidencyConfig, RunResidencyManager
from art.megatron.tensor_snapshot import SnapshotReadBarrier


class _DeferredCopyEvent:
    def __init__(self, source: torch.Tensor, target: torch.Tensor) -> None:
        self.source = source
        self.target = target
        self.completed = False

    def complete(self) -> None:
        self.target.copy_(self.source)
        self.completed = True

    def synchronize(self) -> None:
        raise AssertionError("mutation fencing synchronized the CPU")


class _MutationStream:
    def __init__(self) -> None:
        self.waited: list[_DeferredCopyEvent] = []

    def wait_event(self, event: _DeferredCopyEvent) -> None:
        self.waited.append(event)
        event.complete()


def _barrier(event: _DeferredCopyEvent) -> SnapshotReadBarrier:
    barrier = SnapshotReadBarrier()
    pending = SimpleNamespace(
        fences=(SimpleNamespace(device=0, event=event),),
    )
    barrier.register(cast(Any, pending))
    return barrier


def _manager(tmp_path: Path) -> RunResidencyManager:
    capacity = TierCapacity(max_bytes=64 << 20)
    return RunResidencyManager(
        RunResidencyConfig(
            limits=ResidencyLimits(
                l1_gpu=capacity,
                l2_cpu=capacity,
                l3_nvme=capacity,
            ),
            nvme=NvmeResidencyStoreConfig(root=str(tmp_path / "l3")),
        ),
        snapshot_barrier=SnapshotReadBarrier(),
    )


def _key(
    representation: Literal["weights", "optimizer", "accumulator"],
    *,
    generation_id: str = "generation",
) -> ResidencyKey:
    return ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id=generation_id,
        representation=representation,
        accumulator_revision=1 if representation == "accumulator" else 0,
        topology_fingerprint="topology",
        adapter_layout_fingerprint="adapter",
    )


def test_working_set_fence_orders_every_component_without_blocking_unrelated_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    keys = tuple(
        _key(component) for component in ("weights", "optimizer", "accumulator")
    )
    unrelated = _key("weights", generation_id="unrelated-generation")
    sources = {key: torch.full((4,), index + 1.0) for index, key in enumerate(keys)}
    images = {key: torch.empty_like(source) for key, source in sources.items()}
    events = {key: _DeferredCopyEvent(sources[key], images[key]) for key in keys}
    unrelated_event = _DeferredCopyEvent(torch.ones(1), torch.empty(1))
    for key, source in sources.items():
        manager.register_l2(key, (source,))
    manager.register_l2(unrelated, (torch.ones(1),))
    for key, event in (*events.items(), (unrelated, unrelated_event)):
        manager._mutation_barriers[key] = _barrier(event)
    stream = _MutationStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)

    manager.wait_before_mutation_working_set(keys)
    for key, source in sources.items():
        assert events[key].completed
        source.add_(10)

    assert stream.waited == list(events.values())
    assert not unrelated_event.completed
    for index, key in enumerate(keys):
        torch.testing.assert_close(images[key], torch.full((4,), index + 1.0))
        torch.testing.assert_close(sources[key], torch.full((4,), index + 11.0))
    weights_barrier = manager._mutation_barriers[keys[0]]
    manager._mutation_barriers.pop(keys[1])

    with pytest.raises(RuntimeError, match="missing mutation fence.*optimizer"):
        manager.wait_before_mutation_working_set(keys)

    assert manager._mutation_barriers[keys[0]] is weights_barrier
    manager.close()


def test_each_l2_snapshot_precedes_mutation_of_same_residency_key(
    tmp_path: Path,
) -> None:
    capacity = TierCapacity(max_bytes=64 << 20)
    manager = RunResidencyManager(
        RunResidencyConfig(
            limits=ResidencyLimits(
                l1_gpu=capacity,
                l2_cpu=capacity,
                l3_nvme=capacity,
            ),
            nvme=NvmeResidencyStoreConfig(root=str(tmp_path / "l3")),
        ),
        snapshot_barrier=SnapshotReadBarrier(),
    )
    key = ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id="generation",
        representation="accumulator",
        accumulator_revision=1,
        topology_fingerprint="topology",
        adapter_layout_fingerprint="adapter",
    )
    source = torch.ones(8 << 20, dtype=torch.float32, device="cuda")
    l2 = manager.register_l1(key, (source,))

    manager.acquire_l1(key)
    manager.wait_before_mutation_working_set((key,))
    source.fill_(2)
    manager.release_l1(key)

    first_image = l2.result(timeout=10)
    assert torch.all(first_image.tensors()[0] == 1)

    manager.ensure_l3(key).result(timeout=10)
    manager.evict_l2(key)
    second_l2 = manager.ensure_l2(key)
    manager.acquire_l1(key)
    manager.wait_before_mutation_working_set((key,))
    source.fill_(3)
    manager.release_l1(key)

    second_image = second_l2.result(timeout=10)
    assert torch.all(second_image.tensors()[0] == 2)
    torch.cuda.synchronize()
    assert torch.all(source == 3)
    manager.close()
