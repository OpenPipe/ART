from pathlib import Path
from typing import Literal

import pytest
import torch

from art.megatron.runtime.nvme_residency import NvmeResidencyStoreConfig
from art.megatron.runtime.residency import (
    ResidencyCapacityUnavailable,
    ResidencyDemand,
    ResidencyKey,
    ResidencyLedger,
    ResidencyLimits,
    TierCapacity,
)
from art.megatron.runtime.run_residency import (
    RunResidencyConfig,
    RunResidencyManager,
)
from art.megatron.runtime.tensor_residency import TensorResidencyTransition
from art.megatron.tensor_snapshot import SnapshotReadBarrier
from art.utils.disk_admission import DiskAdmissionConfig

GIB = 1 << 30


def key(
    representation: Literal["weights", "optimizer", "accumulator", "sampler"],
    *,
    generation: str = "generation",
    revision: int = 0,
) -> ResidencyKey:
    return ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id=generation,
        representation=representation,
        accumulator_revision=revision,
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )


def manager(
    root: Path,
    *,
    l1_bytes: int = 1 << 20,
    l2_bytes: int = 1 << 20,
) -> RunResidencyManager:
    return RunResidencyManager(
        RunResidencyConfig(
            limits=ResidencyLimits(
                l1_gpu=TierCapacity(max_bytes=l1_bytes),
                l2_cpu=TierCapacity(max_bytes=l2_bytes),
                l3_nvme=TierCapacity(max_bytes=max(l1_bytes, l2_bytes, 1 << 20)),
                max_concurrent_transitions=2,
            ),
            nvme=NvmeResidencyStoreConfig(
                root=str(root),
                disk_admission=DiskAdmissionConfig(
                    shared_storage_mount=root.parent,
                    storage_identity=f"test-{root.parent.name}",
                    node_identity="test-node",
                    runtime_free_floor_bytes=0,
                ),
            ),
            device="cuda:0",
        ),
        snapshot_barrier=SnapshotReadBarrier(),
    )


def test_14_771_gib_demand_above_high_watermark_reclaims_only_cache() -> None:
    total = 14_771 * GIB // 1000
    sizes = (5 * GIB, 8 * GIB, total - 13 * GIB)
    keys = (
        key("weights"),
        key("optimizer"),
        key("accumulator", revision=1),
    )
    cache = key("weights", generation="unrelated")
    ledger = ResidencyLedger(
        ResidencyLimits(
            l1_gpu=TierCapacity(max_bytes=16 * GIB),
            l2_cpu=TierCapacity(max_bytes=32 * GIB),
            l3_nvme=TierCapacity(max_bytes=32 * GIB),
        )
    )
    l2 = ledger.reserve_many(
        tuple(
            ResidencyDemand(
                key=item, source=None, target="l2_cpu", byte_count=byte_count
            )
            for item, byte_count in (*zip(keys, sizes, strict=True), (cache, GIB))
        )
    )
    ledger.commit_many(l2)
    cache_l1 = ledger.reserve(cache, source="l2_cpu", target="l1_gpu", byte_count=GIB)
    ledger.commit(cache_l1)

    assert total > ledger.limits.l1_gpu.high_bytes
    assert ledger.admission_evictions("l1_gpu", total, total, protected=keys) == (
        cache,
    )
    ledger.drop(cache, "l1_gpu")
    reservations = ledger.reserve_many(
        tuple(
            ResidencyDemand(
                key=item,
                source="l2_cpu",
                target="l1_gpu",
                byte_count=byte_count,
            )
            for item, byte_count in zip(keys, sizes, strict=True)
        )
    )

    assert sum(item.byte_count for item in reservations) == total
    assert ledger.usage().reserved_bytes["l1_gpu"] == total
    ledger.commit_many(reservations)
    assert ledger.usage().ready_bytes["l1_gpu"] == total


def test_over_max_working_set_fails_without_partial_l1_admission(
    tmp_path: Path,
) -> None:
    residency = manager(tmp_path, l1_bytes=1000, l2_bytes=4096)
    keys = (
        key("weights"),
        key("optimizer"),
        key("accumulator", revision=1),
    )
    tensors = (
        torch.empty(400, dtype=torch.uint8),
        torch.empty(400, dtype=torch.uint8),
        torch.empty(201, dtype=torch.uint8),
    )
    for item, tensor in zip(keys, tensors, strict=True):
        residency.register_l2(item, (tensor,))

    with pytest.raises(ResidencyCapacityUnavailable, match="demanded=1001, max=1000"):
        residency.prefetch_l1_working_set(keys)

    assert residency.ledger.usage().ready_bytes["l1_gpu"] == 0
    assert residency.ledger.usage().reserved_bytes["l1_gpu"] == 0
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    residency.close()


def test_grouped_prefetch_and_acquire_cover_complete_working_set(
    tmp_path: Path,
) -> None:
    residency = manager(tmp_path, l1_bytes=1024, l2_bytes=4096)
    keys = (
        key("weights"),
        key("optimizer"),
        key("accumulator", revision=1),
    )
    tensors = (
        torch.empty(400, dtype=torch.uint8),
        torch.empty(400, dtype=torch.uint8),
        torch.empty(144, dtype=torch.uint8),
    )
    for item, tensor in zip(keys, tensors, strict=True):
        residency.register_l2(item, (tensor,))

    residency.prefetch_l1_working_set(keys)

    transitions = tuple(residency._state(item).l1_transition for item in keys)
    assert transitions[0] is not None
    assert all(transition is transitions[0] for transition in transitions)
    assert residency.ledger.usage().ready_bytes["l1_gpu"] == 944
    residency.acquire_l1_working_set(keys)
    assert all(residency.ledger.entry(item).pin_counts["l1_gpu"] == 1 for item in keys)
    residency.release_l1_working_set(keys)
    residency.close()


def test_advance_defers_retirement_for_overlapping_l2_save_lease(
    tmp_path: Path,
) -> None:
    residency = manager(tmp_path)
    source = key("weights", generation="parent")
    target = key("weights", generation="child")
    tensor = torch.arange(64, dtype=torch.float32, device="cuda:0")
    residency.register_l1(source, (tensor,)).result()

    with residency.borrow_l2(source) as image:
        target_l2 = residency.advance_l1(source, target, (tensor,), retire_source=True)
        retirement = residency.retire_async(source)
        assert not retirement.done()
        assert retirement not in residency._futures
        assert residency.ledger.has_copy(source, "l2_cpu")
        assert torch.equal(image.tensors()[0], torch.arange(64, dtype=torch.float32))
        target_l2.result()

    retirement.result(timeout=2)
    assert source not in residency.keys("run")
    assert residency.ledger.has_copy(target, "l1_gpu")
    residency.close()


def test_retiring_an_unacquired_load_releases_transition_and_gpu_copy(
    tmp_path: Path,
) -> None:
    residency = manager(tmp_path)
    state = key("weights", generation="abandoned-load")
    tensor = torch.nn.Parameter(torch.arange(64, dtype=torch.float32))
    residency.register_l2(state, (tensor,))
    residency.prefetch_l1(state)

    retirement = residency.retire_async(state)

    retirement.result(timeout=2)
    assert tensor.device.type == "cpu"
    assert residency.keys("run") == ()
    assert residency.ledger.entries() == ()
    assert residency._active_transitions == 0
    residency.close()


def test_prefetch_retains_event_without_cpu_synchronize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    residency = manager(tmp_path)
    state = key("weights", generation="prefetch")
    tensor = torch.nn.Parameter(torch.arange(64, dtype=torch.float32))
    residency.register_l2(state, (tensor,))
    waits: list[TensorResidencyTransition] = []
    original_wait = TensorResidencyTransition.wait_on_current_stream

    def fail_synchronize(_transition: TensorResidencyTransition) -> None:
        raise AssertionError("prefetch synchronized the CPU")

    def record_wait(transition: TensorResidencyTransition) -> None:
        waits.append(transition)
        original_wait(transition)

    monkeypatch.setattr(TensorResidencyTransition, "synchronize", fail_synchronize)
    monkeypatch.setattr(
        TensorResidencyTransition, "wait_on_current_stream", record_wait
    )

    residency.prefetch_l1(state)
    transition = residency._state(state).l1_transition
    assert transition is not None
    assert waits == []

    residency.acquire_l1(state)
    assert waits == [transition]
    assert residency._state(state).l1_transition is None
    residency.release_l1(state)
    residency.close()


def test_l2_working_set_failure_commits_no_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    residency = manager(tmp_path)
    keys = (key("weights"), key("optimizer"))
    tensors = (torch.arange(8, dtype=torch.uint8), torch.arange(12, dtype=torch.uint8))
    source_ptrs = tuple(tensor.data_ptr() for tensor in tensors)
    adopt_host_image = residency._mover.adopt_host_image
    calls = 0

    def fail_second(component: tuple[torch.Tensor, ...]):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced second-component failure")
        return adopt_host_image(component)

    monkeypatch.setattr(residency._mover, "adopt_host_image", fail_second)

    with pytest.raises(RuntimeError, match="forced second-component failure"):
        residency.register_l2_working_set(zip(keys, ((tensors[0],), (tensors[1],))))

    assert residency.keys("run") == ()
    assert residency.ledger.entries() == ()
    assert residency._mutation_barriers == {}
    assert residency.ledger.usage().ready_bytes["l2_cpu"] == 0
    assert residency.ledger.usage().reserved_bytes["l2_cpu"] == 0
    assert residency._active_transitions == 0
    assert tuple(tensor.data_ptr() for tensor in tensors) == source_ptrs
    with pytest.raises(RuntimeError, match="forced second-component failure"):
        residency.close()


def test_l2_working_set_adopts_and_commits_every_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    residency = manager(tmp_path)
    keys = (key("weights"), key("optimizer"))
    tensors = (torch.arange(8, dtype=torch.uint8), torch.arange(12, dtype=torch.uint8))
    source_ptrs = tuple(tensor.data_ptr() for tensor in tensors)

    def reject_copy(*_args, **_kwargs):
        raise AssertionError("L2 registration copied prepared CPU storage")

    monkeypatch.setattr(torch, "empty_like", reject_copy)
    monkeypatch.setattr(torch.Tensor, "copy_", reject_copy)

    images = residency.register_l2_working_set(
        zip(keys, ((tensors[0],), (tensors[1],)))
    )

    assert len(images) == 2
    assert set(residency.keys("run")) == set(keys)
    assert all(residency._state(item).l2 is image for item, image in zip(keys, images))
    assert all(residency.ledger.has_copy(item, "l2_cpu") for item in keys)
    assert set(residency._mutation_barriers) == set(keys)
    assert residency.ledger.usage().ready_bytes["l2_cpu"] == 20
    assert residency.ledger.usage().reserved_bytes["l2_cpu"] == 0
    assert tuple(image.storage_bytes()[0].data_ptr() for image in images) == source_ptrs
    assert tuple(tensor.data_ptr() for tensor in tensors) == source_ptrs
    residency.close()


def test_l2_working_set_validates_every_component_before_admission(
    tmp_path: Path,
) -> None:
    residency = manager(tmp_path)
    weights = key("weights")
    optimizer = key("optimizer")
    tensor = torch.ones(1)

    with pytest.raises(RuntimeError, match="non-empty CPU tensor tuple"):
        residency.register_l2_working_set(
            ((weights, (tensor,)), (optimizer, (torch.ones(1, device="meta"),)))
        )
    with pytest.raises(ValueError, match="keys must be unique"):
        residency.register_l2_working_set(
            ((weights, (tensor,)), (weights, (tensor.clone(),)))
        )

    assert residency.keys("run") == ()
    assert residency.ledger.entries() == ()
    assert residency._active_transitions == 0
    residency.close()
