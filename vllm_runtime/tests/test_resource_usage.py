from concurrent.futures import Future
from types import SimpleNamespace

from art_vllm_runtime.resource_usage import (
    GPUServiceTracker,
    KVUsageOwner,
    PhysicalKVTracker,
    _account_future,
    _GPUUsageBatchTracker,
)


def _scheduled(**weights: int) -> SimpleNamespace:
    return SimpleNamespace(
        num_scheduled_tokens=weights,
        total_num_scheduled_tokens=sum(weights.values()),
    )


def test_gpu_service_conserves_overlapping_world_time() -> None:
    timestamps = iter((0, 10, 20, 30))
    tracker = GPUServiceTracker(2, monotonic_ns=lambda: next(timestamps))
    first = _scheduled(a=1, b=1)
    second = _scheduled(c=2)

    first_id = tracker.start(first)
    second_id = tracker.start(second)
    tracker.finish(first_id, first)
    tracker.finish(second_id, second)

    assert first._art_gpu_allocations == {"a": 15, "b": 15}
    assert second._art_gpu_allocations == {"c": 30}
    assert (
        sum(first._art_gpu_allocations.values())
        + sum(second._art_gpu_allocations.values())
        == 60
    )


def test_physical_kv_counts_cached_blocks_once_until_eviction() -> None:
    timestamps = iter((10, 20, 30))
    config = SimpleNamespace(
        num_blocks=2,
        _art_physical_kv_bytes_per_block=64,
    )
    tracker = PhysicalKVTracker(config, monotonic_ns=lambda: next(timestamps))
    owner = KVUsageOwner("tenant", "run", "standard", "model")
    blocks = [
        SimpleNamespace(block_id=0, is_null=False, ref_cnt=1, block_hash=None),
        SimpleNamespace(block_id=1, is_null=False, ref_cnt=1, block_hash="cached"),
    ]

    with tracker.batch():
        tracker.assign(blocks, owner)
        tracker.assign(blocks, owner)
    blocks[0].ref_cnt = 0
    tracker.release_unresident(blocks)
    blocks[1].ref_cnt = 0
    blocks[1].block_hash = None
    tracker.release_unresident(blocks)

    assert [update["byte_count"] for update in tracker.take_updates()] == [128, 64, 0]


def test_gpu_usage_completes_after_all_queued_batches_drain() -> None:
    tracker = _GPUUsageBatchTracker()
    owners = {"engine-request": object()}
    tracker.register(owners)
    tracker.register(owners)

    assert tracker.finish(owners, {}) == set()
    assert tracker.finish(owners, {}) == {"engine-request"}


def test_accounting_future_drives_lazy_vllm_future() -> None:
    class LazyFuture(Future[str]):
        def result(self, timeout: float | None = None) -> str:
            if not self.done():
                self.set_result("worker-output")
            return super().result(timeout)

    completed = _account_future(
        LazyFuture(),
        lambda value: f"{value}-accounted",
        lambda: None,
    )

    assert completed.result() == "worker-output-accounted"


def test_accounting_future_finishes_when_source_is_consumed_elsewhere() -> None:
    class LazyFuture(Future[str]):
        def result(self, timeout: float | None = None) -> str:
            if not self.done():
                self.set_result("worker-output")
            return super().result(timeout)

    source = LazyFuture()
    completed = _account_future(
        source,
        lambda value: f"{value}-accounted",
        lambda: None,
    )

    assert source.result() == "worker-output"
    assert completed.done()
    assert completed.result() == "worker-output-accounted"
