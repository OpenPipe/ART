from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any

import pytest

from art.distributed import monarch_actor
from art.distributed.art_runtime import DistributedMoeRoutePrefetch
from art.distributed.host_admission import HostAdmissionRequest
from art.distributed.moe_route_store import (
    MoeRouteObjectBatchTransfer,
    MoeRoutePrefetchRef,
    MoeRouteSlice,
    MoeRouteStoredObject,
)
from art.distributed.monarch_actor import ArtHostService
from art.distributed.object_store import S3ObjectStoreConfig
from art.distributed.packing import PackingRequest
from art.distributed.specs import ArtRuntimeConfig
from art.megatron.training.slot import MegatronTrainingSlot
from art.training.contracts import OperationRef


def _transfer(group_count: int = 1) -> MoeRouteObjectBatchTransfer:
    return MoeRouteObjectBatchTransfer(
        tenant_id="tenant",
        run_id="run",
        store=S3ObjectStoreConfig(
            endpoint_url="https://objects.example.test",
            region="test",
            bucket="routes",
            prefix="training/routes",
        ),
        groups=tuple(
            (
                MoeRouteStoredObject(
                    object_id=f"{index + 1:064x}",
                    byte_count=1,
                    slices=(
                        MoeRouteSlice(
                            trajectory_index=index,
                            scope="exchange",
                            scope_index=0,
                            choice_index=0,
                            offset=0,
                            byte_count=1,
                        ),
                    ),
                ),
            )
            for index in range(group_count)
        ),
    )


def _actor() -> ArtHostService:
    actor = object.__new__(ArtHostService)
    actor._closing = False
    actor._moe_route_prefetches = {}
    actor._moe_route_receive_slots = asyncio.Semaphore(16)
    actor._moe_route_receive_executor = None
    actor._moe_route_receiver = lambda _transfer: object()
    return actor


def test_actor_uses_configured_route_receive_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Inbox:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr(monarch_actor, "PackedBatchInbox", Inbox)
    actor = object.__new__(ArtHostService)
    config = ArtRuntimeConfig()
    actor.__init__(
        HostAdmissionRequest(
            host_id="host",
            node_rank=0,
            expected_gpu_ids=(),
            runtime_packages=(),
        ).model_dump_json(),
        1,
        data_plane_host="127.0.0.1",
        moe_route_receive_concurrency=config.moe_route_receive_concurrency,
    )
    assert actor._moe_route_receive_slots._value == 16
    assert actor._moe_route_receive_executor._max_workers == 16
    actor._moe_route_receive_executor.shutdown()


@pytest.mark.asyncio
async def test_route_receive_opens_the_measured_sixteen_way_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = _transfer(16)
    all_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def receive(_self: object, _receiver: object, _value: object) -> memoryview:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 16:
                all_started.set()
        assert release.wait(2)
        with lock:
            active -= 1
        return memoryview(bytearray(b"x")).toreadonly()

    monkeypatch.setattr(MoeRouteObjectBatchTransfer, "_receive", receive)
    executor = ThreadPoolExecutor(max_workers=16)
    task = asyncio.create_task(
        transfer.receive_groups(
            object(), asyncio.Semaphore(16), timeout_s=5, executor=executor
        )
    )
    try:
        assert await asyncio.to_thread(all_started.wait, 1)
        assert peak == 16
        release.set()
        assert len(await task) == 16
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        executor.shutdown()


async def _start_prefetch(
    actor: ArtHostService,
    transfer: MoeRouteObjectBatchTransfer,
    prefetch: MoeRoutePrefetchRef,
    *,
    batch_id: str = "batch",
    generation_id: str = "generation",
) -> None:
    start = ArtHostService.__dict__["prefetch_moe_routes"]._method.__wrapped__
    await start(actor, transfer, prefetch, batch_id, generation_id, 5.0)


@pytest.mark.asyncio
async def test_prefetched_routes_are_received_and_consumed_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _actor()
    transfer = _transfer()
    prefetch = MoeRoutePrefetchRef(prefetch_id="prefetch", group_count=1)
    started = threading.Event()
    release = threading.Event()
    get_calls = 0

    def receive(_self: object, _receiver: object, _value: object) -> memoryview:
        nonlocal get_calls
        get_calls += 1
        started.set()
        assert release.wait(2)
        return memoryview(bytearray(b"x")).toreadonly()

    monkeypatch.setattr(MoeRouteObjectBatchTransfer, "_receive", receive)
    await _start_prefetch(actor, transfer, prefetch)
    assert await asyncio.to_thread(started.wait, 1)
    request = PackingRequest.model_construct(
        generation_id="generation",
        moe_route_groups=(),
        moe_route_transfer=None,
        moe_route_object_transfer=None,
        moe_route_prefetch=prefetch,
    )
    receive_task = asyncio.create_task(
        actor._receive_moe_route_groups(request, "batch", 5.0)
    )
    release.set()
    groups = await asyncio.wait_for(receive_task, 1)

    assert len(groups) == 1
    assert bytes(groups[0].objects[0].data) == b"x"
    assert get_calls == 1
    with pytest.raises(ValueError, match="already-consumed"):
        await actor._consume_moe_route_prefetch(
            prefetch, batch_id="batch", generation_id="generation"
        )
    assert get_calls == 1


@pytest.mark.asyncio
async def test_prefetch_identity_mismatch_cancels_receive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _actor()
    transfer = _transfer()
    prefetch = MoeRoutePrefetchRef(prefetch_id="prefetch", group_count=1)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def receive(*_args: object, **_kwargs: object) -> tuple[()]:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return ()

    monkeypatch.setattr(MoeRouteObjectBatchTransfer, "receive_groups", receive)
    await _start_prefetch(actor, transfer, prefetch)
    await asyncio.wait_for(started.wait(), 1)
    with pytest.raises(ValueError, match="identity changed"):
        await actor._consume_moe_route_prefetch(
            prefetch, batch_id="batch", generation_id="wrong"
        )
    await asyncio.wait_for(cancelled.wait(), 1)
    assert not actor._moe_route_prefetches


class _Runtime:
    def __init__(self) -> None:
        self.prefetches: list[DistributedMoeRoutePrefetch] = []
        self.discarded: list[DistributedMoeRoutePrefetch] = []

    async def prefetch_moe_routes(
        self, transfer: MoeRouteObjectBatchTransfer, *, generation_id: str
    ) -> DistributedMoeRoutePrefetch:
        value = DistributedMoeRoutePrefetch(
            prefetch_id=f"prefetch-{len(self.prefetches)}",
            batch_id=f"batch-{len(self.prefetches)}",
            generation_id=generation_id,
            source_host="host",
            trainer_hosts=("host",),
            transfer=transfer,
        )
        self.prefetches.append(value)
        return value

    async def discard_moe_route_prefetch(
        self, value: DistributedMoeRoutePrefetch
    ) -> None:
        self.discarded.append(value)


def _ref(index: int) -> OperationRef:
    return OperationRef(
        run_id="run",
        operation_id=f"operation-{index}",
        sequence_id=index,
        learner_parent_version=0,
        kind="forward_backward",
    )


def _slot(runtime: _Runtime) -> MegatronTrainingSlot:
    slot = object.__new__(MegatronTrainingSlot)
    slot.runtime = runtime
    slot._moe_route_prefetches = {}
    slot._require_run = lambda _run_id: object()
    return slot


@pytest.mark.asyncio
async def test_packing_failure_reclaims_prefetch() -> None:
    runtime = _Runtime()
    slot = _slot(runtime)
    ref = _ref(0)
    prefetch = await slot.prefetch_forward_moe_routes(ref, _transfer())

    async def fail(*_args: object) -> Any:
        raise RuntimeError("packing failed")

    slot._prepare_forward_packing_owned = fail
    with pytest.raises(RuntimeError, match="packing failed"):
        await slot.prepare_forward_packing(ref, object(), moe_route_prefetch=prefetch)

    assert runtime.discarded == [prefetch.distributed]
    assert not slot._moe_route_prefetches


@pytest.mark.asyncio
async def test_wrong_operation_prefetch_is_reclaimed() -> None:
    runtime = _Runtime()
    slot = _slot(runtime)
    prefetch = await slot.prefetch_forward_moe_routes(_ref(0), _transfer())

    with pytest.raises(RuntimeError, match="does not own"):
        await slot.prepare_forward_packing(
            _ref(1), object(), moe_route_prefetch=prefetch
        )

    assert runtime.discarded == [prefetch.distributed]
    assert not slot._moe_route_prefetches


@pytest.mark.asyncio
async def test_slot_close_reclaims_every_unconsumed_prefetch() -> None:
    runtime = _Runtime()
    slot = _slot(runtime)
    prefetches = [
        await slot.prefetch_forward_moe_routes(_ref(index), _transfer())
        for index in range(2)
    ]

    class Trainer:
        valid = True

        async def close(self) -> None:
            pass

    slot.trainer = Trainer()
    slot._pending_results = {}
    slot._batch_releases = set()
    slot._batch_release_failures = []
    slot._batch_release_leases = {}
    slot._prepared_saves = {}
    slot._closed = False
    await slot.close()

    assert runtime.discarded == [value.distributed for value in prefetches]
    assert slot._closed
