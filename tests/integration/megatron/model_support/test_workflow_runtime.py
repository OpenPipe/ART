import pytest

from .workflow_runtime import (
    WorkflowDevice,
    WorkflowOperation,
    WorkflowPlan,
    WorkflowResourceRequest,
    WorkflowRuntimeKey,
    WorkflowSession,
    _GpuPool,
    compile_workflow,
    execute_workflow,
)


def _devices(hosts: int = 1, gpus: int = 8) -> list[WorkflowDevice]:
    return [
        WorkflowDevice(host=f"host-{host}", gpu=str(gpu))
        for host in range(hosts)
        for gpu in range(gpus)
    ]


def _runtime(handler: str = "handler") -> WorkflowRuntimeKey:
    return WorkflowRuntimeKey(
        source_fingerprint="source",
        handler=handler,
        fixture="fixture",
        kind="megatron",
    )


def _session(
    session_id: str,
    *,
    gpu_count: int,
    gpu_share: float = 1.0,
    estimated_duration_s: float = 1.0,
) -> WorkflowSession:
    operation = WorkflowOperation(
        id=session_id,
        stage=session_id,
        runtime=_runtime(session_id),
        resources=WorkflowResourceRequest(gpu_count=gpu_count, gpu_share=gpu_share),
        estimated_duration_s=estimated_duration_s,
    )
    return WorkflowSession(
        id=session_id,
        runtime=operation.runtime,
        operations=(operation,),
        resources=operation.resources,
        estimated_duration_s=estimated_duration_s,
    )


def test_fractional_placements_pack_per_host_and_leave_full_width_capacity() -> None:
    pool = _GpuPool(_devices(hosts=3))
    fractional = WorkflowResourceRequest(gpu_count=1, gpu_share=0.125)

    placements = [pool.acquire(fractional) for _ in range(24)]

    assert all(placement is not None for placement in placements)
    assert [placement.host for placement in placements if placement is not None] == [
        f"host-{index % 3}" for index in range(24)
    ]
    assert {
        (placement.host, placement.devices[0].gpu)
        for placement in placements
        if placement is not None
    } == {(f"host-{index}", "0") for index in range(3)}
    full_width = pool.acquire(WorkflowResourceRequest(gpu_count=4))
    assert full_width is not None
    assert tuple(device.gpu for device in full_width.devices) == ("1", "2", "3", "4")


def test_ready_full_width_session_launches_before_fractional_backfill() -> None:
    sessions = (
        _session("fractional", gpu_count=1, gpu_share=0.125, estimated_duration_s=100),
        _session("exclusive-one", gpu_count=1, estimated_duration_s=100),
        _session("full", gpu_count=4, estimated_duration_s=1),
    )
    execution = execute_workflow(
        WorkflowPlan(sessions=sessions),
        devices=_devices(),
        runner=lambda _session, _placement: None,
    )

    assert (
        execution.results["full"].started_monotonic_s
        < execution.results["exclusive-one"].started_monotonic_s
        < execution.results["fractional"].started_monotonic_s
    )
    assert tuple(
        device.gpu for device in execution.results["full"].placement.devices
    ) == ("0", "1", "2", "3")
    assert execution.results["exclusive-one"].placement.devices[0].gpu == "4"
    assert execution.results["fractional"].placement.devices[0].gpu == "5"


def test_cpu_sessions_are_distributed_across_explicit_hosts() -> None:
    sessions = tuple(_session(f"cpu-{index}", gpu_count=0) for index in range(3))
    execution = execute_workflow(
        WorkflowPlan(sessions=sessions),
        devices=_devices(hosts=3),
        runner=lambda _session, placement: placement.host,
    )

    assert [execution.results[f"cpu-{index}"].output for index in range(3)] == [
        "host-0",
        "host-1",
        "host-2",
    ]


def test_distinct_host_affinities_balance_and_remain_pinned() -> None:
    pool = _GpuPool(_devices(hosts=3))
    full = WorkflowResourceRequest(gpu_count=8)
    occupied = []
    for _ in range(3):
        placement = pool.acquire(full)
        assert placement is not None
        occupied.append(placement)
    pool.release(occupied[0], full)
    requests = [
        WorkflowResourceRequest(gpu_count=1, host_affinity=f"model-{index}")
        for index in range(6)
    ]
    hosts = []
    for request in requests:
        placement = pool.acquire(request)
        hosts.append(placement.host if placement is not None else None)
        if placement is not None:
            pool.release(placement, request)

    assert hosts == ["host-0", None, None, "host-0", None, None]
    pool.release(occupied[1], full)
    pool.release(occupied[2], full)
    for request, host in zip(requests, ("host-0", "host-1", "host-2") * 2, strict=True):
        variant_request = request.model_copy(update={"gpu_count": 8})
        placement = pool.acquire(variant_request)
        assert placement is not None
        assert placement.host == host
        pool.release(placement, variant_request)


def test_compiled_session_counts_shared_startup_once() -> None:
    runtime = _runtime()
    operations = tuple(
        WorkflowOperation(
            id=f"operation-{index}",
            stage=f"stage-{index}",
            runtime=runtime,
            estimated_duration_s=duration,
            estimated_shared_startup_s=startup,
        )
        for index, (duration, startup) in enumerate(
            ((60.0, 0.0), (360.0, 200.0), (360.0, 200.0))
        )
    )

    plan = compile_workflow(operations)

    assert plan.sessions[0].estimated_duration_s == pytest.approx(580.0)
