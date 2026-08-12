from types import SimpleNamespace
from typing import Any, cast

from art.megatron import train


class _Schedule:
    def __init__(self, spans: tuple[tuple[Any, Any], ...] = ()) -> None:
        self._spans = iter(spans)
        self._span = None
        self.telemetry = SimpleNamespace(cuda_span=lambda: self._span)

    def run(self, forward_step_func: Any, *, forward_only: bool) -> list[str]:
        assert callable(forward_step_func) and forward_only is False
        self._span = next(self._spans, None)
        return ["output"]


class _CudaEvent:
    def __init__(self, timestamp_ms: float) -> None:
        self.timestamp_ms = timestamp_ms

    def elapsed_time(self, end: "_CudaEvent") -> float:
        return end.timestamp_ms - self.timestamp_ms


def test_inter_forward_backward_timing_uses_rank_local_monotonic_boundaries(
    monkeypatch,
) -> None:
    timestamps = iter((10.0, 12.0, 15.0, 17.0))
    monkeypatch.setattr(train.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(train.torch.distributed, "get_world_size", lambda: 1)
    timing = train._InterForwardBackwardTiming()
    schedule = cast(
        Any,
        _Schedule(
            (
                (_CudaEvent(100.0), _CudaEvent(200.0)),
                (_CudaEvent(260.0), _CudaEvent(400.0)),
            )
        ),
    )

    first, collect_first_metrics = train._run_training_schedule(
        schedule,
        lambda: None,
        timing,
    )
    second, collect_second_metrics = train._run_training_schedule(
        schedule,
        lambda: None,
        timing,
    )

    assert first == second == ["output"]
    assert collect_first_metrics() == {}
    assert collect_second_metrics() == {
        "time/inter_forward_backward_gap_rank_0_s": 3.0,
        "time/inter_forward_backward_gpu_gap_rank_0_s": 0.06,
    }
    assert timing.previous_schedule_end_s == 17.0


def test_inter_forward_backward_timing_gathers_rank_durations_on_cpu_group(
    monkeypatch,
) -> None:
    timestamps = iter((1.0, 2.0, 4.0, 5.0))
    group = SimpleNamespace(backend="gloo")
    timing = train._InterForwardBackwardTiming(metrics_group=group)
    monkeypatch.setattr(train.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(train.torch.distributed, "get_world_size", lambda: 2)

    waits = []

    def gather(output, value, *, group, async_op):
        assert group is timing.metrics_group
        assert async_op is True and value.device.type == "cpu"
        output[0].copy_(value)
        output[1].copy_(value + 0.25)
        return SimpleNamespace(wait=lambda: waits.append(True))

    monkeypatch.setattr(train.torch.distributed, "all_gather", gather)
    schedule = cast(Any, _Schedule())
    train._run_training_schedule(schedule, lambda: None, timing)
    _, collect_metrics = train._run_training_schedule(
        schedule,
        lambda: None,
        timing,
    )

    assert waits == []
    assert collect_metrics() == {
        "time/inter_forward_backward_gap_rank_0_s": 2.0,
        "time/inter_forward_backward_gap_rank_1_s": 2.25,
    }
    assert waits == [True]
