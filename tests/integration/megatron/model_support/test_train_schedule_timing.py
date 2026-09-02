import inspect
from types import SimpleNamespace
from typing import Any, cast

from art.megatron import train


class _Schedule:
    def __init__(self) -> None:
        self.telemetry = SimpleNamespace(metrics=lambda: {"time/schedule_s": 1.25})

    def run(
        self,
        forward_step_func: Any,
        *,
        forward_only: bool,
        collect_non_loss_data: bool,
    ) -> list[str]:
        assert callable(forward_step_func) and forward_only is False
        assert collect_non_loss_data is False
        return ["output"]


def test_training_schedule_does_not_create_or_gather_profiler_state(
    monkeypatch,
) -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("production schedule attempted a profiler collective")

    monkeypatch.setattr(train.torch.distributed, "new_group", unexpected)
    monkeypatch.setattr(train.torch.distributed, "all_gather", unexpected)

    assert train._run_training_schedule(
        cast(Any, _Schedule()),
        lambda: None,
    ) == ["output"]
    assert "inter_forward_backward_timing" not in train.TrainingRuntime.model_fields
    assert "new_group" not in inspect.getsource(train.build_training_runtime)
    assert "all_gather" not in inspect.getsource(train._run_training_schedule)
