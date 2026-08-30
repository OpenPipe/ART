from types import SimpleNamespace

import pytest

from art.megatron.gate_evidence import MegatronGateAttemptPlan, MegatronGateTurn


def test_gate_schedule_allows_serial_runs_without_isolation_capture() -> None:
    runs = tuple(
        SimpleNamespace(
            bootstrap=SimpleNamespace(run_id=run_id),
            commands=(SimpleNamespace(kind="optim_step"),),
        )
        for run_id in ("run-1", "run-2")
    )
    capture_free = MegatronGateAttemptPlan.model_construct(
        slot=None,
        attempt_root="/tmp/gate",
        runs=runs,
        schedule=tuple(
            MegatronGateTurn(run_id=run_id, command_count=1)
            for run_id in ("run-1", "run-2")
        ),
    )
    assert capture_free._validate_runs() is capture_free

    partial_capture = capture_free.model_copy(
        update={
            "schedule": (
                MegatronGateTurn(
                    run_id="run-1", command_count=1, capture_isolation=True
                ),
                MegatronGateTurn(run_id="run-2", command_count=1),
            )
        }
    )
    with pytest.raises(ValueError, match="capture an active turn for every run"):
        partial_capture._validate_runs()
