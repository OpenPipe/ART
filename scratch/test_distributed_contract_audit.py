from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from art.distributed.art_runtime import DistributedPackedBatch
from art.distributed.monarch_actor import _RetainedPackingPlan
from art.distributed.monarch_runtime import _validate_remote_model
from art.distributed.packing import PackingResult
from art.local.backend import PreparedPackedTensors
from art.megatron.training.command_telemetry import (
    RANK_STATISTIC_COUNT,
    RankCommandTelemetry,
    RankTelemetryTopology,
)
from art.megatron.training.commands import packing_metrics
from art.megatron.training.workload import TrainingStepWorkload
from art.preprocessing.pack import PackingTimings
from art.training.contracts import ForwardBackwardResult, PackingOutcome

ROOT = Path(__file__).parents[1]


def _constructor_keywords(path: str, constructor: str) -> tuple[frozenset[str], ...]:
    tree = ast.parse((ROOT / path).read_text())
    return tuple(
        frozenset(keyword.arg for keyword in node.keywords if keyword.arg is not None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == constructor)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == constructor)
        )
    )


@pytest.mark.parametrize(
    ("model", "path", "constructor", "expected_calls"),
    (
        (
            PreparedPackedTensors,
            "src/art/local/backend.py",
            "PreparedPackedTensors",
            1,
        ),
        (
            PreparedPackedTensors,
            "src/art/distributed/monarch_actor.py",
            "PreparedPackedTensors",
            1,
        ),
        (
            _RetainedPackingPlan,
            "src/art/distributed/monarch_actor.py",
            "_RetainedPackingPlan",
            1,
        ),
        (
            PackingResult,
            "src/art/distributed/monarch_actor.py",
            "PackingResult",
            2,
        ),
        (
            DistributedPackedBatch,
            "src/art/distributed/art_runtime.py",
            "DistributedPackedBatch",
            1,
        ),
    ),
)
def test_required_packing_contract_fields_are_explicitly_propagated(
    model: type,
    path: str,
    constructor: str,
    expected_calls: int,
) -> None:
    assert model.model_fields["num_dropped_trajectories"].is_required()
    calls = _constructor_keywords(path, constructor)
    assert len(calls) == expected_calls
    required = {
        name for name, field in model.model_fields.items() if field.is_required()
    }
    assert all(required <= keywords for keywords in calls)


def test_packing_result_preserves_drop_count_across_wire_roundtrip() -> None:
    assert all(field.is_required() for field in PackingResult.model_fields.values())
    assert all(
        field.is_required() for field in DistributedPackedBatch.model_fields.values()
    )
    result = PackingResult(
        ref=None,
        packed_group_shapes=(),
        trainable_assistant_tokens=0,
        loss_bearing_tokens=0,
        non_padding_tokens=0,
        num_dropped_trajectories=7,
        trajectory_log_path=None,
        trajectory_fetch_s=0.0,
        trajectory_receive_s=0.0,
        trajectory_build_s=0.0,
        packing_core_s=0.0,
        packing_lock_wait_s=0.0,
        packing_compute_s=0.0,
        packing_timings=PackingTimings(),
        trajectory_log_wait_s=0.0,
        packed_batch_finalize_s=0.0,
        generation_id="generation",
    )
    assert PackingResult.model_validate_json(result.model_dump_json()) == result
    assert _validate_remote_model(PackingResult, result) == result
    with pytest.raises(ValidationError):
        _validate_remote_model(
            PackingResult,
            {
                key: value
                for key, value in result.model_dump(mode="python").items()
                if key != "num_dropped_trajectories"
            },
        )


def test_canonical_packing_metrics_reports_drop_count() -> None:
    packed = SimpleNamespace(
        num_dropped_trajectories=7,
        trajectory_fetch_s=0.1,
        trajectory_receive_s=0.2,
        trajectory_build_s=0.3,
        packing_core_s=0.4,
        packing_lock_wait_s=0.5,
        packing_compute_s=0.6,
        packing_timings=PackingTimings(),
        trajectory_log_wait_s=0.7,
        packed_batch_finalize_s=0.8,
        packing_rpc_s=0.9,
        packed_batch_fanout_s=1.0,
    )
    assert packing_metrics(packed)["data/step_num_dropped_trajectories"] == 7.0
    backend_tree = ast.parse((ROOT / "src/art/megatron/backend.py").read_text())
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_packing_metrics"
        for node in ast.walk(backend_tree)
    )


def _workload() -> TrainingStepWorkload:
    return TrainingStepWorkload(
        logical_nonpadding_tokens=1,
        loss_bearing_tokens=1,
        executed_token_equivalents=1,
        nominal_schedule_capacity_tokens=1,
        dummy_executed_token_equivalents=0,
        dummy_schedule_capacity_tokens=0,
        real_microbatches=1,
        dummy_microbatches=0,
    )


def _rank_payload() -> dict[str, object]:
    return {
        "program": "rl",
        "backward": True,
        "topology": RankTelemetryTopology(
            global_rank=0,
            dp_cp_ranks=(0,),
            pp_ranks=(0,),
        ),
        "statistics": (0.0,) * RANK_STATISTIC_COUNT,
        "workload": _workload(),
        "pipeline_metrics": {},
        "inter_metrics": {},
        "base_metrics": {},
        "num_gradient_steps": 1,
    }


def test_rank_telemetry_wire_contract_rejects_layout_drift() -> None:
    payload = _rank_payload()
    assert (
        RankCommandTelemetry.model_validate(payload).statistics == payload["statistics"]
    )
    with pytest.raises(ValidationError):
        RankCommandTelemetry.model_validate({**payload, "unknown_metric": 1})
    with pytest.raises(ValidationError):
        RankCommandTelemetry.model_validate({**payload, "statistics": (0.0,)})
    with pytest.raises(ValidationError):
        TrainingStepWorkload.model_validate(
            {**_workload().model_dump(), "merged_branch_field": 1}
        )


def test_zero_loss_forward_backward_result_roundtrips_without_gradient() -> None:
    assert ForwardBackwardResult.model_fields["produced_gradient"].is_required()
    calls = tuple(
        keywords
        for path in (
            "src/art/megatron/training/client.py",
            "src/art/megatron/training/slot.py",
        )
        for keywords in _constructor_keywords(path, "ForwardBackwardResult")
    )
    assert len(calls) == 4
    assert all("produced_gradient" in keywords for keywords in calls)
    packing = PackingOutcome(
        packed_sequence_length=1,
        packed_sequences=0,
        target_packed_sequences=1,
        nominal_capacity_tokens=1,
        physical_tokens=0,
        non_padding_tokens=0,
        loss_bearing_tokens=0,
        trainable_assistant_tokens=0,
        policy_token_counts=None,
        group_shapes=(),
    )
    result = ForwardBackwardResult(
        operation_id="operation",
        packing=packing,
        loss_fn_outputs=(),
        produced_gradient=False,
    )
    assert ForwardBackwardResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValidationError):
        ForwardBackwardResult(
            operation_id="operation",
            packing=packing,
            loss_fn_outputs=(),
            produced_gradient=True,
        )


def test_optimizer_metric_stops_before_residency_snapshot_bookkeeping() -> None:
    source = (ROOT / "src/art/megatron/runtime/executor.py").read_text()
    start = source.rindex("    def execute_optimizer(self, job: OptimizerJobSpec)")
    body = source[start : source.index("\n    def ", start + 5)]
    assert body.index("optimizer_step_s = time.perf_counter() - started") < body.index(
        "checkpoint_slot_residency_tensors"
    )
