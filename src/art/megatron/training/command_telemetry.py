from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Any, Literal

from megatron.core import parallel_state as ps
from pydantic import BaseModel, ConfigDict, Field
import torch

from art.loss import (
    IMPORTANCE_RATIO_HISTOGRAM_BUCKETS,
    PROBABILITY_CORRELATION_STAT_COUNT,
    LossOffPolicyDiagnosticsAccumulator,
    probability_correlation_from_stats,
)
from art.megatron.context_parallel.types import TrainingStepWorkload
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY

from .pipeline_schedule import (
    PipelineScheduleTelemetry,
    aggregate_pipeline_rank_metrics,
    aggregate_pipeline_schedule_metrics,
)

_CORRELATION = slice(2, 2 + PROBABILITY_CORRELATION_STAT_COUNT)
_KL_SUM = _CORRELATION.stop
_KL_COUNT = _KL_SUM + 1
_OFF_POLICY = slice(_KL_COUNT + 1, None)
RANK_STATISTIC_COUNT = _OFF_POLICY.start + 3 + IMPORTANCE_RATIO_HISTOGRAM_BUCKETS


class RankTelemetryTopology(BaseModel):
    model_config = ConfigDict(frozen=True)

    global_rank: int = Field(ge=0)
    dp_cp_ranks: tuple[int, ...] = Field(min_length=1)
    pp_ranks: tuple[int, ...] = Field(min_length=1)


class PendingRankCommandTelemetry(BaseModel):
    """Small rank-local result resolved after the GPU turn is released."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    program: Literal["rl", "sft"]
    backward: bool
    topology: RankTelemetryTopology
    statistics: torch.Tensor
    workload: TrainingStepWorkload
    schedules: tuple[PipelineScheduleTelemetry, ...] = Field(min_length=1)
    inter_metric_readers: tuple[Callable[[], dict[str, float]], ...] = ()
    base_metrics: dict[str, float] = Field(default_factory=dict)
    num_gradient_steps: int = Field(default=1, ge=1)


class RankCommandTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    program: Literal["rl", "sft"]
    backward: bool
    topology: RankTelemetryTopology
    statistics: tuple[float, ...]
    workload: TrainingStepWorkload
    pipeline_metrics: dict[str, float]
    inter_metrics: dict[str, float]
    base_metrics: dict[str, float]
    num_gradient_steps: int = Field(ge=1)


def rank_telemetry_topology() -> RankTelemetryTopology:
    if not torch.distributed.is_initialized():
        return RankTelemetryTopology(global_rank=0, dp_cp_ranks=(0,), pp_ranks=(0,))
    return RankTelemetryTopology(
        global_rank=torch.distributed.get_rank(),
        dp_cp_ranks=tuple(
            torch.distributed.get_process_group_ranks(
                ps.get_data_parallel_group(with_context_parallel=True)
            )
        ),
        pp_ranks=tuple(
            torch.distributed.get_process_group_ranks(
                ps.get_pipeline_model_parallel_group()
            )
        ),
    )


def rank_telemetry_statistics(
    *,
    loss_sum: torch.Tensor,
    token_count: torch.Tensor,
    correlation: torch.Tensor,
    kl_sum: torch.Tensor,
    kl_count: torch.Tensor,
    diagnostics: LossOffPolicyDiagnosticsAccumulator,
) -> torch.Tensor:
    if correlation.shape != (PROBABILITY_CORRELATION_STAT_COUNT,):
        raise ValueError("invalid probability-correlation statistics shape")
    values = torch.cat(
        (
            loss_sum.reshape(1),
            token_count.reshape(1),
            correlation,
            kl_sum.reshape(1),
            kl_count.reshape(1),
            diagnostics.tensor(device=loss_sum.device),
        )
    ).to(dtype=torch.float64)
    if values.shape != (RANK_STATISTIC_COUNT,):
        raise RuntimeError("rank telemetry statistic layout changed")
    return values


def materialize_rank_telemetry(
    pending: PendingRankCommandTelemetry,
    staged_statistics: torch.Tensor,
) -> dict[str, Any]:
    inter_metrics: dict[str, float] = {}
    for reader in pending.inter_metric_readers:
        values = reader()
        overlap = inter_metrics.keys() & values.keys()
        if overlap:
            raise RuntimeError(f"duplicate deferred telemetry metrics: {sorted(overlap)}")
        inter_metrics.update(values)
    return RankCommandTelemetry(
        program=pending.program,
        backward=pending.backward,
        topology=pending.topology,
        statistics=tuple(float(value) for value in staged_statistics.tolist()),
        workload=pending.workload,
        pipeline_metrics=aggregate_pipeline_schedule_metrics(
            pending.schedules, rank_local=True
        ),
        inter_metrics=inter_metrics,
        base_metrics=pending.base_metrics,
        num_gradient_steps=pending.num_gradient_steps,
    ).model_dump(mode="python")


def aggregate_rank_command_telemetry(
    payloads: Sequence[dict[str, Any]],
    *,
    expected_token_count: int,
) -> dict[str, float]:
    rows = tuple(RankCommandTelemetry.model_validate(payload) for payload in payloads)
    if not rows:
        raise ValueError("command telemetry requires rank results")
    by_rank = {row.topology.global_rank: row for row in rows}
    if len(by_rank) != len(rows) or 0 not in by_rank:
        raise RuntimeError("command telemetry has duplicate ranks or no global rank zero")
    coordinator = by_rank[0]
    invariants = {
        (row.program, row.backward, row.num_gradient_steps) for row in rows
    }
    if len(invariants) != 1:
        raise RuntimeError("trainer ranks returned incompatible command telemetry")
    data_rows = tuple(by_rank[rank] for rank in coordinator.topology.dp_cp_ranks)
    if any(row.topology.dp_cp_ranks != coordinator.topology.dp_cp_ranks for row in data_rows):
        raise RuntimeError("data-parallel telemetry groups disagree")
    totals = tuple(
        math.fsum(row.statistics[index] for row in data_rows)
        for index in range(RANK_STATISTIC_COUNT)
    )
    observed_tokens = int(totals[1])
    if observed_tokens != expected_token_count:
        raise RuntimeError(
            "rank telemetry token count differs from packed provenance: "
            f"observed={observed_tokens}, expected={expected_token_count}"
        )
    workload = TrainingStepWorkload(
        **{
            name: sum(getattr(row.workload, name) for row in data_rows)
            for name in TrainingStepWorkload.model_fields
        }
    )
    pipeline_rows = tuple(
        by_rank[rank].pipeline_metrics for rank in coordinator.topology.pp_ranks
    )
    metrics = {
        "loss/train": totals[0] / max(totals[1], 1.0),
        "data/gradient_step_nonpadding_logical_tokens": float(
            workload.logical_nonpadding_tokens
        ),
        "data/gradient_step_loss_bearing_tokens": float(workload.loss_bearing_tokens),
        "data/gradient_step_executed_token_equivalents": float(
            workload.executed_token_equivalents
        ),
        "data/gradient_step_nominal_schedule_capacity_tokens": float(
            workload.nominal_schedule_capacity_tokens
        ),
        "data/gradient_step_dummy_executed_token_equivalents": float(
            workload.dummy_executed_token_equivalents
        ),
        "data/gradient_step_dummy_schedule_capacity_tokens": float(
            workload.dummy_schedule_capacity_tokens
        ),
        "pipeline/gradient_step_real_microbatches": float(workload.real_microbatches),
        "pipeline/gradient_step_dummy_microbatches": float(workload.dummy_microbatches),
        **coordinator.base_metrics,
        **aggregate_pipeline_rank_metrics(pipeline_rows),
    }
    for row in sorted(rows, key=lambda item: item.topology.global_rank):
        overlap = metrics.keys() & row.inter_metrics.keys()
        if overlap:
            raise RuntimeError(f"duplicate rank telemetry metrics: {sorted(overlap)}")
        metrics.update(row.inter_metrics)
    if coordinator.backward:
        metrics[TRAIN_GRADIENT_STEPS_KEY] = float(coordinator.num_gradient_steps)
    correlation = torch.tensor(totals[_CORRELATION], dtype=torch.float64)
    if totals[_CORRELATION.start] >= 2:
        metrics["loss/probs_corr"] = float(
            probability_correlation_from_stats(correlation)
        )
    if totals[_KL_COUNT] > 0:
        metrics["loss/kl_policy_ref"] = totals[_KL_SUM] / totals[_KL_COUNT]
    if totals[_OFF_POLICY.start + 2] > 0:
        diagnostics = LossOffPolicyDiagnosticsAccumulator.from_tensor(
            torch.tensor(totals[_OFF_POLICY], dtype=torch.float64)
        )
        metrics.update(diagnostics.to_metrics())
    return metrics
