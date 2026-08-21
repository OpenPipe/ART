from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.loss import (
    LossOffPolicyDiagnosticsAccumulator,
    compute_probs_corr_stats,
)
from art.megatron.context_parallel.types import TrainingStepWorkload
from art.megatron.training.command_telemetry import (
    PendingRankCommandTelemetry,
    RankTelemetryTopology,
    aggregate_rank_command_telemetry,
    materialize_rank_telemetry,
    rank_telemetry_statistics,
)
from art.megatron.training.finalize_grads import reduce_accumulated_token_count
from art.megatron.training.gradient_accumulator import (
    GradientAccumulator,
    ParameterGradientAccumulator,
)
from art.megatron.training.microbatches import _local_trainable_token_count_tensor
from art.megatron.training.pipeline_schedule import PipelineScheduleTelemetry
from art.trainer_rank._impl import TrainerRank


class _LossInputs:
    def __init__(self, mask: torch.Tensor) -> None:
        self.assistant_mask = mask

    def align_inputs(self) -> "_LossInputs":
        return self


class _MainGradChunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3))
        self.weight.main_grad = torch.zeros_like(self.weight, dtype=torch.float32)

    def zero_grad_buffer(self) -> None:
        self.weight.main_grad.zero_()


@pytest.mark.parametrize("contributions", (1, 2, 8))
def test_accumulators_keep_one_unnormalized_rank_local_sum(
    contributions: int,
) -> None:
    tokens = tuple(index + 1 for index in range(contributions))
    values = tuple(
        torch.full((3,), float((index + 1) * 7)) for index in range(contributions)
    )
    expected = sum(values, torch.zeros(3))

    parameter = torch.nn.Parameter(torch.zeros(3))
    dynamic = ParameterGradientAccumulator(parameters=(parameter,))
    resident_ids = None
    for index, (token_count, value) in enumerate(zip(tokens, values, strict=True)):
        dynamic.before_forward_backward()
        dynamic.record(
            f"dynamic-{index}",
            torch.tensor(token_count),
            (value.clone(),),
            expected_global_token_count=2 * token_count,
        )
        current_ids = tuple(map(id, dynamic.residency_tensors()))
        resident_ids = current_ids if resident_ids is None else resident_ids
        assert current_ids == resident_ids
    dynamic.seal(dynamic.contribution_ids)
    dynamic_sums = dynamic.prepare_optimizer()
    torch.testing.assert_close(dynamic_sums.gradients[0], expected)
    assert int(dynamic_sums.local_token_count) == sum(tokens)
    assert dynamic_sums.expected_global_token_count == 2 * sum(tokens)
    assert len(dynamic.residency_tensors()) == 2

    chunk = _MainGradChunk()
    static = GradientAccumulator.model_construct(model_chunks=[chunk])
    static.model_post_init(None)
    for index, (token_count, value) in enumerate(zip(tokens, values, strict=True)):
        static.before_forward_backward()
        chunk.weight.main_grad.copy_(value)
        static.record(
            f"static-{index}",
            torch.tensor(token_count),
            expected_global_token_count=2 * token_count,
        )
    static.seal(static.contribution_ids)
    static_sums = static.prepare_optimizer()
    torch.testing.assert_close(static_sums.gradients[0], expected)
    assert int(static_sums.local_token_count) == sum(tokens)
    assert static_sums.expected_global_token_count == 2 * sum(tokens)
    assert len(static.residency_tensors()) <= 2


def test_dynamic_accumulator_folds_resident_grads_into_stable_fp32_buffers() -> None:
    first = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    second = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    accumulator = ParameterGradientAccumulator(parameters=(first, second))
    pointers = None

    for index in range(8):
        accumulator.before_forward_backward()
        first.grad = torch.full_like(first, index + 1)
        second.grad = None if index % 2 else torch.full_like(second, 2 * index + 1)
        accumulator.record_parameters(
            f"fb-{index}",
            torch.tensor(index + 1),
            expected_global_token_count=2 * (index + 1),
        )
        current = tuple(
            tensor.data_ptr() for tensor in accumulator.residency_tensors()[1:]
        )
        pointers = current if pointers is None else pointers
        assert current == pointers

    accumulator.seal(accumulator.contribution_ids)
    sums = accumulator.prepare_optimizer()
    assert all(gradient.dtype == torch.float32 for gradient in sums.gradients)
    torch.testing.assert_close(sums.gradients[0], torch.full((3,), 36.0))
    torch.testing.assert_close(sums.gradients[1], torch.full((2,), 28.0))
    assert int(sums.local_token_count) == 36
    assert sums.expected_global_token_count == 72


def test_rejected_resident_gradient_record_does_not_mutate_accumulator() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    accumulator = ParameterGradientAccumulator(parameters=(parameter,))
    parameter.grad = torch.ones_like(parameter)
    accumulator.record_parameters("fb", torch.tensor(1))
    before = tuple(tensor.clone() for tensor in accumulator.residency_tensors())

    parameter.grad = torch.full_like(parameter, 9)
    with pytest.raises(RuntimeError, match="duplicate gradient contribution"):
        accumulator.record_parameters("fb", torch.tensor(9))

    assert accumulator.contribution_ids == ("fb",)
    for actual, expected in zip(accumulator.residency_tensors(), before, strict=True):
        torch.testing.assert_close(actual, expected)


def _new_pair_groups() -> tuple[list[dist.ProcessGroup], list[dist.ProcessGroup]]:
    dp = [dist.new_group((0, 2)), dist.new_group((1, 3))]
    tp = [dist.new_group((0, 1)), dist.new_group((2, 3))]
    return dp, tp


def _gradient_collective_worker(rank: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=4)
    try:
        from megatron.core import parallel_state as ps

        default_dp, default_tp = _new_pair_groups()
        expert_dp, expert_tp = _new_pair_groups()
        dp_index = rank % 2
        tp_index = rank // 2
        ps.get_data_parallel_group = lambda with_context_parallel=True: default_dp[
            dp_index
        ]
        ps.get_expert_data_parallel_group = lambda: expert_dp[dp_index]
        ps.get_tensor_model_parallel_group = (
            lambda check_initialized=False: default_tp[tp_index]
        )
        ps.get_expert_tensor_parallel_group = (
            lambda check_initialized=False: expert_tp[tp_index]
        )

        original_all_reduce = dist.all_reduce
        calls: list[int] = []

        def counted_all_reduce(*args: object, **kwargs: object) -> object:
            group = kwargs.get("group")
            calls.append(id(group))
            return original_all_reduce(*args, **kwargs)

        dist.all_reduce = counted_all_reduce  # type: ignore[assignment]
        for contribution_count in (1, 2, 8):
            default = torch.nn.Parameter(torch.zeros(2))
            default.allreduce = True
            default.grad_sync_domain = "tp_default"
            default.grad_sync_op = "sum"
            expert = torch.nn.Parameter(torch.zeros(2))
            expert.allreduce = False
            expert.grad_sync_domain = "expert_tp"
            expert.grad_sync_op = "sum"
            accumulator = ParameterGradientAccumulator(
                parameters=(default, expert)
            )
            local_tokens = []
            expected_default = torch.zeros(2)
            expected_expert = torch.zeros(2)
            expected_global_tokens = 0
            for index in range(contribution_count):
                dp_coordinate = rank // 2
                token_count = (index + 1) * (dp_coordinate + 1)
                global_tokens = 3 * (index + 1)
                default_value = torch.tensor(
                    ((rank + 1) * (index + 1), rank + index + 2),
                    dtype=torch.float32,
                )
                expert_value = default_value * 3
                accumulator.before_forward_backward()
                accumulator.record(
                    f"fb-{index}",
                    torch.tensor(token_count),
                    (default_value, expert_value),
                    expected_global_token_count=global_tokens,
                )
                local_tokens.append(token_count)
                expected_global_tokens += global_tokens
                for peer in range(4):
                    peer_value = torch.tensor(
                        ((peer + 1) * (index + 1), peer + index + 2),
                        dtype=torch.float32,
                    )
                    expected_default.add_(peer_value)
                    expected_expert.add_(peer_value * 3)

            accumulator.seal(accumulator.contribution_ids)
            local = accumulator.prepare_optimizer()
            calls.clear()
            global_tokens = reduce_accumulated_token_count(
                local.local_token_count,
                expected_global_token_count=expected_global_tokens,
                group=default_dp[dp_index],
            )
            trainer = object.__new__(TrainerRank)
            gradients = trainer._sync_dynamic_grads(
                (default, expert), local.gradients
            )
            torch._foreach_div_(gradients, global_tokens)

            assert len(calls) == 5
            assert sorted(calls.count(group) for group in set(calls)) == [1, 1, 1, 2]
            torch.testing.assert_close(
                gradients[0], expected_default / expected_global_tokens
            )
            torch.testing.assert_close(
                gradients[1], expected_expert / expected_global_tokens
            )
            assert int(local.local_token_count) == sum(local_tokens)
    finally:
        dist.all_reduce = original_all_reduce  # type: ignore[assignment]
        dist.destroy_process_group()


def test_collectives_are_constant_in_contribution_count(tmp_path: Path) -> None:
    mp.spawn(
        _gradient_collective_worker,
        args=(f"file://{tmp_path / 'gradient-collectives'}",),
        nprocs=4,
        join=True,
    )


def _workload(tokens: int) -> TrainingStepWorkload:
    return TrainingStepWorkload(
        logical_nonpadding_tokens=tokens,
        loss_bearing_tokens=tokens,
        executed_token_equivalents=tokens,
        nominal_schedule_capacity_tokens=tokens,
        dummy_executed_token_equivalents=0,
        dummy_schedule_capacity_tokens=0,
        real_microbatches=1,
        dummy_microbatches=0,
    )


def test_pp_telemetry_is_rank_local_and_controller_aggregated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def collective_is_a_bug(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("rank-local telemetry launched a collective")

    monkeypatch.setattr(dist, "all_reduce", collective_is_a_bug)
    monkeypatch.setattr(dist, "all_gather_into_tensor", collective_is_a_bug)
    payloads = []
    for rank in range(4):
        pp_rank = rank // 2
        dp_rank = rank % 2
        tokens = 3 if dp_rank == 0 else 5
        zero = torch.zeros(())
        pending = PendingRankCommandTelemetry(
            program="sft",
            backward=True,
            topology=RankTelemetryTopology(
                global_rank=rank,
                dp_cp_ranks=((0, 1) if pp_rank == 0 else (2, 3)),
                pp_ranks=((0, 2) if dp_rank == 0 else (1, 3)),
            ),
            statistics=rank_telemetry_statistics(
                loss_sum=torch.tensor(2.0 * tokens),
                token_count=torch.tensor(tokens),
                correlation=torch.zeros(6),
                kl_sum=zero,
                kl_count=zero,
                diagnostics=LossOffPolicyDiagnosticsAccumulator(),
            ),
            workload=_workload(tokens),
            schedules=(
                PipelineScheduleTelemetry(
                    pp_rank=pp_rank,
                    pp_size=2,
                    vp_size=1,
                    num_microbatches=1,
                    real_microbatches=1,
                    dummy_microbatches=0,
                    micro_batch_size=1,
                    seq_length=8,
                    microbatch_group_size=1,
                    forward_compute_s_by_chunk={0: float(pp_rank + 1)},
                    backward_compute_s_by_chunk={0: float(pp_rank + 2)},
                    forward_calls_by_chunk={0: 1},
                ),
            ),
            num_gradient_steps=1,
        )
        payloads.append(
            materialize_rank_telemetry(pending, pending.statistics.clone())
        )

    metrics = aggregate_rank_command_telemetry(
        payloads, expected_token_count=8
    )
    assert metrics["loss/train"] == pytest.approx(2.0)
    assert metrics["pipeline/stage_0/forward_compute_s"] == pytest.approx(1.0)
    assert metrics["pipeline/stage_1/forward_compute_s"] == pytest.approx(2.0)
    assert metrics["data/gradient_step_loss_bearing_tokens"] == 8
    assert metrics["data/step_num_gradient_steps"] == 1


def test_micro_loss_telemetry_has_no_host_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        Path(__file__).parents[1] / "src/art/megatron/train.py"
    ).read_text()
    tree = ast.parse(source)
    steps = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_megatron_rl_forward_backward_step"
    ]
    assert len(steps) == 1
    reducers = [
        node
        for node in ast.walk(steps[0])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "reduce_loss"
    ]
    assert reducers
    forbidden = {
        node.func.attr
        for reducer in reducers
        for node in ast.walk(reducer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "item" not in forbidden
    assert "cpu" not in forbidden

    def item_is_a_bug(_self: torch.Tensor) -> float:
        raise AssertionError("micro telemetry synchronized a tensor")

    monkeypatch.setattr(torch.Tensor, "item", item_is_a_bug)
    stats = compute_probs_corr_stats(
        torch.tensor((-1.0, -2.0)), torch.tensor((-1.5, -1.0))
    )
    assert stats.shape == (6,)
    count = _local_trainable_token_count_tensor(
        [_LossInputs(torch.tensor((True, False, True)))],
        device=torch.device("cpu"),
    )
    assert torch.equal(count, torch.tensor(2))
