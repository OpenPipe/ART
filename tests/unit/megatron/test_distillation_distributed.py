from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron import train
from art.megatron.composite_loss import (
    PreparedDistillationSidecars,
    PreparedPolicySidecars,
    composite_prepared_loss_from_logits,
)
from art.megatron.context_parallel.runtime import (
    dispatch_context_parallel_tensor,
)
from art.megatron.context_parallel.types import ParallelTopology, TokenRange
from art.megatron.distillation import PackedDistillationTensors

_INITIAL_LOGITS = torch.tensor(
    [
        [0.2, -0.3, 0.5, 0.1, -0.4],
        [-0.1, 0.4, 0.3, -0.2, 0.6],
        [0.6, 0.1, -0.4, 0.2, -0.3],
        [0.0, -0.2, 0.1, 0.4, 0.3],
    ],
    dtype=torch.float32,
)
_POLICY_CONFIG = {
    "epsilon": 1.0,
    "epsilon_high": 4.0,
    "importance_sampling_level": "token",
}
_COEFFICIENT = 0.7


class _DistributedFakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(_INITIAL_LOGITS.clone())

    def zero_grad_buffer(self) -> None:
        self.zero_grad(set_to_none=True)

    def forward(self, **_kwargs: Any) -> torch.Tensor:
        # Megatron models return sequence-first logits.
        return self.logits.unsqueeze(1)


class _DistributedFakeOptimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.param_groups: list[dict[str, Any]] = [{"params": [parameter], "lr": 0.0}]
        self.last_gradient: torch.Tensor | None = None

    def step(self) -> tuple[bool, float, int]:
        assert self.parameter.grad is not None
        self.last_gradient = self.parameter.grad.detach().clone()
        return (
            True,
            float(torch.linalg.vector_norm(self.parameter.grad).item()),
            int(torch.count_nonzero(self.parameter.grad == 0)),
        )

    def zero_grad(self) -> None:
        self.parameter.grad = None


class _DistributedFakeHandler:
    build_gdn_execution_spec = False

    def get_forward_kwargs(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def zero_internal_padding_grads(self, _model: Any) -> None:
        return

    def zero_internal_padding_params(self, _model: Any) -> None:
        return


def _packed(
    *,
    policy_mask: list[bool],
    target_mask: list[bool],
) -> PackedDistillationTensors:
    policy = torch.tensor([policy_mask], dtype=torch.bool)
    target = torch.tensor([target_mask], dtype=torch.bool)
    topk_ids = torch.full((1, 4, 2), -1, dtype=torch.long)
    teacher_logprobs = torch.zeros((1, 4, 2), dtype=torch.float32)
    tail_logprobs = torch.zeros((1, 4), dtype=torch.float32)
    for position in range(4):
        if target[0, position]:
            topk_ids[0, position] = torch.tensor([position % 5, (position + 2) % 5])
            teacher_logprobs[0, position] = torch.tensor([0.55, 0.25]).log()
            tail_logprobs[0, position] = math.log(0.2)
    return {
        "tokens": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "token_mask": torch.ones((1, 4), dtype=torch.bool),
        "input_pos": torch.arange(4).unsqueeze(0),
        "source_group_ids": torch.tensor([0]),
        "policy_mask": policy,
        "old_logprobs": torch.where(
            policy,
            torch.tensor(-1.0),
            torch.tensor(float("nan")),
        ),
        "policy_advantages": torch.tensor([[0.0, 1.0, -0.5, 0.75]]),
        "policy_weights": policy.float(),
        "policy_group_ids": torch.zeros((1, 4), dtype=torch.long),
        "target_mask": target,
        "distillation_weights": target.float(),
        "topk_token_ids": topk_ids,
        "teacher_logprobs": teacher_logprobs,
        "tail_logprobs": tail_logprobs,
        "temperatures": torch.ones((1, 4)),
    }


def _patch_distributed_step(*, topology: ParallelTopology) -> None:
    train_module = cast(Any, train)
    parallel_state = cast(Any, train.ps)
    train_module._validate_distillation_worker_topology = lambda _model: None
    train_module._infer_parallel_topology = lambda _model: topology
    parallel_state.get_data_parallel_group = lambda with_context_parallel=True: (
        dist.group.WORLD
    )
    parallel_state.get_tensor_model_parallel_rank = lambda: 0
    parallel_state.get_tensor_model_parallel_group = lambda check_initialized=False: (
        None
    )
    train_module.as_megatron_api_chunks = lambda chunks: chunks
    train_module.flush_param_grads_to_main_grads = lambda _chunks: None
    train_module._causal_attention_state = lambda *_args, **_kwargs: object()
    train_module._art_flex_sliding_windows = lambda _provider: ()

    def finalize(chunks: Any, num_tokens: torch.Tensor) -> None:
        global_denominator = num_tokens.detach().clone()
        dist.all_reduce(global_denominator)
        for chunk in chunks:
            for parameter in chunk.parameters():
                assert parameter.grad is not None
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(global_denominator.item())

    train_module.finalize_model_grads_extended = finalize


def _run_distributed_step(
    packed: PackedDistillationTensors,
    *,
    expected_policy_count: int,
    expected_target_count: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    model = _DistributedFakeModel()
    optimizer = _DistributedFakeOptimizer(model.logits)
    result = train.run_megatron_distillation_step(
        model_chunks=cast(Any, [model]),
        provider=object(),
        model_support_handler=_DistributedFakeHandler(),
        optimizer=optimizer,
        learning_rate=0.0,
        packed_tensors=packed,
        sample_indices=[0],
        logical_vocab_size=5,
        temperature=1.0,
        coefficient=_COEFFICIENT,
        compensate_temperature_squared=False,
        policy_config=cast(Any, _POLICY_CONFIG),
        expected_policy_count=expected_policy_count,
        expected_target_count=expected_target_count,
    )
    assert optimizer.last_gradient is not None
    return (
        optimizer.last_gradient,
        result.reduced_loss,
        result.policy_token_count,
        result.target_token_count,
    )


def _dp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _patch_distributed_step(topology=ParallelTopology(tp=1, dp=2, cp=1, pp=1))
        policy_only = _packed(
            policy_mask=[False, True, True, False],
            target_mask=[False, False, False, False],
        )
        kd_only = _packed(
            policy_mask=[False, False, False, False],
            target_mask=[False, False, True, False],
        )
        split_result = _run_distributed_step(
            policy_only if rank == 0 else kd_only,
            expected_policy_count=2,
            expected_target_count=1,
        )

        both = _packed(
            policy_mask=[False, True, False, False],
            target_mask=[False, False, True, False],
        )
        empty = _packed(
            policy_mask=[False, False, False, False],
            target_mask=[False, False, False, False],
        )
        anchor_result = _run_distributed_step(
            both if rank == 0 else empty,
            expected_policy_count=1,
            expected_target_count=1,
        )
        torch.save(
            {
                "split_gradient": split_result[0],
                "split_loss": split_result[1],
                "split_counts": split_result[2:],
                "anchor_gradient": anchor_result[0],
                "anchor_loss": anchor_result[1],
                "anchor_counts": anchor_result[2:],
            },
            Path(output_dir) / f"dp-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _reference(
    packed_batches: list[PackedDistillationTensors],
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _INITIAL_LOGITS.clone().requires_grad_(True)
    policy_sum = logits.new_zeros(())
    kd_sum = logits.new_zeros(())
    policy_count = 0
    target_count = 0
    for packed in packed_batches:
        micro = train._select_distillation_micro(
            packed,
            sample_index=0,
            device=torch.device("cpu"),
        )
        local_policy_count = int(micro["policy_mask"].sum().item())
        local_target_count = int(micro["target_mask"].sum().item())
        policy = (
            PreparedPolicySidecars(
                sampled_token_ids=micro["sampled_token_ids"],
                old_logprobs=micro["old_logprobs"],
                advantages=micro["policy_advantages"],
                weights=micro["policy_weights"],
                mask=micro["policy_mask"],
                group_ids=micro["policy_group_ids"],
            )
            if local_policy_count
            else None
        )
        kd = (
            PreparedDistillationSidecars(
                teacher_topk_ids=micro["topk_token_ids"],
                teacher_topk_logprobs=micro["teacher_logprobs"],
                teacher_tail_logprob=micro["tail_logprobs"],
                mask=micro["target_mask"],
                weights=micro["distillation_weights"],
            )
            if local_target_count
            else None
        )
        if policy is None and kd is None:
            continue
        result = composite_prepared_loss_from_logits(
            logits.unsqueeze(0),
            logical_vocab_size=5,
            policy=policy,
            distillation=kd,
            policy_config=cast(Any, _POLICY_CONFIG),
            distillation_coefficient=_COEFFICIENT,
        )
        if result.policy is not None:
            policy_sum = policy_sum + result.policy.loss_sum
        if result.distillation is not None:
            kd_sum = kd_sum + result.distillation.loss_sum
        policy_count += local_policy_count
        target_count += local_target_count
    loss = policy_sum / policy_count + _COEFFICIENT * kd_sum / target_count
    loss.backward()
    assert logits.grad is not None
    return loss.detach(), logits.grad


def test_dp_additive_global_denominators_and_zero_anchor_match_reference(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _dp_worker,
        args=(2, str(tmp_path / "dp-init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank_results = [
        torch.load(tmp_path / f"dp-rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]

    split_policy = _packed(
        policy_mask=[False, True, True, False],
        target_mask=[False, False, False, False],
    )
    split_kd = _packed(
        policy_mask=[False, False, False, False],
        target_mask=[False, False, True, False],
    )
    split_loss, split_gradient = _reference([split_policy, split_kd])
    anchor_both = _packed(
        policy_mask=[False, True, False, False],
        target_mask=[False, False, True, False],
    )
    anchor_empty = _packed(
        policy_mask=[False, False, False, False],
        target_mask=[False, False, False, False],
    )
    anchor_loss, anchor_gradient = _reference([anchor_both, anchor_empty])

    for result in rank_results:
        torch.testing.assert_close(result["split_loss"], split_loss)
        torch.testing.assert_close(result["split_gradient"], split_gradient)
        assert result["split_counts"] == (2, 1)
        torch.testing.assert_close(result["anchor_loss"], anchor_loss)
        torch.testing.assert_close(result["anchor_gradient"], anchor_gradient)
        assert result["anchor_counts"] == (1, 1)


def _cp_inputs() -> tuple[
    torch.Tensor,
    PreparedPolicySidecars,
    PreparedDistillationSidecars,
]:
    logits = torch.tensor(
        [
            [
                [0.2, -0.1, 0.5, 0.7, -0.4],
                [0.6, 0.1, -0.2, 0.4, 0.8],
                [-0.3, 0.4, 0.2, 0.1, 0.5],
                [0.9, -0.8, 0.3, 0.2, -0.1],
                [0.4, 0.6, -0.5, 0.1, 0.2],
                [-0.7, 0.3, 0.8, -0.2, 0.5],
                [0.1, -0.4, 0.6, 0.9, -0.3],
            ]
        ],
        dtype=torch.float32,
    )
    policy_mask = torch.tensor([[True, True, False, False, False, False, False]])
    policy = PreparedPolicySidecars(
        sampled_token_ids=torch.tensor([[2, 3, 0, 0, 0, 0, 0]]),
        old_logprobs=torch.tensor([[-1.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        advantages=torch.tensor([[1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        weights=policy_mask.float(),
        mask=policy_mask,
        group_ids=torch.zeros((1, 7), dtype=torch.long),
    )
    kd_mask = torch.tensor([[False, False, False, True, False, True, True]])
    topk_ids = torch.full((1, 7, 3), -1, dtype=torch.long)
    teacher_logprobs = torch.zeros((1, 7, 3))
    tail_logprobs = torch.zeros((1, 7))
    for position in (3, 5, 6):
        topk_ids[0, position] = torch.tensor([0, 2, 4])
        teacher_logprobs[0, position] = torch.tensor([0.4, 0.3, 0.2]).log()
        tail_logprobs[0, position] = math.log(0.1)
    kd = PreparedDistillationSidecars(
        teacher_topk_ids=topk_ids,
        teacher_topk_logprobs=teacher_logprobs,
        teacher_tail_logprob=tail_logprobs,
        mask=kd_mask,
        weights=torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.0, 1.25, 2.0]]),
    )
    return logits, policy, kd


def _dispatch_sidecars(
    sidecars: PreparedPolicySidecars | PreparedDistillationSidecars,
    *,
    rank_plan: Any,
) -> PreparedPolicySidecars | PreparedDistillationSidecars:
    dispatch = lambda tensor, pad: dispatch_context_parallel_tensor(
        tensor,
        rank_plan=rank_plan,
        pad_value=pad,
    )
    if isinstance(sidecars, PreparedPolicySidecars):
        return PreparedPolicySidecars(
            sampled_token_ids=dispatch(sidecars.sampled_token_ids, 0),
            old_logprobs=dispatch(sidecars.old_logprobs, 0.0),
            advantages=dispatch(sidecars.advantages, 0.0),
            weights=dispatch(sidecars.weights, 0.0),
            mask=dispatch(sidecars.mask, False),
            group_ids=dispatch(sidecars.group_ids, 0),
        )
    return PreparedDistillationSidecars(
        teacher_topk_ids=dispatch(sidecars.teacher_topk_ids, -1),
        teacher_topk_logprobs=dispatch(sidecars.teacher_topk_logprobs, 0.0),
        teacher_tail_logprob=dispatch(sidecars.teacher_tail_logprob, 0.0),
        mask=dispatch(sidecars.mask, False),
        weights=dispatch(sidecars.weights, 0.0),
    )


def _cp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        ranges = (TokenRange(0, 2), TokenRange(2, 7))
        owned = ranges[rank]
        rank_plan = SimpleNamespace(
            local_valid_lengths=(owned.size(),),
            local_row_ranges=(owned,),
        )
        full_logits, full_policy, full_kd = _cp_inputs()
        local_logits = dispatch_context_parallel_tensor(
            full_logits,
            rank_plan=cast(Any, rank_plan),
            pad_value=0.0,
        ).requires_grad_(True)
        local_policy = cast(
            PreparedPolicySidecars,
            _dispatch_sidecars(full_policy, rank_plan=rank_plan),
        )
        local_kd = cast(
            PreparedDistillationSidecars,
            _dispatch_sidecars(full_kd, rank_plan=rank_plan),
        )
        expected_topk = full_kd.teacher_topk_ids[:, owned.start : owned.end, :]
        torch.testing.assert_close(local_kd.teacher_topk_ids, expected_topk)
        assert local_kd.teacher_topk_ids.shape == (1, owned.size(), 3)

        cast(Any, train.ps).get_data_parallel_group = (
            lambda with_context_parallel=True: dist.group.WORLD
        )
        policy_count, target_count = train._reduce_prepared_counts(
            local_policy_count=int(local_policy.mask.sum().item()),
            local_target_count=int(local_kd.mask.sum().item()),
            device=torch.device("cpu"),
            topology=ParallelTopology(tp=1, dp=1, cp=2, pp=1),
        )
        active_policy = local_policy if bool(local_policy.mask.any()) else None
        active_kd = local_kd if bool(local_kd.mask.any()) else None
        result = composite_prepared_loss_from_logits(
            local_logits,
            logical_vocab_size=5,
            policy=active_policy,
            distillation=active_kd,
            policy_config=cast(Any, _POLICY_CONFIG),
            distillation_coefficient=_COEFFICIENT,
        )
        objective_sum = local_logits.new_zeros(())
        policy_sum = local_logits.new_zeros(())
        kd_sum = local_logits.new_zeros(())
        if result.policy is not None:
            policy_sum = result.policy.loss_sum
            objective_sum = objective_sum + policy_sum
        if result.distillation is not None:
            kd_sum = result.distillation.loss_sum
            objective_sum = objective_sum + (
                kd_sum * _COEFFICIENT * policy_count / target_count
            )
        objective_sum.backward()
        assert local_logits.grad is not None
        local_logits.grad.div_(policy_count)

        statistics = torch.stack((policy_sum.detach(), kd_sum.detach()))
        dist.all_reduce(statistics)
        reconstructed_gradient = torch.zeros_like(full_logits)
        reconstructed_gradient[:, owned.start : owned.end, :] = local_logits.grad
        dist.all_reduce(reconstructed_gradient)
        torch.save(
            {
                "gradient": reconstructed_gradient,
                "loss": (
                    statistics[0] / policy_count
                    + _COEFFICIENT * statistics[1] / target_count
                ),
                "counts": (policy_count, target_count),
                "topk": local_kd.teacher_topk_ids,
            },
            Path(output_dir) / f"cp-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_cp_uneven_dispatch_and_additive_gradient_reconstruct_reference(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _cp_worker,
        args=(2, str(tmp_path / "cp-init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank_results = [
        torch.load(tmp_path / f"cp-rank-{rank}.pt", weights_only=True)
        for rank in range(2)
    ]
    full_logits, policy, kd = _cp_inputs()
    reference_logits = full_logits.requires_grad_(True)
    reference = composite_prepared_loss_from_logits(
        reference_logits,
        logical_vocab_size=5,
        policy=policy,
        distillation=kd,
        policy_config=cast(Any, _POLICY_CONFIG),
        distillation_coefficient=_COEFFICIENT,
    )
    reference.loss.backward()
    assert reference_logits.grad is not None

    for result in rank_results:
        assert result["counts"] == (2, 3)
        torch.testing.assert_close(result["loss"], reference.loss)
        torch.testing.assert_close(result["gradient"], reference_logits.grad)
    assert rank_results[0]["topk"].shape == (1, 2, 3)
    assert rank_results[1]["topk"].shape == (1, 5, 3)
