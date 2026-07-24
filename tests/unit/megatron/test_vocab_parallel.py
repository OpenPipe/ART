from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.composite_loss import (
    PreparedDistillationSidecars,
    PreparedPolicySidecars,
    composite_prepared_loss_from_logits,
)
from art.megatron.context_parallel.runtime import (
    dispatch_context_parallel_tensor,
)
from art.megatron.context_parallel.types import TokenRange
from art.megatron.vocab_parallel import (
    vocabulary_parallel_composite_prepared_loss,
)


def _sidecars() -> tuple[PreparedPolicySidecars, PreparedDistillationSidecars]:
    policy_mask = torch.tensor([[True, False], [False, True]])
    policy = PreparedPolicySidecars(
        sampled_token_ids=torch.tensor([[0, 0], [0, 6]]),
        old_logprobs=torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),
        advantages=torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
        weights=policy_mask.float(),
        mask=policy_mask,
        group_ids=torch.tensor([[0, 0], [0, 1]]),
    )
    kd_mask = torch.ones((2, 2), dtype=torch.bool)
    ids = torch.tensor([[[0, 4], [3, 6]], [[4, 0], [6, 3]]])
    teacher_probs = torch.tensor([0.55, 0.25])
    teacher_logprobs = teacher_probs.log().expand(2, 2, 2).clone()
    distillation = PreparedDistillationSidecars(
        teacher_topk_ids=ids,
        teacher_topk_logprobs=teacher_logprobs,
        teacher_tail_logprob=torch.full((2, 2), math.log(0.2)),
        mask=kd_mask,
        weights=torch.tensor([[1.0, 0.5], [0.25, 1.5]]),
        temperature=1.7,
        compensate_temperature_squared=True,
    )
    return policy, distillation


def _policy_config() -> dict[str, Any]:
    return {
        "epsilon": 1.0,
        "epsilon_high": 4.0,
        "importance_sampling_level": "token",
    }


def _tp_worker(
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
        group = dist.new_group(ranks=list(range(world_size)))
        full = torch.tensor(
            [
                [
                    [0.2, -0.1, 0.5, 0.7, -0.4, 0.3, 0.9, 80.0],
                    [0.6, 0.1, -0.2, 0.4, 0.8, -0.5, 0.2, -80.0],
                ],
                [
                    [-0.3, 0.4, 0.2, 0.1, 0.5, 0.7, -0.6, 70.0],
                    [0.9, -0.8, 0.3, 0.2, -0.1, 0.6, 0.4, -70.0],
                ],
            ],
            dtype=torch.float64,
        )
        local = full[..., rank * 4 : (rank + 1) * 4].clone().requires_grad_(True)
        policy, kd = _sidecars()
        result = vocabulary_parallel_composite_prepared_loss(
            local,
            logical_vocab_size=7,
            tensor_parallel_group=group,
            tensor_parallel_rank=rank,
            tensor_parallel_world_size=world_size,
            policy=policy,
            distillation=kd,
            policy_config=cast(Any, _policy_config()),
            distillation_coefficient=0.7,
        )
        result.loss.backward()
        assert local.grad is not None
        torch.save(
            {
                "loss": result.loss.detach(),
                "policy_loss": result.policy_loss,
                "kd_loss": result.distillation_loss,
                "grad": local.grad,
                "tail": result.distillation.details.student_tail_mass_sum
                if result.distillation is not None
                else None,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_tp2_sparse_scores_and_gradients_match_full_vocab(tmp_path: Path) -> None:
    init_file = tmp_path / "init"
    mp.spawn(
        _tp_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    shards = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)
    ]

    full = torch.tensor(
        [
            [
                [0.2, -0.1, 0.5, 0.7, -0.4, 0.3, 0.9, 80.0],
                [0.6, 0.1, -0.2, 0.4, 0.8, -0.5, 0.2, -80.0],
            ],
            [
                [-0.3, 0.4, 0.2, 0.1, 0.5, 0.7, -0.6, 70.0],
                [0.9, -0.8, 0.3, 0.2, -0.1, 0.6, 0.4, -70.0],
            ],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    policy, kd = _sidecars()
    reference = composite_prepared_loss_from_logits(
        full,
        logical_vocab_size=7,
        policy=policy,
        distillation=kd,
        policy_config=cast(Any, _policy_config()),
        distillation_coefficient=0.7,
    )
    reference.loss.backward()
    assert full.grad is not None

    torch.testing.assert_close(shards[0]["loss"], reference.loss)
    torch.testing.assert_close(shards[1]["loss"], reference.loss)
    torch.testing.assert_close(shards[0]["policy_loss"], reference.policy_loss)
    torch.testing.assert_close(shards[0]["kd_loss"], reference.distillation_loss)
    torch.testing.assert_close(
        torch.cat([shards[0]["grad"], shards[1]["grad"]], dim=-1),
        full.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    assert shards[0]["grad"][..., -1].abs().sum() > 0
    assert shards[1]["grad"][..., 0].abs().sum() > 0


def test_tp_requires_an_explicit_group() -> None:
    policy, _kd = _sidecars()
    with pytest.raises(RuntimeError, match="explicit tensor-parallel group"):
        vocabulary_parallel_composite_prepared_loss(
            torch.zeros((2, 2, 4)),
            logical_vocab_size=7,
            tensor_parallel_group=None,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=2,
            policy=policy,
            policy_config=cast(Any, _policy_config()),
        )


def test_context_parallel_dispatch_preserves_trailing_dimensions() -> None:
    tensor = torch.arange(16).reshape(1, 4, 2, 2)
    rank_plan = SimpleNamespace(
        local_valid_lengths=(2,),
        local_row_ranges=(TokenRange(1, 3),),
    )

    dispatched = dispatch_context_parallel_tensor(
        tensor,
        rank_plan=cast(Any, rank_plan),
        pad_value=-1,
    )

    assert dispatched.shape == (1, 2, 2, 2)
    torch.testing.assert_close(dispatched, tensor[:, 1:3])
