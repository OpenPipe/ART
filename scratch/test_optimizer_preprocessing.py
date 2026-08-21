from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.training.gradient_accumulator import (
    ParameterGradientAccumulator,
)
from art.trainer_rank._impl import (
    AdamParams,
    TrainerRank,
    TrainerRankSlotStateError,
    _CheckpointSlot,
    _compact_optimizer_valid_ranges,
    _distributed_grad_norm,
    _DynamicOptimizer,
    _zero_optimizer_padding,
)


def _reference_grad_norm(grads: tuple[torch.Tensor, ...]) -> float:
    squared = torch.zeros((), dtype=torch.float32)
    for grad in grads:
        squared.add_(grad.float().square().sum())
    return float(torch.sqrt(squared).item())


def _adamw(
    params: tuple[torch.nn.Parameter, ...], config: AdamParams
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=False,
    )


def test_compact_ranges_exactly_replace_dense_masks() -> None:
    masks = (
        torch.tensor(
            [
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
            ]
        ),
        torch.tensor(
            [
                [[False, False, False, True, True]] * 2,
                [[False, False, False, True, True]] * 2,
            ]
        ),
        torch.zeros((2, 3), dtype=torch.bool),
    )
    ranges = tuple(_compact_optimizer_valid_ranges(mask) for mask in masks)
    assert ranges == (
        ((3, 6), 1, ((0, 2), (4, 6))),
        ((2, 2, 5), 2, ((0, 3),)),
        None,
    )

    values = tuple(
        torch.arange(mask.numel(), dtype=torch.float32).reshape(mask.shape) + 1
        for mask in masks
    )
    expected = tuple(
        value.masked_fill(mask, 0) for value, mask in zip(values, masks, strict=True)
    )
    _zero_optimizer_padding(zip(values, ranges, strict=True))
    for actual, target in zip(values, expected, strict=True):
        assert torch.equal(actual, target)

    with pytest.raises(TrainerRankSlotStateError, match="not representable"):
        _compact_optimizer_valid_ranges(torch.eye(3, dtype=torch.bool))


@pytest.mark.parametrize("count", (1, 64, 1024))
def test_foreach_norm_matches_scalar_reduction(count: int) -> None:
    generator = torch.Generator().manual_seed(17)
    grads = tuple(
        torch.randn(17 + index % 31, generator=generator) for index in range(count)
    )
    params = (torch.nn.Parameter(torch.empty(0)),) * count
    actual = _distributed_grad_norm(params, grads)
    expected = _reference_grad_norm(grads)
    assert actual == pytest.approx(expected, rel=2.0e-6, abs=1.0e-7)


def test_sealed_accumulator_is_masked_and_clipped_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AdamParams(
        learning_rate=0.01,
        beta1=0.8,
        beta2=0.9,
        eps=1.0e-8,
        weight_decay=0.05,
        grad_clip_norm=0.3,
    )
    masks = (
        torch.tensor(
            [
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
                [False, False, True, True, False, False],
            ]
        ),
        torch.tensor(
            [[False, False, False, False, False], [True, True, True, True, True]]
        ),
    )
    ranges = tuple(_compact_optimizer_valid_ranges(mask) for mask in masks)
    model_params = tuple(
        torch.nn.Parameter(
            (torch.arange(mask.numel(), dtype=torch.float32) + 1)
            .reshape(mask.shape)
            .masked_fill(mask, 0)
            .mul_(0.01)
        )
        for mask in masks
    )
    first_finite = tuple(
        (torch.arange(mask.numel(), dtype=torch.float32) + 1).reshape(mask.shape)
        for mask in masks
    )
    second_finite = tuple(
        torch.flip(grad, dims=(-1,)).mul(0.25) for grad in first_finite
    )
    first = tuple(
        grad.masked_fill(mask, torch.nan)
        for grad, mask in zip(first_finite, masks, strict=True)
    )
    second = tuple(
        grad.masked_fill(mask, torch.inf)
        for grad, mask in zip(second_finite, masks, strict=True)
    )
    expected_accumulated = tuple(
        (left * 2.0 + right * 3.0) / 5.0
        for left, right in zip(first, second, strict=True)
    )

    accumulator = ParameterGradientAccumulator(parameters=model_params)
    accumulator.record(
        "first", torch.tensor(2.0), tuple(grad.clone() for grad in first)
    )
    accumulator.record(
        "second", torch.tensor(3.0), tuple(grad.clone() for grad in second)
    )
    resident = accumulator.residency_tensors()[1:]
    accumulator.seal(("first", "second"))
    owned = accumulator.prepare_optimizer()
    assert tuple(map(id, owned)) == tuple(map(id, resident))
    for actual, target in zip(owned, expected_accumulated, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0, equal_nan=True)

    reference_grads = tuple(
        grad.masked_fill(mask, 0)
        for grad, mask in zip(expected_accumulated, masks, strict=True)
    )
    reference_norm = _reference_grad_norm(reference_grads)
    clip = min(1.0, config.grad_clip_norm / (reference_norm + 1.0e-6))
    reference_grads = tuple(grad.mul(clip) for grad in reference_grads)

    masters = tuple(
        torch.nn.Parameter(param.detach().clone()) for param in model_params
    )
    reference_masters = tuple(
        torch.nn.Parameter(param.detach().clone()) for param in model_params
    )
    optimizer = _adamw(masters, config)
    reference_optimizer = _adamw(reference_masters, config)
    reference_masters[0].grad = reference_grads[0]
    reference_masters[1].grad = None
    reference_optimizer.step()
    reference_optimizer.zero_grad(set_to_none=True)

    trainer = object.__new__(TrainerRank)
    trainer._checkpoint_slots = {
        "run": _CheckpointSlot(
            params=model_params,
            optimizer=_DynamicOptimizer(optimizer, masters),
            optimizer_valid_ranges=ranges,
        )
    }
    monkeypatch.setattr(trainer, "_slot_ref", lambda _name: None)
    monkeypatch.setattr(trainer, "_prune_slot_graphs", lambda _ref=None: None)
    result = trainer._step_dynamic_optimizer(
        (("run", model_params, owned, (True, False)),),
        params=config,
    )

    assert result["grad_norm"] == pytest.approx(reference_norm, rel=2.0e-6)
    assert tuple(map(id, owned)) == tuple(map(id, resident))
    for actual, target, mask in zip(owned, reference_grads, masks, strict=True):
        torch.testing.assert_close(actual, target, rtol=2.0e-6, atol=1.0e-7)
        assert not torch.count_nonzero(actual.masked_select(mask))
    for actual, target in zip(model_params, reference_masters, strict=True):
        torch.testing.assert_close(actual, target, rtol=2.0e-6, atol=1.0e-7)
    assert trainer._checkpoint_slots["run"].revision == 1


def test_optimizer_master_and_moments_zero_only_padding() -> None:
    mask = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, True, True, False],
        ]
    )
    ranges = (_compact_optimizer_valid_ranges(mask),)
    model = torch.nn.Parameter(torch.zeros(mask.shape))
    master = torch.nn.Parameter(torch.ones(mask.shape))
    optimizer = torch.optim.AdamW((master,), lr=0.01, foreach=False)
    master.grad = torch.ones_like(master)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    master.data.fill_(7)
    for value in optimizer.state[master].values():
        if isinstance(value, torch.Tensor) and value.shape == master.shape:
            value.fill_(7)

    trainer = object.__new__(TrainerRank)
    trainer._checkpoint_slots = {
        "run": _CheckpointSlot(
            params=(model,),
            optimizer=_DynamicOptimizer(optimizer, (master,)),
            optimizer_valid_ranges=ranges,
        )
    }
    trainer._zero_dynamic_optimizer_padding(
        "run", _DynamicOptimizer(optimizer, (master,))
    )

    for value in (master, *optimizer.state[master].values()):
        if isinstance(value, torch.Tensor) and value.shape == master.shape:
            assert torch.equal(value, torch.full_like(value, 7).masked_fill(mask, 0))


def _distributed_worker(rank: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        from megatron.core import parallel_state as ps

        world = dist.group.WORLD
        ps.get_data_parallel_group = lambda with_context_parallel=True: world
        ps.get_expert_data_parallel_group = lambda: world
        ps.get_tensor_model_parallel_group = lambda check_initialized=False: None
        ps.get_expert_tensor_parallel_group = lambda check_initialized=False: None

        replicated = torch.nn.Parameter(torch.empty(0))
        replicated_grad = (
            torch.tensor([3.0, 4.0]) if rank == 0 else torch.tensor([30.0, 40.0])
        )
        assert _distributed_grad_norm((replicated,), (replicated_grad,)) == 5.0

        ps.get_data_parallel_group = lambda with_context_parallel=True: None
        ps.get_expert_data_parallel_group = lambda: None
        sharded = torch.nn.Parameter(torch.empty(0))
        sharded.lora_tp_sharded = True
        sharded_grad = (
            torch.tensor([3.0, 4.0]) if rank == 0 else torch.tensor([0.0, 12.0])
        )
        assert _distributed_grad_norm((sharded,), (sharded_grad,)) == 13.0
    finally:
        dist.destroy_process_group()


def test_distributed_norm_counts_replicas_once_and_all_shards(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _distributed_worker,
        args=(f"file://{tmp_path / 'optimizer_norm'}",),
        nprocs=2,
        join=True,
    )
