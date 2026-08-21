from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import TrainerRank, _impl
from art.trainer_rank._impl import (
    PreparedTrainerRankOptimizerState,
    TrainerRankOptimizerLayout,
    TrainerRankOptimizerSnapshotSource,
    _CheckpointSlot,
    _DynamicOptimizer,
)


def _layout(
    shapes: Sequence[tuple[int, ...]], target: str, rank: int
) -> TrainerRankOptimizerLayout:
    return {
        "parallel": (1, 0, 1, 0, 1, 0, 1, 0),
        "parameters": tuple(
            (
                (f"{target}.lora_{index}.rank_{rank}",),
                shape,
                "torch.float32",
                "tp",
                False,
                None,
                "uniform",
                (),
            )
            for index, shape in enumerate(shapes)
        ),
    }


def _slot(
    shapes: Sequence[tuple[int, ...]], *, initialized: bool = True
) -> tuple[_CheckpointSlot, torch.optim.AdamW, dict[str, int]]:
    params = tuple(torch.nn.Parameter(torch.randn(shape)) for shape in shapes)
    masters = tuple(torch.nn.Parameter(param.detach().clone()) for param in params)
    optimizer = torch.optim.AdamW(masters, lr=0.01, foreach=False)
    if initialized:
        for master in masters:
            master.grad = torch.ones_like(master)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    calls = {"state_dict": 0}
    state_dict = optimizer.state_dict

    def counted_state_dict() -> dict[str, Any]:
        calls["state_dict"] += 1
        return state_dict()

    optimizer.state_dict = counted_state_dict  # type: ignore[method-assign]
    return (
        _CheckpointSlot(
            params=params,
            optimizer=_DynamicOptimizer(optimizer, masters),
            optimizer_padding_masks=tuple(
                torch.zeros_like(param, dtype=torch.bool) for param in params
            ),
        ),
        optimizer,
        calls,
    )


def _trainer(
    monkeypatch: pytest.MonkeyPatch,
    slots: dict[str, _CheckpointSlot],
    layouts: dict[str, TrainerRankOptimizerLayout],
) -> tuple[TrainerRank, dict[str, int]]:
    trainer = object.__new__(TrainerRank)
    trainer._checkpoint_slots = slots
    calls = {name: 0 for name in slots}

    def dynamic_layout(name: str) -> TrainerRankOptimizerLayout:
        calls[name] = calls.get(name, 0) + 1
        return layouts[name]

    monkeypatch.setattr(trainer, "_dynamic_optimizer_layout", dynamic_layout)
    return trainer, calls


def _optimizer_payload(source: TrainerRankOptimizerSnapshotSource) -> dict[str, Any]:
    return cast(dict[str, Any], source.state["optimizer"])


def test_cache_reuses_structure_and_freezes_generation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot, optimizer, calls = _slot(((2, 3), (3, 4)))
    trainer, layout_calls = _trainer(
        monkeypatch,
        {"run": slot},
        {"run": _layout(((2, 3), (3, 4)), "q_proj", 2)},
    )

    residency = trainer.checkpoint_slot_residency_tensors("run")
    first = trainer.checkpoint_slot_optimizer_residency_source("run")
    assert first is not None
    first_group = _optimizer_payload(first)["param_groups"][0]
    for _ in range(50):
        assert trainer.checkpoint_slot_residency_tensors("run") is residency
        assert trainer.checkpoint_slot_optimizer_residency_source("run") is not None
    assert calls["state_dict"] == layout_calls["run"] == 1
    snapshot = trainer.checkpoint_slot_optimizer_snapshot_sources("run")
    assert snapshot is not None
    assert isinstance(snapshot["layout"], dict)
    assert isinstance(cast(dict[str, Any], snapshot["optimizer"])["state"], dict)
    assert calls["state_dict"] == layout_calls["run"] == 1

    optimizer.param_groups[0]["lr"] = 0.25
    optimizer.param_groups[0]["step"] = 7
    second = trainer.checkpoint_slot_optimizer_residency_source("run")
    assert second is not None
    second_group = _optimizer_payload(second)["param_groups"][0]
    assert first_group["lr"] == 0.01
    assert "step" not in first_group
    assert second_group["lr"] == 0.25
    assert second_group["step"] == 7
    assert _optimizer_payload(first)["state"] is _optimizer_payload(second)["state"]
    assert calls["state_dict"] == layout_calls["run"] == 1
    with pytest.raises(TypeError):
        cast(dict[str, Any], first.state["layout"])["parallel"] = ()
    with pytest.raises(TypeError):
        cast(dict[int, Any], _optimizer_payload(first)["state"])[-1] = {}

    replacements = tuple(
        torch.full_like(tensor, index + 1) for index, tensor in enumerate(first.tensors)
    )
    bound = first.bind(replacements)
    source_tensors = _impl._unique_tensors(_impl._nested_tensors(first.state))
    bound_tensors = _impl._unique_tensors(_impl._nested_tensors(bound))
    replacements_by_id = {
        id(source): replacement
        for source, replacement in zip(first.tensors, replacements, strict=True)
    }
    assert tuple(map(id, bound_tensors)) == tuple(
        id(replacements_by_id[id(source)]) for source in source_tensors
    )
    restored_params = tuple(
        torch.nn.Parameter(torch.zeros_like(param)) for param in slot.params
    )
    restored = torch.optim.AdamW(restored_params)
    restored.load_state_dict(cast(dict[str, Any], bound["optimizer"]))
    assert len(restored.state) == len(restored_params)


def test_cache_is_exact_across_shapes_ranks_and_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small, _small_optimizer, small_calls = _slot(((1, 4),))
    wide, _wide_optimizer, wide_calls = _slot(((8, 4), (8, 6)))
    layouts = {
        "small": _layout(((1, 4),), "q_proj", 1),
        "wide": _layout(((8, 4), (8, 6)), "q_proj+v_proj", 8),
    }
    trainer, layout_calls = _trainer(
        monkeypatch, {"small": small, "wide": wide}, layouts
    )

    small_source = trainer.checkpoint_slot_optimizer_residency_source("small")
    wide_source = trainer.checkpoint_slot_optimizer_residency_source("wide")
    assert small_source is not None and wide_source is not None
    assert small_source.state["layout"] == layouts["small"]
    assert wide_source.state["layout"] == layouts["wide"]
    assert small_source.state["layout"] != wide_source.state["layout"]
    assert small_calls["state_dict"] == wide_calls["state_dict"] == 1

    replacement, _optimizer, replacement_calls = _slot(((3, 5), (3, 7)))
    trainer._checkpoint_slots["small"] = replacement
    layouts["small"] = _layout(((3, 5), (3, 7)), "gate_proj+up_proj", 3)
    replacement_source = trainer.checkpoint_slot_optimizer_residency_source("small")
    assert replacement_source is not None
    assert replacement_source.state["layout"] == layouts["small"]
    assert tuple(map(id, replacement_source.tensors)) != tuple(
        map(id, small_source.tensors)
    )
    assert replacement_calls["state_dict"] == 1
    assert layout_calls == {"small": 2, "wide": 1}


def test_cache_rebuilds_for_lazy_state_and_optimizer_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot, optimizer, calls = _slot(((2, 2), (2, 3)), initialized=False)
    layout = _layout(((2, 2), (2, 3)), "down_proj", 2)
    trainer, layout_calls = _trainer(monkeypatch, {"run": slot}, {"run": layout})

    empty = trainer.checkpoint_slot_optimizer_residency_source("run")
    assert empty is not None and not _optimizer_payload(empty)["state"]
    for master in cast(_DynamicOptimizer, slot.optimizer).master_params:
        master.grad = torch.ones_like(master)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    initialized = trainer.checkpoint_slot_optimizer_residency_source("run")
    assert initialized is not None
    assert len(_optimizer_payload(initialized)["state"]) == 2
    assert calls["state_dict"] == layout_calls["run"] == 2
    trainer.checkpoint_slot_optimizer_residency_source("run")
    assert calls["state_dict"] == layout_calls["run"] == 2

    masters = tuple(torch.nn.Parameter(param.detach().clone()) for param in slot.params)
    prepared = PreparedTrainerRankOptimizerState(
        layout=layout,
        master_params=masters,
        state=tuple(
            {
                "step": torch.zeros(()),
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
            }
            for master in masters
        ),
        param_group={"lr": 0.5},
        padding_masks=cast(tuple[torch.Tensor, ...], slot.optimizer_padding_masks),
    )
    trainer._bind_prepared_checkpoint_slot_optimizer(
        "run", prepared, torch.optim.AdamW(masters)
    )
    assert slot.optimizer_residency_descriptor is None
    rebound = trainer.checkpoint_slot_optimizer_residency_source("run")
    assert rebound is not None
    assert tuple(map(id, rebound.tensors)) != tuple(map(id, initialized.tensors))
    assert layout_calls["run"] == 3

    trainer.clear_checkpoint_slot_optimizer("run")
    assert slot.optimizer is None
    assert slot.optimizer_residency_descriptor is None
    assert trainer.checkpoint_slot_optimizer_residency_source("run") is None


def test_cache_rejects_detached_state_dict_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slot, optimizer, _calls = _slot(((2, 2),))
    trainer, _layout_calls = _trainer(
        monkeypatch,
        {"run": slot},
        {"run": _layout(((2, 2),), "q_proj", 2)},
    )

    def clone(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        return value

    optimizer.state_dict = lambda: cast(  # type: ignore[method-assign]
        dict[str, Any], clone(torch.optim.Optimizer.state_dict(optimizer))
    )
    with pytest.raises(
        _impl.TrainerRankSlotStateError, match="preserve live tensor identity"
    ):
        trainer.checkpoint_slot_optimizer_residency_source("run")
