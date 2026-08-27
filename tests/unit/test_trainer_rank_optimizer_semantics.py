from __future__ import annotations

import pytest
import torch

from art.megatron.portable_optimizer_archive import PortableOptimizerArchiveMetadata
from art.trainer_rank._optimizer_semantics import (
    require_uniform_optimizer_steps,
    shared_optimizer_step,
)


def test_shared_optimizer_step_reads_te_group_state() -> None:
    assert shared_optimizer_step(
        {"step": 7},
        ({"exp_avg": torch.zeros(1)}, {"exp_avg_sq": torch.zeros(1)}),
    ) == 7.0


def test_shared_optimizer_step_projects_uniform_logical_state() -> None:
    assert shared_optimizer_step(
        {},
        ({"step": torch.tensor(3.0)}, {"step": 3}),
    ) == 3.0


@pytest.mark.parametrize(
    ("group", "states", "message"),
    (
        ({"step": 2}, ({"step": 1},), "differs from the shared"),
        ({}, ({"step": 1}, {}), "incomplete"),
        ({}, ({"step": 1}, {"step": 2}), "differ within"),
    ),
)
def test_shared_optimizer_step_rejects_ambiguous_state(
    group: dict[str, object],
    states: tuple[dict[str, object], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        shared_optimizer_step(group, states)


def test_logical_optimizer_steps_must_collapse_to_one_te_group_step() -> None:
    assert require_uniform_optimizer_steps((4, torch.tensor(4.0))) == 4.0
    with pytest.raises(ValueError, match="logical optimizer steps differ"):
        require_uniform_optimizer_steps((4, 5))


def test_portable_optimizer_metadata_rejects_ambiguous_step_contract() -> None:
    with pytest.raises(ValueError, match="format_version"):
        PortableOptimizerArchiveMetadata(
            format_version=1,  # type: ignore[arg-type]
            source_rank=0,
            source_world_size=1,
            logical_keys=("weight",),
            steps={"weight": 1.0},
            param_group={"step": 1.0},
        )
    with pytest.raises(ValueError, match="requires step"):
        PortableOptimizerArchiveMetadata(
            source_rank=0,
            source_world_size=1,
            logical_keys=("weight",),
            steps={"weight": 1.0},
            param_group={"lr": 3e-5},
        )
