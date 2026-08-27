from __future__ import annotations

import pytest
import torch

from art.megatron.portable_optimizer_archive import PortableOptimizerArchiveMetadata
from art.trainer_rank._optimizer_semantics import (
    optimizer_iteration,
    require_uniform_optimizer_iterations,
    shared_optimizer_iteration,
)


def test_shared_iteration_reads_te_parameter_group_state() -> None:
    assert shared_optimizer_iteration(
        {"step": 7},
        ({"exp_avg": torch.zeros(1)}, {"exp_avg_sq": torch.zeros(1)}),
    ) == 7


@pytest.mark.parametrize("value", (1.5, torch.tensor(2.25)))
def test_optimizer_iteration_rejects_fractional_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite nonnegative integer"):
        optimizer_iteration(value)


def test_shared_iteration_rejects_per_parameter_or_missing_group_counters() -> None:
    with pytest.raises(ValueError, match="missing its shared iteration"):
        shared_optimizer_iteration({}, ({"exp_avg": torch.zeros(1)},))
    with pytest.raises(ValueError, match="must not contain iteration counters"):
        shared_optimizer_iteration({"step": 3}, ({"step": 3},))


def test_logical_iterations_collapse_to_one_te_counter() -> None:
    assert require_uniform_optimizer_iterations((4, torch.tensor(4.0))) == 4
    with pytest.raises(ValueError, match="logical optimizer iterations differ"):
        require_uniform_optimizer_iterations((4, 5))


def test_portable_metadata_hard_cuts_to_shared_step_contract() -> None:
    with pytest.raises(ValueError, match="format_version"):
        PortableOptimizerArchiveMetadata(
            format_version=1,  # type: ignore[arg-type]
            source_rank=0,
            source_world_size=1,
            logical_keys=("weight",),
            steps={"weight": 1.0},
            param_group={"step": 1},
        )
    with pytest.raises(ValueError, match="requires step"):
        PortableOptimizerArchiveMetadata(
            source_rank=0,
            source_world_size=1,
            logical_keys=("weight",),
            steps={"weight": 1.0},
            param_group={"lr": 3e-5},
        )
    with pytest.raises(ValueError, match="differ from the shared"):
        PortableOptimizerArchiveMetadata(
            source_rank=0,
            source_world_size=1,
            logical_keys=("weight",),
            steps={"weight": 2.0},
            param_group={"step": 1},
        )
