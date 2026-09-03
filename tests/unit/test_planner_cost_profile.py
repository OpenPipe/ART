"""The fitted layout score applies only inside its calibrated capability profile."""

from __future__ import annotations

import torch

from art.trainer_rank._planner_cost import (
    CALIBRATION_PROFILE,
    COEFFICIENT_VERSION,
    COEFFICIENT_VERSION_FALLBACK,
    coefficient_version_for,
    prefix_tree_layout_score,
)
from art.trainer_rank._prefix_tree_planner import (
    build_canonical_prefix_tree,
    prefix_tree_layout_candidates,
)


def _version(
    *,
    device_capability: tuple[int, int] | None = (9, 0),
    param_dtype: str = "torch.bfloat16",
    hidden_size: int = 2_560,
    is_moe: bool = False,
) -> int:
    """The measured configuration, with one fact overridden at a time."""

    return coefficient_version_for(
        device_capability=device_capability,
        param_dtype=param_dtype,
        hidden_size=hidden_size,
        is_moe=is_moe,
    )


def test_profile_admits_the_measured_configuration() -> None:
    assert _version() == COEFFICIENT_VERSION
    assert CALIBRATION_PROFILE.matches(
        device_capability=(9, 0),
        param_dtype="torch.bfloat16",
        hidden_size=2_560,
        is_moe=False,
    )


def test_profile_falls_back_outside_its_domain() -> None:
    # Ampere (unmeasured GPU class), fp16, a 0.6B-class width, an 8B-class
    # width, and mixture-of-experts each leave the calibrated domain.
    assert _version(device_capability=(8, 0)) == COEFFICIENT_VERSION_FALLBACK
    assert _version(param_dtype="torch.float16") == COEFFICIENT_VERSION_FALLBACK
    assert _version(hidden_size=1_024) == COEFFICIENT_VERSION_FALLBACK
    assert _version(hidden_size=4_096) == COEFFICIENT_VERSION_FALLBACK
    assert _version(is_moe=True) == COEFFICIENT_VERSION_FALLBACK


def test_cpu_only_planning_uses_the_fitted_table() -> None:
    # No CUDA device (unit tests): the profile describes GPU execution, so the
    # device is not a reason to fall back.
    assert (
        coefficient_version_for(
            device_capability=None,
            param_dtype="torch.float16",
            hidden_size=8,
            is_moe=False,
        )
        == COEFFICIENT_VERSION
    )


def test_fallback_version_reproduces_the_landing_score() -> None:
    rows = tuple(
        torch.tensor(list(range(100, 2_148)) + [tail], dtype=torch.long)
        for tail in (1, 2, 3, 4)
    )
    tree = build_canonical_prefix_tree(rows)
    for candidate in prefix_tree_layout_candidates(tree):
        layout = candidate.layout
        fallback = prefix_tree_layout_score(
            layout,
            cp_size=4,
            layers=2,
            uses_gdn=True,
            coefficient_version=COEFFICIENT_VERSION_FALLBACK,
        )
        # The landing's hand-shaped formula, written out.
        cp, layers = 4, 2
        segments = len(layout.segments)
        edges = len(layout.selected_decisions)
        depth = layout.maximum_depth
        expected = (
            layers * layout.packed_tokens * 1_024
            + ((layout.packed_tokens + cp - 1) // cp) * (96 + 32 * cp)
            + segments * (96 + 32 * cp) * 1_024
            + edges * (64 + 32 * cp) * 1_024
            + min(1, max(0, depth - 1)) * layers * 768 * 1_024
            + max(0, depth - 2) * layers * 256 * 1_024
        )
        assert fallback == (expected, layout.packed_tokens, segments, depth)
        fitted = prefix_tree_layout_score(layout, cp_size=4, layers=2, uses_gdn=True)
        assert fitted[1:] == fallback[1:]  # same deterministic tie-breaks
