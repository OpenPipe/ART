"""Landing acceptance: knob-free TrainerRank public contract.

These tests encode the public-surface half of the holistic-planner landing
contract (research thread behavior spec, frozen 2026-08-31):

- ``TrainerRank`` exposes no prefix-sharing depth, microbatch width,
  head-chunk, or memory-safety policy knob. Its constructor accepts only the
  training runtime.
- ``forward_micro_batches`` and ``dp_rank_forward`` accept only
  ``inputs``, ``checkpoint``, and ``no_grad``.
- ``TrainerRankMemoryError`` reports only a predicted peak, the usable limit,
  and an actionable reduction suggestion. It carries no infeasibility proof.

These tests define the landing contract: written and fail-verified before
the implementation, they must pass unmodified on the landed tree.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

import art.trainer_rank as trainer_rank

BANNED_KNOBS = (
    "shared_prefix_max_depth",
    "head_chunk_tokens",
    "memory_safety_factor",
    "memory_reserve_fraction",
)

BANNED_PROOF_FIELDS = (
    "feasibility_proof",
    "proven_minimum_bytes",
    "proven_minimum_peak_bytes",
    "limiting_atomic_input",
    "component_breakdown",
    "retained_graph_bytes",
    "ephemeral_bytes",
    "output_head_bytes",
    "gradient_bytes",
    "persistent_capacity_bytes",
)

REQUIRED_MEMORY_ERROR_FIELDS = (
    "predicted_peak_bytes",
    "usable_limit_bytes",
    "suggestion",
)


def _parameters(callable_: Any) -> dict[str, inspect.Parameter]:
    signature = inspect.signature(callable_)
    return {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name not in ("self", "cls")
    }


def test_constructor_accepts_only_the_training_runtime() -> None:
    parameters = _parameters(trainer_rank.TrainerRank.__init__)
    assert list(parameters) == ["runtime"], (
        "TrainerRank must accept exactly one constructor argument (the training"
        f" runtime); found {sorted(parameters)}"
    )


@pytest.mark.parametrize("knob", BANNED_KNOBS)
def test_constructor_rejects_policy_knob(knob: str) -> None:
    parameters = _parameters(trainer_rank.TrainerRank.__init__)
    assert knob not in parameters, f"policy knob {knob!r} must be removed"
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ), "TrainerRank.__init__ must not accept **kwargs (knobs could pass silently)"


@pytest.mark.parametrize("method_name", ("forward_micro_batches", "dp_rank_forward"))
def test_forward_method_signatures_are_knob_free(method_name: str) -> None:
    method = getattr(trainer_rank.TrainerRank, method_name)
    parameters = _parameters(method)
    allowed = {"inputs", "checkpoint", "no_grad"}
    assert set(parameters) <= allowed, (
        f"{method_name} accepts unexpected parameters:"
        f" {sorted(set(parameters) - allowed)}"
    )
    assert "inputs" in parameters, f"{method_name} must accept inputs"
    for name in set(parameters) - {"inputs"}:
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{method_name} parameter {name!r} must be keyword-only"
        )


def test_memory_error_reports_actionable_fields_without_proof() -> None:
    error_class = trainer_rank.TrainerRankMemoryError
    init_parameters = set(_parameters(error_class.__init__))
    annotations = set(getattr(error_class, "__annotations__", {}))
    declared = init_parameters | annotations | set(vars(error_class))
    missing = [field for field in REQUIRED_MEMORY_ERROR_FIELDS if field not in declared]
    assert not missing, (
        "TrainerRankMemoryError must report predicted peak, usable limit, and an"
        f" actionable suggestion; missing {missing} (declared: {sorted(declared)})"
    )
    forbidden = [field for field in BANNED_PROOF_FIELDS if field in declared]
    assert not forbidden, (
        "TrainerRankMemoryError must not carry infeasibility-proof or"
        f" per-component attribution fields; found {forbidden}"
    )


def test_no_public_test_anchor_hook() -> None:
    """Forced layout anchors are test-only; they must not be public API."""

    for method_name in ("__init__", "forward_micro_batches", "dp_rank_forward"):
        parameters = _parameters(getattr(trainer_rank.TrainerRank, method_name))
        leaked = [
            name
            for name in parameters
            if "anchor" in name or "policy" in name or "forced" in name
        ]
        assert not leaked, (
            f"test-only layout forcing leaked into TrainerRank.{method_name}: {leaked}"
        )


def test_no_knob_reexports() -> None:
    exported = set(dir(trainer_rank))
    leaked = [knob for knob in BANNED_KNOBS if knob in exported]
    assert not leaked, f"policy knobs re-exported from art.trainer_rank: {leaked}"
