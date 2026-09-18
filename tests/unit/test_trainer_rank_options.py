from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
import pickle
from typing import Any, cast, get_type_hints

import cloudpickle
import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOptions,
    ResolvedForwardOptions,
    Unset,
    resolve_forward_options,
)
from art.trainer_rank import (
    ImportanceSamplingGradientCorrection as Correction,
)
from art.trainer_rank._corrections import (
    correct_logprob_cotangent,
    importance_weights,
)


@pytest.mark.parametrize(
    "public_type", [ForwardOptions, ResolvedForwardOptions, Correction]
)
def test_public_option_annotations_resolve(public_type: type) -> None:
    hints = get_type_hints(public_type)
    assert hints
    if public_type is ForwardOptions:
        assert hints["max_gradient_staleness"] == int | type(Unset)


def test_options_resolve_per_field_and_preserve_explicit_overrides() -> None:
    constructor = ForwardOptions(max_gradient_staleness=8, output_device="cpu")
    method = ForwardOptions(max_gradient_staleness=4, allow_cpu_offload=False)
    input = ForwardOptions(max_gradient_staleness=0, stale_gradient_corrections=[])
    resolved = resolve_forward_options(constructor, method, input)
    assert resolved == ResolvedForwardOptions(
        max_gradient_staleness=0,
        stale_gradient_corrections=(),
        allow_cpu_offload=False,
        output_device="cpu",
    )
    assert resolve_forward_options().stale_gradient_corrections == (Correction(),)
    assert resolve_forward_options().max_gradient_staleness == 2


def test_options_snapshot_corrections_and_replace_inherited_collection() -> None:
    corrections = [Correction(clip_high=2)]
    options = ForwardOptions(stale_gradient_corrections=corrections)
    corrections.clear()
    resolved = resolve_forward_options(
        options, ForwardOptions(stale_gradient_corrections=[Correction(clip_high=3)])
    )
    assert options.stale_gradient_corrections == (Correction(clip_high=2),)
    assert resolved.stale_gradient_corrections == (Correction(clip_high=3),)
    with pytest.raises(FrozenInstanceError):
        setattr(resolved, "max_gradient_staleness", 0)
    with pytest.raises(FrozenInstanceError):
        setattr(options, "output_device", "model")


@pytest.mark.parametrize(
    "roundtrip",
    [
        copy,
        deepcopy,
        lambda x: pickle.loads(pickle.dumps(x)),
        lambda x: cloudpickle.loads(cloudpickle.dumps(x)),
    ],
)
def test_unset_identity_survives_transport(roundtrip) -> None:
    assert roundtrip(Unset) is Unset
    options = roundtrip(ForwardOptions(max_gradient_staleness=0))
    assert options.output_device is Unset
    assert resolve_forward_options(options).max_gradient_staleness == 0
    request = roundtrip(ForwardInput(input_tokens=torch.tensor([1]), options=options))
    assert request.checkpoint is Unset
    assert request.options == options


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gradient_staleness": -1},
        {"max_gradient_staleness": True},
        {"max_gradient_staleness": 1.5},
        {"allow_cpu_offload": 0},
        {"allow_replay": None},
        {"backward_state": "disk"},
        {"output_device": "cuda"},
        {"stale_gradient_corrections": [Correction(), Correction()]},
    ],
)
def test_invalid_options_fail_before_submission(kwargs) -> None:
    with pytest.raises(ValueError):
        ForwardOptions(**kwargs)


@pytest.mark.parametrize(
    "state,flag", [("cpu", "allow_cpu_offload"), ("replay", "allow_replay")]
)
def test_cross_level_conflicts_are_checked_after_resolution(state, flag) -> None:
    constructor = ForwardOptions(**cast(dict[str, Any], {flag: False}))
    method = ForwardOptions(backward_state=state)
    with pytest.raises(ValueError, match="requires"):
        resolve_forward_options(constructor, method)
    # An input can override either side of the inherited conflict.
    assert (
        resolve_forward_options(
            constructor, method, ForwardOptions(**cast(dict[str, Any], {flag: True}))
        ).backward_state
        == state
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"clip_low": -1},
        {"clip_low": 6},
        {"clip_high": float("inf")},
        {"clip_low": float("nan")},
        {"policy": "sometimes"},
    ],
)
def test_invalid_correction(kwargs) -> None:
    with pytest.raises(ValueError):
        Correction(**kwargs)


def test_score_function_weighting_matches_current_categorical_expectation() -> None:
    # Enumerate all actions: E_old[(p_new/p_old) A grad log p_new].
    logits = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    rewards = torch.tensor([1.0, -0.5, 2.0], dtype=torch.float64)
    current_logprobs = logits.log_softmax(-1)
    weights = importance_weights(old.log(), current_logprobs, Correction(clip_high=100))
    assert not weights.requires_grad
    weighted_score = (old * weights * rewards * current_logprobs).sum()
    actual = torch.autograd.grad(weighted_score, logits, retain_graph=True)[0]
    expected = torch.autograd.grad((current_logprobs.exp() * rewards).sum(), logits)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_ratio_clipping_is_stable_in_low_precision(dtype) -> None:
    original = torch.tensor([-10000, -1, -1, -10000], dtype=dtype, requires_grad=True)
    current = torch.tensor(
        [-1, -10000, -float("inf"), -10000], dtype=dtype, requires_grad=True
    )
    weights = importance_weights(
        original, current, Correction(clip_low=0.1, clip_high=5)
    )
    torch.testing.assert_close(weights, weights.new_tensor([5, 0.1, 0.1, 1]))
    assert weights.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert not weights.requires_grad
    assert torch.isfinite(weights).all()
    assert torch.equal(
        importance_weights(original, current, Correction(clip_high=0)),
        torch.zeros_like(weights),
    )


def test_top_k_uses_original_ids_and_full_distribution_probabilities() -> None:
    old = torch.tensor([[0.4, 0.3]], dtype=torch.float64)
    current = torch.tensor([[0.1, 0.6]], dtype=torch.float64)
    tokens = torch.tensor([[2, 0]])
    cotangent = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    result = correct_logprob_cotangent(
        cotangent,
        original_logprobs=old.log(),
        current_logprobs=current.log(),
        correction=Correction(),
        original_tokens=tokens,
        current_tokens=tokens,
    )
    torch.testing.assert_close(
        result, torch.tensor([[0.75, -4.0]], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="same token IDs"):
        correct_logprob_cotangent(
            cotangent,
            original_logprobs=old.log(),
            current_logprobs=current.log(),
            correction=Correction(),
            original_tokens=tokens,
            current_tokens=tokens.flip(-1),
        )


def test_unavailable_correction_follows_policy_without_forward() -> None:
    grad = torch.ones(2)
    kwargs: dict[str, Any] = dict(
        original_logprobs=torch.zeros(2), current_logprobs=None
    )
    assert correct_logprob_cotangent(grad, correction=Correction(), **kwargs) is grad
    with pytest.raises(RuntimeError, match="requires current logprobs"):
        correct_logprob_cotangent(
            grad, correction=Correction(policy="always"), **kwargs
        )


@pytest.mark.parametrize(
    "original,current",
    [
        ([0.0], [float("nan")]),
        ([0.0], [float("inf")]),
        ([float("-inf")], [-1.0]),
        ([float("-inf")], [float("-inf")]),
    ],
)
def test_undefined_ratios_and_missing_support_raise(original, current) -> None:
    with pytest.raises(ValueError):
        importance_weights(torch.tensor(original), torch.tensor(current), Correction())


def test_correction_rejects_implicit_broadcasting() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        importance_weights(torch.zeros(2, 1), torch.zeros(2), Correction())
    with pytest.raises(ValueError, match="shapes must match"):
        correct_logprob_cotangent(
            torch.ones(2, 1),
            original_logprobs=torch.zeros(2),
            current_logprobs=torch.zeros(2),
            correction=Correction(),
        )


@pytest.mark.parametrize("bound", [1e-300, 1e300])
def test_extreme_finite_bounds_preserve_representable_weights(bound: float) -> None:
    weights = importance_weights(
        torch.tensor([-1.0]),
        torch.tensor([-1.0]),
        Correction(clip_low=bound, clip_high=bound),
    )
    assert weights.dtype == torch.float64
    assert weights.item() == bound


def test_resolved_options_reject_unset_and_unsupported_corrections() -> None:
    with pytest.raises(ValueError, match="cannot contain Unset"):
        ResolvedForwardOptions(max_gradient_staleness=cast(Any, Unset))
    with pytest.raises(TypeError, match="unsupported stale gradient correction"):
        ForwardOptions(stale_gradient_corrections=cast(Any, [object()]))
