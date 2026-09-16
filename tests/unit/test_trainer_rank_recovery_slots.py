"""Checkpoint setup precedes the DP-local recovery loop exactly once."""

from types import SimpleNamespace
from typing import cast

import pytest

from art.trainer_rank import TrainerRank, _impl


@pytest.mark.parametrize("search_count", (1, 2, 3))
@pytest.mark.parametrize("empty", (False, True))
def test_dp_recovery_ensures_before_all_searches(search_count, empty):
    rank = TrainerRank.__new__(TrainerRank)
    rank.device = _impl.torch.device("cpu")
    requests = [] if empty else [object()]
    checkpoint = object()
    events = []
    plan = cast(
        _impl._AnyForwardPlan, SimpleNamespace(packed_tokens=1, logical_tokens=1)
    )
    bad = _impl._MemoryCheck(80, 10, False)
    fit = (plan, _impl._MemoryCheck(80, 200, True))
    refused = _impl._ForwardRefusal(plan, bad, "too large")
    results = {1: [fit], 2: [refused, fit], 3: [(plan, bad), refused, fit]}[
        search_count
    ]

    def ensure(actual, **kwargs):
        assert actual is requests and kwargs == {"checkpoint": checkpoint}
        events.append("ensure")

    def search(actual, **kwargs):
        assert actual is requests
        assert kwargs == dict(
            checkpoint=checkpoint,
            refusal_prefix="forward is predicted to exceed available memory",
            ensure_slots=False,
        )
        events.append("search")
        return results.pop(0)

    rank._ensure_checkpoint_slots_for = ensure
    rank._find_admissible_forward = search
    rank._snapshot_planning_telemetry = lambda *args: None
    rank._try_cache_recovery = lambda *args, **kwargs: True
    result = rank._plan_admissible_forward(
        requests, checkpoint=checkpoint, context="dp_rank_forward"
    )
    assert result == fit and not results
    assert events == ["ensure"] + ["search"] * search_count


@pytest.mark.parametrize("error_type", (ValueError, KeyboardInterrupt))
def test_checkpoint_error_precedes_search_and_preserves_identity(error_type):
    rank = TrainerRank.__new__(TrainerRank)
    error = error_type("checkpoint setup failed")
    error.__cause__, error.__context__ = LookupError("cause"), KeyError("context")
    error.__suppress_context__ = True
    cause, context = error.__cause__, error.__context__
    events = []

    def ensure(*args, **kwargs):
        events.append("ensure")
        raise error

    def forbidden(*args, **kwargs):
        raise AssertionError("search/recovery must not begin")

    rank._ensure_checkpoint_slots_for = ensure
    rank._recover_admission = forbidden
    rank._find_admissible_forward = forbidden
    with pytest.raises(error_type) as captured:
        rank._plan_admissible_forward([], checkpoint=None, context="dp_rank_forward")
    assert captured.value is error
    assert error.__cause__ is cause and error.__context__ is context
    assert error.__suppress_context__ and events == ["ensure"]


@pytest.mark.parametrize("ensure_slots", (None, False, True))
def test_direct_search_keeps_default_setup(ensure_slots):
    rank = TrainerRank.__new__(TrainerRank)
    events = []
    plan, check = object(), _impl._MemoryCheck(80, 200, True)
    rank._ensure_checkpoint_slots_for = lambda *a, **kw: events.append("ensure")

    def materialize(*args, **kwargs):
        assert kwargs == dict(checkpoint=None, ensure_slots=False)
        events.append("plan")
        return plan

    rank._plan_flat_forward = materialize
    rank._memory_check = lambda actual: check if actual is plan else None
    options = {} if ensure_slots is None else {"ensure_slots": ensure_slots}
    assert rank._find_admissible_forward(
        [], checkpoint=None, refusal_prefix="refused", **options
    ) == (plan, check)
    assert events == (["plan"] if ensure_slots is False else ["ensure", "plan"])
