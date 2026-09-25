"""Test-first: one existing-budget recovery before a fitting minimum-wave split.

No real allocator, process group, model, or GPU is used. Real planner integration
uses tiny CPU inputs; recovery uses the existing scalar clock/allocator facade.
"""

from dataclasses import replace
import gc
import types
from typing import cast
import unittest
import weakref

import pytest
import test_trainer_rank_cache_recovery as recovery_fixture
from test_trainer_rank_split import _packed_budget, _rank, _request

from art.trainer_rank import _impl


class TestSplitRecovery(unittest.TestCase):
    make = recovery_fixture.TestRecovery.make

    def candidate(self, required=5, *, split=True, target=True):
        plan = types.SimpleNamespace(
            packed_tokens=40, logical_tokens=40, subforward_count=2 if split else 1
        )
        check = _impl._MemoryCheck(required, 10, required <= 10)
        value = _impl._CandidateMicroBatch(
            inputs=[["A", "B"]],
            indices=(0,),
            plan=cast(_impl._AnyForwardPlan, plan),
            check=check,
            stats_global_count=1,
            rejected_candidates=1,
            cold_start=True,
        )
        original = types.SimpleNamespace(packed_tokens=40, logical_tokens=40)
        failed = _impl._MemoryCheck(80, 10, False)
        # Inject the private contract on the old tree so red tests demonstrate
        # ignored recovery, rather than stopping at a constructor TypeError.
        object.__setattr__(
            value, "recovery_target", (original, failed) if target else None
        )
        return value

    def recover(self, rank, candidates):
        calls = []

        def search():
            value = candidates[min(len(calls), len(candidates) - 1)]
            calls.append(value)
            if isinstance(value, BaseException):
                raise value
            return value

        result = rank._recover_admission(
            search,
            lambda value: (value.plan, value.check),
            lambda value, check: replace(value, check=check),
            context="forward_micro_batches",
            sync_across_dp=True,
        )
        return result, calls

    def setup_recovery(self):
        q, c, k, n = self.make()
        c.memory_reserved = lambda device: 128
        return q, c, k, n

    def test_exact_target_once_then_fresh_search_and_selected_check(self):
        q, c, k, n = self.setup_recovery()
        first = self.candidate()
        selected = self.candidate(80, split=False, target=False)
        seen = []
        original = q._try_cache_recovery

        def observed(check, **kwargs):
            seen.append((check, kwargs))
            return original(check, **kwargs)

        q._try_cache_recovery = observed
        result, calls = self.recover(q, [first, selected])
        self.assertEqual(c.events.count("release"), 1)
        self.assertEqual(len(calls), 2)
        self.assertIs(seen[0][0], first.recovery_target[1])
        self.assertTrue(seen[0][1]["require_unused_cache"])
        self.assertIs(result.plan, selected.plan)
        self.assertEqual(result.check.estimated_required_bytes, 80)
        self.assertEqual(result.check.available_bytes, 170)
        self.assertIsNone(result.recovery_target)

    def test_budget_denial_preserves_fitting_split(self):
        q, c, k, n = self.setup_recovery()
        state = q._recovery_state()
        state.first_consumed = True
        state.cost, state.high, state.work = 1.0, 1.0, 1.0
        first = self.candidate()
        result, calls = self.recover(q, [first])
        self.assertIs(result.plan, first.plan)
        self.assertIsNone(result.recovery_target)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("release", c.events)
        self.assertGreater(state.cost, 1.0)
        self.assertIsNone(state.owner)

    def test_no_unused_cache_preserves_split_without_release(self):
        q, c, k, n = self.setup_recovery()
        c.memory_reserved = lambda device: c.memory_allocated(device)
        first = self.candidate()
        result, calls = self.recover(q, [first])
        self.assertIs(result.plan, first.plan)
        self.assertIsNone(result.recovery_target)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("release", c.events)
        self.assertFalse(q._recovery_state().first_consumed)

    def test_existing_budget_allows_later_recovery_without_new_allowance(self):
        q, c, k, n = self.setup_recovery()
        state = q._recovery_state()
        state.first_consumed = True
        state.cost, state.high, state.work = 0.01, 0.01, 100.0
        result, calls = self.recover(
            q, [self.candidate(), self.candidate(80, target=False)]
        )
        self.assertEqual(c.events.count("release"), 1)
        self.assertEqual(len(calls), 2)
        self.assertGreater(state.cost, 0.01)
        self.assertTrue(state.first_consumed)

    def test_unsupported_allocator_or_invalid_budget_keeps_fitting_split(self):
        for condition in ("backend", "invalid_budget"):
            with self.subTest(condition=condition):
                q, c, k, n = self.setup_recovery()
                if condition == "backend":
                    c.backend = "cudaMallocAsync"
                    c.memory_reserved = lambda device: 16
                else:
                    q._recovery_state().invalid = True
                first = self.candidate()
                result, calls = self.recover(q, [first])
                self.assertIs(result.plan, first.plan)
                self.assertIsNone(result.recovery_target)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("release", c.events)
                self.assertFalse(q._recovery_state().first_consumed)

    def test_no_hint_means_no_optional_recovery(self):
        for split in (False, True):
            with self.subTest(split=split):
                q, c, k, n = self.setup_recovery()
                first = self.candidate(split=split, target=False)
                q._try_cache_recovery = lambda *a, **kw: self.fail(
                    "unrequested recovery"
                )
                result, calls = self.recover(q, [first])
                self.assertIs(result.plan, first.plan)
                self.assertEqual(len(calls), 1)

    def test_insufficient_release_can_keep_new_fitting_split(self):
        q, c, k, n = self.setup_recovery()
        first, later = self.candidate(), self.candidate(8)
        result, calls = self.recover(q, [first, later])
        self.assertEqual(c.events.count("release"), 1)
        self.assertEqual(len(calls), 2)
        self.assertIs(result.plan, later.plan)
        self.assertEqual(result.check.estimated_required_bytes, 8)

    def test_failed_research_never_gets_second_release(self):
        q, c, k, n = self.setup_recovery()
        refusal = _impl._ForwardRefusal(
            cast(
                _impl._AnyForwardPlan,
                types.SimpleNamespace(packed_tokens=200, logical_tokens=200),
            ),
            _impl._MemoryCheck(200, 170, False),
            "still cannot fit",
        )
        with self.assertRaises(_impl.TrainerRankMemoryError):
            self.recover(q, [self.candidate(), refusal])
        self.assertEqual(c.events.count("release"), 1)
        self.assertIsNone(q._recovery_state().owner)

    def test_oversized_return_releases_recovery_target_after_world_refresh_fails(self):
        q, c, k, n = self.setup_recovery()
        q._allow_oversized_batches = True
        available = iter((10, 3))

        def refresh(check, **kwargs):
            free = next(available)
            return replace(
                check, available_bytes=free, fits=check.estimated_required_bytes <= free
            )

        q._refresh_memory_check = refresh

        class Target:
            packed_tokens = 40

        first, later = self.candidate(), self.candidate(6)
        later = replace(later, recovery_target=(Target(), later.recovery_target[1]))
        target = weakref.ref(later.recovery_target[0])
        plan, inputs, indices = later.plan, later.inputs, later.indices
        pending = [first, later]
        result = q._recover_admission(
            lambda: pending.pop(0),
            lambda value: (value.plan, value.check),
            lambda value, check: replace(value, check=check),
            context="forward_micro_batches",
            sync_across_dp=True,
            admit_refusal=lambda refusal: self.fail("existing candidate must be used"),
        )
        self.assertEqual(pending, [])
        self.assertEqual(c.events.count("release"), 1)
        self.assertIs(result.plan, plan)
        self.assertIs(result.inputs, inputs)
        self.assertIs(result.indices, indices)
        self.assertEqual(result.check, _impl._MemoryCheck(6, 3, False))
        self.assertIsNone(result.recovery_target)
        self.assertIsNotNone(later.recovery_target)
        del first, later
        gc.collect()
        self.assertIsNone(target())

    def test_release_error_preserves_primary_instead_of_running_split(self):
        q, c, k, n = self.setup_recovery()
        primary = RuntimeError("release failed")
        c.failure = primary
        with self.assertRaises(RuntimeError) as caught:
            self.recover(q, [self.candidate()])
        self.assertIs(caught.exception, primary)
        self.assertEqual(c.events.count("release"), 1)
        self.assertTrue(q._recovery_state().first_consumed)
        self.assertIsNone(q._recovery_state().owner)

    def test_research_error_keeps_primary_and_owner_closes(self):
        q, c, k, n = self.setup_recovery()
        primary = ValueError("planner failure")
        with self.assertRaises(ValueError) as caught:
            self.recover(q, [self.candidate(), primary])
        self.assertIs(caught.exception, primary)
        self.assertEqual(c.events.count("release"), 1)
        self.assertIsNone(q._recovery_state().owner)

    def test_stale_fallback_after_budget_denial_searches_without_release(self):
        q, c, k, n = self.setup_recovery()
        state = q._recovery_state()
        state.first_consumed = True
        state.cost, state.high, state.work = 1.0, 1.0, 1.0
        original = q._try_cache_recovery

        def deny(check, **kwargs):
            result = original(check, **kwargs)
            self.assertFalse(result)
            c.free = 33  # Available 3, original split requires 5.
            return result

        q._try_cache_recovery = deny
        later = self.candidate(2, target=False)
        result, calls = self.recover(q, [self.candidate(), later])
        self.assertIs(result.plan, later.plan)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("release", c.events)

    def test_asymmetric_peers_follow_same_recovery_collective_order(self):
        transcripts = []
        for deficient in (False, True):
            with self.subTest(deficient=deficient):
                q, c, k, n = self.setup_recovery()
                c.free = 40 if deficient else 200
                q._refresh_memory_check = lambda check, **kw: replace(
                    check,
                    available_bytes=170,
                    fits=check.estimated_required_bytes <= 170,
                )
                expected = [
                    ("SUM", 2, [0.02, 0.0]),
                    ("MAX", 4, [80, 100, 0, 0]),
                    ("MIN", 4, [10, -1, 1, 1]),
                    ("MIN", 2, [170, -1]),
                ]
                transcript = []

                def reduce(values, *, op, sync_across_dp):
                    self.assertTrue(sync_across_dp)
                    want_op, length, result = expected[len(transcript)]
                    self.assertEqual((op, len(values)), (want_op, length))
                    transcript.append((op, len(values)))
                    return result

                q._recovery_reduce = reduce
                result, calls = self.recover(
                    q, [self.candidate(), self.candidate(80, split=False, target=False)]
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual(c.events.count("release"), int(deficient))
                self.assertEqual(len(transcript), 4)
                transcripts.append(transcript)
        self.assertEqual(*transcripts)


def test_pure_finder_retains_exact_unsplit_plan_and_check(
    monkeypatch: pytest.MonkeyPatch,
):
    rank = _rank(monkeypatch)
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *_a, **_kw: 0)
    _packed_budget(monkeypatch, rank, 20)
    observed = []
    original = rank._memory_check

    def check(plan, **kwargs):
        value = original(plan, **kwargs)
        observed.append((plan, value))
        return value

    monkeypatch.setattr(rank, "_memory_check", check)
    monkeypatch.setattr(
        rank, "_try_cache_recovery", lambda *a, **kw: pytest.fail("impure search")
    )
    targets = []
    found = rank._find_admissible_forward(
        [_request(i) for i in range(4)],
        checkpoint=_impl.Unset,
        refusal_prefix="test",
        unsplit_targets=targets,
    )
    assert not isinstance(found, _impl._ForwardRefusal)
    plan, selected = found
    assert isinstance(plan, _impl._SplitForwardPlan)
    assert len(targets) == 1
    target, denied = targets[0]
    assert isinstance(target, _impl._FlatForwardPlan)
    assert not denied.fits and selected.fits
    assert any(p is target and c is denied for p, c in observed)
    assert denied is not selected


def test_minimum_wave_hint_does_not_broaden_dp_rank_forward(
    monkeypatch: pytest.MonkeyPatch,
):
    rank = _rank(monkeypatch)
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *_a, **_kw: 0)
    _packed_budget(monkeypatch, rank, 20)
    inputs = [_request(i) for i in range(4)]
    candidate = rank._search_next_micro_batch([inputs], 0)
    assert isinstance(candidate, _impl._CandidateMicroBatch)
    assert isinstance(candidate.plan, _impl._SplitForwardPlan)
    assert candidate.stats_global_count == 1
    assert candidate.recovery_target is not None
    assert not candidate.recovery_target[1].fits
    # The ordinary DP-local return remains its existing (plan, check) pair.
    found = rank._find_admissible_forward(
        inputs, checkpoint=_impl.Unset, refusal_prefix="test"
    )
    assert isinstance(found, tuple) and len(found) == 2


def test_profile_only_minimum_wave_has_no_recovery_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    rank = _rank(monkeypatch)
    _packed_budget(monkeypatch, rank, 1000)
    monkeypatch.setattr(rank, "_all_ranks_have_memory_profile", lambda **kw: False)
    candidate = rank._search_next_micro_batch([[_request(0), _request(1)]], 0)
    assert isinstance(candidate, _impl._CandidateMicroBatch)
    assert isinstance(candidate.plan, _impl._FlatForwardPlan)
    assert candidate.recovery_target is None


@pytest.mark.parametrize("peer", (0, 1, 2))
def test_search_healthy_peer_carries_target_when_other_peer_splits(monkeypatch, peer):
    rank = _rank(monkeypatch)
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (peer, 3))
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *a, **kw: 0)
    _packed_budget(monkeypatch, rank, 20 if peer == 1 else 1000)
    local_check = rank._memory_check_required

    def check(required, *, sync_across_dp=False):
        return (
            _impl._MemoryCheck(40, 20, False)
            if sync_across_dp
            else local_check(required)
        )

    monkeypatch.setattr(rank, "_memory_check_required", check)
    events = []

    def outcome(local):
        events.append(("outcome", local))
        return 2

    def all_true(local):
        assert not events, "extra vote after the existing outcome collective"
        return local

    monkeypatch.setattr(rank, "_admission_outcome", outcome)
    monkeypatch.setattr(rank, "_all_ranks_true", all_true)
    monkeypatch.setattr(
        rank, "_try_cache_recovery", lambda *a, **kw: pytest.fail("impure search")
    )
    inputs = [[_request(i) for i in range(2)], [_request(i + 2) for i in range(4)]]
    candidate = rank._search_next_micro_batch(inputs, 0)
    assert isinstance(candidate, _impl._CandidateMicroBatch)
    assert candidate.indices == ((peer,) if peer < 2 else ())
    assert candidate.stats_global_count == 2
    assert isinstance(candidate.plan, _impl._FlatForwardPlan) == (peer != 1)
    assert candidate.recovery_target is not None
    assert candidate.recovery_target[1].fits == (peer != 1)
    assert events == [("outcome", 2 if peer == 1 else 3)]


@pytest.mark.parametrize("outcome", (0, 1))
def test_peer_error_or_refusal_precedes_new_split_vote(monkeypatch, outcome):
    rank = _rank(monkeypatch)
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *a, **kw: 0)
    _packed_budget(monkeypatch, rank, 20)
    voted = []

    def exchange(local):
        assert local == 2
        voted.append(local)
        return outcome

    def all_true(local):
        assert not voted, "new split vote after collective failure/refusal"
        return local

    monkeypatch.setattr(rank, "_admission_outcome", exchange)
    monkeypatch.setattr(rank, "_all_ranks_true", all_true)
    inputs = [[_request(i) for i in range(4)]]
    if outcome == 0:
        with pytest.raises(RuntimeError, match="another DP rank"):
            rank._search_next_micro_batch(inputs, 0)
    else:
        result = rank._search_next_micro_batch(inputs, 0)
        assert isinstance(result, _impl._ForwardRefusal)
    assert voted == [2]


def test_existing_outcome_all_flat_does_not_request_recovery(monkeypatch):
    rank = _rank(monkeypatch)
    _packed_budget(monkeypatch, rank, 1000)
    local_check = rank._memory_check_required

    def check(required, *, sync_across_dp=False):
        return (
            _impl._MemoryCheck(40, 20, False)
            if sync_across_dp
            else local_check(required)
        )

    monkeypatch.setattr(rank, "_memory_check_required", check)
    outcomes = []

    def outcome(local):
        outcomes.append(local)
        return local

    monkeypatch.setattr(rank, "_admission_outcome", outcome)
    candidate = rank._search_next_micro_batch([[_request(i) for i in range(4)]], 0)
    assert isinstance(candidate, _impl._CandidateMicroBatch)
    assert outcomes == [3]
    assert candidate.recovery_target is None
    assert isinstance(candidate.plan, _impl._FlatForwardPlan)
