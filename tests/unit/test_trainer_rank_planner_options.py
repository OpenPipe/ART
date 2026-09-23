"""Opt-in planner fallback/report boundaries, using real CPU plans and fake counters."""

from contextlib import nullcontext
import gc
import json
from pathlib import Path

import pytest
from test_trainer_rank_split import _packed_budget, _rank, _recording_executor, _request
import torch

from art.trainer_rank import _impl as tr
from art.trainer_rank._planner_misses import Reporter


def _oversized(monkeypatch, *, limit=5):
    rank = _rank(monkeypatch)
    rank._allow_oversized_batches = True
    monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *a, **k: 0)
    _packed_budget(monkeypatch, rank, limit)
    return rank


def test_default_refusal_does_not_execute(monkeypatch):
    rank = _oversized(monkeypatch)
    rank._allow_oversized_batches = False
    executed = _recording_executor(monkeypatch, rank)
    with pytest.raises(tr.TrainerRankMemoryError):
        rank.dp_rank_forward([_request(i) for i in range(4)])
    assert executed == []


def test_override_exhausts_recovery_then_runs_lowest_exact_split(monkeypatch):
    rank = _oversized(monkeypatch)
    events = []
    monkeypatch.setattr(
        rank, "_try_cache_recovery", lambda *a, **k: events.append("recovery") or False
    )
    executed = _recording_executor(monkeypatch, rank)
    rank.dp_rank_forward([_request(i) for i in range(4)])
    assert events == ["recovery"]
    assert len(executed) == 4
    assert all(plan.packed_tokens == 10 for plan in executed)
    assert rank.last_forward_telemetry()["predicted_peak_bytes"] == 10


def test_fitting_split_is_unchanged_when_override_enabled(monkeypatch):
    rank = _oversized(monkeypatch, limit=20)
    executed = _recording_executor(monkeypatch, rank)
    rank.dp_rank_forward([_request(i) for i in range(4)])
    assert [plan.packed_tokens for plan in executed] == [20, 20]


def test_cache_recovery_fit_wins_over_unsafe_minimum(monkeypatch):
    rank = _oversized(monkeypatch)

    def recover(*args, **kwargs):
        _packed_budget(monkeypatch, rank, 20)
        return True

    monkeypatch.setattr(rank, "_try_cache_recovery", recover)
    executed = _recording_executor(monkeypatch, rank)
    rank.dp_rank_forward([_request(i) for i in range(4)])
    assert [plan.packed_tokens for plan in executed] == [20, 20]


def test_override_selects_best_rung_not_last(monkeypatch):
    rank = _oversized(monkeypatch)
    monkeypatch.setattr(
        rank,
        "_estimate_required_memory_bytes_from_values",
        lambda *, packed_tokens, **k: {40: 400, 20: 20, 10: 30}[packed_tokens],
    )
    executed = _recording_executor(monkeypatch, rank)
    rank.dp_rank_forward([_request(i) for i in range(4)])
    assert [plan.packed_tokens for plan in executed] == [20, 20]


def test_ep_unsupported_split_is_never_overridden(monkeypatch):
    rank = _oversized(monkeypatch)
    monkeypatch.setattr(rank, "_expert_parallel_active", lambda: True)
    executed = _recording_executor(monkeypatch, rank)
    with pytest.raises(tr.TrainerRankMemoryError, match="expert parallelism"):
        rank.dp_rank_forward([_request(i) for i in range(2)])
    assert not executed


def test_disagreeing_peer_refuses_override(monkeypatch):
    rank = _oversized(monkeypatch)
    monkeypatch.setattr(rank, "_try_cache_recovery", lambda *a, **k: False)
    seen = []

    def reduce(values, *, op, sync_across_dp):
        seen.append((values, op, sync_across_dp))
        return [0.0]

    monkeypatch.setattr(rank, "_recovery_reduce", reduce)
    with pytest.raises(tr.TrainerRankMemoryError):
        rank.dp_rank_forward([_request(0)])
    assert seen == [([1.0, 0.0, 0.0], "MIN", False)]


def test_microbatch_override_keeps_minimum_wave_inputs(monkeypatch):
    rank = _oversized(monkeypatch)
    wave = [_request(i) for i in range(4)]
    candidate = rank._select_next_micro_batch([wave, [_request(99)]], 0)
    assert candidate.inputs == [wave]
    assert candidate.indices == (0,)
    assert candidate.stats_global_count == 1
    assert candidate.plan.subforward_count == 4
    assert candidate.check.estimated_required_bytes == 10


def test_nonmemory_validation_is_not_overridden(monkeypatch):
    rank = _oversized(monkeypatch)
    with pytest.raises(ValueError, match="exceeds vocabulary"):
        rank.dp_rank_forward(
            [tr.ForwardInput(input_tokens=torch.tensor([1]), top_k=1000)]
        )


def _reporting_rank(monkeypatch, tmp_path):
    rank = _rank(monkeypatch)
    rank._planner_reporter = Reporter(5, spool_dir=tmp_path)
    plan = rank._plan_flat_forward([_request(0)])
    monkeypatch.setattr(
        rank, "_estimate_required_memory_bytes_from_values", lambda **k: 220
    )
    counters = {"allocated": 100, "peak": 100, "syncs": 0}
    monkeypatch.setattr(rank, "device", torch.device("cuda:0"))
    monkeypatch.setattr(tr, "_telemetry_phase", lambda *a, **k: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def synchronize(*a):
        counters["syncs"] += 1

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda *a: counters["allocated"]
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: counters["peak"])
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda *a: counters.update(peak=counters["allocated"]),
    )
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda *a: {"num_ooms": 1})

    def execute(plan):
        counters.update(allocated=150, peak=350)
        return [
            tr.ForwardOutput(torch.tensor([1.0]), None, None, None)
            for _ in range(plan.request_count)
        ]

    monkeypatch.setattr(rank, "_execute_flat_plan", execute)
    return rank, plan, counters


def _records(root: Path):
    return [json.loads(path.read_text()) for path in root.rglob("*.json")]


def test_report_uses_local_prediction_before_profile_update(monkeypatch, tmp_path):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(9999, 10000, True), context="dp_rank_forward"
    )
    counters["peak"] = 500  # caller backward exceeds the earlier forward peak
    rank._complete_planner_observation()
    [record] = _records(tmp_path)
    assert record["predicted_peak_bytes"] == 200
    assert record["admission_peak_bytes"] == 9999
    assert record["observed_peak_bytes"] == 400
    assert record["replay"]["memory_replay"]["estimates"][0]["profile"] is None
    assert (
        record["replay"]["requests"][0]["input_tokens"]
        == _request(0).input_tokens.tolist()
    )
    assert counters["syncs"] == 2  # unchanged forward's two existing syncs
    rank.finish_planner_observation()
    assert len(_records(tmp_path)) == 1


def test_caught_backward_oom_is_reported_once_without_completed_peak(
    monkeypatch, tmp_path
):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    error = torch.cuda.OutOfMemoryError("caller backward allocation")
    rank.report_planner_oom(error)
    rank.report_planner_oom(error)
    rank.finish_planner_observation()
    [record] = _records(tmp_path)
    assert record["oom"]
    assert record["observed_peak_bytes"] is None
    assert record["error_pct"] is None
    assert record["partial_peak_bytes"] == 250
    assert record["replay"]["oom"]["message"] == str(error)
    assert record["phase"] == "forward_and_caller"
    assert counters["syncs"] == 2


def test_forward_oom_report_preserves_original_cause(monkeypatch, tmp_path):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    error = torch.cuda.OutOfMemoryError("forward allocation")

    def fail(plan):
        counters["peak"] = 180
        raise error

    monkeypatch.setattr(rank, "_execute_flat_plan", fail)
    with pytest.raises(tr.TrainerRankMemoryError) as raised:
        rank._run_flat_plan_with_memory_tracking(
            plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
        )
    assert raised.value.__cause__ is error
    [record] = _records(tmp_path)
    assert record["phase"] == "forward"
    assert record["partial_peak_bytes"] == 80
    assert counters["syncs"] == 1


def test_snapshot_failure_still_persists_minimal_oom(monkeypatch, tmp_path):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    monkeypatch.setattr(rank, "_plan_cost", lambda plan: 1 / 0)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    rank.report_planner_oom(torch.cuda.OutOfMemoryError("backward"))
    [record] = _records(tmp_path)
    assert record["oom"] and not record["replay_complete"]
    assert "planner_snapshot_unavailable" in record["incomplete_reasons"]


def test_nonoom_does_not_mint_oom_report(monkeypatch, tmp_path):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    rank.report_planner_oom(RuntimeError("unrelated failure"))
    rank.discard_planner_observation()
    assert not _records(tmp_path)


@pytest.mark.parametrize("oom", [False, True])
def test_delayed_report_keeps_original_request_options(monkeypatch, tmp_path, oom):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    request = plan.groups[0].items[0].request
    original = {
        "top_k": request.top_k,
        "logits": request.logits,
        "hidden_states": request.hidden_states,
        "no_grad": request.no_grad,
        "checkpoint": str(request.checkpoint),
    }
    output_bytes = plan.output_bytes
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    # Caller/backward work can reuse the mutable ForwardInput before emission.
    request.top_k = 9
    request.logits = request.hidden_states = request.no_grad = True
    request.checkpoint = "replacement"
    if oom:
        rank.report_planner_oom(torch.cuda.OutOfMemoryError("caller backward"))
    else:
        rank._complete_planner_observation()
    rank.finish_planner_observation()
    [record] = _records(tmp_path)
    replay = record["replay"]
    assert {name: replay["requests"][0][name] for name in original} == original
    assert replay["memory_replay"]["estimates"][0]["arguments"]["output_bytes"] == (
        output_bytes
    )
    assert record["replay_complete"] is False
    assert "immutable runtime" in record["incomplete_reasons"][0]
    assert record["oom"] is oom
    assert counters["syncs"] == 2


def test_changed_tokens_mark_replay_incomplete(monkeypatch, tmp_path):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    plan.groups[0].items[0].input_ids[0] = 123
    rank._complete_planner_observation()
    [record] = _records(tmp_path)
    assert not record["replay_complete"]
    assert "device_or_modified_input" in record["incomplete_reasons"]


def test_disabled_reports_do_not_sample_extra_counters(monkeypatch, tmp_path):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    rank._planner_reporter = Reporter(None, spool_dir=tmp_path)
    monkeypatch.setattr(
        rank,
        "_plan_cost",
        lambda plan: pytest.fail("disabled observation built snapshot"),
    )
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    rank.finish_planner_observation()
    assert not _records(tmp_path)
    assert counters["syncs"] == 2


def test_mixed_rank_flags_do_not_add_divergent_split_pricing(monkeypatch):
    rank = _oversized(monkeypatch)
    monkeypatch.setattr(rank, "_try_cache_recovery", lambda *a, **k: False)
    seen = []
    monkeypatch.setattr(
        rank,
        "_recovery_reduce",
        lambda values, **kw: seen.append((values, kw)) or [0.0],
    )
    original = rank._plan_flat_forward
    planned = []

    def plan(*args, **kwargs):
        planned.append(len(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(rank, "_plan_flat_forward", plan)
    with pytest.raises(tr.TrainerRankMemoryError):
        rank.dp_rank_forward([_request(i) for i in range(4)])
    assert planned == [4, 4]  # disabled peer keeps lower-bound pruning everywhere
    assert all(entry[1]["sync_across_dp"] is False for entry in seen)


def test_fitting_path_adds_no_option_agreement_collective(monkeypatch):
    rank = _oversized(monkeypatch, limit=100)
    monkeypatch.setattr(
        rank,
        "_recovery_reduce",
        lambda *a, **k: pytest.fail("new normal-path collective"),
    )
    _recording_executor(monkeypatch, rank)
    rank.dp_rank_forward([_request(0)])


def test_iterator_close_preserves_pending_backward_oom_context(monkeypatch, tmp_path):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda sample=None: 10000)
    iterator = rank.forward_micro_batches([_request(0)])
    next(iterator)
    iterator.close()
    assert rank._planner_observation is not None
    assert not _records(tmp_path)
    rank.report_planner_oom(
        torch.cuda.OutOfMemoryError("backward after generator close")
    )
    [record] = _records(tmp_path)
    assert record["oom"]


def test_split_report_keeps_first_child_peak_across_resets(monkeypatch, tmp_path):
    rank, first, counters = _reporting_rank(monkeypatch, tmp_path)
    second = rank._plan_flat_forward([_request(1)])
    split = tr._SplitForwardPlan((first, second), ((0,), (1,)), 2)
    executions = iter([(150, 700), (170, 300)])

    def execute(plan):
        allocated, peak = next(executions)
        counters.update(allocated=allocated, peak=peak)
        return [tr.ForwardOutput(torch.tensor([1.0]), None, None, None)]

    monkeypatch.setattr(rank, "_execute_flat_plan", execute)
    rank._execute_split_plan_with_memory_tracking(
        split, check=tr._MemoryCheck(440, 10000, True), context="dp_rank_forward"
    )
    counters["peak"] = 350
    rank._complete_planner_observation()
    [record] = _records(tmp_path)
    assert record["predicted_peak_bytes"] == 400
    assert record["observed_peak_bytes"] == 600
    assert record["replay"]["subforward_request_indices"] == [[0], [1]]
    assert counters["syncs"] == 4


def test_volatile_budget_uses_smaller_materialized_candidate(monkeypatch):
    rank = _oversized(monkeypatch, limit=100)
    # Force the maintained search to materialize its widths rather than use
    # cheap scalar estimates. All fitting plans are real considered options.
    monkeypatch.setattr(rank, "_estimate_flat_forward", lambda *a, **k: None)
    items = [_request(i) for i in range(4)]
    selected = rank._search_next_micro_batch(items, 0)
    assert isinstance(selected, tr._CandidateMicroBatch)
    assert selected.stats_global_count == 4
    assert selected.fallback is not None
    assert selected.fallback.indices == (0,)
    assert selected.fallback.check.estimated_required_bytes == 10
    monkeypatch.setattr(
        rank,
        "_memory_check_required",
        lambda required, **k: tr._MemoryCheck(required, 1, False),
    )
    result = rank._recover_admission(
        lambda: selected,
        lambda value: (value.plan, value.check),
        lambda value, check: tr.replace(value, check=check),
        context="forward_micro_batches",
        sync_across_dp=True,
        admit_refusal=lambda refusal: pytest.fail("lost original candidate mapping"),
    )
    assert result is selected.fallback or result == selected.fallback
    assert result.inputs == [items[0]]
    assert result.plan.request_count == 1


def test_interleaved_execution_scopes_preserve_both_oom_inputs(monkeypatch, tmp_path):
    rank, first, _ = _reporting_rank(monkeypatch, tmp_path)
    second = rank._plan_flat_forward([_request(1)])
    one, two = {}, {}
    check = tr._MemoryCheck(220, 10000, True)
    with rank.planner_observation_scope(one):
        rank._run_flat_plan_with_memory_tracking(
            first, check=check, context="dp_rank_forward"
        )
    with rank.planner_observation_scope(two):
        rank.discard_planner_observation()  # new actor execution cannot erase one
        rank._run_flat_plan_with_memory_tracking(
            second, check=check, context="dp_rank_forward"
        )
    with rank.planner_observation_scope(one):
        rank.report_planner_oom(torch.cuda.OutOfMemoryError("first backward"))
    with rank.planner_observation_scope(two):
        rank.report_planner_oom(torch.cuda.OutOfMemoryError("second backward"))
    records = _records(tmp_path)
    assert len(records) == 2
    assert {r["replay"]["requests"][0]["input_tokens"][-1] for r in records} == {0, 1}
    assert all(
        r["partial_peak_bytes"] is None and r["observed_peak_bytes"] is None
        for r in records
    )
    assert all(
        "overlapping_forward_memory_window" in r["replay"]["window_reasons"]
        and r["replay"]["measurement_valid"] is False
        and r["replay_complete"] is False
        and "immutable runtime" in r["incomplete_reasons"][0]
        for r in records
    )
    assert not rank._planner_active_observations


def test_overlapping_scope_does_not_claim_completed_comparison(monkeypatch, tmp_path):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    one, two = {}, {}
    check = tr._MemoryCheck(220, 10000, True)
    for context in (one, two):
        with rank.planner_observation_scope(context):
            rank._run_flat_plan_with_memory_tracking(
                plan, check=check, context="dp_rank_forward"
            )
    for context in (one, two):
        with rank.planner_observation_scope(context):
            rank._complete_planner_observation()
            rank.finish_planner_observation()
    assert not _records(tmp_path)
    assert not rank._planner_active_observations


def test_sequential_scopes_keep_independent_completed_peaks(monkeypatch, tmp_path):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    for _ in range(2):
        counters.update(allocated=100, peak=100)
        with rank.planner_observation_scope({}):
            rank._run_flat_plan_with_memory_tracking(
                plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
            )
            rank._complete_planner_observation()
    assert len(_records(tmp_path)) == 2
    assert not rank._planner_active_observations


@pytest.mark.parametrize("local_refusal", [50, 150])
def test_dp_refusal_prices_share_world_scope_before_best_choice(
    monkeypatch, local_refusal
):
    rank = _oversized(monkeypatch, limit=1)
    items = [_request(i) for i in range(4)]
    wide = rank._plan_flat_forward(items)
    small = rank._plan_flat_forward(items[:1])
    candidate = tr._CandidateMicroBatch(
        items, (0, 1, 2, 3), wide, tr._MemoryCheck(100, 200, True), 4, 0, False
    )
    refusal = tr._ForwardRefusal(
        small, tr._MemoryCheck(local_refusal, 1, False), "minimum wave refused"
    )
    searches = iter((candidate, refusal))
    prices = []

    def world_price(required, *, sync_across_dp):
        prices.append((required, sync_across_dp))
        return tr._MemoryCheck(100 if len(prices) == 1 else 150, 1, False)

    monkeypatch.setattr(rank, "_memory_check_required", world_price)
    monkeypatch.setattr(rank, "_try_cache_recovery", lambda *a, **k: False)
    result = rank._recover_admission(
        lambda: next(searches),
        lambda value: (value.plan, value.check),
        lambda value, check: tr.replace(value, check=check),
        context="forward_micro_batches",
        sync_across_dp=True,
        admit_refusal=lambda refused: tr.replace(
            candidate, plan=refused.plan, check=refused.check, stats_global_count=1
        ),
    )
    assert prices == [(100, True), (local_refusal, True)]
    assert result.stats_global_count == 4
    assert result.plan is wide


def test_dp_disagreeing_fallback_widths_refuse_before_execution(monkeypatch):
    rank = _oversized(monkeypatch, limit=1)
    items = [_request(0)]
    plan = rank._plan_flat_forward(items)
    refusal = tr._ForwardRefusal(plan, tr._MemoryCheck(10, 1, False), "refused")
    monkeypatch.setattr(rank, "_try_cache_recovery", lambda *a, **k: False)
    monkeypatch.setattr(rank, "_recovery_reduce", lambda *a, **k: [1.0, 1.0, -4.0])
    with pytest.raises(tr.TrainerRankMemoryError):
        rank._recover_admission(
            lambda: refusal,
            lambda value: (value.plan, value.check),
            lambda value, check: value,
            context="forward_micro_batches",
            sync_across_dp=True,
            admit_refusal=lambda r: tr._CandidateMicroBatch(
                items, (0,), r.plan, r.check, 1, 0, True
            ),
        )


def test_dp_forward_reports_at_forward_boundary_not_later_optimizer(
    monkeypatch, tmp_path
):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    monkeypatch.setattr(
        rank,
        "_plan_admissible_forward",
        lambda *a, **k: (plan, tr._MemoryCheck(220, 10000, True)),
    )
    rank.dp_rank_forward([_request(0)])
    [record] = _records(tmp_path)
    assert record["observed_peak_bytes"] == 250
    assert record["phase"] == "forward"
    assert rank._planner_active_observations
    assert rank._planner_observation["window_open"] is False
    counters["peak"] = 10000  # unrelated later optimizer allocation
    rank.finish_planner_observation()
    assert _records(tmp_path) == [record]
    assert counters["syncs"] == 2


def test_closed_forward_keeps_backward_oom_without_false_partial_peak(
    monkeypatch, tmp_path
):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    rank._execute_admitted_plan(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    counters["peak"] = 10000
    rank.report_planner_oom(torch.cuda.OutOfMemoryError("later backward"))
    records = _records(tmp_path)
    assert len(records) == 2
    [oom] = [record for record in records if record["oom"]]
    assert oom["phase"] == "caller_after_profile"
    assert oom["partial_peak_bytes"] is None
    assert oom["observed_peak_bytes"] is None and oom["error_pct"] is None
    assert oom["replay"]["window_reasons"] == ["oom_after_profiled_window"]
    assert oom["replay_complete"] is False
    assert "immutable runtime" in oom["incomplete_reasons"][0]
    assert not rank._planner_active_observations


def test_execution_finish_does_not_complete_abandoned_microbatch_window(
    monkeypatch, tmp_path
):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="forward_micro_batches"
    )
    counters["peak"] = 10000
    rank.finish_planner_observation()
    assert not _records(tmp_path)
    assert not rank._planner_active_observations


def test_abandoned_execution_does_not_poison_future_measurements(monkeypatch, tmp_path):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    abandoned = {}
    with rank.planner_observation_scope(abandoned):
        rank._run_flat_plan_with_memory_tracking(
            plan,
            check=tr._MemoryCheck(220, 10000, True),
            context="forward_micro_batches",
        )
    assert rank._planner_active_observations
    del abandoned
    gc.collect()
    assert not rank._planner_active_observations
    counters.update(allocated=100, peak=100)
    with rank.planner_observation_scope({}):
        rank._execute_admitted_plan(
            plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
        )
    [record] = _records(tmp_path)
    assert record["observed_peak_bytes"] == 250


def test_diagnostic_registry_failure_does_not_replace_forward_oom(
    monkeypatch, tmp_path
):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    error = torch.cuda.OutOfMemoryError("original allocation failure")

    def execute(plan):
        del rank._planner_active_observations
        raise error

    monkeypatch.setattr(rank, "_execute_flat_plan", execute)
    with pytest.raises(tr.TrainerRankMemoryError) as raised:
        rank._run_flat_plan_with_memory_tracking(
            plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
        )
    assert raised.value.__cause__ is error
    rank.finish_planner_observation()  # diagnostic cleanup must also be auxiliary


def test_complete_diagnostic_failure_and_cancellation_semantics(monkeypatch, tmp_path):
    rank, plan, _ = _reporting_rank(monkeypatch, tmp_path)
    rank._run_flat_plan_with_memory_tracking(
        plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
    )
    del rank._planner_overlap_generation
    rank._complete_planner_observation()  # ordinary diagnostic AttributeError is contained
    assert not _records(tmp_path)

    def cancel():
        raise KeyboardInterrupt("do not swallow cancellation")

    monkeypatch.setattr(rank, "discard_planner_observation", cancel)
    with pytest.raises(KeyboardInterrupt):
        rank.finish_planner_observation()


def test_closed_dp_context_invalidates_other_executions_open_caller_window(
    monkeypatch, tmp_path
):
    rank, plan, counters = _reporting_rank(monkeypatch, tmp_path)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda sample=None: 10000)
    one, two = {}, {}
    with rank.planner_observation_scope(one):
        rank._execute_admitted_plan(
            plan, check=tr._MemoryCheck(220, 10000, True), context="dp_rank_forward"
        )
    [forward_report] = _records(tmp_path)
    assert forward_report["observed_peak_bytes"] == 250
    assert one["observation"]["window_open"] is False

    iterator = rank.forward_micro_batches([_request(1)])
    with rank.planner_observation_scope(two):
        next(iterator)
    # A resumes caller backward while B's yielded microbatch interval is open.
    # No new forward occurs to announce that allocator contamination.
    with rank.planner_observation_scope(one):
        counters["peak"] = 10000
        rank.finish_planner_observation()
    with rank.planner_observation_scope(two):
        with pytest.raises(StopIteration):
            next(iterator)
        rank.finish_planner_observation()
    assert _records(tmp_path) == [forward_report]
    assert not rank._planner_active_observations
