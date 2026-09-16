"""Actual admission methods with scalar allocator/search/clock facades, no CUDA."""

from contextlib import nullcontext
import math
import os
import types
import unittest
from unittest.mock import patch

from art.trainer_rank import _impl

Refusal = _impl.TrainerRankMemoryError
Partial = _impl.TrainerRankPartialExecutionError
OOM = _impl.torch.cuda.OutOfMemoryError


class Scalar:
    def __init__(self, t, i):
        self.t, self.i = t, i

    def item(self):
        return self.t.data[self.i]


class Tensor:
    def __init__(self, data, **kw):
        self.data = data if isinstance(data, list) else [data]

    def __getitem__(self, i):
        return Scalar(self, i)

    def __setitem__(self, i, v):
        self.data[i] = v

    def tolist(self):
        return self.data

    def item(self):
        return self.data[0]


class CUDA:
    OutOfMemoryError = OOM

    def synchronize(self, device):
        self.events.append("sync")

    def reset_peak_memory_stats(self, device):
        self.events.append("reset")

    def max_memory_allocated(self, device):
        return self.allocated

    def __init__(self):
        self.free = 40
        self.total = 1000
        self.allocated = 0
        self.backend = "native"
        self.events = []
        self.failure = None

    def is_available(self):
        return True

    def get_allocator_backend(self):
        return self.backend

    def mem_get_info(self, device):
        self.events.append("sample")
        return self.free, self.total

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return self.allocated

    def empty_cache(self):
        self.events.append("release")
        if self.failure is not None:
            raise self.failure
        self.free = 200


class Clock:
    def __init__(self):
        self.value = 0.0
        self.next = None

    def perf_counter(self):
        if self.next is not None:
            return self.next
        self.value += 0.01
        return self.value


def plan(required=80):
    return types.SimpleNamespace(packed_tokens=required, logical_tokens=required)


def fail(ns, required=80):
    return ns["_ForwardRefusal"](
        plan(required),
        ns["_MemoryCheck"](required, 10, False),
        "smallest actual refusal",
    )


def success(ns, required=80):
    return (plan(required), ns["_MemoryCheck"](required, 170, True))


def run(q, items, *, sync=True):
    calls = []

    def search():
        calls.append(1)
        value = items[min(len(calls) - 1, len(items) - 1)]
        if isinstance(value, BaseException):
            raise value
        return value

    try:
        value = q._recover_admission(
            search,
            lambda v: v,
            lambda v, c: (v[0], c),
            context="forward_micro_batches" if sync else "dp_rank_forward",
            sync_across_dp=sync,
        )
    except BaseException as e:
        return None, e, len(calls)
    return value, None, len(calls)


class TestRecovery(unittest.TestCase):
    def make(self):
        cuda = CUDA()
        clock = Clock()
        for name, value in dict(
            torch=types.SimpleNamespace(
                cuda=cuda, tensor=Tensor, float64="f64", int32="i32"
            ),
            time=clock,
            dist=types.SimpleNamespace(
                is_available=lambda: False, is_initialized=lambda: False
            ),
            _telemetry_phase=lambda *a, **kw: nullcontext(),
            _TEST_HOOKS_ENV="CONTROL_ART_HOOK",
            _TEST_MEMORY_LIMIT_ENV="CONTROL_ART_LIMIT",
        ).items():
            patcher = patch.object(_impl, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        q = object.__new__(_impl.TrainerRank)
        q.device = types.SimpleNamespace(type="cuda")
        q._update_peak_memory_profile = lambda *a: None
        q._execute_flat_plan = lambda p: [object() for _ in range(p.request_count)]
        q._telemetry_signature = lambda p: {}
        q._telemetry_plan_signature = lambda p: {}
        q._snapshot_planning_telemetry = lambda *a: None
        q._forward_memory_group = lambda: None
        return (
            q,
            cuda,
            clock,
            dict(
                _MemoryCheck=_impl._MemoryCheck, _ForwardRefusal=_impl._ForwardRefusal
            ),
        )

    def test_first_once_and_fresh_return(self):
        q, c, k, n = self.make()
        v, e, calls = run(q, [fail(n), success(n)])
        self.assertIsNone(e)
        self.assertEqual(calls, 2)
        self.assertEqual(c.events.count("release"), 1)
        self.assertEqual(v[1].available_bytes, 170)
        self.assertTrue(q._recovery_state().first_consumed)

    def test_small_fit_without_release(self):
        q, c, k, n = self.make()
        c.free = 200
        v, e, count = run(q, [success(n)])
        self.assertIsNone(e)
        self.assertEqual(count, 1)
        self.assertNotIn("release", c.events)

    def test_one_bounded_smaller_refresh(self):
        q, c, k, n = self.make()
        v, e, count = run(q, [success(n, 192), success(n, 5)])
        self.assertIsNone(e)
        self.assertEqual(count, 2)
        self.assertEqual(v[1].available_bytes, 10)
        self.assertNotIn("release", c.events)
        self.assertGreater(q._recovery_state().cost, 0)
        self.assertFalse(q._recovery_state().first_consumed)

    def test_moving_final_counters_do_not_reclaim(self):
        q, c, k, n = self.make()
        v, e, count = run(q, [success(n, 192), success(n, 192)])
        self.assertIsInstance(e, Refusal)
        self.assertEqual(count, 2)
        self.assertNotIn("release", c.events)

    def test_final_refresh_then_exhaustion_then_one_recovery(self):
        q, c, k, n = self.make()
        v, e, count = run(q, [success(n, 192), fail(n), success(n)])
        self.assertIsNone(e)
        self.assertEqual(count, 3)
        self.assertEqual(c.events.count("release"), 1)

    def test_search_runtime_error_never_recovered(self):
        q, c, k, n = self.make()
        original = RuntimeError("peer runtime error")
        v, e, count = run(q, [original])
        self.assertIs(e, original)
        self.assertEqual(count, 1)
        self.assertNotIn("release", c.events)

    def test_failed_release_original_and_consumed(self):
        for typ in (RuntimeError, KeyboardInterrupt, SystemExit):
            q, c, k, n = self.make()
            original = typ("release error")
            c.failure = original
            v, e, count = run(q, [fail(n), success(n)])
            self.assertIs(e, original)
            self.assertEqual(count, 1)
            self.assertTrue(q._recovery_state().first_consumed)
            self.assertIsNone(q._recovery_state().owner)

    def test_second_failed_fit_no_second_release(self):
        q, c, k, n = self.make()
        v, e, count = run(q, [fail(n), fail(n, 200)])
        self.assertIsInstance(e, Refusal)
        self.assertEqual(count, 2)
        self.assertEqual(c.events.count("release"), 1)

    def test_lower_after_release_refuses(self):
        q, c, k, n = self.make()

        def release():
            c.events.append("release")
            c.free = 31

        c.empty_cache = release
        v, e, count = run(q, [fail(n), success(n, 80)])
        self.assertIsInstance(e, Refusal)
        self.assertEqual(e.usable_limit_bytes, 1)
        self.assertEqual(c.events.count("release"), 1)

    def test_quota_stops_with_persistent_first_debt(self):
        q, c, k, n = self.make()
        run(q, [fail(n), success(n)])
        c.free = 40
        v, e, count = run(q, [fail(n), success(n)])
        self.assertIsInstance(e, Refusal)
        self.assertEqual(count, 1)
        self.assertEqual(c.events.count("release"), 1)
        self.assertGreater(q._recovery_state().cost, 0)

    def test_completed_forward_earns_next_trial(self):
        q, c, k, n = self.make()
        run(q, [fail(n), success(n)])
        q._record_recovery_work("dp_rank_forward", 2.0)
        c.free = 40
        v, e, count = run(q, [fail(n), success(n)])
        self.assertIsNone(e)
        self.assertEqual(c.events.count("release"), 2)

    def test_no_first_trial_per_entrypoint(self):
        q, c, k, n = self.make()
        run(q, [fail(n), success(n)])
        c.free = 40
        v, e, count = run(q, [fail(n), success(n)], sync=False)
        self.assertIsInstance(e, Refusal)
        self.assertEqual(c.events.count("release"), 1)

    def test_invalid_clock_does_not_release(self):
        for x in (math.nan, math.inf):
            q, c, k, n = self.make()
            k.next = x
            v, e, count = run(q, [fail(n), success(n)])
            self.assertIsInstance(e, Refusal)
            self.assertNotIn("release", c.events)
            self.assertTrue(q._recovery_state().invalid)

    def test_test_cap_never_triggers_release_even_physical_low(self):
        for free in (40, 200):
            q, c, k, n = self.make()
            c.free = free
            os.environ["CONTROL_ART_HOOK"] = "1"
            os.environ["CONTROL_ART_LIMIT"] = "20"
            try:
                v, e, count = run(q, [fail(n), success(n)])
            finally:
                os.environ.pop("CONTROL_ART_HOOK")
                os.environ.pop("CONTROL_ART_LIMIT")
            self.assertIsInstance(e, Refusal)
            self.assertNotIn("release", c.events)

    def test_sufficient_space_after_refusal_retries_without_release(self):
        q, c, k, n = self.make()
        c.free = 200
        v, e, count = run(q, [fail(n), success(n)])
        self.assertIsNone(e)
        self.assertEqual(count, 2)
        self.assertNotIn("release", c.events)

    def test_original_over_timing_secondary(self):
        q, c, k, n = self.make()
        original = RuntimeError("release original")
        c.failure = original
        old = k.perf_counter
        calls = []

        def timer():
            calls.append(1)
            if len(calls) > 2:
                raise ValueError("secondary timer")
            return old()

        k.perf_counter = timer
        v, e, count = run(q, [fail(n), success(n)])
        self.assertIs(e, original)
        self.assertTrue(q._recovery_state().invalid)
        self.assertIsNone(q._recovery_state().owner)

    def test_no_work_for_unowned_context(self):
        q, c, k, n = self.make()
        q._record_recovery_work("arbitrary caller time", 100.0)
        self.assertEqual(q._recovery_state().work, 0)

    def test_foreign_owner_is_retained(self):
        q, c, k, n = self.make()
        foreign = object()
        q._recovery_state().owner = foreign
        v, e, count = run(q, [fail(n), success(n)])
        self.assertIsInstance(e, Refusal)
        self.assertIs(q._recovery_state().owner, foreign)
        self.assertTrue(q._recovery_state().invalid)
        self.assertNotIn("release", c.events)

    def test_cap_refusal_in_later_trial_does_not_use_work(self):
        q, c, k, n = self.make()
        run(q, [fail(n), success(n)])
        q._record_recovery_work("dp_rank_forward", 100)
        c.free = 40
        os.environ["CONTROL_ART_HOOK"] = "1"
        os.environ["CONTROL_ART_LIMIT"] = "20"
        try:
            v, e, count = run(q, [fail(n), success(n)])
        finally:
            os.environ.pop("CONTROL_ART_HOOK")
            os.environ.pop("CONTROL_ART_LIMIT")
        self.assertIsInstance(e, Refusal)
        self.assertEqual(c.events.count("release"), 1)

    def test_fresh_pre_release_fit_skips_call_consumes_trial(self):
        q, c, k, n = self.make()
        reads = []
        old = c.mem_get_info

        def sample(d):
            reads.append(1)
            if len(reads) >= 3:
                c.free = 200
            return old(d)

        c.mem_get_info = sample
        v, e, count = run(q, [fail(n), success(n)])
        self.assertIsNone(e)
        self.assertNotIn("release", c.events)
        self.assertTrue(q._recovery_state().first_consumed)

    def test_overflowing_work_disables_recovery(self):
        q, c, k, n = self.make()
        q._record_recovery_work("dp_rank_forward", 1e308)
        q._record_recovery_work("dp_rank_forward", 1e308)
        self.assertTrue(q._recovery_state().invalid)
        v, e, count = run(q, [fail(n), success(n)])
        self.assertNotIn("release", c.events)

    def test_finite_inputs_overflow_derived_quota(self):
        q, c, k, n = self.make()
        state = q._recovery_state()
        state.first_consumed = True
        state.high = 1e308
        state.cost = 1e308
        state.work = 1e308
        v, e, count = run(q, [fail(n), success(n)])
        self.assertTrue(state.invalid)
        self.assertNotIn("release", c.events)

    def test_forward_clock_failures_preserve_success(self):
        for failing_call in (1, 2):
            q, c, k, n = self.make()
            old = k.perf_counter
            calls = []

            def timer():
                calls.append(1)
                if len(calls) == failing_call:
                    raise ValueError("instrumentation only")
                return old()

            k.perf_counter = timer
            p = plan()
            p.request_count = 1
            outputs, baseline = q._run_flat_plan_with_memory_tracking(
                p, check=n["_MemoryCheck"](80, 170, True), context="dp_rank_forward"
            )
            self.assertEqual(len(outputs), 1)
            self.assertTrue(q._recovery_state().invalid)
            self.assertEqual(q._recovery_state().work, 0)

    def test_forward_recording_failure_preserves_success(self):
        q, c, k, n = self.make()
        p = plan()
        p.request_count = 1

        def record(*a):
            raise ValueError("recording only")

        q._record_recovery_work = record
        outputs, baseline = q._run_flat_plan_with_memory_tracking(
            p, check=n["_MemoryCheck"](80, 170, True), context="dp_rank_forward"
        )
        self.assertEqual(len(outputs), 1)
        self.assertTrue(q._recovery_state().invalid)

    def test_execution_oom_keeps_admission_and_cause(self):
        q, c, k, n = self.make()
        p = plan()
        p.request_count = 1
        check = n["_MemoryCheck"](80, 170, True)
        original = OOM("execution")

        def execute(p):
            raise original

        q._execute_flat_plan = execute
        try:
            q._run_flat_plan_with_memory_tracking(
                p, check=check, context="dp_rank_forward"
            )
        except Refusal as e:
            self.assertIs(e.__cause__, original)
            self.assertEqual(e.usable_limit_bytes, check.available_bytes)
            self.assertEqual(e.predicted_peak_bytes, check.estimated_required_bytes)
        else:
            self.fail("expected exact original wrapped OOM")
        self.assertNotIn("release", c.events)
        self.assertEqual(q._recovery_state().work, 0)

    def test_split_children_credit_once_and_rollback(self):
        for fail_at in (None, 2):
            q, c, k, n = self.make()
            p = plan()
            p.request_count = 1
            count = []

            def execute(p):
                count.append(1)
                if len(count) == fail_at:
                    raise OOM("second child")
                return [object()]

            q._execute_flat_plan = execute
            split = types.SimpleNamespace(
                subforwards=(p, p),
                request_indices=((0,), (1,)),
                request_count=2,
                subforward_count=2,
            )
            state = q._recovery_state()
            state.work = 1.0
            try:
                outputs, baseline, peak = q._execute_split_plan_with_memory_tracking(
                    split,
                    check=n["_MemoryCheck"](80, 170, True),
                    context="dp_rank_forward",
                )
            except Partial:
                self.assertEqual(fail_at, 2)
                self.assertEqual(state.work, 1.0)
            else:
                self.assertIsNone(fail_at)
                self.assertAlmostEqual(state.work, 1.02)

    def test_split_mapping_error_rolls_back(self):
        q, c, k, n = self.make()
        p = plan()
        p.request_count = 2
        split = types.SimpleNamespace(
            subforwards=(p,),
            request_indices=((0,),),
            request_count=1,
            subforward_count=1,
        )
        state = q._recovery_state()
        state.work = 1.0
        with self.assertRaises(ValueError):
            q._execute_split_plan_with_memory_tracking(
                split, check=n["_MemoryCheck"](80, 170, True), context="dp_rank_forward"
            )
        self.assertEqual(state.work, 1.0)

    def test_cross_entrypoint_progress_with_work_and_no_new_first_trial(self):
        q, c, k, n = self.make()
        run(q, [fail(n), success(n)], sync=True)
        q._record_recovery_work("dp_rank_forward", 2.0)
        c.free = 40
        v, e, count = run(q, [fail(n), success(n)], sync=False)
        self.assertIsNone(e)
        self.assertEqual(c.events.count("release"), 2)

    def test_normal_dp_local_has_no_added_check(self):
        q, c, k, n = self.make()
        value = success(n)
        q._find_admissible_forward = lambda *a, **kw: value

        def forbidden(*a, **kw):
            raise AssertionError("unnecessary added admission collective")

        q._memory_check_required = forbidden
        result = q._plan_admissible_forward(
            [], checkpoint=None, context="dp_rank_forward"
        )
        self.assertIs(result[1], value[1])
        self.assertNotIn("release", c.events)

    def test_final_bookkeeping_failure_preserves_success(self):
        q, c, k, n = self.make()
        calls = []

        def search():
            calls.append(1)
            if len(calls) == 1:
                return fail(n)
            q._recovery_state().high = object()
            return success(n)

        result = q._recover_admission(
            search,
            lambda v: v,
            lambda v, c: (v[0], c),
            context="forward_micro_batches",
            sync_across_dp=True,
        )
        self.assertTrue(result[1].fits)
        self.assertTrue(q._recovery_state().invalid)
        self.assertIsNone(q._recovery_state().owner)

    def test_owner_cleanup_keeps_original_runtime_error(self):
        for secondary_type in (KeyboardInterrupt, SystemExit):
            q, c, k, n = self.make()
            state = q._recovery_state()
            original = RuntimeError("original release failure")
            secondary = secondary_type("owner cleanup cancellation")
            c.failure = original

            class Lock:
                calls = 0

                def __enter__(self):
                    self.calls += 1
                    if self.calls == 4:
                        raise secondary

                def __exit__(self, *args):
                    return False

            state.lock = Lock()
            _, error, _ = run(q, [fail(n), success(n)])
            self.assertIs(error, original)
            self.assertEqual(state.lock.calls, 4)
            self.assertTrue(state.invalid)

    def test_first_bookkeeping_cancellation_keeps_identity(self):
        q, c, k, n = self.make()
        state = q._recovery_state()
        original = KeyboardInterrupt("first bookkeeping cancellation")
        secondary = SystemExit("second owner cleanup cancellation")

        class Lock:
            calls = 0

            def __enter__(self):
                self.calls += 1
                if self.calls == 3:
                    raise original
                if self.calls == 4:
                    raise secondary

            def __exit__(self, *args):
                return False

        state.lock = Lock()
        _, error, _ = run(q, [fail(n), success(n)])
        self.assertIs(error, original)
        self.assertEqual(state.lock.calls, 4)
        self.assertTrue(state.invalid)

    def test_ordinary_owner_cleanup_failure_keeps_success(self):
        q, c, k, n = self.make()
        state = q._recovery_state()

        class Lock:
            calls = 0

            def __enter__(self):
                self.calls += 1
                if self.calls == 4:
                    raise RuntimeError("ordinary owner cleanup diagnostic")

            def __exit__(self, *args):
                return False

        state.lock = Lock()
        value, error, _ = run(q, [fail(n), success(n)])
        self.assertIsNone(error)
        self.assertTrue(value[1].fits)
        self.assertTrue(state.invalid)

    def test_sampling_and_reduction_errors_keep_both_diagnostics(self):
        for phase in ("pre_sample", "release", "post_sample"):
            for mode in ("local", "transport", "both"):
                with self.subTest(phase=phase, mode=mode):
                    q, c, k, n = self.make()
                    local = ValueError("local sample or release failed")
                    cause, context = TypeError("original cause"), LookupError("context")
                    local.__cause__, local.__context__ = cause, context
                    local.__suppress_context__ = True
                    local.add_note("existing note")
                    transport = RuntimeError("memory reduction failed")
                    transport.__cause__ = OSError("transport cause")
                    samples, reductions = [], []

                    def available():
                        samples.append(1)
                        if mode != "transport" and (
                            phase == "pre_sample"
                            and len(samples) == 1
                            or phase == "post_sample"
                            and len(samples) == 2
                        ):
                            raise local
                        return 10

                    def reduce(values, *, op, sync_across_dp):
                        reductions.append((op, list(values), sync_across_dp))
                        if mode != "local" and len(reductions) == (
                            3 if phase == "pre_sample" else 4
                        ):
                            raise transport
                        return values

                    q._available_memory_bytes = available
                    q._recovery_reduce = reduce
                    if phase == "release" and mode != "transport":
                        c.failure = local
                    _, error, calls = run(q, [fail(n), success(n)])
                    self.assertIs(error, transport if mode == "transport" else local)
                    if mode != "transport":
                        self.assertIs(local.__cause__, cause)
                        self.assertIs(local.__context__, context)
                        self.assertTrue(local.__suppress_context__)
                        self.assertEqual(local.__notes__[0], "existing note")
                        self.assertEqual(
                            len(local.__notes__), 2 if mode == "both" else 1
                        )
                        if mode == "both":
                            self.assertIn(
                                "OSError: transport cause", local.__notes__[1]
                            )
                            self.assertIn(
                                "RuntimeError: memory reduction failed",
                                local.__notes__[1],
                            )
                            self.assertIn("raise transport", local.__notes__[1])
                    self.assertEqual(calls, 1)
                    self.assertEqual(
                        [x[0] for x in reductions],
                        ["SUM", "MAX", "MIN"]
                        + ([] if phase == "pre_sample" else ["MIN"]),
                    )
                    state = q._recovery_state()
                    self.assertIsNone(state.owner)
                    self.assertEqual(state.work, 0)
                    self.assertEqual(state.cost, state.high)
                    self.assertGreater(state.cost, 0)
                    self.assertEqual(state.first_consumed, phase != "pre_sample")

    def test_admission_sampling_and_reduction_error_keep_original_chain(self):
        q, c, k, n = self.make()
        local = ValueError("admission sample")
        context, cause = LookupError("context"), TypeError("cause")
        local.__context__, local.__cause__ = context, cause
        transport = RuntimeError("admission reduction")
        calls = []

        def available():
            raise local

        def all_reduce(value, *, op, group):
            calls.append(op)
            if op == "MIN":
                raise transport

        q._available_memory_bytes = available
        with patch.object(
            _impl,
            "dist",
            types.SimpleNamespace(
                is_available=lambda: True,
                is_initialized=lambda: True,
                ReduceOp=types.SimpleNamespace(MAX="MAX", MIN="MIN"),
                all_reduce=all_reduce,
            ),
        ):
            with self.assertRaises(ValueError) as captured:
                q._memory_check_required(80, sync_across_dp=True)
        self.assertIs(captured.exception, local)
        self.assertIs(local.__context__, context)
        self.assertIs(local.__cause__, cause)
        self.assertIn("RuntimeError: admission reduction", local.__notes__[0])
        self.assertEqual(calls, ["MAX", "MIN"])

    def test_secondary_note_failure_never_replaces_primary(self):
        primary, secondary = ValueError("primary"), RuntimeError("secondary")
        context = LookupError("original context")
        primary.__context__ = context
        helper = _impl.TrainerRank._memory_error_with_reduction_note
        with patch.object(
            _impl.traceback, "format_exception", side_effect=SystemExit("renderer")
        ):
            self.assertIs(helper(primary, secondary), primary)
        self.assertIs(primary.__context__, context)
        primary.__dict__["__notes__"] = 42
        self.assertIs(helper(primary, secondary), primary)
        self.assertEqual(primary.__notes__, 42)
        self.assertIs(primary.__context__, context)
