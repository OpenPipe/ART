"""Prospective bounded accounting controls; no Torch/model/device import.

Load the actual helper with a scalar Torch facade, and extract the actual
recovery/rollback methods rather than copy their decision arithmetic.
"""

import ast
from asyncio import CancelledError
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import gc
import importlib.util
import math
import os
from pathlib import Path
import sys
import threading
import traceback
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[2]


class Clock:
    value = 1000

    def perf_counter_ns(self):
        self.value += 1
        return self.value


class Tensor:
    def __init__(self, device):
        self.device, self.requires_grad = device, True
        self.hooks = {}

    def register_hook(self, callback):
        key = len(self.hooks)
        self.hooks[key] = callback
        ref = weakref.ref(self)

        def remove():
            tensor = ref()
            if tensor is not None:
                tensor.hooks.pop(key, None)

        return NS(remove=remove)


class CUDA:
    def __init__(self):
        self.events = []
        self.free, self.releases = 0, 0
        self.failure = None

    def Event(self, *, enable_timing):
        assert enable_timing is False
        if self.failure is not None:
            raise self.failure
        event = NS(ready=False, stream=None, queries=0)

        def query():
            event.queries += 1
            return event.ready

        event.query = query
        event.record = lambda stream: setattr(event, "stream", stream)
        self.events.append(event)
        return event

    def current_stream(self, device):
        return ("caller", device.index)

    def is_available(self):
        return True

    def get_allocator_backend(self):
        return "native"

    def mem_get_info(self, device):
        return self.free, 1000

    def memory_allocated(self, device):
        return 0

    def memory_reserved(self, device):
        return 100

    def empty_cache(self):
        self.releases += 1
        self.free = 500


class TestBackwardWork(unittest.TestCase):
    def setUp(self):
        self.clock, self.cuda = Clock(), CUDA()
        self.task, self.callbacks = 0, []
        self.device = NS(type="cuda", index=0)
        self.torch = NS(
            cuda=self.cuda,
            compiler=NS(is_compiling=lambda: False),
            _C=NS(_current_graph_task_id=lambda: self.task),
            autograd=NS(
                Variable=NS(
                    _execution_engine=NS(
                        queue_callback=self.callbacks.append,
                    )
                )
            ),
        )
        name = "_art_backward_work_control"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "src/art/trainer_rank/_backward_work.py"
        )
        assert spec is not None and spec.loader is not None
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": self.torch, name: self.module}):
            spec.loader.exec_module(self.module)
        setattr(self.module, "time", self.clock)
        self.work = self.module.BackwardWork(threading.RLock(), self.device)
        self.addCleanup(self.work.close)

    def output(self, tensor):
        return NS(
            target_logprobs=tensor,
            logits=tensor,
            hidden_states=tensor,
            top_k=NS(logprobs=tensor),
        )

    def start(self, task):
        self.task = task
        self.work._start()

    def finish(self, task, *, ready=True):
        self.task = task
        callback = self.callbacks.pop(0)
        callback()
        if self.cuda.events:
            self.cuda.events[-1].ready = ready

    def completed(self, task=1, *, ready=True):
        self.start(task)
        self.clock.value += 1000
        self.finish(task, ready=ready)
        return self.work.rows[task].ended - self.work.rows[task].started

    def actual_rank(self):
        source = ast.parse((ROOT / "src/art/trainer_rank/_impl.py").read_text())
        names = {"_CacheRecoveryState", "_MemoryCheck"}
        methods = {
            "_recovery_state",
            "_backward_work",
            "_try_cache_recovery",
            "_cache_recovery_episode",
            "_release_cached_memory_for_backward",
            "_memory_error_with_reduction_note",
            "_execute_split_plan_with_memory_tracking",
        }
        selected = [
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name in names
        ]
        trainer = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == "TrainerRank"
        )
        trainer.body = [
            node
            for node in trainer.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        ns = dict(
            __name__=__name__,
            dataclass=dataclass,
            dataclass_field=field,
            threading=threading,
            contextmanager=contextmanager,
            traceback=traceback,
            BackwardWork=self.module.BackwardWork,
            _backward_region=self.module.region,
            torch=self.torch,
            math=math,
            os=os,
            time=self.clock,
            cast=lambda typ, value: value,
            TrainerRankMemoryError=type("MemoryRefusal", (RuntimeError,), {}),
            _TEST_HOOKS_ENV="ART_BACKWARD_CONTROL_ONLY",
            _TEST_MEMORY_LIMIT_ENV="ART_BACKWARD_CONTROL_LIMIT",
            _MEMORY_RESERVE_FRACTION=0.05,
        )
        tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *selected,
                trainer,
            ],
            type_ignores=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(tree), "actual-accounting-methods", "exec"
            ),
            ns,
        )
        rank = object.__new__(ns["TrainerRank"])
        rank.device = self.device
        state = rank._recovery_state()
        self.work.lock = state.lock
        state.backward = self.work
        rank._recovery_clock = lambda: 1.0
        rank._available_memory_bytes = lambda: self.cuda.free
        return rank, state, ns

    def test_fixed_endpoint_unready_survives_later_forward_and_idle(self):
        duration = self.completed(ready=False)
        ended = self.work.rows[1].ended
        self.work.harvest()
        self.assertEqual(self.work.work_ns, 0)
        cost = self.work.cost_ns
        with self.work.region():
            self.clock.value += 10**12
        self.assertLess(self.work.cost_ns - cost, 1000)
        self.assertEqual(self.work.depth, 0)
        self.assertFalse(self.work.rows[1].blocked)
        self.assertEqual(self.work.rows[1].ended, ended)
        self.cuda.events[0].ready = True
        self.work.harvest()
        self.assertEqual(self.work.work_ns, duration)
        self.work.harvest()
        self.assertEqual(self.work.work_ns, duration)
        self.assertEqual(self.cuda.events[0].stream, ("caller", 0))

    def test_four_fields_repeated_attach_and_multiple_outputs_queue_once(self):
        first, second = Tensor(self.device), Tensor(self.device)
        for _ in range(2):
            self.work.attach([self.output(first), self.output(second)])
        self.assertEqual((len(first.hooks), len(second.hooks)), (1, 1))
        self.task = 1
        gradient = object()
        for tensor in (first, second):
            self.assertIsNone(next(iter(tensor.hooks.values()))(gradient))
        self.assertEqual(len(self.callbacks), 1)
        self.finish(1)
        self.work.harvest()
        self.assertGreater(self.work.work_ns, 0)

    def test_retained_graph_new_ids_and_retired_id_cannot_revive_credit(self):
        expected = self.completed(1)
        self.work.harvest()
        expected += self.completed(2)
        self.work.harvest()
        self.assertEqual(self.work.work_ns, expected)
        self.start(1)
        self.assertTrue(self.work.disabled)
        self.work.harvest()
        self.assertEqual(self.work.work_ns, expected)

    def test_nested_and_forward_overlap_retire_zero(self):
        self.start(1)
        self.start(2)
        # The engine callbacks may complete inner first.
        self.callbacks.reverse()
        self.finish(2)
        self.finish(1)
        self.work.harvest()
        self.assertEqual(self.work.work_ns, 0)
        with self.work.region():
            self.completed(3)
        self.work.harvest()
        self.assertEqual(self.work.work_ns, 0)
        self.start(4)
        with self.work.region():
            pass
        self.finish(4)
        self.work.harvest()
        self.assertEqual(self.work.work_ns, 0)

    def test_failed_pending_blocks_commit_without_clock_timeout(self):
        self.start(1)  # Node failure means its completion callback never arrives.
        self.callbacks.clear()
        self.completed(2)
        self.clock.value += 10**12
        self.work.harvest()
        self.assertEqual(self.work.work_ns, 0)
        self.assertIn(1, self.work.rows)

    def test_bounded_unready_retirement_and_long_stream_keep_committed_total(self):
        committed = self.completed(1)
        self.work.harvest()
        for task in range(2, 12):
            self.completed(task, ready=False)
        self.assertEqual(len(self.work.rows), 8)
        self.assertNotIn(2, self.work.rows)
        self.assertFalse(self.work.disabled)
        for event in self.cuda.events:
            event.ready = True
        committed += sum(row.ended - row.started for row in self.work.rows.values())
        self.work.harvest()
        self.assertEqual(self.work.work_ns, committed)
        for task in range(12, 1012):
            committed += self.completed(task)
            self.work.harvest()
            self.assertEqual(len(self.work.rows), 0)
        self.assertEqual(self.work.work_ns, committed)

    def test_pending_overflow_preserves_prior_credit_and_positive_cost(self):
        committed = self.completed(1)
        self.work.harvest()
        for task in range(2, 11):
            self.start(task)
        self.assertTrue(self.work.disabled)
        self.assertLessEqual(len(self.work.rows), 8)
        cost = self.work.cost_ns
        self.work.harvest()
        self.assertEqual(self.work.work_ns, committed)
        self.assertGreaterEqual(self.work.cost_ns, cost)

    def test_original_exception_identity_cause_and_observer_fault_neutrality(self):
        primary, cause = KeyboardInterrupt("original"), ValueError("cause")
        rank = NS(_backward_work=lambda: self.work)

        @self.module.region
        def fail(rank):
            self.cuda.failure = RuntimeError("diagnostic tail")
            self.start(1)
            self.finish(1)
            raise primary from cause

        with self.assertRaises(KeyboardInterrupt) as caught:
            fail(rank)
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cause)
        self.assertTrue(self.work.disabled)
        self.assertGreater(self.work.cost_ns, 0)
        self.assertEqual(self.work.depth, 0)

    def test_ordinary_queue_fault_does_not_replace_gradient_or_erase_previous_credit(
        self,
    ):
        expected = self.completed(1)
        self.work.harvest()
        tensor = Tensor(self.device)
        self.work.attach([self.output(tensor)])
        self.task = 2
        with patch.object(
            self.torch.autograd.Variable._execution_engine,
            "queue_callback",
            side_effect=RuntimeError("queue"),
        ):
            self.assertIsNone(next(iter(tensor.hooks.values()))(object()))
        self.assertTrue(self.work.disabled)
        self.work.harvest()
        self.assertEqual(self.work.work_ns, expected)

    def test_new_cancellation_at_each_observer_boundary_propagates_exact_object(self):
        original_work = self.work
        for error_type in (
            KeyboardInterrupt,
            SystemExit,
            GeneratorExit,
            CancelledError,
        ):
            for stage in ("attach", "hook", "callback", "harvest", "leave", "close"):
                with self.subTest(error=error_type, stage=stage):
                    self.work = self.module.BackwardWork(threading.RLock(), self.device)
                    self.callbacks.clear()
                    self.task = 1
                    primary, cause = error_type(stage), ValueError("original cause")
                    primary.__cause__ = cause
                    tensor = Tensor(self.device)
                    if stage == "attach":
                        target, name = tensor, "register_hook"
                        action = lambda: self.work.attach([self.output(tensor)])
                    elif stage == "hook":
                        self.work.attach([self.output(tensor)])
                        target = self.torch.autograd.Variable._execution_engine
                        name = "queue_callback"
                        action = lambda: next(iter(tensor.hooks.values()))(object())
                    elif stage == "callback":
                        self.start(1)
                        target, name = self.cuda, "Event"
                        action = self.callbacks.pop(0)
                    elif stage == "harvest":
                        self.completed(1)
                        target, name = self.cuda.events[-1], "query"
                        action = self.work.harvest
                    elif stage == "leave":
                        scope = self.work.region()
                        scope.__enter__()
                        target, name = self.clock, "perf_counter_ns"
                        action = lambda: scope.__exit__(None, None, None)
                    else:
                        self.work.attach([self.output(tensor)])
                        target, name = (
                            next(iter(self.work.outputs.values()))[1],
                            "remove",
                        )
                        action = self.work.close
                    with patch.object(target, name, side_effect=primary):
                        with self.assertRaises(BaseException) as caught:
                            action()
                    self.assertIs(caught.exception, primary)
                    self.assertIs(primary.__cause__, cause)
                    self.assertTrue(self.work.disabled)
                    self.assertEqual(self.work.work_ns, 0)
                    self.assertEqual(self.work.depth, 0)
                    self.work.close()
        self.work = original_work

    def test_meter_propagates_new_cancellation_and_preserves_inflight_primary(self):
        primary, secondary = KeyboardInterrupt("queue"), SystemExit("meter")
        cause = ValueError("cause")
        primary.__cause__ = cause
        self.task = 1
        with patch.object(
            self.clock, "perf_counter_ns", side_effect=[2000, 2001, secondary]
        ):
            with patch.object(
                self.torch.autograd.Variable._execution_engine,
                "queue_callback",
                side_effect=primary,
            ):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    self.work._start()
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cause)
        self.assertTrue(self.work.invalid)
        # With no primary, a newly delivered meter cancellation must escape.
        for action in (lambda: self.work._charge(2000), self.work.harvest):
            self.work.disabled = False
            self.work.rows.clear()
            clock = [secondary] if action != self.work.harvest else [3000, secondary]
            with patch.object(self.clock, "perf_counter_ns", side_effect=clock):
                with self.assertRaises(SystemExit) as caught:
                    action()
            self.assertIs(caught.exception, secondary)

    def test_actual_lookup_and_constructor_cancellation_identity(self):
        rank, state, ns = self.actual_rank()
        primary, secondary = CancelledError("lookup"), GeneratorExit("meter")
        cause = ValueError("lookup cause")
        primary.__cause__ = cause
        with patch.object(
            self.clock, "perf_counter_ns", side_effect=[primary, secondary]
        ):
            with self.assertRaises(CancelledError) as caught:
                rank._backward_work()
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cause)
        self.assertTrue(state.invalid)
        self.assertTrue(self.work.invalid)
        state.invalid, state.backward = False, None

        def construction(*args):
            raise primary

        with patch.dict(ns, BackwardWork=construction):
            with self.assertRaises(CancelledError) as caught:
                rank._backward_work()
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cause)
        self.assertTrue(state.invalid)

    def test_region_cleanup_preserves_body_error_but_new_cancellation_escapes(self):
        rank = NS(_backward_work=lambda: self.work)
        secondary = KeyboardInterrupt("leave")
        for primary in (None, ValueError("body"), CancelledError("body cancellation")):
            with self.subTest(primary=primary):
                cause = RuntimeError("body cause")

                @self.module.region
                def body(rank):
                    if primary is not None:
                        raise primary from cause
                    return "original result"

                with patch.object(
                    self.clock,
                    "perf_counter_ns",
                    side_effect=[10000, 10001, secondary, 10003],
                ):
                    with self.assertRaises(BaseException) as caught:
                        body(rank)
                self.assertIs(
                    caught.exception, secondary if primary is None else primary
                )
                self.assertEqual(self.work.depth, 0)
                if primary is not None:
                    self.assertIs(primary.__cause__, cause)
        # Enter has incremented its own count when its final charge cancels.
        # Cleanup must remove only that count, retaining an existing outer one.
        called = []

        @self.module.region
        def untouched(rank):
            called.append(True)

        with self.work.region():
            self.assertEqual(self.work.depth, 1)
            with patch.object(
                self.clock,
                "perf_counter_ns",
                side_effect=[20000, secondary, 20002, 20003],
            ):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    untouched(rank)
            self.assertIs(caught.exception, secondary)
            self.assertEqual(self.work.depth, 1)
            self.assertEqual(called, [])
            scope = self.work.region()
            scope.__enter__()
            self.assertEqual(self.work.depth, 2)
            another = SystemExit("secondary exit meter")
            with patch.object(
                self.clock, "perf_counter_ns", side_effect=[secondary, another]
            ):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    scope.__exit__(None, None, None)
            self.assertIs(caught.exception, secondary)
            self.assertEqual(self.work.depth, 1)
        self.assertEqual(self.work.depth, 0)

    def handoff_entry_failure(
        self, stage, primary, *, foreign_owner=False, exchange_error=None
    ):
        self.work = self.module.BackwardWork(threading.RLock(), self.device)
        self.addCleanup(self.work.close)
        rank, state, _ = self.actual_rank()
        state.owner = original_owner = object() if foreign_owner else None
        state.work, state.cost = 17.0, 3.0
        self.work.work_ns = 19
        secondary = KeyboardInterrupt(stage)
        votes = []

        def reduce(values, **kwargs):
            votes.append((values, kwargs))
            if exchange_error is not None:
                raise exchange_error
            return values

        rank._recovery_reduce = reduce
        clock = self.clock.perf_counter_ns
        count = 0

        def observer_clock():
            nonlocal count
            count += 1
            if (
                count
                == {"lookup_clock": 1, "region_clock": 3, "region_charge": 4}[stage]
            ):
                raise secondary
            return clock()

        with ExitStack() as patches:
            if stage in ("lookup_clock", "region_clock", "region_charge"):
                patches.enter_context(
                    patch.object(self.clock, "perf_counter_ns", observer_clock)
                )
            elif stage in ("lookup_state", "episode_state"):
                values = (
                    [secondary, state]
                    if stage == "lookup_state"
                    else [state, secondary]
                )
                patches.enter_context(
                    patch.object(rank, "_recovery_state", side_effect=values)
                )
            elif stage == "episode_clock":
                patches.enter_context(
                    patch.object(rank, "_recovery_clock", side_effect=[secondary, 2.0])
                )
            else:
                # Cancel after owner assignment, including while a foreign
                # episode owns the state: cleanup must never clear its token.
                lock = state.lock
                cancel_exit = False

                def episode_clock():
                    nonlocal cancel_exit
                    if count == 0:
                        cancel_exit = True
                    return 1.0

                class Lock:
                    def __enter__(self):
                        return lock.__enter__()

                    def __exit__(self, *args):
                        nonlocal cancel_exit, count
                        result = lock.__exit__(*args)
                        if cancel_exit:
                            cancel_exit = False
                            count += 1
                            raise secondary
                        return result

                state.lock = self.work.lock = Lock()
                rank._recovery_clock = episode_clock
            with self.assertRaises(BaseException) as caught:
                rank._release_cached_memory_for_backward(
                    NS(groups=[NS(grad_enabled=True)]), error=primary
                )
        self.assertIs(caught.exception, secondary if primary is None else primary)
        self.assertEqual(
            votes,
            []
            if primary is None
            else [([1.0, 1.0], {"op": "MAX", "sync_across_dp": True})],
        )
        self.assertIs(state.owner, original_owner)
        self.assertEqual(self.work.depth, 0)
        self.assertEqual((state.work, self.work.work_ns), (17.0, 19))
        self.assertGreaterEqual(state.cost, 3.0)
        self.assertEqual(self.cuda.releases, 0)
        self.assertTrue(state.invalid or self.work.invalid or self.work.disabled)

    def test_saved_forward_error_reaches_failure_vote_despite_entry_cancellation(self):
        for stage in (
            "lookup_state",
            "lookup_clock",
            "region_clock",
            "region_charge",
            "episode_state",
            "episode_clock",
            "episode_owner",
        ):
            for error_type in (RuntimeError, CancelledError):
                with self.subTest(stage=stage, error_type=error_type):
                    primary = error_type("forward")
                    primary.__cause__ = cause = ValueError("original cause")
                    primary.__context__ = context = LookupError("original context")
                    self.handoff_entry_failure(stage, primary)
                    self.assertIs(primary.__cause__, cause)
                    self.assertIs(primary.__context__, context)
                    self.assertTrue(primary.__suppress_context__)
        self.handoff_entry_failure(
            "episode_owner", RuntimeError("forward"), foreign_owner=True
        )
        primary = RuntimeError("forward before poisoned exchange")
        self.handoff_entry_failure(
            "episode_clock", primary, exchange_error=SystemExit("exchange cancellation")
        )
        self.assertIn("exchange cancellation", "\n".join(primary.__notes__))

    def test_new_entry_cancellation_propagates_and_cleans_owned_state(self):
        for stage in (
            "lookup_state",
            "lookup_clock",
            "region_clock",
            "region_charge",
            "episode_state",
            "episode_clock",
            "episode_owner",
        ):
            with self.subTest(stage=stage):
                self.handoff_entry_failure(stage, None)
        self.handoff_entry_failure("episode_owner", None, foreign_owner=True)

    def test_recovery_episode_still_records_cost_and_preserves_foreign_owner(self):
        rank, state, _ = self.actual_rank()
        state.cost, state.high = 3.0, 0.25
        for foreign_owner in (None, object()):
            with self.subTest(foreign_owner=foreign_owner):
                state.owner = foreign_owner
                with patch.object(rank, "_recovery_clock", side_effect=[10.0, 10.5]):
                    with rank._cache_recovery_episode() as (owner, started):
                        self.assertEqual(started, 10.0)
                        self.assertIs(
                            state.owner,
                            owner if foreign_owner is None else foreign_owner,
                        )
                self.assertIs(state.owner, foreign_owner)
                self.assertFalse(state.invalid)
        self.assertEqual((state.cost, state.high), (4.0, 0.5))

    def test_clock_fault_cost_overflow_and_unknown_readiness_fail_closed(self):
        prior = self.work.cost_ns
        with patch.object(
            self.clock, "perf_counter_ns", side_effect=RuntimeError("clock")
        ):
            self.work.harvest()
        self.assertTrue(self.work.invalid)
        self.assertEqual(self.work.cost_ns, prior)
        self.work.invalid = False
        self.work.cost_ns = self.module._MAX_NS
        self.work.harvest()
        self.assertTrue(self.work.invalid)
        self.assertEqual(self.work.cost_ns, self.module._MAX_NS)

    def test_wrong_device_mode_id_and_nonbool_readiness_withhold(self):
        for task in (-1, 2**31, True):
            with self.subTest(task=task):
                self.work.disabled = False
                self.start(task)
                self.assertTrue(self.work.disabled)
        self.work.disabled = False
        tensor = Tensor(NS(type="cuda", index=1))
        self.work.attach([self.output(tensor)])
        self.assertTrue(self.work.disabled)
        self.assertFalse(tensor.hooks)
        self.work.disabled = False
        with patch.dict(
            sys.modules,
            {"torch._dynamo.compiled_autograd": NS(compiled_autograd_enabled=True)},
        ):
            self.start(1)
        self.assertTrue(self.work.disabled)
        self.work.disabled = False
        self.completed(2)
        self.cuda.events[-1].ready = 1
        self.work.harvest()
        self.assertTrue(self.work.disabled)
        self.assertEqual(self.work.work_ns, 0)

    def test_weak_outputs_owner_and_close_remove_only_owned_hooks(self):
        tensor = Tensor(self.device)
        tensor.register_hook(lambda grad: grad)
        self.work.attach([self.output(tensor)])
        self.assertEqual(len(tensor.hooks), 2)
        self.work.close()
        self.assertEqual(len(tensor.hooks), 1)
        ref = weakref.ref(tensor)
        del tensor
        gc.collect()
        self.assertIsNone(ref())
        owner = self.module.BackwardWork(threading.RLock(), self.device)
        tensor = Tensor(self.device)
        owner.attach([self.output(tensor)])
        ref = weakref.ref(owner)
        del owner
        gc.collect()
        self.assertIsNone(ref())
        self.assertNotIn("__del__", vars(self.module.BackwardWork))
        # No Python finalizer can intercept cancellation. Caller-retained tensors
        # may keep a bounded weak hook; it is inert after the rank owner is gone.
        self.assertEqual(len(tensor.hooks), 1)
        self.assertIsNone(next(iter(tensor.hooks.values()))(object()))

    def test_actual_reducer_uses_sum_cost_and_max_local_sum_not_sum_of_maxima(self):
        rank, state, ns = self.actual_rank()
        state.work, state.cost, state.high = 100.0, 4.0, 1.0
        state.first_consumed, state.owner = True, object()
        calls = []

        def reduce(values, *, op, sync_across_dp):
            calls.append((op, list(values)))
            if op == "SUM":
                return [values[0] + 3.0, values[1] + 1.0]
            if op == "MAX":
                # Other rank has F=0, B=100; MAX(local F+B)=100, not 200.
                return [values[0], max(values[1], 100.0), values[2], values[3]]
            return values

        rank._recovery_reduce = reduce
        result = rank._try_cache_recovery(
            ns["_MemoryCheck"](80, 0, False),
            sync_across_dp=True,
            owner=state.owner,
            started=0.0,
        )
        self.assertFalse(result)
        self.assertEqual([op for op, _ in calls], ["SUM", "MAX", "MIN"])
        self.assertEqual(calls[1][1][1], 100.0)
        self.assertGreater(calls[0][1][0], 5.0)  # Original C+elapsed plus measured O.
        self.assertEqual(self.cuda.releases, 0)

    def test_actual_first_repeat_rule_and_invalid_meter(self):
        rank, state, ns = self.actual_rank()
        rank._recovery_reduce = lambda values, **kw: values
        state.owner, state.high = object(), 0.1
        check = ns["_MemoryCheck"](80, 0, False)
        self.assertTrue(
            rank._try_cache_recovery(
                check, sync_across_dp=False, owner=state.owner, started=0.0
            )
        )  # Original first free release.
        self.cuda.free = 0
        self.assertFalse(
            rank._try_cache_recovery(
                check, sync_across_dp=False, owner=state.owner, started=0.0
            )
        )
        # Use a genuinely completed host interval in this scalar fixture.
        self.start(1)
        self.clock.value += 30_000_000_000
        self.finish(1)
        self.assertTrue(
            rank._try_cache_recovery(
                check, sync_across_dp=False, owner=state.owner, started=0.0
            )
        )
        self.cuda.free = 0
        self.work.invalid = True
        self.assertFalse(
            rank._try_cache_recovery(
                check, sync_across_dp=False, owner=state.owner, started=0.0
            )
        )

    def test_actual_split_rollback_keeps_b_and_o_and_primary_error(self):
        rank, state, ns = self.actual_rank()
        self.completed()
        self.work.harvest()
        credited, cost = self.work.work_ns, self.work.cost_ns
        state.work = 5.0
        primary = ValueError("second subforward")
        count = 0

        def execute(*args, **kw):
            nonlocal count
            count += 1
            state.work += 10.0
            if count == 2:
                raise primary
            return [object()], None

        rank._run_flat_plan_with_memory_tracking = execute
        plan = NS(
            request_count=2,
            subforwards=[object(), object()],
            request_indices=[[0], [1]],
            subforward_count=2,
        )
        with self.assertRaises(ValueError) as caught:
            rank._execute_split_plan_with_memory_tracking(
                plan, check=None, context="test"
            )
        self.assertIs(caught.exception, primary)
        self.assertEqual(state.work, 5.0)
        self.assertEqual(self.work.work_ns, credited)
        self.assertGreater(self.work.cost_ns, cost)

    def test_original_invalid_forward_guard_cannot_be_masked_by_positive_b(self):
        rank, state, ns = self.actual_rank()
        rank._recovery_reduce = lambda values, **kw: values
        state.owner, state.work, state.high = object(), -1.0, 0.1
        self.start(1)
        self.clock.value += 30_000_000_000
        self.finish(1)
        self.assertFalse(
            rank._try_cache_recovery(
                ns["_MemoryCheck"](80, 0, False),
                sync_across_dp=False,
                owner=state.owner,
                started=0.0,
            )
        )
        self.assertTrue(state.invalid)
        self.assertEqual(self.cuda.releases, 0)


if __name__ == "__main__":
    unittest.main()
