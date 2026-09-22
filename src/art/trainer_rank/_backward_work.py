"""Private, bounded accounting for completed ordinary-engine backward phases.

The currency is host elapsed time, including intervening engine callbacks and
scheduling, gated by a completed CUDA tail. It is not CUDA kernel time. Observer
spans are charged conservatively, including spans also timed by recovery.
"""

from dataclasses import dataclass
from functools import wraps
import sys
import time
from typing import Any
import weakref

import torch

_MAX_NS = 2**63 - 1


def _measured(function):
    @wraps(function)
    def measured(self, *args, **kwargs):
        started = None
        try:
            started = time.perf_counter_ns()
            return function(self, *args, **kwargs)
        except BaseException:
            # Accounting must not replace a training error, cancellation or grad.
            self.disabled = True
        finally:
            self._charge(started)

    return measured


def region(function):
    """Exclude original forward/recovery work; meter only our transitions."""
    @wraps(function)
    def wrapped(rank, *args, **kwargs):
        work = rank._backward_work()
        entered = work.enter() if work is not None else False
        try:
            return function(rank, *args, **kwargs)
        finally:
            if entered:
                work.leave()

    return wrapped


@dataclass
class _Invocation:
    started: int
    blocked: bool
    ended: int | None = None
    tail: Any = None


class BackwardWork:
    def __init__(self, lock, device):
        self.lock = lock
        self.cost_ns = 0
        self.invalid = False
        started = time.perf_counter_ns()
        self.device = device
        self.work_ns = 0
        self.disabled = device.type != "cuda" or getattr(device, "index", None) is None
        self.closed = False
        self.depth = 0
        self.highest_task = -1
        # Object identity is the owner generation; callbacks hold only weak refs.
        self.rows: dict[int, _Invocation] = {}
        self.outputs: dict[int, tuple[Any, Any]] = {}
        self._charge(started)

    def _charge(self, started):
        try:
            ended = time.perf_counter_ns()
            with self.lock:
                if (
                    type(started) is not int
                    or type(ended) is not int
                    or not 0 <= started <= ended <= _MAX_NS
                    or self.cost_ns + ended - started > _MAX_NS
                ):
                    self.invalid = True
                else:
                    self.cost_ns += ended - started
        except BaseException:
            # Preserve the positive cost already recorded, never replace by zero.
            self.invalid = True

    def _ordinary(self):
        module = sys.modules.get("torch._dynamo.compiled_autograd")
        return not torch.compiler.is_compiling() and (
            module is None
            or not (
                getattr(module, "compiled_autograd_enabled", False)
                or getattr(module, "in_compiled_autograd_region", False)
            )
        )

    @_measured
    def attach(self, outputs):
        with self.lock:
            if self.disabled or self.closed:
                return
            if not self._ordinary():
                self.disabled = True
                return
            for key, (ref, handle) in list(self.outputs.items()):
                if ref() is None:
                    handle.remove()
                    del self.outputs[key]
            for output in outputs:
                top_k = output.top_k
                for tensor in (
                    output.target_logprobs,
                    output.logits,
                    output.hidden_states,
                    None if top_k is None else top_k.logprobs,
                ):
                    if tensor is None or not tensor.requires_grad:
                        continue
                    if tensor.device != self.device:
                        self.disabled = True
                        return
                    key = id(tensor)
                    if key in self.outputs and self.outputs[key][0]() is tensor:
                        continue
                    if len(self.outputs) >= 256:
                        self.disabled = True
                        return
                    ref = weakref.ref(self)

                    def hook(grad, ref=ref):
                        work = ref()
                        if work is not None:
                            work._start()
                        # Returning None preserves the original gradient object.

                    self.outputs[key] = (weakref.ref(tensor), tensor.register_hook(hook))

    @_measured
    def _start(self):
        with self.lock:
            if self.disabled or self.closed:
                return
            task = torch._C._current_graph_task_id()
            if not self._ordinary() or type(task) is not int or not 0 <= task < 2**31:
                self.disabled = True
                return
            if task in self.rows:
                if self.rows[task].ended is not None:
                    self.disabled = True
                return
            if task <= self.highest_task:
                self.disabled = True
                return
            if len(self.rows) >= 8:
                # Explicit bounded retirement of an already closed interval;
                # never guess that a pending/failed engine has completed.
                retired = next(
                    (key for key, row in self.rows.items() if row.ended is not None),
                    None,
                )
                if retired is None:
                    self.disabled = True
                    return
                del self.rows[retired]
            self.highest_task = task
            pending = [row for row in self.rows.values() if row.ended is None]
            for row in pending:
                row.blocked = True
            self.rows[task] = _Invocation(
                time.perf_counter_ns(), bool(self.depth or pending)
            )
            ref = weakref.ref(self)

            def finish():
                work = ref()
                if work is not None:
                    work._finish(task)

            torch.autograd.Variable._execution_engine.queue_callback(finish)

    @_measured
    def _finish(self, task):
        ended = time.perf_counter_ns()
        with self.lock:
            if self.disabled or self.closed:
                return
            row = self.rows[task]
            observed = torch._C._current_graph_task_id()
            if (
                not self._ordinary()
                or type(observed) is not int
                or observed != task
                or row.ended is not None
                or type(row.started) is not int
                or type(ended) is not int
                or not 0 <= row.started <= ended <= _MAX_NS
            ):
                self.disabled = True
                return
            tail = torch.cuda.Event(enable_timing=False)
            tail.record(torch.cuda.current_stream(self.device))
            row.ended, row.tail = ended, tail

    @_measured
    def enter(self):
        with self.lock:
            if self.depth >= 16:
                self.disabled = True
                return False
            self.depth += 1
            for row in self.rows.values():
                if row.ended is None:
                    row.blocked = True
            return True

    @_measured
    def leave(self):
        with self.lock:
            if self.depth <= 0:
                self.disabled = True
            else:
                self.depth -= 1

    @_measured
    def harvest(self):
        with self.lock:
            if self.disabled or self.closed:
                self.rows.clear()
                return
            if any(row.ended is None for row in self.rows.values()):
                # A failed or unresolved engine is not a quiescent frontier.
                return
            addition = 0
            retired = []
            for task, row in self.rows.items():
                ready = row.tail.query()
                if type(ready) is not bool:
                    self.disabled = True
                    return
                if ready and not row.blocked:
                    addition += row.ended - row.started
                if ready or row.blocked:
                    retired.append(task)
            if self.work_ns + addition > _MAX_NS:
                self.disabled = True
                return
            self.work_ns += addition
            # Preserve completed/unready rows until a later real query or bounded
            # capacity retirement. A later forward never blocks a closed interval.
            for task in retired:
                del self.rows[task]

    @_measured
    def close(self):
        with self.lock:
            self.closed = True
            self.rows.clear()
            for _, handle in self.outputs.values():
                handle.remove()
            self.outputs.clear()

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass
