"""Driver-owned calibration phases around local work and orchestration votes.

An open phase starts with one entry vote and ends with one exit vote. Before
an existing orchestration collective, every peer exits its current local
phase; a local failure joins that exit instead of entering a different vote.
After the collective, the next local phase starts. Finalizers never vote.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Callable

    from art.trainer_rank._impl import TrainerRank


OUTER_ERROR_PHASE = 1
_ACTIVE: ContextVar[Protocol | None] = ContextVar("calibration_protocol", default=None)


def _phase_payload(stage: str, failed: bool, execute: bool) -> list[float]:
    """Prepare local vote operands before touching the reserved control buffer."""
    return [float(failed), float(execute), -float(execute)]


def vote(rank: TrainerRank, values, phase: int, buffer=None) -> list[float]:
    """A fixed five-double envelope also tags callers' outer error votes."""
    count = len(values)
    if count > 3:
        raise ValueError("calibration control votes have at most three operands")
    if not (dist.is_available() and dist.is_initialized()):
        return list(values)
    if buffer is None:
        buffer = torch.empty(
            5, dtype=torch.float64, device=rank._forward_collective_device()
        )
    for index in range(3):
        buffer[index] = values[index] if index < count else 0
    buffer[3], buffer[4] = phase, -phase
    dist.all_reduce(
        buffer,
        op=dist.ReduceOp.MAX,
        group=getattr(rank, "_forward_replica_group", None),
    )
    result = buffer.tolist()
    if result[3] != -result[4]:
        active = _ACTIVE.get()
        error = RuntimeError("Calibration phase mismatch")
        if active is not None:
            active.open = False
            active.broken = True
            active.failure = error
        raise error
    return result[:count]


class Protocol:
    def __init__(self, rank: TrainerRank):
        self.rank = rank
        self.sequence = 0
        self.open = False
        self.broken = False
        self.failure: BaseException | None = None
        self.distributed = dist.is_available() and dist.is_initialized()
        # Healthy control transport is established before the first phase.
        # Subsequent entry/exit votes reuse this reserved buffer.
        self.buffer = (
            torch.empty(
                5, dtype=torch.float64, device=rank._forward_collective_device()
            )
            if self.distributed
            else None
        )

    def fail(self, error: BaseException | None):
        if not self.distributed and error is not None:
            raise error
        self.failure = RuntimeError("Calibration failed on one or more ranks")
        raise self.failure from error

    def _vote(self, stage: str, error=None, execute=False):
        try:
            values = _phase_payload(stage, error is not None, execute)
        except BaseException as exc:
            error = exc if error is None else error
            values = [1.0, float(execute), -float(execute)]
        tag = 1024 + self.sequence * 2 + (stage == "exit")
        if self.distributed:
            values = self.rank._recovery_reduce(
                values,
                op="MAX",
                sync_across_dp=True,
                _phase=tag,
                _buffer=self.buffer,
            )
        return values, error

    def start(self, error=None):
        assert not self.open and not self.broken
        sequence = self.sequence + 1
        try:
            self.sequence = sequence
            self.open = True
        except BaseException as exc:
            # Commit the same must-vote state if reservation was interrupted.
            self.sequence, self.open = sequence, True
            error = exc if error is None else error
        values, error = self._vote("entry", error)
        if values[0]:
            self.end(error or RuntimeError("Calibration entry preparation failed"))

    def end(self, error=None, *, execute=False):
        if not self.open:
            if self.failure is not None:
                raise self.failure
            if error is not None:
                raise error
            return
        self.open = False
        values, error = self._vote("exit", error, execute)
        if values[0] or values[1] != -values[2]:
            self.fail(error)

    def transition(self, *, execute=False):
        self.end(execute=execute)
        self.start()

    def collective(self, action, *, prepare=None):
        self.end()
        result = action()
        error = None
        if prepare is not None:
            try:
                result = prepare()
            except BaseException as exc:
                error = exc
        self.start(error)
        return result

    def abandon_local(self):
        if self.distributed:
            raise AssertionError("Calibration close must use the collective driver")
        self.open = False


class Driver:
    """One explicit owner, reused by nested preparation and callback scopes."""

    def __init__(self, rank: TrainerRank):
        self.rank = rank
        self.owner = False
        self.token = None

    def __enter__(self):
        active = _ACTIVE.get()
        if active is None:
            self.protocol = Protocol(self.rank)
            self.owner = True
            self.token = _ACTIVE.set(self.protocol)
            try:
                self.protocol.start()
            except BaseException:
                _ACTIVE.reset(self.token)
                raise
        else:
            assert active.rank is self.rank, (
                "Calibration cannot nest different trainers"
            )
            self.protocol = active
        return self.protocol

    def finish(self, error=None, *, execute=False):
        try:
            self.protocol.end(error, execute=execute)
            if not self.owner:
                self.protocol.start()
        finally:
            if self.owner and self.token is not None:
                _ACTIVE.reset(self.token)

    def __exit__(self, kind, error, traceback):
        if self.owner:
            self.finish(error)
        return False


def collective(action: Callable[[], Any]):
    active = _ACTIVE.get()
    return action() if active is None else active.collective(action)


def all_gather_object(object_list, obj, group=None, *, _finish=None):
    """Prepare object bytes in-phase, preserving the two tensor wire gathers."""
    from torch.distributed import distributed_c10d as c10d

    current = _ACTIVE.get()
    if current is None or c10d._rank_not_in_group(group):
        return collective(lambda: dist.all_gather_object(object_list, obj, group=group))
    device = c10d._get_object_coll_device(group)
    payload, local_size = c10d._object_to_tensor(obj, device, group)
    count = dist.get_world_size(group)
    sizes = torch.zeros(count, dtype=torch.long, device=device)
    size_list = [sizes[i].unsqueeze(0) for i in range(count)]

    current.collective(lambda: dist.all_gather(size_list, local_size, group=group))
    # The gathered maximum is needed before allocating either padded buffer.
    # Prepare both in the reopened phase, then vote before the payload gather.
    maximum = int(sizes.max().item())
    payload.resize_(maximum)
    storage = torch.empty(maximum * count, dtype=torch.uint8, device=device)
    outputs = [storage[maximum * i : maximum * (i + 1)] for i in range(count)]

    def finish():
        for index, tensor in enumerate(outputs):
            object_list[index] = c10d._tensor_to_object(tensor, size_list[index], group)
        return _finish() if _finish is not None else None

    if _finish is not None:
        # Transaction readiness includes decoding and result construction. Carry
        # their failures into the existing post-payload entry vote.
        return current.collective(
            lambda: dist.all_gather(outputs, payload, group=group), prepare=finish
        )
    current.collective(lambda: dist.all_gather(outputs, payload, group=group))
    finish()


def active() -> Protocol | None:
    return _ACTIVE.get()


def checkpoint() -> None:
    """Finish local preparation before entering the model execution boundary."""
    current = _ACTIVE.get()
    if current is not None:
        current.transition()


def owned(function):
    @wraps(function)
    def run(rank, *args, **kwargs):
        rank._guard_forward_collective(function.__name__)
        with Driver(rank):
            return function(rank, *args, **kwargs)

    return run
