"""Logical callback leaders and ordered physical-rank participation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Generator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
import inspect
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast
import weakref

import cloudpickle
import torch
import torch.distributed as dist

from . import _impl, _transport

if TYPE_CHECKING:
    from . import TrainerRank
    from ._heads import ModuleHandle

Mode = Literal["rank", "zero"]
T = TypeVar("T")


def _coordinate_call(call: Callable[[], T], *, group: dist.ProcessGroup | None) -> T:
    result, error = None, None
    try:
        result = call()
    except BaseException as exc:
        error = exc
    failures = [None if error is None else f"{type(error).__name__}: {error}"]
    if dist.is_initialized():
        local = failures[0]
        failures = [None] * dist.get_world_size(group)
        dist.all_gather_object(failures, local, group=group)
    if any(failures):
        if error is not None:
            raise error
        raise RuntimeError(f"Physical trainer preflight failed: {failures}")
    return cast(T, result)


@dataclass(frozen=True)
class RankCallbackResult:
    logical_rank: int | None
    value: Any = None


@dataclass(frozen=True)
class _Command:
    sequence: int
    operation: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    grad_enabled: bool


def _encode_command(command: _Command) -> bytes:
    return _transport.encode(command)


@dataclass(frozen=True)
class _OutputPacket:
    packet: Any
    cpu: tuple[bool, ...]
    managed: bool


@dataclass
class _Release:
    completed: asyncio.Future[None]
    gathered: list[Any]
    finish: Callable[[_Release], None]


@dataclass
class _State:
    sequence: int = 0
    graphs: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    exports: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    collector: Any = None
    released: set[str] = field(default_factory=set)
    iterators: dict[str, Iterator[Any]] = field(default_factory=dict)
    batch_inputs: dict[str, Any] = field(default_factory=dict)
    control_groups: dict[tuple[int, ...], dist.ProcessGroup] = field(
        default_factory=dict
    )
    release_group: dist.ProcessGroup | None = None
    pending_release: _Release | None = None
    release_error: str | None = None


def get_rank_callback_metadata(rank: TrainerRank) -> int | None:
    """Logical DP index on its leader; None on internal TP/CP participants."""
    if not dist.is_initialized():
        return 0
    from megatron.core import parallel_state as ps

    if ps.get_tensor_model_parallel_rank() or ps.get_context_parallel_rank():
        return None
    return rank._dp_rank_and_size()[0]


def rank_callback_leader(rank: TrainerRank, *, mode: Mode = "rank") -> bool:
    return (
        not dist.is_initialized() or dist.get_rank() == 0
        if mode == "zero"
        else get_rank_callback_metadata(rank) is not None
    )


async def join_rank_callback_release(
    rank: TrainerRank,
) -> asyncio.CancelledError | None:
    """Settle prior callback cleanup, returning any deferred cancellation."""
    state: _State | None = getattr(rank, "_rank_command_state", None)
    if state is None:
        return None
    cancelled = None
    if release := state.pending_release:
        while True:
            try:
                await asyncio.shield(release.completed)
                break
            except asyncio.CancelledError as error:
                if release.completed.cancelled():
                    break
                cancelled = error
            except Exception:
                break  # Finalization records and reports the terminal error.
        # A done callback may still be queued; ownership must be settled
        # synchronously before this actor can execute another command.
        release.finish(release)
    if state.release_error is not None:
        raise RuntimeError(state.release_error)
    return cancelled


class _Executor:
    def __init__(self, rank: TrainerRank, mode: Mode) -> None:
        from ._tensors import CotangentCollector

        if mode not in ("rank", "zero"):
            raise ValueError(f"Unknown callback mode {mode!r}")
        self.rank, self.mode = rank, mode
        self.group = None
        self.distributed = dist.is_initialized()
        if self.distributed and mode == "rank":
            from megatron.core import parallel_state as ps

            self.group = ps.get_tensor_and_context_parallel_group()
        self.members = (
            dist.get_process_group_ranks(self.group)
            if self.group is not None
            else list(range(dist.get_world_size()))
            if self.distributed
            else [0]
        )
        self.leader = self.members[0]
        self.is_leader = not self.distributed or dist.get_rank() == self.leader
        self.dp_rank = rank._dp_rank_and_size()[0]
        state = getattr(rank, "_rank_command_state", None)
        if state is None:
            state = _State(collector=CotangentCollector())
            setattr(rank, "_rank_command_state", state)
        self.state: _State = state
        if self.distributed and state.release_group is None:
            state.release_group = dist.new_group(backend="gloo")
        if self.distributed and dist.get_backend(self.group) != "gloo":
            key = tuple(self.members)
            if key not in state.control_groups:
                state.control_groups[key] = dist.new_group(
                    self.members, backend="gloo", use_local_synchronization=True
                )
            self.group = state.control_groups[key]
        self.iterators: dict[int, Generator[Any, None, None]] = {}
        self.stopped = False

    def _start_release(self) -> None:
        if self.state.release_error is not None:
            raise RuntimeError(self.state.release_error)
        if self.state.pending_release is not None:
            raise RuntimeError("Previous callback release is still pending")
        synchronize_heads = (
            self.stopped
            and self.mode == "rank"
            and self.rank._dp_rank_and_size()[1] > 1
            and any(
                custom.kind == "buffer"
                or (
                    custom.kind == "module"
                    and next(cast(torch.nn.Module, custom.value).buffers(), None)
                    is not None
                )
                for slot in getattr(self.rank, "_checkpoint_slots", {}).values()
                for custom in slot.custom.values()
            )
        )
        # Piggyback presence so empty replicas still join a needed reconciliation.
        pending = (tuple(self.state.released), synchronize_heads)
        gathered: list[Any] = [pending]
        loop = asyncio.get_running_loop()
        if self.distributed:
            gathered = [None] * dist.get_world_size()
            # A finished DP callback must leave its actor loop available while
            # another DP callback awaits unrelated async work. Only Gloo runs
            # off-thread; dropping graph ownership stays on the actor thread.
            completed = loop.run_in_executor(
                None,
                partial(
                    dist.all_gather_object,
                    gathered,
                    pending,
                    group=self.state.release_group,
                ),
            )
        else:
            completed = loop.create_future()
            completed.set_result(None)
        release = self.state.pending_release = _Release(
            completed, gathered, self._finish_release
        )
        # The callback's exception can reach its controller before every DP
        # sibling exits. Keep ownership until their matching cleanup completes.
        completed.add_done_callback(lambda _: self._finish_release(release))

    def _finish_release(self, release: _Release) -> None:
        if self.state.pending_release is not release or not release.completed.done():
            return
        self.state.pending_release = None
        try:
            release.completed.result()
            handles = {handle for values, _ in release.gathered for handle in values}
            for handle in handles:
                self.state.graphs.pop(handle, None)
            self.state.released.difference_update(handles)
            if any(synchronize for _, synchronize in release.gathered):
                from ._heads import synchronize_head_buffers

                # Every DP session has stopped. Unequal callbacks/yields cannot
                # enter this WORLD collective early, including after failure.
                synchronize_head_buffers(self.rank)
        except BaseException as error:
            self.state.release_error = (
                f"Callback release reconciliation failed: {error}"
            )
            release.completed.get_loop().call_exception_handler(
                {"message": self.state.release_error, "exception": error}
            )

    async def _join_release(self) -> asyncio.CancelledError | None:
        return await join_rank_callback_release(self.rank)

    async def reconcile_releases(
        self, *, defer_cancellation: bool = False
    ) -> asyncio.CancelledError | None:
        """Join prior cleanup before entering another all-rank boundary."""
        cancelled = await self._join_release()
        self._start_release()
        cancelled = await self._join_release() or cancelled
        if cancelled is not None and not defer_cancellation:
            raise cancelled
        return cancelled

    @asynccontextmanager
    async def release_on_exit(self) -> AsyncGenerator[None, None]:
        try:
            yield
        except GeneratorExit:
            await self.reconcile_releases()
            raise
        except BaseException:
            # FIRST_EXCEPTION controllers need this error to cancel siblings
            # which may still be awaiting user work. Their cleanup joins ours.
            self._start_release()
            raise
        else:
            await self.reconcile_releases()

    def _broadcast(self, command: _Command | None) -> _Command:
        if not self.distributed or len(self.members) == 1:
            assert command is not None
            return command
        payload = None
        if command is not None:
            try:
                payload = _encode_command(command)
            except Exception as exc:
                command = _Command(
                    command.sequence,
                    "error",
                    (f"Command serialization failed: {exc}",),
                    {},
                    False,
                )
                payload = _encode_command(command)
        objects: list[Any] = [payload]
        dist.broadcast_object_list(objects, src=self.leader, group=self.group)
        return self._decode(objects[0])

    def _decode(self, payload: Any) -> _Command:
        error, decoded = None, None
        try:
            decoded = _transport.decode(payload)
        except Exception as exc:
            error = f"Command deserialization failed: {exc}"
        failures = self._gather(error)
        if any(failures):
            return _Command(self.state.sequence, "error", (str(failures),), {}, False)
        assert isinstance(decoded, _Command)
        return decoded

    def invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        if self.stopped:
            raise RuntimeError("Trainer callback session has stopped")
        self.state.sequence += 1
        command = _Command(
            self.state.sequence, operation, args, kwargs, torch.is_grad_enabled()
        )
        command = self._broadcast(command)
        return self._execute(command)

    async def serve(self) -> None:
        deferred: BaseException | None = None
        try:
            while True:
                objects: list[Any] = [None]
                # Only the Gloo CPU receive leaves the actor thread. Decoding,
                # model commands and iterator cleanup retain its CUDA context.
                # Use a Future, not a child Task that asyncio.run shutdown could
                # cancel independently before the serving task joins it.
                received = asyncio.get_running_loop().run_in_executor(
                    None,
                    partial(
                        dist.broadcast_object_list,
                        objects,
                        src=self.leader,
                        group=self.group,
                    ),
                )
                while True:
                    try:
                        await asyncio.shield(received)
                        break
                    except asyncio.CancelledError as error:
                        # An abandoned receive could consume the next callback's
                        # command. Drain this session through its leader stop.
                        deferred = error if deferred is None else deferred
                command = self._decode(objects[0])
                self.state.sequence = max(self.state.sequence, command.sequence)
                if command.operation == "stop":
                    if deferred is not None:
                        raise deferred
                    return
                try:
                    self._execute(command)
                except BaseException as error:
                    # The leader receives the same coordinated error and chooses
                    # whether to catch it, continue, or stop the callback.
                    # Cancellation must not abandon the leader's stop command.
                    if not isinstance(error, Exception):
                        deferred = error if deferred is None else deferred
        finally:
            self.stopped = True
            self._close_iterators()

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            self.state.sequence += 1
            self._broadcast(_Command(self.state.sequence, "stop", (), {}, False))
        finally:
            self._close_iterators()

    def _close_iterators(self) -> None:
        iterators, self.iterators = self.iterators, {}
        for iterator in iterators.values():
            iterator.close()

    def _gather(self, value: Any) -> list[Any]:
        if not self.distributed or len(self.members) == 1:
            return [value]
        gathered: list[Any] = [None] * len(self.members)
        dist.all_gather_object(gathered, value, group=self.group)
        return gathered

    def _execute(self, command: _Command) -> Any:
        result, error = None, None
        try:
            with torch.set_grad_enabled(command.grad_enabled):
                result = self._dispatch(command)
        except BaseException as exc:
            error = exc
        errors = self._gather(
            None if error is None else f"{type(error).__name__}: {error}"
        )
        if any(errors):
            self.state.graphs.pop(
                f"{self.mode}:{command.sequence}:dp:{self.dp_rank}", None
            )
            if error is not None:
                raise error
            raise RuntimeError(
                f"Physical trainer command {command.operation!r} failed: {errors}"
            )
        if command.operation in ("forward", "next", "batches_next"):
            value = (
                result if get_rank_callback_metadata(self.rank) is not None else None
            )
            try:
                return self._gather_outputs(value)
            except BaseException:
                self.state.graphs.pop(
                    f"{self.mode}:{command.sequence}:dp:{self.dp_rank}", None
                )
                raise
        return result

    def _gather_outputs(self, value: Any) -> list[Any] | None:
        if not self.distributed or len(self.members) == 1:
            return [value]

        def admit_serialization() -> None:
            from ._tensors import flatten_tensors

            tensors, _ = flatten_tensors(value)
            # Pickling creates storage bytes before gather admission can sample
            # their size. Reserve tensor storage, a copy, and per-leaf metadata.
            required = 2 * sum(t.numel() * t.element_size() for t in tensors)
            required += 4096 * (len(tensors) + 1)
            if required > self._available_host_memory():
                raise MemoryError(
                    f"Trainer output serialization requires {required} CPU bytes"
                )

        self._coordinated_preflight(admit_serialization)
        payload = self._coordinated_preflight(lambda: cloudpickle.dumps(value))
        sizes = self._gather(len(payload))

        def admit() -> None:
            available = self._available_host_memory()
            # Gloo gather_object pads every sender to the largest serialized
            # payload. Include receive storage and unpickling copies on leader.
            padded = max(sizes) + 1024
            required = 2 * padded
            if self.is_leader:
                required += 2 * len(sizes) * padded + sum(sizes)
            if required > available:
                raise MemoryError(
                    f"Trainer output transfer requires {required} CPU bytes, "
                    f"but the per-process shared-host budget has {available}"
                )

        self._coordinated_preflight(admit)
        values: list[Any] | None = (
            [None] * len(self.members) if self.is_leader else None
        )
        dist.gather_object(payload, values, dst=self.leader, group=self.group)

        def decode() -> list[Any] | None:
            if values is None:
                return None
            decoded = [cloudpickle.loads(item) for item in values]
            return [item for item in decoded if item is not None]

        return self._coordinated_preflight(decode)

    def _available_host_memory(self) -> int:
        from ._memory_policy import host_memory_budget, local_rank_count

        if hasattr(self.rank, "_available_cpu_memory_bytes"):
            return self.rank._available_cpu_memory_bytes()
        return host_memory_budget(
            local_world_size=local_rank_count(
                world_size=dist.get_world_size() if self.distributed else 1
            )
        ).available_bytes

    def _packet(self, tree: Any, sequence: int) -> Any:
        from ._tensors import ManagedTensor, detach_tree, flatten_tensors

        handle = f"{self.mode}:{sequence}:dp:{self.dp_rank}"
        tensors, _ = flatten_tensors(tree)
        if any(tensor.requires_grad for tensor in tensors):
            self.state.graphs[handle] = tuple(tensors)
        if get_rank_callback_metadata(self.rank) is None:
            return None
        required = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        if required > self._available_host_memory():
            raise MemoryError(f"Trainer output snapshot requires {required} CPU bytes")
        return _OutputPacket(
            detach_tree(handle, tree, device="cpu"),
            tuple(tensor.device.type == "cpu" for tensor in tensors),
            any(isinstance(tensor, ManagedTensor) for tensor in tensors),
        )

    def _dispatch(self, command: _Command) -> Any:
        op, args, kwargs = command.operation, command.args, command.kwargs
        if op == "error":
            raise RuntimeError(args[0])
        if op == "forward":
            return self._packet(self.rank.forward(*args, **kwargs), command.sequence)
        if op == "batches":
            self.iterators[command.sequence] = cast(
                Generator[Any, None, None], self.rank.forward_batches(*args, **kwargs)
            )
            return command.sequence
        if op == "batches_open":
            handle = f"{self.mode}:batches:{command.sequence}"
            self.state.iterators[handle] = self.rank.forward_batches(*args, **kwargs)
            return handle
        if op in ("next", "batches_next"):
            iterator = (
                self.iterators[args[0]]
                if op == "next"
                else self.state.iterators[args[0]]
            )
            batch = next(iterator, None)
            if batch is None:
                return (None, None)
            return replace(batch, inputs=[], outputs=[]), self._packet(
                batch.outputs, command.sequence
            )
        if op in ("close", "batches_close"):
            iterator = (
                self.iterators.pop(args[0], None)
                if op == "close"
                else self.state.iterators.pop(args[0], None)
            )
            if op == "batches_close":
                self.state.batch_inputs.pop(args[0], None)
            if iterator is not None:
                cast(Generator[Any, None, None], iterator).close()
            return None
        if op == "release":
            for handle in args[0]:
                self.state.graphs.pop(handle, None)
            return None
        if op == "backward":
            return self._backward(*args, **kwargs)
        if op == "head":
            from ._heads import execute_head_operation, synchronize_head_buffers

            if self.mode == "zero" and args[0] == "head_export":
                synchronize_head_buffers(self.rank)
            return execute_head_operation(
                self.rank,
                *args,
                coordinate=self._coordinated_preflight,
                local_lookup=self.mode == "rank",
                **kwargs,
            )
        if op == "reduce_value":
            tensor = args[0].to(self.rank.device)
            self.rank.reduce(tensor, **kwargs)
            return tensor
        if op == "pop_checkpoint_context":

            def validate() -> None:
                ref = self.rank._slot_ref(args[0])
                if not self.rank._slot_stack or self.rank._slot_stack[-1] != ref:
                    raise RuntimeError(
                        "Pushed checkpoint stack changed before context exit"
                    )

            self._coordinated_preflight(validate)
            return self.rank.pop_checkpoint()
        return getattr(self.rank, op)(*args, **kwargs)

    def _coordinated_preflight(self, validate: Callable[[], T]) -> T:
        return _coordinate_call(validate, group=self.group)

    def _backward(self, packets: Sequence[Any], *, retain_graph: bool) -> None:
        outputs, gradients, handles, head_gradients = [], [], [], []

        def prepare() -> None:
            for packet in packets:
                if packet.handle.startswith("head:"):
                    from ._heads import head_gradient_targets

                    targets = head_gradient_targets(
                        self.rank, packet, materialize=False
                    )
                    # The global caller's head loss occurs once; DP SUM must
                    # count it once while TP/CP retain their replicated contract.
                    if self.mode != "zero" or self.dp_rank == 0:
                        head_gradients.extend(targets)
                    continue
                tensors = self.state.graphs.get(packet.handle)
                if tensors is None:
                    if packet.handle.endswith(f":dp:{self.dp_rank}"):
                        raise ValueError(
                            f"Unknown or released forward {packet.handle!r}"
                        )
                    continue  # Other DP rank owns this root.
                if len(tensors) != len(packet.gradients):
                    raise ValueError(
                        "Cotangent packet does not match its physical forward"
                    )
                handles.append(packet.handle)
                for tensor, gradient in zip(tensors, packet.gradients, strict=True):
                    if gradient is not None:
                        if gradient.shape != tensor.shape or not tensor.requires_grad:
                            raise ValueError(
                                "Cotangent does not match its physical forward"
                            )
                        outputs.append(tensor)
                        gradients.append(
                            gradient.to(device=tensor.device, dtype=tensor.dtype)
                        )

        self._coordinated_preflight(prepare)
        transaction = (
            self.rank._gradient_transaction(before_commit=self._coordinated_preflight)
            if hasattr(self.rank, "_gradient_transaction")
            else nullcontext()
        )
        try:
            with transaction:

                def stage_heads() -> None:
                    gradient = None
                    try:
                        for version, maximum, parameter, source in head_gradients:
                            gradient = source.to(
                                device=parameter.device, dtype=parameter.dtype
                            )
                            self.rank._commit_versioned_gradients(
                                ((version, maximum, parameter, gradient),)
                            )
                            gradient = None
                    finally:
                        gradient = None

                # Every peer finishes (or rolls back) copies/staging before any
                # participant can enter model backward's TP/CP collectives.
                if any(packet.handle.startswith("head:") for packet in packets):
                    self._coordinated_preflight(stage_heads)
                if hasattr(self.rank, "_forward_cotangent_collector"):
                    inner: list[tuple[str, Any]] = []

                    def collect() -> None:
                        if outputs:
                            packets = self.rank._forward_cotangent_collector().backward(
                                outputs, gradients, retain_graph=retain_graph
                            )
                            inner.extend(
                                (packet.handle, packet.gradients) for packet in packets
                            )
                        self.rank._forward_graph_cache().validate_many(inner)

                    self._coordinated_preflight(collect)
                    self.rank._forward_graph_cache().backward_many(
                        inner,
                        retain_graph=retain_graph,
                        coordinate=lambda call: _coordinate_call(
                            call, group=self.rank._forward_memory_group()
                        ),
                    )
                elif outputs:
                    torch.autograd.backward(
                        outputs, gradients, retain_graph=retain_graph
                    )
        finally:
            if not retain_graph:
                for handle in handles:
                    self.state.graphs.pop(handle, None)


_COLLECTIVE_METHODS = frozenset(
    {
        "zero_grad",
        "load_checkpoint",
        "snapshot_checkpoint",
        "_push_checkpoint_sync",
        "prefetch_checkpoints",
        "pop_checkpoint",
        "save_checkpoint",
        "prepare_checkpoint_save",
        "finish_checkpoint_save",
        "abort_checkpoint_save",
        "export_lora",
        "optim_step",
    }
)


class _PushedCheckpoint(_impl.PushedCheckpoint):
    def _pop(self) -> None:
        if not self._entered:
            return
        cast("_RankView", self._trainer)._invoke("pop_checkpoint_context", self._path)
        self._entered = False
        self._closed = True


class _RankView:
    def __init__(self, executor: _Executor) -> None:
        self._executor = executor
        self._rank = executor.rank
        self._transport_handles: list[str] | None = None

    @property
    def device(self) -> torch.device:
        return self._rank.device

    @property
    def hidden_size(self) -> int:
        return self._rank.hidden_size

    def __getattribute__(self, name: str) -> Any:
        if name in _COLLECTIVE_METHODS:
            return lambda *args, **kwargs: self._invoke(name, *args, **kwargs)
        return object.__getattribute__(self, name)

    def _flush_heads(self) -> None:
        if hasattr(self._rank, "_logical_head_handles"):
            from ._heads import flush_logical_heads

            flush_logical_heads(self)

    def _refresh_heads(self) -> None:
        if hasattr(self._rank, "_logical_head_handles"):
            from ._heads import refresh_logical_heads

            refresh_logical_heads(self)

    def _invoke(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        if self._executor.stopped:
            raise RuntimeError("Trainer callback session has stopped")
        released = self._executor.state.released
        handles = tuple(
            handle
            for handle in released
            if handle.startswith(f"{self._executor.mode}:")
        )
        if handles:
            self._executor.invoke("release", handles)
            released.difference_update(handles)
        if operation != "head":
            self._flush_heads()
        result = self._executor.invoke(operation, *args, **kwargs)
        if operation in {
            "optim_step",
            "load_checkpoint",
            "pop_checkpoint",
            "pop_checkpoint_context",
            "_push_checkpoint_sync",
        }:
            self._refresh_heads()
        return result

    def push_checkpoint(self, checkpoint: Any) -> _impl.PushedCheckpoint:
        path, directory = self._rank._checkpoint_source(checkpoint)
        return _PushedCheckpoint(cast("TrainerRank", self), path, directory)

    def forward(self, inputs: _impl.ForwardInputs, **kwargs: Any) -> Any:
        materialized = _impl._materialize(inputs)
        if self._executor.mode == "rank":
            if hasattr(self._rank, "_capture_forward_options"):
                materialized = self._rank._capture_forward_options(
                    materialized, kwargs.get("options")
                )
            packets = self._invoke("forward", materialized, **kwargs)
            with self._release_on_error([packet.packet.handle for packet in packets]):
                (packet,) = packets
                return self._attach(self._place_outputs([(packet, materialized)])[0])
        single = isinstance(materialized, _impl.ForwardInput)
        roots = [materialized] if single else materialized
        outputs: list[Any] = []
        handles: list[str] = []
        with self._release_on_error(handles):
            items = self._prepare_batches(roots, kwargs)
            for batch in self._iterate_batches(items, kwargs, handles):
                outputs.extend(batch.outputs)
            return (
                outputs[0]
                if single
                else _impl._rebuild_forward_tree(materialized, outputs)
            )

    @contextmanager
    def _release_on_error(self, handles: Sequence[str]) -> Iterator[None]:
        try:
            yield
        except BaseException:
            if handles:
                try:
                    # Followers must release their graphs too; head publication
                    # is unrelated to reclaiming outputs never delivered.
                    self._executor.invoke("release", tuple(handles))
                except BaseException:
                    self._executor.state.released.update(handles)
            raise

    def forward_batches(self, inputs: Any, **kwargs: Any) -> Iterator[_impl.MicroBatch]:
        items = self._prepare_batches(inputs, kwargs)
        return self._iterate_batches(items, kwargs)

    def _prepare_batches(self, inputs: Any, kwargs: dict[str, Any]) -> Any:
        items = [_impl._materialize(item) for item in inputs]
        if kwargs.get("no_grad") is None:
            kwargs["no_grad"] = not torch.is_grad_enabled()
        if hasattr(self._rank, "_capture_forward_options"):
            items = self._rank._capture_forward_options(items, kwargs.get("options"))
        if self._executor.mode == "zero":
            kwargs["yield_empty"] = True
        return items

    def open_forward_batches(self, inputs: Any, **kwargs: Any) -> str:
        """Bind a lazily pulled iterator that survives callback boundaries."""
        items = self._prepare_batches(inputs, kwargs)
        if kwargs.get("checkpoint", _impl.Unset) is _impl.Unset:
            stack = getattr(self._rank, "_slot_stack", ())
            selected = (
                stack[-1] if stack else getattr(self._rank, "_default_slot_ref", None)
            )
            kwargs["checkpoint"] = None if selected is None else selected.name
        handle = self._invoke("batches_open", items, **kwargs)
        self._executor.state.batch_inputs[handle] = items
        return handle

    def next_forward_batch(self, handle: str) -> _impl.MicroBatch | None:
        items = self._executor.state.batch_inputs.get(handle)
        if items is None:
            return None
        try:
            batch = self._combine_wave(self._invoke("batches_next", handle), items)
        except BaseException:
            self._close_failed_iterator("batches_close", handle)
            raise
        if batch is None:
            self.close_forward_batches(handle)
        return batch

    def close_forward_batches(self, handle: str) -> None:
        self._invoke("batches_close", handle)

    def _close_failed_iterator(self, operation: str, handle: int | str) -> None:
        try:
            # Pending releases and head publication must not prevent closure.
            self._executor.invoke(operation, handle)
        except BaseException:
            pass  # Preserve the delivery error and any queued graph release.

    def _iterate_batches(
        self,
        items: Any,
        kwargs: dict[str, Any],
        handles: list[str] | None = None,
    ) -> Iterator[_impl.MicroBatch]:
        identifier = self._invoke("batches", items, **kwargs)
        delivery_failed = False
        try:
            while True:
                try:
                    batch = self._combine_wave(
                        self._invoke("next", identifier), items, handles
                    )
                except BaseException:
                    delivery_failed = True
                    raise
                if batch is None:
                    return
                yield batch
                del batch
        finally:
            # This iterator belongs to its creating callback, even if a retained
            # traceback delays its finalizer until a later callback is serving.
            if not self._executor.stopped:
                if delivery_failed:
                    self._close_failed_iterator("close", identifier)
                else:
                    self._invoke("close", identifier)

    def _combine_wave(
        self, wave: Any, items: Any, accumulated: list[str] | None = None
    ) -> _impl.MicroBatch | None:
        handles = [packet.packet.handle for _, packet in wave if packet is not None]
        with self._release_on_error(handles):
            batch = self._assemble_wave(wave, items)
            if accumulated is not None:
                accumulated.extend(handles)
            return batch

    def _assemble_wave(self, wave: Any, items: Any) -> _impl.MicroBatch | None:
        if all(batch is None for batch, _ in wave):
            return None
        if any(batch is None for batch, _ in wave):
            raise RuntimeError("Physical forward iterators ended on different waves")
        packets = self._place_outputs(
            [
                (packet, [items[index] for index in batch.indices])
                for batch, packet in wave
            ]
        )
        batches = [
            replace(
                batch,
                inputs=[items[index] for index in batch.indices],
                outputs=self._attach(packet),
            )
            for (batch, _), packet in zip(wave, packets, strict=True)
        ]
        if self._executor.mode == "rank":
            return batches[0]
        rows = sorted(
            (index, output)
            for batch in batches
            for index, output in zip(batch.indices, batch.outputs, strict=True)
        )
        batch = batches[0]
        return replace(
            batch,
            inputs=[items[index] for index, _ in rows],
            outputs=[output for _, output in rows],
            indices=[index for index, _ in rows],
            stats=replace(batch.stats, local_count=len(rows)),
        )

    def _place_outputs(
        self, outputs: Sequence[tuple[_OutputPacket, Any]]
    ) -> list[_OutputPacket]:
        if self._transport_handles is not None:
            self._transport_handles.extend(
                output.packet.handle for output, _ in outputs
            )
            return [
                replace(output, cpu=(True,) * len(output.cpu), managed=True)
                for output, _ in outputs
            ]
        from ._memory_policy import choose_output_placements
        from ._options import resolve_forward_options
        from ._tensors import flatten_tensors, unflatten_tensors

        costs: list[tuple[int, Literal["auto", "model", "cpu"]]] = []
        for output, inputs in outputs:
            policies: dict[int, Literal["auto", "model", "cpu"]] = {}

            def visit(request: Any, result: Any) -> None:
                if isinstance(request, _impl.ForwardInput):
                    policy = resolve_forward_options(
                        input=request.options
                    ).output_device
                    tensors, _ = flatten_tensors(result)
                    for tensor in tensors:
                        policies[id(tensor)] = policy
                else:
                    for child, value in zip(request, result, strict=True):
                        visit(child, value)

            visit(inputs, unflatten_tensors(output.packet.spec, output.packet.tensors))
            costs.extend(
                (
                    tensor.numel() * tensor.element_size(),
                    "cpu" if cpu else policies[id(tensor)],
                )
                for tensor, cpu in zip(output.packet.tensors, output.cpu, strict=True)
            )
        available = (
            self._rank._available_memory_bytes()
            if hasattr(self._rank, "_available_memory_bytes")
            else 1 << 60
        )
        if hasattr(self._rank, "_pending_backward_memory"):
            available -= sum(self._rank._pending_backward_memory())
        try:
            placements = iter(
                choose_output_placements(costs, gpu_available_bytes=available)
            )
        except BaseException:
            self._invoke(
                "release", tuple(output.packet.handle for output, _ in outputs)
            )
            raise
        result = []
        for output, _ in outputs:
            cpu = tuple(next(placements) == "cpu" for _ in output.cpu)
            result.append(
                replace(output, cpu=cpu, managed=output.managed or cpu != output.cpu)
            )
        return result

    def _attach(self, output: _OutputPacket) -> Any:
        packet = replace(
            output.packet,
            tensors=tuple(
                tensor if cpu else tensor.to(self.device)
                for tensor, cpu in zip(output.packet.tensors, output.cpu, strict=True)
            ),
        )
        state = self._executor.state
        state_ref, handle = weakref.ref(state), packet.handle

        def release() -> None:
            if owner := state_ref():
                owner.released.add(handle)

        with torch.enable_grad():
            return state.collector.attach(
                packet, managed=output.managed, on_release=release
            )

    def backward(
        self, loss: Any, gradient: Any = None, *, retain_graph: bool = False
    ) -> None:
        packets = self._executor.state.collector.backward(
            loss, gradient, retain_graph=retain_graph
        )
        self._submit_backward(packets, retain_graph=retain_graph)

    def _submit_backward(self, packets: Sequence[Any], *, retain_graph: bool) -> None:
        self._invoke(
            "backward",
            tuple(
                replace(
                    packet,
                    gradients=tuple(
                        None if value is None else value.cpu()
                        for value in packet.gradients
                    ),
                )
                for packet in packets
            ),
            retain_graph=retain_graph,
        )

    def export_forward(self, tree: Any) -> Any:
        from ._tensors import detach_tree, flatten_tensors

        state = self._executor.state
        state.sequence += 1
        handle = f"client:{state.sequence}:dp:{self._executor.dp_rank}"
        tensors, _ = flatten_tensors(tree)
        if self._transport_handles is not None:
            # Gathered CPU storage remains owned by exports until backward or
            # release, and fresh host/cgroup headroom excludes that live use.
            # Reserve the additional independent reply snapshot before copying.
            required = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            if required > self._executor._available_host_memory():
                raise MemoryError(
                    f"Trainer export snapshot requires {required} CPU bytes"
                )
        packet = detach_tree(handle, tree, device="cpu")
        if any(tensor.requires_grad for tensor in tensors):
            state.exports[handle] = tuple(
                reply
                if self._transport_handles is not None and not tensor.requires_grad
                else tensor
                for tensor, reply in zip(tensors, packet.tensors, strict=True)
            )
        return packet

    def release_forward(self, handles: Sequence[str]) -> None:
        """Idempotently release exported client graphs after caller collection."""
        for handle in handles:
            self._executor.state.exports.pop(handle, None)
        self._invoke("release", ())

    def backward_packets(
        self, packets: Sequence[Any], *, retain_graph: bool = False
    ) -> None:
        handles = tuple(
            packet.handle for packet in packets if not packet.handle.startswith("head:")
        )
        try:
            outputs, gradients, heads = [], [], []
            for packet in packets:
                if packet.handle.startswith("head:"):
                    heads.append(packet)
                    continue
                tensors = self._executor.state.exports[packet.handle]
                if len(tensors) != len(packet.gradients):
                    raise ValueError(
                        "Client cotangent packet does not match its forward"
                    )
                for tensor, gradient in zip(tensors, packet.gradients, strict=True):
                    if gradient is not None:
                        outputs.append(tensor)
                        gradients.append(
                            gradient.to(device=tensor.device, dtype=tensor.dtype)
                        )
            model = (
                self._executor.state.collector.backward(
                    outputs, gradients, retain_graph=retain_graph
                )
                if outputs
                else ()
            )
            self._submit_backward((*model, *heads), retain_graph=retain_graph)
        finally:
            if not retain_graph:
                for handle in handles:
                    self._executor.state.exports.pop(handle, None)

    def module(
        self, name: str, factory: Callable[[], Any], **kwargs: Any
    ) -> ModuleHandle:
        return cast("ModuleHandle", self._register("module", name, factory, **kwargs))

    def parameter(
        self, name: str, factory: Callable[[], Any], **kwargs: Any
    ) -> torch.nn.Parameter:
        return cast(
            torch.nn.Parameter, self._register("parameter", name, factory, **kwargs)
        )

    def buffer(
        self, name: str, factory: Callable[[], Any], **kwargs: Any
    ) -> torch.Tensor:
        return cast(torch.Tensor, self._register("buffer", name, factory, **kwargs))

    def _register(
        self,
        kind: Literal["module", "parameter", "buffer"],
        name: str,
        factory: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        from ._heads import logical_register_head

        return logical_register_head(self, kind, name, factory, **kwargs)

    def last_forward_telemetry(self) -> dict[str, Any]:
        return self._rank.last_forward_telemetry()


class TrainerRankZero(_RankView):
    """Single callback view over every physical trainer rank, without reduce."""


def _view(executor: _Executor) -> _RankView:
    from . import TrainerRank

    class LogicalTrainerRank(_RankView, TrainerRank):
        def reduce(self, tensor: torch.Tensor, **kwargs: Any) -> None:
            result = self._invoke("reduce_value", tensor.detach().cpu(), **kwargs)
            tensor.copy_(result.to(tensor.device))

    return (
        TrainerRankZero(executor)
        if executor.mode == "zero"
        else LogicalTrainerRank(executor)
    )


async def run_rank_callback(
    rank: TrainerRank, callback: Callable[[Any], Any], *, mode: Mode = "rank"
) -> RankCallbackResult:
    executor = _Executor(rank, mode)
    cancelled = await executor.reconcile_releases(defer_cancellation=True)
    async with executor.release_on_exit():
        if not executor.is_leader:
            await executor.serve()
            if cancelled is not None:
                raise cancelled
            return RankCallbackResult(None)
        view = _view(executor)
        try:
            if cancelled is not None:
                raise cancelled
            view._refresh_heads()
            result = callback(view)
            if inspect.isawaitable(result):
                result = await result
            if inspect.isgenerator(result) or inspect.isasyncgen(result):
                raise TypeError("Use run_rank_callback_stream for generator callbacks")
            return RankCallbackResult(0 if mode == "zero" else executor.dp_rank, result)
        finally:
            try:
                view._flush_heads()
            finally:
                executor.stop()


async def run_rank_callback_stream(
    rank: TrainerRank, callback: Callable[[Any], Any], *, mode: Mode = "rank"
) -> AsyncGenerator[RankCallbackResult, Any]:
    """Drive one user generator on each logical leader, forwarding sends."""
    executor = _Executor(rank, mode)
    cancelled = await executor.reconcile_releases(defer_cancellation=True)
    if not executor.is_leader:
        async with executor.release_on_exit():
            await executor.serve()
            if cancelled is not None:
                raise cancelled
        yield RankCallbackResult(None)
        return
    iterator = value = None
    view = _view(executor)
    async with executor.release_on_exit():
        try:
            if cancelled is not None:
                raise cancelled
            view._refresh_heads()
            iterator = callback(view)
            if inspect.isawaitable(iterator):
                iterator = await iterator
            if not (inspect.isgenerator(iterator) or inspect.isasyncgen(iterator)):
                raise TypeError("Stream callback must return a generator")
            sent = None
            while True:
                try:
                    value = (
                        await iterator.asend(sent)
                        if inspect.isasyncgen(iterator)
                        else iterator.send(sent)
                    )
                except (StopIteration, StopAsyncIteration):
                    return
                sent = yield RankCallbackResult(
                    0 if mode == "zero" else executor.dp_rank, value
                )
        finally:
            try:
                if inspect.isasyncgen(iterator):
                    await iterator.aclose()
                elif inspect.isgenerator(iterator):
                    iterator.close()
                view._flush_heads()
            finally:
                value = None
                executor.stop()
