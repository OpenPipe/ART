"""Evictable physical forward graphs, separate from caller autograd graphs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
import random
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4
import weakref

import torch
from torch._C._autograd import _get_current_graph_task_keep_graph
from torch.multiprocessing.reductions import StorageWeakRef

from art._tensor_residency import observe_resident_tensors

from ._tensors import _map_tensor_arguments

ForwardHandle = str
type Retention = Literal["gpu", "cpu", "replay"]


@dataclass(frozen=True)
class _InputTensor:
    value: torch.Tensor
    device: torch.device
    requires_grad: bool


def _snapshot(value: Any, *, inputs: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        if inputs:
            return _InputTensor(
                value.detach().to("cpu", copy=True), value.device, value.requires_grad
            )
        return value.detach().clone().requires_grad_(value.requires_grad)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                f.name: _snapshot(getattr(value, f.name), inputs=inputs)
                for f in fields(value)
                if f.init
            },
        )
    if isinstance(value, tuple):
        items = tuple(_snapshot(item, inputs=inputs) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, list):
        return [_snapshot(item, inputs=inputs) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot(item, inputs=inputs) for key, item in value.items()}
    return deepcopy(value)


def _restore_inputs(value: Any) -> Any:
    if isinstance(value, _InputTensor):
        return value.value.to(value.device, copy=True).requires_grad_(
            value.requires_grad
        )
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                f.name: _restore_inputs(getattr(value, f.name))
                for f in fields(value)
                if f.init
            },
        )
    return _map_tensor_arguments(_restore_inputs, value)


def _tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _storage_sizes(tensors: Iterable[torch.Tensor]) -> tuple[int, int]:
    sizes: dict[tuple[torch.device, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        sizes[(tensor.device, storage.data_ptr())] = storage.nbytes()
    return (
        sum(size for (device, _), size in sizes.items() if device.type != "cpu"),
        sum(size for (device, _), size in sizes.items() if device.type == "cpu"),
    )


@dataclass
class _RNGState:
    cpu: torch.Tensor
    cuda: dict[int, torch.Tensor]
    python: tuple[Any, ...]
    tracker: Any = None

    @classmethod
    def capture(cls, devices: Sequence[int], tracker: Any) -> _RNGState:
        return cls(
            torch.get_rng_state(),
            {device: torch.cuda.get_rng_state(device) for device in devices},
            random.getstate(),
            None if tracker is None else _snapshot(tracker.get_states()),
        )

    def restore(self, tracker: Any) -> None:
        torch.set_rng_state(self.cpu)
        for device, state in self.cuda.items():
            torch.cuda.set_rng_state(state, device)
        random.setstate(self.python)
        if tracker is not None:
            tracker.set_states(_snapshot(self.tracker))

    @contextmanager
    def replay(self, tracker: Any):
        ambient = self.capture(tuple(self.cuda), tracker)
        try:
            self.restore(tracker)
            yield
        finally:
            ambient.restore(tracker)


@dataclass
class _TransferStats:
    """Completed saved-storage copies, cumulative across released graphs."""

    offload_bytes: int = 0
    offload_seconds: float = 0.0
    offload_count: int = 0
    offload_max_bytes: int = 0
    restore_bytes: int = 0
    restore_seconds: float = 0.0
    restore_count: int = 0
    restore_max_bytes: int = 0

    @torch.compiler.disable
    def copy(self, tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        start = perf_counter()
        if tensor.device.type == "cuda" and torch.device(device).type == "cpu":
            # Pin the only host copy of each storage. Keep copies blocking so
            # offload releases GPU ownership before returning, even on user
            # streams, and unpack never exposes an unfinished restore.
            result = torch.empty_like(tensor, device="cpu", pin_memory=True)
            result.copy_(tensor)
        else:
            result = tensor.to(device, copy=True)
        elapsed = perf_counter() - start
        size = tensor.numel() * tensor.element_size()
        if result.device.type == "cpu":
            self.offload_bytes += size
            self.offload_seconds += elapsed
            self.offload_count += 1
            self.offload_max_bytes = max(self.offload_max_bytes, size)
        else:
            self.restore_bytes += size
            self.restore_seconds += elapsed
            self.restore_count += 1
            self.restore_max_bytes = max(self.restore_max_bytes, size)
        return result


@dataclass
class _SavedTensor:
    tensor: torch.Tensor
    device: torch.device
    managed: bool
    source: StorageWeakRef
    restored: dict[tuple[torch.device, StorageWeakRef], torch.Tensor]
    transfer_stats: _TransferStats

    def view(self, storage, device) -> torch.Tensor:
        tensor = self.tensor
        result = torch.empty(0, dtype=tensor.dtype, device=device).set_(
            storage, tensor.storage_offset(), tensor.size(), tensor.stride()
        )
        if tensor.is_conj():
            result = result.conj()
        if tensor.is_neg():
            result = torch._neg_view(result)
        return result

    def offload(self, copies: dict[StorageWeakRef, torch.Tensor]) -> None:
        if self.managed and self.tensor.device.type != "cpu":
            tensor = self.tensor
            storage = tensor.untyped_storage()
            # Weak storage identity prevents allocator address reuse from
            # confusing distinct activations without pinning CUDA storage.
            key = self.source
            if key not in copies:
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                copies[key] = self.transfer_stats.copy(raw, "cpu")
            self.tensor = self.view(copies[key].untyped_storage(), "cpu")

    def unpack(self) -> torch.Tensor:
        if self.tensor.device == self.device:
            # TE frees unpacked tensor data by rebinding Tensor.data. A retained
            # graph needs its saved object's metadata intact for the next call.
            if _get_current_graph_task_keep_graph():
                return self.tensor.detach()
            return self.tensor
        key = (self.device, self.source)
        if key not in self.restored:
            storage = self.tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.uint8).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
            self.restored[key] = self.transfer_stats.copy(raw, self.device)
        return self.view(self.restored[key].untyped_storage(), self.device)


@dataclass(frozen=True)
class GraphState:
    retention: Retention
    gpu_bytes: int
    cpu_bytes: int
    offload_bytes: int
    replay_bytes: int
    replayable: bool
    replay_count: int
    offloadable: bool
    restore_workspace_bytes: int = 0
    checkpoint_versions: tuple[Any, ...] = ()
    non_offloadable_bytes: int | None = None


@dataclass
class _ForwardRecord:
    execute: Callable[[Any], Sequence[torch.Tensor]]
    inputs: Any
    context_factory: Callable[[], AbstractContextManager[Any]]
    validate_backward: Callable[[], None] | None
    retention: Retention
    checkpoint_versions: tuple[Any, ...]
    options: Any
    rng: _RNGState
    rng_tracker: Any
    keep_on_device: Callable[[torch.Tensor], bool] | None
    transfer_stats: _TransferStats
    outputs: tuple[torch.Tensor, ...] | None = None
    saved: list[weakref.ReferenceType[_SavedTensor]] | None = None
    resident: list[weakref.ReferenceType[torch.Tensor]] | None = None
    metadata: tuple[tuple[torch.Size, torch.dtype, torch.device, bool], ...] = ()
    replay_count: int = 0
    autocast: tuple[tuple[str, bool, torch.dtype], ...] = ()
    corrections: Any = None
    is_stale: Callable[[], bool] | None = None
    current_context_factory: Callable[[], AbstractContextManager[Any]] | None = None
    replay_with_current: bool = False
    restored: dict[tuple[torch.device, StorageWeakRef], torch.Tensor] = field(
        default_factory=dict
    )
    execution_peak_bytes: int = 0

    def run(
        self,
        *,
        context_factory: Callable[[], AbstractContextManager[Any]] | None = None,
        store: bool = True,
        grad_enabled: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        saved: list[weakref.ReferenceType[_SavedTensor]] = []
        resident: list[weakref.ReferenceType[torch.Tensor]] = []
        observed = False
        copies: dict[StorageWeakRef, torch.Tensor] = {}
        retention, keep_on_device, restored, transfer_stats = (
            self.retention,
            self.keep_on_device,
            self.restored,
            self.transfer_stats,
        )

        def observe(tensors: Sequence[torch.Tensor]) -> None:
            nonlocal observed
            observed = True
            resident.extend(weakref.ref(tensor) for tensor in tensors)

        def pack(tensor: torch.Tensor) -> _SavedTensor:
            cell = _SavedTensor(
                tensor.detach(),
                tensor.device,
                not (keep_on_device and keep_on_device(tensor)),
                StorageWeakRef(tensor.untyped_storage()),
                restored,
                transfer_stats,
            )
            if retention == "cpu":
                cell.offload(copies)
            saved.append(weakref.ref(cell))
            return cell

        with (
            torch.enable_grad(),
            (context_factory or self.context_factory)(),
            ExitStack() as stack,
        ):
            for device, enabled, dtype in self.autocast:
                stack.enter_context(
                    torch.autocast(device, enabled=enabled, dtype=dtype)
                )
            with (
                torch.set_grad_enabled(grad_enabled),
                torch.autograd.graph.saved_tensors_hooks(pack, _SavedTensor.unpack),
                observe_resident_tensors(observe),
            ):
                try:
                    outputs = tuple(self.execute(_restore_inputs(self.inputs)))
                except BaseException:
                    # A failed forward has no usable graph. Release partial
                    # host allocations even if its traceback retains outputs.
                    for reference in saved:
                        if (cell := reference()) is not None:
                            cell.tensor = torch.empty(0)
                    raise
                finally:
                    copies.clear()
        if store:
            # Outputs and external autograd contexts already own these storages.
            # Moving their saved aliases would retain both CUDA and CPU copies,
            # then allocate a duplicate CUDA restore beside the original storage.
            output_storages = {
                StorageWeakRef(value.untyped_storage()): value.untyped_storage()
                for value in (*outputs, *(ref() for ref in resident))
                if value is not None
            }
            for reference in saved:
                cell = reference()
                if cell is not None and cell.managed and cell.source in output_storages:
                    cell.tensor = cell.view(output_storages[cell.source], cell.device)
                    cell.managed = False
            self.outputs, self.saved = outputs, saved
            self.resident = resident if observed else None
        return outputs


class GraphCache:
    """A rank-local cache. Its caller serializes model operations and policies."""

    def __init__(self) -> None:
        self._records: dict[ForwardHandle, _ForwardRecord] = {}
        self.transfer_stats = _TransferStats()

    def run(
        self,
        execute: Callable[[Any], Sequence[torch.Tensor]],
        inputs: Any,
        *,
        context_factory: Callable[[], AbstractContextManager[Any]] = nullcontext,
        validate_backward: Callable[[], None] | None = None,
        retention: Retention = "gpu",
        checkpoint_versions: tuple[Any, ...] = (),
        options: Any = None,
        cuda_devices: Sequence[int] = (),
        rng_tracker: Any = None,
        keep_on_device: Callable[[torch.Tensor], bool] | None = None,
        output_device: torch.device | str | None = None,
        execution_peak_bytes: int = 0,
    ) -> tuple[ForwardHandle, tuple[torch.Tensor, ...]]:
        if retention not in ("gpu", "cpu", "replay"):
            raise ValueError(f"Unknown graph retention {retention!r}")
        if retention == "cpu" and not getattr(options, "allow_cpu_offload", True):
            raise ValueError("CPU graph offload is disabled for this forward")
        if retention == "replay" and not getattr(options, "allow_replay", True):
            raise ValueError("Graph replay is disabled for this forward")
        record = _ForwardRecord(
            execute,
            _snapshot(inputs, inputs=True),
            context_factory,
            validate_backward,
            retention,
            checkpoint_versions,
            options,
            _RNGState.capture(cuda_devices, rng_tracker),
            rng_tracker,
            keep_on_device,
            self.transfer_stats,
        )
        record.autocast = tuple(
            (
                device,
                torch.is_autocast_enabled(device),
                torch.get_autocast_dtype(device),
            )
            for device in ("cpu", "cuda")
        )
        record.execution_peak_bytes = execution_peak_bytes
        physical = record.run()
        record.metadata = tuple(
            (value.shape, value.dtype, value.device, value.requires_grad)
            for value in physical
        )
        # clone: a detached view may still pin a much larger model activation.
        detached = tuple(
            value.detach()
            .to(device=output_device, copy=True)
            .requires_grad_(value.requires_grad)
            for value in physical
        )
        handle = uuid4().hex
        self._records[handle] = record
        if retention == "replay":
            self.evict(handle)
        return handle, detached

    def handles(self) -> tuple[ForwardHandle, ...]:
        return tuple(self._records)

    def set_corrections(
        self,
        handle: ForwardHandle,
        context: Any,
        *,
        is_stale: Callable[[], bool],
        current_context_factory: Callable[[], AbstractContextManager[Any]]
        | None = None,
    ) -> None:
        record = self._records[handle]
        record.corrections = context
        record.is_stale = is_stale
        record.current_context_factory = current_context_factory

    def state(self, handle: ForwardHandle) -> GraphState:
        record = self._records[handle]
        cells = [
            cell
            for ref in record.saved or ()
            if (cell := ref()) is not None and cell.managed
        ]
        correction_tensors = (
            () if record.corrections is None else record.corrections.tensors
        )
        resident = tuple(
            value for ref in record.resident or () if (value := ref()) is not None
        )
        gpu, cpu = _storage_sizes(
            (
                *_tensors(record.inputs),
                *_tensors(record.rng),
                *correction_tensors,
                *(cell.tensor for cell in cells),
                *(record.outputs or ()),
                *resident,
            )
        )
        offload, _ = _storage_sizes(cell.tensor for cell in cells)
        _, replay = _storage_sizes(
            (*_tensors(record.inputs), *_tensors(record.rng), *correction_tensors)
        )
        policy = getattr(record.options, "backward_state", "auto")
        return GraphState(
            record.retention,
            gpu,
            cpu,
            offload,
            replay,
            getattr(record.options, "allow_replay", True)
            and policy in ("auto", "replay"),
            record.replay_count,
            getattr(record.options, "allow_cpu_offload", True)
            and policy in ("auto", "cpu"),
            max(0, record.execution_peak_bytes - gpu),
            record.checkpoint_versions,
            None
            if record.resident is None
            else _storage_sizes((*(record.outputs or ()), *resident))[0],
        )

    def offload(self, handle: ForwardHandle) -> None:
        record = self._records[handle]
        if not getattr(record.options, "allow_cpu_offload", True):
            raise RuntimeError("CPU graph offload is disabled for this forward")
        if record.outputs is None:
            return
        copies: dict[StorageWeakRef, torch.Tensor] = {}
        try:
            for ref in record.saved or ():
                if (cell := ref()) is not None:
                    cell.offload(copies)
        finally:
            copies.clear()
        record.retention = "cpu"

    def evict(
        self, handle: ForwardHandle, *, replay_with_current: bool | None = None
    ) -> None:
        record = self._records[handle]
        if not getattr(record.options, "allow_replay", True):
            raise RuntimeError("Graph replay is disabled for this forward")
        if replay_with_current is not None:
            if replay_with_current and record.current_context_factory is None:
                raise ValueError("Current-weight replay requires a version context")
            record.replay_with_current = replay_with_current
        record.outputs = record.saved = record.resident = None
        record.restored.clear()
        record.retention = "replay"

    def release(self, handle: ForwardHandle) -> None:
        if (record := self._records.pop(handle, None)) is not None:
            # Saved-variable hooks can outlive their Python outputs. Break all
            # ownership edges even when a caller retains a failure traceback.
            record.outputs = record.saved = record.resident = None
            record.restored.clear()
            record.inputs = record.corrections = None
            record.execute = lambda _: ()
            record.context_factory = nullcontext
            record.validate_backward = record.current_context_factory = None
            record.is_stale = record.keep_on_device = None

    def backward(
        self,
        handle: ForwardHandle,
        gradients: Sequence[torch.Tensor | None],
        *,
        retain_graph: bool = False,
    ) -> None:
        self.backward_many(((handle, gradients),), retain_graph=retain_graph)

    def validate_many(
        self,
        packets: Sequence[tuple[ForwardHandle, Sequence[torch.Tensor | None]]],
    ) -> None:
        handles: set[str] = set()
        for handle, gradients in packets:
            if handle in handles:
                raise ValueError("Duplicate forward handle in one backward")
            handles.add(handle)
            record = self._records[handle]
            if record.validate_backward is not None:
                record.validate_backward()
            if len(gradients) != len(record.metadata):
                raise ValueError("Cotangent count does not match forward outputs")
            for gradient, (shape, dtype, _, requires_grad) in zip(
                gradients, record.metadata, strict=True
            ):
                if gradient is not None and (
                    not requires_grad
                    or gradient.shape != shape
                    or gradient.dtype != dtype
                ):
                    raise ValueError(
                        "Cotangent shape/dtype or output requires_grad mismatch"
                    )
            if (
                record.corrections is not None
                and record.is_stale is not None
                and record.is_stale()
            ):
                if (
                    record.corrections.requires_current(gradients)
                    and record.current_context_factory is None
                ):
                    raise RuntimeError(
                        "Stale correction requires current model outputs but no current version context is available"
                    )

    def backward_many(
        self,
        packets: Sequence[tuple[ForwardHandle, Sequence[torch.Tensor | None]]],
        *,
        retain_graph: bool = False,
        coordinate: Callable[[Callable[[], Any]], Any] | None = None,
    ) -> None:
        self.validate_many(packets)
        try:
            self._backward_validated(
                packets,
                retain_graph=retain_graph,
                coordinate=coordinate or (lambda function: function()),
            )
        except BaseException:
            # A replay/backward failure consumes the operation. The enclosing
            # checkpoint transaction discards unpublished optimizer gradients.
            for handle, _ in packets:
                self.release(handle)
            raise

    def _backward_validated(
        self,
        packets: Sequence[tuple[ForwardHandle, Sequence[torch.Tensor | None]]],
        *,
        retain_graph: bool,
        coordinate: Callable[[Callable[[], Any]], Any],
    ) -> None:
        # Random wire handles differ across physical ranks. Creation order is
        # shared by TP/CP peers and therefore fixes backward collective order.
        ordinals = {handle: index for index, handle in enumerate(self._records)}
        packets = sorted(packets, key=lambda packet: ordinals[packet[0]])
        prepared = {}
        stale = {
            handle: record.is_stale is not None and record.is_stale()
            for handle, _ in packets
            for record in (self._records[handle],)
        }
        # The coordinator uses this logical rank's TP/CP group, never all DP
        # ranks: different DP owners may have different numbers of graphs.
        for handle, gradients in packets:
            record = self._records[handle]
            prepared[handle] = coordinate(
                lambda: self._prepare_correction(record, gradients, stale[handle])
            )
        for handle, gradients in packets:
            record = self._records[handle]
            pairs = coordinate(
                lambda: self._prepare_backward(
                    record, gradients, stale[handle], prepared[handle]
                )
            )
            try:
                coordinate(
                    lambda: (
                        torch.autograd.backward(
                            [output for output, _ in pairs],
                            [gradient for _, gradient in pairs],
                            retain_graph=retain_graph,
                        )
                        if pairs
                        else None
                    )
                )
            finally:
                record.restored.clear()
            del pairs
            if not retain_graph:
                self.release(handle)
            elif record.retention == "replay":
                self.evict(handle)

    @staticmethod
    def _prepare_correction(record, gradients, stale):
        if not (
            stale
            and record.corrections is not None
            and record.corrections.requires_current(gradients)
            and not (record.outputs is None and record.replay_with_current)
        ):
            return None
        # Explicit always may add a no-grad forward. Stage every correction
        # before physical backward; changed cotangents wait on CPU.
        with record.rng.replay(record.rng_tracker):
            current = record.run(
                context_factory=record.current_context_factory,
                store=False,
                grad_enabled=False,
            )
        corrected = record.corrections.correct(gradients, current)
        return tuple(
            value.to("cpu") if value is not None and value is not original else value
            for value, original in zip(corrected, gradients, strict=True)
        )

    @staticmethod
    def _prepare_backward(record, gradients, stale, prepared):
        gradients = gradients if prepared is None else prepared
        if not any(gradient is not None for gradient in gradients):
            return []
        current_replay = stale and record.replay_with_current
        if record.outputs is None:
            with record.rng.replay(record.rng_tracker):
                physical = record.run(
                    context_factory=record.current_context_factory
                    if current_replay
                    else None
                )
            metadata = tuple(
                (value.shape, value.dtype, value.device, value.requires_grad)
                for value in physical
            )
            if metadata != record.metadata:
                record.outputs = record.saved = None
                raise RuntimeError(
                    "Replayed output metadata differs from original forward"
                )
            record.replay_count += 1
        if prepared is None and stale and record.corrections is not None:
            if current_replay:
                record.corrections.validate_replay(gradients, record.outputs)
            gradients = record.corrections.correct(
                gradients, record.outputs if current_replay else None
            )
        assert record.outputs is not None
        return [
            (output, gradient.to(output.device))
            for output, gradient in zip(record.outputs, gradients, strict=True)
            if gradient is not None
        ]
