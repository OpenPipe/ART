"""Live checkpoint-owned modules and their serializable client state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, cast
import weakref

import torch

from ._tensors import _map_tensor_arguments, _map_tensors

if TYPE_CHECKING:
    from ._impl import TrainerRank, _CustomObject, _CustomTensorTracker

HeadKind = Literal["module", "parameter", "buffer"]
_parameter_transform: ContextVar[bool] = ContextVar(
    "head_parameter_transform", default=False
)
_head_call: ContextVar[tuple[tuple[object, ...], dict[int, torch.Tensor]] | None] = (
    ContextVar("head_call", default=None)
)


def head_call_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Route preexisting live buffer aliases into their active module captures."""
    call = _head_call.get()
    if call is None:
        return None
    changed = False

    def replace(value: torch.Tensor) -> torch.Tensor:
        nonlocal changed
        if id(value) in call[1]:
            changed = True
            return call[1][id(value)]
        return value

    result = _map_tensors(replace, (args, kwargs))
    return result if changed else None


TENSOR_METADATA_FUNCTIONS = frozenset(
    {
        "size",
        "numel",
        "nelement",
        "dim",
        "ndimension",
        "stride",
        "storage_offset",
        "element_size",
        "is_contiguous",
        "is_floating_point",
        "is_complex",
        "is_signed",
        "get_device",
    }
)
_TENSOR_INSPECTION_PROPERTIES = frozenset(
    {
        "_backward_hooks",
        "_base",
        "_cdata",
        "_grad",
        "_grad_fn",
        "_has_symbolic_sizes_strides",
        "_post_accumulate_grad_hooks",
        "_python_dispatch",
        "_version",
        "data",
        "device",
        "dtype",
        "grad",
        "grad_dtype",
        "grad_fn",
        "is_cpu",
        "is_cuda",
        "is_ipu",
        "is_leaf",
        "is_maia",
        "is_meta",
        "is_mkldnn",
        "is_mps",
        "is_mtia",
        "is_nested",
        "is_quantized",
        "is_sparse",
        "is_sparse_csr",
        "is_vulkan",
        "is_xla",
        "is_xpu",
        "itemsize",
        "layout",
        "name",
        "names",
        "nbytes",
        "ndim",
        "output_nr",
        "requires_grad",
        "retains_grad",
        "shape",
        "volatile",
    }
)
_TENSOR_MUTATION_DUNDERS = frozenset(
    {
        "__setitem__",
        "__set__",
        "__iadd__",
        "__isub__",
        "__imul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__imatmul__",
        "__iand__",
        "__ior__",
        "__ixor__",
        "__ilshift__",
        "__irshift__",
    }
)
_CLIENT_BUFFER_MUTATIONS = (_TENSOR_MUTATION_DUNDERS - {"__set__", "__imatmul__"}) | {
    "copy_",
    "fill_",
    "zero_",
    "add_",
    "sub_",
    "mul_",
    "div_",
    "true_divide_",
    "floor_divide_",
    "remainder_",
    "fmod_",
    "pow_",
    "lerp_",
    "bitwise_and_",
    "bitwise_or_",
    "bitwise_xor_",
    "bitwise_left_shift_",
    "bitwise_right_shift_",
    "masked_fill_",
    "masked_scatter_",
    "scatter_",
    "scatter_add_",
    "index_copy_",
    "index_add_",
    "index_fill_",
    "index_put_",
    "put_",
    "clamp_",
    "clamp_min_",
    "clamp_max_",
}


def tensor_metadata_function(func: Callable[..., Any]) -> bool:
    """Inspect handle metadata/state without capturing a weight-sized snapshot.

    Tensor-valued views (T, mT, H, mH, real, imag) and unknown descriptors must
    use the ordinary computation path. Autograd/storage inspection retains its
    existing handle semantics, including the client's separate .data guard.
    """
    name = getattr(func, "__name__", "")
    return name in TENSOR_METADATA_FUNCTIONS or (
        name == "__get__"
        and getattr(getattr(func, "__self__", None), "__name__", None)
        in _TENSOR_INSPECTION_PROPERTIES
    )


def mutates_tensor(func: Callable[..., Any], kwargs: Mapping[str, Any]) -> bool:
    name = getattr(func, "__name__", "")
    return (
        (name.endswith("_") and not name.endswith("__"))
        or name in _TENSOR_MUTATION_DUNDERS
        or kwargs.get("out") is not None
        or kwargs.get("inplace") is True
    )


def tensor_mutation_targets(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> set[int]:
    from ._impl import _walk_objects

    if not mutates_tensor(func, kwargs):
        return set()
    values = kwargs["out"] if kwargs.get("out") is not None else args[:1]
    return {
        id(value) for value in _walk_objects(values) if isinstance(value, torch.Tensor)
    }


def readonly_buffer_views(result: Any, snapshots: list[torch.Tensor]) -> Any:
    def wrap(value: torch.Tensor) -> torch.Tensor:
        with torch._C.DisableTorchFunctionSubclass():
            if any(
                getattr(torch._C, "_is_alias_of")(value, snapshot)
                for snapshot in snapshots
            ):
                value.__class__ = _BufferSnapshotView
        return value

    return _map_tensors(wrap, result) if snapshots else result


class _BufferSnapshotView(torch.Tensor):
    """A readable snapshot view that cannot silently impersonate a live write."""

    def __deepcopy__(self, memo: dict[int, Any]) -> torch.Tensor:
        result = _plain(self)
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        return _plain(self).__reduce_ex__(proto)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        from ._impl import _walk_objects

        kwargs = kwargs or {}
        if getattr(func, "__name__", "") in {
            "batch_norm",
            "instance_norm",
            "embedding",
            "embedding_bag",
        }:
            raise RuntimeError(
                "Stateful operations on buffer snapshot views are unsupported; use the live buffer or an explicit clone()"
            )
        targets = tensor_mutation_targets(func, args, kwargs)
        if any(
            isinstance(value, cls) and id(value) in targets
            for value in _walk_objects((args, kwargs))
        ):
            raise RuntimeError(
                "Views of live checkpoint buffers are read-only; use buffer[index] = value or buffer.copy_(), or clone() for a writable private copy"
            )
        snapshots = []

        def plain(value: torch.Tensor) -> torch.Tensor:
            if isinstance(value, cls):
                with torch._C.DisableTorchFunctionSubclass():
                    value = value.as_subclass(torch.Tensor)
                snapshots.append(value)
            return value

        args, kwargs = _map_tensors(plain, (args, kwargs))
        return readonly_buffer_views(func(*args, **kwargs), snapshots)


def _plain(tensor: torch.Tensor) -> torch.Tensor:
    with torch._C.DisableTorchFunctionSubclass():
        return tensor.as_subclass(torch.Tensor).detach().clone()


def _restore_module(module: torch.nn.Module) -> torch.nn.Module:
    return module


def _preserve_module_aliases(module: torch.nn.Module, apply: Callable[[], Any]) -> None:
    groups: dict[tuple[str, int], list[str]] = {}
    for kind, values in (
        ("_parameters", module.named_parameters(remove_duplicate=False)),
        ("_buffers", module.named_buffers(remove_duplicate=False)),
    ):
        for key, value in values:
            groups.setdefault((kind, id(value)), []).append(key)
    apply()
    for (kind, _), keys in groups.items():
        prefix, _, key = keys[0].rpartition(".")
        value = getattr(module.get_submodule(prefix), kind)[key]
        for alias in keys[1:]:
            prefix, _, key = alias.rpartition(".")
            getattr(module.get_submodule(prefix), kind)[key] = value


def move_module(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    _preserve_module_aliases(module, lambda: module.to(device=device))
    return module


def head_staleness(trainer: TrainerRank) -> int:
    from ._options import resolve_forward_options

    return resolve_forward_options(
        getattr(trainer, "_forward_options", None)
    ).max_gradient_staleness


class ModuleHandle(torch.nn.Module):
    """A reusable module whose calls capture current checkpoint tensor versions.

    Each successful call publishes its buffer changes. A failed call leaves buffers
    unchanged. Parameters and buffers retained by earlier calls are never modified.
    The proxy delegates attributes, children, and named parameters to the source;
    its type is ModuleHandle, so isinstance(handle, factory_class) is not preserved.
    For an external activation-checkpoint closure, capture ``head.snapshot()``
    before calling ``torch.utils.checkpoint``. Snapshot buffers are private;
    their mutations are not published to the checkpoint. Client and public logical
    callback snapshots contain cotangent bridges: use ``use_reentrant=False``.
    Internal physical snapshots support either mode under a gradient transaction.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        capture: Callable[[], tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
        publish: Callable[[Mapping[str, torch.Tensor]], None],
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_source", module)
        object.__setattr__(self, "_capture", capture)
        object.__setattr__(self, "_publish", publish)
        self._parameters = module._parameters
        self._buffers = module._buffers
        self._modules = module._modules
        self._non_persistent_buffers_set = module._non_persistent_buffers_set
        self.training = module.training

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._source, name)

    def __getitem__(self, key: Any) -> Any:
        return self._source[key]

    def train(self, mode: bool = True) -> ModuleHandle:
        self._source.train(mode)
        self.training = mode
        return self

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> ModuleHandle:
        def convert(value: torch.Tensor) -> torch.Tensor:
            result = fn(value)
            if isinstance(value, _ClientBuffer):
                return _ClientBuffer(result, value._head_owner)
            if not isinstance(value, torch.nn.Parameter) and hasattr(
                value, "_art_tracker"
            ):
                return type(value)(result, value._art_tracker)
            return result

        token = _parameter_transform.set(True)
        try:
            _preserve_module_aliases(
                self._source, lambda: self._source._apply(convert, recurse)
            )
        finally:
            _parameter_transform.reset(token)
        self._parameters, self._buffers, self._modules = (
            self._source._parameters,
            self._source._buffers,
            self._source._modules,
        )
        return self

    def requires_grad_(self, requires_grad: bool = True) -> ModuleHandle:
        raise RuntimeError(
            "Set checkpoint parameter trainability in the module factory"
        )

    def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
        if (call := _head_call.get()) is not None and any(
            head is self for head in call[0]
        ):
            return super()._call_impl(*args, **kwargs)
        return self._captured_call(
            lambda: super(ModuleHandle, self)._call_impl(*args, **kwargs)
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if (call := _head_call.get()) is not None and any(
            head is self for head in call[0]
        ):
            return self._source(*args, **kwargs)
        return self._captured_call(lambda: self._source(*args, **kwargs))

    def _captured_call(self, call: Callable[[], Any]) -> Any:
        if torch._C._current_graph_task_id() >= 0:
            raise RuntimeError(
                "Live module called during backward recomputation; pass head.snapshot() to activation checkpointing (client and logical callback handles require use_reentrant=False)"
            )
        parameters, buffers = self._capture()
        parameter_versions = {
            name: value._version for name, value in parameters.items()
        }
        # Hooks on the handle must see the same private tensors as forward.
        # Internal checkpoint closures retain the captured source after bindings
        # on the public handle are restored. Memo substitution avoids extra copies.
        captured = self._captured_module(parameters, buffers)
        original = self._source
        active, aliases = _head_call.get() or ((), {})
        token = _head_call.set(
            (
                (*active, self),
                aliases
                | {id(value): buffers[key] for key, value in original.named_buffers()},
            )
        )
        try:
            object.__setattr__(self, "_source", captured)
            self._parameters, self._buffers, self._modules = (
                captured._parameters,
                captured._buffers,
                captured._modules,
            )
            result = call()
            current_parameters = dict(captured.named_parameters())
            if current_parameters.keys() != parameters.keys() or any(
                current_parameters[name] is not value
                or value._version != parameter_versions[name]
                for name, value in parameters.items()
            ):
                raise RuntimeError(
                    "Checkpoint module forward must not mutate parameters"
                )
            updated_buffers = dict(captured.named_buffers())
        finally:
            object.__setattr__(self, "_source", original)
            self._parameters, self._buffers, self._modules = (
                original._parameters,
                original._buffers,
                original._modules,
            )
            _head_call.reset(token)
        self._publish(updated_buffers)
        return result

    def _captured_module(
        self,
        parameters: Mapping[str, torch.Tensor],
        buffers: Mapping[str, torch.Tensor],
    ) -> torch.nn.Module:
        memo = {
            id(value): parameters[key] for key, value in self._source.named_parameters()
        } | {id(value): buffers[key] for key, value in self._source.named_buffers()}
        return deepcopy(self._source, memo)

    def snapshot(self) -> torch.nn.Module:
        """Capture an external closure; cotangent bridges require use_reentrant=False."""
        if torch._C._current_graph_task_id() >= 0:
            raise RuntimeError(
                "Capture head.snapshot() before entering activation checkpointing"
            )
        return self._captured_module(*self._capture())

    def __deepcopy__(self, memo: dict[int, object]) -> torch.nn.Module:
        return deepcopy(self._source, memo)

    def __reduce_ex__(self, protocol: SupportsIndex) -> tuple[Any, ...]:
        return _restore_module, (self._source,)


class _NativeModuleState:
    def __init__(
        self,
        trainer: TrainerRank,
        tracker: _CustomTensorTracker,
        module: torch.nn.Module,
    ):
        self.trainer = weakref.ref(trainer)
        self.tracker = tracker
        self.module = module

    def capture(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        trainer = self.tracker.validate()
        assert self.tracker.ref.name is not None
        version = trainer._capture_checkpoint_version(self.tracker.ref.name)
        from ._impl import _track_slot_graph_tensor

        parameters = {}
        marker = torch.zeros((), dtype=torch.bool, device="cpu")
        for key, parameter in self.module.named_parameters():
            with torch._C.DisableTorchFunctionSubclass():
                value = trainer._snapshot_parameter(
                    parameter, version, head_staleness(trainer)
                )
                if value.requires_grad and torch.is_grad_enabled():
                    value = _track_slot_graph_tensor(value, marker)
                parameters[key] = value
        if any(value.requires_grad for value in parameters.values()):
            self.tracker.record(marker)
        buffers = {key: _plain(value) for key, value in self.module.named_buffers()}
        return parameters, buffers

    def publish(self, buffers: Mapping[str, torch.Tensor]) -> None:
        self.tracker.validate()
        staged = _stage_local_buffers(dict(self.module.named_buffers()), buffers)
        with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
            for target, value in staged:
                target.copy_(value)
        if staged:
            self.tracker.buffer_revision += 1


def _stage_local_buffers(
    targets: Mapping[str, torch.Tensor], values: Mapping[str, torch.Tensor]
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if targets.keys() != values.keys():
        raise ValueError("Checkpoint module forward must preserve buffer names")
    staged = []
    with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
        for key, target in targets.items():
            value = values[key]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(
                    "Checkpoint module forward must preserve buffer shape and dtype"
                )
            value = value.to(device=target.device)
            if not torch.equal(target, value):
                prepared = _plain(target)
                prepared.copy_(value)
                staged.append((target, prepared))
    return staged


def native_module_handle(
    trainer: TrainerRank, custom: _CustomObject, tracker: _CustomTensorTracker
) -> ModuleHandle:
    state = _NativeModuleState(trainer, tracker, cast(torch.nn.Module, custom.value))
    return ModuleHandle(state.module, state.capture, state.publish)


@dataclass(frozen=True)
class HeadRegistration:
    checkpoint: Any
    name: str
    kind: HeadKind
    value: torch.nn.Module | torch.Tensor | None
    factory_error: str | None = None


@dataclass(frozen=True)
class HeadState:
    version: Any
    name: str
    kind: HeadKind
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    buffer_revision: int
    max_gradient_staleness: int = 2


@dataclass(frozen=True)
class HeadObservation:
    sequence: int
    state: HeadState | None


def _observe_head(trainer: TrainerRank, state: HeadState | None) -> HeadObservation:
    sequence = getattr(trainer, "_head_observation_sequence", 0) + 1
    setattr(trainer, "_head_observation_sequence", sequence)
    return HeadObservation(sequence, state)


@dataclass(frozen=True)
class HeadBufferUpdate:
    version: Any
    name: str
    buffer_revision: int
    buffers: dict[str, torch.Tensor]


def _custom_tracker(custom: _CustomObject) -> _CustomTensorTracker:
    if custom.kind == "module":
        assert isinstance(custom.handle, ModuleHandle)
        return custom.handle._capture.__self__.tracker
    return cast(Any, custom.value)._art_tracker


def _head_tensors(
    kind: HeadKind, source: torch.nn.Module | torch.Tensor, *, parameters: bool
) -> dict[str, torch.Tensor]:
    if kind == "module":
        module = cast(torch.nn.Module, source)
        return dict(module.named_parameters() if parameters else module.named_buffers())
    target_kind = "parameter" if parameters else "buffer"
    return {"": cast(torch.Tensor, source)} if kind == target_kind else {}


def export_head(trainer: TrainerRank, checkpoint: str, name: str) -> HeadState:
    custom = trainer._checkpoint_slots[checkpoint].custom[name]
    parameters, buffers = (
        _head_tensors(custom.kind, custom.value, parameters=True),
        _head_tensors(custom.kind, custom.value, parameters=False),
    )
    return HeadState(
        trainer._capture_checkpoint_version(checkpoint),
        name,
        custom.kind,
        {
            key: _plain(value).cpu().requires_grad_(value.requires_grad)
            for key, value in parameters.items()
        },
        {key: _plain(value).cpu() for key, value in buffers.items()},
        _custom_tracker(custom).buffer_revision,
        head_staleness(trainer),
    )


def execute_head_operation(
    trainer: Any,
    kind: str,
    payload: Any,
    *,
    coordinate: Callable[[Callable[[], None]], None] | None = None,
    local_lookup: bool = False,
) -> Any:
    """Execute on every physical rank inside the owning trainer operation queue."""
    if hasattr(trainer, "_rank"):
        return trainer._invoke("head", kind, payload)
    if kind == "head_lookup":
        from ._impl import Unset

        checkpoint, name = payload
        if local_lookup and checkpoint is Unset:
            ref = (
                trainer._slot_stack[-1]
                if trainer._slot_stack
                else trainer._default_slot_ref
            )
            checkpoint = None if ref is None else ref.name
        # Reopening a loaded head is DP-local; lazy loading is still global.
        if not local_lookup or checkpoint is None:
            checkpoint = trainer._resolve_custom_checkpoint(checkpoint)
        elif checkpoint not in trainer._checkpoint_slots:
            raise trainer._slot_state_error(
                f"Load checkpoint {checkpoint!r} across all ranks before a logical-DP head lookup"
            )
        assert checkpoint is not None
        return checkpoint, export_head(
            trainer, checkpoint, name
        ) if name in trainer._checkpoint_slots[checkpoint].custom else None
    if kind == "head_register":
        from . import _checkpoint

        registration: HeadRegistration = payload
        # Every DP/TP/CP participant must settle leader-side construction before
        # any rank enters checkpoint resolution or mutates its registered heads.
        _checkpoint.raise_distributed(
            None
            if registration.factory_error is None
            else RuntimeError(registration.factory_error),
            f"construct custom object {registration.name!r}",
            trainer._checkpoint_group(),
        )
        checkpoint = trainer._resolve_custom_checkpoint(registration.checkpoint)
        existing = trainer._checkpoint_slots[checkpoint].custom.get(registration.name)
        if existing is not None:
            _validate_registration(trainer, existing, registration)
        trainer._custom_object(
            registration.name,
            registration.kind,
            lambda: deepcopy(registration.value),
            checkpoint=checkpoint,
        )
        return _observe_head(
            trainer, export_head(trainer, checkpoint, registration.name)
        )
    if kind == "head_export":
        return tuple(
            _observe_head(
                trainer,
                export_head(trainer, checkpoint, name)
                if checkpoint in trainer._checkpoint_slots
                and name in trainer._checkpoint_slots[checkpoint].custom
                else None,
            )
            for checkpoint, name in payload
        )
    if kind == "head_publish":
        from . import _checkpoint

        staged, error = [], None

        def prepare() -> None:
            nonlocal staged
            staged = _stage_buffer_publications(trainer, payload)

        if coordinate is not None:
            coordinate(prepare)
        else:
            try:
                prepare()
            except Exception as exc:
                error = exc
            _checkpoint.raise_distributed(
                error, "publish custom buffers", trainer._checkpoint_group()
            )
        with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
            for tracker, values in staged:
                for target, value in values:
                    target.copy_(value)
                tracker.buffer_revision += 1
        return None
    raise ValueError(f"Unknown head operation: {kind!r}")


def _validate_registration(
    trainer: TrainerRank, existing: _CustomObject, registration: HeadRegistration
) -> None:
    from . import _checkpoint
    from ._impl import _custom_signature, _custom_state, _CustomObject

    error = None
    try:
        value = registration.value
        if existing.kind != registration.kind:
            raise ValueError("Custom checkpoint object kind differs from registration")
        if existing.kind == "module":
            if not isinstance(value, torch.nn.Module):
                raise TypeError("module() factory must return torch.nn.Module")
            if trainer._checkpoint_slots[registration.checkpoint].snapshot:
                value = deepcopy(value).requires_grad_(False)
            candidate = _CustomObject("module", value, object())
            if _custom_signature(
                registration.name, candidate, _custom_state(candidate)
            ) != _custom_signature(
                registration.name, existing, _custom_state(existing)
            ):
                raise ValueError(
                    "Custom module schema differs from its registered checkpoint state"
                )
        elif (
            not isinstance(value, torch.Tensor)
            or value.shape != cast(torch.Tensor, existing.value).shape
            or value.dtype != cast(torch.Tensor, existing.value).dtype
        ):
            raise ValueError(
                "Custom tensor shape or dtype differs from its registered checkpoint state"
            )
    except Exception as exc:
        error = exc
    _checkpoint.raise_distributed(
        error, "validate custom handle", trainer._checkpoint_group()
    )


def _stage_buffer_publications(trainer: TrainerRank, updates: Any) -> list[Any]:
    staged = []
    for update in updates:
        current = trainer._capture_checkpoint_version(update.version.checkpoint)
        if current.generation != update.version.generation:
            raise trainer._slot_state_error(
                "Custom head checkpoint was replaced before buffer publication"
            )
        custom = trainer._checkpoint_slots[update.version.checkpoint].custom[
            update.name
        ]
        tracker = _custom_tracker(custom)
        if tracker.buffer_revision != update.buffer_revision:
            raise RuntimeError(
                f"Custom module {update.name!r} buffers changed before publication"
            )
        targets = _head_tensors(custom.kind, custom.value, parameters=False)
        if update.buffers.keys() != targets.keys():
            raise ValueError("Custom module buffer keys changed before publication")
        values = []
        for key, value in update.buffers.items():
            target = targets[key]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Custom buffer {key!r} shape or dtype changed")
            values.append((target, value.to(target.device).clone()))
        staged.append((tracker, values))
    return staged


def head_gradient_targets(
    trainer: TrainerRank, packet: Any, *, materialize: bool = True
) -> list[tuple[Any, int, torch.nn.Parameter, torch.Tensor]]:
    """Translate one client head cotangent; caller commits all operation targets together."""
    import json

    from ._versions import CheckpointVersion

    if not packet.handle.startswith("head:"):
        raise ValueError("Not a custom head cotangent")
    metadata = json.loads(packet.handle[5:])
    version = CheckpointVersion(
        metadata["checkpoint"], metadata["generation"], metadata["revision"]
    )
    trainer._validate_checkpoint_version(
        version, max_gradient_staleness=metadata["max_gradient_staleness"]
    )
    custom = trainer._checkpoint_slots[version.checkpoint].custom[metadata["name"]]
    parameters = _head_tensors(custom.kind, custom.value, parameters=True)
    if len(metadata["keys"]) != len(packet.gradients):
        raise ValueError("Custom head cotangent count does not match parameters")
    for key, gradient in zip(metadata["keys"], packet.gradients, strict=True):
        parameter = parameters[key]
        if gradient is not None and (
            gradient.shape != parameter.shape
            or any(
                tensor.layout != torch.strided
                for tensor in (parameter, gradient, parameter.grad)
                if tensor is not None
            )
        ):
            raise ValueError(
                "Custom head cotangent shape/layout does not match parameter"
            )
    return [
        (
            version,
            metadata["max_gradient_staleness"],
            cast(torch.nn.Parameter, parameters[key]),
            gradient.to(device=parameters[key].device, dtype=parameters[key].dtype)
            if materialize
            else gradient,
        )
        for key, gradient in zip(metadata["keys"], packet.gradients, strict=True)
        if gradient is not None
    ]


class _ClientParameter(torch.nn.Parameter):
    _head_owner: LiveHead
    _head_key: str

    def __new__(cls, data: torch.Tensor, owner: LiveHead, key: str) -> _ClientParameter:
        result = super().__new__(
            cls, data, requires_grad=owner.state.parameters[key].requires_grad
        )
        result._head_owner, result._head_key = owner, key
        return result

    def __init__(self, data: torch.Tensor, owner: LiveHead, key: str) -> None:
        pass

    def register_hook(self, hook: Any) -> Any:
        """Run once on summed captured uses per backward; removal affects old graphs."""
        from ._parameter_hooks import register_parameter_hook

        self._head_owner._validate()
        return register_parameter_hook(self, hook)

    def register_post_accumulate_grad_hook(self, hook: Any) -> Any:
        from ._parameter_hooks import reject_post_accumulate_hook

        return reject_post_accumulate_hook()

    @property
    def data(self) -> torch.Tensor:
        raise RuntimeError(
            "Client checkpoint parameters do not expose mutable .data; use detach() to read a snapshot"
        )

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        if not _parameter_transform.get():
            raise RuntimeError(
                "Client checkpoint parameters may only be changed by trainer.optim_step"
            )
        with torch._C.DisableTorchFunctionSubclass():
            cast(Any, torch.Tensor.data).__set__(self, value)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if tensor_metadata_function(func) or name in {
            "__format__",
            "__hash__",
            "__len__",
            "__repr__",
            "__str__",
        }:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        if torch._C._current_graph_task_id() >= 0:
            raise RuntimeError(
                "Live parameter used during backward recomputation; capture parameter.clone() before activation checkpointing and use use_reentrant=False for client or logical callback handles"
            )
        if name == "__set__" and _parameter_transform.get():
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        mutation_targets = tensor_mutation_targets(func, args, kwargs)
        captures: dict[int, torch.Tensor] = {}

        def replace(value: Any) -> Any:
            if isinstance(value, _ClientParameter):
                if id(value) in mutation_targets:
                    raise RuntimeError(
                        "Client checkpoint parameters may only be changed by trainer.optim_step"
                    )
                if id(value) not in captures:
                    captures[id(value)] = value._head_owner.capture((value._head_key,))[
                        value._head_key
                    ]
                return captures[id(value)]
            return _map_tensor_arguments(replace, value)

        return func(*replace(args), **replace(kwargs))

    def __deepcopy__(self, memo: dict[int, Any]) -> torch.nn.Parameter:
        with torch._C.DisableTorchFunctionSubclass():
            result = torch.nn.Parameter(_plain(self), requires_grad=self.requires_grad)
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        with torch._C.DisableTorchFunctionSubclass():
            return torch.nn.Parameter(
                _plain(self), requires_grad=self.requires_grad
            ).__reduce_ex__(proto)


class _ClientBuffer(torch.Tensor):
    _head_owner: LiveHead

    @staticmethod
    def __new__(cls, data: torch.Tensor, owner: LiveHead) -> _ClientBuffer:
        result = torch.Tensor._make_subclass(cls, data.detach(), require_grad=False)
        result._head_owner = owner
        return result

    @property
    def data(self) -> torch.Tensor:
        raise RuntimeError("Use checkpoint buffer.copy_() to change its values")

    @data.setter
    def data(self, value: torch.Tensor) -> None:
        raise RuntimeError("Use checkpoint buffer.copy_() to change its values")

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if (captured := head_call_arguments(args, kwargs)) is not None:
            return func(*captured[0], **captured[1])
        name = getattr(func, "__name__", "")
        from ._impl import _walk_objects

        mutation_targets = tensor_mutation_targets(func, args, kwargs)
        mutating = any(
            isinstance(value, _ClientBuffer) and id(value) in mutation_targets
            for value in _walk_objects((args, kwargs))
        )
        if mutating and (
            kwargs.get("out") is not None or name not in _CLIENT_BUFFER_MUTATIONS
        ):
            raise RuntimeError(
                f"Unsupported checkpoint buffer mutation {name}; use buffer.copy_() with unchanged shape and dtype"
            )
        copies, originals = {}, {}
        if tensor_metadata_function(func):
            for value in _walk_objects((args, kwargs)):
                if isinstance(value, _ClientBuffer):
                    value._head_owner._validate()
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

        def replace(value: Any) -> Any:
            if isinstance(value, _ClientBuffer):
                value._head_owner._validate()
                if id(value) not in copies:
                    with torch._C.DisableTorchFunctionSubclass():
                        copies[id(value)] = (
                            value.as_subclass(torch.Tensor)
                            if id(value) in mutation_targets
                            else _plain(value)
                        )
                    originals[id(copies[id(value)])] = value
                return copies[id(value)]
            return _map_tensor_arguments(replace, value)

        result = func(*replace(args), **replace(kwargs))
        copied = [
            (original, copies[id(original)])
            for original in originals.values()
            if id(original) not in mutation_targets
        ]
        staged = _stage_local_buffers(
            {str(index): original for index, (original, _) in enumerate(copied)},
            {str(index): copy for index, (_, copy) in enumerate(copied)},
        )
        with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
            for original, copy in staged:
                original.copy_(copy)
                cast(_ClientBuffer, original)._head_owner.pending = True
        if mutating:
            for original in originals.values():
                if id(original) in mutation_targets:
                    original._head_owner.pending = True
            result = originals.get(id(result), result)
        return readonly_buffer_views(result, [copy for _, copy in copied])

    def __deepcopy__(self, memo: dict[int, Any]) -> torch.Tensor:
        result = _plain(self)
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto: SupportsIndex) -> Any:
        return _plain(self).__reduce_ex__(proto)


class LiveHead:
    """Client/sandbox state shared by all references to one registered head."""

    def __init__(
        self,
        state: HeadState,
        value: torch.nn.Module | torch.Tensor,
        collector: Any,
        *,
        max_gradient_staleness: int | None = None,
    ):
        self.state = state
        self.collector = collector
        self.max_gradient_staleness = (
            state.max_gradient_staleness
            if max_gradient_staleness is None
            else max_gradient_staleness
        )
        self.pending = False
        self.invalid = False
        self.invalid_reason = "its checkpoint was replaced"
        if state.kind == "module":
            assert isinstance(value, torch.nn.Module)
            value = deepcopy(value)
            self.source = value
            replaced = {}
            for key, parameter in value.named_parameters():
                replaced[id(parameter)] = _ClientParameter(
                    state.parameters[key].to(parameter.device), self, key
                )
            for child in value.modules():
                for key, parameter in child._parameters.items():
                    if parameter is not None:
                        child._parameters[key] = replaced[id(parameter)]
            buffers = {
                id(buffer): _ClientBuffer(_plain(buffer), self)
                for buffer in value.buffers()
            }
            for child in value.modules():
                for key, buffer in child._buffers.items():
                    if buffer is not None:
                        child._buffers[key] = buffers[id(buffer)]
            self.value = ModuleHandle(value, self.capture_module, self.publish)
        elif state.kind == "parameter":
            assert isinstance(value, torch.Tensor)
            self.source = self.value = _ClientParameter(
                state.parameters[""].to(value.device), self, ""
            )
        else:
            assert isinstance(value, torch.Tensor)
            self.source = self.value = _ClientBuffer(
                state.buffers[""].to(value.device).clone(), self
            )
        self.refresh(state)

    def _validate(self) -> None:
        if self.invalid:
            raise RuntimeError(
                f"Custom checkpoint object {self.state.name!r} is stale because {self.invalid_reason}"
            )

    def invalidate(self, reason: str) -> None:
        self.invalid = True
        self.invalid_reason = reason

    def parameters(self) -> dict[str, torch.Tensor]:
        return _head_tensors(self.state.kind, self.source, parameters=True)

    def buffers(self) -> dict[str, torch.Tensor]:
        return _head_tensors(self.state.kind, self.source, parameters=False)

    def capture(self, keys: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
        import json
        from uuid import uuid4

        from ._tensors import detach_tree

        self._validate()
        state = self.state
        keys = tuple(state.parameters) if keys is None else keys
        version = state.version
        handle = "head:" + json.dumps(
            {
                "checkpoint": version.checkpoint,
                "generation": version.generation,
                "revision": version.revision,
                "name": state.name,
                "keys": keys,
                "max_gradient_staleness": self.max_gradient_staleness,
                "capture": uuid4().hex,
            },
            separators=(",", ":"),
        )
        current = self.parameters()
        parameters = {
            key: state.parameters[key].to(
                device=current[key].device, dtype=current[key].dtype
            )
            for key in keys
        }
        if not torch.is_grad_enabled():
            return {key: value.detach().clone() for key, value in parameters.items()}
        from ._parameter_hooks import parameter_hooks

        registries = self.collector._head_hooks
        registries[handle] = tuple(parameter_hooks(current[key]) for key in keys)
        try:
            return self.collector.attach(
                detach_tree(handle, parameters),
                on_release=lambda: registries.pop(handle, None),
            )
        except BaseException:
            registries.pop(handle, None)
            raise

    def capture_module(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        parameters = self.capture()
        return parameters, {key: _plain(value) for key, value in self.buffers().items()}

    def publish(self, buffers: Mapping[str, torch.Tensor]) -> None:
        self._validate()
        staged = _stage_local_buffers(self.buffers(), buffers)
        with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
            for target, value in staged:
                target.copy_(value)
        self.pending = self.pending or bool(staged)

    def refresh(self, state: HeadState) -> None:
        if state.version.generation != self.state.version.generation:
            self.invalid = True
            return
        self._validate()
        if state.version.revision < self.state.version.revision:
            return
        if (
            state.kind != self.state.kind
            or state.parameters.keys() != self.state.parameters.keys()
            or state.buffers.keys() != self.state.buffers.keys()
        ):
            raise ValueError("Custom checkpoint object schema changed")
        targets = self.parameters() | self.buffers()
        keep_buffers = (
            self.pending or state.buffer_revision < self.state.buffer_revision
        )
        if keep_buffers:
            state = replace(
                state,
                buffers=self.state.buffers,
                buffer_revision=self.state.buffer_revision,
            )
        with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
            for key, value in (
                state.parameters | ({} if keep_buffers else state.buffers)
            ).items():
                targets[key].copy_(value.to(targets[key].device))
        self.state = state

    def take_publication(self) -> HeadBufferUpdate | None:
        self._validate()
        current = self.buffers()
        if not self.pending and all(
            torch.equal(_plain(value).cpu(), self.state.buffers[key])
            for key, value in current.items()
        ):
            return None
        self.pending = False
        result = HeadBufferUpdate(
            self.state.version,
            self.state.name,
            self.state.buffer_revision,
            {
                key: _plain(value).to(device="cpu", dtype=self.state.buffers[key].dtype)
                for key, value in current.items()
            },
        )
        # Publication is ordered before subsequent operations by the owner. Keep
        # local revision in sequence for calls made before that operation resolves.
        self.state = replace(
            self.state,
            buffers=result.buffers,
            buffer_revision=self.state.buffer_revision + 1,
        )
        return result


def synchronize_head_buffers(trainer: TrainerRank, checkpoints: Any = None) -> None:
    """Publish logical DP zero's persistent buffers at an ordered boundary.

    Parameter gradients still reduce across DP. Arbitrary counters and running
    statistics are copied from the authority, never averaged.
    """
    import torch.distributed as dist

    from . import _checkpoint

    if not (dist.is_available() and dist.is_initialized()):
        return
    group = trainer._checkpoint_group()
    names = sorted(trainer._checkpoint_slots if checkpoints is None else checkpoints)
    targets = {}
    for checkpoint in names:
        for name, custom in trainer._checkpoint_slots[checkpoint].custom.items():
            if custom.kind == "parameter":
                continue
            if custom.kind == "buffer":
                buffers = _head_tensors(custom.kind, custom.value, parameters=False)
            else:
                persistent = {
                    id(buffer)
                    for child in cast(torch.nn.Module, custom.value).modules()
                    for key, buffer in child._buffers.items()
                    if buffer is not None
                    and key not in child._non_persistent_buffers_set
                }
                buffers = {
                    key: value
                    for key, value in _head_tensors(
                        custom.kind, custom.value, parameters=False
                    ).items()
                    if id(value) in persistent
                }
            targets[(checkpoint, name)] = (_custom_tracker(custom), buffers)
    revisions = _checkpoint._gather(
        {key: tracker.buffer_revision for key, (tracker, _) in targets.items()}, group
    )
    if any(peer.keys() != targets.keys() for peer in revisions):
        raise trainer._slot_state_error(
            "Custom buffer registrations differ across ranks"
        )
    payload = _checkpoint._phase(
        lambda: (
            {
                key: (
                    tracker.buffer_revision,
                    {name: _plain(value).cpu() for name, value in buffers.items()},
                )
                for key, (tracker, buffers) in targets.items()
            }
            if dist.get_rank(group) == 0
            else None
        ),
        "snapshot synchronized buffers",
        group,
    )
    authoritative = _checkpoint._gather(payload, group)[0]
    assert authoritative is not None
    if authoritative.keys() != targets.keys():
        raise trainer._slot_state_error(
            "Custom buffer registrations differ across ranks"
        )
    staged, local_differences, error = {}, {}, None
    try:
        for key, (tracker, buffers) in targets.items():
            staged[key] = _stage_local_buffers(buffers, authoritative[key][1])
            local_differences[key] = tracker.buffer_revision != authoritative[key][
                0
            ] or bool(staged[key])
    except Exception as exc:
        error = exc
    _checkpoint.raise_distributed(error, "validate synchronized buffers", group)
    differences = _checkpoint._gather(local_differences, group)
    with torch.no_grad(), torch._C.DisableTorchFunctionSubclass():
        for key, (tracker, _) in targets.items():
            for target, value in staged[key]:
                target.copy_(value)
            tracker.buffer_revision = max(peer[key] for peer in revisions) + int(
                any(peer[key] for peer in differences)
            )


def _logical_heads(view: Any) -> dict[tuple[str, str], LiveHead]:
    rank = view._rank
    if not hasattr(rank, "_logical_head_handles"):
        rank._logical_head_handles = {}
    return rank._logical_head_handles


def logical_register_head(
    view: Any,
    kind: HeadKind,
    name: str,
    factory: Callable[[], Any],
    *,
    checkpoint: Any = ...,
) -> Any:
    from ._options import resolve_forward_options

    if checkpoint is ...:
        from ._impl import Unset

        checkpoint = Unset

    flush_logical_heads(view)

    checkpoint, state = view._invoke("head", "head_lookup", (checkpoint, name))
    if state is not None and state.kind != kind:
        raise ValueError(f"Checkpoint object {name!r} is already a {state.kind}")
    registry = _logical_heads(view)
    current = registry.get((checkpoint, name))
    if current is not None and not current.invalid and state is not None:
        current.refresh(state)
        if not current.invalid:
            return current.value
    if state is None:
        value, error, factory_error = None, None, None
        try:
            value = factory()
        except BaseException as exc:
            error = exc
            factory_error = type(exc).__name__
            try:
                factory_error += f": {exc}"
            except BaseException:
                pass
        try:
            state = view._invoke(
                "head",
                "head_register",
                HeadRegistration(checkpoint, name, kind, value, factory_error),
            ).state
        except BaseException:
            if error is None:
                raise
        # Preserve the factory's original type, identity and chain at its owner;
        # physical command peers receive an ordinary coordinated failure.
        if error is not None:
            raise error
    else:
        value = deepcopy(view._rank._checkpoint_slots[checkpoint].custom[name].value)
    assert value is not None
    value = (
        move_module(value, view.device)
        if isinstance(value, torch.nn.Module)
        else value.to(view.device)
    )
    maximum = resolve_forward_options(
        getattr(view._rank, "_forward_options", None)
    ).max_gradient_staleness
    head = LiveHead(
        state, value, view._executor.state.collector, max_gradient_staleness=maximum
    )
    registry[(checkpoint, name)] = head
    return head.value


def flush_logical_heads(view: Any) -> None:
    updates = tuple(
        update
        for head in _logical_heads(view).values()
        if not head.invalid and (update := head.take_publication()) is not None
    )
    if updates:
        try:
            view._executor.invoke("head", "head_publish", updates)
        except BaseException:
            for update in updates:
                _logical_heads(view)[
                    (update.version.checkpoint, update.name)
                ].invalidate("buffer publication failed; register the object again")
            raise


def refresh_logical_heads(view: Any) -> None:
    registry = _logical_heads(view)
    keys = tuple(key for key, head in registry.items() if not head.invalid)
    if keys:
        states = view._executor.invoke("head", "head_export", keys)
        for key, observation in zip(keys, states, strict=True):
            state = observation.state
            if state is None:
                registry[key].invalid = True
            else:
                registry[key].refresh(state)
