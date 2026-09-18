"""Detached output trees and transaction-scoped, first-order cotangents."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from threading import Lock
from typing import Any
import weakref

import torch


@dataclass(frozen=True)
class TensorTreeSpec:
    kind: str
    context: Any = None
    children: tuple[TensorTreeSpec, ...] = ()


def flatten_tensors(tree: Any) -> tuple[tuple[torch.Tensor, ...], TensorTreeSpec]:
    """Flatten tensor leaves once per identity, preserving container structure."""
    from ._impl import Unset

    tensors: list[torch.Tensor] = []
    indices: dict[int, int] = {}

    def visit(value: Any) -> TensorTreeSpec:
        if value is Unset:
            return TensorTreeSpec("unset")
        if isinstance(value, torch.Tensor):
            if id(value) not in indices:
                indices[id(value)] = len(tensors)
                tensors.append(value)
            return TensorTreeSpec("tensor", indices[id(value)])
        if is_dataclass(value) and not isinstance(value, type):
            names = tuple(field.name for field in fields(value))
            children = tuple(visit(getattr(value, name)) for name in names)
            if isinstance(value, dict):
                return TensorTreeSpec(
                    "dataclass_dict",
                    (
                        type(value),
                        names,
                        tuple(value),
                        getattr(value, "default_factory", None),
                    ),
                    children + tuple(visit(item) for item in value.values()),
                )
            return TensorTreeSpec("dataclass", (type(value), names), children)
        if isinstance(value, dict):
            return TensorTreeSpec(
                "dict",
                (type(value), tuple(value), getattr(value, "default_factory", None)),
                tuple(visit(item) for item in value.values()),
            )
        if isinstance(value, (list, tuple)):
            return TensorTreeSpec("sequence", type(value), tuple(map(visit, value)))
        return TensorTreeSpec("constant", value)

    try:
        spec = visit(tree)
        return tuple(tensors), spec
    finally:
        # The recursive closure otherwise retains its tensor list until cyclic GC.
        del visit


def unflatten_tensors(spec: TensorTreeSpec, tensors: Sequence[torch.Tensor]) -> Any:
    if spec.kind == "unset":
        from ._impl import Unset

        return Unset
    if spec.kind == "tensor":
        return tensors[spec.context]
    if spec.kind == "constant":
        return spec.context
    values = [unflatten_tensors(child, tensors) for child in spec.children]
    if spec.kind in {"dataclass", "dataclass_dict"}:
        cls, names = spec.context[:2]
        if spec.kind == "dataclass_dict":
            # ModelOutput-style dataclasses have a builtin mapping allocation
            # and may prohibit update(); restore their actual mapping separately.
            base = OrderedDict if issubclass(cls, OrderedDict) else dict
            result = base.__new__(cls)
            base.__init__(
                result, zip(spec.context[2], values[len(names) :], strict=True)
            )
        else:
            result = object.__new__(cls)
        for name, value in zip(names, values[: len(names)], strict=True):
            object.__setattr__(result, name, value)
        return result
    if spec.kind == "dict":
        cls, keys, factory = spec.context
        result = cls() if factory is None else cls(factory)
        result.update(zip(keys, values, strict=True))
        return result
    if spec.kind == "sequence":
        cls = spec.context
        return cls(*values) if hasattr(cls, "_fields") else cls(values)
    raise ValueError(f"Unknown tensor tree node: {spec.kind!r}")


def _map_tensors(fn: Callable[[torch.Tensor], torch.Tensor], tree: Any) -> Any:
    tensors, spec = flatten_tensors(tree)
    return unflatten_tensors(spec, tuple(map(fn, tensors)))


def _plain(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, ManagedTensor):
        with torch.enable_grad(), torch._C.DisableTorchFunctionSubclass():
            return tensor.as_subclass(torch.Tensor)
    return tensor


@dataclass(frozen=True)
class TensorPacket:
    handle: str
    spec: TensorTreeSpec
    tensors: tuple[torch.Tensor, ...]
    requires_grad: tuple[bool, ...]


@dataclass(frozen=True)
class CotangentPacket:
    handle: str
    gradients: tuple[torch.Tensor | None, ...]


def _validate_output_spec(spec: TensorTreeSpec) -> None:
    def metadata(value: Any) -> None:
        if type(value) is tuple:
            for item in value:
                metadata(item)
        elif type(value) not in (
            type(None),
            bool,
            int,
            float,
            complex,
            str,
            bytes,
        ) and not isinstance(
            value, (torch.dtype, torch.device, torch.layout, torch.memory_format)
        ):
            raise TypeError(
                f"Unsupported output tree metadata {type(value).__name__}; "
                "use tensor leaves, dataclasses, dictionaries, lists, tuples and scalar metadata."
            )

    if spec.kind == "constant":
        metadata(spec.context)
    elif spec.kind == "dict":
        metadata(spec.context[1])
        metadata(spec.context[2])
    elif spec.kind == "dataclass_dict":
        metadata(spec.context[2])
        metadata(spec.context[3])
    for child in spec.children:
        _validate_output_spec(child)


def detach_tree(
    handle: str, tree: Any, *, device: torch.device | str | None = None
) -> TensorPacket:
    """Snapshot supported output containers; reject opaque tensor-bearing objects."""
    tensors, spec = flatten_tensors(tree)
    _validate_output_spec(spec)
    return TensorPacket(
        handle,
        spec,
        tuple(
            _plain(tensor).detach().to(device=device, copy=True) for tensor in tensors
        ),
        tuple(tensor.requires_grad for tensor in tensors),
    )


class _OutputBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchor, collector, handle, requires_grad, on_release, *values):
        ctx.collector, ctx.handle = collector, handle
        ctx.signature = tuple(
            (value.shape, value.dtype, value.device, required)
            for value, required in zip(values, requires_grad, strict=True)
        )
        if on_release is not None:
            weakref.finalize(ctx, on_release).atexit = False
        ctx.set_materialize_grads(False)
        # Accessed by backward so PyTorch enforces retain_graph on repeated use.
        ctx.save_for_backward(anchor)
        ctx.mark_non_differentiable(
            *(
                value
                for value, required in zip(values, requires_grad, strict=True)
                if not required
            )
        )
        return tuple(values)

    @staticmethod
    def backward(ctx, *gradients):
        ctx.saved_tensors
        ctx.collector._record(ctx.handle, ctx.signature, gradients)
        return (None,) * (5 + len(gradients))


class _CollectionRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, collector, *outputs):
        ctx.collector = collector
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(
            *(value for value in outputs if not value.requires_grad)
        )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *gradients):
        with ctx.collector._lock:
            ctx.collector._graph_task_id = torch._C._current_graph_task_id()
        return (None, *gradients)


class CotangentCollector:
    """Collect every local cotangent before the caller commits any remote work.

    One instance belongs to one caller/owner, including its head snapshots. A
    concurrent or nested remote backward on the same collector is rejected; pass
    coupled losses together instead. Ordinary local recomputation is supported. Local
    parameter gradients follow normal PyTorch accumulation semantics; remote
    cotangents are discarded if any part of local backward raises.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: dict[str, tuple[torch.Tensor | None, ...]] | None = None
        self._signatures: dict[str, tuple[Any, ...]] = {}
        self._graph_task_id: int | None = None
        self._head_hooks: dict[str, tuple[Any, ...]] = {}

    def attach(
        self,
        packet: TensorPacket,
        *,
        managed: bool = False,
        on_release: Callable[[], None] | None = None,
    ) -> Any:
        """Attach outputs; optionally release their owner when the graph dies.

        The callback follows the autograd context, including dependent losses,
        and must be nonblocking and safe after explicit graph consumption. For
        packets without differentiable outputs, release happens immediately.
        Grad mode applies normally. Differentiable outputs are read-only custom
        Function views; clone before in-place changes, including under no_grad.
        """
        _validate_output_spec(packet.spec)
        if len(packet.tensors) != len(packet.requires_grad):
            raise ValueError(
                "Tensor packet values and requires_grad flags differ in length"
            )
        values = tuple(_plain(tensor).detach() for tensor in packet.tensors)
        if any(packet.requires_grad):
            for value, required in zip(values, packet.requires_grad, strict=True):
                if required and not (value.is_floating_point() or value.is_complex()):
                    raise ValueError(
                        "Only floating-point or complex outputs can require gradients"
                    )
            values = _OutputBridge.apply(
                torch.empty(0, requires_grad=True),
                self,
                packet.handle,
                packet.requires_grad,
                on_release,
                *values,
            )
        elif on_release is not None:
            on_release()
        if managed:
            values = tuple(map(managed_tensor, values))
        return unflatten_tensors(packet.spec, values)

    def _record(
        self,
        handle: str,
        signature: tuple[Any, ...],
        gradients: Sequence[torch.Tensor | None],
    ) -> None:
        copies = tuple(
            None if grad is None else _plain(grad).detach().clone()
            for grad in gradients
        )
        with self._lock:
            if (
                self._pending is None
                or self._graph_task_id != torch._C._current_graph_task_id()
            ):
                raise RuntimeError(
                    "Use the owning trainer.backward(loss) to collect remote cotangents; "
                    "unscoped or nested remote backward is unsupported. Pass coupled losses together."
                )
            if handle in self._signatures and self._signatures[handle] != signature:
                raise ValueError(
                    f"Incompatible output signatures for repeated handle {handle!r}"
                )
            self._signatures[handle] = signature
            previous = self._pending.get(handle)
            if previous is not None:
                copies = tuple(
                    new
                    if old is None
                    else old
                    if new is None
                    else new + old
                    if old.layout != torch.strided and new.layout == torch.strided
                    else old + new
                    for old, new in zip(previous, copies, strict=True)
                )
            self._pending[handle] = copies

    def backward(
        self,
        loss: torch.Tensor | Sequence[torch.Tensor],
        gradient: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
        *,
        retain_graph: bool = False,
    ) -> tuple[CotangentPacket, ...]:
        from ._heads import _ClientParameter
        from ._impl import _TrackedParameter

        with self._lock:
            if self._pending is not None:
                raise RuntimeError(
                    "A backward collection is already active on this owner"
                )
            self._pending = {}
        try:
            outputs = (loss,) if isinstance(loss, torch.Tensor) else tuple(loss)
            # A single root records the engine task before any upstream hooks or
            # bridges execute, including when local backward runs on CUDA workers.
            with torch.enable_grad():
                # Direct live roots need the same version capture as arithmetic.
                # Deduplicate aliases so a repeated root shares one snapshot.
                outputs = _map_tensors(
                    lambda value: (
                        value.clone()
                        if isinstance(value, _ClientParameter | _TrackedParameter)
                        else value
                    ),
                    outputs,
                )
                roots = _CollectionRoot.apply(self, *map(_plain, outputs))
            head_hooks = self._head_hooks.copy()
            torch.autograd.backward(roots, gradient, retain_graph=retain_graph)
            with self._lock:
                packets = tuple(
                    CotangentPacket(handle, grads)
                    for handle, grads in sorted(self._pending.items())
                )
            from ._parameter_hooks import apply_head_hooks

            return apply_head_hooks(packets, head_hooks)
        finally:
            with self._lock:
                self._pending = None
                self._signatures.clear()
                self._graph_task_id = None


# Implicit copies are only safe for known pure operations. Stateful functional
# calls (for example BatchNorm and embedding(max_norm=...)) can mutate arguments
# without an in-place name or even a tensor version-counter change.
_MIXED_DEVICE_PURE_OPS = frozenset(
    "__add__ __radd__ __sub__ __rsub__ __mul__ __rmul__ __truediv__ __rtruediv__ "
    "__floordiv__ __rfloordiv__ __pow__ __rpow__ __mod__ __rmod__ __matmul__ __rmatmul__ "
    "__eq__ __ne__ __lt__ __le__ __gt__ __ge__ __and__ __rand__ __or__ __ror__ __xor__ __rxor__ "
    "__getitem__ add sub subtract mul multiply div divide true_divide floor_divide pow "
    "remainder fmod maximum minimum fmax fmin eq ne lt le gt ge equal allclose isclose "
    "logical_and logical_or logical_xor bitwise_and bitwise_or bitwise_xor "
    "matmul mm bmm mv dot vdot inner outer addmm addbmm baddbmm addmv addr linear "
    "einsum tensordot bilinear cat concat concatenate stack hstack vstack dstack "
    "where lerp clamp clip masked_select gather take take_along_dim index_select "
    "scatter scatter_add scatter_reduce index_add index_copy index_fill index_put "
    "mse_loss l1_loss smooth_l1_loss huber_loss binary_cross_entropy "
    "binary_cross_entropy_with_logits cross_entropy nll_loss kl_div poisson_nll_loss "
    "cosine_similarity cosine_embedding_loss hinge_embedding_loss margin_ranking_loss "
    "triplet_margin_loss pairwise_distance pdist cdist".split()
)


class ManagedTensor(torch.Tensor):
    """Eager CPU/CUDA interop that copies operands through ordinary autograd.

    Computation follows managed placement; CPU wins when managed operands
    disagree. Results remain managed. Mixed-device mutation and multiple CUDA
    devices require an explicit move. Mixed-device support is limited to pure
    arithmetic, linear algebra, indexing, tensor combination and common losses.
    Stateful or unknown mixed-device operations require explicit placement.
    Arbitrary subclasses and compiled graphs are outside this eager interface.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # Let live parameter/buffer proxies capture their values before running
        # the operation with subclass dispatch disabled.
        if not all(issubclass(cls, other) for other in types):
            return NotImplemented
        name = getattr(func, "__name__", "")
        identity_property = name == "__get__" and getattr(
            getattr(func, "__self__", None), "__name__", ""
        ) not in {"T", "mT", "H", "mH", "real", "imag"}
        if (
            identity_property
            or func
            in (
                torch.autograd.grad,
                torch.autograd.backward,
                torch.Tensor.backward,
            )
            or name
            in {
                "__set__",
                "register_hook",
                "register_post_accumulate_grad_hook",
                "retain_grad",
                "requires_grad_",
                "detach_",
            }
        ):
            # Autograd targets and metadata belong to the original tensor, not
            # a fresh alias (which is absent from the user's existing graph).
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **(kwargs or {}))
        kwargs = kwargs or {}
        originals, _ = flatten_tensors((args, kwargs))
        identities = {id(tensor) for tensor in originals}
        # Operations must receive the original TensorImpl/autograd identity.
        # Unwrapping into aliases loses metadata mutations and existing hooks.
        with torch._C.DisableTorchFunctionSubclass():
            devices = {tensor.device for tensor in originals}
            if len(devices) > 1 and name not in {"to", "type_as"}:
                accelerators = {device for device in devices if device.type != "cpu"}
                if len(accelerators) != 1 or next(iter(accelerators)).type != "cuda":
                    raise RuntimeError(
                        "Managed tensors require an explicit move between accelerator devices"
                    )
                if name not in _MIXED_DEVICE_PURE_OPS or kwargs.get("out") is not None:
                    raise RuntimeError(
                        f"Managed mixed-device operation {name!r} across "
                        f"{', '.join(sorted(map(str, devices)))} may perform mutation "
                        "or is unsupported; use explicit tensor .to(device) placement."
                    )
                managed_devices = {
                    tensor.device
                    for tensor in originals
                    if isinstance(tensor, ManagedTensor)
                }
                device = (
                    torch.device("cpu")
                    if torch.device("cpu") in managed_devices
                    else next(iter(managed_devices))
                )
                args, kwargs = _map_tensors(
                    lambda tensor: tensor.to(device), (args, kwargs)
                )
            result = func(*args, **kwargs)

            def wrap(tensor: torch.Tensor) -> torch.Tensor:
                # Keep no-op/in-place/out identities, including ordinary operands.
                # New results already own the right autograd/view metadata; an
                # as_subclass alias would turn even clone() into a view and lose
                # its hooks when a later in-place operation rebases that view.
                if id(tensor) not in identities:
                    tensor.__class__ = ManagedTensor
                return tensor

            return _map_tensors(wrap, result)


def managed_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, ManagedTensor):
        return tensor
    with torch.enable_grad(), torch._C.DisableTorchFunctionSubclass():
        return tensor.as_subclass(ManagedTensor)


def managed_tree(tree: Any, *, device: torch.device | str | None = None) -> Any:
    """Place a nested output tree while preserving its original gradient paths."""
    return _map_tensors(lambda tensor: managed_tensor(tensor.to(device=device)), tree)
