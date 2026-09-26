"""Persistent live-parameter hooks applied to one backward's summed gradient."""

from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
from typing import Any
import weakref

import torch
from torch.utils.hooks import RemovableHandle

parameter_hook_active: ContextVar[bool] = ContextVar(
    "parameter_hook_active", default=False
)


@dataclass(eq=False)
class ParameterHooks:
    parameter: weakref.ReferenceType[torch.Tensor]
    hooks: OrderedDict[int, Any] = field(default_factory=OrderedDict)

    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        for hook in tuple(self.hooks.values()):
            token = parameter_hook_active.set(True)
            try:
                result = hook(gradient)
            finally:
                parameter_hook_active.reset(token)
            if result is None:
                continue
            if not isinstance(result, torch.Tensor) or (
                result.shape != gradient.shape
                or result.dtype != gradient.dtype
                or result.device != gradient.device
                or result.layout != gradient.layout
            ):
                raise RuntimeError(
                    "Live parameter hook must return None or a gradient with unchanged shape, dtype, device and layout"
                )
            gradient = result
        return gradient


def parameter_hooks(parameter: torch.Tensor) -> ParameterHooks:
    registry = getattr(parameter, "_art_parameter_hooks", None)
    if registry is None:
        registry = ParameterHooks(weakref.ref(parameter))
        setattr(parameter, "_art_parameter_hooks", registry)
    return registry


def register_parameter_hook(parameter: torch.Tensor, hook: Any) -> RemovableHandle:
    """Run once per explicit backward, before accumulating into existing .grad."""
    if not parameter.requires_grad:
        raise RuntimeError(
            "cannot register a hook on a tensor that doesn't require gradient"
        )
    if not callable(hook):
        raise TypeError("parameter hook must be callable")
    registry = parameter_hooks(parameter)
    handle = RemovableHandle(registry.hooks)
    registry.hooks[handle.id] = hook
    return handle


def reject_post_accumulate_hook(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "Live checkpoint parameters do not support register_post_accumulate_grad_hook; use register_hook before transactional gradient publication"
    )


def apply_parameter_hooks(
    parameter: torch.Tensor, gradient: torch.Tensor
) -> torch.Tensor:
    registry = getattr(parameter, "_art_parameter_hooks", None)
    if registry is not None:
        result = registry.apply(gradient)
        if result is not gradient:
            gradient.copy_(result)
    return gradient


def apply_head_hooks(
    packets: tuple[Any, ...], registries: dict[str, tuple[ParameterHooks, ...]]
) -> tuple[Any, ...]:
    """Combine hooked client captures without weakening any origin's expiry."""
    from ._tensors import CotangentPacket

    groups: dict[ParameterHooks, list[tuple[int, int, Any]]] = {}
    gradients = [list(packet.gradients) for packet in packets]
    for index, packet in enumerate(packets):
        for column, registry in enumerate(registries.get(packet.handle, ())):
            if registry.hooks and gradients[index][column] is not None:
                groups.setdefault(registry, []).append(
                    (index, column, json.loads(packet.handle[5:]))
                )
    combined = []
    with torch.no_grad():
        for registry, entries in groups.items():
            oldest = min(entries, key=lambda entry: entry[2]["revision"])
            metadata = dict(oldest[2])
            gradient = gradients[oldest[0]][oldest[1]]
            parameter = registry.parameter()
            if parameter is not None:
                gradient = gradient.to(device=parameter.device, dtype=parameter.dtype)
            gradient = gradient.clone()
            for index, column, _ in entries:
                if (index, column) != oldest[:2]:
                    addition = gradients[index][column].to(gradient)
                    if (
                        gradient.layout != torch.strided
                        and addition.layout == torch.strided
                    ):
                        gradient = addition + gradient
                    else:
                        gradient.add_(addition)
                gradients[index][column] = None
            gradient = registry.apply(gradient)
            metadata["keys"] = [metadata["keys"][oldest[1]]]
            metadata["capture"] = "hooks:" + metadata["capture"]
            metadata["max_gradient_staleness"] = (
                min(
                    origin["revision"] + origin["max_gradient_staleness"]
                    for _, _, origin in entries
                )
                - metadata["revision"]
            )
            combined.append(
                CotangentPacket(
                    "head:" + json.dumps(metadata, separators=(",", ":")), (gradient,)
                )
            )
    # Keep original packets for validation even when their gradients were folded
    # into a single packet. The combined expiry also governs later optim_step.
    return tuple(
        CotangentPacket(packet.handle, tuple(values))
        for packet, values in zip(packets, gradients, strict=True)
    ) + tuple(combined)
