"""Packed frozen expert bases. Activation belongs to resident TrainerRank."""

from collections.abc import Callable
from typing import Any
import weakref

import torch


def _pack_weights(linear: torch.nn.Module) -> tuple[int, int, int]:
    weights = [getattr(linear, f"weight{i}") for i in range(linear.num_gemms)]
    # The registered Parameters remain the only persistent storage owners. A view
    # made in forward is an input to normal outer compile, not a cached GPU copy.
    with torch.no_grad():
        packed = torch.stack(weights)
        for weight, view in zip(weights, packed, strict=True):
            weight.data = view
    return tuple(packed.shape)


def _supported(linear: torch.nn.Module) -> bool:
    if not hasattr(torch, "_grouped_mm") or linear.num_gemms < 1:
        return False
    config = linear.config
    if (
        linear.tp_size != 1
        or linear.sequence_parallel
        or linear.te_return_bias
        or linear.te_quant_params is not None
        or config.fp8
        or config.delay_wgrad_compute
        or getattr(config, "fp4", None)
    ):
        return False
    weights = [getattr(linear, f"weight{i}") for i in range(linear.num_gemms)]
    first = weights[0]
    if (
        first.device.type != "cuda"
        or first.dtype != torch.bfloat16
        or first.ndim != 2
        or any(size % 8 for size in first.shape)
        or torch.cuda.get_device_capability(first.device) != (9, 0)
    ):
        return False
    return all(
        type(weight) is torch.nn.Parameter
        and not weight.requires_grad
        and weight.shape == first.shape
        and weight.dtype == first.dtype
        and weight.device == first.device
        and weight.is_contiguous()
        for weight in weights
    ) and not any(
        name.startswith("bias") and value.numel()
        for name, value in linear.named_parameters(recurse=False)
    )


def _first_layout(weight: torch.nn.Parameter) -> tuple[Any, ...]:
    storage = weight.untyped_storage()
    return (
        id(weight),
        weight.device,
        weight.dtype,
        tuple(weight.shape),
        weight.stride(),
        weight.storage_offset(),
        storage.data_ptr(),
        storage.nbytes(),
    )


@torch.compiler.disable
def _offsets(
    owner: "FrozenGroupedBase", x: torch.Tensor, splits: list[int] | torch.Tensor
) -> torch.Tensor | None:
    # Standard child moves/views transform all Parameters. Check one layout here
    # before constructing a view, without scanning expert storage or graph breaks
    # beyond this existing routing boundary. Nonuniform raw .data replacement is
    # outside the resident owner's contract.
    if _first_layout(owner._grouped_linear().weight0) != owner._grouped_first:
        owner._retire_grouped_base()
        return None
    # Python routing lists otherwise become outer-Dynamo value guards, creating
    # a new graph for each routing pattern. Keep values in an invocation tensor.
    return torch.as_tensor(splits, device=x.device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )


def prepare_grouped_bases(model: list[torch.nn.Module]) -> None:
    for chunk in model:
        for module in chunk.modules():
            if isinstance(module, FrozenGroupedBase):
                module._prepare_grouped_base(True)


class FrozenGroupedBase(torch.nn.Module):
    """Lifecycle boundary for FC LoRA wrappers; unprepared calls retain TE.

    Standard wrapper/root moves and state loading retire the packed route. A new
    resident TrainerRank may prepare again, outside any running job. Offload and
    streaming owners exclude their modules before changing weight storage.
    """

    def __init__(self) -> None:
        super().__init__()
        self._grouped_shape: tuple[int, int, int] | None = None
        self._grouped_weights: tuple[torch.nn.Parameter, ...] = ()
        self._grouped_first: tuple[Any, ...] | None = None
        self._grouped_resident = True
        self._grouped_preparations = 0
        self._grouped_load_hook = None

    def _retire_grouped_base(self) -> None:
        self._grouped_shape = None
        self._grouped_weights = ()
        self._grouped_first = None

    def _apply(self, fn: Callable, recurse: bool = True) -> Any:
        self._retire_grouped_base()
        return super()._apply(fn, recurse=recurse)

    def requires_grad_(self, requires_grad: bool = True) -> Any:
        self._retire_grouped_base()
        return super().requires_grad_(requires_grad)

    def _prepare_grouped_base(self, enabled: bool) -> None:
        if not enabled or not self._grouped_resident:
            self._retire_grouped_base()
            return
        if self._grouped_shape is not None:
            return
        linear = self._grouped_linear()
        if not _supported(linear):
            return
        if self._grouped_load_hook is None:
            owner = weakref.ref(self)

            def retire(*_args: Any) -> None:
                if (module := owner()) is not None:
                    module._retire_grouped_base()

            self._grouped_load_hook = linear.register_load_state_dict_pre_hook(retire)
        self._grouped_shape = _pack_weights(linear)
        self._grouped_weights = tuple(
            getattr(linear, f"weight{i}") for i in range(linear.num_gemms)
        )
        self._grouped_first = _first_layout(linear.weight0)
        self._grouped_preparations += 1

    def _grouped_linear(self) -> torch.nn.Module:
        raise NotImplementedError

    def _base_forward(
        self, x: torch.Tensor, splits: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        linear = self._grouped_linear()
        shape = self._grouped_shape
        if (
            shape is None
            or x.dtype != torch.bfloat16
            or any(weight.requires_grad for weight in self._grouped_weights)
        ):
            return linear(x, splits)
        _, n, k = shape
        # Invocation-owned offsets survive overlapping forwards and recompute.
        offsets = _offsets(self, x, splits)
        if offsets is None:
            return linear(x, splits)
        weight = linear.weight0.as_strided(shape, (n * k, k, 1))
        result = torch._grouped_mm(x, weight.transpose(1, 2), offs=offsets)
        linear.is_first_microbatch = False
        return result, None
