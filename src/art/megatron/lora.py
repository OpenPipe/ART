from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass, replace
import functools
import importlib
import json
import math
import os
import re
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    NotRequired,
    TypedDict,
    TypeVar,
    cast,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core import parallel_state as ps
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelGroupedLinear,
    TERowParallelLinear,
)
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_layer import TransformerLayer
import torch

from art.megatron._collective import dtype_name as _dtype_name

from .kernels.cute_grouped_lora_quack import (
    quack_grouped_lora,
    quack_grouped_lora_dual,
)

MOE_LORA_RANK = 1
DENSE_LORA_RANK = 8
LORA_ALPHA = 32
MEGATRON_LORA_RANK_ENV = "ART_MEGATRON_LORA_RANK"
MEGATRON_LORA_TARGET_MODULES_ENV = "ART_MEGATRON_LORA_TARGET_MODULES"
_LAYER_BLOCK_RE = re.compile(r"^(?P<block>.*\.layers\.\d+)\.")

ShardDomain = Literal["tp", "expert_tp"]
GradSyncDomain = Literal["tp_default", "expert_tp"]
GradSyncOp = Literal["none", "sum", "avg"]
_F = TypeVar("_F", bound=Callable[..., Any])

TP_DEFAULT_GRAD_SYNC_DOMAIN: GradSyncDomain = "tp_default"
EXPERT_TP_GRAD_SYNC_DOMAIN: GradSyncDomain = "expert_tp"
GRAD_SYNC_OP_NONE: GradSyncOp = "none"
GRAD_SYNC_OP_SUM: GradSyncOp = "sum"
GRAD_SYNC_OP_AVG: GradSyncOp = "avg"


@dataclass(frozen=True)
class LoRASlotRef:
    name: str | None


_CURRENT_LORA_SLOT: contextvars.ContextVar[LoRASlotRef | None] = contextvars.ContextVar(
    "art_megatron_current_lora_slot", default=None
)


@contextmanager
def use_lora_slot(ref: LoRASlotRef | None) -> Iterator[None]:
    token = _CURRENT_LORA_SLOT.set(ref)
    try:
        yield
    finally:
        _CURRENT_LORA_SLOT.reset(token)


def _with_captured_lora_slot(function: _F) -> _F:
    context = _CURRENT_LORA_SLOT.get()

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        token = _CURRENT_LORA_SLOT.set(context)
        try:
            return function(*args, **kwargs)
        finally:
            _CURRENT_LORA_SLOT.reset(token)

    return cast(_F, wrapped)


def _patch_function_once(module: Any, name: str, wrapper: Callable[[_F], _F]) -> None:
    original = getattr(module, name, None)
    if original is None or getattr(original, "_art_lora_slot_context_patch", False):
        return
    patched = wrapper(original)
    setattr(patched, "_art_lora_slot_context_patch", True)
    setattr(module, name, patched)


def install_lora_checkpoint_context_hooks() -> None:
    """Preserve the selected dynamic LoRA slot across activation recompute."""

    def wrap_checkpoint(original: _F, function_index: int) -> _F:
        @functools.wraps(original)
        def checkpoint(*args: Any, **kwargs: Any) -> Any:
            if len(args) > function_index:
                args = (
                    *args[:function_index],
                    _with_captured_lora_slot(args[function_index]),
                    *args[function_index + 1 :],
                )
            elif "function" in kwargs:
                kwargs = {
                    **kwargs,
                    "function": _with_captured_lora_slot(kwargs["function"]),
                }
            elif "forward_func" in kwargs:
                kwargs = {
                    **kwargs,
                    "forward_func": _with_captured_lora_slot(kwargs["forward_func"]),
                }
            else:
                raise TypeError("checkpoint wrapper could not find callable argument")
            return original(*args, **kwargs)

        return cast(_F, checkpoint)

    def patch(target: str, name: str, function_index: int) -> None:
        try:
            module_name, _, attr_path = target.partition(":")
            target_obj = importlib.import_module(module_name)
            for attr in attr_path.split(".") if attr_path else ():
                target_obj = getattr(target_obj, attr, None)
                if target_obj is None:
                    return
            _patch_function_once(
                target_obj,
                name,
                lambda original: wrap_checkpoint(original, function_index),
            )
        except Exception:
            pass

    for target, name, function_index in (
        ("torch.utils.checkpoint", "checkpoint", 0),
        ("megatron.core.tensor_parallel", "checkpoint", 0),
        ("megatron.core.tensor_parallel.random", "checkpoint", 0),
        (
            "megatron.core.tensor_parallel.random:CheckpointWithoutOutput",
            "checkpoint",
            1,
        ),
        ("megatron.core.transformer.transformer_block", "te_checkpoint", 0),
        ("transformer_engine.pytorch.distributed", "checkpoint", 0),
    ):
        patch(target, name, function_index)


install_lora_checkpoint_context_hooks()


@dataclass(frozen=True)
class LoRAParallelSpec:
    # This only describes TP / expert-TP; DP/CP vs expert-DP is selected by `allreduce`.
    shard_domain: ShardDomain = "tp"
    sharded: bool = False
    shard_dim: int | None = None
    grad_sync_domain: GradSyncDomain = TP_DEFAULT_GRAD_SYNC_DOMAIN
    grad_sync_op: GradSyncOp = GRAD_SYNC_OP_NONE


class LoraShardManifest(TypedDict):
    domain: NotRequired[ShardDomain]
    sharded: bool
    shard_dim: NotRequired[int | None]
    shard_world_size: int
    shard_rank: int
    export_shard_dim: NotRequired[int]
    export_shard_strategy: NotRequired[str]
    component_sizes: NotRequired[Sequence[int]]


class LoraShardMeta(NamedTuple):
    key: str
    owner_rank: int
    shape: tuple[int, ...]
    dtype_name: str
    manifest: LoraShardManifest
    block: str


def _distributed_initialized() -> bool:
    is_initialized = getattr(torch.distributed, "is_initialized", None)
    return (
        torch.distributed.is_available()
        and callable(is_initialized)
        and bool(is_initialized())
    )


def _get_shard_world_size(domain: ShardDomain) -> int:
    if not _distributed_initialized():
        return 1
    if domain == "tp":
        return ps.get_tensor_model_parallel_world_size()
    group = ps.get_expert_tensor_parallel_group(check_initialized=False)
    if group is None:
        return 1
    return group.size()


def _get_shard_rank(domain: ShardDomain) -> int:
    if not _distributed_initialized():
        return 0
    if domain == "tp":
        return ps.get_tensor_model_parallel_rank()
    group = ps.get_expert_tensor_parallel_group(check_initialized=False)
    if group is None:
        return 0
    return group.rank()


def _block_for_key(key: str) -> str:
    match = _LAYER_BLOCK_RE.match(key)
    if match is not None:
        return match.group("block")
    return "__global__"


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"Invalid shard axis {axis} for tensor ndim={ndim}")
    return axis


def _shard_weight_by_components(
    weight: torch.Tensor,
    *,
    axis: int,
    component_sizes: Sequence[int],
    world_size: int,
    rank: int,
) -> torch.Tensor:
    if sum(component_sizes) != weight.shape[axis]:
        raise ValueError(
            f"Component sizes {tuple(component_sizes)} do not match axis {axis} "
            f"extent {weight.shape[axis]}"
        )
    local_components: list[torch.Tensor] = []
    for component in torch.split(weight, list(component_sizes), dim=axis):
        if component.shape[axis] % world_size != 0:
            raise ValueError(
                f"Component shape {tuple(component.shape)} is not divisible by "
                f"world size {world_size} on axis {axis}"
            )
        local_size = component.shape[axis] // world_size
        local_components.append(component.narrow(axis, rank * local_size, local_size))
    return torch.cat(local_components, dim=axis).contiguous()


def _linear_disables_tensor_parallel_comm(linear: Any) -> bool:
    return getattr(linear, "parallel_mode", "") is None or getattr(
        linear, "explicit_expert_comm", False
    )


def default_lora_rank_for_handler(handler: Any) -> int:
    return MOE_LORA_RANK if bool(getattr(handler, "is_moe", False)) else DENSE_LORA_RANK


def _configured_lora_rank(provider: Any, handler: Any) -> int:
    rank = getattr(provider, "_art_lora_rank", None)
    if rank is None:
        rank = os.environ.get(MEGATRON_LORA_RANK_ENV)
    if rank is None:
        return default_lora_rank_for_handler(handler)
    return int(rank)


def _configured_lora_target_modules(provider: Any, spec: Any) -> list[str]:
    target_modules = getattr(provider, "_art_lora_target_modules", None)
    if target_modules is None and (
        raw_target_modules := os.environ.get(MEGATRON_LORA_TARGET_MODULES_ENV)
    ):
        target_modules = json.loads(raw_target_modules)
    if target_modules is None:
        target_modules = spec.default_target_modules
    return [str(target_module) for target_module in target_modules]


def _column_parallel_lora_input(x: torch.Tensor, linear: Any) -> torch.Tensor:
    if _linear_disables_tensor_parallel_comm(linear):
        return x
    if (
        bool(getattr(linear, "sequence_parallel", False))
        and int(getattr(linear, "tp_size", 1)) > 1
    ):
        return gather_from_sequence_parallel_region(x)
    return x


def _set_lora_parallel_metadata(
    param: torch.nn.Parameter,
    *,
    parallel_spec: LoRAParallelSpec,
    allreduce: bool,
) -> None:
    replicated = not parallel_spec.sharded
    setattr(param, "lora_shard_domain", parallel_spec.shard_domain)
    setattr(param, "lora_tp_sharded", parallel_spec.sharded)
    setattr(param, "lora_tp_replicated", replicated)
    setattr(param, "lora_tp_shard_dim", parallel_spec.shard_dim)
    setattr(param, "grad_sync_domain", parallel_spec.grad_sync_domain)
    setattr(param, "grad_sync_op", parallel_spec.grad_sync_op)
    setattr(param, "allreduce", allreduce)

    setattr(
        param,
        "average_gradients_across_tp_domain",
        (
            replicated
            and parallel_spec.grad_sync_domain == TP_DEFAULT_GRAD_SYNC_DOMAIN
            and parallel_spec.grad_sync_op == GRAD_SYNC_OP_AVG
        ),
    )

    if parallel_spec.sharded:
        shard_dim = parallel_spec.shard_dim
        if shard_dim is None:
            raise ValueError("LoRAParallelSpec.shard_dim must be set when sharded=True")
        setattr(param, "tensor_model_parallel", True)
        setattr(param, "partition_dim", _normalize_axis(shard_dim, param.ndim))
        setattr(param, "partition_stride", 1)
    else:
        setattr(param, "tensor_model_parallel", False)
        setattr(param, "partition_dim", -1)
        setattr(param, "partition_stride", 1)


def _set_lora_shard_strategy_metadata(
    param: torch.nn.Parameter,
    *,
    strategy: str,
    component_sizes: Sequence[int] | None = None,
) -> None:
    setattr(param, "lora_tp_shard_strategy", strategy)
    if component_sizes is not None:
        setattr(
            param,
            "lora_tp_component_sizes",
            tuple(int(size) for size in component_sizes),
        )


def _exported_shard_dim(param: torch.nn.Parameter) -> int:
    axis = _normalize_axis(param.lora_tp_shard_dim, param.ndim)  # ty: ignore[unresolved-attribute]
    # LoRA exports always serialize a 2D tensor:
    # - non-expert params export `param.T`
    # - expert params export `param[expert].T`
    if param.ndim == 3:
        if axis == 0:
            raise ValueError("LoRA expert shard_dim cannot reference the expert axis")
        axis -= 1
    if axis not in (0, 1):
        raise ValueError(
            f"Unsupported exported LoRA shard axis {axis} for ndim={param.ndim}"
        )
    return 1 - axis


def _copy_lora_param_metadata(
    source: torch.nn.Parameter,
    target: torch.nn.Parameter,
) -> None:
    for name in (
        "lora_shard_domain",
        "lora_tp_sharded",
        "lora_tp_replicated",
        "lora_tp_shard_dim",
        "grad_sync_domain",
        "grad_sync_op",
        "allreduce",
        "average_gradients_across_tp_domain",
        "tensor_model_parallel",
        "partition_dim",
        "partition_stride",
        "lora_tp_shard_strategy",
        "lora_tp_component_sizes",
    ):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    setattr(target, "_art_dynamic_lora_slot", True)


class LoRASlot(torch.nn.Module):
    def __init__(
        self,
        *,
        ref: LoRASlotRef,
        a_t: torch.Tensor,
        b_t: torch.Tensor,
        alpha: float,
        a_template: torch.nn.Parameter,
        b_template: torch.nn.Parameter,
    ) -> None:
        super().__init__()
        self.ref = ref
        self.alpha = float(alpha)
        self.A_T = torch.nn.Parameter(a_t.detach().clone())
        self.B_T = torch.nn.Parameter(b_t.detach().clone())
        _copy_lora_param_metadata(a_template, self.A_T)
        _copy_lora_param_metadata(b_template, self.B_T)

    @property
    def rank(self) -> int:
        return int(self.A_T.shape[-1])

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


class LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
        device: torch.device,
        num_local_experts: int = 1,
        a_parallel_spec: LoRAParallelSpec = LoRAParallelSpec(),
        b_parallel_spec: LoRAParallelSpec = LoRAParallelSpec(),
        allreduce: bool = True,
    ) -> None:
        super().__init__()
        assert num_local_experts == 1 or "{expert}" in adapter_model_prefix, (
            "adapter_model_prefix must contain the '{expert}' format placeholder if num_local_experts > 1"
        )
        self.adapter_model_prefix = adapter_model_prefix
        self.alpha = float(alpha)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scale = alpha / rank
        self._slot_modules = torch.nn.ModuleDict()
        self._next_slot_key = 0
        self._slot_keys: dict[LoRASlotRef, str] = {}
        self.A_T = torch.nn.Parameter(
            torch.zeros(
                num_local_experts, in_features, rank, dtype=dtype, device=device
            ).squeeze(0)
        )
        self.B_T = torch.nn.Parameter(
            torch.zeros(
                num_local_experts, rank, out_features, dtype=dtype, device=device
            ).squeeze(0)
        )
        _set_lora_parallel_metadata(
            self.A_T,
            parallel_spec=a_parallel_spec,
            allreduce=allreduce,
        )
        _set_lora_parallel_metadata(
            self.B_T,
            parallel_spec=b_parallel_spec,
            allreduce=allreduce,
        )
        self._expert_offset = ps.get_expert_model_parallel_rank() * num_local_experts
        self.reset_lora_parameters()

    @property
    def num_local_experts(self) -> int:
        return self.A_T.shape[0] if self.A_T.ndim == 3 else 1

    def _broadcast_if_replicated(self, param: torch.nn.Parameter) -> None:
        if not param.lora_tp_replicated:  # ty: ignore[unresolved-attribute]
            return
        domain = param.lora_shard_domain  # ty: ignore[unresolved-attribute]
        world_size = _get_shard_world_size(domain)
        if world_size <= 1:
            return
        group = (
            ps.get_tensor_model_parallel_group()
            if domain == "tp"
            else ps.get_expert_tensor_parallel_group(check_initialized=False)
        )
        if group is None:
            raise RuntimeError(
                f"{self.adapter_model_prefix}: missing process group for replicated parameter domain={domain}"
            )
        src = torch.distributed.get_global_rank(  # ty: ignore[possibly-missing-attribute]
            group, 0
        )
        torch.distributed.broadcast(  # ty: ignore[possibly-missing-attribute]
            param.data,
            src=src,
            group=group,
        )

    def reset_lora_parameters(self) -> None:
        """Initialize LoRA weights (A=Kaiming, B=zeros) like PEFT defaults."""
        if self.A_T.ndim == 3:
            for expert in range(self.A_T.shape[0]):
                torch.nn.init.kaiming_uniform_(self.A_T[expert].T, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.A_T.T, a=math.sqrt(5))
        torch.nn.init.zeros_(self.B_T)
        self._broadcast_if_replicated(self.A_T)
        self._broadcast_if_replicated(self.B_T)

    def _expected_weight_keys(self, suffix: str) -> list[str]:
        if "{expert}" in self.adapter_model_prefix:
            return [
                f"{self.adapter_model_prefix.format(expert=expert + self._expert_offset)}.{suffix}.weight"
                for expert in range(self.num_local_experts)
            ]
        return [f"{self.adapter_model_prefix}.{suffix}.weight"]

    def load_lora_slot(
        self,
        ref: LoRASlotRef,
        adapter_model: dict[str, torch.Tensor],
        *,
        alpha: float = LORA_ALPHA,
        localized: bool = False,
    ) -> bool:
        if ref.name is None:
            raise ValueError("base-model slot refs do not own LoRA tensors")
        weights = self._adapter_weights(adapter_model, require=False)
        if weights is None:
            return False
        a_t = (
            weights[0].contiguous()
            if localized
            else self._localized_weight(weights[0], into=self.A_T)
        )
        b_t = (
            weights[1].contiguous()
            if localized
            else self._localized_weight(weights[1], into=self.B_T)
        )
        if (
            a_t.ndim != self.A_T.ndim
            or tuple(a_t.shape[:-1]) != tuple(self.A_T.shape[:-1])
            or b_t.ndim != self.B_T.ndim
            or tuple(b_t.shape[:-2]) != tuple(self.B_T.shape[:-2])
            or int(a_t.shape[-1]) != int(b_t.shape[-2])
            or int(b_t.shape[-1]) != int(self.B_T.shape[-1])
        ):
            raise ValueError(
                f"{self.adapter_model_prefix}: local LoRA checkpoint shape mismatch"
            )
        slot_key = self._slot_keys.get(ref)
        if slot_key is not None and self._has_live_slot_grads(ref):
            raise RuntimeError(
                f"Cannot overwrite live LoRA slot {ref.name!r} for "
                f"{self.adapter_model_prefix}; clear grads/backward graph first."
            )
        slot = LoRASlot(
            ref=ref,
            a_t=a_t,
            b_t=b_t,
            alpha=alpha,
            a_template=self.A_T,
            b_template=self.B_T,
        )
        if slot_key is None:
            while f"slot_{self._next_slot_key}" in self._slot_modules:
                self._next_slot_key += 1
            slot_key = f"slot_{self._next_slot_key}"
            self._next_slot_key += 1
        self._slot_modules[slot_key] = slot
        self._slot_keys[ref] = slot_key
        return True

    def lora_slot_params(self, ref: LoRASlotRef) -> list[torch.nn.Parameter]:
        slot = self._slot(ref)
        if slot is None:
            return []
        return [slot.A_T, slot.B_T]

    def _slot(self, ref: LoRASlotRef) -> LoRASlot | None:
        key = self._slot_keys.get(ref)
        if key is None:
            return None
        return cast(LoRASlot, self._slot_modules[key])

    def _has_live_slot_grads(self, ref: LoRASlotRef) -> bool:
        slot = self._slot(ref)
        return slot is not None and any(
            param.grad is not None for param in (slot.A_T, slot.B_T)
        )

    def load_lora(self, adapter_model: dict[str, torch.Tensor]) -> None:
        weights = self._adapter_weights(adapter_model, require=False)
        if weights is None:
            self.reset_lora_parameters()
            return
        self._load_weight(weights[0], into=self.A_T)
        self._load_weight(weights[1], into=self.B_T)

    def _adapter_weights(
        self,
        adapter_model: dict[str, torch.Tensor],
        *,
        require: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        all_keys = [
            key
            for suffix in ("lora_A", "lora_B")
            for key in self._expected_weight_keys(suffix)
        ]
        missing = [key for key in all_keys if key not in adapter_model]
        if len(missing) == len(all_keys) and not require:
            return None
        if missing:
            state = "Missing" if require else "Incomplete"
            raise KeyError(
                f"{state} LoRA adapter keys for {self.adapter_model_prefix}: "
                f"{sorted(missing)}"
            )
        return (
            self._adapter_weight(adapter_model, suffix="lora_A"),
            self._adapter_weight(adapter_model, suffix="lora_B"),
        )

    def _adapter_weight(
        self,
        adapter_model: dict[str, torch.Tensor],
        *,
        suffix: str,
    ) -> torch.Tensor:
        keys = self._expected_weight_keys(suffix)
        if self.num_local_experts > 1:
            return torch.stack([adapter_model[key].T for key in keys])
        return adapter_model[keys[0]].T

    def _localized_weight(
        self, weight: torch.Tensor, *, into: torch.nn.Parameter
    ) -> torch.Tensor:
        domain = into.lora_shard_domain  # ty: ignore[unresolved-attribute]
        if into.lora_tp_sharded:  # ty: ignore[unresolved-attribute]
            axis = into.lora_tp_shard_dim  # ty: ignore[unresolved-attribute]
            axis = _normalize_axis(axis, weight.ndim)
            world_size = _get_shard_world_size(domain)
            rank = _get_shard_rank(domain)
            strategy = getattr(into, "lora_tp_shard_strategy", "uniform")
            if strategy == "componentwise":
                component_sizes = tuple(
                    int(size) for size in getattr(into, "lora_tp_component_sizes", ())
                )
                if not component_sizes:
                    raise ValueError(
                        f"{self.adapter_model_prefix}: missing component sizes for shard strategy={strategy}"
                    )
                weight = _shard_weight_by_components(
                    weight,
                    axis=axis,
                    component_sizes=component_sizes,
                    world_size=world_size,
                    rank=rank,
                )
            elif strategy == "uniform":
                if weight.shape[axis] % world_size != 0:
                    raise ValueError(
                        f"{self.adapter_model_prefix}: weight shape {tuple(weight.shape)} is not divisible by world size "
                        f"{world_size} on axis {axis}"
                    )
                local_size = weight.shape[axis] // world_size
                if into.shape[axis] != local_size:
                    raise ValueError(
                        f"{self.adapter_model_prefix}: expected local shard size {into.shape[axis]}, got {local_size}"
                    )
                weight = weight.narrow(axis, rank * local_size, local_size)
            else:
                raise ValueError(
                    f"{self.adapter_model_prefix}: unsupported shard strategy={strategy}"
                )
        return weight.contiguous()

    def _load_weight(self, weight: torch.Tensor, *, into: torch.nn.Parameter) -> None:
        weight = self._localized_weight(weight, into=into)
        if tuple(weight.shape) != tuple(into.shape):
            raise ValueError(
                f"{self.adapter_model_prefix}: sharded load shape mismatch, got {tuple(weight.shape)} "
                f"expected {tuple(into.shape)}"
            )
        into.data.copy_(weight)
        into.requires_grad = True

    def _should_export_parameter(self, param: torch.nn.Parameter) -> bool:
        """
        Determine if the given LoRA param should be exported in the sharded LoRA state dict
        (drop replicated ranks/params).
        """
        if "{expert}" in self.adapter_model_prefix:
            if ps.get_expert_data_parallel_rank() != 0:
                return False
        else:  # self is a non-MoE layer
            # dp x cp rank 0 participates
            if ps.get_data_parallel_rank(with_context_parallel=True) != 0:
                return False

        # this param is fully sharded, all shard ranks participate
        if param.lora_tp_sharded:  # ty: ignore[unresolved-attribute]
            return True
        # param is replicated, tp rank 0 or etp rank 0 participates
        return _get_shard_rank(param.lora_shard_domain) == 0  # ty: ignore[unresolved-attribute]

    def _manifest_for_param(self, param: torch.nn.Parameter) -> LoraShardManifest:
        manifest: LoraShardManifest = {
            "domain": param.lora_shard_domain,  # ty: ignore[unresolved-attribute]
            "sharded": param.lora_tp_sharded,  # ty: ignore[unresolved-attribute]
            "shard_dim": param.lora_tp_shard_dim,  # ty: ignore[unresolved-attribute]
            "shard_world_size": _get_shard_world_size(param.lora_shard_domain)  # ty: ignore[unresolved-attribute]
            if param.lora_tp_sharded  # ty: ignore[unresolved-attribute]
            else 1,
            "shard_rank": _get_shard_rank(param.lora_shard_domain)  # ty: ignore[unresolved-attribute]
            if param.lora_tp_sharded  # ty: ignore[unresolved-attribute]
            else 0,
        }
        if param.lora_tp_sharded:  # ty: ignore[unresolved-attribute]
            manifest["export_shard_dim"] = _exported_shard_dim(param)
            manifest["export_shard_strategy"] = getattr(
                param,
                "lora_tp_shard_strategy",
                "uniform",
            )
            component_sizes = list(getattr(param, "lora_tp_component_sizes", ()))
            if component_sizes:
                manifest["component_sizes"] = component_sizes
        return manifest

    def _lora_params(
        self, ref: LoRASlotRef | None = None
    ) -> list[tuple[str, torch.nn.Parameter]]:
        if ref is not None:
            slot = self._slot(ref)
            if slot is None:
                return []
            return [
                ("lora_A.weight", slot.A_T),
                ("lora_B.weight", slot.B_T),
            ]
        return [
            ("lora_A.weight", self.A_T),
            ("lora_B.weight", self.B_T),
        ]

    def _export_items(
        self,
        ref: LoRASlotRef | None = None,
        *,
        include_replicas: bool = False,
    ) -> list[tuple[str, torch.nn.Parameter, int | None]]:
        export_items: list[tuple[str, torch.nn.Parameter, int | None]] = []
        for key, param in self._lora_params(ref):
            if not include_replicas and not self._should_export_parameter(param):
                continue
            if "{expert}" in self.adapter_model_prefix:
                for expert in range(self.num_local_experts):
                    full_key = f"{self.adapter_model_prefix.format(expert=expert + self._expert_offset)}.{key}"
                    export_items.append(
                        (full_key, param, expert if param.ndim == 3 else None)
                    )
            else:
                export_items.append((f"{self.adapter_model_prefix}.{key}", param, None))
        return export_items

    def sharded_lora_manifest(
        self,
        ref: LoRASlotRef | None = None,
        *,
        include_replicas: bool = False,
    ) -> dict[str, LoraShardManifest]:
        return {
            key: self._manifest_for_param(param)
            for key, param, _expert in self._export_items(
                ref, include_replicas=include_replicas
            )
        }

    def sharded_lora_state_dict(
        self,
        ref: LoRASlotRef | None = None,
        *,
        include_replicas: bool = False,
    ) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for key, param, expert in self._export_items(
            ref, include_replicas=include_replicas
        ):
            state[key] = param.data[expert].T if expert is not None else param.data.T
        return state

    def sharded_lora_grad_dict(self) -> dict[str, torch.Tensor]:
        grads: dict[str, torch.Tensor] = {}
        for key, param, expert in self._export_items():
            if not hasattr(param, "main_grad"):
                raise RuntimeError(
                    f"LoRA param missing main_grad attribute for key '{key}'"
                )
            grad = cast(torch.Tensor, param.main_grad)
            if grad is None:
                raise RuntimeError(f"LoRA param main_grad is None for key '{key}'")
            if hasattr(grad, "_local_tensor"):
                grad = cast(Any, grad)._local_tensor
            local_grad = grad[expert] if expert is not None else grad
            grads[key] = local_grad.T
        return grads

    def active_lora_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
        ref = _CURRENT_LORA_SLOT.get()
        if ref is None:
            return self.A_T, self.B_T, self.scale
        if ref.name is None:
            return None
        slot = self._slot(ref)
        if slot is None:
            return None
        return slot.A_T, slot.B_T, slot.scale

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        active = self.active_lora_tensors()
        if active is None:
            return x.new_zeros((*x.shape[:-1], self.out_features))
        a_t, b_t, scale = active
        if tokens_per_expert is not None:
            assert self.num_local_experts > 1, (
                "tokens_per_expert is only supported if num_local_experts > 1"
            )
            bsz = tokens_per_expert
            if isinstance(bsz, list):
                bsz = torch.tensor(bsz, dtype=torch.int64, device="cpu")
            if x.shape[0] == 0:
                return x.new_zeros((*x.shape[:-1], self.out_features))
            return quack_grouped_lora(x, a_t, b_t, bsz, scale=scale)
        out = (x @ a_t) @ b_t
        return out if scale == 1.0 else out * scale


@torch.compiler.disable
def _expert_grouped_lora_forward(
    lora: LoRA,
    x: torch.Tensor,
    tokens_per_expert: list[int] | torch.Tensor,
    out_features: int,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return x.new_zeros((x.shape[0], out_features))
    return lora(x, tokens_per_expert=tokens_per_expert)


def _out_features(module: object) -> int:
    out_features = getattr(module, "out_features", None)
    if not isinstance(out_features, int):
        raise TypeError(f"{type(module).__name__} has no integer out_features")
    return out_features


@torch.compiler.disable
def _expert_grouped_lora_dual_forward(
    module: "MLPExpertsLinearFC1LoRA",
    x: torch.Tensor,
    tokens_per_expert: list[int] | torch.Tensor,
) -> torch.Tensor:
    counts = tokens_per_expert
    if isinstance(counts, list):
        counts = torch.tensor(counts, dtype=torch.int64, device="cpu")
    if x.shape[0] == 0:
        return x.new_zeros((x.shape[0], module.out_features))
    gate = module.gate_lora.active_lora_tensors()
    up = module.up_lora.active_lora_tensors()
    if gate is None or up is None:
        return torch.cat(
            [
                module.gate_lora(x, tokens_per_expert=counts),
                module.up_lora(x, tokens_per_expert=counts),
            ],
            dim=-1,
        )
    gate_a_t, gate_b_t, gate_scale = gate
    up_a_t, up_b_t, up_scale = up
    return quack_grouped_lora_dual(
        x,
        gate_a_t,
        gate_b_t,
        up_a_t,
        up_b_t,
        counts,
        scale_gate=gate_scale,
        scale_up=up_scale,
    )


def _parallel_lora(
    *,
    adapter_model_prefix: str,
    linear: Any,
    out_features: int,
    rank: int,
    alpha: float,
    layout: Literal["column", "row"],
    shard_domain: ShardDomain = "tp",
    grad_sync_domain: GradSyncDomain = TP_DEFAULT_GRAD_SYNC_DOMAIN,
    allreduce: bool = True,
    num_local_experts: int = 1,
) -> LoRA:
    weight = getattr(linear, "weight0", None)
    if weight is None:
        weight = getattr(linear, "weight", None)
    assert isinstance(weight, torch.Tensor)
    row_layout = layout == "row"
    a_parallel_spec = LoRAParallelSpec(
        shard_domain=shard_domain,
        sharded=row_layout,
        shard_dim=-2 if row_layout else None,
        grad_sync_domain=grad_sync_domain,
        grad_sync_op=GRAD_SYNC_OP_NONE if row_layout else GRAD_SYNC_OP_SUM,
    )
    b_parallel_spec = replace(
        a_parallel_spec,
        sharded=not row_layout,
        shard_dim=None if row_layout else -1,
        grad_sync_domain=grad_sync_domain,
        grad_sync_op=GRAD_SYNC_OP_SUM if row_layout else GRAD_SYNC_OP_NONE,
    )
    return LoRA(
        adapter_model_prefix=adapter_model_prefix,
        in_features=linear.in_features,
        out_features=out_features,
        rank=rank,
        alpha=alpha,
        dtype=weight.dtype,
        device=weight.device,
        num_local_experts=num_local_experts,
        a_parallel_spec=a_parallel_spec,
        b_parallel_spec=b_parallel_spec,
        allreduce=allreduce,
    )


def _parallel_lora_pair(
    *,
    adapter_model_prefix: str,
    linear: Any,
    out_features: int,
    rank: int,
    alpha: float,
    layout: Literal["column", "row"],
    suffixes: tuple[str, str],
    num_local_experts: int = 1,
) -> tuple[LoRA, LoRA]:
    expert_parallel = num_local_experts > 1
    return cast(
        tuple[LoRA, LoRA],
        tuple(
            _parallel_lora(
                adapter_model_prefix=f"{adapter_model_prefix}.{suffix}",
                linear=linear,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                layout=layout,
                shard_domain="expert_tp" if expert_parallel else "tp",
                grad_sync_domain=(
                    EXPERT_TP_GRAD_SYNC_DOMAIN
                    if expert_parallel
                    else TP_DEFAULT_GRAD_SYNC_DOMAIN
                ),
                allreduce=not expert_parallel,
                num_local_experts=num_local_experts,
            )
            for suffix in suffixes
        ),
    )


class SelfAttentionLinearProjLoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_proj: TERowParallelLinear,
        rank: int,
        alpha: float,
        provider: GPTModelProvider,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.linear_proj = linear_proj
        self.reduce_output = reduce_output
        self.lora = _parallel_lora(
            adapter_model_prefix=adapter_model_prefix,
            linear=linear_proj,
            out_features=linear_proj.out_features,
            rank=rank,
            alpha=alpha,
            layout="row",
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_output, bias_output = self.linear_proj(x)
        assert isinstance(base_output, torch.Tensor)
        assert isinstance(bias_output, (torch.Tensor, type(None)))

        lora_output = self.lora(x)
        if self.reduce_output and self.provider.tensor_model_parallel_size > 1:
            if self.provider.sequence_parallel:
                lora_output = reduce_scatter_to_sequence_parallel_region(lora_output)
            else:
                lora_output = reduce_from_tensor_model_parallel_region(lora_output)
        return base_output + lora_output, bias_output


class SelfAttentionLinearQKVLoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_qkv: TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
        provider: GPTModelProvider,
        target_modules: set[str],
    ) -> None:
        super().__init__()
        self.provider = provider
        linear_qkv.return_layernorm_output = True
        linear_qkv.return_layernorm_output_gathered = True
        self.linear_qkv = linear_qkv
        assert self.provider.kv_channels is not None
        assert self.provider.num_query_groups is not None
        assert self.provider.num_attention_heads is not None
        if self.provider.num_attention_heads % self.provider.num_query_groups != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_query_groups for QKV LoRA"
            )
        weight = linear_qkv.weight
        assert isinstance(weight, torch.Tensor)
        total_out_features_per_rank = int(weight.shape[0])
        kv_out_features = self.provider.kv_channels * self.provider.num_query_groups
        tp_world_size = ps.get_tensor_model_parallel_world_size()
        assert kv_out_features % tp_world_size == 0, (
            "kv_out_features must be divisible by tensor parallel size"
        )
        q_out_features = self.provider.kv_channels * self.provider.num_attention_heads
        assert q_out_features % tp_world_size == 0, (
            "q_out_features must be divisible by tensor parallel size"
        )
        q_out_features_per_rank = q_out_features // tp_world_size
        kv_out_features_per_rank = kv_out_features // tp_world_size
        self.attention_output_gate = bool(
            getattr(self.provider, "attention_output_gate", False)
        )
        q_and_gate_out_features_per_rank = total_out_features_per_rank - (
            2 * kv_out_features_per_rank
        )
        expected_q_out_features_per_rank = q_out_features_per_rank * (
            2 if self.attention_output_gate else 1
        )
        assert q_and_gate_out_features_per_rank == expected_q_out_features_per_rank, (
            "Unexpected per-rank QKV packing for this attention layout"
        )
        self.q_and_gate_out_features_per_rank = q_and_gate_out_features_per_rank
        self.kv_out_features_per_rank = kv_out_features_per_rank
        self.num_query_groups_per_partition = (
            self.provider.num_query_groups // tp_world_size
        )
        self.num_attention_heads_per_group = (
            self.provider.num_attention_heads // self.provider.num_query_groups
        )
        self.hidden_size_per_attention_head = self.provider.kv_channels
        self.q_proj_lora = (
            self._build_qkv_lora(
                adapter_model_prefix=f"{adapter_model_prefix}.q_proj",
                linear_qkv=linear_qkv,
                rank=rank,
                alpha=alpha,
                out_features=q_and_gate_out_features_per_rank,
            )
            if _targets_include(target_modules, "q_proj")
            else None
        )
        self.k_proj_lora = (
            self._build_qkv_lora(
                adapter_model_prefix=f"{adapter_model_prefix}.k_proj",
                linear_qkv=linear_qkv,
                rank=rank,
                alpha=alpha,
                out_features=kv_out_features_per_rank,
            )
            if _targets_include(target_modules, "k_proj")
            else None
        )
        self.v_proj_lora = (
            self._build_qkv_lora(
                adapter_model_prefix=f"{adapter_model_prefix}.v_proj",
                linear_qkv=linear_qkv,
                rank=rank,
                alpha=alpha,
                out_features=kv_out_features_per_rank,
            )
            if _targets_include(target_modules, "v_proj")
            else None
        )

    @staticmethod
    def _build_qkv_lora(
        *,
        adapter_model_prefix: str,
        linear_qkv: TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
        out_features: int,
    ) -> LoRA:
        assert isinstance(linear_qkv.weight, torch.Tensor)
        a_parallel_spec = LoRAParallelSpec(
            shard_domain="tp",
            sharded=False,
            shard_dim=None,
            grad_sync_domain=TP_DEFAULT_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_SUM,  # sum replicated TP contributions
        )
        b_parallel_spec = replace(
            a_parallel_spec,
            sharded=True,
            shard_dim=-1,
            grad_sync_op=GRAD_SYNC_OP_NONE,  # only need DP-type reductions
        )
        return LoRA(
            adapter_model_prefix=adapter_model_prefix,
            in_features=linear_qkv.in_features,
            out_features=out_features,
            rank=rank,
            alpha=alpha,
            dtype=linear_qkv.weight.dtype,
            device=linear_qkv.weight.device,
            a_parallel_spec=a_parallel_spec,
            b_parallel_spec=b_parallel_spec,
            # Non-expert LoRA params use Megatron's dense DP/CP gradient buckets.
            allreduce=True,
        )

    def _qkv_lora_output(
        self,
        lora: LoRA | None,
        layernorm_output: torch.Tensor,
        out_features: int,
    ) -> torch.Tensor:
        if lora is not None:
            return lora(layernorm_output)
        return layernorm_output.new_zeros(
            (*layernorm_output.shape[:-1], out_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        (
            linear_output_and_layernorm_output,
            bias,
        ) = self.linear_qkv(x)
        linear_output, layernorm_output = linear_output_and_layernorm_output
        assert isinstance(linear_output, torch.Tensor)
        assert isinstance(layernorm_output, torch.Tensor)
        assert isinstance(bias, (torch.Tensor, type(None)))

        query_and_gate = self._qkv_lora_output(
            self.q_proj_lora,
            layernorm_output,
            self.q_and_gate_out_features_per_rank,
        )
        key = self._qkv_lora_output(
            self.k_proj_lora,
            layernorm_output,
            self.kv_out_features_per_rank,
        )
        value = self._qkv_lora_output(
            self.v_proj_lora,
            layernorm_output,
            self.kv_out_features_per_rank,
        )
        query_and_gate_5d = query_and_gate.reshape(
            query_and_gate.shape[0],
            query_and_gate.shape[1],
            self.num_query_groups_per_partition,
            self.num_attention_heads_per_group
            * (2 if self.attention_output_gate else 1),
            self.hidden_size_per_attention_head,
        )
        key_5d = key.reshape(
            key.shape[0],
            key.shape[1],
            self.num_query_groups_per_partition,
            1,
            self.hidden_size_per_attention_head,
        )
        value_5d = value.reshape(
            value.shape[0],
            value.shape[1],
            self.num_query_groups_per_partition,
            1,
            self.hidden_size_per_attention_head,
        )
        qkv_5d = torch.cat([query_and_gate_5d, key_5d, value_5d], dim=3)
        adapter_output = qkv_5d.reshape(qkv_5d.shape[0], qkv_5d.shape[1], -1)

        return linear_output + adapter_output, bias


class GatedDeltaNetInProjLoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        in_proj: TELayerNormColumnParallelLinear,
        gated_delta_net: GatedDeltaNet,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        in_proj.return_layernorm_output = True
        in_proj.return_layernorm_output_gathered = True
        self.in_proj = in_proj
        self.num_value_heads_per_partition = (
            gated_delta_net.num_value_heads // ps.get_tensor_model_parallel_world_size()
        )
        qkv_out_features_per_partition = (
            gated_delta_net.qk_dim * 2 + gated_delta_net.v_dim
        ) // ps.get_tensor_model_parallel_world_size()
        z_out_features_per_partition = (
            gated_delta_net.v_dim // ps.get_tensor_model_parallel_world_size()
        )
        self.qkv_lora = _parallel_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.in_proj_qkv",
            linear=in_proj,
            out_features=qkv_out_features_per_partition,
            rank=rank,
            alpha=alpha,
            layout="column",
        )
        _set_lora_shard_strategy_metadata(
            self.qkv_lora.B_T,
            strategy="componentwise",
            component_sizes=(
                gated_delta_net.qk_dim,
                gated_delta_net.qk_dim,
                gated_delta_net.v_dim,
            ),
        )
        self.z_lora = _parallel_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.in_proj_z",
            linear=in_proj,
            out_features=z_out_features_per_partition,
            rank=rank,
            alpha=alpha,
            layout="column",
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        linear_output_and_layernorm_output, bias = self.in_proj(x)
        linear_output, layernorm_output = linear_output_and_layernorm_output
        assert isinstance(linear_output, torch.Tensor)
        assert isinstance(layernorm_output, torch.Tensor)
        assert isinstance(bias, (torch.Tensor, type(None)))

        qkv = self.qkv_lora(layernorm_output)
        z = self.z_lora(layernorm_output)
        beta = qkv.new_zeros(
            qkv.shape[0],
            qkv.shape[1],
            self.num_value_heads_per_partition,
        )
        alpha = beta.clone()
        adapter_output = torch.cat([qkv, z, beta, alpha], dim=-1)
        return linear_output + adapter_output, bias


class MLPExpertsLinearFC1LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc1: TEColumnParallelGroupedLinear,
        rank: int,
        alpha: float,
        num_local_experts: int,
        fused_gate_up: bool = False,
    ) -> None:
        super().__init__()
        self.linear_fc1 = linear_fc1
        self.out_features = _out_features(linear_fc1)
        self.fused_gate_up = bool(fused_gate_up)
        if self.fused_gate_up:
            self.lora = _parallel_lora(
                adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.gate_up_proj",
                linear=linear_fc1,
                out_features=self.out_features,
                rank=rank,
                alpha=alpha,
                layout="column",
                shard_domain="expert_tp",
                grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
                allreduce=False,
                num_local_experts=num_local_experts,
            )
            gate_out_features = self.out_features // 2
            expert_tp_world_size = _get_shard_world_size("expert_tp")
            _set_lora_shard_strategy_metadata(
                self.lora.B_T,
                strategy="componentwise",
                component_sizes=(
                    gate_out_features * expert_tp_world_size,
                    gate_out_features * expert_tp_world_size,
                ),
            )
        else:
            self.gate_lora, self.up_lora = _parallel_lora_pair(
                adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}",
                linear=linear_fc1,
                out_features=self.out_features // 2,
                rank=rank,
                alpha=alpha,
                layout="column",
                suffixes=("gate_proj", "up_proj"),
                num_local_experts=num_local_experts,
            )

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_out, bias_out = cast(
            Callable[
                [torch.Tensor, list[int] | torch.Tensor],
                tuple[torch.Tensor, torch.Tensor | None],
            ],
            self.linear_fc1,
        )(x, tokens_per_expert)
        adapter_out = (
            _expert_grouped_lora_forward(
                self.lora,
                x,
                tokens_per_expert,
                self.out_features,
            )
            if self.fused_gate_up
            else _expert_grouped_lora_dual_forward(self, x, tokens_per_expert)
        )
        return base_out + adapter_out, bias_out


class MLPExpertsLinearFC2LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc2: TERowParallelGroupedLinear,
        rank: int,
        alpha: float,
        num_local_experts: int,
    ) -> None:
        super().__init__()
        self.linear_fc2 = linear_fc2
        self.out_features = _out_features(linear_fc2)
        self.lora = _parallel_lora(
            adapter_model_prefix=f"{adapter_model_prefix}.{{expert}}.down_proj",
            linear=linear_fc2,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            layout="row",
            shard_domain="expert_tp",
            grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
            allreduce=False,
            num_local_experts=num_local_experts,
        )

    def forward(
        self, x: torch.Tensor, tokens_per_expert: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        base_out, bias_out = cast(
            Callable[
                [torch.Tensor, list[int] | torch.Tensor],
                tuple[torch.Tensor, torch.Tensor | None],
            ],
            self.linear_fc2,
        )(x, tokens_per_expert)
        adapter_out = _expert_grouped_lora_forward(
            self.lora,
            x,
            tokens_per_expert,
            self.out_features,
        )
        # the reason there is no TP comm here is because the MoE token routing handles
        # expert TP comm externally
        return base_out + adapter_out, bias_out


class SharedExpertsLinearFC1LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc1: TEColumnParallelLinear | TELayerNormColumnParallelLinear,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if isinstance(linear_fc1, TELayerNormColumnParallelLinear):
            linear_fc1.return_layernorm_output = True
            linear_fc1.return_layernorm_output_gathered = True
        self.linear_fc1 = linear_fc1
        self.gate_lora, self.up_lora = _parallel_lora_pair(
            adapter_model_prefix=adapter_model_prefix,
            linear=linear_fc1,
            out_features=linear_fc1.out_features // 2,
            rank=rank,
            alpha=alpha,
            layout="column",
            suffixes=("gate_proj", "up_proj"),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if int(x.numel()) == 0:
            zero = x.sum() * 0.0
            weight = getattr(self.linear_fc1, "weight", None)
            if isinstance(weight, torch.Tensor):
                zero = zero + weight.to(dtype=x.dtype).sum() * 0.0
            for lora in (self.gate_lora, self.up_lora):
                zero = zero + lora.A_T.to(dtype=x.dtype).sum() * 0.0
                zero = zero + lora.B_T.to(dtype=x.dtype).sum() * 0.0
            return zero.expand(
                *x.shape[:-1], self.linear_fc1.out_features
            ).clone(), None
        base_output, bias_out = self.linear_fc1(x)
        if isinstance(base_output, tuple):
            base_out, lora_input = base_output
        else:
            base_out = base_output
            lora_input = _column_parallel_lora_input(x, self.linear_fc1)
        adapter_out = torch.cat(
            [self.gate_lora(lora_input), self.up_lora(lora_input)],
            dim=-1,
        )
        if adapter_out.shape != base_out.shape:
            adapter_model_prefix = self.gate_lora.adapter_model_prefix.rsplit(".", 1)[0]
            raise RuntimeError(
                f"{adapter_model_prefix}: LoRA adapter output shape "
                f"{tuple(adapter_out.shape)} does not match base output shape "
                f"{tuple(base_out.shape)}"
            )
        return base_out + adapter_out, bias_out


class SharedExpertsLinearFC2LoRA(torch.nn.Module):
    def __init__(
        self,
        adapter_model_prefix: str,
        linear_fc2: TERowParallelLinear,
        rank: int,
        alpha: float,
        provider: GPTModelProvider,
    ) -> None:
        super().__init__()
        self.row_parallel_lora = SelfAttentionLinearProjLoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.down_proj",
            linear_proj=linear_fc2,
            rank=rank,
            alpha=alpha,
            provider=provider,
            reduce_output=not _linear_disables_tensor_parallel_comm(linear_fc2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.row_parallel_lora(x)


def _unwrap_attr(
    value: Any,
    attr_name: str,
    expected_type: type[Any] | tuple[type[Any], ...],
) -> Any:
    if isinstance(value, expected_type):
        return value
    unwrapped = getattr(value, attr_name)
    assert isinstance(unwrapped, expected_type)
    return unwrapped


def _adapter_model_prefix(module: TransformerLayer) -> str:
    return f"base_model.model.model.layers.{module.layer_number - 1}"


def _is_language_transformer_layer_name(module_name: str) -> bool:
    while module_name.startswith("module."):
        module_name = module_name.removeprefix("module.")
    return module_name.startswith(("decoder.layers.", "language_model.decoder.layers."))


def _targets_include(target_modules: set[str], *names: str) -> bool:
    return not target_modules or any(name in target_modules for name in names)


def wrap_standard_self_attention(
    self_attention: SelfAttention,
    *,
    adapter_model_prefix: str,
    provider: GPTModelProvider,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    if _targets_include(target_modules, "o_proj"):
        self_attention_linear_proj = _unwrap_attr(
            self_attention.linear_proj,
            "linear_proj",
            TERowParallelLinear,
        )
        self_attention.linear_proj = SelfAttentionLinearProjLoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.self_attn.o_proj",
            linear_proj=self_attention_linear_proj,
            rank=rank,
            alpha=alpha,
            provider=provider,
        )
    if _targets_include(target_modules, "q_proj", "k_proj", "v_proj"):
        self_attention_linear_qkv = _unwrap_attr(
            self_attention.linear_qkv,
            "linear_qkv",
            TELayerNormColumnParallelLinear,
        )
        linear_qkv_lora = SelfAttentionLinearQKVLoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.self_attn",
            linear_qkv=self_attention_linear_qkv,
            rank=rank,
            alpha=alpha,
            provider=provider,
            target_modules=target_modules,
        )
        setattr(self_attention, "linear_qkv", linear_qkv_lora)


def wrap_gated_delta_net_attention(
    self_attention: GatedDeltaNet,
    *,
    adapter_model_prefix: str,
    provider: GPTModelProvider,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    if _targets_include(target_modules, "out_proj"):
        gated_delta_net_out_proj = _unwrap_attr(
            self_attention.out_proj,
            "out_proj",
            TERowParallelLinear,
        )
        self_attention.out_proj = SelfAttentionLinearProjLoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.linear_attn.out_proj",
            linear_proj=gated_delta_net_out_proj,
            rank=rank,
            alpha=alpha,
            provider=provider,
        )
    if _targets_include(target_modules, "in_proj_qkv", "in_proj_z"):
        gated_delta_net_in_proj = _unwrap_attr(
            self_attention.in_proj,
            "in_proj",
            TELayerNormColumnParallelLinear,
        )
        self_attention.in_proj = GatedDeltaNetInProjLoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.linear_attn",
            in_proj=gated_delta_net_in_proj,
            gated_delta_net=self_attention,
            rank=rank,
            alpha=alpha,
        )


def wrap_grouped_moe_experts(
    experts: TEGroupedMLP,
    *,
    adapter_model_prefix: str,
    target_modules: set[str],
    rank: int,
    alpha: int,
    fused_gate_up: bool = False,
) -> None:
    wrap_fc1 = (
        _targets_include(target_modules, "experts")
        if fused_gate_up
        else _targets_include(target_modules, "gate_proj", "up_proj")
    )
    if wrap_fc1:
        mlp_experts_linear_fc1 = _unwrap_attr(
            experts.linear_fc1,
            "linear_fc1",
            TEColumnParallelGroupedLinear,  # type: ignore
        )
        linear_fc1_lora = MLPExpertsLinearFC1LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
            linear_fc1=mlp_experts_linear_fc1,
            rank=rank,
            alpha=alpha,
            num_local_experts=experts.num_local_experts,
            fused_gate_up=fused_gate_up,
        )
        setattr(experts, "linear_fc1", linear_fc1_lora)
    wrap_fc2 = (
        wrap_fc1 if fused_gate_up else _targets_include(target_modules, "down_proj")
    )
    if wrap_fc2:
        linear_fc2 = _unwrap_attr(
            experts.linear_fc2,
            "linear_fc2",
            TERowParallelGroupedLinear,  # type: ignore
        )
        linear_fc2_lora = MLPExpertsLinearFC2LoRA(
            adapter_model_prefix=f"{adapter_model_prefix}.mlp.experts",
            linear_fc2=linear_fc2,
            rank=rank,
            alpha=alpha,
            num_local_experts=experts.num_local_experts,
        )
        setattr(experts, "linear_fc2", linear_fc2_lora)


def wrap_split_mlp_lora(
    mlp: Any,
    *,
    adapter_model_prefix: str,
    provider: GPTModelProvider,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    if _targets_include(target_modules, "gate_proj", "up_proj"):
        linear_fc1 = _unwrap_attr(
            mlp.linear_fc1,
            "linear_fc1",
            (TEColumnParallelLinear, TELayerNormColumnParallelLinear),
        )
        mlp.linear_fc1 = SharedExpertsLinearFC1LoRA(
            adapter_model_prefix=adapter_model_prefix,
            linear_fc1=linear_fc1,
            rank=rank,
            alpha=alpha,
        )
    if _targets_include(target_modules, "down_proj"):
        linear_fc2 = _unwrap_attr(
            mlp.linear_fc2,
            "linear_fc2",
            TERowParallelLinear,
        )
        mlp.linear_fc2 = SharedExpertsLinearFC2LoRA(
            adapter_model_prefix=adapter_model_prefix,
            linear_fc2=linear_fc2,
            rank=rank,
            alpha=alpha,
            provider=provider,
        )


def wrap_grouped_moe_experts_3d(
    experts: TEGroupedMLP,
    *,
    adapter_model_prefix: str,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    wrap_grouped_moe_experts(
        experts,
        adapter_model_prefix=adapter_model_prefix,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        fused_gate_up=True,
    )


def wrap_dense_mlp(
    mlp: Any,
    *,
    adapter_model_prefix: str,
    provider: GPTModelProvider,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    wrap_split_mlp_lora(
        mlp,
        adapter_model_prefix=f"{adapter_model_prefix}.mlp",
        provider=provider,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
    )


def wrap_shared_experts_mlp(
    shared_experts: Any,
    *,
    adapter_model_prefix: str,
    provider: GPTModelProvider,
    target_modules: set[str],
    rank: int,
    alpha: int,
) -> None:
    wrap_split_mlp_lora(
        shared_experts,
        adapter_model_prefix=f"{adapter_model_prefix}.mlp.shared_experts",
        provider=provider,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
    )


def apply_lora_adapters(
    model: Sequence[torch.nn.Module],
    provider: GPTModelProvider,
) -> list[torch.nn.Module]:
    provider = cast(Any, provider)
    handler = provider._art_model_support_handler
    spec = provider._art_model_support_spec
    target_modules = _configured_lora_target_modules(provider, spec)
    rank = _configured_lora_rank(provider, handler)
    handler.apply_lora_adapters(
        model,
        provider,
        target_modules=target_modules,
        rank=rank,
        alpha=LORA_ALPHA,
    )
    return list(model)


def load_lora_slot_into_model(
    model: Sequence[torch.nn.Module],
    ref: LoRASlotRef,
    adapter_model: dict[str, torch.Tensor],
    *,
    alpha: float = LORA_ALPHA,
    localized: bool = False,
) -> int:
    loaded = 0
    for chunk in model:
        for module in chunk.modules():
            if isinstance(module, LoRA) and module.load_lora_slot(
                ref,
                adapter_model,
                alpha=alpha,
                localized=localized,
            ):
                loaded += 1
    if loaded == 0 and ref.name is not None:
        raise RuntimeError(f"LoRA slot {ref.name!r} loaded no adapter sites")
    return loaded


def delete_lora_slot_from_model(
    model: Sequence[torch.nn.Module], ref: LoRASlotRef
) -> None:
    for chunk in model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            key = module._slot_keys.pop(ref, None)
            if key is not None and key in module._slot_modules:
                del module._slot_modules[key]


def _lora_slot_replacements(
    model: Sequence[torch.nn.Module],
    source: LoRASlotRef,
    destination: LoRASlotRef,
) -> list[tuple[LoRA, str | None, str | None]]:
    replacements: list[tuple[LoRA, str | None, str | None]] = []
    for chunk in model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            source_key = module._slot_keys.get(source)
            destination_key = module._slot_keys.get(destination)
            for ref, key in ((source, source_key), (destination, destination_key)):
                if key is None:
                    continue
                if key not in module._slot_modules:
                    raise RuntimeError(
                        f"LoRA slot {ref.name!r} maps to missing module {key!r}"
                    )
                slot = cast(LoRASlot, module._slot_modules[key])
                if slot.ref != ref:
                    raise RuntimeError(
                        f"LoRA slot {ref.name!r} has inconsistent module metadata"
                    )
            replacements.append((module, source_key, destination_key))
    return replacements


type _LoraSlotSnapshot = tuple[
    tuple[
        LoRA,
        dict[LoRASlotRef, str],
        dict[str, LoRASlot],
        dict[str, LoRASlotRef],
    ],
    ...,
]


def _snapshot_lora_slots(
    model: Sequence[torch.nn.Module],
) -> _LoraSlotSnapshot:
    snapshots = []
    seen: set[int] = set()
    for chunk in model:
        for module in chunk.modules():
            if not isinstance(module, LoRA) or id(module) in seen:
                continue
            seen.add(id(module))
            slots = {
                key: cast(LoRASlot, slot) for key, slot in module._slot_modules.items()
            }
            snapshots.append(
                (
                    module,
                    dict(module._slot_keys),
                    slots,
                    {key: slot.ref for key, slot in slots.items()},
                )
            )
    return tuple(snapshots)


def _restore_lora_slots(snapshot: _LoraSlotSnapshot) -> None:
    for module, keys, slots, refs in snapshot:
        module._slot_keys.clear()
        module._slot_keys.update(keys)
        module._slot_modules = torch.nn.ModuleDict(slots)
        for key, ref in refs.items():
            slots[key].ref = ref


def validate_lora_slot_replacement(
    model: Sequence[torch.nn.Module],
    source: LoRASlotRef,
    destination: LoRASlotRef,
) -> None:
    _lora_slot_replacements(model, source, destination)


def replace_lora_slot_in_model(
    model: Sequence[torch.nn.Module],
    source: LoRASlotRef,
    destination: LoRASlotRef,
) -> None:
    replacements = _lora_slot_replacements(model, source, destination)
    for module, source_key, destination_key in replacements:
        if source_key is None:
            if destination_key is not None:
                module._slot_keys.pop(destination)
                del module._slot_modules[destination_key]
            continue
        slot = cast(LoRASlot, module._slot_modules[source_key])
        slot.ref = destination
        module._slot_keys.pop(source)
        module._slot_keys[destination] = source_key
        if destination_key is not None and destination_key != source_key:
            del module._slot_modules[destination_key]


def iter_lora_slot_parameters(
    model: Sequence[torch.nn.Module],
    ref: LoRASlotRef,
) -> Iterator[torch.nn.Parameter]:
    seen: set[int] = set()
    for chunk in model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            for param in module.lora_slot_params(ref):
                param_id = id(param)
                if param_id in seen:
                    continue
                seen.add(param_id)
                yield param


def iter_lora_sites(
    model: Sequence[torch.nn.Module],
) -> Iterator[tuple[str, torch.nn.Parameter, torch.nn.Parameter]]:
    """Yield every ambient and dynamic LoRA parameter pair exactly once."""
    seen: set[int] = set()
    for chunk in model:
        for module in chunk.modules():
            prefix = getattr(module, "adapter_model_prefix", None)
            a_t = getattr(module, "A_T", None)
            b_t = getattr(module, "B_T", None)
            if (
                not isinstance(prefix, str)
                or not isinstance(a_t, torch.nn.Parameter)
                or not isinstance(b_t, torch.nn.Parameter)
                or id(module) in seen
            ):
                continue
            seen.add(id(module))
            yield prefix, a_t, b_t
            for slot in getattr(module, "_slot_modules", {}).values():
                yield prefix, slot.A_T, slot.B_T
