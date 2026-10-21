"""Wave-local adapter ownership for expert-major dispatched rows.

The plan follows the dispatcher's actual row order, including repeated top-k
rows and padding. Shared-outer factors stay compact; Parameters are never
replaced or expanded across experts. Transport must carry adapter IDs with rows.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .lora import LoRASlotRef


@dataclass(frozen=True)
class ExpertAdapterPlan:
    refs: tuple["LoRASlotRef | None", ...]
    permutation: torch.Tensor
    inverse: torch.Tensor
    counts: tuple[tuple[int, ...], ...]  # adapter-major, then expert

    @classmethod
    def build(
        cls,
        refs: Sequence["LoRASlotRef | None"],
        adapter_ids: torch.Tensor,
        expert_counts: Sequence[int] | torch.Tensor,
    ) -> "ExpertAdapterPlan":
        """Build once after dispatch; CPU metadata validation synchronizes CUDA."""
        if not refs or len(set(refs)) != len(refs):
            raise ValueError("expert plan requires unique adapter refs")
        if adapter_ids.ndim != 1 or adapter_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("adapter IDs must be a 1D int32/int64 table")
        raw = torch.as_tensor(expert_counts)
        if (
            raw.ndim != 1
            or raw.dtype not in (torch.int32, torch.int64)
            or not raw.numel()
        ):
            raise ValueError("expert counts must be a nonempty integer vector")
        counts = raw.detach().cpu().tolist()
        ids = adapter_ids.detach().cpu().tolist()
        if any(n < 0 for n in counts) or sum(counts) != len(ids):
            raise ValueError("expert counts must cover every dispatched row")
        if any(i < 0 or i >= len(refs) for i in ids):
            raise ValueError("adapter IDs must select a wave adapter")
        experts = torch.repeat_interleave(
            torch.arange(len(counts), device=adapter_ids.device),
            torch.tensor(counts, device=adapter_ids.device),
            output_size=adapter_ids.numel(),
        )
        groups = adapter_ids.to(torch.int64) * len(counts) + experts
        permutation = groups.argsort(stable=True)
        inverse = permutation.argsort()
        histogram = torch.bincount(groups.cpu(), minlength=len(refs) * len(counts))
        return cls(
            tuple(refs),
            permutation,
            inverse,
            tuple(tuple(row) for row in histogram.reshape(len(refs), -1).tolist()),
        )

    @property
    def rows(self) -> int:
        return sum(map(sum, self.counts))


_CURRENT_EXPERT_ADAPTER_PLAN: ContextVar[ExpertAdapterPlan | None] = ContextVar(
    "art_megatron_expert_adapter_plan", default=None
)


@contextmanager
def use_expert_adapter_plan(plan: ExpertAdapterPlan | None) -> Iterator[None]:
    token = _CURRENT_EXPERT_ADAPTER_PLAN.set(plan)
    try:
        yield
    finally:
        _CURRENT_EXPERT_ADAPTER_PLAN.reset(token)


def expert_lora(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    counts: Sequence[int],
    *,
    scale: float = 1.0,
    shared_outer_kernel: bool = True,
) -> torch.Tensor:
    """Per-adapter math for per-expert or compact shared-outer A/B factors.

    H200 reuses the ordinary native kernels with the adapter's own row geometry.
    Other devices and FP32 use the Torch reference below.

    Shared A projects all of an adapter's rows together before expert B. Shared
    B projects all intermediate rows together after expert A. This avoids both
    expert-sized shared weights and expert-sized shared gradient temporaries.
    """
    experts = len(counts)
    if (
        x.ndim != 2
        or a.ndim != 3
        or b.ndim != 3
        or a.shape[0] not in (1, experts)
        or b.shape[0] not in (1, experts)
        or a.shape[1] != x.shape[1]
        or a.shape[2] != b.shape[1]
        or sum(counts) != x.shape[0]
        or any(n < 0 for n in counts)
    ):
        raise ValueError("expert rows/counts/factor shapes disagree")
    if not shared_outer_kernel:
        a = a.expand(experts, -1, -1)
        b = b.expand(experts, -1, -1)
    if (
        x.shape[0] > 0
        and x.is_cuda
        and x.dtype in {torch.bfloat16, torch.float16}
        and torch.cuda.get_device_capability(x.device)[0] == 9
    ):
        if a.shape[0] == b.shape[0] == experts:
            from .kernels.cute_grouped_lora_quack import quack_grouped_lora

            return quack_grouped_lora(x, a, b, list(counts), scale=scale)
        if (a.shape[0], b.shape[0]) in {(1, experts), (experts, 1)}:
            from .kernels.shared_outer_lora import shared_outer_lora

            return shared_outer_lora(
                x,
                a,
                b,
                list(counts),
                scale=scale,
                shared_factor="A" if a.shape[0] == 1 else "B",
            )
    if a.shape[0] == 1:
        tmp = x @ a[0]
    else:
        tmp = torch.cat([chunk @ a[e] for e, chunk in enumerate(x.split(counts))])
    if b.shape[0] == 1:
        out = tmp @ b[0]
    else:
        out = torch.cat([chunk @ b[e] for e, chunk in enumerate(tmp.split(counts))])
    return out if scale == 1.0 else out * scale


def segmented_expert_lora(
    x: torch.Tensor,
    factors: Sequence[tuple[torch.Tensor, torch.Tensor, float] | None],
    plan: ExpertAdapterPlan,
    *,
    out_features: int,
    shared_outer_kernel: bool = True,
) -> torch.Tensor:
    """One base expert pass can use distinct original adapter parameters."""
    if x.ndim != 2 or x.shape[0] != plan.rows or len(factors) != len(plan.refs):
        raise ValueError("projection rows/factors disagree with expert plan")
    sorted_x = x.index_select(0, plan.permutation)
    outputs = []
    for chunk, pair, counts in zip(
        sorted_x.split(tuple(map(sum, plan.counts))), factors, plan.counts, strict=True
    ):
        if pair is None:
            outputs.append(chunk.new_zeros((chunk.shape[0], out_features)))
        else:
            a, b, scale = pair
            if b.shape[-1] != out_features:
                raise ValueError("expert output width disagrees with factors")
            outputs.append(
                expert_lora(
                    chunk,
                    a,
                    b,
                    counts,
                    scale=scale,
                    shared_outer_kernel=shared_outer_kernel,
                )
            )
    return torch.cat(outputs).index_select(0, plan.inverse)


@dataclass(frozen=True)
class AdapterTokens:
    refs: tuple["LoRASlotRef | None", ...]
    ids: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(self, "refs", tuple(self.refs))
        if not 1 <= len(self.refs) <= 256 or len(set(self.refs)) != len(self.refs):
            raise ValueError("transport supports 1..256 distinct adapter refs")
        if self.ids.ndim != 1 or self.ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token adapter IDs must be a 1D integer vector")
        ids = self.ids.detach().clone()
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= len(self.refs)):
            raise ValueError("token adapter ID is outside the wave")
        object.__setattr__(self, "ids", ids)


_CURRENT_ADAPTER_TOKENS: ContextVar[AdapterTokens | None] = ContextVar(
    "art_megatron_adapter_tokens", default=None
)


@contextmanager
def multi_adapter_tokens(tokens: AdapterTokens | None) -> Iterator[None]:
    """Scope immutable source IDs for native EP1 dispatch and recomputation."""
    token = _CURRENT_ADAPTER_TOKENS.set(tokens)
    plan_token = _CURRENT_EXPERT_ADAPTER_PLAN.set(None)
    try:
        yield
    finally:
        _CURRENT_EXPERT_ADAPTER_PLAN.reset(plan_token)
        _CURRENT_ADAPTER_TOKENS.reset(token)


def capture_expert_context(function):
    """Restore source IDs and dispatched plans on recomputation worker threads."""
    from functools import wraps

    tokens = _CURRENT_ADAPTER_TOKENS.get()
    plan = _CURRENT_EXPERT_ADAPTER_PLAN.get()

    @wraps(function)
    def wrapped(*args, **kwargs):
        with multi_adapter_tokens(tokens), use_expert_adapter_plan(plan):
            return function(*args, **kwargs)

    return wrapped


def _ep1_dispatch_postprocess(dispatcher, original, hidden_states, probs):
    """Replay EP1's exact native permutation on a detached scalar ID column."""
    from megatron.core.transformer.moe.moe_utils import permute

    tokens = _CURRENT_ADAPTER_TOKENS.get()
    if tokens is None:
        return original(hidden_states, probs)
    if (
        tokens.ids.device != hidden_states.device
        or tokens.ids.numel() != dispatcher.routing_map.shape[0]
    ):
        raise ValueError("source adapter IDs must match native routing rows/device")
    result = original(hidden_states, probs)
    # TP=EP=1 makes native permutation 2 an identity. Use the same permutation
    # implementation/flags as hidden states, including TE's fused row map.
    ids, _, _, _, _ = permute(
        tokens.ids[:, None].to(hidden_states.dtype),
        dispatcher.routing_map,
        # TE zeros the payload for zero-probability padding slots. Metadata
        # must retain IDs even there, so use nonzero detached dummy scores.
        probs=torch.ones_like(dispatcher.routing_map, dtype=torch.float32),
        num_out_tokens=dispatcher.num_out_tokens,
        fused=dispatcher.config.moe_permute_fusion,
        drop_and_pad=False,
    )
    _CURRENT_EXPERT_ADAPTER_PLAN.set(
        ExpertAdapterPlan.build(tokens.refs, ids[:, 0].to(torch.int64), result[1])
    )
    return result


def _router_forward(original, hidden_states, padding_mask=None):
    """Keep native gate geometry, capacity and auxiliary losses per adapter."""
    from .segmented_lora import _CURRENT_LORA_SEGMENTS, slice_lora_segments

    segments = _CURRENT_LORA_SEGMENTS.get()
    if segments is None:
        return original(hidden_states, padding_mask=padding_mask)
    if (
        hidden_states.ndim != 3
        or hidden_states.shape[1] != 1
        or hidden_states.shape[0] != segments.rows
        or not segments.rows
        or (padding_mask is not None and padding_mask.shape != hidden_states.shape[:2])
    ):
        raise ValueError("segmented router requires complete batch-one adapter rows")
    results = []
    start = 0
    for count in segments.counts:
        stop = start + count
        if count:
            with slice_lora_segments(start, stop):
                results.append(
                    original(
                        hidden_states[start:stop].contiguous(),
                        padding_mask=None
                        if padding_mask is None
                        else padding_mask[start:stop],
                    )
                )
        start = stop
    return tuple(torch.cat(parts, dim=0) for parts in zip(*results, strict=True))


def _row_forward(original, hidden_states, *args, **kwargs):
    """Preserve native row shapes for TE normalization/projection algorithms."""
    from .canonical.operators import _join
    from .segmented_lora import _CURRENT_LORA_SEGMENTS, slice_lora_segments

    segments = _CURRENT_LORA_SEGMENTS.get()
    if (
        segments is None
        or hidden_states.shape[0] != segments.rows
        or sum(count > 0 for count in segments.counts) < 2
    ):
        return original(hidden_states, *args, **kwargs)
    outputs = []
    start = 0
    for count in segments.counts:
        stop = start + count
        if count:
            with slice_lora_segments(start, stop):
                outputs.append(
                    original(hidden_states[start:stop].contiguous(), *args, **kwargs)
                )
        start = stop
    return _join(outputs, next(count for count in segments.counts if count))


def install_segmented_row_modules(model) -> None:
    """Opt into native adapter geometry without changing ordinary forwards."""
    from .canonical.operators import _module_types
    from .lora import LoRA

    row_types, _, _ = _module_types()
    for module in model.modules():
        if (
            isinstance(module, row_types)
            and not isinstance(module, LoRA)
            and not getattr(module, "_art_segmented_rows", False)
        ):
            module.forward = partial(_row_forward, module.forward)
            module._art_segmented_rows = True


def install_segmented_dispatcher(moe_layer) -> None:
    """Opt into native TP1/CP1/EP1 adapter IDs; ordinary dispatch stays native."""
    config = moe_layer.config
    if (
        getattr(config, "tensor_model_parallel_size", 1) != 1
        or getattr(config, "context_parallel_size", 1) != 1
        or getattr(config, "expert_model_parallel_size", 1) != 1
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or getattr(config, "moe_router_enable_expert_bias", False)
        or getattr(getattr(moe_layer, "router", None), "global_tokens_per_expert", None)
        is not None
    ):
        raise ValueError(
            "segmented transport requires stateless TP1/CP1/EP1 BF16/FP16 routing"
        )
    dispatcher = moe_layer.token_dispatcher
    if getattr(dispatcher, "_art_segmented_ep1", False):
        return
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    if type(dispatcher) is not MoEAlltoAllTokenDispatcher:
        raise ValueError("unsupported native segmented dispatcher")
    if dispatcher.tp_size != 1 or dispatcher.ep_size != 1 or dispatcher.drop_and_pad:
        raise ValueError(
            "native segmented dispatch requires TP1/EP1 without capacity padding"
        )
    dispatcher.dispatch_postprocess = partial(
        _ep1_dispatch_postprocess, dispatcher, dispatcher.dispatch_postprocess
    )
    if hasattr(moe_layer, "router"):
        moe_layer.router.forward = partial(_router_forward, moe_layer.router.forward)
    dispatcher._art_segmented_ep1 = True
