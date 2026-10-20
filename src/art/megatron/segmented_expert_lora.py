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
) -> torch.Tensor:
    """Reference math for per-expert or singleton shared-outer A/B factors.

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
            outputs.append(expert_lora(chunk, a, b, counts, scale=scale))
    return torch.cat(outputs).index_select(0, plan.inverse)


# HybridEP's BF16 TMA payload width must be divisible by 16. IDs 0..255 are
# represented exactly in BF16; the other header columns are zero padding.
HEADER_COLUMNS = 16


@dataclass(frozen=True)
class AdapterTokens:
    refs: tuple["LoRASlotRef | None", ...]
    ids: torch.Tensor

    def __post_init__(self) -> None:
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
    """Scope immutable source IDs and restore graph-owned HybridEP geometry."""
    previous = _CURRENT_ADAPTER_TOKENS.get()
    buffer = None
    if tokens is not None and tokens.ids.is_cuda:
        from megatron.core.transformer.moe import fused_a2a

        buffer = fused_a2a._hybrid_ep_buffer
        if buffer is not None:
            buffer.set_num_tokens_per_rank(tokens.ids.numel())
    token = _CURRENT_ADAPTER_TOKENS.set(tokens)
    plan_token = _CURRENT_EXPERT_ADAPTER_PLAN.set(None)
    try:
        yield
    finally:
        _CURRENT_EXPERT_ADAPTER_PLAN.reset(plan_token)
        _CURRENT_ADAPTER_TOKENS.reset(token)
        if buffer is not None and previous is not None:
            buffer.set_num_tokens_per_rank(previous.ids.numel())


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


def pack_adapter_header(x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or ids.shape != (x.shape[0],):
        raise ValueError("one adapter ID is required for every hidden row")
    if x.dtype not in (torch.bfloat16, torch.float16) or x.shape[1] % 16:
        raise ValueError("HybridEP header requires aligned BF16/FP16 rows")
    header = x.new_zeros((x.shape[0], HEADER_COLUMNS))
    header[:, 0] = ids
    return torch.cat((x, header), dim=1)


class SegmentedHybridEPManager:
    """Instance proxy retaining Megatron's native dispatch/combine autograd."""

    def __init__(self, manager):
        self.manager = manager

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def dispatch(self, hidden_states, *args, **kwargs):
        tokens = _CURRENT_ADAPTER_TOKENS.get()
        if tokens is None:
            raise RuntimeError("segmented dispatch requires multi_adapter_tokens")
        payload = self.manager.dispatch(
            pack_adapter_header(hidden_states, tokens.ids), *args, **kwargs
        )
        hidden = payload[:, :-HEADER_COLUMNS].contiguous()
        ids = payload[:, -HEADER_COLUMNS].to(torch.int64)
        _CURRENT_EXPERT_ADAPTER_PLAN.set(
            ExpertAdapterPlan.build(
                tokens.refs, ids, self.manager.get_number_of_tokens_per_expert()
            )
        )
        return hidden

    def combine(self, hidden_states, *args, **kwargs):
        padded = torch.nn.functional.pad(hidden_states, (0, HEADER_COLUMNS))
        return self.manager.combine(padded, *args, **kwargs)[
            :, :-HEADER_COLUMNS
        ].contiguous()


def _combine_postprocess(dispatcher, hidden_states):
    result = type(dispatcher).combine_postprocess(dispatcher, hidden_states)
    manager = dispatcher._comm_manager.manager
    manager.routing_map = manager.token_probs = manager.dispatched_probs = None
    return result


def install_segmented_dispatcher(moe_layer) -> None:
    """Install before execution, keeping ordinary per-run slot Parameters.

    The caller must initialize HybridEP buffers for hidden_size+16 and the
    maximum admitted wave size before any graph is live. No buffer may grow
    while retained graphs exist. Only TP1/CP1/EP1 is admitted by v0.
    """
    config = moe_layer.config
    if (
        getattr(config, "tensor_model_parallel_size", 1) != 1
        or getattr(config, "context_parallel_size", 1) != 1
        or getattr(config, "expert_model_parallel_size", 1) != 1
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
    ):
        raise ValueError("segmented transport requires TP1/CP1/EP1 BF16/FP16")
    dispatcher = moe_layer.token_dispatcher
    if isinstance(getattr(dispatcher, "_comm_manager", None), SegmentedHybridEPManager):
        return
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEFlexTokenDispatcher,
        _HybridEPManager,
    )

    if not hasattr(dispatcher, "_comm_manager"):
        # Flex's constructor rejects TPxEP=1 although HybridEP supports it. Use
        # its initialized native groups and the existing flex methods, as in
        # the qualified EP1 spike. Never copy instance-bound ART callbacks.
        flex = object.__new__(MoEFlexTokenDispatcher)
        flex.__dict__.update(dispatcher.__dict__)
        flex.__dict__.pop("dispatch_preprocess", None)
        flex.__dict__.pop("combine_postprocess", None)
        flex.num_local_experts = moe_layer.experts.num_local_experts
        flex.local_expert_indices = list(range(flex.num_local_experts))
        flex._comm_manager = _HybridEPManager(
            dispatcher.tp_ep_group,
            flex.num_local_experts,
            flex.num_local_experts,
            config,
        )
        flex.cudagraph_attrs = [
            "_comm_manager.token_probs",
            "_comm_manager.routing_map",
        ]
        moe_layer._segmented_original_token_dispatcher = dispatcher
        moe_layer.token_dispatcher = dispatcher = flex
    manager = dispatcher._comm_manager
    if type(manager).__name__ != "_HybridEPManager":
        raise ValueError("segmented transport requires the native HybridEP manager")
    dispatcher._comm_manager = SegmentedHybridEPManager(manager)
    if getattr(config, "cuda_graph_impl", "none") == "none":
        dispatcher.combine_postprocess = partial(_combine_postprocess, dispatcher)
