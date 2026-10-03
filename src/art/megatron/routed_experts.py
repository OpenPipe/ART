"""Graph-owned expert replay for TrainerRank, including activation recompute."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from types import MethodType
from typing import Any

import torch


@dataclass(frozen=True)
class RoutingContext:
    targets: dict[int, dict[str, torch.Tensor]]
    layout: str = "attention"


CURRENT_ROUTES: ContextVar[RoutingContext | None] = ContextVar(
    "art_routed_experts", default=None
)


@contextmanager
def use_routes(context: RoutingContext | None):
    token = CURRENT_ROUTES.set(context)
    try:
        yield
    finally:
        CURRENT_ROUTES.reset(token)


def set_routing_layout(layout: str) -> None:
    if (context := CURRENT_ROUTES.get()) is not None:
        # Checkpoint closures retain this immutable layout snapshot.
        CURRENT_ROUTES.set(replace(context, layout=layout))


def capture_routes(function):
    context = CURRENT_ROUTES.get()

    @wraps(function)
    def wrapped(*args, **kwargs):
        with use_routes(context):
            return function(*args, **kwargs)

    return wrapped


@torch.compiler.disable
def _routing(router: Any, logits: torch.Tensor, *args: Any, **kwargs: Any):
    context = CURRENT_ROUTES.get()
    original = router._art_rank_original_routing
    if context is None:
        return original(logits, *args, **kwargs)
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    indices = context.targets[id(router)][context.layout]
    if indices.shape != (logits.numel() // logits.shape[-1], router.topk):
        raise ValueError("routed_experts do not match local router input rows")
    replay = router.router_replay
    previous = replay.target_topk_idx, replay.router_replay_action
    # Do not append to MCore's global/FIFO backward state. The checkpoint
    # closure selects this graph's exact indices for every recomputation.
    replay.target_topk_idx = indices
    replay.router_replay_action = RouterReplayAction.REPLAY_FORWARD
    try:
        return original(logits, *args, **kwargs)
    finally:
        replay.target_topk_idx, replay.router_replay_action = previous


def router_bindings(model: list[Any]) -> list[tuple[Any, int]]:
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.router_replay import RouterReplay

    from .routing_replay import _global_layer_prefixes

    bindings = []
    for chunk in model:
        prefixes = _global_layer_prefixes(chunk)
        for name, router in chunk.named_modules():
            if not isinstance(router, TopKRouter):
                continue
            if (
                router.config.moe_router_fusion
                or router.routing_type == "sinkhorn"
                or getattr(router, "_art_routing_replay_target_patched", False)
            ):
                raise ValueError("routed_experts requires unfused top-k routing")
            layer = next(
                (index for prefix, index in prefixes if name.startswith(prefix + ".")),
                None,
            )
            if layer is None:
                raise ValueError(f"Cannot identify routed_experts layer for {name}")
            if not hasattr(router, "_art_rank_original_routing"):
                if router.router_replay is None:
                    router.router_replay = RouterReplay()
                router._art_rank_original_routing = router.routing
                router.routing = MethodType(_routing, router)
            bindings.append((router, layer))
    if not bindings:
        raise ValueError("routed_experts requires a supported MoE model")
    return bindings


def validate_routes(
    routes: torch.Tensor, length: int, layers: int, bindings
) -> torch.Tensor:
    if (
        routes.dtype == torch.bool
        or routes.dtype.is_floating_point
        or routes.dtype.is_complex
    ):
        raise ValueError("routed_experts must contain integer expert IDs")
    if routes.ndim != 3 or tuple(routes.shape[:2]) != (length, layers):
        raise ValueError(
            "routed_experts must have shape [input_tokens, model_layers, top-k]"
        )
    routes = routes.detach().to(device="cpu", dtype=torch.int64)
    if bool((routes < 0).any()):
        raise ValueError("routed_experts contains missing or negative expert IDs")
    for router, layer in bindings:
        values = routes[:, layer]
        if values.shape[1] != router.topk or bool(
            (values >= router.config.num_moe_experts).any()
        ):
            raise ValueError("routed_experts does not match model expert count/top-k")
        ordered = values.sort(dim=-1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("routed_experts contains duplicate expert IDs")
    return routes.clone()


def prepare_routes(items, packed, prepared, bindings, device) -> RoutingContext | None:
    if not items or items[0].routed_experts is None:
        return None
    from .routing_replay import MoeRoutingReplayController
    from .training.trace import _routing_replay_token_uid_sets

    # Shared prefixes deliberately use the first sequence's routes, matching
    # the training packer. Different routes do not split a shared token prefix.
    routes = torch.cat(
        [
            items[segment.sequence_indices[0]].routed_experts[
                segment.start : segment.end
            ]
            for segment in packed.segments
        ]
    )
    uid_sets = _routing_replay_token_uid_sets(
        prepared.token_uids
        if prepared.token_uids is not None
        else torch.arange(packed.tokens.numel()).unsqueeze(0),
        attention_state=prepared.attention_state,
    )
    targets = {}
    for router, layer in bindings:
        layouts = {}
        for layout, uids in uid_sets.items():
            assert uids is not None
            local = MoeRoutingReplayController._token_uids_for_router_binding(
                uids, sequence_parallel=router.config.sequence_parallel
            )
            valid = (local >= 0) & (local < len(routes))
            indices = routes[:, layer].index_select(0, local.clamp(0, len(routes) - 1))
            # Only physical TP/CP padding may use placeholder routes.
            indices[~valid] = torch.arange(router.topk, dtype=torch.int64)
            layouts[layout] = indices.to(device=device)
        targets[id(router)] = layouts
    return RoutingContext(targets)
