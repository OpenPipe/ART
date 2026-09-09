from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from art.megatron.routing_replay import (
    ROUTER_KEY_FORMAT_VERSION,
    ROUTER_NAME_TOKEN,
    MoeRoutingReplayBundle,
    ParallelTopology,
    RouterCallRoute,
    StepRouterRoutes,
    StepRoutes,
    build_router_key_from_module_name,
)


def _flatten_router_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise RuntimeError(
            f"Router tensor must have rank >=2, got shape={tuple(tensor.shape)}"
        )
    num_experts = int(tensor.shape[-1])
    return tensor.reshape(-1, num_experts).contiguous()


def _extract_router_output_tensors(
    call_entry: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = None
    routing_map = None
    if isinstance(call_entry, dict):
        output = call_entry.get("output")
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            probs, routing_map = output[0], output[1]
        elif isinstance(output, dict):
            probs = output.get("probs")
            routing_map = output.get("routing_map")
    elif isinstance(call_entry, (list, tuple)) and len(call_entry) >= 2:
        probs, routing_map = call_entry[0], call_entry[1]
    else:
        raise RuntimeError(f"Unsupported router output type: {type(call_entry)}")

    if not isinstance(probs, torch.Tensor):
        raise RuntimeError(f"Expected probs tensor, got {type(probs)}")
    if not isinstance(routing_map, torch.Tensor):
        raise RuntimeError(f"Expected routing_map tensor, got {type(routing_map)}")

    probs_2d = _flatten_router_tensor(probs.to(torch.float32))
    routing_map_2d = _flatten_router_tensor(routing_map.bool())
    if probs_2d.shape != routing_map_2d.shape:
        raise RuntimeError(
            "Router output shape mismatch: "
            f"probs={tuple(probs_2d.shape)} routing_map={tuple(routing_map_2d.shape)}"
        )
    return probs_2d, routing_map_2d


def _extract_dp_slot_from_rank_meta(rank_meta: Any) -> tuple[int, int] | None:
    if isinstance(rank_meta, dict):
        rank_meta = [rank_meta]
    if not isinstance(rank_meta, list) or not rank_meta:
        return None
    dp_ranks = {
        int(item["dp_rank"])
        for item in rank_meta
        if isinstance(item, dict) and "dp_rank" in item
    }
    dp_world_sizes = {
        int(item["dp_world_size"])
        for item in rank_meta
        if isinstance(item, dict) and "dp_world_size" in item
    }
    if len(dp_ranks) != 1 or len(dp_world_sizes) != 1:
        return None
    return next(iter(dp_ranks)), next(iter(dp_world_sizes))


def _trace_call_route_metadata(
    call_entry: dict[str, Any],
) -> tuple[int | None, int | None]:
    sample_index = call_entry.get("micro_sample_index")
    if isinstance(sample_index, int):
        return int(sample_index), None
    dp_slot = _extract_dp_slot_from_rank_meta(call_entry.get("rank_meta"))
    micro_order = int(call_entry.get("micro_order", 0))
    if dp_slot is None:
        return None, micro_order
    dp_rank, dp_world_size = dp_slot
    return None, micro_order * dp_world_size + dp_rank


def _compact_route_from_dense(
    probs_2d: torch.Tensor,
    routing_map_2d: torch.Tensor,
) -> RouterCallRoute:
    num_tokens, num_experts = probs_2d.shape
    if num_tokens == 0:
        return RouterCallRoute(
            expert_indices=torch.zeros((0, 0), dtype=torch.int32),
            expert_probs=torch.zeros((0, 0), dtype=torch.float32),
            expert_mask=torch.zeros((0, 0), dtype=torch.bool),
            num_experts=num_experts,
        )

    max_topk = int(routing_map_2d.sum(dim=1).max().item())
    expert_indices = torch.zeros((num_tokens, max_topk), dtype=torch.int32)
    expert_probs = torch.zeros((num_tokens, max_topk), dtype=torch.float32)
    expert_mask = torch.zeros((num_tokens, max_topk), dtype=torch.bool)
    for token_index in range(num_tokens):
        expert_ids = torch.nonzero(
            routing_map_2d[token_index], as_tuple=False
        ).flatten()
        slot_count = int(expert_ids.numel())
        if slot_count == 0:
            continue
        expert_indices[token_index, :slot_count] = expert_ids.to(torch.int32)
        expert_probs[token_index, :slot_count] = probs_2d[token_index, expert_ids].to(
            torch.float32
        )
        expert_mask[token_index, :slot_count] = True

    return RouterCallRoute(
        expert_indices=expert_indices,
        expert_probs=expert_probs,
        expert_mask=expert_mask,
        num_experts=num_experts,
    )


def _rank_token_counts(
    call_entry: dict[str, Any], token_count: int
) -> tuple[int, ...] | None:
    row_splits = call_entry.get("primary_output__row_splits")
    if not isinstance(row_splits, list):
        return None
    counts = tuple(int(count) for count in row_splits)
    if sum(counts) != token_count:
        return None
    return counts


def _route_token_uids(
    call_entry: dict[str, Any], token_count: int
) -> torch.Tensor | None:
    token_uids = call_entry.get("row_token_uids")
    if not isinstance(token_uids, torch.Tensor):
        return None
    token_uids = token_uids.to(dtype=torch.int64).reshape(-1).contiguous()
    if int(token_uids.numel()) != token_count:
        raise RuntimeError(
            "Router row token UID count must match route rows: "
            f"uids={int(token_uids.numel())}, routes={token_count}"
        )
    if bool((token_uids < 0).any().item()) or int(token_uids.unique().numel()) != int(
        token_uids.numel()
    ):
        raise RuntimeError("Router row token UIDs must be unique and non-negative")
    return token_uids


def _expand_route_to_token_span(
    route: RouterCallRoute,
    token_uids: torch.Tensor | None,
    token_count: int,
) -> RouterCallRoute:
    if token_uids is None:
        if route.num_global_tokens != token_count:
            raise RuntimeError(
                "A compact router route requires row token UIDs: "
                f"routes={route.num_global_tokens}, token_span={token_count}"
            )
        return route
    identity = torch.arange(token_count, dtype=torch.int64)
    if route.num_global_tokens == token_count and torch.equal(token_uids, identity):
        return route
    if int(token_uids.numel()) > 0 and int(token_uids.max().item()) >= token_count:
        raise RuntimeError(
            "Router row token UID exceeds the replay token span: "
            f"max_uid={int(token_uids.max().item())}, token_span={token_count}"
        )

    rows = torch.arange(token_count, dtype=torch.int64).unsqueeze(1)
    slots = torch.arange(route.max_topk, dtype=torch.int64).unsqueeze(0)
    expert_indices = ((rows + slots) % route.num_experts).to(torch.int32)
    expert_indices.index_copy_(0, token_uids, route.expert_indices)
    expert_probs = None
    if route.expert_probs is not None:
        expert_probs = torch.zeros((token_count, route.max_topk), dtype=torch.float32)
        expert_probs.index_copy_(0, token_uids, route.expert_probs)
    expert_mask = None
    if route.expert_mask is not None:
        expert_mask = torch.ones((token_count, route.max_topk), dtype=torch.bool)
        expert_mask.index_copy_(0, token_uids, route.expert_mask)
    return RouterCallRoute(
        expert_indices=expert_indices,
        expert_probs=expert_probs,
        expert_mask=expert_mask,
        num_experts=route.num_experts,
        sample_index=route.sample_index,
        micro_slot=route.micro_slot,
    )


def _dedupe_checkpoint_router_calls(
    call_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    previous_call_key: tuple[int | None, int | None, int] | None = None
    previous_route: RouterCallRoute | None = None
    for call_entry in call_entries:
        probs_2d, routing_map_2d = _extract_router_output_tensors(call_entry)
        compact_route = _compact_route_from_dense(probs_2d, routing_map_2d)
        sample_index, micro_slot = _trace_call_route_metadata(call_entry)
        call_key = (
            sample_index,
            micro_slot,
            int(call_entry.get("micro_order", 0)),
        )
        is_checkpoint_duplicate = (
            previous_call_key == call_key
            and previous_route is not None
            and torch.equal(compact_route.expert_indices, previous_route.expert_indices)
            and (
                compact_route.expert_probs is None
                and previous_route.expert_probs is None
                or compact_route.expert_probs is not None
                and previous_route.expert_probs is not None
                and torch.equal(compact_route.expert_probs, previous_route.expert_probs)
            )
            and compact_route.expert_mask is not None
            and previous_route.expert_mask is not None
            and torch.equal(compact_route.expert_mask, previous_route.expert_mask)
        )
        if is_checkpoint_duplicate:
            continue
        deduped.append(call_entry)
        previous_call_key = call_key
        previous_route = compact_route
    return deduped


def _build_router_key_from_trace_name(trace_module_name: str) -> str:
    if not trace_module_name.startswith("chunk"):
        raise RuntimeError(
            "Forward trace router module name must start with 'chunk<idx>.'; "
            f"got '{trace_module_name}'"
        )
    chunk_prefix, separator, module_name = trace_module_name.partition(".")
    if not separator or not chunk_prefix.removeprefix("chunk").isdigit():
        raise RuntimeError(
            "Forward trace router module name must start with 'chunk<idx>.'; "
            f"got '{trace_module_name}'"
        )
    return build_router_key_from_module_name(
        chunk_index=int(chunk_prefix.removeprefix("chunk")),
        module_name=module_name,
    )


def build_bundle_from_forward_trace_dir(
    *,
    traces_dir: str | Path,
    num_steps: int,
    topology: ParallelTopology,
) -> MoeRoutingReplayBundle:
    trace_dir = Path(traces_dir)
    steps: dict[int, StepRoutes] = {}
    router_keys_union: set[str] = set()
    max_topk = 0

    for step_index in range(num_steps):
        trace_path = trace_dir / f"forward_trace_step_{step_index:03d}.pt"
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Missing forward trace for step={step_index}: {trace_path}"
            )
        step_trace: dict[str, list[dict[str, Any]]] = torch.load(
            trace_path, map_location="cpu", weights_only=False
        )

        step_routers: dict[str, StepRouterRoutes] = {}
        step_global_tokens: int | None = None
        route_token_uids: dict[tuple[str, int], torch.Tensor | None] = {}
        for module_name in sorted(step_trace.keys()):
            if ROUTER_NAME_TOKEN not in module_name:
                continue
            router_key = _build_router_key_from_trace_name(module_name)
            router_calls: dict[int, RouterCallRoute] = {}
            deduped_router_calls = _dedupe_checkpoint_router_calls(
                step_trace[module_name]
            )
            for call_index, call_entry in enumerate(deduped_router_calls):
                probs_2d, routing_map_2d = _extract_router_output_tensors(call_entry)
                compact_route = _compact_route_from_dense(probs_2d, routing_map_2d)
                sample_index, micro_slot = _trace_call_route_metadata(call_entry)
                compact_route.sample_index = sample_index
                compact_route.micro_slot = micro_slot
                compact_route.rank_token_counts = _rank_token_counts(
                    call_entry, compact_route.num_global_tokens
                )
                router_calls[call_index] = compact_route
                token_uids = _route_token_uids(
                    call_entry, compact_route.num_global_tokens
                )
                route_token_uids[(router_key, call_index)] = token_uids
                max_topk = max(max_topk, compact_route.max_topk)
                token_count = compact_route.num_global_tokens
                if token_uids is not None and int(token_uids.numel()) > 0:
                    token_count = max(token_count, int(token_uids.max().item()) + 1)
                step_global_tokens = (
                    token_count
                    if step_global_tokens is None
                    else max(step_global_tokens, token_count)
                )

            if not router_calls:
                raise RuntimeError(
                    f"Router trace has no calls for module '{module_name}' at step={step_index}"
                )
            step_routers[router_key] = StepRouterRoutes(calls=router_calls)
            router_keys_union.add(router_key)

        if not step_routers:
            raise RuntimeError(
                f"No router traces found for step={step_index} in {trace_path}"
            )
        if step_global_tokens is None:
            raise RuntimeError(
                f"Could not infer token count for step={step_index} from router traces"
            )
        for router_key, router_routes in step_routers.items():
            for call_index, route in router_routes.calls.items():
                router_routes.calls[call_index] = _expand_route_to_token_span(
                    route,
                    route_token_uids[(router_key, call_index)],
                    step_global_tokens,
                )
        global_token_uids = torch.arange(step_global_tokens, dtype=torch.int64)
        steps[step_index] = StepRoutes(
            routers=step_routers,
            global_token_uids=global_token_uids,
        )

    router_keys = sorted(router_keys_union)
    for step_index, step_routes in steps.items():
        if set(step_routes.routers.keys()) != set(router_keys):
            raise RuntimeError(
                f"Step {step_index} router keys differ from global set: "
                f"step_keys={sorted(step_routes.routers.keys())}, router_keys={router_keys}"
            )

    return MoeRoutingReplayBundle(
        format_version=ROUTER_KEY_FORMAT_VERSION,
        topology=topology,
        num_steps=num_steps,
        max_topk=max_topk,
        router_keys=router_keys,
        steps=steps,
    )
