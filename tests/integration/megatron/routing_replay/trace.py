from __future__ import annotations

import types
from typing import Any, Callable

from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.moe.moe_utils import permute, sort_chunks_by_idxs
import torch

TRACE_ROW_TOKEN_UIDS_ATTR = "_art_trace_row_token_uids"
TRACE_UID_SPAN_ATTR = "_art_trace_uid_span"
_CONTROLLER_GETTER: Callable[[], Any | None] | None = None


def _active_controller() -> Any | None:
    if _CONTROLLER_GETTER is None:
        return None
    return _CONTROLLER_GETTER()


@torch._dynamo.disable
def _dispatcher_local_token_uids(
    controller: Any,
    dispatcher: Any,
    *,
    num_local_tokens: int,
) -> torch.Tensor:
    num_local_tokens = int(num_local_tokens)
    step_routes = controller._active_step_routes
    if step_routes is None:
        raise RuntimeError("Routing replay dispatcher used without an active step")
    sequence_parallel = bool(
        getattr(getattr(dispatcher, "config", None), "sequence_parallel", False)
    )
    prepared_uids = getattr(
        controller,
        "local_token_uids_for_active_dispatch",
        None,
    )
    if callable(prepared_uids):
        local_uids = prepared_uids(
            num_local_tokens=num_local_tokens,
            sequence_parallel=sequence_parallel,
        )
    else:
        local_uids = None
    if local_uids is None:
        route_uids, rank_sliced = _active_route_global_token_uids(controller)
        if int(route_uids.numel()) == num_local_tokens:
            local_uids = route_uids
        elif rank_sliced:
            local_uids = torch.arange(num_local_tokens, dtype=torch.int64)
        else:
            local_uids = controller.local_token_indexer.build_local_token_uids(
                global_token_uids=route_uids,
                num_local_tokens=num_local_tokens,
                sequence_parallel=sequence_parallel,
                context_parallel_size=int(
                    getattr(
                        getattr(dispatcher, "config", None),
                        "context_parallel_size",
                        1,
                    )
                ),
            )
    sample_index = getattr(controller, "_active_sample_index", None)
    uid_span = int(step_routes.global_token_uids.numel())
    if isinstance(sample_index, int) and sample_index >= 0 and uid_span > 0:
        if bool((local_uids >= 0).all().item()):
            local_uids = local_uids + sample_index * uid_span
        else:
            local_uids = local_uids.clone()
            valid_mask = local_uids >= 0
            local_uids[valid_mask] += sample_index * uid_span
    return local_uids


def _active_route_global_token_uids(controller: Any) -> tuple[torch.Tensor, bool]:
    step_routes = controller._active_step_routes
    if step_routes is None:
        raise RuntimeError("Routing replay dispatcher used without an active step")
    active_call_key = controller._active_router_call_key()
    if active_call_key is None:
        return step_routes.global_token_uids, False
    for router_key in sorted(controller._local_router_keys):
        router_calls = step_routes.routers[router_key].calls
        last_call_index = controller._router_last_call_indices.get(router_key)
        if last_call_index in router_calls:
            route = router_calls[last_call_index]
            if controller._router_call_key(route) == active_call_key:
                return _route_token_uids_for_rank(route)
        for route in router_calls.values():
            if controller._router_call_key(route) == active_call_key:
                return _route_token_uids_for_rank(route)
    return step_routes.global_token_uids, False


def _route_token_uids_for_rank(route: Any) -> tuple[torch.Tensor, bool]:
    token_uids = torch.arange(route.num_global_tokens, dtype=torch.int64)
    rank_token_counts = getattr(route, "rank_token_counts", None)
    if (
        rank_token_counts is None or not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
    ):
        return token_uids, False
    world_size = int(torch.distributed.get_world_size())  # ty: ignore[possibly-missing-attribute]
    if len(rank_token_counts) != world_size:
        return token_uids, False
    rank = int(torch.distributed.get_rank())  # ty: ignore[possibly-missing-attribute]
    start = sum(int(count) for count in rank_token_counts[:rank])
    end = start + int(rank_token_counts[rank])
    return token_uids[start:end].contiguous(), True


def _trace_row_uids_from_source(source: Any) -> tuple[torch.Tensor | None, int | None]:
    row_token_uids = getattr(source, TRACE_ROW_TOKEN_UIDS_ATTR, None)
    if not isinstance(row_token_uids, torch.Tensor):
        return None, None
    uid_span = getattr(source, TRACE_UID_SPAN_ATTR, None)
    uid_span_int = uid_span if isinstance(uid_span, int) and uid_span > 0 else None
    return row_token_uids, uid_span_int


def _attach_trace_row_uids(
    target: Any,
    *,
    row_token_uids: torch.Tensor,
    uid_span: int | None,
) -> None:
    setattr(
        target,
        TRACE_ROW_TOKEN_UIDS_ATTR,
        row_token_uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1),
    )
    setattr(target, TRACE_UID_SPAN_ATTR, uid_span)


def _attach_trace_row_uids_recursive(
    target: Any,
    *,
    row_token_uids: torch.Tensor,
    uid_span: int | None,
) -> None:
    if isinstance(target, torch.Tensor):
        _attach_trace_row_uids(
            target,
            row_token_uids=row_token_uids,
            uid_span=uid_span,
        )
        return
    if isinstance(target, dict):
        for value in target.values():
            _attach_trace_row_uids_recursive(
                value,
                row_token_uids=row_token_uids,
                uid_span=uid_span,
            )
        return
    if isinstance(target, (list, tuple)):
        for value in target:
            _attach_trace_row_uids_recursive(
                value,
                row_token_uids=row_token_uids,
                uid_span=uid_span,
            )


@torch._dynamo.disable
def _attach_router_output_trace_row_uids(
    *,
    controller: Any,
    router_key: str,
    output: Any,
) -> None:
    target_key = getattr(controller, "_router_prepared_target_keys", {}).get(router_key)
    if target_key is None:
        return
    token_uid_key, _call_index = target_key
    token_uids = getattr(controller, "_prepared_uid_sets", {}).get(token_uid_key)
    binding = getattr(controller, "_router_bindings", {}).get(router_key)
    if token_uids is None or binding is None:
        return
    router_token_uids = controller._token_uids_for_router_binding(
        token_uids,
        sequence_parallel=bool(binding["sequence_parallel"]),
    )
    _attach_trace_row_uids_recursive(
        output,
        row_token_uids=router_token_uids,
        uid_span=getattr(controller, "_global_uid_count", None),
    )


def _install_router_output_trace_hooks(controller: Any | None) -> None:
    if controller is None:
        return
    for router_key, binding in getattr(controller, "_router_bindings", {}).items():
        module = binding.get("module")
        if module is None or getattr(module, "_art_oracle_router_trace_patched", False):
            continue
        original_routing = module.routing

        def patched_routing(
            router_module: Any,
            *args: Any,
            _original_routing: Any = original_routing,
            _router_key: str = router_key,
            **kwargs: Any,
        ) -> Any:
            del router_module
            output = _original_routing(*args, **kwargs)
            controller = _active_controller()
            if controller is not None:
                _attach_router_output_trace_row_uids(
                    controller=controller,
                    router_key=_router_key,
                    output=output,
                )
            return output

        module.routing = types.MethodType(patched_routing, module)
        setattr(module, "_art_oracle_router_trace_patched", True)


@torch._dynamo.disable
def _propagate_grouped_mlp_trace_row_uids(source: Any, linear_fc2: Any) -> None:
    row_token_uids, uid_span = _trace_row_uids_from_source(source)
    if row_token_uids is None:
        return
    _attach_trace_row_uids(
        linear_fc2,
        row_token_uids=row_token_uids,
        uid_span=uid_span,
    )


@torch._dynamo.disable
def _propagate_fc2_trace_row_uids(
    *,
    x: Any,
    module: Any,
    linear_fc2: Any,
    lora: Any,
) -> None:
    row_token_uids, uid_span = _trace_row_uids_from_source(x)
    if row_token_uids is None:
        row_token_uids, uid_span = _trace_row_uids_from_source(module)
    if row_token_uids is None:
        return
    _attach_trace_row_uids(
        linear_fc2,
        row_token_uids=row_token_uids,
        uid_span=uid_span,
    )
    _attach_trace_row_uids(
        lora,
        row_token_uids=row_token_uids,
        uid_span=uid_span,
    )


def _canonicalize_expert_token_order(
    expert_inputs: torch.Tensor,
    expert_probs: torch.Tensor,
    expert_token_uids: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor | list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = (
        [int(count) for count in tokens_per_expert.tolist()]
        if isinstance(tokens_per_expert, torch.Tensor)
        else [int(count) for count in tokens_per_expert]
    )
    if sum(counts) != int(expert_token_uids.numel()):
        raise RuntimeError(
            "Expert token uid count mismatch after dispatch: "
            f"uids={int(expert_token_uids.numel())}, "
            f"tokens_per_expert_sum={sum(counts)}"
        )

    order_segments: list[torch.Tensor] = []
    cursor = 0
    for count in counts:
        if count <= 1:
            order_segments.append(
                torch.arange(cursor, cursor + count, dtype=torch.long)
            )
            cursor += count
            continue
        segment_uids = expert_token_uids[cursor : cursor + count].to(device="cpu")
        order_segments.append(torch.argsort(segment_uids, stable=True) + cursor)
        cursor += count
    if not order_segments:
        empty = torch.empty(0, dtype=torch.long)
        return expert_inputs, expert_probs, expert_token_uids, empty

    canonical_order_cpu = torch.cat(order_segments, dim=0)
    inverse_order_cpu = torch.empty_like(canonical_order_cpu)
    inverse_order_cpu[canonical_order_cpu] = torch.arange(
        canonical_order_cpu.numel(), dtype=torch.long
    )
    canonical_order = canonical_order_cpu.to(
        device=expert_inputs.device, dtype=torch.long
    )
    return (
        expert_inputs.index_select(0, canonical_order),
        expert_probs.index_select(0, canonical_order),
        expert_token_uids.index_select(
            0,
            canonical_order_cpu.to(device=expert_token_uids.device, dtype=torch.long),
        ),
        inverse_order_cpu,
    )


def _canonical_trace_row_uids(
    expert_token_uids: torch.Tensor,
    *,
    tokens_per_expert: torch.Tensor | list[int],
    local_expert_indices: list[int] | tuple[int, ...] | None,
    sample_uid_span: int,
    num_experts: int,
) -> tuple[torch.Tensor, int]:
    counts = (
        [int(count) for count in tokens_per_expert.tolist()]
        if isinstance(tokens_per_expert, torch.Tensor)
        else [int(count) for count in tokens_per_expert]
    )
    expert_indices = (
        [int(expert_index) for expert_index in local_expert_indices]
        if local_expert_indices is not None
        else list(range(len(counts)))
    )
    if len(expert_indices) != len(counts):
        raise RuntimeError(
            "Local expert index metadata mismatch: "
            f"num_expert_indices={len(expert_indices)}, num_counts={len(counts)}"
        )
    row_uid_span = sample_uid_span * max(int(num_experts), 1)
    row_uid_chunks: list[torch.Tensor] = []
    cursor = 0
    for global_expert_id, count in zip(expert_indices, counts, strict=True):
        segment = expert_token_uids[cursor : cursor + count].to(dtype=torch.int64)
        sample_ids = torch.div(segment, sample_uid_span, rounding_mode="floor")
        local_token_ids = torch.remainder(segment, sample_uid_span)
        row_uid_chunks.append(
            sample_ids * row_uid_span
            + int(global_expert_id) * sample_uid_span
            + local_token_ids
        )
        cursor += count
    if cursor != int(expert_token_uids.numel()):
        raise RuntimeError(
            "Canonical trace row uid construction did not consume all expert rows: "
            f"consumed={cursor}, total={int(expert_token_uids.numel())}"
        )
    if not row_uid_chunks:
        return expert_token_uids.new_empty((0,), dtype=torch.int64), row_uid_span
    return torch.cat(row_uid_chunks, dim=0).contiguous(), row_uid_span


@torch._dynamo.disable
def _build_dispatch_postprocess_trace(
    *,
    dispatcher: Any,
    controller: Any,
    global_input_token_uids: torch.Tensor,
    expert_inputs: torch.Tensor,
    expert_probs: torch.Tensor,
    tokens_per_expert: torch.Tensor | list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    expert_token_uids = global_input_token_uids
    if dispatcher.num_local_experts > 1:
        sorted_token_uids = sort_chunks_by_idxs(
            expert_token_uids.unsqueeze(-1),
            dispatcher.num_global_tokens_per_local_expert.ravel(),
            dispatcher.sort_input_by_local_experts,
            fused=False,
        )[0]
        expert_token_uids = sorted_token_uids.reshape(-1).contiguous()
    (
        expert_inputs,
        expert_probs,
        canonical_expert_token_uids,
        inverse_order_cpu,
    ) = _canonicalize_expert_token_order(
        expert_inputs,
        expert_probs,
        expert_token_uids,
        tokens_per_expert=tokens_per_expert,
    )
    active_step_routes = controller._active_step_routes
    if active_step_routes is None:
        raise RuntimeError("MoE replay dispatcher preprocess called before set_step")
    trace_row_uids, trace_uid_span = _canonical_trace_row_uids(
        canonical_expert_token_uids,
        tokens_per_expert=tokens_per_expert,
        local_expert_indices=getattr(dispatcher, "local_expert_indices", None),
        sample_uid_span=int(active_step_routes.global_token_uids.numel()),
        num_experts=int(getattr(dispatcher, "num_experts", 1)),
    )
    return (
        expert_inputs,
        expert_probs,
        inverse_order_cpu,
        trace_row_uids,
        trace_uid_span,
    )


def install_moe_routing_trace_hooks(
    controller_getter: Callable[[], Any | None],
) -> None:
    global _CONTROLLER_GETTER
    _CONTROLLER_GETTER = controller_getter
    _install_router_output_trace_hooks(_active_controller())
    try:
        from megatron.core.transformer.moe.experts import TEGroupedMLP
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
        )

        from art.megatron.lora import MLPExpertsLinearFC2LoRA
    except Exception:
        return

    if hasattr(MoEAlltoAllTokenDispatcher, "_art_oracle_trace_patched"):
        return

    original_preprocess = MoEAlltoAllTokenDispatcher.preprocess
    original_dispatch_preprocess = MoEAlltoAllTokenDispatcher.dispatch_preprocess
    original_token_dispatch = MoEAlltoAllTokenDispatcher.token_dispatch
    original_dispatch_postprocess = MoEAlltoAllTokenDispatcher.dispatch_postprocess
    original_combine_preprocess = MoEAlltoAllTokenDispatcher.combine_preprocess
    original_te_grouped_mlp_forward = TEGroupedMLP.forward
    original_fc2_forward = MLPExpertsLinearFC2LoRA.forward

    def patched_preprocess(
        self: Any, routing_map: torch.Tensor, *args: Any, **kwargs: Any
    ):
        result = original_preprocess(self, routing_map, *args, **kwargs)
        if (
            not getattr(self, "drop_and_pad", False)
            and getattr(self.config, "moe_expert_capacity_factor", None) is None
            and not (
                getattr(self.config, "moe_router_padding_for_quantization", None)
                or getattr(self.config, "moe_router_padding_for_fp8", None)
            )
        ):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    def patched_dispatch_preprocess(
        self: Any,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor,
        probs: torch.Tensor,
    ):
        result = original_dispatch_preprocess(self, hidden_states, routing_map, probs)
        self._art_replay_permuted_local_token_uids = None
        self._art_replay_global_input_token_uids = None
        self._art_replay_expert_input_inverse_permutation = None

        controller = _active_controller()
        if controller is None:
            return result
        local_token_uids = _dispatcher_local_token_uids(
            controller,
            self,
            num_local_tokens=int(routing_map.shape[0]),
        )
        permuted_local_uids = permute(
            local_token_uids.to(
                device=hidden_states.device, dtype=torch.int64
            ).unsqueeze(-1),
            self.routing_map,
            num_out_tokens=self.num_out_tokens,
            fused=False,
            drop_and_pad=self.drop_and_pad,
        )[0]
        self._art_replay_permuted_local_token_uids = permuted_local_uids.reshape(
            -1
        ).contiguous()
        return result

    def patched_token_dispatch(
        self: Any,
        permutated_local_input_tokens: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        result = original_token_dispatch(
            self,
            permutated_local_input_tokens,
            permuted_probs,
        )
        controller = _active_controller()
        permuted_local_token_uids = getattr(
            self, "_art_replay_permuted_local_token_uids", None
        )
        if controller is None or permuted_local_token_uids is None:
            return result

        global_token_uids = permuted_local_token_uids.to(
            device=permutated_local_input_tokens.device, dtype=torch.int64
        ).unsqueeze(-1)
        if self.ep_size > 1:
            global_token_uids = all_to_all(
                self.ep_group,
                global_token_uids,
                self.output_splits,
                self.input_splits,
            )
        if self.tp_size > 1:
            output_split_sizes = (
                None
                if self.output_splits_tp is None
                else self.output_splits_tp.tolist()
            )
            global_token_uids = gather_from_sequence_parallel_region(
                global_token_uids,
                group=self.tp_group,
                output_split_sizes=output_split_sizes,
            )
        self._art_replay_global_input_token_uids = global_token_uids.reshape(
            -1
        ).contiguous()
        return result

    def patched_dispatch_postprocess(
        self: Any,
        global_input_tokens: torch.Tensor,
        global_probs: torch.Tensor,
    ):
        expert_inputs, tokens_per_expert, expert_probs = original_dispatch_postprocess(
            self,
            global_input_tokens,
            global_probs,
        )
        controller = _active_controller()
        global_input_token_uids = getattr(
            self, "_art_replay_global_input_token_uids", None
        )
        if controller is None or global_input_token_uids is None or self.drop_and_pad:
            return expert_inputs, tokens_per_expert, expert_probs

        (
            expert_inputs,
            expert_probs,
            inverse_order_cpu,
            trace_row_uids,
            trace_uid_span,
        ) = _build_dispatch_postprocess_trace(
            dispatcher=self,
            controller=controller,
            global_input_token_uids=global_input_token_uids,
            expert_inputs=expert_inputs,
            expert_probs=expert_probs,
            tokens_per_expert=tokens_per_expert,
        )
        self._art_replay_expert_input_inverse_permutation = inverse_order_cpu
        _attach_trace_row_uids(
            expert_inputs,
            row_token_uids=trace_row_uids,
            uid_span=trace_uid_span,
        )
        return expert_inputs, tokens_per_expert, expert_probs

    def patched_combine_preprocess(self: Any, hidden_states: torch.Tensor):
        inverse_order_cpu = getattr(
            self, "_art_replay_expert_input_inverse_permutation", None
        )
        if inverse_order_cpu is not None and inverse_order_cpu.numel() > 0:
            hidden_states = hidden_states.index_select(
                0,
                inverse_order_cpu.to(device=hidden_states.device, dtype=torch.long),
            )
        self._art_replay_expert_input_inverse_permutation = None
        return original_combine_preprocess(self, hidden_states)

    def patched_te_grouped_mlp_forward(
        self: Any,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        _propagate_grouped_mlp_trace_row_uids(
            permuted_local_hidden_states,
            self.linear_fc2,
        )
        return original_te_grouped_mlp_forward(
            self,
            permuted_local_hidden_states,
            tokens_per_expert,
            permuted_probs,
        )

    def patched_fc2_forward(
        self: Any,
        x: torch.Tensor,
        tokens_per_expert: list[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _propagate_fc2_trace_row_uids(
            x=x,
            module=self,
            linear_fc2=self.linear_fc2,
            lora=self.lora,
        )
        return original_fc2_forward(self, x, tokens_per_expert)

    setattr(MoEAlltoAllTokenDispatcher, "preprocess", patched_preprocess)
    setattr(
        MoEAlltoAllTokenDispatcher,
        "dispatch_preprocess",
        patched_dispatch_preprocess,
    )
    setattr(MoEAlltoAllTokenDispatcher, "token_dispatch", patched_token_dispatch)
    setattr(
        MoEAlltoAllTokenDispatcher,
        "dispatch_postprocess",
        patched_dispatch_postprocess,
    )
    setattr(
        MoEAlltoAllTokenDispatcher,
        "combine_preprocess",
        patched_combine_preprocess,
    )
    setattr(TEGroupedMLP, "forward", patched_te_grouped_mlp_forward)
    setattr(MLPExpertsLinearFC2LoRA, "forward", patched_fc2_forward)
    setattr(MoEAlltoAllTokenDispatcher, "_art_oracle_trace_patched", True)
