from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, cast

import torch

from .trace_uids import (
    TRACE_ROW_TOKEN_UIDS_ATTR,
    TRACE_UID_SPAN_ATTR,
    expand_token_uids_for_heads,
    extract_tensor_attr,
    normalize_row_token_uids,
    row_token_uids_from_trace_sources,
)

CAPTURE_NAME_TOKENS = (
    ".self_attention",
    ".self_attention.in_proj",
    ".self_attention.in_proj.in_proj",
    ".self_attention.in_proj.qkv_lora",
    ".self_attention.in_proj.z_lora",
    ".self_attention.out_norm",
    ".self_attention.out_proj",
    ".self_attention.out_proj.lora",
    ".self_attention.linear_qkv",
    ".self_attention.linear_qkv.q_proj_lora",
    ".self_attention.linear_qkv.k_proj_lora",
    ".self_attention.linear_qkv.v_proj_lora",
    ".self_attention.core_attention",
    ".self_attention.linear_proj",
    ".self_attention.linear_proj.lora",
    ".pre_mlp_layernorm",
    ".mlp",
    ".mlp.router",
    ".mlp.experts.linear_fc1",
    ".mlp.experts.linear_fc1.gate_lora",
    ".mlp.experts.linear_fc1.up_lora",
    ".mlp.experts.linear_fc2",
    ".mlp.experts.linear_fc2.lora",
    ".mlp.linear_fc1",
    ".mlp.linear_fc1.gate_lora",
    ".mlp.linear_fc1.up_lora",
    ".mlp.linear_fc2",
    ".mlp.linear_fc2.row_parallel_lora",
    ".mlp.linear_fc2.row_parallel_lora.lora",
)
ROUTER_NAME_TOKEN = ".mlp.router"
CORE_ATTENTION_NAME_TOKEN = ".self_attention.core_attention"
PRIMARY_OUTPUT_CANONICAL_KEY = "primary_output__is_canonical"
CAPTURE_INPUT_NAME_TOKENS = (
    ".self_attention.linear_qkv",
    ".self_attention.linear_qkv.q_proj_lora",
    ".self_attention.linear_qkv.k_proj_lora",
    ".self_attention.linear_qkv.v_proj_lora",
)


def _module_layer_index(module_name: str) -> int | None:
    marker = "decoder.layers."
    marker_index = module_name.find(marker)
    if marker_index < 0:
        return None
    raw = module_name[marker_index + len(marker) :].split(".", 1)[0]
    return int(raw) if raw.isdigit() else None


def _trace_hook(fn: Callable[..., Any]) -> Callable[..., Any]:
    return torch.compiler.disable(fn)


def _normalize_trace_module_name(module_name: str) -> str:
    """Strips compile-wrapper path segments from trace module names."""
    return module_name.replace("._orig_mod", "")


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerces scalar values to int for trace metadata."""
    try:
        return int(value)
    except Exception:
        return default


def _call_cp_world_size(call: dict[str, Any]) -> int:
    rank_meta = call.get("rank_meta")
    if isinstance(rank_meta, list) and rank_meta:
        rank_meta = rank_meta[0]
    if not isinstance(rank_meta, dict):
        return 1
    return _safe_int(rank_meta.get("cp_world_size"), 1)


def _safe_ps_stat(name: str, default: int) -> int:
    """Reads one Megatron parallel-state integer when available."""
    try:
        from megatron.core import parallel_state as ps

        getter = getattr(ps, name)
        return _safe_int(getter(), default)
    except Exception:
        return default


def _rank_metadata() -> dict[str, int]:
    """Builds lightweight distributed metadata for one trace call."""
    rank = 0
    world_size = 1
    if torch.distributed.is_initialized():  # ty: ignore[possibly-missing-attribute]
        rank = _safe_int(torch.distributed.get_rank(), 0)  # ty: ignore[possibly-missing-attribute]
        world_size = _safe_int(torch.distributed.get_world_size(), 1)  # ty: ignore[possibly-missing-attribute]
    return {
        "global_rank": rank,
        "world_size": world_size,
        "tp_rank": _safe_ps_stat("get_tensor_model_parallel_rank", 0),
        "tp_world_size": _safe_ps_stat("get_tensor_model_parallel_world_size", 1),
        "cp_rank": _safe_ps_stat("get_context_parallel_rank", 0),
        "cp_world_size": _safe_ps_stat("get_context_parallel_world_size", 1),
        "ep_rank": _safe_ps_stat("get_expert_model_parallel_rank", 0),
        "ep_world_size": _safe_ps_stat("get_expert_model_parallel_world_size", 1),
        "etp_rank": _safe_ps_stat("get_expert_tensor_parallel_rank", 0),
        "etp_world_size": _safe_ps_stat("get_expert_tensor_parallel_world_size", 1),
        "dp_rank": _safe_ps_stat("get_data_parallel_rank", 0),
        "dp_world_size": _safe_ps_stat("get_data_parallel_world_size", 1),
        "expert_dp_rank": _safe_ps_stat("get_expert_data_parallel_rank", 0),
        "expert_dp_world_size": _safe_ps_stat("get_expert_data_parallel_world_size", 1),
    }


def _extract_dp_slot_from_rank_meta(rank_meta: Any) -> tuple[int, int] | None:
    """Returns one stable `(dp_rank, dp_world_size)` pair from merged rank metadata."""
    if isinstance(rank_meta, dict):
        rank_meta = [rank_meta]
    if not isinstance(rank_meta, list) or not rank_meta:
        return None
    dp_ranks = {
        _safe_int(item.get("dp_rank"), 0)
        for item in rank_meta
        if isinstance(item, dict) and "dp_rank" in item
    }
    dp_world_sizes = {
        _safe_int(item.get("dp_world_size"), 1)
        for item in rank_meta
        if isinstance(item, dict) and "dp_world_size" in item
    }
    if len(dp_ranks) != 1 or len(dp_world_sizes) != 1:
        return None
    return next(iter(dp_ranks)), next(iter(dp_world_sizes))


def _trace_call_sort_key(call: dict[str, Any]) -> tuple[int, int]:
    """Builds a stable micro identity for merged trace ordering."""
    sample_index = call.get("micro_sample_index")
    if isinstance(sample_index, int):
        return 0, int(sample_index)
    micro_order = _safe_int(call.get("micro_order"), 0)
    dp_slot = _extract_dp_slot_from_rank_meta(call.get("rank_meta"))
    if dp_slot is None:
        return 1, micro_order
    dp_rank, dp_world_size = dp_slot
    return 1, micro_order * dp_world_size + dp_rank


def _local_dummy_micro_slot(micro_order: int) -> int:
    """Builds the stable dummy-micro slot used when one micro has no sample id."""
    dp_rank = _safe_ps_stat("get_data_parallel_rank", 0)
    dp_world_size = _safe_ps_stat("get_data_parallel_world_size", 1)
    return micro_order * dp_world_size + dp_rank


def _captured_output_sort_key(
    sample_index: int | None,
    micro_order: int,
    micro_slot: int | None,
) -> tuple[int, int, int]:
    """Builds the deterministic ordering used for captured root outputs."""
    if isinstance(sample_index, int):
        return 0, int(sample_index), micro_order
    return 1, _safe_int(micro_slot, micro_order), 0


def _shard_world_size_for_domain(domain: Any) -> int:
    """Returns shard-group world size for one LoRA shard domain."""
    if domain == "tp":
        return _safe_ps_stat("get_tensor_model_parallel_world_size", 1)
    if domain == "expert_tp":
        return _safe_ps_stat("get_expert_tensor_parallel_world_size", 1)
    return 1


def _world_size_key_for_domain(domain: Any) -> str | None:
    if domain == "tp":
        return "tp_world_size"
    if domain == "expert_tp":
        return "etp_world_size"
    return None


def _extract_primary_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_primary_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_primary_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(tensor, "full_tensor", None)
    if callable(full_tensor):
        tensor = cast(torch.Tensor, full_tensor())
    else:
        to_local = getattr(tensor, "to_local", None)
        if callable(to_local):
            tensor = cast(torch.Tensor, to_local())
        else:
            local_tensor = getattr(tensor, "_local_tensor", None)
            if isinstance(local_tensor, torch.Tensor):
                tensor = local_tensor
    return tensor.detach().cpu()


def _extract_router_topk(
    output: Any, *, topk_hint: int | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(output, tuple) or len(output) < 2:
        return None
    probs = output[0]
    routing_map = output[1]
    if not isinstance(probs, torch.Tensor) or not isinstance(routing_map, torch.Tensor):
        return None
    probs = _materialize_tensor(probs.float())
    routing_map = _materialize_tensor(routing_map)
    if int(routing_map.shape[0]) == 0:
        topk = int(topk_hint or 0)
    else:
        topk = int(routing_map.sum(dim=-1).max().item())
    if topk < 0:
        raise RuntimeError(f"Invalid router topk={topk}")
    if topk == 0:
        topk_scores = probs.new_zeros((probs.shape[0], 0))
        topk_ids = torch.zeros((probs.shape[0], 0), dtype=torch.int64)
    else:
        topk_scores, topk_ids = torch.topk(probs, k=topk, dim=-1)
    return topk_ids.contiguous(), topk_scores.contiguous()


def _extract_router_output(output: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(output, tuple) or len(output) < 2:
        return None
    probs = output[0]
    routing_map = output[1]
    if not isinstance(probs, torch.Tensor) or not isinstance(routing_map, torch.Tensor):
        return None
    return {
        "probs": _materialize_tensor(probs.float()),
        "routing_map": _materialize_tensor(routing_map.bool()),
    }


def _extract_core_attention_inputs(inputs: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(inputs, tuple) or len(inputs) < 3:
        return None
    query, key, value = inputs[:3]
    if not (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        return None
    return {
        "core_attention_query": _materialize_tensor(query),
        "core_attention_key": _materialize_tensor(key),
        "core_attention_value": _materialize_tensor(value),
    }


class ForwardTraceCapture:
    def __init__(
        self,
        model_chunks: list[Any],
        *,
        enabled: bool,
        capture_name_tokens: tuple[str, ...] = CAPTURE_NAME_TOKENS,
        capture_layer_outputs: bool = True,
        max_layer_index: int | None = None,
        micro_start_callback: Callable[[int | None, int], None] | None = None,
        strict_output_match: bool = True,
    ) -> None:
        self.enabled = enabled
        self.capture_name_tokens = capture_name_tokens
        self.capture_layer_outputs = capture_layer_outputs
        self.max_layer_index = max_layer_index
        self.micro_start_callback = micro_start_callback
        self.strict_output_match = strict_output_match
        self.current_step_index: int | None = None
        self.current_step_trace: dict[str, list[dict[str, Any]]] = {}
        self.current_micro_sample_index: int | None = None
        self.current_micro_order = 0
        self.current_micro_module_call_counts: dict[str, int] = {}
        self.current_step_sample_indices: list[int | None] = []
        self.current_step_outputs: list[
            tuple[int | None, int, int | None, torch.Tensor, torch.Tensor | None]
        ] = []
        self._trace_metadata_by_name: dict[str, dict[str, Any]] = {}
        self._next_micro_order = 0
        self._inside_root_forward = False
        self._hook_handles: list[Any] = []
        if not enabled:
            return
        self._register_hooks(model_chunks)

    def _register_hooks(self, model_chunks: list[Any]) -> None:
        if not model_chunks:
            raise RuntimeError("Expected at least one model chunk for forward tracing")
        root_module = model_chunks[0]
        self._hook_handles.append(
            root_module.register_forward_pre_hook(_trace_hook(self._root_pre_hook))
        )
        self._hook_handles.append(
            root_module.register_forward_hook(_trace_hook(self._root_post_hook))
        )
        for chunk_index, chunk in enumerate(model_chunks):
            named_modules = list(chunk.named_modules())
            module_by_name = dict(named_modules)
            for module_name, module in named_modules:
                layer_index = _module_layer_index(module_name)
                if (
                    self.max_layer_index is not None
                    and layer_index is not None
                    and layer_index > self.max_layer_index
                ):
                    continue
                trace_module_name = _normalize_trace_module_name(
                    f"chunk{chunk_index}.{module_name}"
                )
                metadata = self._build_module_trace_metadata(
                    module_name=module_name,
                    module=module,
                    module_by_name=module_by_name,
                )
                if metadata:
                    self._trace_metadata_by_name[trace_module_name] = metadata
                is_layer_output = (
                    self.capture_layer_outputs
                    and ".decoder.layers." in module_name
                    and module_name.rsplit(".", 1)[-1].isdigit()
                )
                if not is_layer_output and not any(
                    module_name.endswith(token) for token in self.capture_name_tokens
                ):
                    continue
                self._hook_handles.append(
                    module.register_forward_hook(
                        _trace_hook(self._make_hook(trace_module_name, module))
                    )
                )

    @classmethod
    def _build_module_trace_metadata(
        cls,
        *,
        module_name: str,
        module: Any,
        module_by_name: dict[str, Any],
    ) -> dict[str, Any]:
        if module_name.endswith(".self_attention.in_proj"):
            sizes = cls._gdn_in_proj_component_sizes(module)
            return {"component_sizes": sizes} if sizes is not None else {}
        if module_name.endswith(".self_attention.in_proj.in_proj"):
            parent_module = module_by_name[module_name.rsplit(".", 1)[0]]
            sizes = cls._gdn_in_proj_component_sizes(parent_module)
            return {"component_sizes": sizes} if sizes is not None else {}
        if module_name.endswith(".self_attention.out_norm"):
            gdn_module = module_by_name[module_name.removesuffix(".out_norm")]
            return {
                "local_heads": int(gdn_module.num_value_heads // gdn_module.tp_size),
            }
        return {}

    @staticmethod
    def _gdn_in_proj_component_sizes(module: Any) -> tuple[int, ...] | None:
        if not hasattr(module, "qkv_lora") or not hasattr(module, "z_lora"):
            return None
        qkv_sizes = tuple(
            int(size)
            for size in getattr(module.qkv_lora.B_T, "lora_tp_component_sizes")
        )
        z_world_size = _shard_world_size_for_domain(module.z_lora.B_T.lora_shard_domain)
        tp_world_size = _safe_ps_stat("get_tensor_model_parallel_world_size", 1)
        return (
            *qkv_sizes,
            int(module.z_lora.B_T.shape[-1]) * z_world_size,
            int(module.num_value_heads_per_partition) * tp_world_size,
            int(module.num_value_heads_per_partition) * tp_world_size,
        )

    @staticmethod
    def _sequence_parallel_enabled(module: Any) -> bool:
        """Returns sequence-parallel flag from module/provider/config when present."""
        for owner in (
            module,
            getattr(module, "provider", None),
            getattr(module, "config", None),
        ):
            if owner is None:
                continue
            value = getattr(owner, "sequence_parallel", None)
            if isinstance(value, bool):
                return value
        return False

    @staticmethod
    def _lora_primary_output_merge_hint(module: Any) -> dict[str, Any] | None:
        """Infers the correct output merge op for LoRA modules."""
        if module.__class__.__name__ != "LoRA":
            return None
        lora_module = module
        b_param = getattr(lora_module, "B_T", None)
        if b_param is None:
            return None
        b_domain = getattr(b_param, "lora_shard_domain", None)
        b_world_size = _shard_world_size_for_domain(b_domain)
        if bool(getattr(b_param, "lora_tp_sharded", False)) and b_world_size > 1:
            shard_dim = getattr(b_param, "lora_tp_shard_dim", None)
            if isinstance(shard_dim, int):
                hint: dict[str, Any] = {"op": "concat", "dim": shard_dim}
                component_sizes = tuple(
                    int(size)
                    for size in getattr(b_param, "lora_tp_component_sizes", ())
                )
                world_size_key = _world_size_key_for_domain(b_domain)
                if component_sizes and world_size_key is not None:
                    hint.update(
                        {
                            "layout": "componentwise",
                            "component_sizes": component_sizes,
                            "world_size_key": world_size_key,
                        }
                    )
                return hint
        a_param = getattr(lora_module, "A_T", None)
        if a_param is None:
            return None
        a_domain = getattr(a_param, "lora_shard_domain", None)
        a_world_size = _shard_world_size_for_domain(a_domain)
        if bool(getattr(a_param, "lora_tp_sharded", False)) and a_world_size > 1:
            return {"op": "sum"}
        return None

    def _infer_primary_output_merge_hint(
        self, name: str, module: Any
    ) -> dict[str, Any] | None:
        """Chooses canonical cross-rank concat axis for one module output."""
        if ROUTER_NAME_TOKEN in name:
            return {"op": "concat", "dim": 0}

        lora_hint = self._lora_primary_output_merge_hint(module)
        if lora_hint is not None:
            return lora_hint

        trace_metadata = self._trace_metadata_by_name.get(name, {})
        component_sizes = trace_metadata.get("component_sizes")
        if isinstance(component_sizes, tuple) and component_sizes:
            return {
                "op": "concat",
                "dim": -1,
                "layout": "componentwise",
                "component_sizes": component_sizes,
                "world_size_key": "tp_world_size",
            }

        local_heads = trace_metadata.get("local_heads")
        if isinstance(local_heads, int) and local_heads > 0:
            return {
                "op": "concat",
                "dim": 0,
                "layout": "rank_blocked_token_heads",
                "local_heads": local_heads,
                "world_size_key": "tp_world_size",
            }

        # Base MoE expert linears need expert-TP aware merge semantics.
        # With etp>1:
        # - FC1 (column-parallel) shards output features -> concat on feature dim.
        # - FC2 (row-parallel) emits partial output contributions -> sum across ranks.
        # With etp==1, keep the existing token-row concat behavior.
        etp_world_size = _safe_ps_stat("get_expert_tensor_parallel_world_size", 1)
        if ".mlp.experts.linear_fc1" in name and ".lora" not in name:
            if etp_world_size > 1:
                return {
                    "op": "concat",
                    "dim": -1,
                    "layout": "gate_up_rank_interleaved",
                    "world_size_key": "etp_world_size",
                }
            return {"op": "concat", "dim": 0}
        if ".mlp.experts.linear_fc2" in name and ".lora" not in name:
            if etp_world_size > 1:
                return {"op": "sum"}
            return {"op": "concat", "dim": 0}

        gather_output = getattr(module, "gather_output", None)
        if isinstance(gather_output, bool) and not gather_output:
            return {"op": "concat", "dim": -1}

        if ".self_attention.linear_qkv" in name:
            return {"op": "concat", "dim": -1}
        if ".self_attention.core_attention" in name:
            return {"op": "concat", "dim": -1}
        if name.endswith(".self_attention.in_proj"):
            return {"op": "concat", "dim": -1}
        if name.endswith(
            ".self_attention.out_proj"
        ) and self._sequence_parallel_enabled(module):
            return {"op": "concat", "dim": 0}
        if name.endswith(".self_attention") and self._sequence_parallel_enabled(module):
            return {"op": "concat", "dim": 0}

        if ".mlp.linear_fc1" in name and ".lora" not in name:
            tp_world_size = _safe_ps_stat("get_tensor_model_parallel_world_size", 1)
            if tp_world_size > 1:
                return {
                    "op": "concat",
                    "dim": -1,
                    "layout": "gate_up_rank_interleaved",
                    "world_size_key": "tp_world_size",
                }
            return {"op": "concat", "dim": -1}
        if ".mlp.linear_fc2.row_parallel_lora" in name and ".lora" not in name:
            if self._sequence_parallel_enabled(module):
                return {"op": "concat", "dim": 0}
            return None
        if ".mlp.linear_fc2" in name and ".lora" not in name:
            row_parallel_lora = getattr(module, "row_parallel_lora", None)
            if row_parallel_lora is not None and self._sequence_parallel_enabled(
                row_parallel_lora
            ):
                return {"op": "concat", "dim": 0}
            return None

        if ".mlp.experts." in name:
            return {"op": "concat", "dim": 0}

        if bool(
            getattr(module, "input_is_parallel", False)
        ) and self._sequence_parallel_enabled(module):
            return {"op": "concat", "dim": 0}

        return None

    def _build_merge_hints(self, name: str, module: Any) -> dict[str, dict[str, Any]]:
        """Builds field-level tensor merge hints for one call record."""
        hints: dict[str, dict[str, Any]] = {}
        primary_output_hint = self._infer_primary_output_merge_hint(name, module)
        if primary_output_hint is not None:
            hints["primary_output"] = primary_output_hint
        if ROUTER_NAME_TOKEN in name:
            concat_dim0 = {"op": "concat", "dim": 0}
            hints["output"] = concat_dim0
            hints["router_topk_ids"] = concat_dim0
            hints["router_topk_scores"] = concat_dim0
        if CORE_ATTENTION_NAME_TOKEN in name:
            concat_heads = {"op": "concat", "dim": 2}
            hints["core_attention_query"] = concat_heads
            hints["core_attention_key"] = concat_heads
            hints["core_attention_value"] = concat_heads
        return hints

    def _make_hook(self, name: str, module: Any):
        def _hook(_module: Any, inputs: Any, output: Any) -> None:
            if self.current_step_index is None or not self._inside_root_forward:
                return
            micro_call_index = self.current_micro_module_call_counts.get(name, 0)
            self.current_micro_module_call_counts[name] = micro_call_index + 1
            trace_item: dict[str, Any] = {
                "micro_call_index": micro_call_index,
                "micro_order": self.current_micro_order,
                "micro_sample_index": self.current_micro_sample_index,
                "module_type": module.__class__.__name__,
                "rank_meta": _rank_metadata(),
                "merge_hints": self._build_merge_hints(name, module),
                # Keep live trace capture passive. Recursively materializing full
                # hook inputs/outputs here performs large device-to-host copies and
                # previously perturbed correctness in the real training forward.
                "primary_output": self.guess_primary_tensor(output),
            }
            if os.environ.get(
                "ART_MEGATRON_FORWARD_TRACE_CAPTURE_INPUTS"
            ) == "1" and any(
                name.endswith(token) for token in CAPTURE_INPUT_NAME_TOKENS
            ):
                primary_input = self.guess_primary_tensor(inputs)
                if primary_input is not None:
                    trace_item["primary_input"] = primary_input
            if ROUTER_NAME_TOKEN in name:
                router_output = _extract_router_output(output)
                if router_output is not None:
                    trace_item["output"] = router_output
                topk_hint = getattr(
                    getattr(module, "config", None), "moe_router_topk", None
                )
                router_topk = _extract_router_topk(
                    output,
                    topk_hint=int(topk_hint) if topk_hint is not None else None,
                )
                if router_topk is not None:
                    topk_ids, topk_scores = router_topk
                    trace_item["router_topk_ids"] = topk_ids
                    trace_item["router_topk_scores"] = topk_scores
            if CORE_ATTENTION_NAME_TOKEN in name:
                core_inputs = _extract_core_attention_inputs(inputs)
                if core_inputs is not None:
                    trace_item.update(core_inputs)
            primary_output = trace_item.get("primary_output")
            primary_row_count = (
                int(primary_output.shape[0])
                if isinstance(primary_output, torch.Tensor) and primary_output.ndim > 0
                else None
            )
            row_token_uids, _uid_span = self._row_token_uids_for_capture(
                module_name=name,
                inputs=inputs,
                output=output,
                module=module,
                row_count=primary_row_count,
            )
            if (
                isinstance(primary_output, torch.Tensor)
                and primary_output.ndim > 0
                and isinstance(row_token_uids, torch.Tensor)
                and int(row_token_uids.numel()) == int(primary_output.shape[0])
            ):
                trace_item["row_token_uids"] = row_token_uids
                if isinstance(_uid_span, int) and _uid_span > 0:
                    trace_item["row_uid_span"] = int(_uid_span)
            trace_items = self._split_expert_trace_items(
                module_name=name,
                module=module,
                inputs=inputs,
                trace_item=trace_item,
            )
            trace_calls = self.current_step_trace.setdefault(name, [])
            for split_item in trace_items:
                split_item["call_index"] = len(trace_calls)
                trace_calls.append(split_item)

        return _hook

    @staticmethod
    def guess_primary_tensor(value: Any) -> torch.Tensor | None:
        tensor = _extract_primary_tensor(value)
        if tensor is None:
            return None
        return _materialize_tensor(tensor)

    def _sample_index_for_micro(self, micro_order: int) -> int | None:
        if micro_order < len(self.current_step_sample_indices):
            return self.current_step_sample_indices[micro_order]
        return None

    def _root_pre_hook(self, _module: Any, _args: Any) -> None:
        if self.current_step_index is None:
            return
        self._inside_root_forward = True
        micro_order = self._next_micro_order
        sample_index = self._sample_index_for_micro(micro_order)
        self.begin_micro(sample_index=sample_index, micro_order=micro_order)

    def _root_post_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        if self.current_step_index is None:
            return
        output_tensor = self.guess_primary_tensor(output)
        if output_tensor is None:
            raise RuntimeError(
                f"Expected root forward output to contain a tensor, got {type(output)}"
            )
        sample_index = self.current_micro_sample_index
        micro_order = self.current_micro_order
        self.current_step_outputs.append(
            (
                sample_index,
                micro_order,
                None
                if sample_index is not None
                else _local_dummy_micro_slot(micro_order),
                output_tensor.float(),
                getattr(_module, "_art_root_output_token_uids", None),
            )
        )
        self._next_micro_order = micro_order + 1
        self._inside_root_forward = False

    def set_step(
        self,
        step_index: int,
        sample_indices: list[int | None] | None = None,
    ) -> None:
        self.current_step_index = step_index
        self.current_step_trace = {}
        self.current_step_sample_indices = list(sample_indices or [])
        self.current_step_outputs = []
        self.current_micro_sample_index = None
        self.current_micro_order = 0
        self.current_micro_module_call_counts = {}
        self._next_micro_order = 0
        self._inside_root_forward = False

    def begin_micro(self, sample_index: int | None, micro_order: int) -> None:
        self.current_micro_sample_index = sample_index
        self.current_micro_order = micro_order
        self.current_micro_module_call_counts = {}
        if self.micro_start_callback is not None:
            self.micro_start_callback(sample_index, micro_order)

    @staticmethod
    def _row_token_uids_for_trace(
        *,
        inputs: Any,
        output: Any = None,
        module: Any,
        row_count: int | None = None,
        prefer_uid_span: bool = False,
    ) -> tuple[torch.Tensor | None, int | None]:
        return row_token_uids_from_trace_sources(
            inputs=inputs,
            output=output,
            module=module,
            row_count=row_count,
            prefer_uid_span=prefer_uid_span,
        )

    @staticmethod
    def _module_row_token_uids(
        module: Any,
        *,
        row_count: int,
    ) -> tuple[torch.Tensor | None, int | None]:
        row_token_uids = normalize_row_token_uids(
            getattr(module, TRACE_ROW_TOKEN_UIDS_ATTR, None)
        )
        if row_token_uids is None or int(row_token_uids.numel()) != int(row_count):
            return None, None
        uid_span = getattr(module, TRACE_UID_SPAN_ATTR, None)
        return (
            row_token_uids,
            uid_span if isinstance(uid_span, int) and uid_span > 0 else None,
        )

    @classmethod
    def _row_token_uids_for_capture(
        cls,
        *,
        module_name: str,
        inputs: Any,
        output: Any,
        module: Any,
        row_count: int | None,
    ) -> tuple[torch.Tensor | None, int | None]:
        if row_count is not None and not cls._is_moe_expert_forward_module(module_name):
            row_token_uids, uid_span = cls._module_row_token_uids(
                module,
                row_count=row_count,
            )
            if row_token_uids is not None:
                return row_token_uids, uid_span
        return cls._row_token_uids_for_trace(
            inputs=inputs,
            output=output,
            module=module,
            row_count=row_count,
        )

    @classmethod
    def _slice_row_aligned_value(
        cls,
        value: Any,
        *,
        row_indices: torch.Tensor,
        total_rows: int,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == total_rows:
                return value.index_select(0, row_indices)
            return value
        if isinstance(value, dict):
            return {
                key: cls._slice_row_aligned_value(
                    item,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._slice_row_aligned_value(
                    item,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                cls._slice_row_aligned_value(
                    item,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for item in value
            )
        return value

    @classmethod
    def _split_expert_trace_items(
        cls,
        *,
        module_name: str,
        module: Any,
        inputs: Any,
        trace_item: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not cls._is_moe_expert_forward_module(module_name):
            return [trace_item]

        primary_output = trace_item.get("primary_output")
        if not isinstance(primary_output, torch.Tensor) or primary_output.ndim == 0:
            return [trace_item]

        row_token_uids, uid_span = cls._row_token_uids_for_trace(
            inputs=inputs,
            module=module,
            row_count=int(primary_output.shape[0]),
            prefer_uid_span=True,
        )
        if row_token_uids is None:
            return [trace_item]

        total_rows = int(row_token_uids.numel())
        if total_rows == 0 or int(primary_output.shape[0]) != total_rows:
            return [trace_item]

        valid_rows = torch.nonzero(row_token_uids >= 0, as_tuple=False).reshape(-1)
        if int(valid_rows.numel()) == 0:
            return []
        if int(valid_rows.numel()) != total_rows:
            trace_item = {
                key: cls._slice_row_aligned_value(
                    value,
                    row_indices=valid_rows,
                    total_rows=total_rows,
                )
                for key, value in trace_item.items()
                if key not in {"call_index", "micro_sample_index", "row_token_uids"}
            }
            row_token_uids = row_token_uids.index_select(0, valid_rows)
            total_rows = int(row_token_uids.numel())

        trace_item["row_token_uids"] = row_token_uids
        if uid_span is None:
            return [trace_item]
        trace_item["row_uid_span"] = int(uid_span)

        sample_ids = torch.div(row_token_uids, uid_span, rounding_mode="floor")
        ordered_sample_ids: list[int] = []
        seen_sample_ids: set[int] = set()
        for sample_id in sample_ids.tolist():
            sample_id_int = int(sample_id)
            if sample_id_int in seen_sample_ids:
                continue
            seen_sample_ids.add(sample_id_int)
            ordered_sample_ids.append(sample_id_int)

        if len(ordered_sample_ids) <= 1:
            if ordered_sample_ids and not isinstance(
                trace_item.get("micro_sample_index"), int
            ):
                trace_item["micro_sample_index"] = ordered_sample_ids[0]
            return [trace_item]

        split_items: list[dict[str, Any]] = []
        for sample_id in ordered_sample_ids:
            row_indices = (sample_ids == sample_id).nonzero(as_tuple=False).reshape(-1)
            split_item = {
                key: cls._slice_row_aligned_value(
                    value,
                    row_indices=row_indices,
                    total_rows=total_rows,
                )
                for key, value in trace_item.items()
                if key not in {"call_index", "micro_sample_index", "row_token_uids"}
            }
            split_item["micro_sample_index"] = sample_id
            split_item["row_token_uids"] = row_token_uids.index_select(0, row_indices)
            split_item["row_uid_span"] = int(uid_span)
            split_items.append(split_item)
        return split_items

    @staticmethod
    def _is_moe_expert_forward_module(module_name: str) -> bool:
        """Returns whether one module emits MoE expert forward outputs."""
        if ".mlp.experts." not in module_name:
            return False
        if ".mlp.router" in module_name:
            return False
        return ".linear_fc1" in module_name or ".linear_fc2" in module_name

    @staticmethod
    def _primary_output_merge_hint(call: dict[str, Any]) -> dict[str, Any] | None:
        """Reads primary-output merge metadata from one call payload."""
        merge_hints = call.get("merge_hints")
        if not isinstance(merge_hints, dict):
            return None
        primary_hint = merge_hints.get("primary_output")
        if not isinstance(primary_hint, dict):
            return None
        return primary_hint

    @classmethod
    def _canonicalize_gate_up_rank_interleaved_feature_layout(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Normalizes TP/ETP fused gate-up fc1 output feature order."""
        del module_name
        primary_hint = cls._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return tensor
        if primary_hint.get("layout") != "gate_up_rank_interleaved":
            return tensor
        world_size_key = primary_hint.get("world_size_key")
        if not isinstance(world_size_key, str):
            raise RuntimeError("gate_up_rank_interleaved hint requires world_size_key")
        rank_meta = call.get("rank_meta")
        rank_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                rank_world_size = first_meta.get(world_size_key)
        elif isinstance(rank_meta, dict):
            rank_world_size = rank_meta.get(world_size_key)
        if not isinstance(rank_world_size, int) or rank_world_size <= 1:
            return tensor
        block_count = 2 * rank_world_size
        if tensor.ndim < 1 or tensor.shape[-1] % block_count != 0:
            raise RuntimeError(
                "gate_up_rank_interleaved tensor feature size must divide by "
                f"{block_count}, got shape={tuple(tensor.shape)}"
            )
        blocks = torch.chunk(tensor, block_count, dim=-1)
        reordered = [blocks[index] for index in range(0, block_count, 2)] + [
            blocks[index] for index in range(1, block_count, 2)
        ]
        return torch.cat(reordered, dim=-1).contiguous()

    @classmethod
    def _canonicalize_componentwise_feature_layout(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Normalizes fused componentwise TP output order, e.g. GDN q/k/v."""
        del module_name
        primary_hint = cls._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return tensor
        if primary_hint.get("layout") != "componentwise":
            return tensor
        dim = primary_hint.get("dim")
        component_sizes = primary_hint.get("component_sizes")
        world_size_key = primary_hint.get("world_size_key")
        if not isinstance(dim, int) or not isinstance(world_size_key, str):
            raise RuntimeError("componentwise hint requires dim and world_size_key")
        if not isinstance(component_sizes, tuple) or not all(
            isinstance(size, int) and size > 0 for size in component_sizes
        ):
            raise RuntimeError("componentwise hint requires positive component sizes")
        rank_meta = call.get("rank_meta")
        rank_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                rank_world_size = first_meta.get(world_size_key)
        elif isinstance(rank_meta, dict):
            rank_world_size = rank_meta.get(world_size_key)
        if not isinstance(rank_world_size, int) or rank_world_size <= 1:
            return tensor
        axis = dim if dim >= 0 else tensor.ndim + dim
        if axis < 0 or axis >= tensor.ndim:
            raise RuntimeError(
                f"Invalid componentwise axis {dim} for {tensor.ndim}D tensor"
            )
        if sum(component_sizes) != tensor.shape[axis]:
            raise RuntimeError(
                "componentwise component sizes must match tensor extent, got "
                f"sizes={component_sizes} shape={tuple(tensor.shape)} axis={axis}"
            )
        if any(size % rank_world_size != 0 for size in component_sizes):
            raise RuntimeError(
                "componentwise component sizes must divide rank world size, got "
                f"sizes={component_sizes} world_size={rank_world_size}"
            )
        local_sizes = [size // rank_world_size for size in component_sizes]
        rank_chunks: list[list[torch.Tensor]] = []
        cursor = 0
        for _rank in range(rank_world_size):
            rank_components = []
            for local_size in local_sizes:
                rank_components.append(tensor.narrow(axis, cursor, local_size))
                cursor += local_size
            rank_chunks.append(rank_components)
        ordered = [
            rank_chunks[rank][component_index]
            for component_index in range(len(component_sizes))
            for rank in range(rank_world_size)
        ]
        return torch.cat(ordered, dim=axis).contiguous()

    @classmethod
    def _canonicalize_rank_blocked_token_heads(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        del module_name
        primary_hint = cls._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return tensor
        if primary_hint.get("layout") != "rank_blocked_token_heads":
            return tensor
        local_heads = primary_hint.get("local_heads")
        world_size_key = primary_hint.get("world_size_key")
        if not isinstance(local_heads, int) or local_heads <= 0:
            raise RuntimeError("rank_blocked_token_heads hint requires local_heads")
        if not isinstance(world_size_key, str):
            raise RuntimeError("rank_blocked_token_heads hint requires world_size_key")
        rank_meta = call.get("rank_meta")
        rank_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                rank_world_size = first_meta.get(world_size_key)
        elif isinstance(rank_meta, dict):
            rank_world_size = rank_meta.get(world_size_key)
        if not isinstance(rank_world_size, int) or rank_world_size <= 1:
            return tensor
        if tensor.ndim != 2:
            raise RuntimeError(
                "rank_blocked_token_heads expects a 2D [rows, head_dim] tensor, "
                f"got shape={tuple(tensor.shape)}"
            )
        tensor = cls._canonicalize_rank_blocked_rows(
            tensor,
            local_heads=local_heads,
            rank_world_size=rank_world_size,
            call=call,
        )
        row_token_uids = call.get("row_token_uids")
        if (
            isinstance(row_token_uids, torch.Tensor)
            and row_token_uids.ndim == 1
            and int(row_token_uids.numel()) == int(tensor.shape[0])
        ):
            call["row_token_uids"] = cls._canonicalize_rank_blocked_rows(
                row_token_uids,
                local_heads=local_heads,
                rank_world_size=rank_world_size,
                call=call,
            )
        return tensor

    @classmethod
    def _canonicalize_rank_blocked_rows(
        cls,
        tensor: torch.Tensor,
        *,
        local_heads: int,
        rank_world_size: int,
        call: dict[str, Any],
    ) -> torch.Tensor:
        rank_meta = call.get("rank_meta")
        row_splits = call.get("primary_output__row_splits")
        if (
            isinstance(rank_meta, list)
            and isinstance(row_splits, list)
            and len(rank_meta) == len(row_splits)
            and sum(int(split) for split in row_splits) == int(tensor.shape[0])
        ):
            chunks = list(torch.split(tensor, [int(split) for split in row_splits], 0))
            grouped_indices: dict[int, list[int]] = {}
            for index, meta in enumerate(rank_meta):
                if not isinstance(meta, dict):
                    return cls._canonicalize_rank_blocked_rows_without_splits(
                        tensor,
                        local_heads=local_heads,
                        rank_world_size=rank_world_size,
                    )
                cp_rank = _safe_int(meta.get("cp_rank"), 0)
                grouped_indices.setdefault(cp_rank, []).append(index)
            ordered_groups: list[torch.Tensor] = []
            for cp_rank in sorted(grouped_indices):
                group_indices = sorted(
                    grouped_indices[cp_rank],
                    key=lambda index: _safe_int(
                        cast(dict[str, Any], rank_meta[index]).get("tp_rank"),
                        index,
                    ),
                )
                group_chunks = [chunks[index] for index in group_indices]
                canonical_group = cls._canonicalize_rank_blocked_group_rows(
                    group_chunks,
                    local_heads=local_heads,
                    rank_world_size=rank_world_size,
                )
                ordered_groups.append(canonical_group)
            return torch.cat(ordered_groups, dim=0).contiguous()
        return cls._canonicalize_rank_blocked_rows_without_splits(
            tensor,
            local_heads=local_heads,
            rank_world_size=rank_world_size,
        )

    @staticmethod
    def _canonicalize_rank_blocked_group_rows(
        chunks: list[torch.Tensor],
        *,
        local_heads: int,
        rank_world_size: int,
    ) -> torch.Tensor:
        if len(chunks) != rank_world_size:
            raise RuntimeError(
                "rank_blocked_token_heads expected one chunk per tensor-parallel "
                f"rank, got {len(chunks)} for world_size={rank_world_size}"
            )
        rows_per_rank = int(chunks[0].shape[0])
        if any(int(chunk.shape[0]) != rows_per_rank for chunk in chunks[1:]):
            raise RuntimeError(
                "rank_blocked_token_heads CP group rank chunks must have equal rows"
            )
        token_count, head_remainder = divmod(rows_per_rank, local_heads)
        if head_remainder != 0:
            raise RuntimeError(
                "rank_blocked_token_heads rows per rank must divide local_heads, got "
                f"rows_per_rank={rows_per_rank} local_heads={local_heads}"
            )
        if rows_per_rank == 0:
            return torch.cat(chunks, dim=0).contiguous()
        grouped = torch.cat(chunks, dim=0)
        tail_shape = tuple(grouped.shape[1:])
        return (
            grouped.reshape(rank_world_size, token_count, local_heads, *tail_shape)
            .permute(1, 0, 2, *range(3, 3 + len(tail_shape)))
            .reshape(rows_per_rank * rank_world_size, *tail_shape)
            .contiguous()
        )

    @staticmethod
    def _canonicalize_rank_blocked_rows_without_splits(
        tensor: torch.Tensor,
        *,
        local_heads: int,
        rank_world_size: int,
    ) -> torch.Tensor:
        rows_per_rank, remainder = divmod(int(tensor.shape[0]), rank_world_size)
        if remainder != 0:
            raise RuntimeError(
                "rank_blocked_token_heads rows must divide rank world size, got "
                f"shape={tuple(tensor.shape)} world_size={rank_world_size}"
            )
        token_count, head_remainder = divmod(rows_per_rank, local_heads)
        if head_remainder != 0:
            raise RuntimeError(
                "rank_blocked_token_heads rows per rank must divide local_heads, got "
                f"rows_per_rank={rows_per_rank} local_heads={local_heads}"
            )
        tail_shape = tuple(tensor.shape[1:])
        return (
            tensor.reshape(rank_world_size, token_count, local_heads, *tail_shape)
            .permute(1, 0, 2, *range(3, 3 + len(tail_shape)))
            .reshape(tensor.shape)
            .contiguous()
        )

    @classmethod
    def _canonicalize_row_aligned_value(
        cls,
        value: Any,
        *,
        order: torch.Tensor,
        total_rows: int,
    ) -> Any:
        """Applies one row-token ordering to every row-aligned tensor value."""
        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and int(value.shape[0]) == total_rows:
                return value.index_select(0, order).contiguous()
            return value
        if isinstance(value, dict):
            return {
                key: cls._canonicalize_row_aligned_value(
                    item,
                    order=order,
                    total_rows=total_rows,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._canonicalize_row_aligned_value(
                    item,
                    order=order,
                    total_rows=total_rows,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                cls._canonicalize_row_aligned_value(
                    item,
                    order=order,
                    total_rows=total_rows,
                )
                for item in value
            )
        return value

    @classmethod
    def _canonicalize_call_row_token_order(cls, call: dict[str, Any]) -> None:
        """Canonicalizes all row-aligned call tensors to global token order."""
        cls._align_exact_zero_padding_row_token_uids(call)
        row_token_uids = call.get("row_token_uids")
        if not isinstance(row_token_uids, torch.Tensor) or row_token_uids.ndim != 1:
            return
        total_rows = int(row_token_uids.numel())
        if total_rows <= 1:
            return
        order = torch.argsort(row_token_uids, stable=True)
        if bool(torch.equal(order, torch.arange(order.numel(), dtype=order.dtype))):
            return
        original_call = dict(call)
        for key, value in original_call.items():
            if key == "row_token_uids":
                continue
            call[key] = cls._canonicalize_row_aligned_value(
                value,
                order=order,
                total_rows=total_rows,
            )
        call["row_token_uids"] = row_token_uids.index_select(0, order).contiguous()

    @staticmethod
    def _align_exact_zero_padding_row_token_uids(call: dict[str, Any]) -> None:
        """Moves padding UID markers onto exact-zero sequence-parallel pad rows."""
        row_token_uids = call.get("row_token_uids")
        tensor = call.get("primary_output")
        if (
            not isinstance(row_token_uids, torch.Tensor)
            or row_token_uids.ndim != 1
            or not isinstance(tensor, torch.Tensor)
            or tensor.ndim == 0
            or int(tensor.shape[0]) != int(row_token_uids.numel())
        ):
            return
        row_count = int(row_token_uids.numel())
        if row_count <= 1 or not bool((row_token_uids < 0).any().item()):
            return
        flat = tensor.detach().reshape(row_count, -1)
        zero_rows = torch.nonzero(
            (flat == 0).all(dim=1) & (row_token_uids >= 0),
            as_tuple=False,
        ).reshape(-1)
        negative_rows = torch.nonzero(
            (row_token_uids < 0) & ~(flat == 0).all(dim=1),
            as_tuple=False,
        ).reshape(-1)
        if int(zero_rows.numel()) == 0 or int(zero_rows.numel()) != int(
            negative_rows.numel()
        ):
            return
        aligned = row_token_uids.clone()
        for zero_pos, negative_pos in zip(
            zero_rows.tolist(), negative_rows.tolist(), strict=True
        ):
            zero_pos = int(zero_pos)
            negative_pos = int(negative_pos)
            if zero_pos >= negative_pos:
                return
            shifted = aligned[zero_pos:negative_pos].clone()
            aligned[zero_pos] = -1
            aligned[zero_pos + 1 : negative_pos + 1] = shifted
        call["row_token_uids"] = aligned

    @classmethod
    def _canonicalize_primary_output_tensor(
        cls,
        *,
        module_name: str,
        tensor: torch.Tensor,
        call: dict[str, Any],
    ) -> torch.Tensor:
        """Runs all remaining primary-output canonicalization passes for one call."""
        tensor = cls._canonicalize_gate_up_rank_interleaved_feature_layout(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )
        tensor = cls._canonicalize_componentwise_feature_layout(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )
        tensor = cls._canonicalize_rank_blocked_token_heads(
            module_name=module_name,
            tensor=tensor,
            call=call,
        )
        return tensor

    @staticmethod
    def _decoder_layer_trace_key(
        module_name: str,
        call: dict[str, Any],
    ) -> tuple[str, int, int, int] | None:
        module_name = _normalize_trace_module_name(module_name)
        if ".decoder.layers." not in module_name:
            return None
        tensor = call.get("primary_output")
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            return None
        return (
            module_name.split(".self_attention", 1)[0].split(".mlp", 1)[0],
            _safe_int(call.get("micro_sample_index"), -1),
            _safe_int(call.get("micro_order"), -1),
            int(tensor.shape[0]),
        )

    @staticmethod
    def _decoder_micro_trace_key(
        module_name: str,
        call: dict[str, Any],
    ) -> tuple[str, int, int] | None:
        module_name = _normalize_trace_module_name(module_name)
        if ".decoder.layers." not in module_name:
            return None
        return (
            module_name.split(".self_attention", 1)[0].split(".mlp", 1)[0],
            _safe_int(call.get("micro_sample_index"), -1),
            _safe_int(call.get("micro_order"), -1),
        )

    @staticmethod
    def _is_attention_output_trace(module_name: str) -> bool:
        module_name = _normalize_trace_module_name(module_name)
        return module_name.endswith(".self_attention")

    @staticmethod
    def _rank_blocked_token_head_count(call: dict[str, Any]) -> int | None:
        primary_hint = ForwardTraceCapture._primary_output_merge_hint(call)
        if not isinstance(primary_hint, dict):
            return None
        if primary_hint.get("layout") != "rank_blocked_token_heads":
            return None
        local_heads = primary_hint.get("local_heads")
        world_size_key = primary_hint.get("world_size_key")
        if not isinstance(local_heads, int) or local_heads <= 0:
            return None
        if not isinstance(world_size_key, str):
            return None
        rank_meta = call.get("rank_meta")
        rank_world_size = None
        if isinstance(rank_meta, list) and rank_meta:
            first_meta = rank_meta[0]
            if isinstance(first_meta, dict):
                rank_world_size = first_meta.get(world_size_key)
        elif isinstance(rank_meta, dict):
            rank_world_size = rank_meta.get(world_size_key)
        if not isinstance(rank_world_size, int) or rank_world_size <= 0:
            return None
        return int(local_heads) * int(rank_world_size)

    @classmethod
    def _propagate_decoder_row_token_uids(
        cls,
        trace: dict[str, list[dict[str, Any]]],
    ) -> None:
        row_uids_by_key: dict[tuple[str, int, int, int], torch.Tensor] = {}
        for module_name in sorted(trace.keys()):
            for call in trace[module_name]:
                row_token_uids = call.get("row_token_uids")
                if not isinstance(row_token_uids, torch.Tensor):
                    continue
                key = cls._decoder_layer_trace_key(module_name, call)
                if key is None or key in row_uids_by_key:
                    continue
                row_uids_by_key[key] = row_token_uids
        for module_name in sorted(trace.keys()):
            for call in trace[module_name]:
                if isinstance(call.get("row_token_uids"), torch.Tensor):
                    continue
                key = cls._decoder_layer_trace_key(module_name, call)
                if key is None:
                    continue
                row_token_uids = row_uids_by_key.get(key)
                if row_token_uids is None:
                    continue
                call["row_token_uids"] = row_token_uids

    @classmethod
    def _propagate_attention_output_row_token_uids(
        cls,
        trace: dict[str, list[dict[str, Any]]],
    ) -> None:
        token_uids_by_key: dict[tuple[str, int, int], torch.Tensor] = {}
        for module_name in sorted(trace.keys()):
            if not cls._is_attention_output_trace(module_name):
                continue
            for call in trace[module_name]:
                tensor = call.get("primary_output")
                row_token_uids = call.get("row_token_uids")
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.ndim == 0
                    or not isinstance(row_token_uids, torch.Tensor)
                    or row_token_uids.ndim != 1
                    or int(row_token_uids.numel()) != int(tensor.shape[0])
                ):
                    continue
                key = cls._decoder_micro_trace_key(module_name, call)
                if key is not None and key not in token_uids_by_key:
                    token_uids_by_key[key] = row_token_uids

        if not token_uids_by_key:
            return

        for module_name in sorted(trace.keys()):
            if cls._is_attention_output_trace(module_name):
                continue
            for call in trace[module_name]:
                tensor = call.get("primary_output")
                if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
                    continue
                key = cls._decoder_micro_trace_key(module_name, call)
                if key is None:
                    continue
                token_uids = token_uids_by_key.get(key)
                if token_uids is None:
                    continue
                existing_uids = call.get("row_token_uids")
                if (
                    isinstance(existing_uids, torch.Tensor)
                    and existing_uids.ndim == 1
                    and int(existing_uids.numel()) == int(tensor.shape[0])
                ):
                    continue
                if int(token_uids.numel()) == int(tensor.shape[0]):
                    call["row_token_uids"] = token_uids
                    continue
                head_count = cls._rank_blocked_token_head_count(call)
                if head_count is not None and int(
                    token_uids.numel()
                ) * head_count == int(tensor.shape[0]):
                    call["row_token_uids"] = expand_token_uids_for_heads(
                        token_uids,
                        head_count=head_count,
                    )

    @classmethod
    def _restore_non_cp_sequence_row_token_uids(
        cls,
        trace: dict[str, list[dict[str, Any]]],
    ) -> None:
        for module_name, calls in trace.items():
            if cls._is_moe_expert_forward_module(module_name):
                continue
            for call in calls:
                if (
                    "rank_meta" not in call
                    or _call_cp_world_size(call) > 1
                    or "row_uid_span" in call
                ):
                    continue
                tensor = call.get("primary_output")
                row_token_uids = call.get("row_token_uids")
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.ndim == 0
                    or not isinstance(row_token_uids, torch.Tensor)
                    or row_token_uids.ndim != 1
                    or int(row_token_uids.numel()) != int(tensor.shape[0])
                    or not bool((row_token_uids >= 0).all().item())
                    or int(row_token_uids.unique().numel())
                    != int(row_token_uids.numel())
                ):
                    continue
                call["row_token_uids"] = torch.arange(
                    row_token_uids.numel(),
                    dtype=torch.int64,
                )

    @classmethod
    def canonicalize_trace(
        cls,
        trace: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Canonicalizes topology-dependent trace outputs in place."""
        for module_name in sorted(trace.keys()):
            calls = trace[module_name]
            for call_offset, call in enumerate(calls):
                call_index = int(call.get("call_index", call_offset))
                tensor = call.get("primary_output")
                if isinstance(tensor, torch.Tensor) and not bool(
                    call.get(PRIMARY_OUTPUT_CANONICAL_KEY)
                ):
                    call["primary_output"] = cls._canonicalize_primary_output_tensor(
                        module_name=module_name,
                        tensor=tensor,
                        call=call,
                    )
                call[PRIMARY_OUTPUT_CANONICAL_KEY] = True
        cls._propagate_decoder_row_token_uids(trace)
        cls._propagate_attention_output_row_token_uids(trace)
        cls._restore_non_cp_sequence_row_token_uids(trace)
        for calls in trace.values():
            for call in calls:
                cls._canonicalize_call_row_token_order(call)
        return trace

    @classmethod
    def flatten_trace_tensors(
        cls,
        trace: dict[str, list[dict[str, Any]]],
        *,
        value_key: str,
    ) -> dict[str, Any]:
        """Flattens trace calls into deterministic key->value tensor maps."""
        if value_key == "primary_output":
            cls.canonicalize_trace(trace)
        flattened: dict[str, Any] = {}
        for module_name in sorted(trace.keys()):
            for call_offset, call in enumerate(trace[module_name]):
                tensor = call.get(value_key)
                if tensor is None:
                    continue
                call_index = call.get("call_index", call_offset)
                flattened[
                    f"{_normalize_trace_module_name(module_name)}.call_{call_index}"
                ] = tensor
        return flattened

    @classmethod
    def _merge_rank_values(
        cls,
        values_by_rank: list[Any],
        *,
        preferred_cat_dim: int | None = None,
        preferred_reduce: str | None = None,
    ) -> Any:
        if not values_by_rank:
            raise RuntimeError("Cannot merge empty rank value list")
        if all(isinstance(value, torch.Tensor) for value in values_by_rank):
            tensors = cast(list[torch.Tensor], values_by_rank)
            if preferred_reduce == "sum" and all(
                tensors[0].shape == tensor.shape for tensor in tensors[1:]
            ):
                return torch.stack(tensors, dim=0).sum(dim=0)
            if (
                preferred_cat_dim is not None
                and all(tensor.ndim > 0 for tensor in tensors)
                and cls._can_cat_along_dim(tensors, dim=preferred_cat_dim)
            ):
                return torch.cat(tensors, dim=preferred_cat_dim)
            if all(tensor.ndim > 0 for tensor in tensors):
                if cls._can_cat_along_dim(tensors, dim=0):
                    return torch.cat(tensors, dim=0)
                if cls._can_cat_along_dim(tensors, dim=-1):
                    return torch.cat(tensors, dim=-1)
            if all(tensors[0].shape == tensor.shape for tensor in tensors[1:]):
                return torch.stack(tensors, dim=0)
            return tensors
        if all(isinstance(value, dict) for value in values_by_rank):
            dicts = cast(list[dict[str, Any]], values_by_rank)
            keys = sorted(set().union(*(value.keys() for value in dicts)))
            return {
                key: cls._merge_rank_values(
                    [value[key] for value in dicts if key in value],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for key in keys
            }
        if all(isinstance(value, list) for value in values_by_rank):
            lists = cast(list[list[Any]], values_by_rank)
            if any(len(values) != len(lists[0]) for values in lists[1:]):
                return lists
            return [
                cls._merge_rank_values(
                    [value[index] for value in lists],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for index in range(len(lists[0]))
            ]
        if all(isinstance(value, tuple) for value in values_by_rank):
            tuples = cast(list[tuple[Any, ...]], values_by_rank)
            if any(len(values) != len(tuples[0]) for values in tuples[1:]):
                return tuples
            return tuple(
                cls._merge_rank_values(
                    [value[index] for value in tuples],
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                for index in range(len(tuples[0]))
            )
        if all(value == values_by_rank[0] for value in values_by_rank[1:]):
            return values_by_rank[0]
        return values_by_rank

    @staticmethod
    def _expert_parallel_group_key(entry: dict[str, Any]) -> tuple[int, int] | None:
        """Returns the expert-data/expert-parallel group for one rank call."""
        rank_meta = entry.get("rank_meta")
        if not isinstance(rank_meta, dict):
            return None
        return (
            _safe_int(rank_meta.get("expert_dp_rank"), 0),
            _safe_int(rank_meta.get("ep_rank"), 0),
        )

    @classmethod
    def _merge_expert_tensor_parallel_values(
        cls,
        *,
        module_name: str,
        key: str,
        rank_call_entries: list[dict[str, Any]],
        preferred_cat_dim: int | None,
        preferred_reduce: str | None,
    ) -> Any | None:
        """Merges ETP shards before concatenating independent expert rows."""
        if not cls._is_moe_expert_forward_module(module_name):
            return None
        if preferred_cat_dim != -1 and preferred_reduce != "sum":
            return None
        entry_values = [
            (entry, entry[key]) for entry in rank_call_entries if key in entry
        ]
        if not entry_values or not all(
            isinstance(value, torch.Tensor) for _, value in entry_values
        ):
            return None

        grouped: dict[tuple[int, int], list[tuple[dict[str, Any], torch.Tensor]]] = {}
        for entry, value in entry_values:
            group_key = cls._expert_parallel_group_key(entry)
            if group_key is None:
                return None
            grouped.setdefault(group_key, []).append((entry, cast(torch.Tensor, value)))

        merged_groups: list[torch.Tensor] = []
        for group_key in sorted(grouped):
            group_values = [value for _, value in grouped[group_key]]
            if key == "row_token_uids":
                first = group_values[0]
                if not all(
                    first.shape == value.shape and torch.equal(first, value)
                    for value in group_values[1:]
                ):
                    raise RuntimeError(
                        "Expert tensor-parallel trace row UIDs diverged within "
                        f"group={group_key} module={module_name}"
                    )
                merged_groups.append(first)
                continue
            if preferred_reduce == "sum":
                merged = cls._merge_rank_values(
                    group_values,
                    preferred_reduce="sum",
                )
            else:
                merged = cls._merge_rank_values(
                    group_values,
                    preferred_cat_dim=preferred_cat_dim,
                )
            if not isinstance(merged, torch.Tensor):
                return None
            merged_groups.append(merged)

        if len(merged_groups) == 1:
            return merged_groups[0]
        if cls._can_cat_along_dim(merged_groups, dim=0):
            return torch.cat(merged_groups, dim=0)
        return None

    @classmethod
    def _merge_rank_call_entries(
        cls,
        module_name: str,
        rank_call_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merges one module call across ranks using per-field merge hints."""
        merged_call: dict[str, Any] = {}
        keys = sorted(set().union(*(entry.keys() for entry in rank_call_entries)))
        for key in keys:
            value_entries = [entry for entry in rank_call_entries if key in entry]
            values = [entry[key] for entry in value_entries]
            if key == "rank_meta":
                merged_call[key] = values
                continue
            if key == "row_token_uids":
                primary_hint = next(
                    (
                        cls._primary_output_merge_hint(entry)
                        for entry in value_entries
                        if cls._primary_output_merge_hint(entry) is not None
                    ),
                    None,
                )
                preferred_cat_dim = None
                preferred_reduce = None
                if isinstance(primary_hint, dict):
                    if primary_hint.get("op") == "sum":
                        preferred_reduce = "sum"
                    elif primary_hint.get("op") == "concat" and isinstance(
                        primary_hint.get("dim"), int
                    ):
                        preferred_cat_dim = int(primary_hint["dim"])
                    if primary_hint.get("layout") == "rank_blocked_token_heads":
                        merged_call[key] = cls._merge_rank_values_with_cp_groups(
                            values_by_rank=values,
                            rank_call_entries=value_entries,
                            preferred_cat_dim=preferred_cat_dim,
                            preferred_reduce=preferred_reduce,
                        )
                        continue
                expert_merged = cls._merge_expert_tensor_parallel_values(
                    module_name=module_name,
                    key=key,
                    rank_call_entries=value_entries,
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
                merged_call[key] = (
                    expert_merged
                    if expert_merged is not None
                    else cls._merge_row_token_uids(
                        values_by_rank=values,
                        rank_call_entries=value_entries,
                    )
                )
                continue
            preferred_cat_dim: int | None = None
            preferred_reduce: str | None = None
            if values and key not in {"merge_hints", "call_index", "module_type"}:
                hint_values = [
                    cast(dict[str, Any], entry["merge_hints"]).get(key)
                    for entry in value_entries
                    if isinstance(entry.get("merge_hints"), dict)
                ]
                op_hints = [
                    hint
                    for hint in hint_values
                    if isinstance(hint, dict) and isinstance(hint.get("op"), str)
                ]
                if op_hints:
                    selected_hint = op_hints[0]
                    op = selected_hint.get("op")
                    if op == "concat":
                        dim = selected_hint.get("dim")
                        if isinstance(dim, int):
                            preferred_cat_dim = dim
                    elif op == "sum":
                        preferred_reduce = "sum"
                if (
                    preferred_reduce is None
                    and preferred_cat_dim == 0
                    and all(isinstance(value, torch.Tensor) for value in values)
                ):
                    merged_call[f"{key}__row_splits"] = [
                        int(cast(torch.Tensor, value).shape[0]) for value in values
                    ]
            expert_merged = cls._merge_expert_tensor_parallel_values(
                module_name=module_name,
                key=key,
                rank_call_entries=value_entries,
                preferred_cat_dim=preferred_cat_dim,
                preferred_reduce=preferred_reduce,
            )
            merged_call[key] = (
                expert_merged
                if expert_merged is not None
                else cls._merge_rank_values_with_cp_groups(
                    values_by_rank=values,
                    rank_call_entries=value_entries,
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
            )
        return merged_call

    @classmethod
    def _merge_row_token_uids(
        cls,
        *,
        values_by_rank: list[Any],
        rank_call_entries: list[dict[str, Any]],
    ) -> Any:
        """Preserves row identities across feature-sharded ranks."""
        if not all(isinstance(value, torch.Tensor) for value in values_by_rank):
            return cls._merge_rank_values(values_by_rank, preferred_cat_dim=0)

        tensors = cast(list[torch.Tensor], values_by_rank)
        grouped_indices: dict[int, list[int]] = {}
        for index, entry in enumerate(rank_call_entries):
            rank_meta = entry.get("rank_meta")
            if not isinstance(rank_meta, dict):
                return cls._merge_rank_values(values_by_rank, preferred_cat_dim=0)
            cp_rank = _safe_int(rank_meta.get("cp_rank"), 0)
            grouped_indices.setdefault(cp_rank, []).append(index)

        merged_by_cp: list[torch.Tensor] = []
        for cp_rank in sorted(grouped_indices):
            group_tensors = [tensors[index] for index in grouped_indices[cp_rank]]
            first = group_tensors[0]
            if all(
                first.shape == tensor.shape and torch.equal(first, tensor)
                for tensor in group_tensors[1:]
            ):
                merged_by_cp.append(first)
                continue
            merged = cls._merge_rank_values(group_tensors, preferred_cat_dim=0)
            if not isinstance(merged, torch.Tensor):
                return merged
            merged_by_cp.append(merged)

        if len(merged_by_cp) == 1:
            return merged_by_cp[0]
        if cls._can_cat_along_dim(merged_by_cp, dim=0):
            return torch.cat(merged_by_cp, dim=0)
        return merged_by_cp

    @classmethod
    def _merge_rank_values_with_cp_groups(
        cls,
        *,
        values_by_rank: list[Any],
        rank_call_entries: list[dict[str, Any]],
        preferred_cat_dim: int | None,
        preferred_reduce: str | None,
    ) -> Any:
        """Merges rank values, preserving CP row shards when features are also sharded."""
        cp_world_sizes: set[int] = set()
        for entry in rank_call_entries:
            rank_meta = entry.get("rank_meta")
            if isinstance(rank_meta, dict):
                cp_world_sizes.add(_safe_int(rank_meta.get("cp_world_size"), 1))
        if len(cp_world_sizes) != 1 or next(iter(cp_world_sizes), 1) <= 1:
            return cls._merge_rank_values(
                values_by_rank,
                preferred_cat_dim=preferred_cat_dim,
                preferred_reduce=preferred_reduce,
            )
        if preferred_cat_dim != -1 and preferred_reduce != "sum":
            return cls._merge_rank_values(
                values_by_rank,
                preferred_cat_dim=preferred_cat_dim,
                preferred_reduce=preferred_reduce,
            )

        grouped_indices: dict[int, list[int]] = {}
        for index, entry in enumerate(rank_call_entries):
            rank_meta = entry.get("rank_meta")
            if not isinstance(rank_meta, dict):
                return cls._merge_rank_values(
                    values_by_rank,
                    preferred_cat_dim=preferred_cat_dim,
                    preferred_reduce=preferred_reduce,
                )
            cp_rank = _safe_int(rank_meta.get("cp_rank"), 0)
            grouped_indices.setdefault(cp_rank, []).append(index)
        if len(grouped_indices) <= 1:
            return cls._merge_rank_values(
                values_by_rank,
                preferred_cat_dim=preferred_cat_dim,
                preferred_reduce=preferred_reduce,
            )

        merged_by_cp = [
            cls._merge_rank_values(
                [values_by_rank[index] for index in grouped_indices[cp_rank]],
                preferred_cat_dim=preferred_cat_dim,
                preferred_reduce=preferred_reduce,
            )
            for cp_rank in sorted(grouped_indices)
        ]
        if all(isinstance(value, torch.Tensor) for value in merged_by_cp):
            tensors = cast(list[torch.Tensor], merged_by_cp)
            if cls._can_cat_along_dim(tensors, dim=0):
                return torch.cat(tensors, dim=0)
        return merged_by_cp

    @staticmethod
    def _can_cat_along_dim(tensors: list[torch.Tensor], dim: int) -> bool:
        if not tensors:
            return False
        if tensors[0].ndim == 0:
            return False
        ndim = tensors[0].ndim
        axis = dim if dim >= 0 else ndim + dim
        if axis < 0 or axis >= ndim:
            return False
        if any(tensor.ndim != ndim for tensor in tensors[1:]):
            return False
        for dim_index in range(ndim):
            if dim_index == axis:
                continue
            dim_size = tensors[0].shape[dim_index]
            if any(tensor.shape[dim_index] != dim_size for tensor in tensors[1:]):
                return False
        return True

    @classmethod
    def _merge_rank_traces(
        cls,
        rank_traces: list[dict[str, list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        if len(rank_traces) == 1:
            return rank_traces[0]
        merged: dict[str, list[dict[str, Any]]] = {}
        module_names = sorted(set().union(*(trace.keys() for trace in rank_traces)))
        for module_name in module_names:
            module_calls: list[dict[str, Any]] = []
            grouped_calls: dict[
                tuple[int, int, int, int],
                list[dict[str, Any]],
            ] = {}
            for trace in rank_traces:
                for call in trace.get(module_name, []):
                    sample_kind, sample_sort_index = _trace_call_sort_key(call)
                    merge_key = (
                        sample_kind,
                        sample_sort_index,
                        int(call.get("micro_order", 0)),
                        int(call.get("micro_call_index", call.get("call_index", 0))),
                    )
                    grouped_calls.setdefault(merge_key, []).append(call)
            for merged_index, merge_key in enumerate(sorted(grouped_calls)):
                merged_call = cls._merge_rank_call_entries(
                    module_name,
                    grouped_calls[merge_key],
                )
                merged_call["call_index"] = merged_index
                module_calls.append(merged_call)
            merged[module_name] = module_calls
        return merged

    @staticmethod
    def _gather_rank_traces(
        local_trace: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, list[dict[str, Any]]]] | None:
        if (
            not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
            or torch.distributed.get_world_size() == 1  # ty: ignore[possibly-missing-attribute]
        ):
            return [local_trace]
        gathered: list[dict[str, list[dict[str, Any]]] | None] = [
            None
        ] * torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
        torch.distributed.all_gather_object(gathered, local_trace)  # ty: ignore[possibly-missing-attribute]
        if torch.distributed.get_rank() != 0:  # ty: ignore[possibly-missing-attribute]
            return None
        return cast(list[dict[str, list[dict[str, Any]]]], gathered)

    @staticmethod
    def _merge_group_tensor(
        tensors: list[tuple[torch.Tensor, torch.Tensor | None]],
    ) -> torch.Tensor:
        if len(tensors) == 1:
            return tensors[0][0]

        tensor_values = [tensor for tensor, _ in tensors]
        first = tensor_values[0]
        if all(tensor.shape == first.shape for tensor in tensor_values[1:]) and all(
            torch.equal(first, tensor) for tensor in tensor_values[1:]
        ):
            return first

        uid_values = [uids for _, uids in tensors]
        if any(uids is None for uids in uid_values):
            raise RuntimeError(
                "Mismatched output captures for the same micro output across non-DP ranks"
            )

        typed_uid_values = cast(list[torch.Tensor], uid_values)
        typed_tensors = [(tensor, cast(torch.Tensor, uids)) for tensor, uids in tensors]
        if any(tensor.ndim != 2 for tensor in tensor_values) or any(
            uids.ndim != 2 for uids in typed_uid_values
        ):
            raise RuntimeError(
                "Root output UID merge currently requires rank-local 2D tensors"
            )
        if any(tensor.shape != uids.shape for tensor, uids in typed_tensors):
            raise RuntimeError(
                "Root output tensor/token UID shape mismatch during CP merge"
            )

        batch_size = int(first.shape[0])
        max_row_length = 1
        for uids in typed_uid_values:
            valid_uids = uids[uids >= 0]
            if int(valid_uids.numel()) > 0:
                max_row_length = max(
                    max_row_length,
                    int(valid_uids.max().item()) + 1,
                )

        merged = first.new_zeros((batch_size, max_row_length))
        filled = torch.zeros((batch_size, max_row_length), dtype=torch.bool)
        for tensor, uids in typed_tensors:
            for row_index in range(batch_size):
                row_uids = uids[row_index]
                valid_mask = row_uids >= 0
                if not bool(valid_mask.any()):
                    continue
                row_positions = row_uids[valid_mask].to(dtype=torch.long)
                row_values = tensor[row_index, valid_mask]
                existing_mask = filled[row_index].index_select(0, row_positions)
                if bool(existing_mask.any()):
                    existing_values = merged[row_index].index_select(0, row_positions)
                    if not torch.equal(existing_values, row_values):
                        raise RuntimeError(
                            "Conflicting CP output values for the same token UID"
                        )
                merged[row_index].index_copy_(0, row_positions, row_values)
                filled[row_index].index_fill_(0, row_positions, True)

        for row_index in range(batch_size):
            row_filled = filled[row_index]
            present = torch.nonzero(row_filled, as_tuple=False).reshape(-1)
            if int(present.numel()) == 0:
                continue
            expected = torch.arange(int(present.numel()), dtype=torch.long)
            if not torch.equal(present, expected):
                raise RuntimeError(
                    "CP output token UIDs did not form a contiguous row-major prefix"
                )
        return merged

    @staticmethod
    def _gather_rank_outputs(
        local_outputs: list[
            tuple[int | None, int, int | None, torch.Tensor, torch.Tensor | None]
        ],
    ) -> (
        list[
            list[
                tuple[
                    int | None,
                    int,
                    int | None,
                    torch.Tensor,
                    torch.Tensor | None,
                ]
            ]
        ]
        | None
    ):
        if (
            not torch.distributed.is_initialized()  # ty: ignore[possibly-missing-attribute]
            or torch.distributed.get_world_size() == 1  # ty: ignore[possibly-missing-attribute]
        ):
            return [local_outputs]
        gathered: list[
            list[
                tuple[
                    int | None,
                    int,
                    int | None,
                    torch.Tensor,
                    torch.Tensor | None,
                ]
            ]
            | None
        ] = [None] * torch.distributed.get_world_size()  # ty: ignore[possibly-missing-attribute]
        torch.distributed.all_gather_object(gathered, local_outputs)  # ty: ignore[possibly-missing-attribute]
        if torch.distributed.get_rank() != 0:  # ty: ignore[possibly-missing-attribute]
            return None
        return cast(
            list[
                list[
                    tuple[
                        int | None,
                        int,
                        int | None,
                        torch.Tensor,
                        torch.Tensor | None,
                    ]
                ]
            ],
            gathered,
        )

    def ordered_step_outputs(self) -> list[torch.Tensor] | None:
        ordered = self.ordered_step_outputs_with_sample_indices()
        if ordered is None:
            return None
        _, outputs = ordered
        return outputs

    def ordered_step_outputs_with_sample_indices(
        self,
    ) -> tuple[list[int | None], list[torch.Tensor]] | None:
        if not self.enabled:
            return None
        gathered_outputs = self._gather_rank_outputs(self.current_step_outputs)
        if gathered_outputs is None:
            return None
        grouped: dict[
            tuple[int | None, int | None, int],
            list[tuple[torch.Tensor, torch.Tensor | None]],
        ] = {}
        for rank_outputs in gathered_outputs:
            for (
                sample_index,
                micro_order,
                micro_slot,
                tensor,
                token_uids,
            ) in rank_outputs:
                group_key = (sample_index, micro_slot, micro_order)
                grouped.setdefault(group_key, []).append((tensor, token_uids))
        ordered_group_keys = sorted(
            grouped,
            key=lambda item: _captured_output_sort_key(item[0], item[2], item[1]),
        )
        return (
            [sample_index for sample_index, _, _ in ordered_group_keys],
            [
                self._merge_group_tensor(grouped[group_key])
                for group_key in ordered_group_keys
            ],
        )

    def save_current_step(self, traces_dir: Path) -> Path | None:
        if not self.enabled or self.current_step_index is None:
            return None
        gathered_traces = self._gather_rank_traces(self.current_step_trace)
        if gathered_traces is None:
            return None
        merged_trace = self.canonicalize_trace(self._merge_rank_traces(gathered_traces))
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_path = traces_dir / f"forward_trace_step_{self.current_step_index:03d}.pt"
        tmp_trace_path = trace_path.with_suffix(f"{trace_path.suffix}.tmp")
        torch.save(merged_trace, tmp_trace_path)
        os.replace(tmp_trace_path, trace_path)
        return trace_path

    @classmethod
    def load_trace(cls, trace_path: Path) -> dict[str, list[dict[str, Any]]]:
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        return cls.canonicalize_trace(trace)

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
