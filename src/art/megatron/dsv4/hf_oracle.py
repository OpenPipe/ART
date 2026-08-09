from types import MethodType
from typing import Any

import torch
from torch import nn

from art.megatron.dsv4.compressor import (
    Dsv4CompressionLayout,
    build_prefix_tree_compression_layouts,
    compressed_layout_visibility,
)
from art.megatron.dsv4.kernel.precision_aligned_ops import linear_bf16_fp32

_COMPRESSOR_TYPES = {"DeepseekV4CSACompressor", "DeepseekV4HCACompressor"}


def _aligned_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return linear_bf16_fp32(x, module.weight)


def _patch_aligned_linear(module: nn.Linear) -> None:
    if module.bias is not None:
        raise RuntimeError("DSV4 compressor oracle projections must be bias-free")
    if getattr(module, "_art_dsv4_aligned", False):
        return
    module.forward = MethodType(_aligned_linear_forward, module)
    module._art_dsv4_aligned = True


def _cast_compressor_output(
    _module: nn.Module,
    inputs: tuple[Any, ...],
    output: tuple[torch.Tensor, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    compressed_kv, block_bias = output
    return compressed_kv.to(inputs[0].dtype), block_bias


def _cast_indexer_key(
    _module: nn.Module,
    inputs: tuple[Any, ...],
) -> tuple[Any, ...]:
    q, compressed_kv, *rest = inputs
    return q, compressed_kv.to(q.dtype), *rest


def _gather_projected(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if int(tensor.shape[0]) != 1:
        raise ValueError("DSV4 HF prefix compression requires batch size one")
    safe_indices = indices.clamp(0, max(int(tensor.shape[1]) - 1, 0))
    gathered = tensor[0].index_select(0, safe_indices.reshape(-1))
    return gathered.view(1, *indices.shape, tensor.shape[-1])


def _compress_prefix_projected(
    module: Any,
    kv: torch.Tensor,
    gate: torch.Tensor,
    layout: Dsv4CompressionLayout,
) -> torch.Tensor:
    ratio = int(module.compress_rate)
    current_valid = layout.current_indices >= 0
    current_kv = _gather_projected(kv, layout.current_indices)
    current_gate = _gather_projected(gate, layout.current_indices)
    current_kv = torch.where(
        current_valid.unsqueeze(-1), current_kv, torch.zeros_like(current_kv)
    )
    current_gate = torch.where(
        current_valid.unsqueeze(-1),
        current_gate,
        torch.full_like(current_gate, float("-inf")),
    )
    position_bias = module.position_bias.view(1, 1, ratio, -1)
    if ratio == 4:
        head_dim = int(module.head_dim)
        previous_valid = layout.previous_indices >= 0
        previous_kv = _gather_projected(kv, layout.previous_indices)
        previous_gate = _gather_projected(gate, layout.previous_indices)
        previous_kv = torch.where(
            previous_valid.unsqueeze(-1),
            previous_kv,
            torch.zeros_like(previous_kv),
        )
        previous_gate = torch.where(
            previous_valid.unsqueeze(-1),
            previous_gate,
            torch.full_like(previous_gate, float("-inf")),
        )
        current_gate = current_gate + position_bias
        previous_gate = previous_gate + position_bias
        slots_kv = torch.cat(
            [previous_kv[..., :head_dim], current_kv[..., head_dim:]], dim=2
        )
        slots_gate = torch.cat(
            [previous_gate[..., :head_dim], current_gate[..., head_dim:]], dim=2
        )
    else:
        slots_kv = current_kv
        slots_gate = current_gate + position_bias
    compressed = (
        slots_kv * slots_gate.softmax(dim=2, dtype=torch.float32).to(slots_kv.dtype)
    ).sum(dim=2)
    compressed = module.kv_norm(compressed)
    positions = layout.entry_start_positions.unsqueeze(0)
    cos, sin = module.rotary_emb(
        compressed, position_ids=positions, layer_type=module.rope_layer_type
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
    )

    return apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)


def _require_fresh_compressor_cache(past_key_values: Any, layer_idx: int) -> None:
    if past_key_values is None:
        return
    cache_layer = past_key_values.layers[layer_idx]
    nonempty = []
    for name in ("buffer_kv", "buffer_gate", "compressed_kv"):
        values = getattr(cache_layer, name, {})
        nonempty.extend(
            f"{name}.{key}" for key, value in values.items() if value is not None
        )
    for name in ("overlap_kv", "overlap_gate"):
        values = getattr(cache_layer, name, {})
        nonempty.extend(
            f"{name}.{key}" for key, value in values.items() if value is not None
        )
    nonempty.extend(
        f"entry_count.{key}={value}"
        for key, value in getattr(cache_layer, "entry_count", {}).items()
        if value
    )
    if nonempty:
        raise ValueError(
            "DSV4 HF prefix oracle requires fresh compressor cache state, got "
            + ", ".join(nonempty)
        )


def _prefix_indexer_forward(
    module: Any,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Any = None,
    layer_idx: int = 0,
) -> torch.Tensor:
    layout = getattr(module, "_art_dsv4_prefix_layout", None)
    if layout is None:
        if q_residual is None and position_ids is None and past_key_values is None:
            return module._art_dsv4_flat_forward(hidden_states)
        return module._art_dsv4_flat_forward(
            hidden_states, q_residual, position_ids, past_key_values, layer_idx
        )
    if q_residual is None or position_ids is None:
        raise ValueError("DSV4 HF prefix indexer requires query and position inputs")
    _require_fresh_compressor_cache(past_key_values, layer_idx)
    batch, seq_len, _ = hidden_states.shape
    kv = module.kv_proj(hidden_states)
    gate = module.gate_proj(hidden_states)
    compressed = _compress_prefix_projected(module, kv, gate, layout)
    cos, sin = module.rotary_emb(
        hidden_states,
        position_ids=position_ids,
        layer_type=module.rope_layer_type,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
    )

    q = module.q_b_proj(q_residual).view(
        batch, seq_len, module.num_heads, module.head_dim
    )
    q = apply_rotary_pos_emb(q.transpose(1, 2), cos, sin).transpose(1, 2)
    scores = module.scorer(q, compressed, hidden_states)
    visible = compressed_layout_visibility(layout, position_ids=position_ids)
    scores = scores.masked_fill(~visible, float("-inf"))
    top_k = min(int(module.index_topk), int(compressed.shape[1]))
    indices = scores.topk(top_k, dim=-1).indices
    valid = visible.gather(-1, indices)
    return torch.where(valid, indices, torch.full_like(indices, -1))


def _prefix_compressor_forward(
    module: Any,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Any = None,
    layer_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    layout = getattr(module, "_art_dsv4_prefix_layout", None)
    if layout is None:
        if q_residual is None and position_ids is None and past_key_values is None:
            return module._art_dsv4_flat_forward(hidden_states)
        return module._art_dsv4_flat_forward(
            hidden_states, q_residual, position_ids, past_key_values, layer_idx
        )
    if q_residual is None or position_ids is None:
        raise ValueError("DSV4 HF prefix compressor requires query and position inputs")
    _require_fresh_compressor_cache(past_key_values, layer_idx)
    kv = module.kv_proj(hidden_states)
    gate = module.gate_proj(hidden_states)
    compressed = _compress_prefix_projected(module, kv, gate, layout)
    compressed_kv = compressed.unsqueeze(1)
    if hasattr(module, "indexer"):
        top_k_indices = module.indexer(
            hidden_states, q_residual, position_ids, past_key_values, layer_idx
        )
        compressed_len = int(compressed.shape[1])
        valid = top_k_indices >= 0
        safe_indices = torch.where(
            valid, top_k_indices, torch.full_like(top_k_indices, compressed_len)
        )
        block_bias = compressed.new_full(
            (*safe_indices.shape[:2], 1, compressed_len + 1), float("-inf")
        ).transpose(1, 2)
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, block_bias[..., :compressed_len]
    visible = compressed_layout_visibility(layout, position_ids=position_ids).unsqueeze(
        1
    )
    block_bias = compressed.new_zeros(visible.shape).masked_fill(
        ~visible, float("-inf")
    )
    return compressed_kv, block_bias


def _patch_prefix_forward(module: Any, forward: Any) -> None:
    module._art_dsv4_flat_forward = module.forward
    module.forward = MethodType(forward, module)


def prepare_hf_reference_model(model: Any) -> Any:
    """Align native HF compressor precision with the training/serving path."""
    compressors = [
        module
        for module in model.modules()
        if type(module).__name__ in _COMPRESSOR_TYPES
    ]
    if not compressors:
        raise RuntimeError("Native DSV4 HF model has no recognized compressor")
    for compressor in compressors:
        _patch_aligned_linear(compressor.kv_proj)
        _patch_aligned_linear(compressor.gate_proj)
        _patch_prefix_forward(compressor, _prefix_compressor_forward)
        compressor.register_forward_hook(_cast_compressor_output)
        indexer = getattr(compressor, "indexer", None)
        if indexer is None:
            continue
        _patch_aligned_linear(indexer.kv_proj)
        _patch_aligned_linear(indexer.gate_proj)
        _patch_prefix_forward(indexer, _prefix_indexer_forward)
        indexer.scorer.register_forward_pre_hook(_cast_indexer_key)
    return model


def set_hf_reference_prefix_tree(
    model: Any,
    *,
    position_ids: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> None:
    device = next(model.parameters()).device
    layouts = build_prefix_tree_compression_layouts(
        position_ids=position_ids.unsqueeze(0),
        group_ids=group_ids.unsqueeze(0),
        parent_ids=parent_ids.unsqueeze(0),
        device=device,
    )
    for module in model.modules():
        if type(module).__name__ not in _COMPRESSOR_TYPES:
            continue
        layout = layouts[int(module.compress_rate)]
        module._art_dsv4_prefix_layout = layout
        indexer = getattr(module, "indexer", None)
        if indexer is not None:
            indexer._art_dsv4_prefix_layout = layout
