from __future__ import annotations

from contextlib import ExitStack
from typing import Any

import torch
import torch.distributed as dist


def _torch_chunk_gated_delta_rule_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
    )

    if kwargs:
        raise TypeError(
            f"Unsupported Qwen3.5 GDN fp32 reference kwargs: {sorted(kwargs)}"
        )
    if scale is not None and scale != float(key.shape[-1] ** -0.5):
        raise ValueError(
            "Qwen3.5 torch GDN reference only supports the model-default scale"
        )
    if cu_seqlens is None:
        return torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    if query.shape[0] != 1:
        raise RuntimeError(
            "Qwen3.5 packed GDN fp32 reference expects packed batch size 1, "
            f"got {query.shape[0]}"
        )
    starts = cu_seqlens.detach().cpu().tolist()
    outputs: list[torch.Tensor] = []
    finals: list[torch.Tensor] = []
    for index, (start, end) in enumerate(zip(starts, starts[1:])):
        state = None if initial_state is None else initial_state[index : index + 1]
        output, final = torch_chunk_gated_delta_rule(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        outputs.append(output)
        if final is not None:
            finals.append(final)
    return torch.cat(outputs, dim=1), torch.cat(finals, dim=0) if finals else None


def _pad_sequence_dim(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
    pad_tokens = target_tokens - int(tensor.shape[1])
    if pad_tokens == 0:
        return tensor
    if pad_tokens < 0:
        raise ValueError(
            f"Cannot pad tensor with {int(tensor.shape[1])} tokens to {target_tokens}"
        )
    padding = tensor.new_zeros(*tensor.shape[:1], pad_tokens, *tensor.shape[2:])
    return torch.cat((tensor, padding), dim=1)


def _autograd_all_gather_varlen(
    tensor: torch.Tensor,
    *,
    group: Any,
    token_counts: list[int],
) -> list[torch.Tensor]:
    from torch.distributed.nn.functional import all_gather

    max_tokens = max(token_counts)
    padded = _pad_sequence_dim(tensor, max_tokens)
    gathered = all_gather(padded, group=group)
    return [
        rank_tensor[:, :token_count]
        for rank_tensor, token_count in zip(gathered, token_counts)
    ]


def _split_segments_by_rank(
    gathered: list[torch.Tensor],
    lengths_by_rank_cpu: torch.Tensor,
) -> list[list[torch.Tensor]]:
    return [
        list(rank_tensor.split(lengths_by_rank_cpu[rank].tolist(), dim=1))
        for rank, rank_tensor in enumerate(gathered)
    ]


def _cat_non_empty(tensors: list[torch.Tensor], *, dim: int) -> torch.Tensor:
    non_empty = [tensor for tensor in tensors if int(tensor.shape[dim]) != 0]
    if non_empty:
        return torch.cat(non_empty, dim=dim)
    return tensors[0]


def _torch_chunk_gated_delta_rule_native_cp_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    group: Any,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    lengths_by_rank_cpu: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if group is None:
        raise ValueError("Qwen3.5 GDN fp32 CP reference requires a process group")
    if lengths_by_rank_cpu is None:
        raise ValueError("Qwen3.5 GDN fp32 CP reference requires all-rank lengths")
    if lengths_by_rank_cpu.device.type != "cpu":
        raise ValueError("Qwen3.5 GDN fp32 CP reference lengths must stay on CPU")
    world_size = dist.get_world_size(group)  # ty: ignore[possibly-missing-attribute]
    rank = dist.get_rank(group)  # ty: ignore[possibly-missing-attribute]
    segment_count = int(initial_state.shape[0])
    if tuple(lengths_by_rank_cpu.shape) != (world_size, segment_count):
        raise ValueError(
            "Qwen3.5 GDN fp32 CP reference lengths must be [world_size, segments], "
            f"got {tuple(lengths_by_rank_cpu.shape)}"
        )
    local_tokens = int(lengths_by_rank_cpu[rank].sum().item())
    if int(q.shape[1]) != local_tokens:
        raise ValueError(
            "Qwen3.5 GDN fp32 CP reference local token count mismatch: "
            f"q has {int(q.shape[1])}, metadata has {local_tokens}"
        )
    if cu_seqlens is not None and cu_seqlens_cpu is None:
        raise ValueError("Qwen3.5 GDN fp32 CP reference requires CPU cu_seqlens")

    token_counts = [
        int(lengths_by_rank_cpu[peer].sum().item()) for peer in range(world_size)
    ]
    q_by_segment = _split_segments_by_rank(
        _autograd_all_gather_varlen(q, group=group, token_counts=token_counts),
        lengths_by_rank_cpu,
    )
    k_by_segment = _split_segments_by_rank(
        _autograd_all_gather_varlen(k, group=group, token_counts=token_counts),
        lengths_by_rank_cpu,
    )
    v_by_segment = _split_segments_by_rank(
        _autograd_all_gather_varlen(v, group=group, token_counts=token_counts),
        lengths_by_rank_cpu,
    )
    g_by_segment = _split_segments_by_rank(
        _autograd_all_gather_varlen(g, group=group, token_counts=token_counts),
        lengths_by_rank_cpu,
    )
    beta_by_segment = _split_segments_by_rank(
        _autograd_all_gather_varlen(beta, group=group, token_counts=token_counts),
        lengths_by_rank_cpu,
    )

    local_outputs: list[torch.Tensor] = []
    final_states: list[torch.Tensor] = []
    for segment_index in range(segment_count):
        full_output, full_final = _torch_chunk_gated_delta_rule_reference(
            _cat_non_empty(
                [rank_segments[segment_index] for rank_segments in q_by_segment],
                dim=1,
            ),
            _cat_non_empty(
                [rank_segments[segment_index] for rank_segments in k_by_segment],
                dim=1,
            ),
            _cat_non_empty(
                [rank_segments[segment_index] for rank_segments in v_by_segment],
                dim=1,
            ),
            g=_cat_non_empty(
                [rank_segments[segment_index] for rank_segments in g_by_segment],
                dim=1,
            ),
            beta=_cat_non_empty(
                [rank_segments[segment_index] for rank_segments in beta_by_segment],
                dim=1,
            ),
            initial_state=initial_state[segment_index : segment_index + 1],
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            scale=scale,
        )
        segment_start = int(lengths_by_rank_cpu[:rank, segment_index].sum().item())
        segment_len = int(lengths_by_rank_cpu[rank, segment_index].item())
        local_outputs.append(
            full_output[:, segment_start : segment_start + segment_len]
        )
        if full_final is not None:
            final_states.append(full_final)
    output = torch.cat(local_outputs, dim=1)
    final_state = torch.cat(final_states, dim=0) if final_states else None
    return output, final_state


def install_megatron_qwen35_gdn_fp32_reference(
    stack: ExitStack,
    *,
    base_model: str,
) -> None:
    from art.megatron.model_support.registry import get_model_support_handler

    handler = get_model_support_handler(base_model)
    if handler.key not in {"qwen3_5_dense", "qwen3_5_moe"}:
        return
    from art.megatron.gdn import operator as gdn_operator

    original_single_rank = gdn_operator._chunk_gated_delta_rule
    original_native_cp = gdn_operator.chunk_gated_delta_rule_native_cp
    setattr(
        gdn_operator,
        "_chunk_gated_delta_rule",
        _torch_chunk_gated_delta_rule_reference,
    )
    setattr(
        gdn_operator,
        "chunk_gated_delta_rule_native_cp",
        _torch_chunk_gated_delta_rule_native_cp_reference,
    )
    stack.callback(
        setattr, gdn_operator, "chunk_gated_delta_rule_native_cp", original_native_cp
    )
    stack.callback(
        setattr, gdn_operator, "_chunk_gated_delta_rule", original_single_rank
    )
