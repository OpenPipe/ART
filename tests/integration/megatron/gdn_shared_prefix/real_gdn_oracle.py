from __future__ import annotations

from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict
import torch
from torch import Tensor
import torch.nn.functional as F

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.gdn_prefix_tree import FLA_CHUNK_SIZE
from art.megatron.gdn.operator import (
    _apply_gated_rms_norm,
    _chunk_gated_delta_rule,
    _disable_reentrant_te_linear_transpose_cache,
    _in_proj,
    _l2norm,
    _local_key_heads,
    _local_value_dim,
    _local_value_heads,
    _out_proj,
    _zero_conv_state,
    _zero_recurrent_state,
    gdn_prefix_tree_forward,
)

from .layout_reference import build_test_gdn_cp_layout_plan
from .metrics import (
    mean_abs_pct,
    parameter_grad_mean_abs_pct_with_name,
    stable_output_mse_loss,
)
from .parser_import import parse_gdn_prefix_tree_segments


class RealGdnOracleMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    loss_mean_abs_pct: float
    loss_abs_diff: float
    output_mean_abs_pct: float
    hidden_grad_mean_abs_pct: float
    param_grad_mean_abs_pct: float


class GdnChainBoundaryDebug(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    family_index: int
    segment_kind: str
    child_index: int | None
    boundary_kind: str
    shard_index: int
    token_offset: int
    conv_initial: Tensor
    recurrent_initial: Tensor


GdnChainMutation = Literal[
    "detach_prefix_state", "zero_conv_tail", "zero_recurrent_parent"
]


class _TreeFamily(NamedTuple):
    row_index: int
    family_index: int
    prefix: Any
    completions: tuple[Any, ...]
    segment_indices: tuple[int, ...]
    parent_indices: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return self.prefix.length + sum(segment.length for segment in self.completions)


def _segment_path(spec: Any, segment_index: int) -> tuple[Any, ...]:
    path = []
    cursor = segment_index
    while cursor >= 0:
        path.append(cursor)
        cursor = spec.tree_parent_indices[cursor]
    return tuple(spec.tree_segments[index] for index in reversed(path))


def _tree_families(spec: Any) -> tuple[_TreeFamily, ...]:
    families = []
    for root_index, root in enumerate(spec.tree_segments):
        if spec.tree_parent_indices[root_index] >= 0:
            continue
        segment_indices = [root_index]
        for index in range(root_index + 1, len(spec.tree_segments)):
            parent = spec.tree_parent_indices[index]
            while parent >= 0:
                if parent == root_index:
                    segment_indices.append(index)
                    break
                parent = spec.tree_parent_indices[parent]
        segments = tuple(spec.tree_segments[index] for index in segment_indices)
        families.append(
            _TreeFamily(
                row_index=root.row_index,
                family_index=root_index,
                prefix=root,
                completions=segments[1:],
                segment_indices=tuple(segment_indices),
                parent_indices=tuple(
                    spec.tree_parent_indices[index] for index in segment_indices
                ),
            )
        )
    return tuple(families)


def compare_real_gdn_cp1_to_flattened(
    *,
    packed_gdn: Any,
    flat_gdn: Any,
    hidden_states: Tensor,
    group_ids: Tensor,
    parent_ids: Tensor,
    assistant_mask: Tensor,
) -> RealGdnOracleMetrics:
    packed_hidden = hidden_states.clone().detach().requires_grad_(True)
    flat_hidden = hidden_states.clone().detach().requires_grad_(True)

    packed_out, _ = gdn_prefix_tree_forward(
        packed_gdn,
        packed_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    flat_out = run_real_gdn_flattened_reference(
        flat_gdn,
        flat_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )

    packed_loss = _masked_quadratic_loss(packed_out, assistant_mask)
    flat_loss = _masked_quadratic_loss(flat_out, assistant_mask)
    packed_loss.backward()
    flat_loss.backward()

    return RealGdnOracleMetrics(
        loss_mean_abs_pct=mean_abs_pct(flat_loss.detach(), packed_loss.detach()),
        loss_abs_diff=float(
            (flat_loss.detach().float() - packed_loss.detach().float()).abs()
        ),
        output_mean_abs_pct=mean_abs_pct(flat_out.detach(), packed_out.detach()),
        hidden_grad_mean_abs_pct=mean_abs_pct(
            _require_grad(flat_hidden), _require_grad(packed_hidden)
        ),
        param_grad_mean_abs_pct=parameter_grad_mean_abs_pct_with_name(
            flat_gdn, packed_gdn
        )[1],
    )


def compare_real_gdn_cp1_to_flattened_with_output_grad(
    *,
    packed_gdn: Any,
    flat_gdn: Any,
    hidden_states: Tensor,
    group_ids: Tensor,
    parent_ids: Tensor,
    output_grad: Tensor,
) -> RealGdnOracleMetrics:
    packed_hidden = hidden_states.clone().detach().requires_grad_(True)
    flat_hidden = hidden_states.clone().detach().requires_grad_(True)

    packed_out, _ = gdn_prefix_tree_forward(
        packed_gdn,
        packed_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    flat_out = run_real_gdn_flattened_reference(
        flat_gdn,
        flat_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )

    real_mask = (group_ids != -1).transpose(0, 1).unsqueeze(-1)
    loss_denominator = real_mask.expand_as(output_grad).sum()
    packed_loss = stable_output_mse_loss(
        packed_out,
        output_grad,
        mask=real_mask,
        denominator=loss_denominator,
    )
    flat_loss = stable_output_mse_loss(
        flat_out,
        output_grad,
        mask=real_mask,
        denominator=loss_denominator,
    )
    packed_loss.backward()
    flat_loss.backward()

    return RealGdnOracleMetrics(
        loss_mean_abs_pct=mean_abs_pct(flat_loss.detach(), packed_loss.detach()),
        loss_abs_diff=float(
            (flat_loss.detach().float() - packed_loss.detach().float()).abs()
        ),
        output_mean_abs_pct=mean_abs_pct(flat_out.detach(), packed_out.detach()),
        hidden_grad_mean_abs_pct=mean_abs_pct(
            _require_grad(flat_hidden), _require_grad(packed_hidden)
        ),
        param_grad_mean_abs_pct=parameter_grad_mean_abs_pct_with_name(
            flat_gdn, packed_gdn
        )[1],
    )


def attach_main_grads(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if not hasattr(parameter, "main_grad"):
            setattr(parameter, "main_grad", torch.zeros_like(parameter))


def zero_parameter_grads(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.grad = None
        main_grad = getattr(parameter, "main_grad", None)
        if main_grad is not None:
            main_grad.zero_()


def _run_gdn_segment(
    gdn: Any,
    hidden_states: Tensor,
    *,
    conv_initial: Tensor,
    recurrent_initial: Tensor,
    output_final_state: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    _disable_reentrant_te_linear_transpose_cache(gdn)
    seq_len, batch_size, _ = hidden_states.shape
    if int(conv_initial.shape[0]) != batch_size:
        raise ValueError(
            "conv_initial batch must match hidden_states batch, got "
            f"{tuple(conv_initial.shape)} for hidden {tuple(hidden_states.shape)}"
        )
    if int(recurrent_initial.shape[0]) != batch_size:
        raise ValueError(
            "recurrent_initial batch must match hidden_states batch, got "
            f"{tuple(recurrent_initial.shape)} for hidden {tuple(hidden_states.shape)}"
        )

    qkvzba, _ = _in_proj(gdn, hidden_states)
    qkvzba = qkvzba.transpose(0, 1)
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            (gdn.qk_dim * 2 + gdn.v_dim) // gdn.tp_size,
            gdn.v_dim // gdn.tp_size,
            gdn.num_value_heads // gdn.tp_size,
            gdn.num_value_heads // gdn.tp_size,
        ],
        dim=-1,
    )
    key_heads = _local_key_heads(gdn)
    value_heads = _local_value_heads(gdn)
    gate = gate.reshape(batch_size, seq_len, value_heads, gdn.value_head_dim)
    beta = beta.reshape(batch_size, seq_len, value_heads)
    alpha = alpha.reshape(batch_size, seq_len, value_heads)

    qkv = qkv.transpose(1, 2)
    qkv, conv_final = _dense_causal_conv1d_with_state(
        gdn,
        qkv,
        conv_initial,
        output_final_state=output_final_state,
    )
    qkv = qkv.transpose(1, 2)

    query, key, value = torch.split(
        qkv,
        [
            gdn.qk_dim // gdn.tp_size,
            gdn.qk_dim // gdn.tp_size,
            gdn.v_dim // gdn.tp_size,
        ],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, key_heads, gdn.key_head_dim)
    key = key.reshape(batch_size, seq_len, key_heads, gdn.key_head_dim)
    value = value.reshape(batch_size, seq_len, value_heads, gdn.value_head_dim)
    if gdn.use_qk_l2norm:
        query = _l2norm(query.contiguous())
        key = _l2norm(key.contiguous())
    if gdn.num_value_heads // gdn.num_key_heads > 1:
        repeat = gdn.num_value_heads // gdn.num_key_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    g = -gdn.A_log.exp() * F.softplus(alpha.float() + gdn.dt_bias)
    beta = beta.sigmoid()
    recurrent_out, recurrent_final = _chunk_gated_delta_rule(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=recurrent_initial,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
    )
    norm_out = _apply_gated_rms_norm(gdn, recurrent_out, gate.contiguous())
    norm_out = norm_out.reshape(batch_size, seq_len, _local_value_dim(gdn))
    norm_out = norm_out.transpose(0, 1).contiguous()
    out, out_bias = _out_proj(gdn, norm_out)
    return out, out_bias, conv_final, recurrent_final


def _dense_causal_conv1d_with_state(
    gdn: Any,
    qkv: Tensor,
    conv_initial: Tensor,
    *,
    output_final_state: bool,
) -> tuple[Tensor, Tensor | None]:
    weight = gdn.conv1d.weight.squeeze(1)
    bias = gdn.conv1d.bias
    dtype = qkv.dtype
    extended = torch.cat([conv_initial, qkv], dim=-1)
    out = F.conv1d(
        extended, weight.unsqueeze(1), bias, padding=0, groups=extended.shape[1]
    )
    out = gdn.act_fn(out[..., : qkv.shape[-1]]).to(dtype=dtype)
    tail_width = int(weight.shape[1]) - 1
    final = (
        extended[..., -tail_width:].to(dtype=dtype)
        if tail_width
        else extended[..., :0].to(dtype=dtype)
    )
    return out, final if output_final_state else None


def run_real_gdn_flattened_reference(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    execution_spec: Any | None = None,
) -> Tensor:
    spec = execution_spec or parse_gdn_prefix_tree_segments(group_ids, parent_ids)
    output = torch.zeros_like(hidden_states)
    for segment_index, segment in enumerate(spec.tree_segments):
        flat_hidden = torch.cat(
            [
                hidden_states[
                    node.start : node.end,
                    node.row_index : node.row_index + 1,
                    :,
                ]
                for node in _segment_path(spec, segment_index)
            ],
            dim=0,
        )
        flat_out, _, _, _ = _run_gdn_segment(
            gdn,
            flat_hidden,
            conv_initial=_zero_conv_state(gdn, hidden_states, segment.row_index),
            recurrent_initial=_zero_recurrent_state(
                gdn, hidden_states, segment.row_index
            ),
            output_final_state=False,
        )
        output[
            segment.start : segment.end,
            segment.row_index : segment.row_index + 1,
            :,
        ] = flat_out[-segment.length :]
    return output


def run_real_gdn_physical_stream(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
) -> Tensor:
    output = torch.zeros_like(hidden_states)
    for row in range(hidden_states.shape[1]):
        valid_length = int((group_ids[row] != -1).sum().item())
        if valid_length == 0:
            continue
        row_out, _, _, _ = _run_gdn_segment(
            gdn,
            hidden_states[:valid_length, row : row + 1, :],
            conv_initial=_zero_conv_state(gdn, hidden_states, row),
            recurrent_initial=_zero_recurrent_state(gdn, hidden_states, row),
            output_final_state=False,
        )
        output[:valid_length, row : row + 1, :] = row_out
    return output


def run_real_gdn_local_fork_reference(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    cp_size: int,
    attention_token_layout_index: TokenLayoutIndex | None = None,
) -> Tensor:
    spec = parse_gdn_prefix_tree_segments(group_ids, parent_ids)
    gdn_token_indices_by_rank = _split_gdn_families_by_rank(spec, cp_size=cp_size)
    gdn_token_ranges_by_rank = _rank_ranges_from_tokens_by_rank(
        gdn_token_indices_by_rank
    )
    plan = build_test_gdn_cp_layout_plan(
        group_ids=group_ids,
        parent_ids=parent_ids,
        cp_size=cp_size,
        attention_token_layout_index=attention_token_layout_index,
        gdn_token_ranges_by_rank=gdn_token_ranges_by_rank,
    )
    flat_hidden = hidden_states.transpose(0, 1).reshape(-1, hidden_states.shape[-1])
    attention_inputs = _rank_tensors_from_flat(
        flat_hidden, _tokens_by_rank_from_ranges(plan.attention_token_ranges_by_rank)
    )
    gdn_inputs = _simulate_all_to_all_single(attention_inputs, plan.attention_to_gdn)
    gdn_outputs = tuple(
        _run_local_fork_rank(gdn, rank_hidden, spec, local_token_indices)
        for rank_hidden, local_token_indices in zip(
            gdn_inputs,
            _tokens_by_rank_from_ranges(plan.gdn_token_ranges_by_rank),
            strict=True,
        )
    )
    attention_outputs = _simulate_all_to_all_single(gdn_outputs, plan.gdn_to_attention)
    flat_output = flat_hidden.new_zeros(flat_hidden.shape)
    for rank_output, token_indices in zip(
        attention_outputs,
        _tokens_by_rank_from_ranges(plan.attention_token_ranges_by_rank),
        strict=True,
    ):
        if token_indices:
            index = torch.tensor(
                token_indices, device=rank_output.device, dtype=torch.long
            )
            flat_output = flat_output.index_copy(0, index, rank_output)
    return (
        flat_output.reshape(group_ids.shape[0], group_ids.shape[1], -1)
        .transpose(0, 1)
        .contiguous()
    )


def _split_gdn_families_by_rank(
    spec: Any,
    *,
    cp_size: int,
) -> tuple[tuple[int, ...], ...]:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    ranks: list[list[int]] = [[] for _ in range(cp_size)]
    loads = [0] * cp_size
    for family in _tree_families(spec):
        rank = min(range(cp_size), key=lambda index: (loads[index], index))
        family_tokens = tuple(
            token
            for segment in (family.prefix, *family.completions)
            for token in _segment_linear_indices(segment, spec.sequence_length)
        )
        ranks[rank].extend(family_tokens)
        loads[rank] += len(family_tokens)
    return tuple(tuple(rank_tokens) for rank_tokens in ranks)


def _simulate_all_to_all_single(
    tensors_by_rank: tuple[Tensor, ...],
    plan: Any,
) -> tuple[Tensor, ...]:
    if len(tensors_by_rank) != int(plan.cp_size):
        raise ValueError(
            f"expected {plan.cp_size} rank tensors, got {len(tensors_by_rank)}"
        )
    sample = next((tensor for tensor in tensors_by_rank if tensor.numel()), None)
    if sample is None:
        sample = tensors_by_rank[0]
    outputs = []
    for dest_rank in range(int(plan.cp_size)):
        pieces: list[Tensor | None] = [
            None for _ in range(int(plan.dest_token_counts_by_rank[dest_rank]))
        ]
        for transfer in plan.transfers:
            if int(transfer.dest_rank) != dest_rank:
                continue
            source_tensor = tensors_by_rank[int(transfer.source_rank)]
            source_positions = _transfer_positions(
                transfer.source_positions_tensor,
                count=int(transfer.token_count),
            )
            dest_positions = _transfer_positions(
                transfer.dest_positions_tensor,
                count=int(transfer.token_count),
            )
            for source_position, dest_position in zip(
                source_positions,
                dest_positions,
                strict=True,
            ):
                pieces[dest_position] = source_tensor[source_position]
        if not pieces:
            outputs.append(sample.new_empty((0, *sample.shape[1:])))
            continue
        if any(piece is None for piece in pieces):
            raise RuntimeError(
                f"exchange plan left holes for destination rank {dest_rank}"
            )
        outputs.append(torch.stack([piece for piece in pieces if piece is not None]))
    return tuple(outputs)


def _segment_linear_indices(segment: Any, sequence_length: int) -> range:
    base = int(segment.row_index) * int(sequence_length)
    return range(base + int(segment.start), base + int(segment.end))


def _transfer_positions(tensor: Tensor | None, *, count: int) -> tuple[int, ...]:
    if tensor is None:
        return tuple(range(count))
    return tuple(int(value) for value in tensor.cpu().tolist())


def _rank_ranges_from_tokens_by_rank(
    tokens_by_rank: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    return tuple(_rank_ranges_from_tokens(tokens) for tokens in tokens_by_rank)


def _rank_ranges_from_tokens(
    tokens: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    if not tokens:
        return ()
    ranges = []
    start = tokens[0]
    end = start + 1
    position = 0
    for local_position, token in enumerate(tokens[1:], start=1):
        if token == end:
            end += 1
            continue
        ranges.append((start, end, position))
        start = token
        end = token + 1
        position = local_position
    ranges.append((start, end, position))
    return tuple(ranges)


def _tokens_by_rank_from_ranges(
    ranges_by_rank: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(token for start, end, _ in ranges for token in range(start, end))
        for ranges in ranges_by_rank
    )


def run_real_gdn_suffix_only_chain_reference(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    cp_size: int,
    mutation: GdnChainMutation | None = None,
    boundary_debug: list[GdnChainBoundaryDebug] | None = None,
) -> Tensor:
    spec = parse_gdn_prefix_tree_segments(group_ids, parent_ids)
    output = torch.zeros_like(hidden_states)
    for family in _tree_families(spec):
        row = family.row_index
        zero_conv = _zero_conv_state(gdn, hidden_states, batch_size=1)
        zero_rec = _zero_recurrent_state(gdn, hidden_states, batch_size=1)
        prefix_hidden = hidden_states[
            family.prefix.start : family.prefix.end, row : row + 1, :
        ]
        prefix_out, prefix_conv, prefix_rec = _run_gdn_segment_suffix_only_chain_shards(
            gdn,
            prefix_hidden,
            segment=family.prefix,
            cp_size=cp_size,
            conv_initial=zero_conv,
            recurrent_initial=zero_rec,
            mutation=mutation,
            boundary_debug=boundary_debug,
        )
        output[family.prefix.start : family.prefix.end, row : row + 1, :] = prefix_out
        completion_conv = prefix_conv
        completion_rec = prefix_rec
        if mutation == "detach_prefix_state":
            completion_conv = completion_conv.detach()
            completion_rec = completion_rec.detach()
        for completion in family.completions:
            completion_hidden = hidden_states[
                completion.start : completion.end, row : row + 1, :
            ]
            completion_out, _, _ = _run_gdn_segment_suffix_only_chain_shards(
                gdn,
                completion_hidden,
                segment=completion,
                cp_size=cp_size,
                conv_initial=completion_conv,
                recurrent_initial=completion_rec,
                mutation=mutation,
                boundary_debug=boundary_debug,
            )
            output[completion.start : completion.end, row : row + 1, :] = completion_out
    return output


def run_real_gdn_chunk_native_reference(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
) -> Tensor:
    spec = parse_gdn_prefix_tree_segments(group_ids, parent_ids)
    output = torch.zeros_like(hidden_states)
    for family in _tree_families(spec):
        _scatter_family_output(
            output,
            family,
            _run_gdn_family_chunk_native(gdn, hidden_states, family),
        )
    return output


def run_real_gdn_mixed_cp_reference(
    gdn: Any,
    hidden_states: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    cp_size: int,
    local_fork_max_tokens: int,
) -> Tensor:
    spec = parse_gdn_prefix_tree_segments(group_ids, parent_ids)
    output = torch.zeros_like(hidden_states)
    local_count = 0
    chain_count = 0
    for family in _tree_families(spec):
        if family.token_count <= local_fork_max_tokens:
            local_count += 1
            _scatter_family_output(
                output,
                family,
                _run_gdn_family_local_fork(gdn, hidden_states, family),
            )
            continue
        chain_count += 1
        _scatter_family_output(
            output,
            family,
            _run_gdn_family_chunk_native(gdn, hidden_states, family),
        )
    if local_count == 0 or chain_count == 0:
        raise ValueError("mixed CP reference requires both local-fork and chain work")
    return output


def _run_gdn_family_chunk_native(
    gdn: Any,
    hidden_states: Tensor,
    family: Any,
) -> Tensor:
    row = family.row_index
    prefix = family.prefix
    boundary_length = (prefix.length // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
    boundary_end = prefix.start + boundary_length
    output = hidden_states.new_zeros((family.token_count, 1, hidden_states.shape[-1]))
    boundary_conv = _zero_conv_state(gdn, hidden_states, batch_size=1)
    boundary_rec = _zero_recurrent_state(gdn, hidden_states, batch_size=1)
    if boundary_length:
        boundary_out, _, boundary_conv, boundary_rec = _run_gdn_segment(
            gdn,
            hidden_states[prefix.start : boundary_end, row : row + 1, :],
            conv_initial=boundary_conv,
            recurrent_initial=boundary_rec,
            output_final_state=True,
        )
        if boundary_conv is None or boundary_rec is None:
            raise RuntimeError("chunk-native boundary must return final states")
        output[:boundary_length] = boundary_out
    tail_hidden = hidden_states[boundary_end : prefix.end, row : row + 1, :]
    if not family.completions:
        if tail_hidden.numel():
            tail_out, _, _, _ = _run_gdn_segment(
                gdn,
                tail_hidden,
                conv_initial=boundary_conv,
                recurrent_initial=boundary_rec,
                output_final_state=False,
            )
            output[boundary_length : boundary_length + int(tail_hidden.shape[0])] = (
                tail_out
            )
        return output
    cursor = prefix.length
    for completion in family.completions:
        completion_hidden = hidden_states[
            completion.start : completion.end, row : row + 1, :
        ]
        segment_hidden = torch.cat((tail_hidden, completion_hidden), dim=0)
        segment_out, _, _, _ = _run_gdn_segment(
            gdn,
            segment_hidden,
            conv_initial=boundary_conv,
            recurrent_initial=boundary_rec,
            output_final_state=False,
        )
        tail_length = int(tail_hidden.shape[0])
        if completion.child_index == 0 and tail_length:
            output[boundary_length : prefix.length] = segment_out[:tail_length]
        next_cursor = cursor + completion.length
        output[cursor:next_cursor] = segment_out[tail_length:]
        cursor = next_cursor
    return output


def _masked_quadratic_loss(output: Tensor, assistant_mask: Tensor) -> Tensor:
    selected = output.transpose(0, 1)[assistant_mask]
    if selected.numel() == 0:
        raise ValueError("assistant_mask selects no tokens")
    return selected.square().sum()


def _run_local_fork_rank(
    gdn: Any,
    rank_hidden: Tensor,
    spec: Any,
    local_token_indices: tuple[int, ...],
) -> Tensor:
    if not local_token_indices:
        return rank_hidden.new_empty(rank_hidden.shape)
    local_group_ids, local_parent_ids = _local_fork_group_tensors(
        spec, local_token_indices, device=rank_hidden.device
    )
    local_output, _ = gdn_prefix_tree_forward(
        gdn,
        rank_hidden.unsqueeze(1).contiguous(),
        group_ids=local_group_ids,
        parent_ids=local_parent_ids,
    )
    return local_output.squeeze(1)


def _run_gdn_family_local_fork(
    gdn: Any,
    hidden_states: Tensor,
    family: Any,
) -> Tensor:
    row = family.row_index
    segments = (family.prefix, *family.completions)
    local_hidden = torch.cat(
        [
            hidden_states[segment.start : segment.end, row : row + 1, :]
            for segment in segments
        ],
        dim=0,
    )
    local_group_ids, local_parent_ids = _family_group_tensors(
        family, device=hidden_states.device
    )
    local_output, _ = gdn_prefix_tree_forward(
        gdn,
        local_hidden,
        group_ids=local_group_ids,
        parent_ids=local_parent_ids,
    )
    return local_output


def _scatter_family_output(output: Tensor, family: Any, family_output: Tensor) -> None:
    row = family.row_index
    cursor = 0
    for segment in (family.prefix, *family.completions):
        next_cursor = cursor + segment.length
        output[segment.start : segment.end, row : row + 1, :] = family_output[
            cursor:next_cursor
        ]
        cursor = next_cursor


def _family_group_tensors(
    family: Any,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    group_ids = []
    parent_ids = []
    local_group_by_global: dict[int, int] = {}
    for local_group_id, (segment, global_index, parent_index) in enumerate(
        zip(
            (family.prefix, *family.completions),
            family.segment_indices,
            family.parent_indices,
            strict=True,
        )
    ):
        local_group_by_global[global_index] = local_group_id
        local_parent_id = (
            local_group_id if parent_index < 0 else local_group_by_global[parent_index]
        )
        group_ids.extend([local_group_id] * segment.length)
        parent_ids.extend([local_parent_id] * segment.length)
    return (
        torch.tensor([group_ids], device=device, dtype=torch.long),
        torch.tensor([parent_ids], device=device, dtype=torch.long),
    )


def _run_gdn_segment_suffix_only_chain_shards(
    gdn: Any,
    hidden_states: Tensor,
    *,
    segment: Any,
    cp_size: int,
    conv_initial: Tensor,
    recurrent_initial: Tensor,
    mutation: GdnChainMutation | None,
    boundary_debug: list[GdnChainBoundaryDebug] | None,
) -> tuple[Tensor, Tensor, Tensor]:
    outputs = []
    conv_state = conv_initial
    recurrent_state = recurrent_initial
    for shard_index, (start, end) in enumerate(
        _non_empty_shard_offsets(segment.length, cp_size)
    ):
        shard_conv = conv_state
        shard_rec = recurrent_state
        if mutation == "zero_conv_tail" and shard_index > 0:
            shard_conv = torch.zeros_like(shard_conv)
        if (
            mutation == "zero_recurrent_parent"
            and segment.kind == "completion"
            and shard_index == 0
        ):
            shard_rec = torch.zeros_like(shard_rec)
        _capture_chain_boundary(
            boundary_debug,
            segment=segment,
            shard_index=shard_index,
            token_offset=start,
            conv_initial=shard_conv,
            recurrent_initial=shard_rec,
        )
        shard_out, _, conv_final, recurrent_final = _run_gdn_segment(
            gdn,
            hidden_states[start:end],
            conv_initial=shard_conv,
            recurrent_initial=shard_rec,
            output_final_state=True,
        )
        if conv_final is None or recurrent_final is None:
            raise RuntimeError("GDN chain shards require final states")
        outputs.append(shard_out)
        conv_state = conv_final
        recurrent_state = recurrent_final
    if not outputs:
        raise ValueError("GDN chain segment must contain at least one token")
    return torch.cat(outputs, dim=0), conv_state, recurrent_state


def _capture_chain_boundary(
    boundary_debug: list[GdnChainBoundaryDebug] | None,
    *,
    segment: Any,
    shard_index: int,
    token_offset: int,
    conv_initial: Tensor,
    recurrent_initial: Tensor,
) -> None:
    if boundary_debug is None:
        return
    is_parent_boundary = segment.kind == "completion" and shard_index == 0
    is_shard_boundary = shard_index > 0
    if not is_parent_boundary and not is_shard_boundary:
        return
    if conv_initial.requires_grad:
        conv_initial.retain_grad()
    if recurrent_initial.requires_grad:
        recurrent_initial.retain_grad()
    boundary_debug.append(
        GdnChainBoundaryDebug(
            family_index=segment.family_index,
            segment_kind=segment.kind,
            child_index=segment.child_index,
            boundary_kind="parent" if is_parent_boundary else "shard",
            shard_index=shard_index,
            token_offset=token_offset,
            conv_initial=conv_initial,
            recurrent_initial=recurrent_initial,
        )
    )


def _non_empty_shard_offsets(
    length: int,
    cp_size: int,
) -> tuple[tuple[int, int], ...]:
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    return tuple(
        (start, end)
        for rank in range(cp_size)
        for start, end in [
            ((length * rank) // cp_size, (length * (rank + 1)) // cp_size)
        ]
        if start < end
    )


def _local_fork_group_tensors(
    spec: Any,
    local_token_indices: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    local_position = {
        token_index: position
        for position, token_index in enumerate(local_token_indices)
    }
    group_ids = torch.full(
        (len(local_token_indices),), -1, device=device, dtype=torch.long
    )
    parent_ids = torch.full_like(group_ids, -1)
    next_group_id = 0
    for family in _tree_families(spec):
        family_segments = (family.prefix, *family.completions)
        family_tokens = tuple(
            token_index
            for segment in family_segments
            for token_index in _segment_linear_indices(segment, spec.sequence_length)
        )
        token_is_local = tuple(
            token_index in local_position for token_index in family_tokens
        )
        if not any(token_is_local):
            continue
        if not all(token_is_local):
            raise ValueError("local-fork execution requires whole prompt families")

        group_by_segment_index: dict[int, int] = {}
        for segment, global_index, parent_index in zip(
            family_segments,
            family.segment_indices,
            family.parent_indices,
            strict=True,
        ):
            group_id = next_group_id
            next_group_id += 1
            group_by_segment_index[global_index] = group_id
            parent_group_id = (
                group_id if parent_index < 0 else group_by_segment_index[parent_index]
            )
            for token_index in _segment_linear_indices(segment, spec.sequence_length):
                position = local_position[token_index]
                group_ids[position] = group_id
                parent_ids[position] = parent_group_id
    if torch.any(group_ids == -1):
        raise RuntimeError("local-fork metadata left unassigned token rows")
    return group_ids.unsqueeze(0), parent_ids.unsqueeze(0)


def _rank_tensors_from_flat(
    flat: Tensor,
    indices_by_rank: tuple[tuple[int, ...], ...],
) -> tuple[Tensor, ...]:
    return tuple(
        flat.index_select(
            0,
            torch.tensor(indices, device=flat.device, dtype=torch.long),
        )
        for indices in indices_by_rank
    )


def _require_grad(tensor: Tensor) -> Tensor:
    if tensor.grad is None:
        raise AssertionError("expected tensor.grad to be populated")
    return tensor.grad


def parameter_grad_mean_abs_pct(left: torch.nn.Module, right: torch.nn.Module) -> float:
    return parameter_grad_mean_abs_pct_with_name(left, right)[1]
