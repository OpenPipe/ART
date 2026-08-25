from __future__ import annotations

from functools import cache
from importlib import import_module
from importlib.metadata import version
from typing import cast

from pydantic import BaseModel, ConfigDict, Field
import torch

from art.megatron.gdn.conv_gelu import packed_varlen_causal_conv

from .plan import MambaConvBucket, MambaExecutionPlan, MambaScanBucket

MAMBA_SSM_VERSION = "2.3.2.post1"


class MambaParameters(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    conv_weight: torch.Tensor
    conv_bias: torch.Tensor | None
    dt_bias: torch.Tensor
    a_log: torch.Tensor
    d: torch.Tensor
    head_dim: int = Field(gt=0)
    state_dim: int = Field(gt=0)
    num_groups: int = Field(gt=0)


def run_mamba_tree(
    projected: torch.Tensor,
    plan: MambaExecutionPlan,
    params: MambaParameters,
) -> torch.Tensor:
    """Run physical convolution once and canonical chunked SSD along every branch."""

    _validate_inputs(projected, plan, params)
    heads = int(params.dt_bias.numel())
    inner = heads * params.head_dim
    groups = params.num_groups * params.state_dim
    z, conv_input, dt = torch.split(
        projected, [inner, inner + 2 * groups, heads], dim=-1
    )
    convolved = _run_convolution(conv_input, plan, params)
    postconv = torch.cat((z, convolved, dt), dim=-1)
    output = projected.new_zeros((plan.tree.token_count, inner))
    states: list[torch.Tensor | None] = []
    zero_state = torch.zeros(
        (heads, params.head_dim, params.state_dim),
        dtype=torch.float32,
        device=projected.device,
    )
    for phase in plan.scan_phases:
        for bucket in phase:
            parents = _parent_states(
                bucket.parent_state_indices, states, zero_state, "SSD"
            )
            dense_output, final = _run_scan_bucket(
                postconv, bucket, parents, params, plan.chunk_size
            )
            write = bucket.output_mask
            output = output.index_copy(
                0,
                bucket.token_indices[write],
                dense_output[write],
            )
            if bucket.needs_final_state:
                if final is None:
                    raise RuntimeError("Mamba SSD omitted a required boundary state")
                _store_states(states, bucket.state_indices, final)
    return output


def _run_convolution(
    conv_input: torch.Tensor,
    plan: MambaExecutionPlan,
    params: MambaParameters,
) -> torch.Tensor:
    output = torch.zeros_like(conv_input)
    states: list[torch.Tensor | None] = []
    zero_state = conv_input.new_zeros(
        (int(params.conv_weight.shape[0]), int(params.conv_weight.shape[1]) - 1)
    )
    for bucket in plan.conv_buckets:
        compact = conv_input.index_select(0, bucket.token_indices)
        parents = _parent_states(
            bucket.parent_indices, states, zero_state, "convolution"
        )
        compact_output, final = packed_varlen_causal_conv(
            compact,
            bucket.cu_seqlens,
            parents,
            params.conv_weight,
            params.conv_bias,
            activation="silu",
            output_final_state=True,
        )
        if final is None:
            raise RuntimeError("Mamba convolution omitted a required boundary state")
        output = output.index_copy(0, bucket.token_indices, compact_output)
        _store_states(states, bucket.segment_indices, final)
    return output


def _run_scan_bucket(
    postconv: torch.Tensor,
    bucket: MambaScanBucket,
    initial_states: torch.Tensor,
    params: MambaParameters,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    heads = int(params.dt_bias.numel())
    inner = heads * params.head_dim
    groups = params.num_groups * params.state_dim
    dense = postconv.index_select(0, bucket.token_indices.flatten()).view(
        bucket.batch_size, bucket.length, -1
    )
    z, convolved, dt = torch.split(dense, [inner, inner + 2 * groups, heads], dim=-1)
    x, b, c = torch.split(convolved, [inner, groups, groups], dim=-1)
    dt = dt.masked_fill(~bucket.real_mask.unsqueeze(-1), -torch.inf)
    result = _mamba_chunk_scan_combined()(
        x.view(bucket.batch_size, bucket.length, heads, params.head_dim),
        dt,
        -torch.exp(params.a_log.float()),
        b.view(bucket.batch_size, bucket.length, params.num_groups, params.state_dim),
        c.view(bucket.batch_size, bucket.length, params.num_groups, params.state_dim),
        chunk_size,
        D=params.d.float(),
        z=z.view(bucket.batch_size, bucket.length, heads, params.head_dim),
        dt_bias=params.dt_bias.float(),
        initial_states=initial_states,
        dt_softplus=True,
        return_final_states=bucket.needs_final_state,
        state_dtype=torch.float32,
    )
    if bucket.needs_final_state:
        output, final = result
    else:
        output, final = result, None
    return output.view(bucket.batch_size, bucket.length, inner), final


def _parent_states(
    parent_indices: tuple[int, ...],
    states: list[torch.Tensor | None],
    zero: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    selected = []
    missing = []
    for parent in parent_indices:
        if parent < 0:
            selected.append(zero)
        elif parent >= len(states) or states[parent] is None:
            missing.append(parent)
        else:
            selected.append(cast(torch.Tensor, states[parent]))
    if missing:
        raise RuntimeError(f"Mamba {kind} is missing parent states {missing}")
    return torch.stack(selected)


def _store_states(
    states: list[torch.Tensor | None],
    indices: tuple[int, ...],
    values: torch.Tensor,
) -> None:
    if int(values.shape[0]) != len(indices):
        raise ValueError("Mamba state rows do not match the execution plan")
    if indices:
        states.extend(None for _ in range(max(indices) + 1 - len(states)))
    for row, index in enumerate(indices):
        states[index] = values[row]


def _validate_inputs(
    projected: torch.Tensor,
    plan: MambaExecutionPlan,
    params: MambaParameters,
) -> None:
    heads = int(params.dt_bias.numel())
    expected_width = (
        2 * heads * params.head_dim + 2 * params.num_groups * params.state_dim + heads
    )
    if tuple(projected.shape) != (plan.tree.token_count, expected_width):
        raise ValueError(
            "Mamba projected input has the wrong token/feature shape: "
            f"got {tuple(projected.shape)}, expected {(plan.tree.token_count, expected_width)}"
        )
    conv_channels = heads * params.head_dim + 2 * params.num_groups * params.state_dim
    if tuple(params.conv_weight.shape[:1]) != (conv_channels,):
        raise ValueError("Mamba convolution channels do not match x/B/C")
    if params.a_log.shape != params.dt_bias.shape:
        raise ValueError("Mamba A_log and dt_bias must contain one value per head")
    if params.d.shape not in (params.dt_bias.shape, (heads, params.head_dim)):
        raise ValueError("Mamba D must be per head or per head dimension")


@cache
def _mamba_chunk_scan_combined():
    if version("mamba-ssm") != MAMBA_SSM_VERSION:
        raise RuntimeError(
            f"ART Mamba requires mamba-ssm {MAMBA_SSM_VERSION}, got "
            f"{version('mamba-ssm')}"
        )
    return import_module("mamba_ssm.ops.triton.ssd_combined").mamba_chunk_scan_combined
