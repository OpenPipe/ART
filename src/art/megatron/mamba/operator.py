from __future__ import annotations

from functools import cache
from importlib import import_module
from importlib.metadata import version

from pydantic import BaseModel, ConfigDict, Field
import torch
import torch.nn.functional as F

from .causal_conv import causal_conv1d
from .permutation import permute_rows
from .plan import MambaConvBucket, MambaExecutionPlan, MambaScanBucket
from .tree_kernels import assemble_rows, assemble_scan_outputs, gather_scan_inputs

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


class _SplitScanInputs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        dense: torch.Tensor,
        heads: int,
        head_dim: int,
        groups: int,
        state_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del ctx
        inner = heads * head_dim
        state_width = groups * state_dim
        return (
            dense[..., :inner].view(*dense.shape[:2], heads, head_dim),
            dense[..., inner : inner + state_width].view(
                *dense.shape[:2], groups, state_dim
            ),
            dense[..., inner + state_width : inner + 2 * state_width].view(
                *dense.shape[:2], groups, state_dim
            ),
            dense[..., inner + 2 * state_width :],
        )

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: object,
        grad_x: torch.Tensor,
        grad_b: torch.Tensor,
        grad_c: torch.Tensor,
        grad_dt: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        del ctx
        return (
            torch.cat(
                (
                    grad_x.flatten(2),
                    grad_b.flatten(2),
                    grad_c.flatten(2),
                    grad_dt,
                ),
                dim=-1,
            ),
            None,
            None,
            None,
            None,
        )


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
    _, conv_input, dt = torch.split(
        projected, [inner, inner + 2 * groups, heads], dim=-1
    )
    convolved = _run_convolution(conv_input, plan, params)
    scan_inputs = gather_scan_inputs(
        convolved, dt, plan.scan_token_positions, plan.scan_token_occurrences
    )
    outputs: list[torch.Tensor] = []
    output_rows: list[torch.Tensor] = []
    output_positions: list[torch.Tensor] = []
    offset = 0
    states: list[torch.Tensor] = []
    zero_state = torch.zeros(
        (heads, params.head_dim, params.state_dim),
        dtype=torch.float32,
        device=projected.device,
    )
    for phase in plan.scan_phases:
        for bucket in phase:
            parents = _parent_states(
                bucket.parent_state_indices, bucket.parent_rows, states, zero_state
            )
            rows = bucket.batch_size * bucket.length
            dense_output, final = _run_scan_bucket(
                scan_inputs.narrow(0, offset, rows),
                bucket,
                parents,
                params,
                plan.chunk_size,
            )
            offset += rows
            outputs.append(dense_output)
            output_rows.append(bucket.output_rows)
            output_positions.append(bucket.output_positions)
            if bucket.needs_final_state:
                if final is None:
                    raise RuntimeError("Mamba SSD omitted a required boundary state")
                states.append(final)
    return assemble_scan_outputs(
        outputs, output_rows, output_positions, plan.tree.token_count
    )


def _run_convolution(
    conv_input: torch.Tensor,
    plan: MambaExecutionPlan,
    params: MambaParameters,
) -> torch.Tensor:
    compacted = permute_rows(conv_input, plan.conv_token_positions)
    outputs = []
    offset = 0
    states: list[torch.Tensor] = []
    zero_state = conv_input.new_zeros(
        (int(params.conv_weight.shape[0]), int(params.conv_weight.shape[1]) - 1)
    )
    for bucket in plan.conv_buckets:
        length = int(bucket.token_indices.numel())
        compact = compacted.narrow(0, offset, length)
        offset += length
        parents = _parent_states(
            bucket.parent_indices, bucket.parent_rows, states, zero_state
        )
        batch = len(bucket.segment_indices)
        channels = int(compact.shape[1])
        dense = compact.view(batch, length // batch, channels).transpose(1, 2)
        if compact.dtype == torch.float32:
            full = torch.cat((parents, dense), dim=-1)
            compact_output = F.silu(
                F.conv1d(
                    full,
                    params.conv_weight.unsqueeze(1),
                    params.conv_bias,
                    groups=channels,
                )
            )
            state_length = int(params.conv_weight.shape[1]) - 1
            final = full[..., -state_length:] if state_length else full[..., :0]
        else:
            initial = parents.transpose(1, 2).contiguous().transpose(1, 2)
            bias = params.conv_bias
            if bias is None:
                bias = params.conv_weight.new_zeros(channels)
            compact_output, final = causal_conv1d(
                dense,
                params.conv_weight,
                bias,
                initial,
            )
        compact_output = compact_output.transpose(1, 2).contiguous().flatten(0, 1)
        outputs.append(compact_output)
        states.append(final)
    return assemble_rows(
        outputs,
        [bucket.token_indices for bucket in plan.conv_buckets],
        plan.tree.token_count,
    )


def _run_scan_bucket(
    scan_inputs: torch.Tensor,
    bucket: MambaScanBucket,
    initial_states: torch.Tensor,
    params: MambaParameters,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    heads = int(params.dt_bias.numel())
    dense = scan_inputs.view(bucket.batch_size, bucket.length, -1)
    x, b, c, dt = _SplitScanInputs.apply(
        dense, heads, params.head_dim, params.num_groups, params.state_dim
    )
    result = _mamba_chunk_scan_combined()(
        x,
        dt,
        -torch.exp(params.a_log.float()),
        b,
        c,
        chunk_size,
        D=params.d.float(),
        z=None,
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
    return output.view(bucket.batch_size, bucket.length, heads * params.head_dim), final


def _parent_states(
    parent_indices: tuple[int, ...],
    parent_rows: torch.Tensor,
    states: list[torch.Tensor],
    zero: torch.Tensor,
) -> torch.Tensor:
    if all(parent < 0 for parent in parent_indices):
        return zero.unsqueeze(0).expand(int(parent_rows.numel()), *zero.shape)
    available = states[0] if len(states) == 1 else torch.cat(states)
    if all(parent >= 0 for parent in parent_indices):
        return available.index_select(0, parent_rows)
    output = zero.unsqueeze(0).expand(int(parent_rows.numel()), *zero.shape).clone()
    positions = (parent_rows >= 0).nonzero().flatten()
    return output.index_copy(
        0, positions, available.index_select(0, parent_rows[positions])
    )


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
