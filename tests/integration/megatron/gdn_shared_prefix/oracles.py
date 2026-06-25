from __future__ import annotations

from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field
import torch
from torch import Tensor
import torch.nn.functional as F

from art.megatron.gdn.gdn_shared_prefix import GdnPackedExecutionSpec, GdnSegmentSpec

from .metrics import (
    mean_abs_pct,
    parameter_grad_mean_abs_pct_with_name,
    stable_output_mse_loss,
)
from .parser_import import parse_gdn_shared_prefix_segments


class ToyGdnConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hidden_size: int = Field(default=8, ge=1)
    conv_width: int = Field(default=4, ge=2)


class ToyOracleMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    loss_mean_abs_pct: float
    output_mean_abs_pct: float
    hidden_grad_mean_abs_pct: float
    param_grad_mean_abs_pct: float


class ToyStatefulGdn(torch.nn.Module):
    """Small stateful block used to validate oracle mechanics on CPU.

    This is not a GDN approximation. It deliberately has the two state classes
    that make GDN shared-prefix execution non-trivial: a finite conv tail and a
    recurrent state. That is enough to prove parser routing, flattened
    accumulation, and known-bad physical-stream sensitivity before the real FLA
    kernels are invoked.
    """

    def __init__(self, config: ToyGdnConfig) -> None:
        super().__init__()
        self.config = config
        self.in_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.rec_proj = torch.nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.out_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.conv_weight = torch.nn.Parameter(
            torch.empty(config.hidden_size, config.conv_width)
        )
        self.conv_bias = torch.nn.Parameter(torch.empty(config.hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.conv_weight, mean=0.0, std=0.15)
        torch.nn.init.normal_(self.conv_bias, mean=0.0, std=0.05)
        for module in (self.in_proj, self.gate_proj, self.rec_proj, self.out_proj):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def zero_conv_state(self, reference: Tensor) -> Tensor:
        return reference.new_zeros(
            self.config.hidden_size,
            self.config.conv_width - 1,
        )

    def zero_recurrent_state(self, reference: Tensor) -> Tensor:
        return reference.new_zeros(self.config.hidden_size)

    def forward_segment(
        self,
        hidden: Tensor,
        *,
        conv_initial: Tensor,
        recurrent_initial: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        projected = self.in_proj(hidden)
        conv_input = torch.cat([conv_initial, projected.T], dim=1)
        conv_out = F.conv1d(
            conv_input.unsqueeze(0),
            self.conv_weight.unsqueeze(1),
            self.conv_bias,
            padding=0,
            groups=self.config.hidden_size,
        ).squeeze(0)
        conv_out = F.silu(conv_out.T)
        conv_final = conv_input[:, -(self.config.conv_width - 1) :]

        recurrent = recurrent_initial
        outputs = []
        gates = torch.sigmoid(self.gate_proj(hidden))
        for token_index in range(hidden.shape[0]):
            recurrent = torch.tanh(recurrent + self.rec_proj(conv_out[token_index]))
            outputs.append(self.out_proj(recurrent * gates[token_index]))
        return torch.stack(outputs), conv_final, recurrent


def run_toy_packed(
    module: ToyStatefulGdn,
    hidden: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
) -> Tensor:
    spec = parse_gdn_shared_prefix_segments(group_ids, parent_ids)
    output = torch.zeros_like(hidden)
    conv_states: list[Tensor] = []
    rec_states: list[Tensor] = []
    for segment_index, segment in enumerate(spec.tree_segments):
        row = segment.row_index
        parent_index = spec.tree_parent_indices[segment_index]
        if parent_index < 0:
            conv_initial = module.zero_conv_state(hidden)
            rec_initial = module.zero_recurrent_state(hidden)
        else:
            conv_initial = conv_states[parent_index]
            rec_initial = rec_states[parent_index]
        segment_out, conv_final, rec_final = module.forward_segment(
            hidden[row, segment.start : segment.end],
            conv_initial=conv_initial,
            recurrent_initial=rec_initial,
        )
        output[row, segment.start : segment.end] = segment_out
        conv_states.append(conv_final)
        rec_states.append(rec_final)
    return output


def run_toy_flattened_reference(
    module: ToyStatefulGdn,
    hidden: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
) -> Tensor:
    spec = parse_gdn_shared_prefix_segments(group_ids, parent_ids)
    output = torch.zeros_like(hidden)
    for segment_index, segment in enumerate(spec.tree_segments):
        path = _segment_path(spec, segment_index)
        flattened = torch.cat(
            [hidden[node.row_index, node.start : node.end] for node in path],
            dim=0,
        )
        flat_out, _, _ = module.forward_segment(
            flattened,
            conv_initial=module.zero_conv_state(hidden),
            recurrent_initial=module.zero_recurrent_state(hidden),
        )
        segment_len = segment.length
        output[segment.row_index, segment.start : segment.end] = flat_out[-segment_len:]
    return output


def _segment_path(
    spec: GdnPackedExecutionSpec,
    segment_index: int,
) -> tuple[GdnSegmentSpec, ...]:
    indices = []
    cursor = segment_index
    while cursor >= 0:
        indices.append(cursor)
        cursor = spec.tree_parent_indices[cursor]
    return tuple(spec.tree_segments[index] for index in reversed(indices))


def run_toy_physical_stream(
    module: ToyStatefulGdn,
    hidden: Tensor,
    *,
    group_ids: Tensor,
) -> Tensor:
    output = torch.zeros_like(hidden)
    for row in range(hidden.shape[0]):
        valid_length = int((group_ids[row] != -1).sum().item())
        if valid_length == 0:
            continue
        row_out, _, _ = module.forward_segment(
            hidden[row, :valid_length],
            conv_initial=module.zero_conv_state(hidden),
            recurrent_initial=module.zero_recurrent_state(hidden),
        )
        output[row, :valid_length] = row_out
    return output


def compare_toy_packed_to_flattened(
    module: ToyStatefulGdn,
    hidden: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    assistant_mask: Tensor,
) -> ToyOracleMetrics:
    packed_module = deepcopy(module)
    flat_module = deepcopy(module)
    packed_hidden = hidden.clone().detach().requires_grad_(True)
    flat_hidden = hidden.clone().detach().requires_grad_(True)

    packed_out = run_toy_packed(
        packed_module,
        packed_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    flat_out = run_toy_flattened_reference(
        flat_module,
        flat_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    packed_loss = _masked_quadratic_loss(packed_out, assistant_mask)
    flat_loss = _masked_quadratic_loss(flat_out, assistant_mask)
    packed_loss.backward()
    flat_loss.backward()

    return ToyOracleMetrics(
        loss_mean_abs_pct=mean_abs_pct(flat_loss.detach(), packed_loss.detach()),
        output_mean_abs_pct=mean_abs_pct(flat_out.detach(), packed_out.detach()),
        hidden_grad_mean_abs_pct=mean_abs_pct(
            _require_grad(flat_hidden), _require_grad(packed_hidden)
        ),
        param_grad_mean_abs_pct=parameter_grad_mean_abs_pct_with_name(
            flat_module, packed_module
        )[1],
    )


def compare_toy_packed_to_flattened_with_output_grad(
    module: ToyStatefulGdn,
    hidden: Tensor,
    *,
    group_ids: Tensor,
    parent_ids: Tensor,
    output_grad: Tensor,
) -> ToyOracleMetrics:
    packed_module = deepcopy(module)
    flat_module = deepcopy(module)
    packed_hidden = hidden.clone().detach().requires_grad_(True)
    flat_hidden = hidden.clone().detach().requires_grad_(True)

    packed_out = run_toy_packed(
        packed_module,
        packed_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    flat_out = run_toy_flattened_reference(
        flat_module,
        flat_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    real_mask = group_ids != -1
    real_mask = (
        real_mask.unsqueeze(-1)
        if output_grad.shape[:2] == real_mask.shape
        else real_mask.transpose(0, 1).unsqueeze(-1)
    )
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

    return ToyOracleMetrics(
        loss_mean_abs_pct=mean_abs_pct(flat_loss.detach(), packed_loss.detach()),
        output_mean_abs_pct=mean_abs_pct(flat_out.detach(), packed_out.detach()),
        hidden_grad_mean_abs_pct=mean_abs_pct(
            _require_grad(flat_hidden), _require_grad(packed_hidden)
        ),
        param_grad_mean_abs_pct=parameter_grad_mean_abs_pct_with_name(
            flat_module, packed_module
        )[1],
    )


def _masked_quadratic_loss(output: Tensor, assistant_mask: Tensor) -> Tensor:
    selected = output[assistant_mask]
    if selected.numel() == 0:
        raise ValueError("assistant_mask selects no tokens")
    return selected.square().sum()


def _require_grad(tensor: Tensor) -> Tensor:
    if tensor.grad is None:
        raise AssertionError("expected tensor.grad to be populated")
    return tensor.grad
