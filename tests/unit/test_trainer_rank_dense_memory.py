"""Dense recompute and no-grad group pricing; CPU admission math, not a bound.

Qwen3.8-27B CP2 allocator traces: the recomputed layer's peak holds its MLP
FC1 stage (7F per row) and one input gradient, and no-grad groups run one
after another.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from test_trainer_rank_checkpoint_memory import price, rank, requests
import torch

from art.trainer_rank._impl import (
    _dense_mlp_recompute_bytes_per_token,
    _GroupLayout,
    _MemorySignature,
)

HIDDEN, LAYERS, FFN = 2048, 40, 5632
STAGE = 7 * FFN * 2


def _module(cls):
    value = cls.__new__(cls)
    torch.nn.Module.__init__(value)
    return value


def _dense_layer() -> Any:
    """The supported gated MLP, from the real owner types."""
    pytest.importorskip("art.megatron.lora")
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.transformer.mlp import MLP

    from art.megatron import lora as lora_module

    mlp = _module(MLP)
    mlp.config = SimpleNamespace(
        ffn_hidden_size=FFN,
        gated_linear_unit=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        sequence_parallel=False,
        fp8=None,
        fp4=None,
        cuda_graph_impl="none",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    mlp.activation_func = torch.nn.functional.silu
    fc1 = _module(lora_module.SharedExpertsLinearFC1LoRA)
    fc1.linear_fc1 = _module(TELayerNormColumnParallelLinear)
    fc1.gate_lora = _module(lora_module.LoRA)
    fc1.up_lora = _module(lora_module.LoRA)
    fc1.non_gated = False
    fc1.out_features = 2 * FFN
    fc2 = _module(lora_module.SharedExpertsLinearFC2LoRA)
    row = _module(lora_module.SelfAttentionLinearProjLoRA)
    row.lora = _module(lora_module.LoRA)
    row.linear_proj = _module(TERowParallelLinear)
    fc2.row_parallel_lora = row
    mlp.linear_fc1 = fc1
    mlp.linear_fc2 = fc2
    layer = torch.nn.Module()
    layer.mlp = mlp
    return layer


def _dense_model(layers: list[Any]) -> Any:
    from megatron.core.transformer.transformer_block import TransformerBlock

    block = _module(TransformerBlock)
    block.layers = torch.nn.ModuleList(layers)
    model: Any = torch.nn.Module()
    model.decoder = block
    model._preprocess = lambda: None  # Marks a GPT model for _language_model.
    return model


def _dense_rank(stage: int = STAGE):
    r = rank()
    r._moe_output_bytes_per_token = r._moe_checkpoint_grad_bytes_per_token = 0
    r._moe_recompute_covered = False
    r._dense_recompute_bytes_per_token = stage
    return r


def test_supported_dense_mlp_prices_its_traced_stage():
    layers = [_dense_layer() for _ in range(3)]
    assert _dense_mlp_recompute_bytes_per_token([_dense_model(layers)]) == STAGE


@pytest.mark.parametrize(
    "change",
    [
        "hook",
        "non_gated",
        "activation",
        "fp8",
        "tensor_parallel",
        "bias",
        "out_features",
        "other_layer",
        "unwrapped_fc1",
        "chunks",
    ],
)
def test_any_unsupported_layer_keeps_the_boundary_allowance(change):
    layers = [_dense_layer() for _ in range(3)]
    mlp = layers[1].mlp
    if change == "hook":
        mlp.linear_fc1.register_forward_hook(lambda *args: None)
    elif change == "non_gated":
        mlp.linear_fc1.non_gated = True
    elif change == "activation":
        mlp.activation_func = torch.nn.functional.gelu
    elif change == "fp8":
        mlp.config.fp8 = "hybrid"
    elif change == "tensor_parallel":
        mlp.config.tensor_model_parallel_size = 2
    elif change == "bias":
        mlp.config.add_bias_linear = True
    elif change == "out_features":
        mlp.linear_fc1.out_features = FFN
    elif change == "other_layer":
        layers[1].mlp = torch.nn.Linear(1, 1)
    else:
        mlp.linear_fc1 = mlp.linear_fc1.linear_fc1
    models = [_dense_model(layers)] * (2 if change == "chunks" else 1)
    assert _dense_mlp_recompute_bytes_per_token(models) == 0


@pytest.mark.parametrize("rows", [67, 1024])
def test_covered_dense_recompute_charges_one_gradient_and_its_stage(rows):
    r = _dense_rank()
    # A short no-grad reference keeps its workspace below the gradient stage.
    values = r._estimate_flat_forward(requests(rows, 16))
    cost = price(r, values)
    assert cost.checkpoint_input_gradient == rows * HIDDEN * 2
    # The recomputed mixer keeps its activations beside the residual, the
    # norm output and the MLP stage.
    per_row = r._recomputed_mixer_bytes_per_token() + 2 * HIDDEN * 2 + STAGE
    assert cost.checkpoint_workspace >= rows * per_row
    assert cost.required >= int(
        (
            cost.checkpoint_retained
            + cost.checkpoint_workspace
            + cost.checkpoint_input_gradient
        )
        * 1.1
    )
    # Without the traced stage, dense keeps one gradient per boundary.
    plain = price(_dense_rank(0), values)
    assert plain.checkpoint_input_gradient == rows * LAYERS * HIDDEN * 2
    assert plain.checkpoint_workspace < cost.checkpoint_workspace


def test_dense_stage_is_not_used_beyond_cp2(monkeypatch):
    r = _dense_rank()
    values = r._estimate_flat_forward(requests(67, 4096))
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 4, 1))
    assert r._dense_recompute_stage_bytes() == 0
    cost = price(r, values)
    assert cost.checkpoint_input_gradient == 67 * LAYERS * HIDDEN * 2


def test_layout_floor_prices_the_dense_stage_per_layer_type():
    r = _dense_rank()
    layers = [SimpleNamespace() for _ in range(4)]
    layout = _GroupLayout(
        attention_rows=(100, 80), gdn_rows=None, attention_retained=(0, 0)
    )
    dense = r._layout_checkpoint_floor(layers, (None,), (None,), (layout,))
    plain = _dense_rank(0)._layout_checkpoint_floor(layers, (None,), (None,), (layout,))
    assert dense[0] == plain[0] == 100 * 4 * HIDDEN * 2
    # The busiest rank's recomputed layer adds its residual, norm and stage.
    assert sum(dense) - sum(plain) == 100 * (2 * HIDDEN * 2 + STAGE)


def _no_grad_required(r, group_rows, *, topology=(1, 1, 2, 1), grad=False):
    signature = _MemorySignature(
        topology, (1, None), len(group_rows), (), grad, (grad,) * len(group_rows)
    )
    return r._estimate_required_memory_bytes_from_values(
        packed_tokens=40_000,
        output_bytes=0,
        signature=signature,
        logical_tokens=40_000,
        group_rows=tuple((rows, grad) for rows in group_rows),
    )


def test_no_grad_groups_price_only_the_largest_one():
    dense, plain = _dense_rank(), _dense_rank(0)
    # One group: the traced per-packed-token floor, unchanged.
    assert _no_grad_required(dense, (12_000,)) == _no_grad_required(plain, (12_000,))
    # Two sequential groups: only the larger group's share of the packed rows.
    two = _no_grad_required(plain, (12_000, 8_000))
    assert _no_grad_required(dense, (12_000, 8_000)) == pytest.approx(
        two * 12_000 / 20_000, rel=1e-3
    )


@pytest.mark.parametrize("case", ["cp1", "cp4", "unsupported"])
def test_no_grad_group_floor_needs_the_traced_shape(case):
    dense = _dense_rank(0 if case == "unsupported" else STAGE)
    kwargs: dict[str, Any] = {
        "cp1": {"topology": (1, 1, 1, 1)},
        "cp4": {"topology": (1, 1, 4, 1)},
        "unsupported": {},
    }[case]
    plain = _dense_rank(0)
    assert _no_grad_required(dense, (12_000, 8_000), **kwargs) == _no_grad_required(
        plain, (12_000, 8_000), **kwargs
    )
