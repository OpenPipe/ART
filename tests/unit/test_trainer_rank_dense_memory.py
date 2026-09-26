"""Dense recompute and no-grad group pricing; CPU admission math, not a bound.

Qwen3.8-27B CP2 allocator traces: the recomputed layer's peak holds its MLP
FC1 stage and one input gradient, and no-grad groups run one after another.
"""

from dataclasses import replace
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from test_trainer_rank_checkpoint_memory import price, rank, requests
from test_trainer_rank_moe_memory import _rank as _moe_rank
from test_trainer_rank_moe_memory import layer  # noqa: F401
import torch

from art.trainer_rank import _impl
from art.trainer_rank._impl import (
    _dense_mlp_recompute_bytes_per_token,
    _GroupLayout,
    _MemorySignature,
)

HIDDEN, LAYERS, FFN, RANK = 2048, 40, 5632, 8
CP2 = (1, 1, 2, 1)
# 7F traced FC1 stage + one more FC1 triplet (6F) of early-run recompile
# residue, plus the adapters' rank-wide intermediates.
STAGE = (13 * FFN + 6 * RANK) * 2
# Three 2F FC1 tensors, residual/norm/CP-gather rows, rank intermediates.
NO_GRAD = (6 * FFN + 6 * HIDDEN + 6 * RANK) * 2


def _module(cls):
    value = cls.__new__(cls)
    torch.nn.Module.__init__(value)
    return value


def _adapter(lora_module, inputs: int, outputs: int, rank: int = RANK):
    lora = _module(lora_module.LoRA)
    lora.A_T = torch.nn.Parameter(torch.empty(inputs, rank, dtype=torch.bfloat16))
    lora.B_T = torch.nn.Parameter(torch.empty(rank, outputs, dtype=torch.bfloat16))
    return lora


def _dense_layer(gdn: bool = False) -> Any:
    """The traced gated MLP, from the real owner types."""
    pytest.importorskip("art.megatron.lora")
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from art.megatron import lora as lora_module

    mlp = _module(MLP)
    mlp.config = SimpleNamespace(
        hidden_size=HIDDEN,
        ffn_hidden_size=FFN,
        gated_linear_unit=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        sequence_parallel=False,
        bias_activation_fusion=True,  # Bridge's Qwen3.5 providers fuse SwiGLU.
        use_te_activation_func=False,
        cpu_offloading=False,
        cuda_graph_impl="none",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        fp8=None,
        fp4=None,
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=None,
        glu_linear_offset=0.0,
    )
    mlp.activation_func = torch.nn.functional.silu
    fc1 = _module(lora_module.SharedExpertsLinearFC1LoRA)
    fc1.linear_fc1 = _module(TELayerNormColumnParallelLinear)
    fc1.gate_lora = _adapter(lora_module, HIDDEN, FFN)
    fc1.up_lora = _adapter(lora_module, HIDDEN, FFN)
    fc1.non_gated = False
    fc1.out_features = 2 * FFN
    fc2 = _module(lora_module.SharedExpertsLinearFC2LoRA)
    row = _module(lora_module.SelfAttentionLinearProjLoRA)
    row.lora = _adapter(lora_module, FFN, HIDDEN)
    row.linear_proj = _module(TERowParallelLinear)
    fc2.row_parallel_lora = row
    mlp.linear_fc1 = fc1
    mlp.linear_fc2 = fc2
    layer = _module(TransformerLayer)
    layer.self_attention = _module(GatedDeltaNet if gdn else SelfAttention)
    layer.mlp = mlp
    return layer


def _wrap_like_art(layer: Any) -> None:
    """ART's GDN island and prefix-tree wrappers, as the traced run had them."""
    from art.megatron.gdn.operator import (
        _gdn_island_layer_forward,
        _prefix_tree_forward,
    )

    # Training compile then wraps the delegate (training/compile.py).
    layer._art_gdn_island_physical_forward = torch.compile(layer.forward)
    layer.forward = MethodType(_gdn_island_layer_forward, layer)
    mixer = layer.self_attention
    if type(mixer).__name__ == "GatedDeltaNet":
        mixer._art_physical_forward = mixer.forward
        mixer.forward = MethodType(_prefix_tree_forward, mixer)


def _dense_model(layers: list[Any]) -> Any:
    from megatron.core.transformer.transformer_block import TransformerBlock

    block = _module(TransformerBlock)
    block.layers = torch.nn.ModuleList(layers)
    model: Any = torch.nn.Module()
    model.decoder = block
    model._preprocess = lambda: None  # Marks a GPT model for _language_model.
    return model


def _dense_rank(stage: int = STAGE, no_grad: int = NO_GRAD):
    r = rank()
    r._moe_output_bytes_per_token = r._moe_checkpoint_grad_bytes_per_token = 0
    r._moe_recompute_covered = False
    r._dense_recompute_bytes_per_token = stage
    r._dense_no_grad_bytes_per_token = no_grad
    return r


def _at_cp2(r):
    # After cheap estimation (which declines under CP): price as a CP2 rank.
    r._topology_key = lambda: CP2
    return r


def test_traced_dense_mlp_prices_its_stage_and_no_grad_transient():
    # A GDN/attention hybrid with ART's wrappers, as traced.
    layers = [_dense_layer(gdn=index != 2) for index in range(3)]
    for layer in layers:
        _wrap_like_art(layer)
    model = _dense_model(layers)
    assert _dense_mlp_recompute_bytes_per_token([model]) == (STAGE, NO_GRAD)
    assert _dense_mlp_recompute_bytes_per_token([model], hidden_size=HIDDEN + 1) == (
        0,
        0,
    )
    # Qwen3.8-27B at rank 8: F 17,408, H 5,120.
    assert (13 * 17408 + 48) * 2 == 452_704


@pytest.mark.parametrize(
    "change",
    [
        "fc1_hook",
        "row_hook",
        "lora_hook",
        "base_hook",
        "layer_hook",
        "layer_forward",
        "layer_delegate",
        "compiled_custom_delegate",
        "layer_type",
        "mixer_delegate",
        "mixer_class_forward",
        "mixer_hook",
        "mixer_forward",
        "no_mixer",
        "decoder_hook",
        "decoder_subclass",
        "non_gated",
        "activation",
        "config_activation",
        "clamp",
        "linear_offset",
        "unfused_activation",
        "te_activation",
        "fp8",
        "fp4",
        "dtype",
        "tensor_parallel",
        "pipeline_parallel",
        "sequence_parallel",
        "cuda_graphs",
        "bias",
        "out_features",
        "other_layer",
        "unwrapped_fc1",
        "unfused_norm",
        "rank",
        "chunks",
    ],
)
def test_anything_but_the_traced_execution_keeps_the_allowance(change):
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from art.megatron import lora as lora_module

    layers = [_dense_layer(gdn=index != 1) for index in range(3)]
    for wrapped in layers:
        _wrap_like_art(wrapped)
    model = _dense_model(layers)
    layer = layers[1]
    gdn_layer = layers[0]
    mlp = layer.mlp
    config = mlp.config

    def hook(module):
        return lambda: module.register_forward_hook(lambda *args: None)

    class Block(TransformerBlock):
        pass

    class Layer(TransformerLayer):
        pass

    class Attention(SelfAttention):
        def forward(self, *args, **kwargs):  # A class-level override.
            return super().forward(*args, **kwargs)

    custom = MethodType(lambda self, *a, **k: None, layer)

    edits = {
        "fc1_hook": hook(mlp.linear_fc1),
        "row_hook": hook(mlp.linear_fc2.row_parallel_lora),
        "lora_hook": hook(mlp.linear_fc1.gate_lora),
        "base_hook": hook(mlp.linear_fc1.linear_fc1),
        "layer_hook": lambda: layer.register_forward_pre_hook(lambda *args: None),
        "layer_forward": lambda: setattr(
            layer, "forward", MethodType(lambda self, *a: None, layer)
        ),
        "layer_delegate": lambda: setattr(
            layer, "_art_gdn_island_physical_forward", custom
        ),
        "compiled_custom_delegate": lambda: setattr(
            layer, "_art_gdn_island_physical_forward", torch.compile(custom)
        ),
        "layer_type": lambda: setattr(layer, "__class__", Layer),
        "mixer_delegate": lambda: setattr(
            gdn_layer.self_attention,
            "_art_physical_forward",
            MethodType(lambda self, *a: None, gdn_layer.self_attention),
        ),
        "mixer_class_forward": lambda: setattr(
            layer.self_attention, "__class__", Attention
        ),
        "mixer_hook": hook(layer.self_attention),
        "mixer_forward": lambda: setattr(
            layer.self_attention, "forward", lambda *a: None
        ),
        "no_mixer": lambda: delattr(layer, "self_attention"),
        "decoder_hook": hook(model.decoder),
        "decoder_subclass": lambda: setattr(model.decoder, "__class__", Block),
        "non_gated": lambda: setattr(mlp.linear_fc1, "non_gated", True),
        "activation": lambda: setattr(mlp, "activation_func", torch.nn.functional.gelu),
        "config_activation": lambda: setattr(
            config, "activation_func", torch.nn.functional.gelu
        ),
        "clamp": lambda: setattr(config, "activation_func_clamp_value", 7.0),
        "linear_offset": lambda: setattr(config, "glu_linear_offset", 1.0),
        "unfused_activation": lambda: setattr(config, "bias_activation_fusion", False),
        "te_activation": lambda: setattr(config, "use_te_activation_func", True),
        "fp8": lambda: setattr(config, "fp8", "hybrid"),
        "fp4": lambda: setattr(config, "fp4", "nvfp4"),
        "dtype": lambda: setattr(config, "params_dtype", torch.float16),
        "tensor_parallel": lambda: setattr(config, "tensor_model_parallel_size", 2),
        "pipeline_parallel": lambda: setattr(config, "pipeline_model_parallel_size", 2),
        "sequence_parallel": lambda: setattr(config, "sequence_parallel", True),
        "cuda_graphs": lambda: setattr(config, "cuda_graph_impl", "local"),
        "bias": lambda: setattr(config, "add_bias_linear", True),
        "out_features": lambda: setattr(mlp.linear_fc1, "out_features", FFN),
        "other_layer": lambda: setattr(layer, "mlp", torch.nn.Linear(1, 1)),
        "unwrapped_fc1": lambda: setattr(mlp, "linear_fc1", mlp.linear_fc1.linear_fc1),
        "unfused_norm": lambda: setattr(
            mlp.linear_fc1, "linear_fc1", _module(TEColumnParallelLinear)
        ),
        "rank": lambda: setattr(
            mlp.linear_fc1, "up_lora", _adapter(lora_module, HIDDEN, FFN, 512)
        ),
        "chunks": lambda: None,
    }
    edits[change]()
    models = [model] * (2 if change == "chunks" else 1)
    assert _dense_mlp_recompute_bytes_per_token(models) == (0, 0)


def test_moe_models_never_price_the_dense_stage(monkeypatch, layer):
    calls = []
    monkeypatch.setattr(
        _impl,
        "_dense_mlp_recompute_bytes_per_token",
        lambda *a, **k: calls.append(1) or (STAGE, NO_GRAD),
    )
    moe = _moe_rank(layer)
    assert moe._moe_layers and not calls
    assert (
        moe._dense_recompute_bytes_per_token,
        moe._dense_no_grad_bytes_per_token,
    ) == (0, 0)


def test_an_active_slot_beyond_the_priced_rank_keeps_the_allowance(monkeypatch):
    r = _at_cp2(_dense_rank())
    policy = r._slot_ref("policy")
    assert r._dense_mlp_widths((None,)) == (STAGE, NO_GRAD)
    monkeypatch.setattr(
        _impl, "_dense_mlp_recompute_bytes_per_token", lambda *a, **k: (0, 0)
    )
    assert r._dense_mlp_widths((policy,)) == (0, 0)
    wider = (STAGE + 96, NO_GRAD + 96)
    monkeypatch.setattr(
        _impl, "_dense_mlp_recompute_bytes_per_token", lambda *a, **k: wider
    )
    assert r._dense_mlp_widths((policy,)) == wider


@pytest.mark.parametrize("rows", [67, 1024])
def test_covered_dense_recompute_charges_one_gradient_and_its_stage(rows):
    r = _dense_rank()
    # A short no-grad reference keeps its transient below the gradient stage.
    values = r._estimate_flat_forward(requests(rows, 16))
    _at_cp2(r)
    cost = price(r, values)
    assert cost.checkpoint_input_gradient == rows * HIDDEN * 2
    # The recomputed mixer keeps its activations beside the residual, the
    # norm output and the MLP stage (with its recompile residue); with no
    # learned profile the floor alone carries it.
    assert r._memory_profiles.get(values[2]) is None
    per_row = r._recomputed_mixer_bytes_per_token() + 2 * HIDDEN * 2 + STAGE
    assert cost.checkpoint_workspace >= rows * per_row + r._te_workspace_growth_bytes()
    assert cost.required >= int(
        (
            cost.checkpoint_retained
            + cost.checkpoint_workspace
            + cost.checkpoint_input_gradient
        )
        * 1.1
    )
    # Without the traced stage, dense keeps one gradient per boundary.
    plain = price(_at_cp2(_dense_rank(0, 0)), values)
    assert plain.checkpoint_input_gradient == rows * LAYERS * HIDDEN * 2


def test_a_larger_no_grad_group_keeps_its_transient_beside_gradient_boundaries():
    r = _dense_rank()
    values = r._estimate_flat_forward(requests(1000, 3000))
    cost = price(_at_cp2(r), values)
    boundaries = 1000 * LAYERS * HIDDEN * 2
    # The later no-grad group's FC1 transient runs beside the retained graph.
    assert cost.checkpoint_workspace >= 3000 * NO_GRAD
    assert cost.required >= int((boundaries + 3000 * NO_GRAD + 1000 * HIDDEN * 2) * 1.1)


@pytest.mark.parametrize("topology", [(1, 1, 1, 1), (1, 1, 4, 1)])
def test_dense_widths_apply_only_at_cp2(topology):
    r = _dense_rank()
    values = r._estimate_flat_forward(requests(67, 16))
    r._topology_key = lambda: topology
    assert r._dense_mlp_widths() == (0, 0)
    assert price(r, values).checkpoint_input_gradient == 67 * LAYERS * HIDDEN * 2


@pytest.mark.parametrize("gdn", [False, True])
def test_layout_floor_prices_the_dense_stage_per_layer_type(gdn):
    r = _at_cp2(_dense_rank())
    plain = _at_cp2(_dense_rank(0, 0))
    if gdn:
        for x in (r, plain):
            x._gdn_layers = 3
            x._geometry = replace(
                x._geometry,
                gdn_key_heads=4,
                gdn_key_head_dim=64,
                gdn_value_heads=8,
                gdn_value_head_dim=64,
            )
    layers = [
        SimpleNamespace(
            _art_gdn_island_boundary=SimpleNamespace(
                input_layout="gdn" if gdn and index else "attention"
            )
        )
        for index in range(4)
    ]
    layout = _GroupLayout(
        attention_rows=(100, 80),
        gdn_rows=(120, 60) if gdn else None,
        attention_retained=(0, 0),
    )
    dense = r._layout_checkpoint_floor(layers, (None,), (None,), (layout,))
    base = plain._layout_checkpoint_floor(layers, (None,), (None,), (layout,))
    assert dense[0] == base[0]
    # Every layer type's recomputed stage gains the residual, norm and MLP
    # stage on its own rows: the busiest rank's largest stage grows by it.
    widths = r._recomputed_mixer_widths(stage_buffers=False)
    rows = {"attention": 100, "gdn": 120} if gdn else {"attention": 100}
    grown = max(rows[k] * (widths[k] + 2 * HIDDEN * 2 + STAGE) for k in rows)
    plain_stage = max(rows[k] * widths[k] for k in rows)
    assert (
        sum(dense) - sum(base) == grown - plain_stage + r._te_workspace_growth_bytes()
    )


def _no_grad_required(r, group_rows, *, topology=CP2, packed=40_000):
    signature = _MemorySignature(
        topology, (1, None), len(group_rows), (), False, (False,) * len(group_rows)
    )
    return r._estimate_required_memory_bytes_from_values(
        packed_tokens=packed,
        output_bytes=0,
        signature=signature,
        logical_tokens=packed,
        group_rows=tuple((rows, False) for rows in group_rows),
    )


def test_no_grad_groups_price_the_largest_groups_own_rows():
    dense, plain = _at_cp2(_dense_rank()), _at_cp2(_dense_rank(0, 0))
    # Today's floor: H bytes per packed token times the layer-count factor.
    per_token = HIDDEN * 2 * min(16, LAYERS // 4 + 4)
    te = dense._te_workspace_growth_bytes()
    # One group: today's per-packed-token floor, unchanged.
    assert _no_grad_required(dense, (12_000,)) == _no_grad_required(plain, (12_000,))
    for groups, packed in (((12_000, 8_000), 40_000), ((58_240, 29_120), 119_119)):
        largest = max(groups)
        # The largest group's own physical rows at the traced width (with
        # TE's workspace growth), and at least its share of today's
        # per-packed-token floor.
        expected = max(
            largest * NO_GRAD + te, -(-packed * per_token * largest // sum(groups))
        )
        assert _no_grad_required(dense, groups, packed=packed) == int(expected * 1.1)
        assert _no_grad_required(dense, groups, packed=packed) < _no_grad_required(
            plain, groups, packed=packed
        )
    # A narrow structural width never drops below the rows' traced need.
    wide = _at_cp2(_dense_rank(STAGE, 10**6))
    assert _no_grad_required(wide, (12_000, 8_000)) == int((12_000 * 10**6 + te) * 1.1)


@pytest.mark.parametrize("case", ["cp1", "cp4", "unsupported"])
def test_no_grad_group_floor_needs_the_traced_shape(case):
    topology = {"cp1": (1, 1, 1, 1), "cp4": (1, 1, 4, 1)}.get(case, CP2)
    dense = _dense_rank(*((0, 0) if case == "unsupported" else (STAGE, NO_GRAD)))
    plain = _dense_rank(0, 0)
    for r in (dense, plain):
        r._topology_key = lambda: topology
    assert _no_grad_required(
        dense, (12_000, 8_000), topology=topology
    ) == _no_grad_required(plain, (12_000, 8_000), topology=topology)


def test_one_unsupported_slot_keeps_both_allowances(monkeypatch):
    """A no-grad reference slot outside the priced rank must not leave the
    gradient group's boundaries on one gradient without the dense stage."""
    r = _at_cp2(_dense_rank())
    policy, reference = r._slot_ref("policy"), r._slot_ref("reference")
    monkeypatch.setattr(
        _impl,
        "_dense_mlp_recompute_bytes_per_token",
        lambda model, ref, **k: (0, 0) if ref == reference else (STAGE, NO_GRAD),
    )
    groups = ((1000, True), (3000, False))
    both = (policy, reference)
    assert (
        r._checkpoint_input_gradient_bytes(groups, both) == 1000 * LAYERS * HIDDEN * 2
    )
    retained, workspace = r._checkpoint_memory_floor(groups, both)
    assert workspace < 3000 * NO_GRAD  # the dense widths fell back together
    supported = (policy, policy)
    assert r._checkpoint_input_gradient_bytes(groups, supported) == 1000 * HIDDEN * 2


def test_no_grad_only_waves_charge_te_workspace_growth():
    r = _at_cp2(_dense_rank())
    _, dense = r._checkpoint_memory_floor(((500, False),))
    _, plain = _at_cp2(_dense_rank(0, 0))._checkpoint_memory_floor(((500, False),))
    assert dense == 500 * NO_GRAD + r._te_workspace_growth_bytes()
    assert plain == 500 * 4 * HIDDEN * 2


def test_the_traced_qwen_attention_mixer_is_accepted():
    bridge = pytest.importorskip(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention"
    )
    layers = [_dense_layer(gdn=index != 1) for index in range(3)]
    qwen = _module(bridge.Qwen3VLSelfAttention)
    layers[1].self_attention = qwen  # Overrides forward, as traced.
    for layer in layers:
        _wrap_like_art(layer)
    assert _dense_mlp_recompute_bytes_per_token([_dense_model(layers)]) == (
        STAGE,
        NO_GRAD,
    )
