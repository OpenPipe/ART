"""Dense CP2 recompute and no-grad group pricing; CPU admission math, not a bound.

Qwen3.8-27B CP2 allocator traces: the recomputed layer's peak holds its MLP
FC1 stage and one input gradient on each rank's own layouts, and no-grad
groups run one after another.
"""

from dataclasses import replace
import functools
import math
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from test_trainer_rank_checkpoint_memory import price, rank, requests
from test_trainer_rank_moe_memory import _rank as _moe_rank
from test_trainer_rank_moe_memory import layer  # noqa: F401
import torch

from art.trainer_rank import ForwardInput, _impl
from art.trainer_rank._impl import (
    _COLD_RECOMPUTE_TRANSIENT_BYTES as COLD,
)
from art.trainer_rank._impl import (
    _TE_CUBLAS_WORKSPACE_BYTES,
    _dense_mlp_recompute_bytes_per_token,
    _GroupLayout,
    _MemorySignature,
)

HIDDEN, LAYERS, FFN, RANK = 2048, 40, 5632, 8
CP2 = (1, 1, 2, 1)
# The 7F traced FC1 stage (no recompile residue), H/4 of q/k norm outputs
# (3.25H of norms with the residual, norm output and input gradient), plus
# the adapters' rank-wide intermediates.
STAGE = (7 * FFN + HIDDEN // 4 + 6 * RANK) * 2
# Three 2F FC1 tensors beside 4.25H of residual pair, embedding and norm
# output (the CP1 floor's 6F + 4H form), rank intermediates.
NO_GRAD = (6 * FFN + 4 * HIDDEN + HIDDEN // 4 + 6 * RANK) * 2


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
    from transformer_engine.pytorch import RMSNorm

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
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    row.linear_proj.get_rng_state_tracker = get_cuda_rng_tracker
    fc2.row_parallel_lora = row
    mlp.linear_fc1 = fc1
    mlp.linear_fc2 = fc2
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp

    layer = _module(TransformerLayer)
    layer.self_attention = _module(GatedDeltaNet if gdn else SelfAttention)
    if gdn:
        layer.self_attention.act_fn = torch.nn.functional.silu
        layer.self_attention.in_proj = _module(TELayerNormColumnParallelLinear)
        layer.self_attention.out_norm = _module(RMSNorm)
        layer.self_attention.out_proj = _module(TERowParallelLinear)
    layer.mlp = mlp
    # The stock spec's bias-dropout-add factories, grad context and cross
    # attention.
    layer.self_attn_bda = layer.mlp_bda = get_bias_dropout_add
    layer.bias_dropout_add_exec_handler = torch.enable_grad
    layer.cross_attn_bda = _module(IdentityFuncOp)
    layer.cross_attention = _module(IdentityOp)
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
    block.group_prefetch_offload_commit_async = lambda x: x  # TE, offload off.
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
    # After cheap estimation (which declines under CP): price as a CP2 rank,
    # standing in for ART's CP attention and GDN islands (``art_cp``).
    r._topology_key = lambda: CP2
    r._layout_pricing_supported = lambda topology: (
        topology[1:] == (1, 2, 1) and bool(r._dense_mlp_widths()[0])
    )
    return r


def test_traced_dense_mlp_prices_its_stage_and_no_grad_transient():
    # A GDN/attention hybrid with ART's wrappers, as traced.
    layers = [_dense_layer(gdn=index != 2) for index in range(3)]
    for layer in layers:
        _wrap_like_art(layer)
    model = _dense_model(layers)
    assert _dense_mlp_recompute_bytes_per_token([model]) == (STAGE, NO_GRAD)
    # The decoder's final norm, untouched or behind ART's empty-safe wrapper.
    from art.megatron.gdn.operator import _empty_safe_norm_forward

    norm: Any = torch.nn.LayerNorm(HIDDEN)
    model.decoder.final_layernorm = norm
    assert _dense_mlp_recompute_bytes_per_token([model]) == (STAGE, NO_GRAD)
    norm._art_empty_safe_norm_physical_forward = norm.forward
    norm.forward = MethodType(_empty_safe_norm_forward, norm)
    assert _dense_mlp_recompute_bytes_per_token([model]) == (STAGE, NO_GRAD)
    assert _dense_mlp_recompute_bytes_per_token([model], hidden_size=HIDDEN + 1) == (
        0,
        0,
    )


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
        "mixer_adapter_rank",
        "mixer_child_hook",
        "mixer_child_forward",
        "norm_delegate",
        "slot_selector",
        "active_selector",
        "child_class_forward",
        "helper_override",
        "helper_partial",
        "helper_callable",
        "final_norm_hook",
        "final_norm_class_forward",
        "final_norm_override",
        "custom_mlp_bda",
        "custom_self_attn_bda",
        "custom_cross_attn_bda",
        "custom_bda_handler",
        "function_cross_attention",
        "gdn_activation",
        "function_in_proj",
        "function_out_norm",
        "function_out_proj",
        "function_linear_proj",
        "backward_hook",
        "backward_pre_hook",
        "global_forward_hook",
        "global_backward_hook",
        "gdn_trace_hooks",
        "cudagraph_manager",
        "fused_residual_norm",
        "fused_residual_norm_subclass",
        "encoder_layer_norm",
        "encoder_final_norm",
        "mlp_norm",
        "custom_rng_tracker",
        "decoder_callback",
        "chunks",
    ],
)
def test_anything_but_the_traced_execution_keeps_the_allowance(change):
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEFusedResidualRMSNorm,
    )
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from art.megatron import lora as lora_module
    from art.megatron.gdn import operator as gdn_operator
    from art.megatron.gdn.operator import _empty_safe_norm_forward

    assert TEFusedResidualRMSNorm is not None
    fused_norm_subclass = type("Norm", (TEFusedResidualRMSNorm,), {})

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

    class Norm(torch.nn.LayerNorm):
        def forward(self, input):  # A class-level forward outside the traced packages.
            return input

    custom = MethodType(lambda self, *a, **k: None, layer)

    def mixer_child(edit):
        # A child the mixer runs, such as its core attention or a norm.
        def apply():
            child = torch.nn.LayerNorm(4)
            edit(child)
            layer.self_attention.core_attention = child

        return apply

    def final_norm(edit):
        # The decoder's final norm runs after the layers.
        def apply():
            norm = torch.nn.LayerNorm(4)
            edit(norm)
            model.decoder.final_layernorm = norm

        return apply

    # Stock torch.nn and Megatron modules a norm builder could return, each
    # running a configured activation that could retain tensors.
    def retaining(x):
        return torch.nn.functional.silu(x)

    def encoder():
        return torch.nn.TransformerEncoderLayer(8, 1, 8, activation=retaining)

    mlp_norm = _module(MLP)
    mlp_norm.activation_func = retaining

    def foreign_norm_delegate(norm):
        norm._art_empty_safe_norm_physical_forward = MethodType(lambda self, x: x, norm)
        norm.forward = MethodType(_empty_safe_norm_forward, norm)

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
        "mixer_adapter_rank": lambda: setattr(
            layer.self_attention, "qkv_lora", _adapter(lora_module, HIDDEN, HIDDEN, 512)
        ),
        "mixer_child_hook": mixer_child(
            lambda child: child.register_forward_hook(lambda *args: None)
        ),
        "mixer_child_forward": mixer_child(
            lambda child: setattr(child, "forward", MethodType(lambda s, x: x, child))
        ),
        "norm_delegate": mixer_child(foreign_norm_delegate),
        # Execution selects tensors through the instance; the gate must too.
        "slot_selector": lambda: setattr(
            mlp.linear_fc1.gate_lora, "_slot", lambda ref: None
        ),
        "active_selector": lambda: setattr(
            mlp.linear_fc1.gate_lora, "active_lora_tensors", lambda: None
        ),
        "child_class_forward": mixer_child(
            lambda child: setattr(child, "__class__", Norm)
        ),
        "helper_override": lambda: setattr(layer, "_forward_mlp", custom),
        "helper_partial": lambda: setattr(
            layer, "_forward_mlp", functools.partial(lambda *a: None)
        ),
        "helper_callable": lambda: setattr(
            layer, "_forward_mlp", type("Helper", (), {"__call__": lambda *a: None})()
        ),
        "final_norm_hook": final_norm(
            lambda norm: norm.register_forward_hook(lambda *args: None)
        ),
        "final_norm_class_forward": final_norm(
            lambda norm: setattr(norm, "__class__", Norm)
        ),
        "final_norm_override": final_norm(
            lambda norm: setattr(norm, "forward", MethodType(lambda s, x: x, norm))
        ),
        "custom_mlp_bda": lambda: setattr(layer, "mlp_bda", lambda *a: None),
        "custom_self_attn_bda": lambda: setattr(
            gdn_layer, "self_attn_bda", functools.partial(lambda *a: None)
        ),
        # A function in place of the cross-attention BDA module.
        "custom_cross_attn_bda": lambda: (
            delattr(layer, "cross_attn_bda"),
            setattr(layer, "cross_attn_bda", lambda *a: None),
        ),
        "custom_bda_handler": lambda: setattr(
            layer, "bias_dropout_add_exec_handler", torch.no_grad
        ),
        "function_cross_attention": lambda: (
            delattr(layer, "cross_attention"),
            setattr(layer, "cross_attention", lambda *a, **k: None),
        ),
        "gdn_activation": lambda: setattr(
            gdn_layer.self_attention, "act_fn", torch.nn.functional.gelu
        ),
        **{
            f"function_{name}": functools.partial(
                lambda name: (
                    delattr(gdn_layer.self_attention, name),
                    setattr(gdn_layer.self_attention, name, lambda *a: None),
                ),
                name,
            )
            for name in ("in_proj", "out_norm", "out_proj")
        },
        "function_linear_proj": lambda: setattr(
            layer.self_attention, "linear_proj", lambda *a: None
        ),
        "backward_hook": lambda: mlp.linear_fc1.register_full_backward_hook(
            lambda *a: None
        ),
        "backward_pre_hook": lambda: layer.register_full_backward_pre_hook(
            lambda *a: None
        ),
        "global_forward_hook": lambda: cleanup.append(
            torch.nn.modules.module.register_module_forward_hook(lambda *a: None)
        ),
        "global_backward_hook": lambda: cleanup.append(
            torch.nn.modules.module.register_module_full_backward_hook(lambda *a: None)
        ),
        "gdn_trace_hooks": lambda: cleanup.append(
            SimpleNamespace(
                remove=functools.partial(
                    gdn_operator.set_gdn_trace_token_uid_hooks,
                    gdn_operator.set_gdn_trace_token_uid_hooks(object()),
                )
            )
        ),
        "cudagraph_manager": lambda: setattr(layer, "cudagraph_manager", object()),
        "fused_residual_norm": mixer_child(
            lambda child: setattr(child, "__class__", TEFusedResidualRMSNorm)
        ),
        "fused_residual_norm_subclass": mixer_child(
            lambda child: setattr(child, "__class__", fused_norm_subclass)
        ),
        "encoder_layer_norm": lambda: setattr(layer, "input_layernorm", encoder()),
        "encoder_final_norm": lambda: setattr(
            model.decoder, "final_layernorm", encoder()
        ),
        "mlp_norm": lambda: setattr(layer, "pre_mlp_layernorm", mlp_norm),
        "custom_rng_tracker": lambda: setattr(
            mlp.linear_fc2.row_parallel_lora.linear_proj,
            "get_rng_state_tracker",
            lambda: None,
        ),
        "decoder_callback": lambda: setattr(model.decoder, "callback", retaining),
        "chunks": lambda: None,
    }
    cleanup: list[Any] = []
    try:
        edits[change]()
        models = [model] * (2 if change == "chunks" else 1)
        assert _dense_mlp_recompute_bytes_per_token(models) == (0, 0)
    finally:
        for handle in cleanup:
            handle.remove()


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


def test_an_active_slot_beyond_the_priced_rank_keeps_mains_pricing(monkeypatch):
    r = _at_cp2(_dense_rank())
    policy, reference = r._slot_ref("policy"), r._slot_ref("reference")
    assert r._dense_mlp_widths((None,)) == (STAGE, NO_GRAD)
    monkeypatch.setattr(
        _impl,
        "_dense_mlp_recompute_bytes_per_token",
        lambda model, ref, **k: (0, 0) if ref == reference else (STAGE, NO_GRAD),
    )
    assert r._dense_mlp_widths((reference,)) == (0, 0)
    # One such gradient slot keeps the whole wave off one gradient and layouts.
    groups = ((1000, True), (800, True))
    layouts = _layouts(groups)
    assert r._dense_layout_floors(groups, (policy, reference), layouts) is None
    assert r._dense_layout_floors(groups, (policy, policy), layouts) is not None
    wider = (STAGE + 96, NO_GRAD + 96)
    monkeypatch.setattr(
        _impl, "_dense_mlp_recompute_bytes_per_token", lambda *a, **k: wider
    )
    assert r._dense_mlp_widths((policy,)) == wider


def _layouts(groups, retained=0):
    """Two-rank layouts with the busiest rank holding each group's rows."""
    return tuple(
        _GroupLayout((rows, rows // 2), None, (retained, retained // 2))
        for rows, _ in groups
    )


def layout_price(r, values, layouts):
    n, out, signature, groups, head = values
    return r._subforward_cost(
        packed_tokens=n,
        output_bytes=out,
        signature=signature,
        logical_tokens=n,
        group_rows=groups,
        group_layouts=layouts,
        head_workspace_bytes=head,
    )


@pytest.mark.parametrize("rows", [67, 1024])
def test_covered_dense_recompute_charges_one_gradient_and_its_stage(rows):
    r = _dense_rank()
    values = r._estimate_flat_forward(requests(rows, 16)[:1])
    _at_cp2(r)
    layouts = _layouts(values[3], retained=4096)
    cost = layout_price(r, values, layouts)
    assert cost.checkpoint_input_gradient == rows * HIDDEN * 2
    # The busiest rank's boundaries, and its recomputed mixer beside the
    # residual, the norm output and the MLP stage; with no learned profile
    # the floor alone carries it.
    assert r._memory_profiles.get(values[2]) is None
    stage = rows * (r._dense_mixer_widths()["attention"] + 2 * HIDDEN * 2 + STAGE)
    assert cost.checkpoint_retained == values[1] + rows * LAYERS * HIDDEN * 2
    assert cost.checkpoint_workspace == (
        stage + 4096 + r._te_workspace_growth_bytes() + COLD
    )
    assert cost.required == int(
        (
            cost.checkpoint_retained
            + cost.checkpoint_workspace
            + cost.checkpoint_input_gradient
        )
        * 1.1
    )
    # The dense head shares the decoder stage (it is not staged).
    head = layout_price(r, (*values[:4], 10**9), layouts)
    assert head.checkpoint_workspace == 10**9 + r._te_workspace_growth_bytes() + COLD
    # Without layouts or the traced stage, dense keeps main's pricing: one
    # gradient per boundary.
    assert price(r, values).checkpoint_input_gradient == rows * LAYERS * HIDDEN * 2
    plain = layout_price(_at_cp2(_dense_rank(0, 0)), values, layouts)
    assert plain == price(_at_cp2(_dense_rank(0, 0)), values)
    assert plain.checkpoint_input_gradient == rows * LAYERS * HIDDEN * 2


def test_mixed_waves_keep_mains_pricing():
    """Gradient beside no-grad groups was not traced at CP2: no layouts, no
    single gradient, and no CP2 no-grad stage."""
    values = _dense_rank()._estimate_flat_forward(requests(1000, 3000))
    dense, plain = _at_cp2(_dense_rank()), _at_cp2(_dense_rank(0, 0))
    layouts = _layouts(values[3])
    assert dense._dense_layout_floors(values[3], None, layouts) is None
    assert layout_price(dense, values, layouts) == price(plain, values)


@pytest.mark.parametrize(
    "topology,sequence_parallel",
    [
        ((1, 1, 1, 1), False),
        ((1, 1, 4, 1), False),
        ((1, 2, 2, 1), False),
        ((1, 2, 2, 1), True),
        ((1, 4, 2, 1), True),
    ],
)
def test_other_topologies_keep_mains_pricing(monkeypatch, topology, sequence_parallel):
    """Away from TP1/CP2 (CP1, CP4 and the TP x SP floor) dense coverage
    changes no gradient, mixed or no-grad price."""
    # Cheap estimates decline under CP: take them at CP1.
    mixed = _dense_rank()._estimate_flat_forward(requests(67, 16))
    gradient = _dense_rank()._estimate_flat_forward(requests(67, 16)[:1])
    costs = []
    for r in (_dense_rank(), _dense_rank(0, 0)):
        r._topology_key = lambda: topology
        decoder = _impl._language_model(r.runtime.model[0]).decoder
        decoder.config.sequence_parallel = sequence_parallel
        monkeypatch.setattr(r, "_sequence_parallel_floor_covered", lambda *_: True)
        assert r._dense_mlp_widths() == (0, 0)
        costs.append(
            (
                price(r, mixed),
                price(r, gradient),
                _no_grad_required(r, (12_000, 8_000), topology=topology),
            )
        )
    assert costs[0] == costs[1]


@pytest.mark.parametrize("gdn", [False, True])
def test_layout_floor_prices_each_rank_and_layer_type(gdn):
    r = _at_cp2(_dense_rank())
    r._geometry = replace(
        r._geometry, num_attention_heads=16, num_query_groups=2, kv_channels=256
    )
    if gdn:
        r._gdn_layers = 3
        r._geometry = replace(
            r._geometry,
            gdn_key_heads=4,
            gdn_key_head_dim=64,
            gdn_value_heads=8,
            gdn_value_head_dim=64,
            gdn_conv_kernel=4,
        )
    # Each layer's saved input layout (``_layer_gdn_inputs``).
    inputs = tuple(bool(gdn and index) for index in range(4))
    r._layer_gdn_inputs = lambda: inputs
    layout = _GroupLayout(
        attention_rows=(100, 80),
        gdn_rows=(120, 60) if gdn else None,
        attention_retained=(5_000, 90_000),
        gdn_states=(3, 40),
    )
    # An attention layer keeps its input norm output and five query- and
    # KV-width tensors beside the executor's records; a GDN layer its traced
    # width, CP exchange included, and its segments' recurrent states.
    widths = r._dense_mixer_widths()
    assert widths["attention"] == (HIDDEN + 5 * 16 * 256 + 5 * 2 * 256) * 2
    states = r._gdn_segment_layer_bytes()
    if gdn:
        assert (
            widths["gdn"]
            == (2 * HIDDEN + 2 * 4 * 64 + 2 * 8 * 64 + 7 * 8 * 64 + 64 * 8) * 2
        )
        assert states == 2 * (4 * 8 * 64 * 64 + 2 * (2 * 4 * 64 + 8 * 64) * 3)

    def floor(rank, grad=True):
        attention = layout.attention_rows[rank]
        gdn_rows = attention if layout.gdn_rows is None else layout.gdn_rows[rank]
        rows = {"attention": attention, "gdn": gdn_rows}
        extra = {
            "attention": layout.attention_retained[rank] * grad,
            "gdn": math.ceil(layout.gdn_states[rank] * states / 2),
        }
        per_row = {
            k: widths[k] + 2 * HIDDEN * 2 + STAGE if grad else NO_GRAD for k in widths
        }
        ledger = HIDDEN * 2 * sum(gdn_rows if g else attention for g in inputs)
        stage = max(rows[k] * per_row[k] + extra[k] for k in widths)
        return ledger * grad, stage

    assert r._dense_layout_floors(((100, True),), None, (layout,)) == (
        floor(0),
        floor(1),
    )
    no_grad = r._dense_layout_floors(((100, False),), None, (layout,))
    assert no_grad == (floor(0, False), floor(1, False))
    # Mixed waves keep main's pricing.
    assert (
        r._dense_layout_floors(((100, True), (9, False)), None, (layout,) * 2) is None
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


def test_cp1_and_cp2_no_grad_stages_share_one_form():
    from art.trainer_rank._impl import _dense_no_grad_row_elements

    assert _dense_no_grad_row_elements(FFN, HIDDEN) == 6 * FFN + 4 * HIDDEN
    assert _dense_no_grad_row_elements(FFN, HIDDEN, 8) == 8 * FFN + 4 * HIDDEN
    # A covered CP2 rank adds H/4 and the adapters' rank intermediates.
    stage = _dense_no_grad_row_elements(FFN, HIDDEN)
    assert NO_GRAD == (stage + HIDDEN // 4 + 6 * RANK) * 2


def test_no_grad_groups_price_the_largest_groups_own_rows():
    dense, plain = _at_cp2(_dense_rank()), _at_cp2(_dense_rank(0, 0))
    te = dense._te_workspace_growth_bytes()
    for groups, packed in (
        ((12_000,), 40_000),
        ((12_000, 8_000), 40_000),
        ((58_240, 29_120), 119_119),
    ):
        # The largest group's own physical rows at the traced width (with
        # TE's workspace growth), in place of the per-packed-token floor.
        expected = int((max(groups) * NO_GRAD + te) * 1.1)
        assert _no_grad_required(dense, groups, packed=packed) == expected
        assert expected < _no_grad_required(plain, groups, packed=packed)
    # A narrow structural width never drops below the rows' traced need.
    wide = _at_cp2(_dense_rank(STAGE, 10**6))
    assert _no_grad_required(wide, (12_000, 8_000)) == int((12_000 * 10**6 + te) * 1.1)


def test_te_workspace_growth_lasts_until_this_devices_gemm_workspace(monkeypatch):
    gemm = pytest.importorskip("transformer_engine.pytorch.cpp_extensions.gemm")

    @functools.lru_cache(maxsize=None)
    def get_cublas_workspace(device, ub, grouped_gemm):
        return None

    monkeypatch.setattr(gemm, "get_cublas_workspace", get_cublas_workspace)
    dense = _at_cp2(_dense_rank())
    device = dense.device.index
    # Grouped, overlap and other devices' workspaces leave this device cold,
    # and process-cold no-grad waves charge the growth too.
    for key in ((device, False, True), (device, True, False), (7, False, False)):
        get_cublas_workspace(*key)
        assert dense._te_workspace_growth_bytes() == _TE_CUBLAS_WORKSPACE_BYTES
    no_grad = _no_grad_required(dense, (500,), packed=1000)
    assert no_grad == int((500 * NO_GRAD + _TE_CUBLAS_WORKSPACE_BYTES) * 1.1)
    get_cublas_workspace(device, False, False)
    assert dense._te_workspace_growth_bytes() == 0
    assert _no_grad_required(dense, (500,), packed=1000) == int(500 * NO_GRAD * 1.1)


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


def test_every_adapter_the_layer_runs_is_priced_beside_arts_norm_wrapper():
    from art.megatron import lora as lora_module
    from art.megatron.gdn.operator import _empty_safe_norm_forward

    layers = [_dense_layer(gdn=index != 2) for index in range(3)]
    for layer in layers:
        _wrap_like_art(layer)
    # A mixer adapter keeps its rank-wide products beside the MLP's.
    layers[1].self_attention.qkv_lora = _adapter(lora_module, HIDDEN, HIDDEN, 32)
    # ART's empty-safe norm wrapper still calls the norm's own forward.
    norm: Any = torch.nn.LayerNorm(HIDDEN)
    norm._art_empty_safe_norm_physical_forward = norm.forward
    norm.forward = MethodType(_empty_safe_norm_forward, norm)
    layers[0].self_attention.q_layernorm = norm
    ranks = 2 * (3 * RANK + 32)
    assert _dense_mlp_recompute_bytes_per_token([_dense_model(layers)]) == (
        (7 * FFN + HIDDEN // 4 + ranks) * 2,
        (6 * FFN + 4 * HIDDEN + HIDDEN // 4 + ranks) * 2,
    )


def test_named_slots_are_read_through_the_lookup_execution_uses():
    from art.megatron import lora as lora_module

    layers = [_dense_layer(gdn=index == 0) for index in range(2)]
    for layer in layers:
        _wrap_like_art(layer)
    model = _dense_model(layers)
    policy = rank()._slot_ref("policy")
    adapters = [
        m for layer in layers for m in layer.modules() if type(m) is lora_module.LoRA
    ]

    def load(adapter, width):
        slot = _module(lora_module.LoRASlot)
        slot.A_T = torch.nn.Parameter(
            torch.empty(adapter.A_T.shape[0], width, dtype=torch.bfloat16)
        )
        slot.B_T = torch.nn.Parameter(
            torch.empty(width, adapter.B_T.shape[1], dtype=torch.bfloat16)
        )
        adapter._slot_keys = {policy: "slot_0"}
        adapter._slot_modules = torch.nn.ModuleDict({"slot_0": slot})

    for adapter in adapters:
        load(adapter, 16)
    widths = (
        (7 * FFN + HIDDEN // 4 + 2 * 3 * 16) * 2,
        (6 * FFN + 4 * HIDDEN + HIDDEN // 4 + 2 * 3 * 16) * 2,
    )
    assert _dense_mlp_recompute_bytes_per_token([model], policy) == widths
    # A slot without an adapter on one module runs the base output there.
    for layer in layers:
        layer.mlp.linear_fc1.up_lora._slot_keys = {}
    assert _dense_mlp_recompute_bytes_per_token([model], policy) == (
        (7 * FFN + HIDDEN // 4 + 2 * 2 * 16) * 2,
        (6 * FFN + 4 * HIDDEN + HIDDEN // 4 + 2 * 2 * 16) * 2,
    )
    load(adapters[0], 300)  # Loaded wider than the priced rank.
    assert _dense_mlp_recompute_bytes_per_token([model], policy) == (0, 0)


@pytest.mark.parametrize("case", ["selective", "eval"])
def test_dense_widths_need_the_checkpoint_floors_decoder(case):
    """The no-grad discount must not outlive the floor that adds TE growth."""
    r = _at_cp2(_dense_rank())
    assert r._dense_mlp_widths() == (STAGE, NO_GRAD)
    decoder = _impl._language_model(r.runtime.model[0]).decoder
    if case == "selective":
        decoder.config.recompute_granularity = "selective"
    else:
        decoder.train(False)
    assert r._checkpoint_memory_floor(((12_000, False), (8_000, False))) == (0, 0)
    assert r._dense_mlp_widths() == (0, 0)
    discounted = _no_grad_required(r, (12_000, 8_000))
    r._dense_recompute_bytes_per_token = r._dense_no_grad_bytes_per_token = 0
    assert _no_grad_required(r, (12_000, 8_000)) == discounted


def art_cp(r, monkeypatch):
    """A 3:1 GDN/attention hybrid at CP2 with ART's CP core attention and a CPU
    planning config."""
    from art.megatron.context_parallel.core_attention import (
        ArtContextParallelCoreAttention,
    )
    from art.megatron.context_parallel.types import ParallelTopology

    r._topology_key = lambda: CP2
    r._geometry = replace(
        r._geometry,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
        gdn_key_heads=16,
        gdn_key_head_dim=128,
        gdn_value_heads=32,
        gdn_value_head_dim=128,
    )
    r._gdn_layers = 30
    for index, layer in enumerate(r.runtime.model[0].decoder.layers):
        is_gdn = index % 4 != 3
        after_gdn = index > 0 and (index - 1) % 4 != 3
        layer._art_gdn_island_boundary = SimpleNamespace(
            is_gdn=is_gdn, input_layout="gdn" if is_gdn and after_gdn else "attention"
        )
        if not is_gdn:
            core = ArtContextParallelCoreAttention.__new__(
                ArtContextParallelCoreAttention
            )
            torch.nn.Module.__init__(core)
            core.softmax_offset = None
            layer.self_attention = torch.nn.Module()
            layer.self_attention.core_attention = core
    provider = r.runtime.provider
    provider.kv_channels = 256
    provider.ffn_hidden_size = 8192
    provider.linear_num_key_heads = 16
    provider.linear_num_value_heads = 32
    provider.linear_key_head_dim = 128
    provider.linear_value_head_dim = 128
    provider.params_dtype = torch.bfloat16
    r.runtime.model_support_handler = SimpleNamespace(
        build_gdn_execution_spec=True,
        context_parallel_workload_profile=lambda provider: None,
    )
    monkeypatch.setattr(r, "_topology", lambda: ParallelTopology(tp=1, cp=2))
    return r


def _cp_requests(lengths=(900, 700, 500)):
    start, out = 0, []
    for n in lengths:
        out.append(
            ForwardInput(
                input_tokens=torch.arange(start, start + n), hidden_states=True
            )
        )
        start += n
    return out


def test_only_a_covered_dense_model_prices_layouts(monkeypatch):
    from art.megatron.context_parallel.executor import retained_stage_record_bytes
    from art.megatron.context_parallel.runtime import context_parallel_rank_layouts
    from art.megatron.context_parallel.types import ParallelTopology
    from art.megatron.training.microbatches import (
        _context_parallel_config_for_provider,
    )

    r = art_cp(_dense_rank(), monkeypatch)
    plan = r._plan_flat_forward(_cp_requests())
    (layout,) = r._plan_group_layouts(plan)
    assert sum(layout.attention_rows) == plan.packed_tokens
    assert layout.gdn_rows is not None and sum(layout.gdn_rows) == plan.packed_tokens
    # Each rank's retention is the executor mirror over that rank's own plan.
    (group,) = plan.groups
    _, _, states, rank_plans = context_parallel_rank_layouts(
        group_ids=group.packed.group_ids,
        parent_ids=group.packed.parent_ids,
        topology=ParallelTopology(tp=1, cp=2),
        config=_context_parallel_config_for_provider(
            r.runtime.provider, r.device, r.runtime.model_support_handler
        ),
        original_seq_len=int(group.packed.tokens.shape[1]),
        build_gdn_execution_spec=True,
    )
    assert layout.attention_retained == tuple(
        retained_stage_record_bytes(
            rank_plan,
            q_heads=16,
            kv_heads=2,
            head_dim=256,
            value_head_dim=256,
            element_size=2,
            block_size=(128, 128),  # the CPU device's flex block
        )
        for rank_plan in rank_plans
    )
    assert states is not None and layout.gdn_states == states
    assert all(n >= 2 for n in states)
    cost = r._plan_cost(plan)
    rows = sum(n for n, grad in r._plan_group_rows(plan) if grad)
    assert cost.checkpoint_input_gradient == rows * HIDDEN * 2
    # MoE and uncovered models keep the busiest-rank pricing.
    for other in (rank(), _dense_rank(0, 0)):
        other = art_cp(other, monkeypatch)
        assert (
            other._plan_group_layouts(other._plan_flat_forward(_cp_requests())) is None
        )
    # No-grad plans are priced on layouts too; mixed plans are not.
    no_grad = [replace(q, no_grad=True) for q in _cp_requests()]
    assert r._plan_group_layouts(r._plan_flat_forward(no_grad)) is not None
    mixed = [no_grad[0], *_cp_requests((700, 500))]
    assert r._plan_group_layouts(r._plan_flat_forward(mixed)) is None


def _short_and_branching(no_grad=False):
    """64 short independent requests, then 8 continuations of one prefix."""
    start, out = 0, []
    for _ in range(64):
        out.append(torch.arange(start, start + 40))
        start += 40
    prefix = torch.arange(start, start + 256)
    out += [
        torch.cat([prefix, torch.arange(9000 + 30 * i, 9030 + 30 * i)])
        for i in range(8)
    ]
    return [
        ForwardInput(input_tokens=t, hidden_states=True, no_grad=no_grad) for t in out
    ]


@pytest.mark.parametrize("no_grad", [False, True])
def test_gdn_segment_states_are_priced_on_each_ranks_layouts(monkeypatch, no_grad):
    r = art_cp(_dense_rank(), monkeypatch)
    plan = r._plan_flat_forward(_short_and_branching(no_grad))
    (layout,) = r._plan_group_layouts(plan)
    # Every independent request and branch is a segment on some rank, with an
    # initial and a final state.
    assert sum(layout.gdn_states) >= 2 * 72
    states = math.ceil(max(layout.gdn_states) * r._gdn_segment_layer_bytes() / 2)
    stateless = replace(layout, gdn_states=())
    monkeypatch.setattr(r, "_plan_group_layouts", lambda plan: (stateless,))
    without = r._plan_cost(plan).required
    monkeypatch.setattr(r, "_plan_group_layouts", lambda plan: (layout,))
    with_states = r._plan_cost(plan).required
    assert int(states * 1.1) // 2 < with_states - without <= int(states * 1.1) + 1


def test_gdn_states_include_parents_imported_from_another_rank(monkeypatch):
    """A prefix tree whose parents and children run on different CP ranks:
    each rank holds an initial and a final state per executed segment plus
    every parent state the executor's exchange imports."""
    from art.megatron.context_parallel.runtime import (
        _get_or_build_planning_bundle,
        _plan_gdn_global_execution,
        _plan_gdn_rank_execution,
    )
    from art.megatron.context_parallel.types import ParallelTopology
    from art.megatron.training.microbatches import (
        _context_parallel_config_for_provider,
        _gdn_planner_config_for_provider,
    )

    r = art_cp(_dense_rank(), monkeypatch)
    prefix, key = torch.arange(1000), 100_000
    requests = []
    for _ in range(3):
        middle = torch.arange(key, key + 300)
        for leaf in range(3):
            tail = torch.arange(key + 1000 * (leaf + 1), key + 1000 * (leaf + 1) + 200)
            requests.append(torch.cat([prefix, middle, tail]))
        key += 10_000
    plan = r._plan_flat_forward(
        [ForwardInput(input_tokens=t, hidden_states=True) for t in requests]
    )
    (layout,) = r._plan_group_layouts(plan)
    (group,) = plan.groups
    topology = ParallelTopology(tp=1, cp=2)
    config = _context_parallel_config_for_provider(
        r.runtime.provider, r.device, r.runtime.model_support_handler
    )
    gdn_config = _gdn_planner_config_for_provider(
        r.runtime.provider, r.runtime.model_support_handler
    )
    planning, bundle, _, _ = _get_or_build_planning_bundle(
        group_ids=group.packed.group_ids,
        parent_ids=group.packed.parent_ids,
        topology=topology,
        config=config,
        original_seq_len=int(group.packed.tokens.shape[1]),
        build_gdn_execution_spec=True,
    )
    decision = _plan_gdn_global_execution(
        planning_key=planning,
        bundle=bundle,
        topology=topology,
        gdn_planner_config=gdn_config,
    )
    imported = [
        sum(
            len(exchange.dest_family_indices)
            for exchange in _plan_gdn_rank_execution(
                planning_key=planning,
                bundle=bundle,
                topology=topology,
                cp_rank=rank,
                gdn_planner_config=gdn_config,
            ).tree_state_exchanges_by_depth
            if exchange is not None
        )
        for rank in range(2)
    ]
    assert any(imported), "the tree must import a parent state across ranks"
    chains = sum(map(len, decision.chain_segments_by_depth))
    executed = [
        chains + sum(map(len, depths)) for depths in decision.segments_by_rank_depth
    ]
    assert layout.gdn_states == tuple(
        2 * n + received for n, received in zip(executed, imported, strict=True)
    )


def test_adapter_gradients_pair_with_each_ranks_boundaries(monkeypatch):
    r = _at_cp2(_dense_rank())
    policy = r._slot_ref("policy")
    groups = ((100, True),)
    layout = _GroupLayout((100, 40), None, (0, 0))
    # Large pending gradients against small boundaries: rank 1 releases less.
    pending = (8_000_000,) * LAYERS + (3_000_000,)
    monkeypatch.setattr(r, "_pending_adapter_gradient_bytes", lambda refs: pending)
    monkeypatch.setattr(
        _impl, "_dense_mlp_recompute_bytes_per_token", lambda *a, **k: (STAGE, NO_GRAD)
    )
    floors = r._dense_layout_floors(groups, (policy,), (layout,))
    boundaries = max(retained for retained, _ in floors)
    workspace = max(map(sum, floors)) - boundaries
    extras = [
        r._adapter_gradient_walk([(pending, (HIDDEN * 2 * rows,) * LAYERS)])
        for rows in layout.attention_rows
    ]
    expected = max(
        rank_retained + max(rank_workspace, workspace) + extra
        for (rank_retained, rank_workspace), extra in zip(floors, extras, strict=True)
    ) - (boundaries + workspace)
    assert expected > 0 and extras[1] > extras[0]
    signature = _MemorySignature(CP2, (1, None), 1, (), True, (True,))
    cost = r._subforward_cost(
        packed_tokens=140,
        output_bytes=0,
        signature=signature,
        logical_tokens=140,
        group_rows=groups,
        group_layouts=(layout,),
        slot_refs=(policy,),
    )
    assert cost.checkpoint_adapter_gradient == expected


def test_no_grad_pricing_needs_the_layout_gate():
    """Without ART's CP attention and GDN islands, no-grad waves keep main's."""
    dense, plain = _dense_rank(), _dense_rank(0, 0)
    for r in (dense, plain):
        r._topology_key = lambda: CP2
    assert dense._dense_mlp_widths() != (0, 0)
    assert not dense._layout_pricing_supported(CP2)
    assert _no_grad_required(dense, (12_000, 8_000)) == _no_grad_required(
        plain, (12_000, 8_000)
    )


def test_split_lower_bound_stays_below_the_exact_dense_price(monkeypatch):
    r = art_cp(_dense_rank(), monkeypatch)
    chunk = _cp_requests((700, 500))
    exact = r._plan_cost(r._plan_flat_forward(chunk)).required
    lower = r._split_chunk_lower_cost(
        chunk, tuple(q.input_tokens for q in chunk), checkpoint=_impl.Unset
    ).required
    assert lower <= exact


def test_replay_refuses_covered_dense_plans(monkeypatch):
    from art.trainer_rank import _planner_replay

    r = art_cp(_dense_rank(), monkeypatch)
    plan = r._plan_flat_forward(_cp_requests())
    with pytest.raises(ValueError, match="dense_runtime_facts_unsupported"):
        _planner_replay.capture(r, plan)
