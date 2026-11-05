from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from art.megatron.gdn.gdn_prefix_tree import parse_gdn_prefix_tree_segments
from art.megatron.glm5next.kda import KimiDeltaAttention
from art.megatron.glm5next.kda_cp import build_prefix_plan
from art.megatron.glm5next.kda_reference import recurrent_kda
from tests.integration.megatron import kda_parity as parity


def config(**kwargs):
    return SimpleNamespace(
        **(
            dict(
                hidden_size=8,
                linear_num_heads=2,
                linear_head_dim=4,
                linear_conv_kernel_dim=4,
                linear_lower_bound=-5.0,
                rms_norm_eps=1e-6,
                params_dtype=torch.float32,
            )
            | kwargs
        )
    )


def layer():
    torch.manual_seed(17)
    return KimiDeltaAttention(config(), backend="reference")


class Adapter(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(2, base.in_features) * 0.1)
        self.B = nn.Parameter(torch.randn(base.out_features, 2) * 0.1)

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        return self.base(x) + (x @ self.A.T) @ self.B.T


def adapt(module):
    module.requires_grad_(False)
    for name in module.lora_targets:
        setattr(module, name, Adapter(getattr(module, name)))


def test_cross_shape_variants_preserve_target_and_exclude_neighbors_from_loss():
    module = layer()
    parity.adapt(module)
    target = torch.randn(9, 1, 8)
    probe = torch.randn_like(target)
    variants = parity.shape_variants(target)
    assert [tuple(value.shape[:2]) for value in variants] == [
        (9, 1),
        (16, 1),
        (23, 2),
        (30, 3),
    ]
    baseline = module(target)[0]
    baseline_grad = parity.gradients(module, baseline, probe)
    for value in variants:
        torch.testing.assert_close(value[:9, :1], target, rtol=0, atol=0)
        output = module(value)[0][:9, :1]
        actual_grad = parity.gradients(module, output, probe)
        torch.testing.assert_close(output, baseline, rtol=1e-5, atol=1e-6)
        assert len(actual_grad) == 8
        for name in baseline_grad:
            torch.testing.assert_close(
                actual_grad[name], baseline_grad[name], rtol=1e-5, atol=1e-6
            )


def test_pairwise_envelope_and_both_metric_thresholds():
    values = [
        torch.tensor([1.0 + step, 2.0], dtype=torch.float64)
        for step in (0, 0.1, 0.2, 0.3)
    ]
    envelope = parity.pairwise_envelope(values)
    assert len(envelope["pairs"]) == 6
    assert envelope["max_abs"] == pytest.approx(0.3)
    assert envelope["rel_l2"] == pytest.approx(0.3 / 5**0.5)
    with pytest.raises(ValueError, match="K >= 4"):
        parity.pairwise_envelope(values[:3])
    expected, actual = torch.tensor([0.0, 4.0]), torch.tensor([3.0, 4.0])
    assert parity.envelope_comparison(
        actual, expected, {"max_abs": 1.5, "rel_l2": 0.375}
    )["pass"]
    for bound in (
        {"max_abs": 1.49, "rel_l2": 0.375},
        {"max_abs": 1.5, "rel_l2": 0.374},
    ):
        result = parity.envelope_comparison(actual, expected, bound)
        assert not result["pass"] and not result["hard_leakage_failure"]
    result = parity.envelope_comparison(actual, expected, {"max_abs": 0, "rel_l2": 0})
    assert not result["pass"] and result["hard_leakage_failure"]
    assert parity.envelope_comparison(expected, expected, {"max_abs": 0, "rel_l2": 0})[
        "pass"
    ]
    with pytest.raises(AssertionError, match="Nonfinite"):
        parity.envelope_comparison(torch.tensor([float("nan"), 4]), expected, envelope)


def test_perturb_pack_keeps_each_target_and_changes_every_neighbor():
    parts = [
        torch.full((length, 1, 8), float(index + 10))
        for index, length in enumerate((9, 10, 13))
    ]
    for index, target in enumerate(parts):
        for delta in (0, 7, -7):
            value, cu, selected = parity.perturb_pack(parts, index, delta)
            lengths = [
                part.shape[0] + (0 if other == index else delta)
                for other, part in enumerate(parts)
            ]
            assert (cu[1:] - cu[:-1]).tolist() == lengths
            torch.testing.assert_close(value[selected], target, rtol=0, atol=0)
            for other, part in enumerate(value.split(lengths)):
                if other != index:
                    assert not torch.any(part == float(other + 10))


def test_packing_gate_detects_cross_sequence_leakage():
    class Layer(nn.Module):
        def __init__(self, leak):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.leak = leak

        def forward(self, x, packed_seq_params=None):  # ty: ignore[invalid-mutable-override]
            assert torch.is_grad_enabled()  # Exercise the same training dispatch.
            output = self.weight * x
            if self.leak and packed_seq_params is not None:
                output = output + x.sum()  # Deliberate wrong cross-sequence dependency.
            return output, None

    x = torch.ones(32, 1, 2)
    cu = torch.tensor([0, 9, 19, 32], dtype=torch.int32)
    for leak in (False, True):
        result = parity.packing_checks(Layer(leak), x, cu, torch.ones_like(x))
        assert result["packed_pass"] is (not leak)
        assert result["leakage_pass"] is (not leak)
        assert result["hard_leakage_failure"] is leak
        assert len(result["targets"]) == 3
        for target in result["targets"]:
            assert len(set(target["gemm_rows"])) == 4
            assert set(target["leakage"]) == {"0", "7", "-7"}
            assert target["output_envelope"]["max_abs"] == 0


def test_recurrence_derivatives():
    torch.manual_seed(3)
    q, k, v, g = [
        torch.randn(1, 3, 2, 2, dtype=torch.float64, requires_grad=True)
        for _ in range(4)
    ]
    beta = torch.randn(1, 3, 2, dtype=torch.float64).sigmoid().requires_grad_()
    assert torch.autograd.gradcheck(
        recurrent_kda, (q, k, v, -g.square(), beta), atol=1e-7, rtol=1e-5
    )


def test_cp_precision_launch_and_alias_scope():
    import importlib

    from art.megatron.glm5next.kda_cp_kernels import (
        _PreciseLaunch,
        install_kda_cp_precision,
    )

    class Kernel:
        def __getitem__(self, grid):
            return lambda **kwargs: (grid, kwargs)

    grid, arguments = _PreciseLaunch(Kernel())[(2, 3)](value=7)
    assert grid == (2, 3)
    assert arguments == {"value": 7, "default_dot_input_precision": "tf32x3"}
    cp = pytest.importorskip("fla.ops.cp.chunk_delta_h")
    before = cp.chunk_gated_delta_rule_fwd_h_pre_process
    install_kda_cp_precision()
    forward = importlib.import_module("fla.ops.kda.chunk_fwd")
    copied = forward.chunk_gated_delta_rule_fwd_h_pre_process
    assert copied is not before and copied.__code__ is before.__code__
    assert cp.chunk_gated_delta_rule_fwd_h_pre_process is before
    assert isinstance(copied.__globals__["merge_fwd_bwd_kernel"], _PreciseLaunch)
    install_kda_cp_precision()
    assert copied is forward.chunk_gated_delta_rule_fwd_h_pre_process


def test_packed_matches_independent_sequences_and_adapter_gradients():
    module = layer()
    adapt(module)
    x = torch.randn(11, 1, 8)
    cu = torch.tensor([0, 3, 4, 11], dtype=torch.int32)
    packed = SimpleNamespace(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    actual, bias = module(x, packed_seq_params=packed)
    assert bias is None
    expected = torch.cat([module(part)[0] for part in x.split([3, 1, 7])])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    parameters = [p for p in module.parameters() if p.requires_grad]
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), parameters)
    expected_grads = torch.autograd.grad((expected * probe).sum(), parameters)
    assert len(parameters) == 8
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_prefix_tree_outputs_and_lora_gradients_match_expanded_branches():
    module = layer()
    adapt(module)
    # Two levels of branches plus an independent root and a padding position.
    groups = torch.tensor([[0, 0, 1, 1, 2, 3, 3, 4, -1]])
    parents = torch.tensor([[-1, -1, 0, 0, 0, 1, 1, 4, -1]])
    state = SimpleNamespace(group_ids=groups, parent_ids=parents, model_state={})
    x = torch.randn(9, 1, 8, requires_grad=True)
    actual, _ = module(x, attention_bias=state)
    # Independent physical sequences: root0->child1->child3, root0->child2, root4.
    first = module(x[[0, 1, 2, 3, 5, 6]])[0]
    second = module(x[[0, 1, 4]])[0]
    third = module(x[7:8])[0]
    expected = torch.cat([first[:4], second[2:], first[4:], third, x[8:9] * 0])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    params = [x] + [p for p in module.parameters() if p.requires_grad]
    probe = torch.randn_like(actual)
    left = torch.autograd.grad((actual * probe).sum(), params, retain_graph=True)
    right = torch.autograd.grad((expected * probe).sum(), params)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert x.grad is not None
    assert x.grad[8].count_nonzero() == 0


def test_fp32_parameters_survive_cast_without_losing_bits():
    module = layer()
    with torch.no_grad():
        for parameter in module._fp32_parameters():
            parameter.copy_(torch.randn_like(parameter))
    before = [p.clone() for p in module._fp32_parameters()]
    module.to(dtype=torch.bfloat16)
    for previous, actual in zip(before, module._fp32_parameters(), strict=True):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(previous, actual, rtol=0, atol=0)
    assert module.q_proj.weight.dtype == torch.bfloat16
    result, _ = module(torch.randn(5, 1, 8, dtype=torch.bfloat16))
    assert result.dtype == torch.bfloat16 and result.isfinite().all()


def test_future_tokens_cannot_change_prefix():
    module = layer()
    x = torch.randn(9, 1, 8)
    changed = x.clone()
    changed[4:] += 10
    torch.testing.assert_close(module(x)[0][:4], module(changed)[0][:4], rtol=0, atol=0)


def test_meta_materialization_and_frozen_input_checkpoint():
    from torch.utils.checkpoint import checkpoint

    meta = KimiDeltaAttention(config(params_dtype=torch.bfloat16), device="meta")
    meta.to_empty(device="cpu")
    assert all(
        p.dtype == torch.float32 and not p.is_meta for p in meta._fp32_parameters()
    )
    module = layer()
    adapt(module)
    x = torch.randn(5, 1, 8)  # frozen embedding: no requires_grad on the input
    expected, _ = module(x)
    actual = checkpoint(lambda value: module(value)[0], x, use_reentrant=False)
    params = [p for p in module.parameters() if p.requires_grad]
    left = torch.autograd.grad(actual.square().sum(), params)
    right = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("cu", [[1, 4], [0, 3], [0, 0, 4], [0, 5, 4]])
def test_invalid_boundaries_fail(cu):
    boundaries = torch.tensor(cu, dtype=torch.int32)
    packed = SimpleNamespace(
        qkv_format="thd", cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries
    )
    with pytest.raises(ValueError, match="boundaries"):
        layer()(torch.randn(4, 1, 8), packed_seq_params=packed)


def test_mcore_config_and_lora_contract_shapes_on_meta():
    cfg = SimpleNamespace(
        hidden_size=4096,
        linear_num_key_heads=64,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    module = KimiDeltaAttention(cfg, device="meta")
    assert [
        tuple(getattr(module, name).weight.shape) for name in module.lora_targets
    ] == [(8192, 4096)] * 3 + [(4096, 8192)]
    assert (
        sum(sum(getattr(module, name).weight.shape) for name in module.lora_targets)
        == 49152
    )
    assert module.f_a_proj.weight.shape == (128, 4096)
    assert module.f_b_proj.weight.shape == (8192, 128)
    assert module.dt_bias.shape == (8192,)


def test_empty_prefix_and_cp_plan_padding():
    module = layer()
    empty = SimpleNamespace(
        group_ids=torch.full((1, 3), -1), parent_ids=torch.full((1, 3), -1)
    )
    x = torch.randn(3, 1, 8, requires_grad=True)
    y, _ = module(x, attention_bias=empty)
    assert y.count_nonzero() == 0
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.count_nonzero() == 0
    spec = parse_gdn_prefix_tree_segments(
        torch.tensor([[0, 0, 1, 2]]), torch.tensor([[-1, -1, 0, 0]])
    )
    plan = build_prefix_plan(spec, device=torch.device("cpu"))
    assert plan.expanded_tokens == 6 and plan.original_tokens == 4
    assert plan.indices.tolist() == [0, 1, 2, 0, 1, 3]


def test_hf_fp32_forward_and_lora_gradients(monkeypatch):
    hf = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    for name in ("causal_conv1d_fn", "chunk_kimi_delta_attention"):
        monkeypatch.setattr(hf, name, inspect.unwrap(getattr(hf, name)))

    module = layer()
    reference = hf.Glm5NextTextLinearAttention(
        Glm5NextTextConfig(
            hidden_size=8,
            num_hidden_layers=4,
            linear_num_heads=2,
            linear_head_dim=4,
            linear_conv_kernel_dim=4,
            linear_lower_bound=-5.0,
            rms_norm_eps=1e-6,
        ),
        layer_idx=0,
    )
    state = module.state_dict()
    hf_state = dict(state)
    for name in ("f_a_proj.weight", "f_b_proj.weight", "A_log", "dt_bias"):
        hf_state["forget_gate." + name] = hf_state.pop(name)
    hf_state["conv1d.weight"] = torch.cat(
        [
            hf_state.pop(name + ".weight")
            for name in ("q_conv1d", "k_conv1d", "v_conv1d")
        ]
    )
    reference.load_state_dict(hf_state, strict=True)
    adapt(module)
    adapt_reference = reference
    adapt_reference.lora_targets = module.lora_targets
    torch.manual_seed(55)
    adapt(reference)
    for name in module.lora_targets:
        getattr(reference, name).load_state_dict(getattr(module, name).state_dict())
    x = torch.randn(7, 1, 8)
    actual, _ = module(x)
    expected = reference(x.transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    probe = torch.randn_like(actual)
    left = torch.autograd.grad(
        (actual * probe).sum(), [p for p in module.parameters() if p.requires_grad]
    )
    right_by_name = dict(reference.named_parameters())
    right = torch.autograd.grad(
        (expected * probe).sum(),
        [
            right_by_name[name]
            for name, p in module.named_parameters()
            if p.requires_grad
        ],
    )
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
