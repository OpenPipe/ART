"""CPU checks using ART's real LoRA storage and runtime wrappers."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")
from art.megatron import lora as lm
from art.megatron.glm5next.lora import lora_targets
from art.megatron.glm5next.lora_layers import _attach_projection


@pytest.fixture(autouse=True)
def local_parallel_state(monkeypatch):
    for name in (
        "get_expert_model_parallel_rank",
        "get_expert_data_parallel_rank",
        "get_data_parallel_rank",
    ):
        monkeypatch.setattr(lm.ps, name, lambda **_: 0)
    monkeypatch.setattr(lm.ps, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        lm.ps, "get_expert_tensor_and_model_parallel_group", lambda **_: None
    )


@pytest.mark.parametrize("rank", (1, 8, 32))
def test_actual_art_unique_storage_counts_on_meta(rank):
    counts = dict.fromkeys(("attn", "mlp", "unembed"), 0)
    for target in lora_targets():
        module = lm.LoRA(
            target.name,
            target.in_features,
            target.out_features,
            rank,
            32,
            torch.bfloat16,
            torch.device("meta"),
            num_local_experts=target.experts or 1,
            lora_layout=lm.LoraLayout(expert="shared_outer"),
        )
        counts[target.group] += sum(p.numel() for p in module.parameters())
    assert counts == {
        "attn": 2_009_088 * rank,
        "mlp": 75_755_520 * rank,
        "unembed": 158_976 * rank,
    }


@pytest.mark.parametrize("tuple_output", (False, True))
def test_projection_bf16_merged_equivalence_with_random_ab(tuple_output):
    class Linear(torch.nn.Linear):
        def forward(self, x):
            output = super().forward(x)
            return (output, None) if tuple_output else output

    torch.manual_seed(225)
    base = Linear(32, 48, bias=False, dtype=torch.bfloat16)
    _attach_projection(base, "base_model.model.lm_head", 8, 32)
    module = base
    with torch.no_grad():
        module.lora.A_T.normal_(std=0.1)
        module.lora.B_T.normal_(std=0.1)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    actual = module(x)
    actual = actual[0] if tuple_output else actual
    merged = (
        base.weight.float() + 4 * module.lora.B_T.T.float() @ module.lora.A_T.T.float()
    )
    expected = torch.nn.functional.linear(x, merged.to(torch.bfloat16))
    rel_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert torch.isfinite(actual).all()
    assert rel_l2 <= 1e-2, rel_l2


def test_native_expert_ep_storage_and_logical_ids():
    # EP8 has 36 expert-specific factors, and one hidden-side factor. Binding
    # logical IDs means exports cannot accidentally number every EP rank 0..35.
    module = lm.LoRA(
        "base_model.model.model.layers.3.mlp.experts.{expert}.gate_proj",
        8,
        6,
        8,
        32,
        torch.float32,
        torch.device("cpu"),
        num_local_experts=36,
        lora_layout=lm.LoraLayout(expert="shared_outer"),
    )
    ids = tuple(range(72, 108))
    module.bind_expert_layout(ids, ())
    assert module.A_T.shape == (1, 8, 8)
    assert module.B_T.shape == (36, 8, 6)
    assert getattr(module.A_T, "grad_sync_domain") == "expert_tp_ep"
    state = module.sharded_lora_state_dict()
    assert len(state) == 72
    assert all(".72." in k or any(f".{i}." in k for i in ids) for k in state)


def test_native_kda_and_head_installer_freezes_only_base():
    from art.megatron.glm5next.lora import LoraFlags, apply_glm5next_lora

    attention = torch.nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "b_proj"):
        setattr(attention, name, torch.nn.Linear(8, 8, bias=False))
    layer = torch.nn.Module()
    layer.layer_number = 1
    layer.self_attention = attention
    layer.mlp = torch.nn.Linear(8, 8)
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([layer])
    model.output_layer = torch.nn.Linear(8, 17, bias=False)
    apply_glm5next_lora(
        [model],
        SimpleNamespace(tensor_model_parallel_size=1),
        rank=8,
        flags=LoraFlags(True, False, True),
    )
    loras = [m for m in model.modules() if isinstance(m, lm.LoRA)]
    assert len(loras) == 5
    assert not hasattr(attention.b_proj, "lora")
    assert "decoder.layers.0.self_attention.q_proj.weight" in dict(
        model.named_parameters()
    )
    assert "output_layer.weight" in dict(model.named_parameters())
    assert all(
        not p.requires_grad
        for n, p in model.named_parameters()
        if not n.endswith(("A_T", "B_T"))
    )
    assert all(p.requires_grad for m in loras for p in m.parameters())
    assert all(m.lora_layout.expert == "shared_outer" for m in loras)
    from art.megatron.glm5next.lora_layers import build_glm5next_adapter_weights_by_base

    weights = build_glm5next_adapter_weights_by_base([model])
    assert set(weights) == {
        *(
            f"decoder.layers.0.self_attention.{p}_proj.weight"
            for p in ("q", "k", "v", "o")
        ),
        "output_layer.weight",
    }
    with pytest.raises(ValueError, match="already"):
        apply_glm5next_lora(
            [model], SimpleNamespace(tensor_model_parallel_size=1), rank=8
        )


def test_all_false_is_frozen_empty_adapter():
    from art.megatron.glm5next.lora import LoraFlags, apply_glm5next_lora

    model = torch.nn.Linear(2, 3)
    apply_glm5next_lora(
        [model], SimpleNamespace(), rank=8, flags=LoraFlags(False, False, False)
    )
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(isinstance(m, lm.LoRA) for m in model.modules())
