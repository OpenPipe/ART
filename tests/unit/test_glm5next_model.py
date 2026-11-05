"""CPU arithmetic and model seams; these do not replace real-weight F4/F8 gates."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

pytest.importorskip("megatron.core")
pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextDecoderLayer,
    Glm5NextTextMLP,
    Glm5NextTextModel,
    Glm5NextTextMoE,
)

from art.megatron.glm5next.config import configure_glm5next_provider
from art.megatron.glm5next.layer import (
    Glm5NextFinalNorm,
    Glm5NextRMSNorm,
    Glm5NextRouter,
    Glm5NextTransformerLayer,
)
from art.megatron.glm5next.reference import HfReferenceAttention


def tiny_config(layers: int = 4) -> Glm5NextTextConfig:
    return Glm5NextTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=8,
        v_head_dim=8,
        linear_head_dim=8,
        linear_num_heads=2,
        linear_conv_kernel_dim=4,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=4,
        index_kpool=4,
        pad_token_id=None,
        use_cache=False,
    )


class _CpuMlp(nn.Module):
    def __init__(self, reference: nn.Module, **_: Any) -> None:
        super().__init__()
        self.reference = reference

    def forward(self, value: torch.Tensor, **_: Any) -> tuple[torch.Tensor, None]:
        return self.reference(value), None


def cpu_layer(config: Any, reference: Any, index: int) -> Glm5NextTransformerLayer:
    # Build the actual ART mHC layer and MCore's constructor. Only the MLP is
    # replaced with a CPU HF reference instead of GPU-only TE grouped GEMM.
    provider = TransformerConfig(
        num_layers=config.num_hidden_layers,
        hidden_size=16,
        num_attention_heads=2,
        use_cpu_initialization=True,
    )
    configure_glm5next_provider(provider, config)
    mlp_type = (
        Glm5NextTextMoE
        if config.mlp_layer_types[index] == "sparse"
        else Glm5NextTextMLP
    )
    mlp = mlp_type(config)
    modules = TransformerLayerSubmodules(
        input_layernorm=Glm5NextRMSNorm,
        self_attention=HfReferenceAttention,
        pre_mlp_layernorm=Glm5NextRMSNorm,
        mlp=ModuleSpec(module=_CpuMlp, params={"reference": mlp}),
    )
    layer = Glm5NextTransformerLayer(
        config=provider,
        submodules=modules,
        layer_number=index + 1,
        pg_collection=ProcessGroupCollection(tp=None, pp=None),
        add_layer_offset=False,
    )
    layer.self_attention.hf_reference.load_state_dict(reference.self_attn.state_dict())
    layer.mlp.reference.load_state_dict(reference.mlp.state_dict())
    cast(nn.Module, layer.input_layernorm).load_state_dict(
        reference.input_layernorm.state_dict()
    )
    cast(nn.Module, layer.pre_mlp_layernorm).load_state_dict(
        reference.post_attention_layernorm.state_dict()
    )
    with torch.no_grad():
        for site, source in (("attn", reference.attn_hc), ("ffn", reference.ffn_hc)):
            for field in ("fn", "base", "scale"):
                getattr(layer, f"hc_{site}_{field}").copy_(getattr(source, field))
    return layer


@pytest.mark.parametrize("index", [0, 3])
def test_actual_mhc_layer_forward_and_backward_against_hf(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transformers.models.glm5_next import modeling_glm5_next as hf_model

    # Select HF's independent PyTorch oracle rather than installed CUDA kernels.
    for name in (
        "causal_conv1d_fn",
        "chunk_kimi_delta_attention",
        "recurrent_kimi_delta_attention",
    ):
        monkeypatch.setattr(hf_model, name, inspect.unwrap(getattr(hf_model, name)))
    torch.manual_seed(7)
    config = tiny_config()
    initialized = Glm5NextTextModel(config)
    reference = cast(Glm5NextTextDecoderLayer, initialized.layers[index])
    layer = cpu_layer(config, reference, index)
    x = (torch.randn(1, 9, 4, 16) * 0.02).requires_grad_()
    candidate_input = x.detach().clone().transpose(0, 1).requires_grad_()
    expected = reference(x, attention_mask=torch.ones(1, 9, dtype=torch.bool))[0]
    actual = layer(candidate_input)[0].transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    assert candidate_input.grad is not None
    assert x.grad is not None
    torch.testing.assert_close(
        candidate_input.grad.transpose(0, 1), x.grad, atol=2e-5, rtol=2e-4
    )
    assert all(
        p.dtype == torch.float32
        for name, p in layer.named_parameters()
        if name.startswith("hc_")
    )


def test_final_norm_contracts_by_mean_and_matches_hf_bf16_rounding() -> None:
    config = SimpleNamespace(params_dtype=torch.bfloat16, sequence_parallel=False)
    norm = Glm5NextFinalNorm(config, hidden_size=16, eps=1e-5)
    x = torch.randn(9, 1, 4, 16, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16))
    mean = x.mean(2)
    value = mean.float()
    expected = norm.weight * (
        value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
    ).to(mean.dtype)
    assert torch.equal(norm(x), expected)
    assert list(norm.state_dict()) == ["weight"]


def test_router_gemm_uses_float32_operands() -> None:
    # Exercise the real override without TopKRouter's GPU-allocated counters.
    router = Glm5NextRouter.__new__(Glm5NextRouter)
    nn.Module.__init__(router)
    router.weight = nn.Parameter(torch.randn(4, 16))
    router.bias = None
    x = torch.randn(9, 16, dtype=torch.bfloat16)
    assert torch.equal(
        router.gating(x), torch.nn.functional.linear(x.float(), router.weight)
    )
    assert router.gating(x).dtype == torch.float32


class _CpuModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = tiny_config()
        reference = Glm5NextTextModel(config)
        self.embedding = reference.embed_tokens
        self.layers = nn.ModuleList(
            [cpu_layer(config, layer, i) for i, layer in enumerate(reference.layers)]
        )
        self.norm = Glm5NextFinalNorm(
            SimpleNamespace(params_dtype=torch.float32), 16, 1e-5
        )
        self.head = nn.Linear(16, 32, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        value = self.embedding(tokens).transpose(0, 1)
        for layer in self.layers:
            value = layer(value)[0]
        return self.head(self.norm(value)).transpose(0, 1)


def test_four_layer_cpu_reference_optimizer_and_base_checkpoint(tmp_path: Path) -> None:
    # A plumbing check with random tiny weights, not the M1 F4 fixture, LoRA
    # learning gate, TrainerRank qualification, or sampler-adapter export.
    torch.manual_seed(41)
    model = _CpuModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    tokens = torch.arange(9).unsqueeze(0)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1)
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
        )
        losses.append(loss.item())
        optimizer.step()
    assert losses[-1] < losses[0]
    path = tmp_path / "base.pt"
    torch.save(model.state_dict(), path)
    restored = _CpuModel()
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("index", [0, 3])
def test_reference_tree_isolates_siblings_and_padding(index: int) -> None:
    torch.manual_seed(81)
    config = tiny_config()
    hf = Glm5NextTextModel(config)
    attention = cpu_layer(config, hf.layers[index], index).self_attention
    value = torch.randn(7, 1, 16, requires_grad=True)
    state = SimpleNamespace(
        group_ids=torch.tensor([[0, 0, 1, 1, 2, 2, -1]]),
        parent_ids=torch.tensor([[0, 0, 0, 0, 0, 0, -1]]),
    )
    actual = attention(value, attention_bias=state)[0]
    root = attention(value[:2])[0]
    left = attention(value[:4])[0][-2:]
    right = attention(torch.cat([value[:2], value[4:6]]))[0][-2:]
    expected = torch.cat([root, left, right, torch.zeros_like(value[:1])])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual[2:4].sum().backward()
    assert value.grad is not None
    assert torch.count_nonzero(value.grad[4:]) == 0
