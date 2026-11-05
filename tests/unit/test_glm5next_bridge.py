from types import SimpleNamespace

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
import pytest
import torch

from art.megatron.glm5next.bridge import (
    ArtGlm5NextBridge,
    ensure_glm5next_bridge_registered,
    glm5next_mapping_registry,
    training_checkpoint_path,
)
from art.megatron.glm5next.config import (
    SERVING_MODEL,
    TRAINING_MODEL,
    TRAINING_REVISION,
)
from art.megatron.glm5next.hf_config import Glm5NextConfig, Glm5NextTextConfig
from art.megatron.glm5next.spec import get_glm5next_decoder_block_spec
from tests.unit.test_glm5next_config import production_config


def test_config_only_provider_on_pinned_bridge() -> None:
    ensure_glm5next_bridge_registered()
    config = Glm5NextConfig(
        text_config=vars(production_config()),
        architectures=["Glm5NextForConditionalGeneration"],
    )
    bridge = AutoBridge.from_hf_config(config)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(bridge._model_bridge, ArtGlm5NextBridge)
    assert provider.num_layers == 45
    assert provider.kv_channels == 256
    assert provider.vocab_size == 154880
    assert provider.transformer_layer_spec is get_glm5next_decoder_block_spec


def test_text_fixture_architecture_dispatches_to_same_bridge() -> None:
    ensure_glm5next_bridge_registered()
    config = Glm5NextTextConfig(
        **vars(production_config()), architectures=["Glm5NextTextOracle"]
    )
    bridge = AutoBridge.from_hf_config(config)
    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(bridge._model_bridge, ArtGlm5NextBridge)
    assert provider.num_layers == 45


def test_native_specs_keep_independent_attention_and_mlp_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import ModuleSpec
    from torch import nn

    ensure_glm5next_bridge_registered()
    cfg = Glm5NextConfig(
        text_config=vars(production_config()),
        architectures=["Glm5NextForConditionalGeneration"],
    )
    provider = AutoBridge.from_hf_config(cfg).to_megatron_provider(load_weights=False)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    # Distinguishable specs expose accidental reuse of one shared MoE spec.
    setattr(
        provider,
        "glm5next_kda_spec",
        ModuleSpec(module=nn.Linear, params={"family": "kda"}),
    )
    setattr(
        provider,
        "glm5next_dsa_spec",
        ModuleSpec(module=nn.Linear, params={"family": "dsa"}),
    )
    block = get_glm5next_decoder_block_spec(provider)
    assert len(block.layer_specs) == len({id(s) for s in block.layer_specs}) == 45
    for index, spec in enumerate(block.layer_specs):
        assert spec.submodules.self_attention.params["family"] == (
            "dsa" if index % 4 == 3 else "kda"
        )
        assert (spec.submodules.mlp.module is MoELayer) == (index >= 3)


@pytest.mark.parametrize("model", [SERVING_MODEL, TRAINING_MODEL])
def test_training_source_is_pinned_before_lazy_state_loader(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/verified/local/bf16/snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert training_checkpoint_path(model) == "/verified/local/bf16/snapshot"
    assert calls[0]["repo_id"] == TRAINING_MODEL
    assert calls[0]["revision"] == TRAINING_REVISION
    assert training_checkpoint_path("/fixtures/F4") == "/fixtures/F4"
    assert len(calls) == 1


def test_mhc_mapping_is_direct_and_has_no_learned_output_head() -> None:
    mappings = glm5next_mapping_registry().mappings
    names = {m.megatron_param: m for m in mappings}
    for site in ("attn", "ffn"):
        for field in ("fn", "base", "scale"):
            mapping = names[f"decoder.layers.*.hc_{site}_{field}"]
            assert isinstance(mapping, ReplicatedMapping)
            assert (
                mapping.hf_param == f"model.language_model.layers.*.hc_{site}_{field}"
            )
    assert not any("hc_head" in name for name in names)
    assert not any("visual" in name or "mtp" in name for name in names)


def test_fp8_local_checkpoint_is_rejected_instead_of_dropping_scales() -> None:
    bridge = ArtGlm5NextBridge()
    weight = torch.zeros(2, 2).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="requires BF16"):
        bridge.maybe_modify_loaded_hf_weight(
            "lm_head.weight", {"lm_head.weight": weight}
        )


def test_handler_export_identity_is_serving_model() -> None:
    from art.megatron.model_support.handlers.glm5next import GLM5NEXT_HANDLER

    config = {"r": 32, "base_model_name_or_path": TRAINING_MODEL}
    assert (
        GLM5NEXT_HANDLER.to_vllm_lora_config(config)["base_model_name_or_path"]
        == SERVING_MODEL
    )
    assert config["base_model_name_or_path"] == TRAINING_MODEL
    provider = SimpleNamespace(
        linear_attention_freq=[1, 1, 1, 0], moe_layer_freq=[0, 0, 0, 1]
    )
    assert {f.key for f in GLM5NEXT_HANDLER.collect_layer_families(provider)} == {
        "mhc",
        "kda_attention",
        "nope_dsa_attention",
        "dense_mlp",
        "grouped_moe_mlp",
        "shared_experts_mlp",
    }


def test_handler_wraps_kda_and_head_and_roundtrips_serving_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch import nn

    from art.megatron import lora as lm
    from art.megatron.glm5next.kda import KimiDeltaAttention
    from art.megatron.model_support.handlers.glm5next import GLM5NEXT_HANDLER

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
    cfg = production_config()
    cfg.num_hidden_layers = 1
    cfg.hidden_size, cfg.vocab_size = 8, 16
    cfg.layer_types, cfg.mlp_layer_types, cfg.indexer_types = (
        ["linear_attention"],
        ["dense"],
        ["full"],
    )
    cfg.linear_attn_config = {
        "num_heads": 1,
        "head_dim": 8,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
    }
    from art.megatron.glm5next.config import configure_glm5next_provider

    provider = SimpleNamespace(
        params_dtype=torch.float32,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )
    configure_glm5next_provider(provider, cfg)
    chunk = nn.Module()
    chunk.decoder = nn.Module()
    layer = nn.Module()
    setattr(layer, "layer_number", 1)
    layer.self_attention = KimiDeltaAttention(provider, backend="reference")
    layer.mlp = nn.Identity()
    chunk.decoder.layers = nn.ModuleList([layer])
    chunk.output_layer = nn.Linear(8, 16, bias=False)
    q_weight = layer.self_attention.q_proj.weight
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "lm_head",
    ]
    GLM5NEXT_HANDLER.apply_lora_adapters(
        [chunk], provider, target_modules=targets, rank=32, alpha=32
    )
    assert layer.self_attention.q_proj.weight is q_weight
    assert not q_weight.requires_grad
    assert not layer.self_attention.b_proj.weight.requires_grad
    optimizer = torch.optim.AdamW(
        [p for p in chunk.parameters() if p.requires_grad], lr=0.001
    )
    torch.manual_seed(19)
    inputs = torch.randn(8, 1, 8)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        logits = chunk.output_layer(layer.self_attention(inputs)[0]).flatten(0, 1)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(8))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in chunk.parameters()
        )
        losses.append(loss.item())
        optimizer.step()
    assert losses[-1] < losses[0]
    native = {}
    for module in chunk.modules():
        if isinstance(module, lm.LoRA):
            native.update(module.sharded_lora_state_dict())
    assert len(native) == 10
    wrapped = nn.Module()
    wrapped.module = chunk
    config = GLM5NEXT_HANDLER.prepare_adapter_config(
        {"r": 32, "lora_alpha": 32}, [wrapped]
    )
    serving, serving_config = GLM5NEXT_HANDLER.to_vllm_lora_tensors(
        native, adapter_config=config
    )
    assert serving_config["base_model_name_or_path"] == SERVING_MODEL
    restored = GLM5NEXT_HANDLER.from_vllm_lora_tensors(
        serving, adapter_config=serving_config
    )
    assert restored.keys() == native.keys()
    for key in native:
        torch.testing.assert_close(restored[key], native[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="changed r"):
        GLM5NEXT_HANDLER.prepare_adapter_config({"r": 8}, [chunk])
