from types import SimpleNamespace
from typing import Any

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import ReplicatedMapping
from megatron.bridge.models.hf_pretrained.state import StateDict, StateSource
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
import pytest
import torch
from torch import nn
import torch.distributed as dist

from art.megatron.glm5next.bridge import ArtGlm5NextBridge
from art.megatron.glm5next.source import Glm5NextTaskSource, materialize_hf_weight
from art.megatron.runtime.bridge_runtime import (
    _optimized_load_weights_hf_to_megatron,
    _validate_local_pretrained_tasks,
)


def test_real_bridge_task_planner_accepts_only_backed_transformers_aliases(tmp_path):
    prefix = "model.language_model.layers.0."
    weights = {
        prefix + "attn_hc.fn": torch.arange(4).reshape(2, 2).float(),
        prefix + "self_attn.forget_gate.A_log": torch.arange(2).float(),
        prefix + "self_attn.conv1d.weight": torch.arange(24).reshape(6, 1, 4).float(),
        prefix + "mlp.experts.gate_up_proj": torch.arange(32).reshape(2, 8, 2).float(),
        prefix + "mlp.experts.down_proj": torch.arange(16).reshape(2, 2, 4).float(),
        prefix + "mlp.gate.e_score_correction_bias": torch.tensor([0.12345, 0.98765]),
    }
    model: Any = nn.Module()
    model.config = TransformerConfig(num_layers=1, hidden_size=2, num_attention_heads=1)
    setattr(model.config, "share_embeddings_and_output_weights", False)
    model.decoder = nn.Module()
    layer = nn.Module()
    model.decoder.layers = nn.ModuleList([layer])
    layer.hc_attn_fn = nn.Parameter(torch.empty(2, 2))
    layer.self_attention = nn.Module()
    layer.self_attention.A_log = nn.Parameter(torch.empty(2))
    layer.self_attention.q_conv1d = nn.Conv1d(2, 2, 4, groups=2, bias=False)
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    for projection in ("linear_fc1", "linear_fc2"):
        linear = nn.Module()
        setattr(layer.mlp.experts, projection, linear)
        for expert in range(2):
            linear.register_parameter(
                f"weight{expert}",
                nn.Parameter(
                    torch.empty(8, 2)
                    if projection == "linear_fc1"
                    else torch.empty(2, 4)
                ),
            )
    layer.mlp.router = nn.Module()
    layer.mlp.router.register_buffer("expert_bias", torch.zeros(2))
    pretrained = SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=2), state=StateDict(weights)
    )
    source = pretrained.state.source
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1
    )
    try:
        parallel_state.initialize_model_parallel()
        bridge = ArtGlm5NextBridge()
        old_tasks = MegatronModelBridge.build_conversion_tasks(
            bridge, pretrained, [model]
        )
        with pytest.raises(RuntimeError, match="7 required local parameter"):
            _validate_local_pretrained_tasks(bridge, [model], old_tasks)
        tasks = bridge.build_conversion_tasks(pretrained, [model])
        _validate_local_pretrained_tasks(bridge, [model], tasks)
        assert sum(t is not None and t.megatron_module is not None for t in tasks) == 8
        assert pretrained.state.source is source  # virtual keys never escape planning
        assert all("hc_attn" not in k for k in source.get_all_keys())
        assert torch.equal(
            materialize_hf_weight(
                prefix + "mlp.experts.1.up_proj.weight", pretrained.state
            ),
            weights[prefix + "mlp.experts.gate_up_proj"][1, 4:],
        )
        assert torch.equal(
            materialize_hf_weight(
                prefix + "self_attn.q_conv1d.weight", pretrained.state
            ),
            weights[prefix + "self_attn.conv1d.weight"][:2],
        )
        # A physically absent fused tensor must still fail the coverage check.
        del weights[prefix + "self_attn.conv1d.weight"]
        broken = SimpleNamespace(config=pretrained.config, state=StateDict(weights))
        with pytest.raises(RuntimeError, match="1 required local parameter"):
            _validate_local_pretrained_tasks(
                bridge, [model], bridge.build_conversion_tasks(broken, [model])
            )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_task_source_never_invents_unbacked_aliases():
    keys = ["x.mlp.experts.gate_up_proj", "x.self_attn.q_conv1d.weight"]
    source = SimpleNamespace(get_all_keys=lambda: keys)
    logical = Glm5NextTaskSource(source, 2).get_all_keys()
    assert "x.mlp.experts.1.up_proj.weight" in logical
    assert "x.mlp.experts.2.up_proj.weight" not in logical
    assert "x.mlp.experts.0.down_proj.weight" not in logical
    assert "x.self_attn.k_conv1d.weight" not in logical


def test_glm_import_copies_each_task_before_staging_the_next(tmp_path, monkeypatch):
    model: Any = nn.Module()
    model.config = SimpleNamespace(share_embeddings_and_output_weights=False)
    model.local = nn.Linear(1, 1, bias=False)
    model.second = nn.Linear(1, 1, bias=False)
    reads = []

    class OrderedSource(StateSource):
        def get_all_keys(self):
            return ["a", "b"]

        def load_tensors(self, keys):
            for key in keys:
                if key == "b":
                    # Bulk prefetch reaches b before a has been copied. One
                    # task at a time bounds staging for both DP replicas.
                    assert float(model.local.weight.detach()[0, 0]) == 1.5
                reads.append(key)
            return {key: torch.full((1, 1), 1.5 if key == "a" else 2.5) for key in keys}

    bridge = ArtGlm5NextBridge()
    monkeypatch.setattr(
        bridge,
        "mapping_registry",
        lambda: MegatronMappingRegistry(
            ReplicatedMapping("local.weight", "a"),
            ReplicatedMapping("second.weight", "b"),
        ),
    )
    pretrained = SimpleNamespace(
        config=SimpleNamespace(n_routed_experts=1),
        state=StateDict(OrderedSource()),
        model_name_or_path=str(tmp_path),
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1
    )
    try:
        parallel_state.initialize_model_parallel()
        monkeypatch.setattr(bridge, "art_stream_import", False)
        with pytest.raises(AssertionError):
            _optimized_load_weights_hf_to_megatron(bridge, pretrained, [model])
        reads.clear()
        monkeypatch.setattr(bridge, "art_stream_import", True)
        _optimized_load_weights_hf_to_megatron(bridge, pretrained, [model])
        assert reads == ["a", "b"]
        assert float(model.second.weight.detach()[0, 0]) == 2.5
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
