from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_lora_delta_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_runtime/src/art_vllm_runtime/lora_delta.py"
    )
    spec = importlib.util.spec_from_file_location("_art_vllm_runtime_lora_delta", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_additive_weight_loader_uses_legacy_loader_for_plain_merged_column_param():
    lora_delta = _load_lora_delta_module()
    param = torch.nn.Parameter(torch.zeros(2, 4))
    loaded = torch.arange(8, dtype=torch.float32).view(2, 4)
    calls = []

    class Owner:
        def weight_loader_v2(self, loader_param, loaded_weight, shard_id):
            del shard_id
            loader_param.load_merged_column_weight(loaded_weight=loaded_weight)

        def weight_loader(self, loader_param, loaded_weight, shard_id):
            calls.append((loader_param, shard_id))
            loader_param.data.copy_(loaded_weight)

    owner = Owner()
    loader = lora_delta._additive_weight_loader(param, owner.weight_loader_v2)
    result = loader(param, loaded, 0)

    assert result is None
    assert calls == [(param, 0)]
    assert torch.equal(param, loaded)


def test_additive_weight_loader_keeps_v2_for_vllm_parameter_like_param():
    lora_delta = _load_lora_delta_module()
    param = torch.nn.Parameter(torch.zeros(2, 4))
    loaded = torch.arange(8, dtype=torch.float32).view(2, 4)
    calls = []

    def load_merged_column_weight(*, loaded_weight, **_kwargs):
        calls.append("v2")
        param.data.copy_(loaded_weight)

    setattr(param, "load_merged_column_weight", load_merged_column_weight)
    owner = SimpleNamespace(
        weight_loader_v2=lambda loader_param, loaded_weight, shard_id: (
            loader_param.load_merged_column_weight(loaded_weight=loaded_weight)
        ),
        weight_loader=lambda *_args, **_kwargs: calls.append("legacy"),
    )
    loader = lora_delta._additive_weight_loader(param, owner.weight_loader_v2)
    loader(param, loaded, 0)

    assert calls == ["v2"]
    assert torch.equal(param, loaded)
