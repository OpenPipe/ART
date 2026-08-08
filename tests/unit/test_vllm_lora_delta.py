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
        def weight_loader_v2(self, loader_param, loaded_weight, shard_id, **kwargs):
            del shard_id, kwargs
            loader_param.load_merged_column_weight(loaded_weight=loaded_weight)

        def weight_loader(self, loader_param, loaded_weight, shard_id, **kwargs):
            calls.append((loader_param, shard_id, kwargs))
            loader_param.data.copy_(loaded_weight)

    owner = Owner()
    loader = lora_delta._additive_weight_loader(owner.weight_loader_v2, {})
    result = loader(
        param=param,
        loaded_weight=loaded,
        shard_id=0,
        return_success=True,
    )

    assert result is None
    assert calls == [(param, 0, {"return_success": True})]
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
    loader = lora_delta._additive_weight_loader(owner.weight_loader_v2, {})
    loader(param, loaded, 0)

    assert calls == ["v2"]
    assert torch.equal(param, loaded)


def test_delta_update_normalizes_missing_quantization_config_during_load():
    lora_delta = _load_lora_delta_module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.config = SimpleNamespace(quantization_config=None)

        def load_weights(self, weights):
            assert self.config.quantization_config == {"quant_method": None}
            for _name, weight in weights:
                getattr(self.weight, "weight_loader")(self.weight, weight)

    model = Model()
    tensors = {
        "base_model.model.weight.lora_A.weight": torch.eye(2),
        "base_model.model.weight.lora_B.weight": torch.eye(2),
    }
    lora_delta.apply_lora_delta_update(
        model=model,
        lora_tensors=tensors,
        adapter_config={"r": 2, "lora_alpha": 2},
        previous_lora_tensors=None,
    )

    assert model.config.quantization_config is None
    assert torch.equal(model.weight, torch.eye(2))


def test_block_fp8_delta_requantizes_weight_and_e8m0_scale() -> None:
    lora_delta = _load_lora_delta_module()
    weight = torch.full((4, 4), 0.5).to(torch.float8_e4m3fn)
    param = torch.nn.Parameter(weight, requires_grad=False)
    scale = torch.nn.Parameter(
        torch.full((2, 2), 0.25).to(torch.float8_e8m0fnu),
        requires_grad=False,
    )
    setattr(param, lora_delta._BLOCK_FP8_SCALE_ATTR, scale)
    setattr(param, lora_delta._BLOCK_FP8_SIZE_ATTR, (2, 2))
    delta = torch.zeros(4, 4)
    delta[:2, :2] = 0.5

    lora_delta._requantize_block_fp8_delta(param, delta)

    expanded = scale.float().repeat_interleave(2, 0).repeat_interleave(2, 1)
    merged = param.float() * expanded
    assert torch.allclose(merged[:2, :2], torch.full((2, 2), 0.625))
    assert torch.allclose(merged[2:, 2:], torch.full((2, 2), 0.125))


def test_block_fp8_delta_supports_expert_leading_dimension() -> None:
    lora_delta = _load_lora_delta_module()
    param = torch.nn.Parameter(
        torch.ones(2, 4, 4).to(torch.float8_e4m3fn), requires_grad=False
    )
    scale = torch.nn.Parameter(torch.ones(2, 2, 2), requires_grad=False)
    setattr(param, lora_delta._BLOCK_FP8_SCALE_ATTR, scale)
    setattr(param, lora_delta._BLOCK_FP8_SIZE_ATTR, (2, 2))
    delta = torch.zeros(2, 4, 4)
    delta[1] = 1.0

    lora_delta._requantize_block_fp8_delta(param, delta)

    expanded = scale.repeat_interleave(2, -2).repeat_interleave(2, -1)
    merged = param.float() * expanded
    assert torch.allclose(merged[0], torch.ones(4, 4))
    assert torch.allclose(merged[1], torch.full((4, 4), 2.0))


def test_block_fp8_delta_supports_grouped_matrix_layout() -> None:
    lora_delta = _load_lora_delta_module()
    param = torch.nn.Parameter(
        torch.ones(2, 2, 4).to(torch.float8_e4m3fn), requires_grad=False
    )
    scale = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    setattr(param, lora_delta._BLOCK_FP8_SCALE_ATTR, scale)
    setattr(param, lora_delta._BLOCK_FP8_SIZE_ATTR, (2, 2))

    lora_delta._requantize_block_fp8_delta(param, torch.ones_like(param).float())

    expanded = scale.repeat_interleave(2, 0).repeat_interleave(2, 1)
    merged = param.flatten(0, 1).float() * expanded
    assert torch.allclose(merged, torch.full((4, 4), 2.0))


def test_block_fp8_expert_loader_updates_only_local_shard() -> None:
    lora_delta = _load_lora_delta_module()
    param = torch.nn.Parameter(
        torch.ones(2, 4, 4).to(torch.float8_e4m3fn), requires_grad=False
    )
    scale = torch.nn.Parameter(torch.ones(2, 2, 2), requires_grad=False)
    setattr(param, lora_delta._BLOCK_FP8_SCALE_ATTR, scale)
    setattr(param, lora_delta._BLOCK_FP8_SIZE_ATTR, (2, 2))

    class Owner:
        @staticmethod
        def _map_global_expert_id_to_local_expert_id(expert_id):
            return {4: 1}.get(expert_id, -1)

        def weight_loader(self, *_args, **_kwargs):
            raise AssertionError("packed expert loader must not allocate full scratch")

    loader = lora_delta._additive_weight_loader(Owner().weight_loader, {})
    assert loader(
        param,
        torch.ones(2, 4),
        shard_id="w3",
        expert_id=4,
    )
    assert not loader(
        param,
        torch.ones(2, 4),
        shard_id="w3",
        expert_id=5,
    )

    expanded = scale.repeat_interleave(2, -2).repeat_interleave(2, -1)
    merged = param.float() * expanded
    assert torch.allclose(merged[0], torch.ones(4, 4))
    assert torch.allclose(merged[1, :2], torch.ones(2, 4))
    assert torch.allclose(merged[1, 2:], torch.full((2, 4), 2.0))
