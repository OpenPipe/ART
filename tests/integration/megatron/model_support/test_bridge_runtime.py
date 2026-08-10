from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

pytest.importorskip("megatron.bridge")

from art.megatron.runtime.bridge_runtime import (
    _optimized_load_weights_hf_to_megatron,
)


class _Mapping:
    def __init__(self, megatron_param: str, hf_param: str) -> None:
        self.megatron_param = megatron_param
        self.hf_param = hf_param
        self.tp_size = 1

    def hf_to_megatron(
        self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module
    ) -> torch.Tensor:
        del megatron_module
        return hf_weights


class _Bridge:
    def __init__(self, tasks: list[Any]) -> None:
        self.tasks = tasks

    def build_conversion_tasks(
        self, hf_pretrained: Any, megatron_model: Any
    ) -> list[Any]:
        del hf_pretrained, megatron_model
        return self.tasks

    def _share_embeddings_and_output_weights(self, config: Any) -> bool:
        return bool(config.share_embeddings_and_output_weights)

    def _is_adapter_param_name(self, name: str) -> bool:
        return ".adapter." in name

    def _with_progress_tracking(self, tasks: list[Any], description: str) -> list[Any]:
        del description
        return tasks

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return state[hf_param]

    def _broadcast_shared_embeddings(self, megatron_model: Any) -> None:
        del megatron_model


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(share_embeddings_and_output_weights=False)
        self.local = torch.nn.Linear(1, 1, bias=False)


def _task(
    mapping: _Mapping,
    *,
    module: torch.nn.Module | None = None,
    weight: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        mapping=mapping,
        megatron_module=module,
        param_weight=weight,
        param_name=mapping.megatron_param,
    )


def test_pretrained_load_rejects_placeholder_for_required_local_parameter() -> None:
    model = _Model()
    bridge = _Bridge([_task(_Mapping("local.weight", "hf.weight"))])
    pretrained = SimpleNamespace(state={}, model_name_or_path="empty-checkpoint")

    with pytest.raises(
        RuntimeError,
        match=r"1 required local parameter\(s\): local.weight",
    ):
        _optimized_load_weights_hf_to_megatron(cast(Any, bridge), pretrained, model)


def test_pretrained_load_allows_nonlocal_placeholder_tasks() -> None:
    model = _Model()
    local_mapping = _Mapping("local.weight", "hf.weight")
    remote_mapping = _Mapping("remote.weight", "hf.remote_weight")
    bridge = _Bridge(
        [
            _task(local_mapping, module=model.local, weight=model.local.weight),
            _task(remote_mapping),
        ]
    )
    expected = torch.tensor([[7.0]])
    pretrained = SimpleNamespace(
        state={"hf.weight": expected}, model_name_or_path="checkpoint"
    )

    result = _optimized_load_weights_hf_to_megatron(
        cast(Any, bridge), pretrained, model
    )

    assert result == [model]
    assert torch.equal(model.local.weight, expected)
