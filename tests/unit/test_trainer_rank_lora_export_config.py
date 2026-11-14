"""TrainerRank publication must prepare model-owned metadata before collection."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from art.megatron import weights
from art.trainer_rank._lora_export import _prepare_vllm_lora_publish


@pytest.mark.parametrize("has_hook", [True, False])
def test_prepare_adapter_config_precedes_layout_and_metadata(
    monkeypatch: pytest.MonkeyPatch, has_hook: bool
) -> None:
    events: list[str] = []
    model: Any = [torch.nn.Linear(1, 1)]
    original = {"r": 8}
    prepared_config = {**original, "model_metadata": "prepared"}

    def prepare_config(config: dict[str, Any], chunks: Any) -> dict[str, Any]:
        assert config is original
        assert chunks is model
        events.append("prepare")
        return prepared_config

    def record_layout(config: dict[str, Any]) -> dict[str, Any]:
        assert config is (prepared_config if has_hook else original)
        events.append("layout")
        return {**config, "layout": "recorded"}

    def packed_groups() -> tuple:
        events.append("groups")
        return ()

    def collect_entries(*args: Any, **kwargs: Any) -> tuple[dict, list]:
        events.append("metadata")
        return {}, []

    def collect_packed_entries(*args: Any, **kwargs: Any) -> tuple[dict, list]:
        events.append("packed_metadata")
        return {}, []

    handler = SimpleNamespace(expert_packed_lora_groups=packed_groups)
    if has_hook:
        handler.prepare_adapter_config = prepare_config
    module = SimpleNamespace(lora_layout=SimpleNamespace(record_config=record_layout))
    # Collection is outside this orchestration test and imports GPU dependencies.
    publish = ModuleType("art.megatron.weights.lora_publish")
    monkeypatch.setattr(
        publish, "iter_lora_modules", lambda _: iter([module]), raising=False
    )
    monkeypatch.setattr(
        publish, "collect_local_lora_entries", collect_entries, raising=False
    )
    monkeypatch.setattr(
        publish,
        "collect_local_packed_expert_entries",
        collect_packed_entries,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, publish.__name__, publish)
    monkeypatch.setattr(weights, "lora_publish", publish, raising=False)

    plan = _prepare_vllm_lora_publish(
        model=model,
        adapter_dtypes={},
        handler=handler,
        adapter_config=original,
        rank=0,
        world_size=1,
        runtime=(0, torch.device("cpu")),
    )

    assert events == (["prepare"] if has_hook else []) + [
        "layout",
        "groups",
        "metadata",
        "packed_metadata",
    ]
    assert plan.adapter_config == {
        **(prepared_config if has_hook else original),
        "layout": "recorded",
    }
    assert original == {"r": 8}
