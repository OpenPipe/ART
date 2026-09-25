import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

qwen35 = pytest.importorskip("art.megatron.model_support.handlers.qwen3_5")

REPO = "Qwen/Qwen3.6-35B-A3B"
PIN = "995ad96eacd98c81ed38be0c5b274b04031597b0"
OTHER = "1" * 40
LAYER = "base_model.model.model.language_model.layers.0.self_attn.q_proj"


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    """An offline cache that holds only the snapshots a test adds."""
    import huggingface_hub.constants

    repo = tmp_path / "models--Qwen--Qwen3.6-35B-A3B"

    def add_snapshot(revision: str, groups: int, *, main: bool = False) -> None:
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_attention_heads": 16,
                    "num_key_value_heads": groups,
                    "head_dim": 256,
                    "hidden_size": 2048,
                }
            )
        )
        if main:
            (repo / "refs").mkdir(exist_ok=True)
            (repo / "refs" / "main").write_text(revision)

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    qwen35._qwen35_text_config.cache_clear()
    yield add_snapshot
    qwen35._qwen35_text_config.cache_clear()


def _dims(revision: str | None = None) -> tuple[int, int, int]:
    config: dict[str, object] = {"base_model_name_or_path": REPO}
    if revision is not None:
        config["revision"] = revision
    return qwen35._qwen35_attention_dims(config)


def test_attention_dims_resolve_the_pinned_snapshot_offline(hub_cache):
    hub_cache(PIN, groups=2)
    assert _dims(PIN) == (16, 2, 256)
    # By name alone, the lookup needs refs/main, which this cache lacks.
    with pytest.raises(OSError):
        _dims()


def test_a_different_main_revision_does_not_change_a_pinned_adapter(hub_cache):
    hub_cache(PIN, groups=2)
    hub_cache(OTHER, groups=4, main=True)
    assert [_dims(PIN), _dims(OTHER), _dims(PIN)] == [
        (16, 2, 256),
        (16, 4, 256),
        (16, 2, 256),
    ]
    assert _dims() == (16, 4, 256)  # unpinned adapters still follow main


def _write_adapter(path: Path, rows: int) -> dict[str, torch.Tensor]:
    from art.megatron.model_support.lora_disk import save_vllm_lora_tensors

    tensors = {
        f"{LAYER}.lora_A.weight": torch.randn(2, 32),
        f"{LAYER}.lora_B.weight": torch.randn(rows, 2),
    }
    # As published for a pinned base model, without attention dimensions.
    config = {
        "base_model_name_or_path": REPO,
        "revision": PIN,
        "r": 2,
        "lora_alpha": 32,
        "target_modules": ["q_proj"],
        "art_lora_format": "vllm",
    }
    save_vllm_lora_tensors(path, tensors, config)
    return tensors


@pytest.mark.parametrize("handler", ["QWEN3_5_DENSE_HANDLER", "QWEN3_5_MOE_HANDLER"])
def test_checkpoint_load_converts_with_the_running_models_attention_shape(
    tmp_path, monkeypatch, handler
):
    from art.trainer_rank import _checkpoint

    def lookup(*_args):
        raise AssertionError("adapter conversion looked the base model up again")

    monkeypatch.setattr(qwen35, "_qwen35_text_config", lookup)
    provider = SimpleNamespace(
        num_attention_heads=4, num_query_groups=2, kv_channels=8, hidden_size=32
    )
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(
            provider=provider, model_support_handler=getattr(qwen35, handler)
        )
    )
    # Two query groups of two heads, each with query and gate rows of size 8.
    tensors = _write_adapter(tmp_path, rows=2 * 2 * 2 * 8)
    art_key = f"{LAYER}.lora_B.weight".replace(".language_model.layers.", ".layers.")
    loaded = _checkpoint._load_adapter(
        cast(Any, trainer),
        cast(Any, SimpleNamespace(manifest=None, path=tmp_path)),
        [art_key],
    )
    expected = qwen35._qwen35_q_proj_lora_b_from_vllm(
        tensors[f"{LAYER}.lora_B.weight"],
        {"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8},
    )
    torch.testing.assert_close(loaded[art_key], expected)


def test_adapter_dimensions_take_precedence_over_the_running_model(
    tmp_path, monkeypatch
):
    from art.megatron.model_support import lora_disk
    from art.trainer_rank import _checkpoint

    seen = []
    monkeypatch.setattr(
        lora_disk,
        "load_lora_tensors_for_megatron",
        lambda path, **kwargs: seen.append(kwargs["adapter_config"]) or {},
    )
    _write_adapter(tmp_path, rows=8)
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    config["num_attention_heads"] = 1
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(
            provider=SimpleNamespace(num_attention_heads=4, num_query_groups=2),
            model_support_handler=qwen35.QWEN3_5_MOE_HANDLER,
        )
    )
    _checkpoint._load_adapter(
        cast(Any, trainer),
        cast(Any, SimpleNamespace(manifest=None, path=tmp_path)),
        [],
    )
    assert seen[0]["num_attention_heads"] == 1
    assert seen[0]["num_key_value_heads"] == 2
    assert seen[0]["revision"] == PIN
