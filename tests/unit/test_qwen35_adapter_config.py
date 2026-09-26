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
    assert _dims("") == (16, 4, 256)  # an empty revision is unpinned


def test_missing_dimensions_come_from_the_pinned_config_not_defaults(hub_cache):
    hub_cache(PIN, groups=2)
    config = {
        "base_model_name_or_path": REPO,
        "revision": PIN,
        "num_attention_heads": 16,
        "hidden_size": 2048,
    }
    # Not one group per head, nor head_dim = hidden_size / heads = 128.
    assert qwen35._qwen35_attention_dims(config) == (16, 2, 256)


def _write_adapter(
    path: Path, rows: int, **dimensions: object
) -> dict[str, torch.Tensor]:
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
        **dimensions,
    }
    save_vllm_lora_tensors(path, tensors, config)
    return tensors


# Four heads in two query groups, with head_dim 8: the runtime's shape. The
# hidden size is not heads x head_dim, as in Qwen3.5, so deriving head_dim from
# it would be caught.
PROVIDER = SimpleNamespace(
    num_attention_heads=4, num_query_groups=2, kv_channels=8, hidden_size=48
)
ROWS = 2 * 2 * 2 * 8  # groups x (query + gate) x heads per group x head_dim
ART_KEY = f"{LAYER}.lora_B.weight".replace(".language_model.layers.", ".layers.")


def _expected(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return qwen35._qwen35_q_proj_lora_b_from_vllm(
        tensors[f"{LAYER}.lora_B.weight"],
        {"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8},
    )


def _forbid_lookup(monkeypatch) -> None:
    def lookup(*_args):
        raise AssertionError("adapter conversion looked the base model up again")

    monkeypatch.setattr(qwen35, "_qwen35_text_config", lookup)


@pytest.mark.parametrize("handler", ["QWEN3_5_DENSE_HANDLER", "QWEN3_5_MOE_HANDLER"])
def test_checkpoint_load_converts_with_the_running_models_attention_shape(
    tmp_path, monkeypatch, handler
):
    from art.trainer_rank import _checkpoint

    _forbid_lookup(monkeypatch)
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(
            provider=PROVIDER, model_support_handler=getattr(qwen35, handler)
        )
    )
    tensors = _write_adapter(tmp_path, ROWS)
    loaded = _checkpoint._load_adapter(
        cast(Any, trainer),
        cast(Any, SimpleNamespace(manifest=None, path=tmp_path)),
        [ART_KEY],
    )
    torch.testing.assert_close(loaded[ART_KEY], _expected(tensors))


def test_null_adapter_dimensions_are_filled_from_the_running_model(
    tmp_path, monkeypatch
):
    from art.megatron.model_support.lora_disk import load_lora_tensors_for_megatron

    _forbid_lookup(monkeypatch)
    # A null group count must not fall back to one group per head.
    tensors = _write_adapter(tmp_path, ROWS, num_key_value_heads=None, head_dim=None)
    loaded = load_lora_tensors_for_megatron(
        tmp_path, handler=qwen35.QWEN3_5_MOE_HANDLER, provider=PROVIDER
    )
    torch.testing.assert_close(loaded[ART_KEY], _expected(tensors))


@pytest.mark.parametrize("handler", ["QWEN3_5_DENSE_HANDLER", "QWEN3_5_MOE_HANDLER"])
def test_adapter_and_model_dimensions_combine(tmp_path, monkeypatch, handler):
    from art.megatron.model_support.lora_disk import load_lora_tensors_for_megatron

    _forbid_lookup(monkeypatch)
    # The adapter has heads and head size, the model has query groups but no
    # head size: together they are complete.
    tensors = _write_adapter(tmp_path, ROWS, num_attention_heads=4, head_dim=8)
    provider = SimpleNamespace(num_attention_heads=4, num_query_groups=2)
    loaded = load_lora_tensors_for_megatron(
        tmp_path, handler=getattr(qwen35, handler), provider=provider
    )
    torch.testing.assert_close(loaded[ART_KEY], _expected(tensors))


def test_adapter_dimensions_take_precedence_over_the_running_model():
    from art.megatron.model_support.lora_disk import with_model_attention_dimensions

    config = with_model_attention_dimensions(
        {"revision": PIN, "num_attention_heads": 1, "num_key_value_heads": None},
        PROVIDER,
    )
    assert config == {
        "revision": PIN,
        "num_attention_heads": 1,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 48,
    }


@pytest.mark.parametrize(
    "provider",
    [
        # Query groups must not default to one per head.
        SimpleNamespace(num_attention_heads=4, kv_channels=8, hidden_size=48),
        # The head size must not be derived from the hidden size.
        SimpleNamespace(num_attention_heads=4, num_query_groups=2, hidden_size=48),
    ],
    ids=["no-query-groups", "no-head-size"],
)
def test_dimensions_the_model_lacks_come_from_the_pinned_lookup(
    tmp_path, monkeypatch, provider
):
    from art.megatron.model_support.lora_disk import load_lora_tensors_for_megatron

    looked_up = []

    def lookup(name, revision):
        looked_up.append((name, revision))
        return SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=8)

    monkeypatch.setattr(qwen35, "_qwen35_text_config", lookup)
    tensors = _write_adapter(tmp_path, ROWS)
    loaded = load_lora_tensors_for_megatron(
        tmp_path, handler=qwen35.QWEN3_5_MOE_HANDLER, provider=provider
    )
    assert looked_up == [(REPO, PIN)]
    torch.testing.assert_close(loaded[ART_KEY], _expected(tensors))


def test_export_of_a_pinned_adapter_resolves_its_revision(hub_cache):
    hub_cache(PIN, groups=2)
    tensor = torch.randn(2 * 2 * 8 * 256, 2)  # 16 heads in 2 groups, head_dim 256
    config = {"base_model_name_or_path": REPO, "revision": PIN}
    art_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    exported, _ = qwen35.QWEN3_5_MOE_HANDLER.to_vllm_lora_tensors(
        {art_key: tensor}, adapter_config=config
    )
    expected = qwen35._qwen35_q_proj_lora_b_to_vllm(
        tensor, {"num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256}
    )
    torch.testing.assert_close(exported[f"{LAYER}.lora_B.weight"], expected)


def test_megatron_service_adapter_load_passes_the_running_model(monkeypatch):
    train = pytest.importorskip("art.megatron.train")
    seen = []
    monkeypatch.setattr(
        train,
        "load_lora_tensors_for_megatron",
        lambda path, **kwargs: seen.append(kwargs) or {},
    )
    monkeypatch.setattr(train, "load_adapter_into_model", lambda *a, **k: None)
    train._load_adapter_into_model([], "adapter", 0, handler=None, provider=PROVIDER)
    assert seen[0]["provider"] is PROVIDER
