from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.integration.megatron.train_inf_mismatch import real_path
from tests.integration.megatron.train_inf_mismatch.output_parity import (
    TrainInfOutputParityConfig,
)
from tests.integration.megatron.train_inf_mismatch.real_path import RealPathConfig


def _config(cache_dir: Path) -> RealPathConfig:
    return RealPathConfig(
        output_parity=TrainInfOutputParityConfig(
            base_model="Qwen/Qwen3-32B",
            lora_target_modules=["q_proj"],
            rollout_modes=["native_lora"],
        ),
        adapter_cache_dir=str(cache_dir),
    )


def _write_adapter(path: Path, *, payload: bytes = b"non-identity") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_model.safetensors").write_bytes(payload)
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")


def _cache_dir(config: RealPathConfig) -> Path:
    assert config.adapter_cache_dir is not None
    return Path(config.adapter_cache_dir)


def test_make_or_reuse_nonzero_adapter_reuses_valid_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    generated = tmp_path / "generated"
    _write_adapter(generated)
    calls = 0

    def make_adapter(*, config, artifact_dir):
        nonlocal calls
        calls += 1
        return str(generated)

    monkeypatch.setattr(real_path, "_make_nonzero_adapter", make_adapter)
    config = _config(tmp_path / "cache")

    first = real_path._make_or_reuse_nonzero_adapter(
        config=config,
        artifact_dir=tmp_path / "artifact_0",
    )
    second = real_path._make_or_reuse_nonzero_adapter(
        config=config,
        artifact_dir=tmp_path / "artifact_1",
    )
    first_path = Path(first.path)
    second_path = Path(second.path)

    assert calls == 1
    assert not first.cache_hit
    assert second.cache_hit
    assert first_path == second_path
    assert first.cache_key == second.cache_key
    assert first_path == _cache_dir(config) / real_path._adapter_cache_key(
        config.output_parity
    )
    assert (first_path / "adapter_model.safetensors").read_bytes() == b"non-identity"


def test_cached_adapter_requires_non_identity_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path / "cache")
    cache_key = real_path._adapter_cache_key(config.output_parity)
    adapter = _cache_dir(config) / cache_key
    _write_adapter(adapter)
    real_path._write_json(
        real_path._adapter_cache_manifest_path(adapter),
        {
            "cache_key": cache_key,
            "non_identity": False,
            "adapter_model_bytes": (adapter / "adapter_model.safetensors")
            .stat()
            .st_size,
        },
    )

    assert not real_path._cached_adapter_is_valid(adapter, cache_key=cache_key)
