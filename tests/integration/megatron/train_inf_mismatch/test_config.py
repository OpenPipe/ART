from __future__ import annotations

from .output_parity import config_from_env


def test_cp_unsupported_model_uses_non_cp_default_topology(monkeypatch) -> None:
    monkeypatch.setenv("BASE_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_CP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_EP", raising=False)

    config = config_from_env()

    assert config.topology.cp == 1
    assert config.topology.tp == 2
    assert config.topology.ep == 2
