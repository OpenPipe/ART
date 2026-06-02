from __future__ import annotations

from .output_parity import config_from_env


def test_cp_unsupported_model_uses_non_cp_default_topology(monkeypatch) -> None:
    monkeypatch.setenv("BASE_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TRAINER_GPU_IDS", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_INFERENCE_GPU_IDS", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_CP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_EP", raising=False)

    config = config_from_env()

    assert config.topology.cp == 1
    assert config.topology.tp == 2
    assert config.topology.ep == 4
    assert config.topology.dp == 2
    assert config.trainer_gpu_ids == [0, 1, 2, 3]
    assert config.inference_gpu_ids == [4, 5, 6, 7]
    assert config.engine_args["tensor_parallel_size"] == 4
    assert config.engine_args["enable_expert_parallel"] is True
