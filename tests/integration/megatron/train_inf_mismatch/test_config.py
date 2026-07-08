from __future__ import annotations

from types import SimpleNamespace

import torch

from . import output_parity
from .output_parity import config_from_env


def test_cp_unsupported_default_converts_cp_to_dp_without_changing_tp(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_CP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_DP", raising=False)
    monkeypatch.setattr(
        output_parity,
        "handler_workflow_resources_for_base_model",
        lambda base_model, *, allow_unvalidated_arch=False: None,
    )
    monkeypatch.setattr(output_parity, "model_support_is_moe", lambda *_, **__: True)
    monkeypatch.setattr(
        output_parity,
        "model_supports_context_parallel",
        lambda *_, **__: False,
    )

    config = config_from_env()

    assert config.topology.tp == 1
    assert config.topology.cp == 1
    assert config.topology.dp == 2
    assert config.topology.ep == 2
    assert config.topology.world_size() == 2


def test_cp_unsupported_model_uses_non_cp_default_topology(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=284 * 1024**3),
    )
    monkeypatch.setenv("BASE_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TRAINER_GPU_IDS", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_INFERENCE_GPU_IDS", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_CP", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_EP", raising=False)
    monkeypatch.setenv("ART_MODEL_SUPPORT_EXTERNAL_VLLM_URL", "http://127.0.0.1:8000")

    config = config_from_env()

    assert config.topology.cp == 1
    assert config.topology.tp == 2
    assert config.topology.ep == 2
    assert config.topology.dp == 1
    assert config.trainer_gpu_ids == [0, 1]
    assert config.inference_gpu_ids == [2, 3]
    assert config.engine_args["tensor_parallel_size"] == 2
    assert config.engine_args["enable_expert_parallel"] is True
    assert config.engine_args["kv_cache_dtype"] == "fp8"
    assert config.engine_args["moe_backend"] == "triton_unfused"
    assert config.megatron_env == {"ART_MEGATRON_STREAMING_WEIGHT_OFFLOAD": "1"}
    assert config.external_vllm_server_url == "http://127.0.0.1:8000"
