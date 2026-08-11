from __future__ import annotations

from types import SimpleNamespace

import torch

from ..model_support.workflow_resources import (
    HandlerWorkflowResources,
    MegatronWorkflowResources,
    MegatronWorkflowTopology,
    VllmWorkflowResources,
    WorkflowStageResources,
)
from . import output_parity
from .output_parity import config_from_env
from .real_path import (
    RealPathConfig,
    _cuda_visible_devices_for_slots,
    _real_path_max_model_len,
)


class _PromptLengthTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        token_count = int(messages[0]["content"])
        return {
            "input_ids": [0] * token_count,
            "attention_mask": [1] * token_count,
        }


def test_real_path_max_model_len_uses_rendered_prompt_length() -> None:
    config = RealPathConfig()
    config.output_parity.packed.sequence_length = 2432
    config.max_completion_tokens = 16

    assert (
        _real_path_max_model_len(
            config,
            tokenizer=_PromptLengthTokenizer(),
            prompts=["2425"],
            chat_template_kwargs={},
        )
        == 2441
    )


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
    assert config.topology.ep == 4
    assert config.topology.dp == 2
    assert config.trainer_gpu_ids == [0, 1, 2, 3]
    assert config.inference_gpu_ids == [2, 3]
    assert config.engine_args["tensor_parallel_size"] == 2
    assert config.engine_args["enable_expert_parallel"] is True
    assert config.engine_args["kv_cache_dtype"] == "fp8"
    assert config.engine_args["moe_backend"] == "triton"
    assert config.streaming_weight_offload is True
    assert config.megatron_env == {}
    assert config.external_vllm_server_url == "http://127.0.0.1:8000"


def test_unconfigured_gpu_defaults_remain_controller_visible_slots(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TRAINER_GPU_IDS", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_INFERENCE_GPU_IDS", raising=False)
    monkeypatch.setattr(
        output_parity,
        "handler_workflow_resources_for_base_model",
        lambda base_model, *, allow_unvalidated_arch=False: None,
    )
    monkeypatch.setattr(output_parity, "model_support_is_moe", lambda *_, **__: True)
    monkeypatch.setattr(
        output_parity,
        "model_supports_context_parallel",
        lambda *_, **__: True,
    )

    config = config_from_env()

    assert config.trainer_gpu_ids == [0, 1]
    assert config.inference_gpu_ids == [2, 3]


def test_explicit_gpu_ids_are_not_reinterpreted_by_config(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_TRAINER_GPU_IDS", "2,3")
    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_INFERENCE_GPU_IDS", "0,1")
    monkeypatch.setattr(
        output_parity,
        "handler_workflow_resources_for_base_model",
        lambda base_model, *, allow_unvalidated_arch=False: None,
    )
    monkeypatch.setattr(output_parity, "model_support_is_moe", lambda *_, **__: True)
    monkeypatch.setattr(
        output_parity,
        "model_supports_context_parallel",
        lambda *_, **__: True,
    )

    config = config_from_env()

    assert config.trainer_gpu_ids == [2, 3]
    assert config.inference_gpu_ids == [0, 1]


def test_resource_gpu_slots_remain_logical_until_runtime_compilation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_TRAINER_GPU_IDS", raising=False)
    monkeypatch.delenv("ART_TRAIN_INF_MISMATCH_INFERENCE_GPU_IDS", raising=False)
    resources = HandlerWorkflowResources(
        train_inf_mismatch=WorkflowStageResources(
            required_world_size=4,
            megatron=MegatronWorkflowResources(
                gpu_ids=[0, 1], topology=MegatronWorkflowTopology()
            ),
            vllm=VllmWorkflowResources(gpu_ids=[2, 3], tensor_parallel_size=2),
        )
    )
    monkeypatch.setattr(
        output_parity,
        "handler_workflow_resources_for_base_model",
        lambda base_model, *, allow_unvalidated_arch=False: resources,
    )
    monkeypatch.setattr(output_parity, "model_support_is_moe", lambda *_, **__: True)
    monkeypatch.setattr(
        output_parity,
        "model_supports_context_parallel",
        lambda *_, **__: True,
    )

    config = config_from_env()

    assert config.trainer_gpu_ids == [0, 1]
    assert config.inference_gpu_ids == [2, 3]


def test_runtime_compiles_logical_gpu_slots_through_outer_mask(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert _cuda_visible_devices_for_slots([0, 1]) == "4,5"
    assert _cuda_visible_devices_for_slots([2, 3]) == "6,7"
