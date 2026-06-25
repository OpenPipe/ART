"""Unit tests for dedicated mode config validation and get_model_config integration."""

import tempfile

import pytest

from art.dev.get_model_config import get_model_config
from art.dev.model import InternalModelConfig, LoRAConfig, PeftArgs
from art.dev.validate import is_dedicated_mode, validate_dedicated_config


@pytest.fixture(autouse=True)
def _stub_model_max_seq_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "art.dev.get_model_config.max_seq_length_from_model_config",
        lambda *_args, **_kwargs: 2048,
    )


def test_shared_mode_empty_config():
    config = InternalModelConfig()
    assert is_dedicated_mode(config) is False


def test_shared_mode_with_other_keys():
    config = InternalModelConfig(init_args={"model_name": "test"})  # type: ignore[typeddict-item]
    assert is_dedicated_mode(config) is False


def test_dedicated_mode_detected():
    config = InternalModelConfig(trainer_gpu_ids=[0], inference_gpu_ids=[1])
    assert is_dedicated_mode(config) is True


def test_valid_shared_mode():
    validate_dedicated_config(InternalModelConfig())


def test_valid_dedicated_two_gpus():
    validate_dedicated_config(
        InternalModelConfig(trainer_gpu_ids=[0], inference_gpu_ids=[1])
    )


def test_valid_dedicated_three_gpus():
    validate_dedicated_config(
        InternalModelConfig(trainer_gpu_ids=[0, 1], inference_gpu_ids=[2])
    )


def test_valid_dedicated_four_gpus():
    validate_dedicated_config(
        InternalModelConfig(trainer_gpu_ids=[0, 1, 2], inference_gpu_ids=[3])
    )


def test_only_trainer_gpu_ids():
    with pytest.raises(ValueError, match="must both be set or both unset"):
        validate_dedicated_config(InternalModelConfig(trainer_gpu_ids=[0]))


def test_only_inference_gpu_ids():
    with pytest.raises(ValueError, match="must both be set or both unset"):
        validate_dedicated_config(InternalModelConfig(inference_gpu_ids=[1]))


def test_empty_trainer_gpu_ids():
    with pytest.raises(ValueError, match="trainer_gpu_ids must be non-empty"):
        validate_dedicated_config(
            InternalModelConfig(trainer_gpu_ids=[], inference_gpu_ids=[1])
        )


def test_empty_inference_gpu_ids():
    with pytest.raises(ValueError, match="inference_gpu_ids must be non-empty"):
        validate_dedicated_config(
            InternalModelConfig(trainer_gpu_ids=[0], inference_gpu_ids=[])
        )


def test_overlapping_gpu_ids():
    with pytest.raises(ValueError, match="must not overlap"):
        validate_dedicated_config(
            InternalModelConfig(trainer_gpu_ids=[0, 1], inference_gpu_ids=[1])
        )


def test_multi_gpu_inference():
    with pytest.raises(ValueError, match="Multi-GPU inference not yet supported"):
        validate_dedicated_config(
            InternalModelConfig(trainer_gpu_ids=[0], inference_gpu_ids=[1, 2])
        )


def test_trainer_not_starting_at_zero():
    with pytest.raises(ValueError, match="must start at GPU 0"):
        validate_dedicated_config(
            InternalModelConfig(trainer_gpu_ids=[1], inference_gpu_ids=[0])
        )


def test_trainer_not_contiguous():
    with pytest.raises(ValueError, match="must be contiguous starting from 0"):
        validate_dedicated_config(
            InternalModelConfig(trainer_gpu_ids=[0, 2], inference_gpu_ids=[1])
        )


def test_dedicated_rejects_fast_inference():
    with pytest.raises(ValueError, match="fast_inference is no longer supported"):
        validate_dedicated_config(
            InternalModelConfig(
                trainer_gpu_ids=[0],
                inference_gpu_ids=[1],
                init_args={"fast_inference": True},  # type: ignore[typeddict-item]
            )
        )


def test_dedicated_rejects_enable_sleep_mode():
    with pytest.raises(
        ValueError, match="enable_sleep_mode is incompatible with dedicated"
    ):
        validate_dedicated_config(
            InternalModelConfig(
                trainer_gpu_ids=[0],
                inference_gpu_ids=[1],
                engine_args={"enable_sleep_mode": True},  # type: ignore[typeddict-item]
            )
        )


def test_dedicated_rejects_fast_inference_false():
    """fast_inference config is removed; vLLM always lives in its own runtime."""
    with pytest.raises(ValueError, match="fast_inference is no longer supported"):
        validate_dedicated_config(
            InternalModelConfig(
                trainer_gpu_ids=[0],
                inference_gpu_ids=[1],
                init_args={"fast_inference": False},  # type: ignore[typeddict-item]
            )
        )


def test_get_model_config_shared_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = get_model_config("test-model", tmpdir, None)
        assert "trainer_gpu_ids" not in result
        assert "inference_gpu_ids" not in result
        assert result["engine_args"]["enable_sleep_mode"] is True
        assert "fast_inference" not in result["init_args"]
        assert result["rollout_weights_mode"] == "lora"
        assert result["lora_config"]["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


@pytest.mark.parametrize(
    "base_model",
    ["Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-397B-A17B"],
)
def test_get_model_config_qwen3_5_moe_target_modules(base_model: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = get_model_config(base_model, tmpdir, None)
        assert result["lora_config"]["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "experts",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


def test_get_model_config_preserves_user_lora_config_target_modules():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = get_model_config(
            "Qwen/Qwen3.5-35B-A3B",
            tmpdir,
            InternalModelConfig(),
            lora_config=LoRAConfig(target_modules=["custom_proj"]),
        )
        assert result["lora_config"]["target_modules"] == ["custom_proj"]


def test_get_model_config_rejects_peft_args_with_migration_message():
    with pytest.raises(ValueError, match="r->rank"):
        PeftArgs(r=4)
    with pytest.raises(ValueError, match="TrainableModel\\(lora_config"):
        get_model_config(
            "test-model",
            "",
            InternalModelConfig(peft_args={"r": 4}),  # type: ignore[typeddict-item]
        )


def test_get_model_config_dedicated_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        )
        result = get_model_config("test-model", tmpdir, config)
        assert result["trainer_gpu_ids"] == [0]
        assert result["inference_gpu_ids"] == [1]
        assert result["engine_args"]["enable_sleep_mode"] is False
        assert "fast_inference" not in result["init_args"]
        assert result["rollout_weights_mode"] == "lora"


def test_get_model_config_dedicated_preserves_user_engine_args():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            engine_args={"max_model_len": 4096},  # type: ignore[typeddict-item]
        )
        result = get_model_config("test-model", tmpdir, config)
        assert result["engine_args"]["max_model_len"] == 4096
        # Sleep mode should still be disabled even if user didn't set it
        assert result["engine_args"]["enable_sleep_mode"] is False


def test_get_model_config_preserves_rollout_weights_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            rollout_weights_mode="merged",
        )
        result = get_model_config("test-model", tmpdir, config)
        assert result["rollout_weights_mode"] == "merged"


def test_invalid_rollout_weights_mode():
    with pytest.raises(
        ValueError, match="rollout_weights_mode must be either 'lora' or 'merged'"
    ):
        validate_dedicated_config(
            InternalModelConfig(rollout_weights_mode="bad-mode")  # type: ignore[typeddict-item]
        )


def test_merged_rollout_weights_requires_dedicated_mode():
    with pytest.raises(
        ValueError, match="rollout_weights_mode='merged' requires dedicated mode"
    ):
        validate_dedicated_config(InternalModelConfig(rollout_weights_mode="merged"))


def test_qwen3_5_moe_allows_default_lora_rollout_weights():
    validate_dedicated_config(
        InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            engine_args={"model": "Qwen/Qwen3.5-35B-A3B"},  # type: ignore[typeddict-item]
        )
    )


def test_qwen3_5_moe_allows_merged_rollout_weights():
    validate_dedicated_config(
        InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            rollout_weights_mode="merged",
            engine_args={"model": "Qwen/Qwen3.5-35B-A3B"},  # type: ignore[typeddict-item]
        )
    )


def test_other_qwen3_5_moe_allows_default_lora_rollout_weights():
    validate_dedicated_config(
        InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
            engine_args={"model": "Qwen/Qwen3.5-397B-A17B"},  # type: ignore[typeddict-item]
        )
    )
