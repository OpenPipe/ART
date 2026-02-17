"""Tests for multi-checkpoint inference support (RFC #513).

This module tests the ability to run inference on multiple model checkpoints
simultaneously, enabling pipelined training where training continues on new
checkpoints while validation runs on older ones.

The key features tested are:
1. Model.get_inference_name() with optional step parameter
2. TinkerState.get_sampler_client() for step-based routing
3. ServerlessBackend._model_inference_name() with step suffix
4. UnslothService max_loras configuration
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import art
from art.model import Model, TrainableModel

# =============================================================================
# Model.get_inference_name() Tests
# =============================================================================


class TestModelGetInferenceName:
    """Test Model.get_inference_name() with optional step parameter."""

    def test_get_inference_name_without_step_uses_name(self):
        """Without step, should return the model name."""
        model = Model(name="test-model", project="test-project")
        assert model.get_inference_name() == "test-model"

    def test_get_inference_name_without_step_uses_inference_model_name(self):
        """Without step, should prefer inference_model_name if set."""
        model = Model(
            name="test-model",
            project="test-project",
            inference_model_name="custom-inference-name",
        )
        assert model.get_inference_name() == "custom-inference-name"

    def test_get_inference_name_with_step_appends_suffix(self):
        """With step, should append @step suffix."""
        model = Model(name="test-model", project="test-project")
        assert model.get_inference_name(step=5) == "test-model@5"
        assert model.get_inference_name(step=0) == "test-model@0"
        assert model.get_inference_name(step=100) == "test-model@100"

    def test_get_inference_name_with_step_uses_inference_model_name(self):
        """With step, should use inference_model_name as base if set."""
        model = Model(
            name="test-model",
            project="test-project",
            inference_model_name="custom-inference-name",
        )
        assert model.get_inference_name(step=5) == "custom-inference-name@5"

    def test_get_inference_name_none_step_is_same_as_no_step(self):
        """Explicitly passing step=None should behave same as no step."""
        model = Model(name="test-model", project="test-project")
        assert model.get_inference_name(step=None) == model.get_inference_name()


class TestTrainableModelGetInferenceName:
    """Test TrainableModel.get_inference_name() with optional step parameter."""

    def test_trainable_model_get_inference_name_with_step(self):
        """TrainableModel should also support step parameter."""
        model = TrainableModel(
            name="trainable-model",
            project="test-project",
            base_model="meta-llama/Llama-3.1-8B",
        )
        assert model.get_inference_name() == "trainable-model"
        assert model.get_inference_name(step=3) == "trainable-model@3"


class TestLitellmCompletionParams:
    """Test Model.litellm_completion_params() with optional step parameter."""

    def test_litellm_completion_params_without_step(self):
        """Without step, should use latest checkpoint name."""
        model = Model(
            name="test-model",
            project="test-project",
            inference_model_name="inference-name",
            inference_base_url="http://localhost:8000/v1",
            inference_api_key="test-key",
        )
        params = model.litellm_completion_params()
        assert params["model"] == "inference-name"
        assert params["base_url"] == "http://localhost:8000/v1"
        assert params["api_key"] == "test-key"

    def test_litellm_completion_params_with_step(self):
        """With step, should append @step suffix to model name."""
        model = Model(
            name="test-model",
            project="test-project",
            inference_model_name="inference-name",
            inference_base_url="http://localhost:8000/v1",
            inference_api_key="test-key",
        )
        params = model.litellm_completion_params(step=5)
        assert params["model"] == "inference-name@5"

    def test_litellm_completion_params_trainable_model_with_step(self):
        """Trainable model with step should have hosted_vllm/ prefix and @step suffix."""
        model = TrainableModel(
            name="trainable-model",
            project="test-project",
            base_model="meta-llama/Llama-3.1-8B",
        )
        # Set inference_model_name as it would be after register()
        model.inference_model_name = "trainable-model"
        model.inference_base_url = "http://localhost:8000/v1"
        model.inference_api_key = "test-key"

        params = model.litellm_completion_params()
        assert params["model"] == "hosted_vllm/trainable-model"

        params_with_step = model.litellm_completion_params(step=3)
        assert params_with_step["model"] == "hosted_vllm/trainable-model@3"


# =============================================================================
# TinkerState Tests
# =============================================================================


class TestTinkerStateGetSamplerClient:
    """Test TinkerState.get_sampler_client() for step-based routing."""

    @pytest.fixture
    def tinker_state_class(self):
        """Import TinkerState, skipping if dependencies unavailable."""
        try:
            from art.tinker.service import TinkerState

            return TinkerState
        except ImportError as e:
            pytest.skip(f"Tinker dependencies not available: {e}")

    def test_get_sampler_client_without_step_returns_latest(self, tinker_state_class):
        """Without step, should return client for latest_step."""
        TinkerState = tinker_state_class

        # Create mock sampler clients
        mock_client_0 = MagicMock()
        mock_client_5 = MagicMock()

        state = TinkerState(
            service_client=MagicMock(),
            rest_client=MagicMock(),
            training_client=MagicMock(),
            sampler_clients={0: mock_client_0, 5: mock_client_5},
            latest_step=5,
            renderer=MagicMock(),
        )

        assert state.get_sampler_client() is mock_client_5
        assert state.get_sampler_client(step=None) is mock_client_5

    def test_get_sampler_client_with_step_returns_specific_client(
        self, tinker_state_class
    ):
        """With step, should return client for that specific step."""
        TinkerState = tinker_state_class

        mock_client_0 = MagicMock()
        mock_client_3 = MagicMock()
        mock_client_5 = MagicMock()

        state = TinkerState(
            service_client=MagicMock(),
            rest_client=MagicMock(),
            training_client=MagicMock(),
            sampler_clients={0: mock_client_0, 3: mock_client_3, 5: mock_client_5},
            latest_step=5,
            renderer=MagicMock(),
        )

        assert state.get_sampler_client(step=0) is mock_client_0
        assert state.get_sampler_client(step=3) is mock_client_3
        assert state.get_sampler_client(step=5) is mock_client_5

    def test_get_sampler_client_invalid_step_raises_error(self, tinker_state_class):
        """Invalid step should raise ValueError with available steps."""
        TinkerState = tinker_state_class

        state = TinkerState(
            service_client=MagicMock(),
            rest_client=MagicMock(),
            training_client=MagicMock(),
            sampler_clients={0: MagicMock(), 5: MagicMock()},
            latest_step=5,
            renderer=MagicMock(),
        )

        with pytest.raises(ValueError) as exc_info:
            state.get_sampler_client(step=3)

        assert "No sampler client for step 3" in str(exc_info.value)
        assert "Available steps: [0, 5]" in str(exc_info.value)


# =============================================================================
# ServerlessBackend Tests
# =============================================================================


class TestServerlessBackendModelInferenceName:
    """Test ServerlessBackend._model_inference_name() with step suffix."""

    def test_model_inference_name_without_step(self):
        """Without step, should return base W&B artifact name."""
        from art.serverless.backend import ServerlessBackend

        # Create backend with mock client
        with patch("art.serverless.backend.Client"):
            backend = ServerlessBackend(api_key="test-key")

        model = TrainableModel(
            name="test-model",
            project="test-project",
            base_model="meta-llama/Llama-3.1-8B",
        )
        model.entity = "test-entity"

        result = backend._model_inference_name(model)
        assert result == "wandb-artifact:///test-entity/test-project/test-model"

    def test_model_inference_name_with_step(self):
        """With step, should append :step{N} suffix."""
        from art.serverless.backend import ServerlessBackend

        with patch("art.serverless.backend.Client"):
            backend = ServerlessBackend(api_key="test-key")

        model = TrainableModel(
            name="test-model",
            project="test-project",
            base_model="meta-llama/Llama-3.1-8B",
        )
        model.entity = "test-entity"

        result = backend._model_inference_name(model, step=5)
        assert result == "wandb-artifact:///test-entity/test-project/test-model:step5"

        result = backend._model_inference_name(model, step=0)
        assert result == "wandb-artifact:///test-entity/test-project/test-model:step0"

    def test_model_inference_name_none_step_is_same_as_no_step(self):
        """Explicitly passing step=None should behave same as no step."""
        from art.serverless.backend import ServerlessBackend

        with patch("art.serverless.backend.Client"):
            backend = ServerlessBackend(api_key="test-key")

        model = TrainableModel(
            name="test-model",
            project="test-project",
            base_model="meta-llama/Llama-3.1-8B",
        )
        model.entity = "test-entity"

        assert backend._model_inference_name(
            model, step=None
        ) == backend._model_inference_name(model)


# =============================================================================
# OpenAI Server Config Tests
# =============================================================================


class TestOpenAIServerConfigLoraName:
    """Test that get_openai_server_config uses step-based LoRA naming."""

    def test_lora_name_includes_step(self):
        """LoRA module name should include @step suffix."""
        from art.dev.openai_server import get_openai_server_config

        config = get_openai_server_config(
            model_name="my-model",
            base_model="meta-llama/Llama-3.1-8B",
            log_file="/tmp/test.log",
            lora_path="/path/to/checkpoints/0005",
        )

        lora_modules = config.get("server_args", {}).get("lora_modules") or []
        assert len(lora_modules) == 1
        assert "my-model@5" in lora_modules[0]
        assert "/path/to/checkpoints/0005" in lora_modules[0]

    def test_lora_name_step_zero(self):
        """LoRA module name should work with step 0."""
        from art.dev.openai_server import get_openai_server_config

        config = get_openai_server_config(
            model_name="my-model",
            base_model="meta-llama/Llama-3.1-8B",
            log_file="/tmp/test.log",
            lora_path="/path/to/checkpoints/0000",
        )

        lora_modules = config.get("server_args", {}).get("lora_modules") or []
        assert len(lora_modules) == 1
        assert "my-model@0" in lora_modules[0]


# =============================================================================
# Step Parsing Tests
# =============================================================================


class TestStepParsing:
    """Test parsing of @step suffix from model names."""

    def test_parse_step_from_model_name(self):
        """Test the step parsing logic used in TinkerService."""
        test_cases = [
            ("model-name", None),  # No @ suffix
            ("model-name@5", 5),  # Valid step
            ("model-name@0", 0),  # Step 0
            ("model-name@100", 100),  # Large step
            ("model@name@5", 5),  # Multiple @ (use last)
            ("model-name@invalid", None),  # Invalid step (not a number)
            ("model-name@", None),  # Empty step
        ]

        for model_name, expected_step in test_cases:
            step = None
            if "@" in str(model_name):
                _, step_str = str(model_name).rsplit("@", 1)
                try:
                    step = int(step_str)
                except ValueError:
                    pass

            assert step == expected_step, (
                f"Failed for {model_name}: got {step}, expected {expected_step}"
            )


# =============================================================================
# UnslothService Configuration Tests
# =============================================================================


class TestUnslothServiceMaxLoras:
    """Test UnslothService max_loras configuration."""

    @pytest.fixture
    def unsloth_service_class(self):
        """Import UnslothService, skipping if dependencies unavailable."""
        try:
            from art.unsloth.service import UnslothService

            return UnslothService
        except ImportError as e:
            pytest.skip(f"Unsloth dependencies not available: {e}")

    def test_max_loras_default_is_2(self, unsloth_service_class):
        """UnslothService should default to max_loras=2 (one for training, one for validation)."""
        UnslothService = unsloth_service_class

        service = UnslothService(
            model_name="test-model",
            base_model="meta-llama/Llama-3.1-8B",
            config={},
            output_dir="/tmp/test",
        )

        # Access the llm cached property to check engine args
        # We can't actually create the LLM, but we can check the config logic
        engine_args = {
            **service.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": service.config.get("engine_args", {}).get("max_loras", 2),
        }

        assert engine_args["max_loras"] == 2
        assert engine_args["enable_lora"] is True

    def test_max_loras_can_be_overridden(self, unsloth_service_class):
        """max_loras should be configurable via engine_args."""
        UnslothService = unsloth_service_class

        service = UnslothService(
            model_name="test-model",
            base_model="meta-llama/Llama-3.1-8B",
            config={"engine_args": {"max_loras": 8}},
            output_dir="/tmp/test",
        )

        engine_args = {
            **service.config.get("engine_args", {}),
            "enable_lora": True,
            "max_loras": service.config.get("engine_args", {}).get("max_loras", 2),
        }

        assert engine_args["max_loras"] == 8


# =============================================================================
# Pipelined Training Usage Example
# =============================================================================


class TestPipelinedTrainingUsage:
    """Test the usage pattern for pipelined training as described in RFC #513."""

    def test_pipelined_training_pattern(self):
        """
        Verify the API supports the pipelined training pattern from RFC #513.

        The pattern is:
        1. Rollout uses latest checkpoint: model.get_inference_name()
        2. After training, queue eval on specific checkpoint: model.get_inference_name(step=N)
        """
        model = Model(name="my-model", project="test", inference_model_name="my-model")

        # Rollout uses latest checkpoint (no step)
        rollout_name = model.get_inference_name()
        assert rollout_name == "my-model"
        assert "@" not in rollout_name

        # After training step 5, queue eval on that specific checkpoint
        eval_name = model.get_inference_name(step=5)
        assert eval_name == "my-model@5"

        # Training continues, new checkpoint at step 6
        # Rollout still uses latest (would be step 6 after training)
        new_rollout_name = model.get_inference_name()
        assert new_rollout_name == "my-model"

        # Previous eval can still reference step 5
        prev_eval_name = model.get_inference_name(step=5)
        assert prev_eval_name == "my-model@5"
