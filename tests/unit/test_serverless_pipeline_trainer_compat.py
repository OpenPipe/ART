from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.serverless.backend import ServerlessBackend
from art.types import TrainConfig


def _make_group() -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=1.0,
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "answer"},
                ],
            )
        ]
    )


def _make_backend() -> ServerlessBackend:
    with patch("art.serverless.backend.Client") as client_cls:
        client = MagicMock()
        client.base_url = "http://serverless.test/v1"
        client_cls.return_value = client
        return ServerlessBackend(api_key="test-key")


@pytest.mark.asyncio
async def test_serverless_train_accepts_pipeline_trainer_kwargs() -> None:
    backend = _make_backend()
    model = TrainableModel(
        name="serverless-pipeline-compat",
        project="pipeline-tests",
        base_model="test-model",
    )
    model.id = "model-id"
    model.entity = "entity"

    seen: dict[str, Any] = {}

    async def fake_train_model(
        _model: TrainableModel,
        _groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dict[str, Any],
        verbose: bool = False,
    ):
        seen["config"] = config
        seen["dev_config"] = dev_config
        seen["verbose"] = verbose
        yield {"loss": 0.25}

    backend._train_model = fake_train_model  # type: ignore[method-assign]
    backend._get_step = AsyncMock(return_value=3)  # type: ignore[method-assign]

    with patch.object(model, "_get_wandb_run", return_value=None):
        result = await backend.train(
            model,
            [_make_group()],
            learning_rate=2e-5,
            loss_fn="ppo",
            normalize_advantages=False,
            save_checkpoint=False,
            packed_sequence_length=4096,
            kl_penalty_coef=0.1,
            kl_ref_adapter_path="/tmp/ref-adapter",
            allow_training_without_logprobs=True,
            plot_tensors=True,
            truncated_importance_sampling=2.0,
            scale_learning_rate_by_reward_std_dev=True,
            logprob_calculation_chunk_size=512,
            num_trajectories_learning_rate_multiplier_power=0.5,
            verbose=True,
        )

    assert result.step == 3
    assert (
        result.artifact_name == "entity/pipeline-tests/serverless-pipeline-compat:step3"
    )
    assert seen["config"].learning_rate == 2e-5
    assert seen["config"].kl_penalty_coef == 0.1
    assert seen["verbose"] is True
    assert seen["dev_config"] == {
        "advantage_balance": 0.0,
        "allow_training_without_logprobs": True,
        "importance_sampling_level": "token",
        "kl_penalty_coef": 0.1,
        "kl_ref_adapter_path": "/tmp/ref-adapter",
        "logprob_calculation_chunk_size": 512,
        "mask_prob_ratio": False,
        "num_trajectories_learning_rate_multiplier_power": 0.5,
        "packed_sequence_length": 4096,
        "plot_tensors": True,
        "ppo": True,
        "precalculate_logprobs": False,
        "scale_learning_rate_by_reward_std_dev": True,
        "scale_rewards": False,
        "truncated_importance_sampling": 2.0,
    }


@pytest.mark.asyncio
async def test_serverless_train_rejects_unsupported_pipeline_kwargs() -> None:
    backend = _make_backend()
    model = TrainableModel(
        name="serverless-pipeline-rejects",
        project="pipeline-tests",
        base_model="test-model",
    )

    with pytest.raises(ValueError, match="loss_fn_config=None"):
        await backend.train(model, [_make_group()], loss_fn_config={"clip": 0.2})

    with pytest.raises(ValueError, match="adam_params=None"):
        await backend.train(model, [_make_group()], adam_params=object())

    with pytest.raises(ValueError, match="conflicting loss_fn and ppo"):
        await backend.train(model, [_make_group()], loss_fn="ppo", ppo=False)


@pytest.mark.asyncio
async def test_serverless_train_model_forwards_experimental_config() -> None:
    backend = _make_backend()
    model = TrainableModel(
        name="serverless-config-payload",
        project="pipeline-tests",
        base_model="test-model",
    )
    model.id = "model-id"

    captured: dict[str, Any] = {}
    backend._client.training_jobs.create = AsyncMock(  # type: ignore[attr-defined]
        side_effect=lambda **kwargs: (
            captured.update(kwargs) or SimpleNamespace(id="training-job-id")
        )
    )

    async def events_list(**_kwargs: Any):
        yield SimpleNamespace(id="event-id", type="training_ended", data={})

    backend._client.training_jobs.events.list = events_list  # type: ignore[attr-defined]

    async def no_sleep(_seconds: float) -> None:
        return None

    with patch("art.serverless.backend.asyncio.sleep", no_sleep):
        async for _ in backend._train_model(
            model,
            [_make_group()],
            TrainConfig(learning_rate=7e-6, kl_penalty_coef=0.2),
            {
                "advantage_balance": 0.3,
                "allow_training_without_logprobs": True,
                "epsilon": 0.1,
                "epsilon_high": 0.2,
                "importance_sampling_level": "sequence",
                "kimi_k2_tau": 0.4,
                "kl_penalty_coef": 0.2,
                "kl_ref_adapter_path": "/tmp/ref",
                "logprob_calculation_chunk_size": 512,
                "mask_prob_ratio": True,
                "max_negative_advantage_importance_sampling_weight": 3.0,
                "num_trajectories_learning_rate_multiplier_power": 0.5,
                "packed_sequence_length": 4096,
                "plot_tensors": True,
                "ppo": True,
                "precalculate_logprobs": True,
                "scale_learning_rate_by_reward_std_dev": True,
                "scale_rewards": False,
                "truncated_importance_sampling": 2.0,
            },
        ):
            pass

    payload = captured["experimental_config"]
    assert payload["learning_rate"] == 7e-6
    assert payload["loss_fn"] == "ppo"
    assert payload["normalize_advantages"] is False
    assert payload["packed_sequence_length"] == 4096
    assert payload["kl_penalty_coef"] == 0.2
    assert payload["kl_ref_adapter_path"] == "/tmp/ref"
    assert payload["allow_training_without_logprobs"] is True
    assert payload["scale_learning_rate_by_reward_std_dev"] is True
