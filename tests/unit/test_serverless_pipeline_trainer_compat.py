from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.serverless.backend import ServerlessBackend
from art.types import TrainConfig, TrainSFTConfig


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
        run_name="serverless-pipeline-compat",
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

    setattr(backend, "_train_model", fake_train_model)
    backend._get_step = AsyncMock(return_value=3)  # type: ignore[method-assign]

    with patch.object(model, "_get_wandb_run", return_value=None):
        result = await backend.train(
            model,
            [_make_group()],
            learning_rate=2e-5,
            loss_fn="ppo",
            normalize_advantages=False,
            save_checkpoint=False,
            optimizer_save_interval=7,
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
        "kl_penalty_source": "sample",
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
        run_name="serverless-pipeline-rejects",
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

    with pytest.raises(ValueError, match="kl_penalty_step_lag must be >= 1"):
        await backend.train(model, [_make_group()], kl_penalty_step_lag=0)

    with pytest.raises(ValueError, match="Only one of"):
        await backend.train(
            model,
            [_make_group()],
            kl_penalty_reference_step=0,
            kl_penalty_step_lag=1,
        )


@pytest.mark.asyncio
async def test_serverless_train_model_forwards_experimental_config() -> None:
    backend = _make_backend()
    model = TrainableModel(
        run_name="serverless-config-payload",
        name="serverless-config-payload",
        project="pipeline-tests",
        base_model="test-model",
    )
    model.id = "model-id"

    captured: dict[str, Any] = {}
    client = SimpleNamespace(
        run_id="run-native",
        projected_learner_version=0,
        next_sequence_id=0,
    )

    async def forward_backward(request: Any) -> Any:
        captured["forward_backward"] = request
        client.next_sequence_id = 1
        result = SimpleNamespace(
            packed_input_capture=None,
            packing=SimpleNamespace(group_shapes=(), packed_sequences=1),
            metrics={},
        )
        return SimpleNamespace(
            ref=SimpleNamespace(operation_id="forward-backward"),
            result=AsyncMock(return_value=result),
        )

    async def optim_step(request: Any) -> Any:
        captured["optim_step"] = request
        client.next_sequence_id = 2
        result = SimpleNamespace(
            checkpoint=SimpleNamespace(learner_version=1), metrics={}
        )
        return SimpleNamespace(
            ref=SimpleNamespace(operation_id="optim-step"),
            result=AsyncMock(return_value=result),
        )

    async def save_weights_for_sampler(request: Any) -> Any:
        captured["save_weights_for_sampler"] = request
        return SimpleNamespace(
            ref=SimpleNamespace(operation_id="save-sampler"),
            result=AsyncMock(return_value=SimpleNamespace(lora="lora-ref")),
        )

    client.forward_backward = forward_backward
    client.optim_step = optim_step
    client.save_weights_for_sampler = save_weights_for_sampler
    backend.training_client = AsyncMock(return_value=client)  # type: ignore[method-assign]

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
            "kl_penalty_reference_step": 0,
            "kl_penalty_source": "sample",
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

    forward = captured["forward_backward"]
    payload = forward.loss.values
    assert payload["learning_rate"] == 7e-6
    assert forward.loss.name == "ppo"
    assert forward.loss.normalize_advantages is False
    assert payload["packed_sequence_length"] == 4096
    assert payload["kl_penalty_coef"] == 0.2
    assert payload["kl_penalty_reference_step"] == 0
    assert payload["kl_penalty_source"] == "sample"
    assert payload["kl_ref_adapter_path"] == "/tmp/ref"
    assert payload["allow_training_without_logprobs"] is True
    assert payload["scale_learning_rate_by_reward_std_dev"] is True


@pytest.mark.asyncio
async def test_serverless_train_sft_uses_native_commands() -> None:
    backend = _make_backend()
    model = TrainableModel(
        run_name="serverless-sft-config-payload",
        name="serverless-sft-config-payload",
        project="pipeline-tests",
        base_model="test-model",
    )
    model.id = "model-id"
    model.entity = "entity"
    model.run_id = "canonical-run-id"

    captured: dict[str, Any] = {}
    client = SimpleNamespace(
        run_id="run-native",
        projected_learner_version=4,
        next_sequence_id=0,
    )
    forward_operation = SimpleNamespace(
        ref=SimpleNamespace(operation_id="sft-forward-backward"),
        result=AsyncMock(
            return_value=SimpleNamespace(
                produced_gradient=True,
                packing=SimpleNamespace(trainable_assistant_tokens=3),
                metrics={"loss/train": 0.5},
            )
        ),
        cancel=AsyncMock(),
    )

    async def forward_backward(request: Any) -> Any:
        captured["forward_backward"] = request
        client.next_sequence_id = 1
        return forward_operation

    async def optim_step(request: Any) -> Any:
        captured["optim_step"] = request
        client.next_sequence_id = 2
        return SimpleNamespace(
            ref=SimpleNamespace(operation_id="sft-optim-step"),
            result=AsyncMock(
                return_value=SimpleNamespace(
                    checkpoint=SimpleNamespace(learner_version=5),
                    metrics={"loss/grad_norm": 0.25},
                )
            ),
        )

    async def save_weights_for_sampler(request: Any) -> Any:
        captured["save_weights_for_sampler"] = request
        client.next_sequence_id = 3
        return SimpleNamespace(
            ref=SimpleNamespace(operation_id="sft-save-sampler"),
            result=AsyncMock(return_value=SimpleNamespace(lora="lora-ref")),
        )

    client.forward_backward = forward_backward
    client.optim_step = optim_step
    client.save_weights_for_sampler = save_weights_for_sampler
    backend.training_client = AsyncMock(return_value=client)  # type: ignore[method-assign]

    trajectory = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "answer"},
        ],
    )

    with patch.object(model, "_get_wandb_run", return_value=None):
        rows = [
            row
            async for row in backend._train_sft(
                model,
                [trajectory],
                TrainSFTConfig(
                    learning_rate=[1e-4],
                    batch_size=2,
                    assistant_turns="last",
                ),
                {"metric_logging": {"enabled": True, "target_training_step": 1}},
            )
        ]

    forward = captured["forward_backward"]
    assert forward.run_id == "run-native"
    assert forward.sequence_id == 0
    assert forward.batch.kind == "sft"
    assert forward.batch.trajectories == (trajectory,)
    assert forward.batch.assistant_turns == "last"
    assert forward.loss.name == "cross_entropy"
    assert forward.return_token_logprobs is False
    assert captured["optim_step"].sequence_id == 1
    assert captured["optim_step"].optimizer.learning_rate == 1e-4
    publication = captured["save_weights_for_sampler"]
    assert publication.sequence_id == 2
    assert publication.checkpoint_name == "step-5"
    assert publication.publication.mode == "versioned_lora"
    assert rows == [
        {
            "loss/train": 0.5,
            "loss/grad_norm": 0.25,
            "data/step_num_trajectories": 1.0,
            "data/step_trainable_assistant_tokens": 3.0,
            "data/step_num_gradient_steps": 1.0,
        }
    ]
    assert backend.logs_sft_metrics_remotely() is False
    assert backend._native_steps[model._storage_name()] == 5
    assert backend._native_artifacts[model._storage_name()] == "lora-ref"
    forward_operation.cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_serverless_train_sft_releases_terminal_zero_gradient() -> None:
    backend = _make_backend()
    model = TrainableModel(
        run_name="serverless-sft-zero-gradient",
        name="serverless-sft-zero-gradient",
        project="pipeline-tests",
        base_model="test-model",
    )
    key = model._storage_name()
    backend._native_steps[key] = 7
    backend._native_artifacts[key] = "existing-lora"

    events: list[str] = []

    async def terminal_result() -> Any:
        events.append("result")
        return SimpleNamespace(
            produced_gradient=False,
            packing=SimpleNamespace(trainable_assistant_tokens=0),
            metrics={"data/sft_zero_work": 1.0},
        )

    async def release_contribution() -> None:
        events.append("cancel")

    forward_operation = SimpleNamespace(
        ref=SimpleNamespace(operation_id="sft-zero-forward-backward"),
        result=AsyncMock(side_effect=terminal_result),
        cancel=AsyncMock(side_effect=release_contribution),
    )
    client = SimpleNamespace(
        run_id="run-native",
        projected_learner_version=7,
        next_sequence_id=0,
        forward_backward=AsyncMock(return_value=forward_operation),
        optim_step=AsyncMock(),
        save_weights_for_sampler=AsyncMock(),
    )
    backend.training_client = AsyncMock(return_value=client)  # type: ignore[method-assign]

    rows = [
        row
        async for row in backend._train_sft(
            model,
            [Trajectory()],
            TrainSFTConfig(learning_rate=1e-4, batch_size=1),
            {},
        )
    ]

    assert rows == []
    assert events == ["result", "cancel"]
    forward_operation.cancel.assert_awaited_once_with()
    client.optim_step.assert_not_awaited()
    client.save_weights_for_sampler.assert_not_awaited()
    assert client.projected_learner_version == 7
    assert backend._native_steps[key] == 7
    assert backend._native_artifacts[key] == "existing-lora"


@pytest.mark.asyncio
async def test_serverless_train_forwards_kl_step_lag() -> None:
    backend = _make_backend()
    model = TrainableModel(
        run_name="serverless-kl-step-lag",
        name="serverless-kl-step-lag",
        project="pipeline-tests",
        base_model="test-model",
    )
    model.id = "model-id"

    seen: dict[str, Any] = {}

    async def fake_train_model(
        _model: TrainableModel,
        _groups: list[TrajectoryGroup],
        _config: TrainConfig,
        dev_config: dict[str, Any],
        verbose: bool = False,
    ):
        del verbose
        seen["dev_config"] = dev_config
        yield {}

    setattr(backend, "_train_model", fake_train_model)
    backend._get_step = AsyncMock(return_value=1)  # type: ignore[method-assign]

    with patch.object(model, "_get_wandb_run", return_value=None):
        await backend.train(
            model,
            [_make_group()],
            kl_penalty_coef=0.2,
            kl_penalty_source="sample",
            kl_penalty_step_lag=3,
        )

    assert seen["dev_config"]["kl_penalty_coef"] == 0.2
    assert seen["dev_config"]["kl_penalty_source"] == "sample"
    assert seen["dev_config"]["kl_penalty_step_lag"] == 3
