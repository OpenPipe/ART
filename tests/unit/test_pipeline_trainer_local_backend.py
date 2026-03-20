import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.dev.model import InternalModelConfig
from art.local import LocalBackend
from art.pipeline_trainer.trainer import PipelineTrainer
from art.utils.output_dirs import get_model_dir


def _make_group(rewards: list[float]) -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                initial_policy_version=0,
                messages_and_choices=[
                    {"role": "user", "content": f"prompt-{idx}"},
                    {"role": "assistant", "content": f"answer-{idx}"},
                ],
            )
            for idx, reward in enumerate(rewards)
        ]
    )


def _make_trainer(
    *,
    model: TrainableModel,
    backend: object,
    **kwargs: Any,
) -> PipelineTrainer:
    return PipelineTrainer(
        model=model,
        backend=backend,  # type: ignore[arg-type]
        rollout_fn=lambda *_args, **_kwargs: asyncio.sleep(0),
        scenarios=[],
        config={},
        num_rollout_workers=1,
        min_batch_size=1,
        max_batch_size=1,
        max_steps=1,
        eval_fn=None,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_pipeline_trainer_preserves_backend_train_kwargs(tmp_path: Path) -> None:
    model = TrainableModel(
        name="pipeline-default-backend-kwargs",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))
    loss_fn_config = {"alpha": 0.1}
    adam_params = object()

    trainer = _make_trainer(
        model=model,
        backend=backend,
        learning_rate=2e-5,
        loss_fn="cispo",
        loss_fn_config=loss_fn_config,
        normalize_advantages=True,
        adam_params=adam_params,
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_args.kwargs == {
        "learning_rate": 2e-5,
        "loss_fn": "cispo",
        "loss_fn_config": loss_fn_config,
        "normalize_advantages": True,
        "save_checkpoint": False,
        "adam_params": adam_params,
    }


@pytest.mark.asyncio
async def test_pipeline_trainer_forwards_kl_kwargs_for_generic_backend(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-generic-backend-kl-kwargs",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))

    trainer = _make_trainer(
        model=model,
        backend=backend,
        kl_penalty_coef=0.25,
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_args.kwargs == {
        "learning_rate": 1e-5,
        "loss_fn": "cispo",
        "loss_fn_config": None,
        "normalize_advantages": True,
        "save_checkpoint": False,
        "adam_params": None,
        "kl_penalty_coef": 0.25,
        "kl_penalty_reference_step": 0,
        "kl_penalty_source": "sample",
    }


@pytest.mark.asyncio
async def test_pipeline_trainer_kl_step_lag_floors_at_zero(
    tmp_path: Path,
) -> None:
    """kl_penalty_step_lag floors at step 0 when lag > current_step."""
    model = TrainableModel(
        name="pipeline-kl-step-lag-floor",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=2, metrics={}))

    trainer = _make_trainer(
        model=model,
        backend=backend,
        kl_penalty_coef=0.25,
        kl_penalty_step_lag=5,
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    # Simulate being at step 1
    trainer.state.next_training_step = 1

    await trainer._training_stage()

    # At step 1, lag=5 → reference = max(0, 1-5) = 0
    assert backend.train.await_args.kwargs["kl_penalty_reference_step"] == 0


@pytest.mark.asyncio
async def test_pipeline_trainer_kl_step_lag_computes_reference(
    tmp_path: Path,
) -> None:
    """kl_penalty_step_lag computes reference from current step."""
    model = TrainableModel(
        name="pipeline-kl-step-lag",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=4, metrics={}))

    trainer = _make_trainer(
        model=model,
        backend=backend,
        kl_penalty_coef=0.25,
        kl_penalty_step_lag=2,
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    # Simulate being at step 3
    trainer.state.next_training_step = 3

    await trainer._training_stage()

    # At step 3, lag=2 → reference = max(0, 3-2) = 1
    assert backend.train.await_args.kwargs["kl_penalty_reference_step"] == 1


@pytest.mark.asyncio
async def test_pipeline_trainer_uses_same_train_kwargs_for_local_backend(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-local-backend-kwargs",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        _internal_config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )
    backend = LocalBackend(path=str(tmp_path))
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))  # type: ignore[method-assign]

    trainer = _make_trainer(
        model=model,
        backend=backend,
        learning_rate=3e-5,
        loss_fn="ppo",
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_args.kwargs == {  # type: ignore[attr-defined]
        "learning_rate": 3e-5,
        "loss_fn": "ppo",
        "loss_fn_config": None,
        "normalize_advantages": True,
        "save_checkpoint": False,
        "adam_params": None,
    }


@pytest.mark.asyncio
async def test_local_backend_train_translates_loss_fn(tmp_path: Path) -> None:
    model = TrainableModel(
        name="local-backend-train-translation",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = LocalBackend(path=str(tmp_path))
    seen: dict[str, Any] = {}

    async def fake_train_model(
        _model: TrainableModel,
        _groups: list[TrajectoryGroup],
        config: Any,
        dev_config: dict[str, Any],
        verbose: bool = False,
    ):
        seen["config"] = config
        seen["dev_config"] = dev_config
        seen["verbose"] = verbose
        yield {}

    backend._train_model = fake_train_model  # type: ignore[method-assign]
    backend._get_step = AsyncMock(return_value=1)  # type: ignore[method-assign]
    with patch.object(model, "_get_wandb_run", return_value=None):
        result = await backend.train(
            model,
            [_make_group([1.0])],
            loss_fn="ppo",
            save_checkpoint=False,
        )

    assert result.step == 1
    assert seen["config"].learning_rate == 5e-6
    assert seen["dev_config"]["ppo"] is True


@pytest.mark.asyncio
async def test_local_backend_train_passes_kl_penalty_source(tmp_path: Path) -> None:
    model = TrainableModel(
        name="local-backend-kl-source",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = LocalBackend(path=str(tmp_path))
    seen: dict[str, Any] = {}

    async def fake_train_model(
        _model: TrainableModel,
        _groups: list[TrajectoryGroup],
        config: Any,
        dev_config: dict[str, Any],
        verbose: bool = False,
    ):
        seen["config"] = config
        seen["dev_config"] = dev_config
        seen["verbose"] = verbose
        yield {}

    backend._train_model = fake_train_model  # type: ignore[method-assign]
    backend._get_step = AsyncMock(return_value=1)  # type: ignore[method-assign]
    with patch.object(model, "_get_wandb_run", return_value=None):
        result = await backend.train(
            model,
            [_make_group([1.0])],
            kl_penalty_coef=0.25,
            kl_penalty_source="sample",
            save_checkpoint=False,
        )

    assert result.step == 1
    assert seen["config"].kl_penalty_source == "sample"
    assert seen["dev_config"]["kl_penalty_source"] == "sample"


@pytest.mark.asyncio
async def test_local_backend_async_context_manager_awaits_async_cleanup(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    calls: list[str] = []

    class FakeService:
        async def aclose(self) -> None:
            calls.append("aclose")

    service = FakeService()
    backend._services["test-service"] = cast(Any, service)

    with patch("art.local.backend.close_proxy") as close_proxy:
        async with backend:
            pass

    assert calls == ["aclose"]
    close_proxy.assert_called_once_with(service)


@pytest.mark.parametrize(
    ("trainer_kwargs", "match"),
    [
        ({"loss_fn": "dro"}, "loss_fn='cispo' or loss_fn='ppo'"),
        ({"loss_fn_config": {"clip": 0.2}}, "loss_fn_config=None"),
        ({"normalize_advantages": False}, "normalize_advantages=True"),
        ({"adam_params": object()}, "adam_params=None"),
    ],
)
def test_pipeline_trainer_rejects_unsupported_local_backend_settings(
    tmp_path: Path,
    trainer_kwargs: dict[str, object],
    match: str,
) -> None:
    model = TrainableModel(
        name="pipeline-local-backend-invalid",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        _internal_config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )

    with pytest.raises(ValueError, match=match):
        _make_trainer(
            model=model,
            backend=LocalBackend(path=str(tmp_path)),
            **trainer_kwargs,
        )


def test_pipeline_trainer_rejects_shared_local_backend(tmp_path: Path) -> None:
    model = TrainableModel(
        name="pipeline-local-backend-shared",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )

    with pytest.raises(
        ValueError, match="only supports LocalBackend in dedicated mode"
    ):
        _make_trainer(model=model, backend=LocalBackend(path=str(tmp_path)))


def test_local_backend_inference_name_prefers_served_step_in_dedicated_mode(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="local-backend-served-step",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        _internal_config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )
    backend = LocalBackend(path=str(tmp_path))
    output_dir = Path(get_model_dir(model=model, art_path=str(tmp_path)))
    (output_dir / "checkpoints" / "3").mkdir(parents=True)
    backend._services[model.name] = cast(Any, SimpleNamespace(_latest_step=2))

    assert backend._model_inference_name(model) == f"{model.name}@2"
    assert backend._model_inference_name(model, step=3) == f"{model.name}@3"
