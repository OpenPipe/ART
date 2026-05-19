import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.dev.model import InternalModelConfig
from art.local import LocalBackend
from art.megatron import MegatronBackend
from art.megatron.train import load_adapter_into_model
from art.pipeline_trainer import (
    CHECKPOINT_CREATED_AT_METRIC,
    CHECKPOINT_EVAL_COMPLETED_METRIC,
    CheckpointRetentionContext,
)
from art.pipeline_trainer.trainer import PipelineTrainer
from art.preprocessing.tokenize import TokenizedResult
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
async def test_pipeline_trainer_forwards_packed_sequence_length_when_set(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-packed-sequence-length",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MagicMock()
    backend.train = AsyncMock(return_value=SimpleNamespace(step=1, metrics={}))

    trainer = _make_trainer(
        model=model,
        backend=backend,
        packed_sequence_length=4096,
    )
    trainer._output_queue = asyncio.Queue()
    await trainer._output_queue.put(_make_group([0.0, 1.0]))
    await trainer._output_queue.put(None)

    await trainer._training_stage()

    assert backend.train.await_args.kwargs["packed_sequence_length"] == 4096


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
            packed_sequence_length=2048,
            save_checkpoint=False,
        )

    assert result.step == 1
    assert seen["config"].learning_rate == 5e-6
    assert seen["dev_config"]["ppo"] is True
    assert seen["dev_config"]["packed_sequence_length"] == 2048


@pytest.mark.asyncio
async def test_local_backend_train_maps_normalize_advantages_to_scale_rewards(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="local-backend-normalize-advantages",
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
        seen["dev_config"] = dev_config
        yield {}

    backend._train_model = fake_train_model  # type: ignore[method-assign]
    backend._get_step = AsyncMock(return_value=1)  # type: ignore[method-assign]
    with patch.object(model, "_get_wandb_run", return_value=None):
        await backend.train(
            model,
            [_make_group([0.0, 1.0])],
            normalize_advantages=False,
            save_checkpoint=False,
        )

    assert seen["dev_config"]["scale_rewards"] is False


@pytest.mark.asyncio
async def test_pipeline_trainer_checkpoint_retention_only_passes_unprotected_steps(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-checkpoint-retention",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    checkpoint_dir = Path(model._get_output_dir()) / "checkpoints"
    for step in range(6):
        (checkpoint_dir / f"{step:04d}").mkdir(parents=True)
    history_path = Path(model._get_output_dir()) / "history.jsonl"
    history_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"step": 2, "val/reward": 1.0},
                {"step": 2, "val/reward": 3.0},
                {
                    "step": 2,
                    CHECKPOINT_CREATED_AT_METRIC: 123.0,
                    CHECKPOINT_EVAL_COMPLETED_METRIC: 1.0,
                },
                {"step": 3, "val/reward": 10.0},
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    backend = MagicMock()
    backend._delete_checkpoint_files = AsyncMock()
    contexts: list[CheckpointRetentionContext] = []

    def strategy(context: CheckpointRetentionContext) -> set[int]:
        contexts.append(context)
        return {1, 4, 99}

    trainer = _make_trainer(
        model=model,
        backend=backend,
        checkpoint_retention_strategy=strategy,
    )
    trainer.state.completed_eval_steps = {2, 3}
    trainer._checkpoint_lease_counts[3] = 1
    trainer._scheduled_eval_steps.add(4)

    await trainer._run_checkpoint_retention(5)

    assert [checkpoint.step for checkpoint in contexts[0].checkpoints] == [0, 1, 2]
    step_two = contexts[0].checkpoints[2]
    assert step_two.is_eval_step is True
    assert step_two.created_at == datetime.fromtimestamp(123.0, timezone.utc)
    assert step_two.metrics["val/reward"] == 2.0
    backend._delete_checkpoint_files.assert_awaited_once_with(  # type: ignore[attr-defined]
        model,
        [1, 3, 4, 5],
    )


@pytest.mark.asyncio
async def test_pipeline_trainer_checkpoint_retention_honors_interval(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-checkpoint-retention-interval",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    checkpoint_dir = Path(model._get_output_dir()) / "checkpoints"
    for step in range(3):
        (checkpoint_dir / f"{step:04d}").mkdir(parents=True)

    backend = MagicMock()
    backend._delete_checkpoint_files = AsyncMock()

    trainer = _make_trainer(
        model=model,
        backend=backend,
        checkpoint_retention_strategy=lambda _context: {0, 1, 2},
        checkpoint_retention_interval=5,
    )

    await trainer._run_checkpoint_retention(4)

    backend._delete_checkpoint_files.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_pipeline_trainer_logs_checkpoint_retention_metadata(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="pipeline-checkpoint-retention-metadata",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        report_metrics=[],
    )
    checkpoint_path = Path(model._get_output_dir()) / "checkpoints" / "0001"
    checkpoint_path.mkdir(parents=True)
    trainer = _make_trainer(model=model, backend=MagicMock())

    await trainer._log_checkpoint_saved(
        SimpleNamespace(step=1, checkpoint_path=str(checkpoint_path))
    )
    await trainer._log_checkpoint_eval_completed(1)

    rows = [
        json.loads(line)
        for line in (Path(model._get_output_dir()) / "history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["checkpoint/saved"] == 1.0
    assert rows[0][CHECKPOINT_CREATED_AT_METRIC] > 0.0
    assert rows[1][CHECKPOINT_EVAL_COMPLETED_METRIC] == 1.0


def _make_tokenized_result(
    trajectory: Trajectory,
    token_ids: list[int],
) -> TokenizedResult:
    tokenizer = cast(
        PreTrainedTokenizerBase,
        SimpleNamespace(eos_token_id=0, decode=lambda token_id: str(token_id)),
    )
    return TokenizedResult(
        advantage=1.0,
        chat="",
        token_ids=token_ids,
        input_pos=list(range(len(token_ids))),
        assistant_mask=[0] * (len(token_ids) - 1) + [1],
        logprobs=[float("nan")] * (len(token_ids) - 1) + [-0.1],
        pixel_values=None,
        image_grid_thw=None,
        trajectory=trajectory,
        choice_offsets=[],
        extra_logprobs={},
        _tokenizer=tokenizer,
        weight=1.0,
        prompt_id=123,
        prompt_length=1,
    )


def test_local_backend_get_packed_tensors_warns_and_drops_overlong_results(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    model = TrainableModel(
        name="local-backend-packed-sequence-length",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    short_trajectory = Trajectory(
        reward=1.0,
        initial_policy_version=0,
        messages_and_choices=[
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    long_trajectory = Trajectory(
        reward=1.0,
        initial_policy_version=0,
        messages_and_choices=[
            {"role": "user", "content": "long"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    short_result = _make_tokenized_result(short_trajectory, [1, 2, 3, 4])
    long_result = _make_tokenized_result(long_trajectory, list(range(10)))

    with (
        patch(
            "art.local.backend.AutoTokenizer.from_pretrained",
            return_value=short_result._tokenizer,
        ),
        patch(
            "art.local.backend.AutoImageProcessor.from_pretrained", return_value=None
        ),
        patch(
            "art.local.backend.tokenize_trajectory_groups",
            return_value=iter([short_result, long_result]),
        ),
        pytest.warns(UserWarning, match="Dropping 1 tokenized results"),
    ):
        packed_tensors = backend._get_packed_tensors(
            model,
            [_make_group([0.0, 1.0])],
            advantage_balance=0.0,
            allow_training_without_logprobs=False,
            scale_rewards=True,
            plot_tensors=False,
            packed_sequence_length=4,
            logprob_calculation_chunk_size=2,
        )

    assert packed_tensors is not None
    assert packed_tensors["tokens"].shape == (1, 4)


@pytest.mark.asyncio
async def test_megatron_backend_train_requires_packed_sequence_length(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="megatron-backend-packed-sequence-length",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    backend = MegatronBackend(path=str(tmp_path))

    with patch.object(model, "_get_wandb_run", return_value=None):
        with pytest.raises(
            ValueError, match="MegatronBackend\\.train requires packed_sequence_length"
        ):
            await backend.train(
                model,
                [_make_group([1.0])],
                save_checkpoint=False,
            )


def test_load_adapter_into_model_reloads_optimizer_when_provided() -> None:
    class FakeModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loaded_adapter: dict[str, torch.Tensor] | None = None

        def load_lora(self, adapter_model: dict[str, torch.Tensor]) -> None:
            self.loaded_adapter = adapter_model

    class FakeOptimizer:
        def __init__(self) -> None:
            self.reload_calls = 0

        def reload_model_params(self) -> None:
            self.reload_calls += 1

    module = FakeModule()
    optimizer = FakeOptimizer()
    adapter_model = {"weight": torch.tensor([1.0])}

    load_adapter_into_model([module], adapter_model, optimizer)

    assert module.loaded_adapter is adapter_model
    assert optimizer.reload_calls == 1


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


@pytest.mark.asyncio
async def test_local_backend_adapter_lease_pins_inference_name_and_prune(
    tmp_path: Path,
) -> None:
    model = TrainableModel(
        name="local-backend-adapter-lease",
        project="pipeline-tests",
        base_model="test-model",
        base_path=str(tmp_path),
        _internal_config=InternalModelConfig(
            trainer_gpu_ids=[0],
            inference_gpu_ids=[1],
        ),
    )
    backend = LocalBackend(path=str(tmp_path))
    service = SimpleNamespace(
        _latest_step=5,
        prune_loaded_adapters=AsyncMock(),
    )
    backend._services[model.name] = cast(Any, service)

    async with backend.adapter_lease(model, 3):
        assert backend._model_inference_name(model) == f"{model.name}@3"
        await backend.prune_model_adapters(model, retain_steps={4, 5})

    assert backend._model_inference_name(model) == f"{model.name}@5"
    service.prune_loaded_adapters.assert_awaited_once_with(retain_steps={3, 4, 5})
