import asyncio
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from openai.types.chat.chat_completion import Choice
import pytest
import torch
from transformers import PreTrainedTokenizerBase

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.dev.model import InternalModelConfig
from art.local import LocalBackend
from art.megatron import MegatronBackend
from art.metrics_taxonomy import TRAIN_GRADIENT_STEPS_KEY, summarize_trajectory_groups
from art.pipeline_trainer.trainer import PipelineTrainer
from art.preprocessing.pack import packed_tensors_from_tokenized_results
from art.preprocessing.tokenize import TokenizedResult, tokenize_trajectory_groups
from art.tinker.backend import TinkerBackend
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir


class _Tokenizer:
    name_or_path = "test-model"
    eos_token_id = 0


def _group(rewards: list[float]) -> TrajectoryGroup:
    return TrajectoryGroup(
        [
            Trajectory(
                reward=reward,
                messages_and_choices=[
                    {"role": "user", "content": "prompt"},
                    Choice.model_validate(
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "answer"},
                            "prompt_token_ids": [10, 11],
                            "token_ids": [20 + index, 30 + index],
                            "logprobs": {
                                "content": [
                                    {
                                        "token": token,
                                        "logprob": -0.1,
                                        "bytes": [],
                                        "top_logprobs": [],
                                    }
                                    for token in ("a", "b")
                                ]
                            },
                        }
                    ),
                ],
            )
            for index, reward in enumerate(rewards)
        ]
    )


def _tokenize(
    groups: list[TrajectoryGroup], *, scale_rewards: bool, drop: bool = True
) -> list[TokenizedResult]:
    return list(
        tokenize_trajectory_groups(
            cast(PreTrainedTokenizerBase, _Tokenizer()),
            groups,
            allow_training_without_logprobs=False,
            scale_rewards=scale_rewards,
            shuffle_group_trajectories=False,
            drop_zero_advantage_trajectories=drop,
        )
    )


@pytest.mark.parametrize("scale_rewards", [True, False])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize(
    "rewards", [[0.1] * 3, [1.0] * 3, [0.0] * 3, [0.1, 0.1 + 5e-13, 0.1 - 5e-13]]
)
def test_zero_variance_groups(
    rewards: list[float], scale_rewards: bool, drop: bool
) -> None:
    group = _group(rewards)
    assert PipelineTrainer._group_zero_variance(group)
    results = _tokenize([group], scale_rewards=scale_rewards, drop=drop)
    if drop:
        assert results == []
        return
    assert len(results) == len(rewards)
    assert all(result.advantage == 0.0 for result in results)
    packed = packed_tensors_from_tokenized_results(results, seq_len=16, verbosity=0)
    assert torch.isfinite(packed["advantages"]).all()
    assert torch.count_nonzero(packed["advantages"]) == 0
    assert torch.isfinite(packed["weights"]).all()


@pytest.mark.parametrize("scale_rewards", [True, False])
@pytest.mark.parametrize("rewards", [[0.0, 0.25, 1.0], [0.0, 2e-12, -2e-12]])
def test_nonzero_variance_advantages_are_unchanged(
    rewards: list[float], scale_rewards: bool
) -> None:
    group = _group(rewards)
    assert not PipelineTrainer._group_zero_variance(group)
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
    expected = [
        (reward - mean) / (std + 1e-6) if scale_rewards else reward - mean
        for reward in rewards
        if reward != mean
    ]
    assert [
        r.advantage for r in _tokenize([group], scale_rewards=scale_rewards)
    ] == expected


@pytest.mark.parametrize("scale_rewards", [True, False])
def test_mixed_batch_matches_normal_groups(scale_rewards: bool) -> None:
    normal = _group([0.0, 0.25, 1.0])
    mixed = _tokenize([normal, _group([0.1] * 3)], scale_rewards=scale_rewards)
    results = _tokenize([normal], scale_rewards=scale_rewards)
    assert len(mixed) == len(results)
    packed = packed_tensors_from_tokenized_results(mixed, seq_len=16, verbosity=0)
    expected = packed_tensors_from_tokenized_results(results, seq_len=16, verbosity=0)
    for key in ("tokens", "assistant_mask", "advantages", "weights"):
        assert torch.equal(packed[key], expected[key])


@pytest.mark.parametrize("advantage", [0.0, 1.0])
def test_zero_weight_denominators_are_finite(advantage: float) -> None:
    results = _tokenize([_group([0.0, 1.0])], scale_rewards=True)
    for result in results:
        result.weight = 0.0
        result.advantage = advantage
    packed = packed_tensors_from_tokenized_results(results, seq_len=16, verbosity=0)
    assert torch.isfinite(packed["weights"]).all()
    assert torch.count_nonzero(packed["weights"]) == 0
    assert torch.isfinite(packed["advantages"]).all()


def test_trainable_group_metrics_use_zero_variance_tolerance() -> None:
    summary = summarize_trajectory_groups(
        [_group([0.1] * 3), _group([0.0, 5e-13]), _group([0.0, 2e-12])]
    )
    assert summary.num_groups_trainable == 1


@pytest.mark.parametrize("backend_class", [LocalBackend, TinkerBackend])
@pytest.mark.parametrize("scale_rewards", [True, False])
@pytest.mark.parametrize("rewards", [[0.1] * 3, [1.0] * 3, []])
async def test_backend_train_skips_zero_variance_without_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_class: type[LocalBackend],
    scale_rewards: bool,
    rewards: list[float],
) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    backend = backend_class(path=str(tmp_path))
    backend._image_processors["test-model"] = None
    model = TrainableModel(
        name="zero-variance",
        run_name="zero-variance",
        project="tests",
        base_model="test-model",
        _internal_config=InternalModelConfig(init_args={"max_seq_length": 16}),
    )
    model_dir = get_model_dir(model, str(tmp_path))
    initial = Path(get_step_checkpoint_dir(model_dir, 0))
    initial.mkdir(parents=True)
    (initial / "adapter.bin").write_bytes(b"unchanged parameters")
    (initial / "optimizer.bin").write_bytes(b"unchanged optimizer")
    service = AsyncMock()
    with (
        patch.object(backend, "_get_service", return_value=service),
        patch("art.local.backend.get_tokenizer", return_value=_Tokenizer()),
        patch.object(model, "_get_wandb_run", return_value=None),
    ):
        result = await backend.train(
            model,
            [_group(rewards)],
            scale_rewards=scale_rewards,
            packed_sequence_length=16,
            logprob_calculation_chunk_size=16,
        )
    service.train.assert_not_called()
    service.register_lora_for_step.assert_awaited_once()
    assert result.step == 1
    assert result.metrics["data/step_num_groups_trainable"] == 0
    assert result.metrics["data/step_trainable_assistant_tokens"] == 0
    assert result.metrics[TRAIN_GRADIENT_STEPS_KEY] == 0
    checkpoint = Path(get_step_checkpoint_dir(model_dir, 1))
    assert (checkpoint / "adapter.bin").read_bytes() == b"unchanged parameters"
    assert (checkpoint / "optimizer.bin").read_bytes() == b"unchanged optimizer"


async def test_megatron_noop_advances_every_rank_without_touching_optimizer() -> None:
    pytest.importorskip("monarch")
    from art.megatron.runtime.executor import MegatronTrainJobExecutor
    from art.megatron.runtime.monarch import MonarchTrainerRun
    from art.megatron.runtime.specs import TrainerGeneration

    source = TrainerGeneration(
        training_session_id="session",
        policy_step=0,
        generation_id="step-00000000-" + "0" * 32,
        adapter_path="/unused/0",
    )
    output = source.model_copy(
        update={"policy_step": 1, "generation_id": "step-00000001-" + "1" * 32}
    )
    executors = []
    optimizers = []
    for _ in range(4):
        optimizer = Mock()
        optimizers.append(optimizer)
        executor = MegatronTrainJobExecutor.__new__(MegatronTrainJobExecutor)
        executor._closed = False
        executor.runtime = SimpleNamespace(
            resident_training_session_id="session",
            resident_policy_step=0,
            resident_generation_id=source.generation_id,
            optimizer_state_loaded=True,
            optimizer=optimizer,
        )
        executors.append(executor)

    async def advance_ranks(source_json, output_json, optimizer_path, adapter):
        assert adapter is None
        return {
            rank: {
                "rank": rank,
                "learner_version": 1,
                "metrics": executor.advance_without_training(
                    source=TrainerGeneration.model_validate_json(source_json),
                    output=TrainerGeneration.model_validate_json(output_json),
                    optimizer_state_path=optimizer_path,
                    adapter=None,
                ),
            }
            for rank, executor in enumerate(executors)
        }

    run = cast(Any, MonarchTrainerRun.__new__(MonarchTrainerRun))
    run._lock = asyncio.Lock()
    run._closed = False
    run._valid = True
    run._active_job_id = None
    run._learner_version = 0
    run.run_spec = SimpleNamespace(training_session_id="session", event_timeout_s=5)
    run.runtime_spec = SimpleNamespace(trainer_mesh=SimpleNamespace(ranks=(0, 1, 2, 3)))
    run._actors = SimpleNamespace(
        advance_without_training=SimpleNamespace(
            call=AsyncMock(side_effect=advance_ranks)
        )
    )

    assert (
        await run.advance_without_training(
            source=source,
            output=output,
            optimizer_state_path="/unused/optimizer",
            adapter=None,
        )
        == {}
    )
    assert run._learner_version == 1
    for executor, optimizer in zip(executors, optimizers, strict=True):
        assert executor.runtime.resident_policy_step == 1
        assert executor.runtime.resident_generation_id == output.generation_id
        assert executor.runtime.optimizer is optimizer
        optimizer.step.assert_not_called()
        optimizer.zero_grad.assert_not_called()


@pytest.mark.parametrize("empty_kind", ["zero_advantages", "zero_weights", "no_tokens"])
def test_backend_packing_skips_batches_without_weighted_advantage(
    tmp_path: Path, empty_kind: str
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    backend._image_processors["test-model"] = None
    model = TrainableModel(
        name="all-zero",
        run_name="all-zero",
        project="tests",
        base_model="test-model",
        _internal_config=InternalModelConfig(init_args={"max_seq_length": 16}),
    )
    group = _group([0.0, 1.0])
    results = _tokenize([group], scale_rewards=True)
    for result in results:
        if empty_kind == "zero_advantages":
            result.advantage = 0.0
        elif empty_kind == "zero_weights":
            result.weight = 0.0
        else:
            result.assistant_mask = [0] * len(result.token_ids)
    with (
        patch("art.local.backend.get_tokenizer", return_value=_Tokenizer()),
        patch(
            "art.local.backend.tokenize_trajectory_groups", return_value=iter(results)
        ),
    ):
        packed = backend._get_packed_tensors(
            model,
            [group],
            advantage_balance=0.0,
            allow_training_without_logprobs=False,
            scale_rewards=True,
            plot_tensors=False,
            packed_sequence_length=16,
            logprob_calculation_chunk_size=16,
        )
    assert packed is None


@pytest.mark.parametrize("scale_rewards", [True, False])
@pytest.mark.parametrize("empty_kind", ["residue", "exact", "no_groups"])
async def test_megatron_train_skips_before_distributed_training(
    tmp_path: Path, scale_rewards: bool, empty_kind: str
) -> None:
    backend = MegatronBackend(path=str(tmp_path), enable_expert_replay=False)
    backend._image_processors["test-model"] = None
    model = TrainableModel(
        name="megatron-zero-variance",
        run_name="megatron-zero-variance",
        project="tests",
        base_model="test-model",
        _internal_config=InternalModelConfig(init_args={"max_seq_length": 16}),
    )
    step = 0

    async def advance(*, expected_step: int, learner_version: int) -> dict[str, float]:
        nonlocal step
        assert expected_step == step
        assert learner_version == step + 1
        step = learner_version
        return {}

    async def pack(request):
        return backend._get_packed_tensors(
            request.model.build(),
            [group.build() for group in request.trajectory_groups],
            advantage_balance=request.advantage_balance,
            allow_training_without_logprobs=request.allow_training_without_logprobs,
            scale_rewards=request.scale_rewards,
            plot_tensors=request.plot_tensors,
            packed_sequence_length=request.packed_sequence_length,
            logprob_calculation_chunk_size=request.logprob_calculation_chunk_size,
            include_moe_routing=request.include_moe_routing,
        )

    service = SimpleNamespace(
        runtime=SimpleNamespace(pack=AsyncMock(side_effect=pack)),
        prepare_for_packing=AsyncMock(side_effect=lambda: step),
        advance_without_training=AsyncMock(side_effect=advance),
        train_packed=Mock(side_effect=AssertionError("must skip trainer dispatch")),
        wait_for_serving=AsyncMock(),
        drain_publication_metrics=Mock(return_value={}),
    )
    backend._services[backend._model_storage_key(model)] = service  # type: ignore[assignment, ty:invalid-assignment]
    groups = (
        []
        if empty_kind == "no_groups"
        else [_group([0.1 if empty_kind == "residue" else 1.0] * 3)]
    )
    with (
        patch.object(backend, "_get_service", return_value=service),
        patch(
            "art.megatron.backend.get_megatron_runtime_config",
            return_value=SimpleNamespace(packed_sequence_length=16),
        ),
        patch("art.local.backend.get_tokenizer", return_value=_Tokenizer()),
        patch.object(model, "_get_wandb_run", return_value=None),
    ):
        result = await asyncio.wait_for(
            backend.train(
                model, groups, scale_rewards=scale_rewards, save_checkpoint=False
            ),
            timeout=60,
        )
    service.train_packed.assert_not_called()
    service.advance_without_training.assert_awaited_once_with(
        expected_step=0, learner_version=1
    )
    service.wait_for_serving.assert_awaited_once_with(1)
    assert result.step == 1
    assert result.metrics[TRAIN_GRADIENT_STEPS_KEY] == 0
    assert result.metrics["data/step_trainable_assistant_tokens"] == 0
