from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import torch

from art import TrainableModel, Trajectory
from art.local import LocalBackend
from art.preprocessing.tokenize import SFTBatch
from art.types import TrainSFTConfig


def _trajectory(content: str) -> Trajectory:
    return Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": content},
        ]
    )


@contextmanager
def _local_sft_patches(
    backend: LocalBackend,
    tokenize_side_effect: Any,
    get_service: AsyncMock,
) -> Iterator[None]:
    with ExitStack() as stack:
        for patcher in (
            patch(
                "art.local.backend.AutoTokenizer.from_pretrained",
                return_value=object(),
            ),
            patch.object(
                backend,
                "_configure_training_tokenizer",
                return_value=object(),
            ),
            patch(
                "art.utils.model_config.get_instruction_response_parts",
                return_value=("<user>", "<assistant>"),
            ),
            patch.object(backend, "_model_max_sequence_length", return_value=None),
            patch(
                "art.local.backend.tokenize_sft_batch",
                side_effect=tokenize_side_effect,
            ),
            patch.object(backend, "_get_service", get_service),
        ):
            stack.enter_context(patcher)
        yield


@pytest.mark.asyncio
async def test_local_sft_does_not_start_service_without_trainable_tokens(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    model = TrainableModel(
        name="empty-sft",
        project="tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    calls: list[dict[str, Any]] = []

    def tokenize(**kwargs: Any) -> SFTBatch:
        calls.append(kwargs)
        return SFTBatch(
            trajectory_tensors=[],
            learning_rate=kwargs["learning_rate"],
            num_trajectories=0,
            num_tokens=0,
            num_trainable_tokens=0,
            num_dropped_trajectories=1,
        )

    get_service = AsyncMock()
    with _local_sft_patches(backend, tokenize, get_service):
        results = [
            result
            async for result in backend._train_sft(
                model,
                [_trajectory("answer")],
                TrainSFTConfig(
                    learning_rate=[1e-4],
                    batch_size=1,
                    assistant_turns="last",
                ),
                {},
            )
        ]

    assert results == []
    assert calls[0]["assistant_turns"] == "last"
    get_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_sft_skipped_batch_does_not_consume_learning_rate(
    tmp_path: Path,
) -> None:
    backend = LocalBackend(path=str(tmp_path))
    model = TrainableModel(
        name="filtered-sft",
        project="tests",
        base_model="test-model",
        base_path=str(tmp_path),
    )
    calls: list[dict[str, Any]] = []

    def tokenize(**kwargs: Any) -> SFTBatch:
        calls.append(kwargs)
        is_first = len(calls) == 1
        trajectory_tensors = (
            []
            if is_first
            else [
                {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                    "labels": torch.tensor([[-100, 2]]),
                }
            ]
        )
        return SFTBatch(
            trajectory_tensors=trajectory_tensors,
            learning_rate=kwargs["learning_rate"],
            num_trajectories=0 if is_first else 1,
            num_tokens=0 if is_first else 2,
            num_trainable_tokens=0 if is_first else 1,
            num_dropped_trajectories=1 if is_first else 0,
        )

    captured_batches: list[SFTBatch] = []

    class Service:
        async def train_sft(
            self,
            batches: list[SFTBatch],
            config: TrainSFTConfig,
            verbose: bool,
        ):
            del config, verbose
            captured_batches.extend(batches)
            yield {"loss/train": 0.5}

    get_service = AsyncMock(return_value=Service())
    with _local_sft_patches(backend, tokenize, get_service):
        results = [
            result
            async for result in backend._train_sft(
                model,
                [_trajectory("dropped"), _trajectory("trained")],
                TrainSFTConfig(
                    learning_rate=[1e-4, 2e-4],
                    batch_size=1,
                    assistant_turns="last",
                ),
                {},
            )
        ]

    assert [call["learning_rate"] for call in calls] == [1e-4, 1e-4]
    assert [batch.learning_rate for batch in captured_batches] == [1e-4]
    assert results[0]["data/step_num_dropped_trajectories"] == 1.0
