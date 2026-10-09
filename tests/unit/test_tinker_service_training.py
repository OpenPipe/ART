import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
import tinker
import torch

from art import TrainConfig, Trajectory
from art.preprocessing.pack import (
    packed_tensors_from_tokenized_results,
    packed_tensors_to_dir,
)
from art.preprocessing.tokenize import TokenizedResult
from art.tinker.service import TinkerService, TinkerState


class FakeTrainingClient:
    def __init__(self) -> None:
        self.batches: list[list[tinker.Datum]] = []
        self.gradients: list[list[torch.Tensor]] = []
        self.events: list[str] = []

    async def forward_backward_custom_async(self, data, loss_fn):
        self.events.append("forward_backward")
        self.batches.append(data)
        logprobs = [
            (
                -0.001
                * (
                    torch.tensor(datum.model_input.to_ints()).cumsum(0)
                    + datum.loss_fn_inputs["target_tokens"].to_torch()
                )
            ).requires_grad_()
            for datum in data
        ]
        loss, metrics = loss_fn(data, logprobs)
        assert torch.isfinite(loss)
        loss.backward()
        gradients = []
        for lp in logprobs:
            assert lp.grad is not None
            assert torch.isfinite(lp.grad).all()
            gradients.append(lp.grad.clone())
        self.gradients.append(gradients)
        future = asyncio.get_running_loop().create_future()
        future.set_result(SimpleNamespace(metrics=metrics))
        return future

    async def optim_step_async(self, adam_params):
        self.events.append("optim_step")
        future = asyncio.get_running_loop().create_future()
        future.set_result(SimpleNamespace(metrics={"optim/step": 1.0}))
        return future


def _result(prompt: list[int], completion: list[int], advantage: float):
    tokens = prompt + completion
    return TokenizedResult(
        advantage=advantage,
        chat="",
        token_ids=tokens,
        input_pos=list(range(len(tokens))),
        assistant_mask=[0] * len(prompt) + [1] * len(completion),
        logprobs=[float("nan")] * len(prompt) + [-0.5] * len(completion),
        pixel_values=None,
        image_grid_thw=None,
        trajectory=Trajectory(reward=advantage),
        choice_offsets=[len(prompt)],
        extra_logprobs={},
        _tokenizer=Mock(),
        weight=1.0,
        prompt_length=len(prompt),
    )


async def _train(tmp_path: Path, monkeypatch, results, *, seq_len=32):
    packed = packed_tensors_from_tokenized_results(
        results,
        seq_len=seq_len,
        min_prefix_tree_shared_segment_length=1,
    )
    disk = packed_tensors_to_dir(packed, str(tmp_path / "tensors"))
    client = FakeTrainingClient()
    service = TinkerService("test", "test-model", {}, str(tmp_path))
    state = TinkerState(
        service_client=Mock(),
        rest_client=Mock(),
        training_client=cast(tinker.TrainingClient, client),
        models={},
    )
    state_future = asyncio.get_running_loop().create_future()
    state_future.set_result(state)
    service.__dict__["_state_task"] = state_future
    monkeypatch.setattr(service, "_get_last_checkpoint_dir", lambda: tmp_path / "0000")
    save_checkpoint = AsyncMock(return_value="tinker://test/0001")
    monkeypatch.setattr(service, "_save_checkpoint", save_checkpoint)
    config = TrainConfig(learning_rate=1e-4)
    metrics = [result async for result in service.train(disk, config, {})]
    save_checkpoint.assert_awaited_once()
    assert state.models == {
        "test@1": "tinker://test/0001",
        "test": "tinker://test/0001",
    }
    return packed, client, metrics


@pytest.mark.parametrize(
    "sequences,expected_inputs",
    [
        (
            [([101, 102], [201, 202]), ([101, 102], [301, 302])],
            [[101], [101, 102, 201, 202], [101, 102, 301, 302]],
        ),
        (
            [
                ([101, 102, 103], [201, 202]),
                ([101, 102, 103], [301, 302]),
                ([101, 104, 105], [401, 402]),
            ],
            [
                [101],
                [101, 102],
                [101, 102, 103, 201, 202],
                [101, 102, 103, 301, 302],
                [101, 104, 105, 401, 402],
            ],
        ),
        (
            [([101, 102], [201, 202]), ([401, 402], [501, 502])],
            [[101, 102, 201, 202], [401, 402, 501, 502]],
        ),
    ],
)
async def test_datums_use_only_their_branch_and_ancestors(
    tmp_path, monkeypatch, sequences, expected_inputs
):
    results = [_result(prompt, completion, 1.0) for prompt, completion in sequences]
    packed, client, _ = await _train(tmp_path, monkeypatch, results)
    assert packed["tokens"].shape[0] == 1
    assert (packed["group_ids"] == -1).any()
    datums = client.batches[0]
    assert [
        datum.model_input.to_ints()
        for datum in datums
        if any(token >= 0 for token in datum.model_input.to_ints())
    ] == expected_inputs
    assert len(datums) == len(expected_inputs)
    for datum, tokens in zip(datums, expected_inputs, strict=True):
        assert datum.loss_fn_inputs["target_tokens"].to_torch().tolist() == (
            tokens[1:] + [0]
        )
        weights = datum.loss_fn_inputs["weights"].to_torch()
        assert weights.dtype == torch.float32
        assert weights.tolist() == [1.0] * len(tokens)

    trained_targets = []
    for datum, grad in zip(datums, client.gradients[0], strict=True):
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        trained_targets.extend(targets[grad != 0].tolist())
    assert sorted(trained_targets) == sorted(
        token for _, completion in sequences for token in completion
    )


async def test_optimizer_steps_once_per_packed_row(tmp_path, monkeypatch):
    results = [
        _result([101, 102], [201, 202], 1.0),
        _result([101, 102], [301, 302], -1.0),
    ]
    packed, client, metrics = await _train(tmp_path, monkeypatch, results, seq_len=4)
    assert packed["tokens"].shape[0] == 2
    assert client.events == [
        "forward_backward",
        "optim_step",
        "forward_backward",
        "optim_step",
    ]
    assert len(metrics) == 2
    service = TinkerService("test", "test-model", {}, str(tmp_path))
    assert await service.resolve_global_grad_accumulation_sequences(TrainConfig()) == 1
    with pytest.raises(ValueError, match="grad_accumulation_sequences=1"):
        await service.resolve_global_grad_accumulation_sequences(
            TrainConfig(grad_accumulation_sequences=2)
        )
