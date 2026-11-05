import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from art import dev
from art.preprocessing.pack import DiskPackedTensors
from art.types import TrainConfig
from art.unsloth.service import UnslothService
import art.unsloth.train as train

_DISK_PACKED_TENSORS = DiskPackedTensors(
    dir="unused", num_sequences=2, sequence_length=1
)


@pytest.mark.parametrize("configured", [2, 4])
@pytest.mark.parametrize("requested", [None, 2])
async def test_service_rejects_unsupported_rl_accumulation(
    tmp_path, configured, requested
):
    service = UnslothService(
        "test",
        "test-model",
        {"trainer_args": {"gradient_accumulation_steps": configured}},
        str(tmp_path),
    )

    with pytest.raises(ValueError, match="set gradient_accumulation_steps=1"):
        await service.resolve_global_grad_accumulation_sequences(
            TrainConfig(grad_accumulation_sequences=requested)
        )
    assert "_state" not in service.__dict__


@pytest.mark.parametrize("configured", [None, 1])
@pytest.mark.parametrize("requested", [None, 1])
async def test_service_accepts_default_rl_accumulation(tmp_path, configured, requested):
    config: dev.InternalModelConfig = {}
    if configured is not None:
        config["trainer_args"] = {"gradient_accumulation_steps": configured}
    service = UnslothService("test", "test-model", config, str(tmp_path))

    assert (
        await service.resolve_global_grad_accumulation_sequences(
            TrainConfig(grad_accumulation_sequences=requested)
        )
        == 1
    )


@pytest.mark.parametrize("configured", [-1, 0])
async def test_service_rejects_invalid_accumulation(tmp_path, configured):
    service = UnslothService(
        "test",
        "test-model",
        {"trainer_args": {"gradient_accumulation_steps": configured}},
        str(tmp_path),
    )

    with pytest.raises(ValueError, match="must be >= 1"):
        await service.resolve_global_grad_accumulation_sequences(TrainConfig())


async def test_service_rejects_mismatched_accumulation(tmp_path):
    service = UnslothService("test", "test-model", {}, str(tmp_path))

    with pytest.raises(ValueError, match="grad_accumulation_sequences=1, got 2"):
        await service.resolve_global_grad_accumulation_sequences(
            TrainConfig(grad_accumulation_sequences=2)
        )


def _context(steps, warmup):
    trainer = Mock()
    trainer.args.gradient_accumulation_steps = steps
    trainer.optimizer = SimpleNamespace(
        state={"sentinel": {}}, param_groups=[{"weight_decay": 0.0}]
    )
    return train.UnslothTrainContext(
        model=Mock(),
        tokenizer=Mock(),
        peft_model=Mock(),
        trainer=trainer,
        inputs_queue=asyncio.Queue(),
        results_queue=asyncio.Queue(),
        warmup_pending=warmup,
        last_training_mode="sft",
    )


@pytest.mark.parametrize("steps", [0, 2, 4])
@pytest.mark.parametrize("warmup", [False, True])
async def test_runtime_rejects_accumulation_before_mutating_context(
    monkeypatch, steps, warmup
):
    ctx = _context(steps, warmup)
    load_tensors = Mock(side_effect=AssertionError("packed tensors must not be read"))
    start_training = Mock(side_effect=AssertionError("trainer must not start"))
    monkeypatch.setattr(train, "packed_tensors_from_dir", load_tensors)
    monkeypatch.setattr(train, "train", start_training)
    message = "must be >= 1" if steps < 1 else "set gradient_accumulation_steps=1"

    with pytest.raises(ValueError, match=message):
        await anext(
            train.run_unsloth_rl_training(ctx, _DISK_PACKED_TENSORS, TrainConfig(), {})
        )

    load_tensors.assert_not_called()
    start_training.assert_not_called()
    assert ctx.train_task is None
    assert ctx.inputs_queue.empty()
    assert ctx.results_queue.empty()
    assert ctx.warmup_pending is warmup
    assert ctx.last_training_mode == "sft"
    assert ctx.trainer.optimizer.state == {"sentinel": {}}
    assert ctx.trainer.optimizer.param_groups == [{"weight_decay": 0.0}]


@pytest.mark.parametrize("warmup", [False, True])
async def test_runtime_default_accumulation_still_produces_results(monkeypatch, warmup):
    ctx = _context(1, warmup)
    consumed = []

    async def trainer_loop(*, trainer, results_queue):
        while True:
            for _ in range(trainer.args.gradient_accumulation_steps):
                consumed.append(await ctx.inputs_queue.get())
            results_queue.put_nowait({"loss": 1.0})

    monkeypatch.setattr(train, "train", trainer_loop)
    monkeypatch.setattr(
        train,
        "packed_tensors_from_dir",
        lambda **kwargs: {"tokens": torch.zeros((2, 1))},
    )
    monkeypatch.setattr(
        train,
        "create_train_inputs",
        lambda packed, offset, config, dev_config, warmup: (offset, warmup),
    )
    monkeypatch.setattr(train, "gc_and_empty_cuda_cache", lambda: None)

    async def collect():
        return [
            result
            async for result in train.run_unsloth_rl_training(
                ctx, _DISK_PACKED_TENSORS, TrainConfig(), {}
            )
        ]

    try:
        results = await asyncio.wait_for(collect(), timeout=5)
        assert results == [{"loss": 1.0}, {"loss": 1.0}]
        assert consumed == ([(0, True)] if warmup else []) + [(0, False), (1, False)]
        assert ctx.warmup_pending is False
        assert ctx.inputs_queue.empty()
        await asyncio.wait_for(ctx.results_queue.join(), timeout=5)
    finally:
        if ctx.train_task is not None:
            ctx.train_task.cancel()
            await asyncio.gather(ctx.train_task, return_exceptions=True)
