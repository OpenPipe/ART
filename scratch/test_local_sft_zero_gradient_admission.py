from types import SimpleNamespace

import pytest

from art.megatron.backend import MegatronBackend
from art.megatron.training.client import LocalMegatronTrainingClient
from art.preprocessing.tokenize import SFTBatch
from art.training.contracts import (
    ForwardBackwardResult,
    ForwardRequest,
    LossConfig,
    SupervisedTrajectoryBatch,
)
from art.trajectories import Trajectory
from art.types import TrainSFTConfig


class _Service:
    def __init__(self) -> None:
        self.sft_calls = 0

    def resolve_sft_global_grad_accumulation_sequences(self, count: int) -> int:
        assert count == 1
        return 1

    async def sft_forward_backward_command(self, *_args, **_kwargs):
        self.sft_calls += 1
        raise AssertionError("empty SFT reached the trainer")

    def retire_command_operation(self, _operation_id: str) -> None:
        pass


class _Client(LocalMegatronTrainingClient):
    def __init__(self, service: _Service) -> None:
        super().__init__(
            run_id="run",
            learner_version=0,
            backend=object(),
            model=object(),
            service=service,
        )
        self.optimizer_submissions = 0

    async def optim_step(self, request):
        self.optimizer_submissions += 1
        return await super().optim_step(request)


class _Backend:
    def __init__(self, client: _Client) -> None:
        self.client = client

    async def training_client(self, _model):
        return self.client

    def _default_sft_batch_size(self) -> int:
        return 1


def _request(sequence_id: int) -> ForwardRequest:
    return ForwardRequest(
        run_id="run",
        request_id=f"next-{sequence_id}",
        sequence_id=sequence_id,
        batch=SupervisedTrajectoryBatch(trajectories=(Trajectory(),)),
        loss=LossConfig(name="cross_entropy"),
    )


@pytest.mark.asyncio
async def test_all_dropped_local_sft_is_noncontributing_and_sequence_safe() -> None:
    service = _Service()
    client = _Client(service)
    client._sft_tokenizer = SimpleNamespace(
        tokenize=lambda *_args, **_kwargs: SFTBatch(
            trajectory_tensors=[],
            learning_rate=0.0,
            num_trajectories=0,
            num_tokens=0,
            num_trainable_tokens=0,
            num_dropped_trajectories=1,
        )
    )
    backend = _Backend(client)

    stream = MegatronBackend._train_sft(
        backend,
        object(),
        (Trajectory(),),
        TrainSFTConfig(batch_size=1),
        {},
    )
    assert [row async for row in stream] == []

    operation = next(iter(client._operations.values()))
    assert await operation.gradient_disposition() == "empty"
    result = await operation.result()
    assert isinstance(result, ForwardBackwardResult)
    assert not result.produced_gradient
    assert result.packing.loss_bearing_tokens == 0
    assert not client._ledger.is_open_forward_backward(operation.ref.operation_id)
    assert client.optimizer_submissions == 0
    assert service.sft_calls == 0

    next_operation = await client.forward(_request(client.next_sequence_id))
    assert (await next_operation.result()).packing.physical_tokens == 0
    assert client.next_sequence_id == 2
    await client.close()
