import asyncio

import pytest

from art.megatron.optimizer_state import CheckpointFile, OptimizerAdapter
from art.megatron.runtime.specs import ResolvedCheckpointState, TrainerGeneration
from art.megatron.training.client import LocalMegatronTrainingClient
from art.training.contracts import (
    LoadStateRequest,
    SamplerPublication,
    SaveWeightsForSamplerRequest,
)


def _generation_id(step: int) -> str:
    return f"step-{step:08d}-{'0' * 32}"


def _adapter(step: int) -> OptimizerAdapter:
    return OptimizerAdapter(
        identity=f"/adapter/{step}",
        training_session_id="session",
        step=step,
        generation_id=_generation_id(step),
        files=(
            CheckpointFile(name="adapter_config.json", size_bytes=1),
            CheckpointFile(name="adapter_model.safetensors", size_bytes=2),
        ),
    )


class _Service:
    rollout_weight_update_mode = "in_flight_lora"

    def __init__(self) -> None:
        self.snapshot_calls = 0
        self.load_calls = []
        self.load_started = asyncio.Event()
        self.load_gate = asyncio.Event()

    async def snapshot_command(self, *_args, **_kwargs):
        self.snapshot_calls += 1
        raise AssertionError("conflicting checkpoint started a snapshot")

    async def load_state_command(self, ref, _source, *, restore_optimizer):
        self.load_calls.append(restore_optimizer)
        self.load_started.set()
        await self.load_gate.wait()
        step = ref.reserved_output_learner_version
        assert step is not None
        generation = TrainerGeneration(
            training_session_id="session",
            policy_step=step,
            generation_id=_generation_id(step),
            adapter_path=f"/adapter/{step}",
        )
        return (
            {"optimizer_restored": restore_optimizer},
            generation,
            {},
            _adapter(step),
        )

    def retire_command_operation(self, _operation_id: str) -> None:
        pass


def _client(service: _Service, *, learner_version: int) -> LocalMegatronTrainingClient:
    return LocalMegatronTrainingClient(
        run_id="run",
        learner_version=learner_version,
        backend=object(),
        model=object(),
        service=service,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_conflicting_sampler_name_fails_before_snapshot(pending: bool) -> None:
    service = _Service()
    client = _client(service, learner_version=4)
    if pending:
        client._claim_checkpoint_name("shared", 3)
    else:
        client._remember_checkpoint(
            "shared", ResolvedCheckpointState(adapter=_adapter(3))
        )

    operation = await client.save_weights_for_sampler(
        SaveWeightsForSamplerRequest(
            run_id="run",
            request_id="save",
            sequence_id=0,
            checkpoint_name="shared",
            publication=SamplerPublication(mode="none"),
        )
    )

    with pytest.raises(RuntimeError, match="identifies different learners"):
        await operation.result()
    assert service.snapshot_calls == 0
    await client.close()


@pytest.mark.asyncio
async def test_load_restore_mode_is_part_of_local_idempotency() -> None:
    service = _Service()
    client = _client(service, learner_version=3)
    client._remember_checkpoint(
        "source",
        ResolvedCheckpointState(
            adapter=_adapter(3),
            optimizer_state_path="/optimizer",
            optimizer_generation_id=_generation_id(3),
        ),
    )
    request = LoadStateRequest(
        run_id="run",
        request_id="load",
        sequence_id=0,
        checkpoint="source",
    )

    operation = await client.load_state(request)
    await service.load_started.wait()
    with pytest.raises(RuntimeError, match="request_id was reused"):
        await client.load_state_with_optimizer(request)

    service.load_gate.set()
    result = await operation.result()
    assert not result.optimizer_restored
    assert service.load_calls == [False]
    await client.close()
