import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import torch

from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.local.backend import _PackedTrainingBatch
from art.megatron.backend import _DistributedBatchPayload
from art.megatron.distributed_service import GenerationSnapshotLaunch
from art.megatron.optimizer_state import CheckpointFile, OptimizerAdapter
from art.megatron.runtime.publication import DurableTrainerPublication
from art.megatron.runtime.specs import ResolvedCheckpointState, TrainerGeneration
from art.megatron.training.client import LocalMegatronTrainingClient
from art.preprocessing.pack import PackingTimings
from art.preprocessing.tokenize import SFTBatch
from art.training import (
    AdamConfig,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadStateRequest,
    LossConfig,
    OptimStepRequest,
    RlTrajectoryBatch,
    SamplerPublication,
    SaveStateRequest,
    SaveWeightsForSamplerRequest,
    SupervisedTrajectoryBatch,
)
from art.trajectories import Trajectory


class _Backend:
    def __init__(self) -> None:
        ref = SimpleNamespace(
            sequence_length=8,
            num_sequences=1,
            training_kind="rl",
            prefix_tree_packing_stats=SimpleNamespace(
                physical_tokens=6,
                policy_token_counts={3: 2},
            ),
        )
        packed = SimpleNamespace(
            leases=SimpleNamespace(ref=ref),
            packed_group_shapes=(),
            non_padding_tokens=6,
            loss_bearing_tokens=2,
            trainable_assistant_tokens=2,
            trajectory_fetch_s=0.1,
            trajectory_receive_s=0.04,
            trajectory_build_s=0.06,
            packing_core_s=0.2,
            packing_lock_wait_s=0.05,
            packing_compute_s=0.15,
            packing_timings=PackingTimings(
                packing_setup_s=0.01,
                packing_tokenization_s=0.02,
                packing_filter_observe_s=0.03,
                prefix_tree_item_build_s=0.04,
                prefix_tree_plan_s=0.05,
                packing_array_allocation_s=0.06,
                packing_row_materialize_s=0.07,
                packing_tensor_finalize_s=0.08,
            ),
            trajectory_log_wait_s=0.3,
            packed_batch_finalize_s=0.4,
            packing_rpc_s=0.5,
            packed_batch_fanout_s=0.6,
        )
        self.batch = _PackedTrainingBatch(
            payload=_DistributedBatchPayload(
                packed=packed,
                groups=(),
                bundles=(),
                selections=(),
                generation_id="generation",
                runtime=object(),
            ),
            num_sequences=1,
            sequence_length=8,
            trainable_assistant_tokens=2,
            loss_bearing_tokens=2,
            non_padding_tokens=6,
            logical_tokens=6,
            physical_tokens=6,
            include_moe_routing=False,
        )

    async def _prepare_training_batch(self, *_args, **_kwargs):
        return self.batch

    async def _release_trajectory_sources(self, _batch, _payload) -> None:
        return None

    def _model_uses_expert_replay(self, _model) -> bool:
        return False

    @asynccontextmanager
    async def _training_batch_lifecycle(self, _batch):
        yield


class _Service:
    rollout_weight_update_mode = "in_flight_lora"
    rollout_weights_mode = "lora"
    optimizer_state_path = "/tmp/optimizer"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.retired_operation_ids: list[str] = []
        self.snapshot_completions: list[asyncio.Task] = []
        self.completion_gate = asyncio.Event()
        self.generation = TrainerGeneration(
            training_session_id="session",
            policy_step=4,
            generation_id="step-00000004-0123456789abcdef0123456789abcdef",
            adapter_path="/tmp/adapter",
        )
        self.adapter = OptimizerAdapter(
            identity=self.generation.adapter_path,
            training_session_id=self.generation.training_session_id,
            step=self.generation.policy_step,
            generation_id=self.generation.generation_id,
            files=(
                CheckpointFile(name="adapter_config.json", size_bytes=1),
                CheckpointFile(name="adapter_model.safetensors", size_bytes=2),
            ),
        )

    def retire_command_operation(self, operation_id: str) -> None:
        self.retired_operation_ids.append(operation_id)

    async def consume_cancelled_command(self, ref) -> None:
        self.calls.append(f"cancelled:{ref.sequence_id}")

    async def start_forward_backward_command(
        self, ref, _batch, _config, _experimental
    ):
        self.calls.append(f"fb:{ref.sequence_id}")
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(
            {"token_logprobs": ((-1.0, -2.0),), "metrics": {"fb": 1.0}}
        )
        return SimpleNamespace(completion=completion)

    async def forward_backward_command(self, ref, _batch, _config, _experimental):
        return await (
            await self.start_forward_backward_command(
                ref, _batch, _config, _experimental
            )
        ).completion

    async def forward_command(self, ref, _batch, _config, _experimental):
        self.calls.append(f"forward:{ref.sequence_id}")
        return {"token_logprobs": ((-1.0, -2.0),), "metrics": {"forward": 1.0}}

    def resolve_sft_global_grad_accumulation_sequences(self, count):
        self.calls.append(f"sft_schedule:{count}")
        return count

    async def resolve_global_grad_accumulation_sequences(self, _config):
        return 1

    async def sft_forward_backward_command(self, ref, _batch, grad_sequences):
        self.calls.append(f"sft_fb:{ref.sequence_id}:{grad_sequences}")
        return {"token_logprobs": ((-1.0, -2.0, -3.0),), "metrics": {"fb": 1.0}}

    async def sft_forward_command(self, ref, _batch, grad_sequences):
        self.calls.append(f"sft_forward:{ref.sequence_id}:{grad_sequences}")
        return {
            "token_logprobs": ((-1.0, -2.0, -3.0),),
            "metrics": {"forward": 1.0},
        }

    async def optimizer_command(self, ref, _optimizer, contributions):
        self.calls.append(f"optim:{ref.sequence_id}:{len(contributions)}")
        return {"metrics": {"optim": 1.0}}, self.generation

    async def snapshot_command(self, ref, *, save_optimizer, activate_serving):
        self.calls.append(
            f"snapshot:{ref.sequence_id}:{save_optimizer}:{activate_serving}"
        )

        async def complete():
            await self.completion_gate.wait()
            return DurableTrainerPublication(
                adapter=self.adapter,
                resume_step=4,
                optimizer_step=4 if save_optimizer else 0,
                optimizer_bytes=1 if save_optimizer else None,
            )

        completion = asyncio.create_task(complete())
        self.snapshot_completions.append(completion)
        return GenerationSnapshotLaunch(metrics={}, completion=completion)

    async def load_state_command(self, ref, source, *, restore_optimizer):
        self.calls.append(
            f"load:{ref.sequence_id}:{source.adapter_step}:{restore_optimizer}"
        )
        generation = self.generation.model_copy(
            update={
                "policy_step": ref.reserved_output_learner_version,
                "generation_id": (
                    f"step-{ref.reserved_output_learner_version:08d}-"
                    "0123456789abcdef0123456789abcdef"
                ),
            }
        )
        adapter = self.adapter.model_copy(
            update={
                "step": generation.policy_step,
                "generation_id": generation.generation_id,
            }
        )
        return (
            {"optimizer_restored": restore_optimizer},
            generation,
            {},
            DurableTrainerPublication(
                adapter=adapter,
                resume_step=generation.policy_step,
                optimizer_step=generation.policy_step,
            ),
        )


class _FailingService(_Service):
    def __init__(self, failure_kind: str) -> None:
        super().__init__()
        self.failure_kind = failure_kind
        self.failure = RuntimeError(f"{failure_kind} failed")
        self.failure_consumed = False
        self.failure_started = asyncio.Event()
        self.failure_release = asyncio.Event()

    async def _fail(self) -> None:
        self.failure_consumed = True
        self.failure_started.set()
        await self.failure_release.wait()
        raise self.failure

    async def start_forward_backward_command(
        self, ref, _batch, _config, _experimental
    ):
        if self.failure_kind == "forward_backward" and not self.failure_consumed:
            self.calls.append(f"fb:{ref.sequence_id}")
            await self._fail()
        return await super().start_forward_backward_command(
            ref, _batch, _config, _experimental
        )

    async def forward_command(self, ref, _batch, _config, _experimental):
        if self.failure_kind == "forward" and not self.failure_consumed:
            self.calls.append(f"forward:{ref.sequence_id}")
            await self._fail()
        return await super().forward_command(ref, _batch, _config, _experimental)

    async def optimizer_command(self, ref, _optimizer, contributions):
        if self.failure_kind == "optim_step" and not self.failure_consumed:
            self.calls.append(f"optim:{ref.sequence_id}:{len(contributions)}")
            await self._fail()
        return await super().optimizer_command(ref, _optimizer, contributions)

    async def load_state_command(self, ref, source, *, restore_optimizer):
        if self.failure_kind == "load_state" and not self.failure_consumed:
            self.calls.append(
                f"load:{ref.sequence_id}:{source.adapter_step}:{restore_optimizer}"
            )
            await self._fail()
        return await super().load_state_command(
            ref, source, restore_optimizer=restore_optimizer
        )


class _DelayedResultService(_Service):
    def __init__(self) -> None:
        super().__init__()
        self.result_release = asyncio.Event()

    async def start_forward_backward_command(
        self, ref, _batch, _config, _experimental
    ):
        self.calls.append(f"fb:{ref.sequence_id}")

        async def complete():
            await self.result_release.wait()
            return {"token_logprobs": ((-1.0, -2.0),), "metrics": {"fb": 1.0}}

        return SimpleNamespace(completion=asyncio.create_task(complete()))


class _Client(LocalMegatronTrainingClient):
    async def _prepare_rl_batch(self, request):
        request.batch.require_local_groups()
        return self._backend.batch


def _batch() -> RlTrajectoryBatch:
    batch = RlTrajectoryBatch(
        groups=(TrajectoryGroupBundle(header=b"header", records=()),),
        min_source_version=3,
        max_source_version=3,
    )
    object.__setattr__(batch, "_local_groups", (SimpleNamespace(trajectories=()),))
    return batch


def test_local_client_executes_one_ordered_command_stream() -> None:
    async def run() -> None:
        backend = _Backend()
        service = _Service()
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=backend,
            model=object(),
            service=service,
        )
        request = ForwardBackwardRequest(
            run_id="run",
            request_id="fb",
            sequence_id=0,
            batch=_batch(),
            loss=LossConfig(name="cispo"),
        )
        forward = await client.forward_backward(request)
        assert await client.forward_backward(request) is forward
        optimizer = await client.optim_step(
            OptimStepRequest(
                run_id="run",
                request_id="optim",
                sequence_id=1,
                optimizer=AdamConfig(learning_rate=1e-3),
            )
        )
        forward_result, optimizer_result = await asyncio.gather(
            forward.result(), optimizer.result()
        )
        assert forward_result.packing.policy_token_counts[0].policy_version == 3
        assert optimizer_result.contributing_forward_backward_operation_ids == (
            forward.ref.operation_id,
        )
        assert service.retired_operation_ids == [forward.ref.operation_id]
        sampler = await client.save_weights_for_sampler(
            SaveWeightsForSamplerRequest(
                run_id="run",
                request_id="sampler",
                sequence_id=2,
                checkpoint_name="step-4",
                publication=SamplerPublication(
                    mode="in_flight_lora", model_alias="model"
                ),
            )
        )
        state = await client.save_state(
            SaveStateRequest(
                run_id="run",
                request_id="state",
                sequence_id=3,
                checkpoint_name="step-4",
            )
        )
        await asyncio.wait_for(
            asyncio.gather(
                asyncio.shield(sampler._ordered),
                asyncio.shield(state._ordered),
            ),
            timeout=1,
        )
        service.completion_gate.set()
        await asyncio.gather(sampler.result(), state.result())
        assert service.calls == [
            "fb:0",
            "optim:1:1",
            "snapshot:2:False:True",
            "snapshot:3:True:False",
        ]
        try:
            await client.forward_backward(request)
        except RuntimeError as error:
            assert "gapless" in str(error)
        else:
            raise AssertionError("retired F/B retry unexpectedly executed")
        await client.close()

    asyncio.run(run())


def test_local_optimizer_starts_before_forward_backward_result_settles() -> None:
    async def run() -> None:
        service = _DelayedResultService()
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=_Backend(),
            model=object(),
            service=service,
        )
        forward = await client.forward_backward(
            ForwardBackwardRequest(
                run_id="run",
                request_id="fb",
                sequence_id=0,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        optimizer = await client.optim_step(
            OptimStepRequest(
                run_id="run",
                request_id="optim",
                sequence_id=1,
                optimizer=AdamConfig(learning_rate=1e-3),
            )
        )
        optimizer_result = await asyncio.wait_for(optimizer.result(), timeout=1)
        assert optimizer_result.contributing_forward_backward_operation_ids == (
            forward.ref.operation_id,
        )
        assert service.calls == ["fb:0", "optim:1:1"]
        assert not forward._result.done()

        service.result_release.set()
        assert (await forward.result()).operation_id == forward.ref.operation_id
        await client.close()

    asyncio.run(run())


def test_local_forward_and_load_use_the_same_ordered_stream() -> None:
    async def run() -> None:
        backend = _Backend()
        service = _Service()
        service.completion_gate.set()
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=backend,
            model=object(),
            service=service,
        )
        forward = await client.forward(
            ForwardRequest(
                run_id="run",
                request_id="forward",
                sequence_id=0,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        state = await client.save_state(
            SaveStateRequest(
                run_id="run",
                request_id="state",
                sequence_id=1,
                checkpoint_name="baseline",
            )
        )
        await asyncio.gather(forward.result(), state.result())
        loaded = await client.load_state(
            LoadStateRequest(
                run_id="run",
                request_id="load-weights",
                sequence_id=2,
                checkpoint="baseline",
            )
        )
        assert not (await loaded.result()).optimizer_restored
        client._remember_checkpoint(
            "exact",
            ResolvedCheckpointState(
                adapter_path="/tmp/adapter",
                adapter_step=4,
                adapter_training_session_id="session",
                adapter_generation_id="generation-4",
                optimizer_state_path="/tmp/optimizer",
                optimizer_generation_id="generation-4",
            ),
        )
        exact = await client.load_state_with_optimizer(
            LoadStateRequest(
                run_id="run",
                request_id="load-exact",
                sequence_id=3,
                checkpoint="exact",
            )
        )
        assert (await exact.result()).optimizer_restored
        assert service.calls == [
            "forward:0",
            "snapshot:1:True:False",
            "load:2:4:False",
            "load:3:4:True",
        ]
        assert service.retired_operation_ids == []
        assert client.projected_learner_version == 5
        await client.close()

    asyncio.run(run())


def test_failed_local_forward_releases_concurrently_submitted_successor() -> None:
    async def run() -> None:
        service = _FailingService("forward")
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=_Backend(),
            model=object(),
            service=service,
        )
        failed = await client.forward(
            ForwardRequest(
                run_id="run",
                request_id="failed-forward",
                sequence_id=0,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        successor = await client.forward(
            ForwardRequest(
                run_id="run",
                request_id="successor-forward",
                sequence_id=1,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        await asyncio.wait_for(service.failure_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert service.calls == ["forward:0"]

        service.failure_release.set()
        with pytest.raises(RuntimeError, match="forward failed"):
            await failed.result()
        assert (await asyncio.wait_for(successor.result(), timeout=1)).operation_id
        assert service.calls == ["forward:0", "forward:1"]
        assert client.next_sequence_id == 2
        await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure_kind", ["forward_backward", "optim_step", "load_state"]
)
def test_failed_local_mutation_poisons_concurrently_submitted_successor(
    failure_kind: str,
) -> None:
    async def run() -> None:
        service = _FailingService(failure_kind)
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=_Backend(),
            model=object(),
            service=service,
        )
        if failure_kind == "forward_backward":
            failed = await client.forward_backward(
                ForwardBackwardRequest(
                    run_id="run",
                    request_id="failed-fb",
                    sequence_id=0,
                    batch=_batch(),
                    loss=LossConfig(name="cispo"),
                )
            )
        elif failure_kind == "optim_step":
            contribution = await client.forward_backward(
                ForwardBackwardRequest(
                    run_id="run",
                    request_id="fb",
                    sequence_id=0,
                    batch=_batch(),
                    loss=LossConfig(name="cispo"),
                )
            )
            await contribution.result()
            failed = await client.optim_step(
                OptimStepRequest(
                    run_id="run",
                    request_id="failed-optim",
                    sequence_id=1,
                    optimizer=AdamConfig(learning_rate=1e-3),
                )
            )
        else:
            client._remember_checkpoint(
                "source",
                ResolvedCheckpointState(
                    adapter_path="/tmp/adapter",
                    adapter_step=3,
                    adapter_training_session_id="session",
                    adapter_generation_id="generation-3",
                    optimizer_state_path="/tmp/optimizer",
                    optimizer_generation_id="generation-3",
                ),
            )
            failed = await client.load_state(
                LoadStateRequest(
                    run_id="run",
                    request_id="failed-load",
                    sequence_id=0,
                    checkpoint="source",
                )
            )

        successor_sequence = client.next_sequence_id
        successor = await client.forward(
            ForwardRequest(
                run_id="run",
                request_id="blocked-forward",
                sequence_id=successor_sequence,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        await asyncio.wait_for(service.failure_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert f"forward:{successor_sequence}" not in service.calls

        service.failure_release.set()
        with pytest.raises(
            RuntimeError, match=f"{failure_kind} failed"
        ) as failed_error:
            await failed.result()
        with pytest.raises(
            RuntimeError, match=f"{failure_kind} failed"
        ) as successor_error:
            await asyncio.wait_for(successor.result(), timeout=1)
        assert failed_error.value is service.failure
        assert successor_error.value is service.failure
        assert f"forward:{successor_sequence}" not in service.calls
        assert client.next_sequence_id == successor_sequence + 1
        await client.close()

    asyncio.run(run())


def test_running_local_mutation_cancellation_matches_remote() -> None:
    async def run() -> None:
        service = _FailingService("forward_backward")
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=_Backend(),
            model=object(),
            service=service,
        )
        failed = await client.forward_backward(
            ForwardBackwardRequest(
                run_id="run",
                request_id="cancelled-fb",
                sequence_id=0,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        successor = await client.forward(
            ForwardRequest(
                run_id="run",
                request_id="blocked-forward",
                sequence_id=1,
                batch=_batch(),
                loss=LossConfig(name="cispo"),
            )
        )
        await asyncio.wait_for(service.failure_started.wait(), timeout=1)
        cancellation = asyncio.create_task(failed.cancel())
        await asyncio.sleep(0)
        assert not cancellation.done()

        service.failure_release.set()
        with pytest.raises(RuntimeError, match="forward_backward failed"):
            await cancellation
        with pytest.raises(RuntimeError, match="forward_backward failed"):
            await failed.result()
        with pytest.raises(RuntimeError, match="forward_backward failed"):
            await asyncio.wait_for(successor.result(), timeout=1)
        assert service.calls == ["fb:0"]
        assert client._sequence_tail is not None
        assert isinstance(client._sequence_tail._future.result(), RuntimeError)
        assert client._sequence_tail._future.exception() is None
        await asyncio.wait_for(client.close(), timeout=1)

    asyncio.run(run())


def test_running_deferred_save_cancellation_awaits_terminal_result() -> None:
    async def run() -> None:
        service = _Service()
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=_Backend(),
            model=object(),
            service=service,
        )
        operation = await client.save_state(
            SaveStateRequest(
                run_id="run",
                request_id="state",
                sequence_id=0,
                checkpoint_name="state",
            )
        )
        await operation._ordered
        cancellation = asyncio.create_task(operation.cancel())
        await asyncio.sleep(0)

        assert len(service.snapshot_completions) == 1
        assert not service.snapshot_completions[0].done()
        assert not cancellation.done()
        service.completion_gate.set()
        await cancellation
        assert (await operation.result()).operation_id == operation.ref.operation_id
        assert not service.snapshot_completions[0].cancelled()
        await client.close()

    asyncio.run(run())


def test_close_cancels_ten_deferred_snapshot_completions() -> None:
    async def run() -> None:
        service = _Service()
        client = _Client(
            run_id="run",
            learner_version=3,
            backend=_Backend(),
            model=object(),
            service=service,
        )
        operations = [
            await client.save_state(
                SaveStateRequest(
                    run_id="run",
                    request_id=f"state-{sequence_id}",
                    sequence_id=sequence_id,
                    checkpoint_name=f"state-{sequence_id}",
                )
            )
            for sequence_id in range(10)
        ]
        await asyncio.gather(*(operation._ordered for operation in operations))
        assert len(service.snapshot_completions) == 10
        assert all(not task.done() for task in service.snapshot_completions)
        assert all(
            raw in operation._state.owned_tasks
            for operation, raw in zip(
                operations, service.snapshot_completions, strict=True
            )
        )

        await asyncio.wait_for(client.close(), timeout=1)
        assert all(task.cancelled() for task in service.snapshot_completions)
        assert all(operation._result.cancelled() for operation in operations)
        assert not any(not task.done() for task in client._completion_tasks)
        assert all(
            all(task.done() for task in operation._state.owned_tasks)
            for operation in operations
        )
        assert client._operations == {}
        assert client._ledger._records == {}
        assert service.retired_operation_ids == [
            operation.ref.operation_id for operation in operations
        ]

    asyncio.run(run())


def test_local_client_dispatches_supervised_commands_without_rl_packing() -> None:
    async def run() -> None:
        backend = _Backend()
        service = _Service()
        client = LocalMegatronTrainingClient(
            run_id="run",
            learner_version=3,
            backend=backend,
            model=object(),
            service=service,
        )
        client._sft_tokenizer = SimpleNamespace(
            tokenize=lambda *_args, **_kwargs: SFTBatch(
                trajectory_tensors=[
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.ones((1, 3), dtype=torch.long),
                        "labels": torch.tensor([[-100, 2, 3]]),
                    }
                ],
                learning_rate=0.0,
                num_trajectories=1,
                num_tokens=3,
                num_trainable_tokens=2,
            )
        )
        forward = await client.forward_backward(
            ForwardBackwardRequest(
                run_id="run",
                request_id="sft-fb",
                sequence_id=0,
                batch=SupervisedTrajectoryBatch(trajectories=(Trajectory(),)),
                loss=LossConfig(name="cross_entropy"),
            )
        )
        optimizer = await client.optim_step(
            OptimStepRequest(
                run_id="run",
                request_id="optim",
                sequence_id=1,
                optimizer=AdamConfig(learning_rate=1e-3),
            )
        )
        result, _ = await asyncio.gather(forward.result(), optimizer.result())
        assert result.packing.policy_token_counts is None
        assert result.packing.loss_bearing_tokens == 2
        assert service.calls == ["sft_schedule:1", "sft_fb:0:1", "optim:1:1"]
        assert service.retired_operation_ids == [forward.ref.operation_id]
        await client.close()

    asyncio.run(run())
