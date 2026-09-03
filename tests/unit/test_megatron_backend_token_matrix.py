from __future__ import annotations

from array import array
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from art import TrainableModel, Trajectory, TrajectoryGroup
from art.distributed.rollout import DistributedTrajectorySelection
import art.megatron.backend as backend_module
from art.megatron.backend import MegatronBackend
from art.pipeline_tuner import PackedGroupShape, PackingLeafShape
from art.preprocessing.moe_routing import MoeRouteArray
from art.preprocessing.tokenize import SFTBatch, TokenizedResult
from art.training import (
    CheckpointRef,
    ForwardBackwardResult,
    NamedLossOutcome,
    OptimStepResult,
    PackingOutcome,
    PolicyTokenCount,
    SamplerWeightsResult,
    TrainingOutcome,
)
from art.types import TrainConfig, TrainSFTConfig


class _Decoder:
    def decode(self, token_id: int, /) -> str:
        return str(token_id)


class _Operation:
    def __init__(self, operation_id: str, result: Any, *, step: int | None = None):
        self.ref = SimpleNamespace(
            operation_id=operation_id,
            run_id="run",
            reserved_output_learner_version=step,
        )
        self._result = result

    async def result(self) -> Any:
        return self._result


class _Client:
    def __init__(self, *, learner_version: int = 0) -> None:
        self.run_id = "run"
        self.projected_learner_version = learner_version
        self.next_sequence_id = 0
        self.requests: list[Any] = []
        self.acknowledged: list[str] = []
        self.retired: list[str] = []

    async def forward_backward(self, request: Any) -> _Operation:
        assert request.sequence_id == self.next_sequence_id
        self.next_sequence_id += 1
        self.requests.append(request)
        matrix = request.batch.matrices[0]
        accepted = 1 if request.loss.name == "cross_entropy" else 2
        policy_counts = (
            None
            if request.loss.name == "cross_entropy"
            else (
                PolicyTokenCount(
                    policy_version=0,
                    accepted_trainable_tokens=accepted,
                ),
            )
        )
        packing = PackingOutcome(
            packed_sequence_length=4,
            packed_sequences=1,
            target_packed_sequences=1,
            logical_tokens=matrix.token_count,
            physical_tokens=matrix.token_count,
            packed_capacity_tokens=4,
            padding_tokens=4 - matrix.token_count,
            group_shapes=(
                PackedGroupShape(
                    leaves=(
                        PackingLeafShape(
                            matrix_id=matrix.matrix_id,
                            token_ids=array(
                                "I", matrix.row("token_ids").dense_values()
                            ),
                            shareable_length=matrix.token_count,
                        ),
                    )
                ),
            ),
        )
        result = ForwardBackwardResult(
            operation_id=f"forward-{request.sequence_id}",
            packing=packing,
            training=TrainingOutcome(
                accepted_trainable_tokens=accepted,
                policy_token_counts=policy_counts,
            ),
            loss=NamedLossOutcome(
                contract_id=request.loss.contract_id,
                value=0.5,
            ),
            metrics={
                "loss/train": 0.5,
                "time/forward_backward_s": 0.25,
            },
        )
        return _Operation(result.operation_id, result)

    async def optim_step(self, request: Any) -> _Operation:
        assert request.sequence_id == self.next_sequence_id
        self.next_sequence_id += 1
        self.requests.append(request)
        self.projected_learner_version += 1
        operation_id = f"optimizer-{request.sequence_id}"
        result = OptimStepResult(
            operation_id=operation_id,
            contributing_forward_backward_operation_ids=(
                f"forward-{request.sequence_id - 1}",
            ),
            checkpoint=CheckpointRef(
                run_id=self.run_id,
                learner_version=self.projected_learner_version,
                checkpoint_id=f"step-{self.projected_learner_version}",
            ),
            metrics={"time/optimizer_step_s": 0.1},
        )
        return _Operation(operation_id, result, step=self.projected_learner_version)

    async def save_weights_for_sampler(self, request: Any) -> _Operation:
        assert request.sequence_id == self.next_sequence_id
        self.next_sequence_id += 1
        self.requests.append(request)
        operation_id = f"publication-{request.sequence_id}"
        result = SamplerWeightsResult(
            operation_id=operation_id,
            checkpoint=CheckpointRef(
                run_id=self.run_id,
                learner_version=self.projected_learner_version,
                checkpoint_id=request.checkpoint_name,
            ),
            lora=f"model@{self.projected_learner_version}",
        )
        return _Operation(operation_id, result)

    async def acknowledge_operation(self, operation_id: str) -> None:
        self.acknowledged.append(operation_id)

    def retire_operation(self, operation_id: str) -> bool:
        self.retired.append(operation_id)
        return True


def _backend(tmp_path: Path, client: _Client) -> MegatronBackend:
    backend = MegatronBackend(
        path=str(tmp_path),
        training_binding=cast(Any, SimpleNamespace(outcome_sink=None)),
    )
    backend._training_client = cast(Any, client)
    backend.training_client = AsyncMock(return_value=client)  # type: ignore[method-assign]
    return backend


def _model() -> TrainableModel:
    return TrainableModel(
        run_name="run",
        name="model",
        project="project",
        base_model="base-model",
    )


def _rollout_result(trajectory: Trajectory) -> TokenizedResult:
    return TokenizedResult(
        advantage=1.0,
        chat="",
        token_ids=[10, 11, 12],
        input_pos=[0, 1, 2],
        assistant_mask=[0, 1, 1],
        logprobs=[float("nan"), -0.2, -0.3],
        pixel_values=None,
        image_grid_thw=None,
        trajectory=trajectory,
        choice_offsets=[1],
        extra_logprobs={},
        moe_routed_experts=MoeRouteArray(
            np.asarray([[[1]], [[2]], [[3]]], dtype=np.uint8),
            num_experts=4,
        ),
        policy_versions=[-1, 0, 0],
        _tokenizer=_Decoder(),
        weight=0.5,
        prompt_id=0,
        prompt_length=1,
    )


@pytest.mark.asyncio
async def test_bound_rl_submits_canonical_token_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _Client()
    backend = _backend(tmp_path, client)
    trajectories = [Trajectory(reward=1.0), Trajectory(reward=0.0)]
    group = TrajectoryGroup(trajectories)
    group._collect_packing_shape = True
    monkeypatch.setattr(
        backend,
        "_tokenize_bound_rollouts",
        lambda *_args: [_rollout_result(trajectories[0])],
    )
    save_token = backend_module._BOUND_SAVE_CHECKPOINT.set(False)
    try:
        rows = [
            row
            async for row in backend._train_bound_rl(
                _model(),
                [group],
                TrainConfig(learning_rate=1e-5),
                {
                    "advantage_balance": 0.25,
                    "epsilon": 0.2,
                    "epsilon_high": 0.3,
                },
                False,
            )
        ]
    finally:
        backend_module._BOUND_SAVE_CHECKPOINT.reset(save_token)

    forward = client.requests[0]
    assert forward.batch.kind == "token_matrix"
    assert [row.name for row in forward.batch.matrices[0].rows] == [
        "token_ids",
        "target_token_ids",
        "loss_weights",
        "advantages",
        "behavior_logprobs",
        "policy_version",
    ]
    assert forward.batch.routes[0].matrix_id == forward.batch.matrices[0].matrix_id
    assert forward.batch.routes[0].expert_ids == bytes([1, 2, 3])
    assert forward.loss.name == "cispo"
    assert forward.loss.normalize_advantages is False
    assert forward.loss.values == {
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.3,
    }
    assert [request.sequence_id for request in client.requests] == [0, 1, 2]
    assert group._packed_group_shape.leaves[0].matrix_id == "rollout-0"
    assert rows[0]["data/step_trainable_assistant_tokens"] == 2.0
    assert client.acknowledged == client.retired


@pytest.mark.asyncio
async def test_bound_rl_zero_work_discards_prepared_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Queue:
        def __init__(self) -> None:
            self.marked = False
            self.releases: list[tuple[Any, str, str | None]] = []

        async def mark_packed(self, *_args: Any) -> None:
            self.marked = True

        async def release_selections(
            self, selections: Any, *, disposition: str, generation_id: str | None = None
        ) -> None:
            self.releases.append((selections, disposition, generation_id))

    client = _Client()
    backend = _backend(tmp_path, client)
    materialized = TrajectoryGroup([Trajectory(reward=1.0)])
    group = TrajectoryGroup()
    queue = _Queue()
    prepared = backend_module._BoundPreparedBatch(
        groups=(group,),
        materialized=(materialized,),
        selections=("lease",),
        queue=queue,
        packing_generation="generation",
    )
    group._prepared_training_batch = prepared
    monkeypatch.setattr(backend, "_tokenize_bound_rollouts", lambda *_args: [])

    rows = [
        row
        async for row in backend._train_bound_rl(
            _model(),
            [group],
            TrainConfig(),
            {},
            False,
        )
    ]

    assert rows == [{"data/step_num_gradient_steps": 0.0}]
    assert not client.requests
    assert queue.marked is False
    assert queue.releases == [(prepared.selections, "discarded", None)]
    assert prepared.released


@pytest.mark.asyncio
async def test_bound_rl_rejects_ppo_before_client_or_lease_claim(
    tmp_path: Path,
) -> None:
    client = _Client()
    backend = _backend(tmp_path, client)

    with pytest.raises(
        ValueError, match="Megatron TokenMatrix training does not support PPO"
    ):
        await anext(
            backend._train_bound_rl(
                _model(),
                [TrajectoryGroup([Trajectory(reward=1.0)])],
                TrainConfig(),
                {"ppo": True},
                False,
            )
        )

    cast(AsyncMock, backend.training_client).assert_not_awaited()
    assert not client.requests


@pytest.mark.asyncio
async def test_ordinary_rl_packs_canonical_token_matrix_and_retains_queue_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    marked = asyncio.Event()

    class _Queue:
        def __init__(self, materialized: TrajectoryGroup) -> None:
            self.materialized = materialized
            self.releases: list[tuple[Any, str, str | None]] = []

        async def materialize_selection(self, _selection: Any) -> TrajectoryGroup:
            events.append("materialize")
            return self.materialized

        async def mark_packed(self, _selections: Any, _generation_id: str) -> None:
            events.append("mark")
            marked.set()

        async def release_selections(
            self,
            selections: Any,
            *,
            disposition: str,
            generation_id: str | None = None,
        ) -> None:
            events.append("release-selection")
            self.releases.append((selections, disposition, generation_id))

    class _Runtime:
        def __init__(self) -> None:
            self.requests: list[Any] = []
            self.released: list[Any] = []

        async def pack(self, request: Any) -> Any:
            events.append("pack-start")
            self.requests.append(request)
            await asyncio.wait_for(marked.wait(), timeout=1.0)
            events.append("pack-end")
            matrix = request.batch.matrices[0]
            shape = PackedGroupShape(
                leaves=(
                    PackingLeafShape(
                        matrix_id=matrix.matrix_id,
                        token_ids=array("I", matrix.row("token_ids").dense_values()),
                        shareable_length=matrix.token_count,
                    ),
                )
            )
            ref = SimpleNamespace(
                num_sequences=1,
                sequence_length=4,
                prefix_tree_packing_stats=SimpleNamespace(
                    logical_tokens=3,
                    physical_tokens=3,
                ),
                training_outcome=TrainingOutcome(
                    accepted_trainable_tokens=2,
                    policy_token_counts=(
                        PolicyTokenCount(
                            policy_version=0,
                            accepted_trainable_tokens=2,
                        ),
                    ),
                ),
                logical_loss_terms=2,
            )
            return SimpleNamespace(
                leases=SimpleNamespace(ref=ref),
                packed_group_shapes=(shape,),
                packing_generation_id=request.generation_id,
            )

        async def release_batch(self, packed: Any) -> None:
            events.append("release-packed")
            self.released.append(packed)

    trajectory = Trajectory(reward=1.0)
    materialized = TrajectoryGroup([trajectory])
    summary = TrajectoryGroup()
    summary._collect_packing_shape = True
    queue = _Queue(materialized)
    lease = SimpleNamespace(
        item=SimpleNamespace(
            ref=SimpleNamespace(descriptor=SimpleNamespace(retained_route_bundles=()))
        )
    )
    selection = DistributedTrajectorySelection(cast(Any, queue), cast(Any, lease))
    summary._distributed_lease = selection
    runtime = _Runtime()
    backend = MegatronBackend(path=str(tmp_path))
    monkeypatch.setattr(
        backend_module,
        "get_megatron_runtime_config",
        lambda: SimpleNamespace(
            topology=SimpleNamespace(cp=1), packed_sequence_length=4
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_service",
        AsyncMock(return_value=SimpleNamespace(runtime=runtime)),
    )
    monkeypatch.setattr(
        backend,
        "_tokenize_bound_rollouts",
        lambda *_args: [_rollout_result(trajectory)],
    )

    batch = await backend._prepare_training_batch(
        _model(),
        [summary],
        {"packed_sequence_length": 4, "scale_rewards": True},
        include_moe_routing=True,
    )

    assert batch is not None
    request = runtime.requests[0]
    assert request.batch.kind == "token_matrix"
    assert request.loss.name == "cispo"
    assert request.loss.values == {
        "clip_low_threshold": 0.0,
        "clip_high_threshold": 5.0,
    }
    assert request.return_token_logprobs is False
    assert request.retained_route_bundles == ()
    assert {
        "model",
        "trajectory_groups",
        "trajectory_sources",
        "group_ids",
        "record_ids",
        "min_source_version",
        "max_source_version",
    }.isdisjoint(type(request).model_fields)
    assert batch.trainable_assistant_tokens == 2
    assert batch.loss_bearing_tokens == 2
    assert batch.non_padding_tokens == 3
    assert summary._packed_group_shape.leaves[0].matrix_id == "rollout-0"
    assert summary._distributed_lease is None
    assert events == ["materialize", "pack-start", "mark", "pack-end"]
    assert queue.releases == []

    await backend._release_distributed_batch(batch, disposition="consumed")

    assert runtime.released == [batch.payload.packed]
    assert queue.releases == [
        ((selection,), "consumed", request.generation_id),
    ]
    assert set(events[-2:]) == {"release-packed", "release-selection"}


@pytest.mark.asyncio
async def test_ordinary_rl_zero_work_discards_unmarked_queue_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Queue:
        def __init__(self) -> None:
            self.marked = False
            self.releases: list[tuple[Any, str, str | None]] = []

        async def materialize_selection(self, _selection: Any) -> TrajectoryGroup:
            return TrajectoryGroup([Trajectory(reward=1.0)])

        async def mark_packed(self, *_args: Any) -> None:
            self.marked = True

        async def release_selection(
            self,
            selection: Any,
            *,
            disposition: str,
            generation_id: str | None = None,
        ) -> None:
            self.releases.append((selection, disposition, generation_id))

    queue = _Queue()
    lease = SimpleNamespace(
        item=SimpleNamespace(
            ref=SimpleNamespace(descriptor=SimpleNamespace(retained_route_bundles=()))
        )
    )
    selection = DistributedTrajectorySelection(cast(Any, queue), cast(Any, lease))
    group = TrajectoryGroup()
    group._distributed_lease = selection
    runtime = SimpleNamespace(pack=AsyncMock())
    backend = MegatronBackend(path=str(tmp_path))
    monkeypatch.setattr(
        backend_module,
        "get_megatron_runtime_config",
        lambda: SimpleNamespace(
            topology=SimpleNamespace(cp=1), packed_sequence_length=4
        ),
    )
    monkeypatch.setattr(
        backend,
        "_get_service",
        AsyncMock(return_value=SimpleNamespace(runtime=runtime)),
    )
    monkeypatch.setattr(backend, "_tokenize_bound_rollouts", lambda *_args: [])

    batch = await backend._prepare_training_batch(
        _model(),
        [group],
        {"packed_sequence_length": 4},
        include_moe_routing=False,
    )

    assert batch is None
    assert group._distributed_lease is None
    assert queue.marked is False
    assert queue.releases == [(selection, "discarded", None)]
    runtime.pack.assert_not_awaited()


@pytest.mark.asyncio
async def test_bound_sft_submits_canonical_token_matrix_and_reads_training_outcome(
    tmp_path: Path,
) -> None:
    class _SftTokenizer:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def tokenize(self, model: Any, trajectories: Any, **kwargs: Any) -> SFTBatch:
            self.calls.append((model, trajectories, kwargs))
            tensors = {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                "labels": torch.tensor([[-100, 2, 3]], dtype=torch.long),
            }
            return SFTBatch(
                trajectory_tensors=[tensors],
                learning_rate=kwargs["learning_rate"],
                num_trajectories=1,
                num_tokens=3,
                num_trainable_tokens=2,
            )

    client = _Client(learner_version=4)
    backend = _backend(tmp_path, client)
    tokenizer = _SftTokenizer()
    backend._sft_tokenizer = cast(Any, tokenizer)
    model = _model()
    trajectory = Trajectory()

    rows = [
        row
        async for row in backend._train_sft(
            model,
            [trajectory],
            TrainSFTConfig(learning_rate=[2e-5], batch_size=1, assistant_turns="last"),
            {},
        )
    ]

    forward = client.requests[0]
    assert forward.batch.kind == "token_matrix"
    assert forward.batch.matrices[0].row("token_ids").dense_values() == (1, 2, 3)
    assert forward.batch.matrices[0].row("target_token_ids").dense_values() == (
        2,
        3,
        0,
    )
    assert forward.loss.name == "cross_entropy"
    assert forward.loss.normalize_advantages is False
    assert forward.return_token_logprobs is False
    assert [request.sequence_id for request in client.requests] == [0, 1, 2]
    assert tokenizer.calls[0][2] == {
        "assistant_turns": "last",
        "learning_rate": 2e-5,
    }
    assert rows[0]["data/step_trainable_assistant_tokens"] == 1.0
    assert rows[0]["data/step_num_dropped_trajectories"] == 0.0
    assert rows[0]["data/step_num_gradient_steps"] == 1.0


@pytest.mark.asyncio
async def test_bound_sft_zero_work_does_not_admit_command(tmp_path: Path) -> None:
    class _ZeroSftTokenizer:
        def tokenize(self, *_args: Any, **_kwargs: Any) -> SFTBatch:
            tensors = {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "labels": torch.tensor([[-100, -100]], dtype=torch.long),
            }
            return SFTBatch(
                trajectory_tensors=[tensors],
                learning_rate=0.0,
                num_trajectories=1,
                num_tokens=2,
                num_trainable_tokens=0,
            )

    client = _Client(learner_version=3)
    backend = _backend(tmp_path, client)
    backend._sft_tokenizer = cast(Any, _ZeroSftTokenizer())

    rows = [
        row
        async for row in backend._train_sft(
            _model(),
            [Trajectory()],
            TrainSFTConfig(learning_rate=1e-5, batch_size=1),
            {},
        )
    ]

    assert rows == [
        {
            "data/step_num_trajectories": 1.0,
            "data/step_trainable_assistant_tokens": 0.0,
            "data/step_num_dropped_trajectories": 0.0,
            "data/sft_zero_work": 1.0,
            "data/step_num_gradient_steps": 0.0,
        }
    ]
    assert not client.requests
    assert client.next_sequence_id == 0
