from types import SimpleNamespace

import pytest

from art.distributed.art_runtime import DistributedPackedBatch
from art.distributed.data_plane import (
    PackedBatchLeaseSet,
    PackedBatchRef,
    PrefixTreePackingStatsSpec,
    TensorSpec,
)
from art.distributed.rollout import RolloutModelSpec
from art.distributed.trajectory_store import TrajectoryGroupBundle
from art.megatron.operation_handler import (
    MegatronOperationConfig,
    MegatronOperationHandler,
)
from art.megatron.runtime.specs import TrainerGeneration
from art.training import (
    AdamConfig,
    ForwardBackwardRequest,
    ForwardRequest,
    LossConfig,
    OperationRef,
    OptimStepRequest,
    RlTrajectoryBatch,
)


def _packed_batch() -> DistributedPackedBatch:
    item_sizes = {
        "tokens": ("int64", 8),
        "group_ids": ("int64", 8),
        "parent_ids": ("int64", 8),
        "input_pos": ("int64", 8),
        "assistant_mask": ("bool", 1),
        "logprobs": ("float32", 4),
        "advantages": ("float32", 4),
        "weights": ("float32", 4),
    }
    offset = 0
    tensors = []
    for name, (dtype, item_size) in item_sizes.items():
        byte_count = 8 * item_size
        tensors.append(
            TensorSpec(
                name=name,
                dtype=dtype,
                shape=(1, 8),
                offset=offset,
                byte_count=byte_count,
            )
        )
        offset += byte_count
    ref = PackedBatchRef(
        batch_id="batch",
        owner_actor_id="owner",
        lease_id="lease",
        shared_memory_name="shm",
        owner_process_id=1,
        tensors=tuple(tensors),
        num_sequences=1,
        sequence_length=8,
        byte_count=offset,
        storage_byte_count=offset,
        pixel_values_present=(False,),
        image_grid_thw_present=(False,),
        prefix_tree_packing_stats=PrefixTreePackingStatsSpec(
            logical_tokens=7, physical_tokens=8
        ),
    )
    return DistributedPackedBatch(
        leases=PackedBatchLeaseSet(ref=ref, host_refs={"host": ref}),
        packed_group_shapes=(),
        trainable_assistant_tokens=4,
        loss_bearing_tokens=4,
        non_padding_tokens=7,
        packing_generation_id="packing",
    )


class _Runtime:
    def __init__(self) -> None:
        self.packed = _packed_batch()
        self.released: list[DistributedPackedBatch] = []

    async def pack(self, _request):
        return self.packed

    async def release_batch(self, batch):
        self.released.append(batch)


class _Trainer:
    def __init__(self) -> None:
        topology = SimpleNamespace(tp=1, cp=1, pp=1)
        self.runtime_spec = SimpleNamespace(
            packed_sequence_length=8,
            enable_moe_routing_replay=False,
            trainer_mesh=SimpleNamespace(ranks=(0,), topology=topology),
        )
        self.fail_optimizer = True

    async def forward(self, job, _batch):
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": 7,
            "executed_token_equivalents": 8,
        }

    async def forward_backward(self, job, _batch):
        return {
            "operation_id": job.operation_id,
            "learner_version": job.expected_learner_version,
            "logical_nonpadding_tokens": 7,
            "executed_token_equivalents": 8,
        }

    async def optim_step(self, job):
        if self.fail_optimizer:
            raise RuntimeError("optimizer failed")
        return {
            "operation_id": job.operation_id,
            "learner_version": job.learner_version,
            "contributing_forward_backward_operation_ids": (
                job.contributing_forward_backward_operation_ids
            ),
        }


def _operation(
    operation_id: str,
    kind: str,
    sequence_id: int,
    *,
    parent: int = 0,
    output: int | None = None,
) -> OperationRef:
    return OperationRef(
        run_id="run",
        operation_id=operation_id,
        sequence_id=sequence_id,
        learner_parent_version=parent,
        reserved_output_learner_version=output,
        kind=kind,
    )


def _batch() -> RlTrajectoryBatch:
    return RlTrajectoryBatch(
        groups=(TrajectoryGroupBundle(header=b"unused", records=()),),
        min_source_version=0,
        max_source_version=0,
    )


@pytest.mark.asyncio
async def test_handler_retains_f_b_input_until_optimizer_commit() -> None:
    runtime = _Runtime()
    trainer = _Trainer()
    handler = MegatronOperationHandler(
        runtime,  # type: ignore[arg-type]
        trainer,
        MegatronOperationConfig(
            run_id="run",
            training_session_id="session",
            source=TrainerGeneration(
                training_session_id="session",
                policy_step=0,
                generation_id=f"step-00000000-{'a' * 32}",
                adapter_path="/adapter/0",
            ),
            optimizer_state_path="/optimizer",
            rollout_model=RolloutModelSpec(payload={}),
            output_adapter_root="/adapter",
        ),
    )
    fb_request = ForwardBackwardRequest(
        run_id="run",
        request_id="fb",
        sequence_id=0,
        batch=_batch(),
        loss=LossConfig(name="cispo"),
    )
    fb = await handler(fb_request, _operation("fb", "forward_backward", 0), ())

    assert fb.packed_input_capture is not None
    assert fb.usage.logical_nonpadding_tokens.value == 7
    assert fb.usage.executed_token_equivalents.value == 8
    assert handler.retained_contribution_inputs() == (("fb", fb.packed_input_capture),)
    assert runtime.released == []

    optim_request = OptimStepRequest(
        run_id="run",
        request_id="optim",
        sequence_id=1,
        optimizer=AdamConfig(learning_rate=1e-5),
    )
    optim = _operation("optim", "optim_step", 1, output=1)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        await handler(optim_request, optim, ("fb",))
    assert handler.retained_contribution_inputs()
    assert runtime.released == []

    trainer.fail_optimizer = False
    result = await handler(optim_request, optim, ("fb",))
    assert result.checkpoint.learner_version == 1
    assert handler.retained_contribution_inputs() == ()
    assert runtime.released == [runtime.packed]

    runtime.packed = _packed_batch()
    forward_request = ForwardRequest(
        run_id="run",
        request_id="forward",
        sequence_id=2,
        batch=_batch(),
        loss=LossConfig(name="cispo"),
    )
    forward = _operation("forward", "forward", 2, parent=1)
    await handler(forward_request, forward, ())
    assert handler.retained_contribution_inputs() == ()
    await handler.release_operation_input("forward")
    assert runtime.released[-1] == runtime.packed
