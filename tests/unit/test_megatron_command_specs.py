import pytest

from art.distributed.data_plane import PackedBatchRef, TensorSpec
from art.megatron.runtime.specs import (
    CurrentTrainConfig,
    ForwardBackwardJobSpec,
    OptimizerJobSpec,
    TrainerGeneration,
)
from art.training import AdamConfig, OperationRef


def _generation(step: int) -> TrainerGeneration:
    return TrainerGeneration(
        training_session_id="session",
        policy_step=step,
        generation_id=f"step-{step:08d}-{'a' * 32}",
        adapter_path=f"/adapter/{step}",
    )


def _batch() -> PackedBatchRef:
    dtypes = {
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
    for name, (dtype, item_size) in dtypes.items():
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
    return PackedBatchRef(
        batch_id="batch",
        owner_actor_id="owner",
        lease_id="lease",
        shared_memory_name="shm",
        owner_process_id=1,
        tensors=tuple(tensors),
        num_sequences=1,
        byte_count=offset,
        storage_byte_count=offset,
        pixel_values_present=(False,),
        image_grid_thw_present=(False,),
        max_source_version=0,
        sequence_length=8,
    )


def test_forward_backward_job_binds_shifted_global_token_provenance() -> None:
    operation = OperationRef(
        run_id="run",
        operation_id="fb",
        sequence_id=0,
        learner_parent_version=0,
        kind="forward_backward",
    )
    job = ForwardBackwardJobSpec(
        operation=operation,
        training_session_id="session",
        source=_generation(0),
        optimizer_state_path="/optimizer",
        batch=_batch(),
        expected_global_loss_bearing_tokens=7,
        config=CurrentTrainConfig(),
    )

    assert job.expected_global_loss_bearing_tokens == 7
    assert job.expected_learner_version == 0
    assert job.operation_id == "fb"

    with pytest.raises(ValueError, match="forward_backward"):
        ForwardBackwardJobSpec(
            **job.model_dump(exclude={"operation"}),
            operation=operation.model_copy(update={"kind": "forward"}),
        )


def test_optimizer_job_requires_exact_next_generation_and_unique_inputs() -> None:
    operation = OperationRef(
        run_id="run",
        operation_id="optim",
        sequence_id=2,
        learner_parent_version=0,
        reserved_output_learner_version=1,
        kind="optim_step",
    )
    job = OptimizerJobSpec(
        operation=operation,
        training_session_id="session",
        generation=_generation(1),
        contributing_forward_backward_operation_ids=("fb-1", "fb-2"),
        optimizer=AdamConfig(learning_rate=1e-5),
    )

    assert job.learner_version == 1
    assert job.contributing_forward_backward_operation_ids == ("fb-1", "fb-2")

    with pytest.raises(ValueError, match="unique"):
        OptimizerJobSpec(
            **job.model_dump(exclude={"contributing_forward_backward_operation_ids"}),
            contributing_forward_backward_operation_ids=("fb-1", "fb-1"),
        )
