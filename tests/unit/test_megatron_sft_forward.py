import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
import torch

from art.megatron.runtime.data_plane import SFTBatchData
from art.megatron.runtime.specs import (
    SFTForwardJobSpec,
    SFTForwardResult,
    TrainerGeneration,
)

LEARNER = TrainerGeneration(
    training_session_id="session",
    policy_step=3,
    generation_id="step-00000003-" + "3" * 32,
    adapter_path="/unused/3",
)


def _batch() -> SFTBatchData:
    tokens = torch.ones((1, 4), dtype=torch.long)
    return SFTBatchData(
        trajectory_tensors=(
            {"input_ids": tokens, "attention_mask": tokens, "labels": tokens},
        ),
        learning_rate=0.0,
        num_trajectories=1,
        num_tokens=4,
        num_trainable_tokens=3,
    )


def _job(**update: Any) -> SFTForwardJobSpec:
    return SFTForwardJobSpec(
        job_id="validation-1",
        run_id="run",
        learner=LEARNER,
        num_batches=2,
    ).model_copy(update=update)


def _run(forward_call: AsyncMock) -> Any:
    pytest.importorskip("monarch")
    from art.megatron.runtime.monarch import MonarchTrainerRun

    run = cast(Any, MonarchTrainerRun.__new__(MonarchTrainerRun))
    run._lock = asyncio.Lock()
    run._closed = False
    run._valid = True
    run._active_job_id = None
    run._active_collective = None
    run._active_receive = None
    run._learner_version = LEARNER.policy_step
    run.run_spec = SimpleNamespace(
        run_id="run", training_session_id="session", event_timeout_s=5
    )
    run.runtime_spec = SimpleNamespace(packed_sequence_length=1024)
    run._supervision = SimpleNamespace(wait=lambda: asyncio.Event().wait())
    run._actors = SimpleNamespace(forward_sft=SimpleNamespace(call=forward_call))
    return run


def _rank_result(rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        **SFTForwardResult(
            job_id="validation-1",
            run_id="run",
            learner=LEARNER,
            loss_sum=12.0,
            num_tokens=8,
        ).model_dump(mode="json"),
    }


async def test_forward_sft_packs_batches_and_returns_the_mesh_totals() -> None:
    forward_call = AsyncMock(return_value={1: _rank_result(1), 0: _rank_result(0)})
    run = _run(forward_call)

    with patch(
        "art.megatron.runtime.monarch.pack_sft_batches",
        side_effect=lambda batches, seq_len: ("packed", len(batches), seq_len),
    ):
        result = await run.forward_sft(_job(), (_batch(), _batch()))

    assert result.loss_sum == 12.0
    assert result.num_tokens == 8
    assert result.loss == 1.5
    assert result.learner == LEARNER
    job_json, packed = forward_call.call_args.args
    assert SFTForwardJobSpec.model_validate_json(job_json) == _job()
    assert packed == ("packed", 2, 1024)
    assert run._active_job_id is None


@pytest.mark.parametrize(
    ("job", "batch_count", "message"),
    [
        (
            _job(
                learner=LEARNER.model_copy(
                    update={
                        "policy_step": 2,
                        "generation_id": "step-00000002-" + "2" * 32,
                    }
                )
            ),
            2,
            "learner version mismatch",
        ),
        (_job(), 1, "batch count does not match"),
    ],
)
async def test_forward_sft_rejects_stale_learners_and_mismatched_payloads(
    job: SFTForwardJobSpec, batch_count: int, message: str
) -> None:
    forward_call = AsyncMock()
    run = _run(forward_call)

    with pytest.raises(ValueError, match=message):
        await run.forward_sft(job, tuple(_batch() for _ in range(batch_count)))

    forward_call.assert_not_called()
    assert run._valid


def test_forward_result_rejects_non_finite_loss() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        SFTForwardResult(
            job_id="validation-1",
            run_id="run",
            learner=LEARNER,
            loss_sum=float("inf"),
            num_tokens=1,
        )
