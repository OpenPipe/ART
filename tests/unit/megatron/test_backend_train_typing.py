from typing import assert_type

from art.backend import AnyTrainableModel
from art.distill import Loss, PreparedTrainingBatch, TrainingObjectives
from art.megatron.backend import MegatronBackend
from art.trajectories import TrajectoryGroup
from art.types import LocalTrainResult


async def _valid_prepared_calls(
    backend: MegatronBackend,
    model: AnyTrainableModel,
    batch: PreparedTrainingBatch,
    session: object,
) -> None:
    standalone = await backend.train(
        model,
        batch,
        objectives=TrainingObjectives(distillation=Loss()),
        idempotency_key="standalone",
    )
    assert_type(standalone, LocalTrainResult)

    additive = await backend.train(
        model,
        batch,
        objectives=TrainingObjectives(
            policy="cispo",
            distillation=Loss(coefficient=0.5),
        ),
        idempotency_key="additive",
        learning_rate=1e-5,
        grad_accumulation_sequences=2,
        optimizer_save_interval=1,
        save_checkpoint=False,
        verbose=True,
        epsilon=1.0,
        epsilon_high=4.0,
        importance_sampling_level="token",
        scale_rewards=False,
        advantage_balance=0.25,
        session=session,
    )
    assert_type(additive, LocalTrainResult)


async def _valid_legacy_rl_call(
    backend: MegatronBackend,
    model: AnyTrainableModel,
    groups: list[TrajectoryGroup],
) -> None:
    result = await backend.train(
        model,
        groups,
        learning_rate=1e-5,
        loss_fn="cispo",
        importance_sampling_level="sequence",
    )
    assert_type(result, LocalTrainResult)


def test_typing_fixture_is_importable() -> None:
    assert callable(_valid_prepared_calls)
    assert callable(_valid_legacy_rl_call)
