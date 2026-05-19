from datetime import datetime, timezone

import pytest

from art.pipeline_trainer import CheckpointInfo, CheckpointRetentionContext
from art.pipeline_trainer.checkpoint_retention import keep_recent_and_top


def _checkpoint(
    step: int,
    *,
    is_eval_step: bool = False,
    reward: float | None = None,
) -> CheckpointInfo:
    metrics = {"val/reward": reward} if reward is not None else {}
    return CheckpointInfo(
        step=step,
        path=f"/tmp/checkpoints/{step:04d}",
        created_at=datetime.fromtimestamp(step, timezone.utc),
        is_eval_step=is_eval_step,
        metrics=metrics,
    )


def test_keep_recent_and_top_returns_kept_steps() -> None:
    strategy = keep_recent_and_top(recent=2, top=1, metric="val/reward")
    context = CheckpointRetentionContext(
        current_step=6,
        checkpoints=[
            _checkpoint(0),
            _checkpoint(1, is_eval_step=True, reward=0.2),
            _checkpoint(2),
            _checkpoint(3, is_eval_step=True, reward=0.8),
            _checkpoint(4),
            _checkpoint(5),
        ],
    )

    assert set(strategy(context)) == {3, 4, 5}


def test_keep_recent_and_top_uses_metric_presence_for_legacy_history() -> None:
    strategy = keep_recent_and_top(recent=0, top=1, metric="val/reward")
    context = CheckpointRetentionContext(
        current_step=3,
        checkpoints=[
            _checkpoint(0),
            _checkpoint(1, reward=0.9),
            _checkpoint(2, reward=0.2),
        ],
    )

    assert set(strategy(context)) == {1}


def test_keep_recent_and_top_handles_zero_limits() -> None:
    strategy = keep_recent_and_top(recent=0, top=0)
    context = CheckpointRetentionContext(
        current_step=3,
        checkpoints=[_checkpoint(0), _checkpoint(1), _checkpoint(2)],
    )

    assert set(strategy(context)) == set()


@pytest.mark.parametrize(
    ("recent", "top", "match"),
    [(-1, 0, "recent must be >= 0"), (0, -1, "top must be >= 0")],
)
def test_keep_recent_and_top_rejects_negative_limits(
    recent: int,
    top: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        keep_recent_and_top(recent=recent, top=top)
