from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from art import Model
from art.utils.benchmarking.log_constant_metrics_wandb import (
    log_constant_metrics_wandb,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("entity", ["test-entity", None])
async def test_log_constant_metrics_wandb_uses_model_entity(
    tmp_path: Path, entity: str | None
) -> None:
    fake_run = MagicMock()
    fake_init = MagicMock(return_value=fake_run)

    model = Model(
        name="test-model",
        project="test-project",
        entity=entity,
        base_path=str(tmp_path),
    )

    with patch(
        "art.utils.benchmarking.log_constant_metrics_wandb.wandb_sdk.init",
        fake_init,
    ):
        await log_constant_metrics_wandb(
            model,
            num_steps=1,
            split_metrics={"val": {"reward": 0.5}},
        )

    init_kwargs = fake_init.call_args.kwargs
    assert init_kwargs["entity"] == entity
    assert init_kwargs["project"] == "test-project"
    assert fake_run.log.call_count == 2
    fake_run.finish.assert_called_once()
