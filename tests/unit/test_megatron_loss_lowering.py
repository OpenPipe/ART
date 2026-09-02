from types import SimpleNamespace
from typing import cast

from art.megatron.operation_handler import _experimental_config
from art.training import ForwardBackwardRequest, LossConfig


def test_route_backed_cispo_preserves_normalization_and_absolute_clip_plan() -> None:
    request = cast(
        ForwardBackwardRequest,
        SimpleNamespace(
            loss=LossConfig(
                name="cispo",
                normalize_advantages=False,
                values={
                    "clip_low_threshold": 0.0,
                    "clip_high_threshold": 4.0,
                },
            )
        ),
    )

    config = _experimental_config(request)

    assert config.scale_rewards is False
    assert config.epsilon == 1.0
    assert config.epsilon_high == 3.0
