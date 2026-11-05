from pathlib import Path

from huggingface_hub.errors import StrictDataclassClassValidationError
import pytest
from transformers import PretrainedConfig

from art.megatron.glm5next.hf_config import Glm5NextTextConfig
from tests.integration.megatron.glm5next_real_config import check_real_config


@pytest.mark.parametrize(
    "name,expected",
    [("full", list(range(3, 45, 4))), ("F4", [3]), ("F8", [3, 7]), ("R2", [1])],
)
def test_real_native_config_without_weights(name: str, expected: list[int]) -> None:
    path = (
        Path(__file__).parent / "fixtures" / "glm5next_configs" / name / "config.json"
    )
    assert check_real_config(path)["dsa_layers"] == expected


def test_alias_roundtrip_and_global_validation_remains_strict() -> None:
    config = Glm5NextTextConfig(num_hidden_layers=1, layer_types=["indexed_attention"])
    assert config.layer_types == ["full_attention"]
    assert config.art_layer_types == ["indexed_attention"]
    restored = Glm5NextTextConfig.from_dict(config.to_dict())
    assert restored.art_layer_types == ["indexed_attention"]
    with pytest.raises(StrictDataclassClassValidationError, match="layer_types"):
        PretrainedConfig.from_dict(
            {"num_hidden_layers": 1, "layer_types": ["indexed_attention"]}
        )
    with pytest.raises(StrictDataclassClassValidationError, match="layer_types"):
        Glm5NextTextConfig(num_hidden_layers=1, layer_types=["unknown_attention"])
    with pytest.raises(ValueError, match="Conflicting"):
        Glm5NextTextConfig(
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            art_layer_types=["indexed_attention"],
        )
