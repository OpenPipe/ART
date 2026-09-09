import pytest
import torch

from art.megatron.lora import LoRA


def test_load_lora_treats_absent_site_as_identity() -> None:
    module = LoRA(
        "base_model.model.foo",
        in_features=3,
        out_features=5,
        rank=2,
        alpha=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    adapter = {
        "base_model.model.foo.lora_A.weight": torch.ones(2, 3),
        "base_model.model.foo.lora_B.weight": torch.ones(5, 2),
    }
    x = torch.ones(4, 3)

    module.load_lora(adapter)
    assert module(x).abs().sum() > 0

    module.load_lora({})
    assert torch.count_nonzero(module.B_T) == 0
    assert torch.allclose(module(x), torch.zeros(4, 5))


def test_load_lora_rejects_partially_present_site() -> None:
    module = LoRA(
        "base_model.model.foo",
        in_features=3,
        out_features=5,
        rank=2,
        alpha=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    with pytest.raises(KeyError, match="Incomplete LoRA adapter keys"):
        module.load_lora({"base_model.model.foo.lora_A.weight": torch.ones(2, 3)})
