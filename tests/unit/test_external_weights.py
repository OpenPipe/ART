"""External materialization can be validated without CUDA or Megatron imports."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "external_weights_under_test",
    Path(__file__).parents[2] / "src/art/megatron/runtime/external_weights.py",
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bind_external_weights = module.bind_external_weights


def test_parameter_metadata_and_external_alias_survive():
    model = torch.nn.Linear(3, 2, device="meta", dtype=torch.bfloat16)
    model.weight.tensor_model_parallel = True
    weight, bias = (
        torch.randn(2, 3, dtype=torch.bfloat16),
        torch.randn(2, dtype=torch.bfloat16),
    )
    assert bind_external_weights([model], {"0:weight": weight, "0:bias": bias}) == [
        model
    ]
    assert model.weight.data_ptr() == weight.data_ptr()
    assert model.bias.data_ptr() == bias.data_ptr()
    assert model.weight.tensor_model_parallel
    assert model.weight.requires_grad


@pytest.mark.parametrize("failure", ["shape", "dtype", "missing", "extra"])
def test_complete_validation_before_binding(failure):
    first = torch.nn.Linear(3, 2, device="meta")
    second = torch.nn.Linear(2, 1, device="meta")
    tensors = {
        "0:weight": torch.randn(2, 3),
        "0:bias": torch.randn(2),
        "1:weight": torch.randn(1, 2),
        "1:bias": torch.randn(1),
    }
    if failure == "shape":
        tensors["1:bias"] = torch.randn(2)
    elif failure == "dtype":
        tensors["1:bias"] = tensors["1:bias"].double()
    elif failure == "missing":
        del tensors["1:bias"]
    else:
        tensors["1:unexpected"] = torch.randn(1)
    with pytest.raises(ValueError):
        bind_external_weights([first, second], tensors)
    assert first.weight.is_meta
    assert second.weight.is_meta
