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
    setattr(model.weight, "tensor_model_parallel", True)
    weight, bias = (
        torch.randn(2, 3, dtype=torch.bfloat16),
        torch.randn(2, dtype=torch.bfloat16),
    )
    assert bind_external_weights([model], {"0:weight": weight, "0:bias": bias}) == [
        model
    ]
    assert model.weight.data_ptr() == weight.data_ptr()
    assert model.bias.data_ptr() == bias.data_ptr()
    assert getattr(model.weight, "tensor_model_parallel")
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


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
@pytest.mark.parametrize("replacement", ["rebind", "data"])
def test_validation_rejects_replaced_registered_storage(kind, replacement):
    model = torch.nn.Module()
    if kind == "parameter":
        model.register_parameter(
            "weight", torch.nn.Parameter(torch.empty(3, device="meta"))
        )
    else:
        model.register_buffer("weight", torch.empty(3, device="meta"))
    source = torch.randn(3)
    bind_external_weights([model], {"0:weight": source})
    module.validate_external_weights([model])
    if replacement == "rebind":
        if kind == "parameter":
            model.weight = torch.nn.Parameter(source.clone())
        else:
            model.weight = source.clone()
    else:
        model.weight.data = source.clone()
    with pytest.raises(RuntimeError, match="external base"):
        module.validate_external_weights([model])


def test_buffers_and_empty_extra_state_are_bound_and_validated():
    model = torch.nn.Module()
    model.register_buffer("state", torch.empty(3, device="meta"))
    source = torch.randn(3)
    bind_external_weights(
        [model], {"0:state": source, "0:layer._extra_state": torch.empty(0)}
    )
    assert model.get_buffer("state").data_ptr() == source.data_ptr()
    module.validate_external_weights([torch.nn.Sequential(model)])


def test_validation_survives_ddp_style_buffers_attribute():
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
            # MCore DDP shadows Module.buffers with its grad-buffer list.
            object.__setattr__(self, "buffers", [])

    model = torch.nn.Module()
    model.register_buffer("state", torch.empty(3, device="meta"))
    source = torch.randn(3)
    bind_external_weights([model], {"0:state": source})
    module.validate_external_weights([Wrapper(model)])
    model.register_buffer("state", source.clone())
    with pytest.raises(RuntimeError, match="external base"):
        module.validate_external_weights([Wrapper(model)])
