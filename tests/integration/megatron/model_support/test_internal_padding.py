from types import SimpleNamespace
from typing import Any

import pytest
import torch

from art.megatron.model_support.handlers.gemma4 import (
    _GEMMA4_LOGICAL_MOE_FFN_ATTR,
    GEMMA4_MOE_HANDLER,
)
from art.megatron.model_support.handlers.gpt_oss import (
    _GPT_OSS_INTERNAL_HIDDEN_ATTR,
    _GPT_OSS_INTERNAL_MOE_FFN_ATTR,
    _GPT_OSS_LOGICAL_HIDDEN_ATTR,
    _GPT_OSS_LOGICAL_MOE_FFN_ATTR,
    GPT_OSS_MOE_HANDLER,
)


class _Lora(torch.nn.Module):
    def __init__(
        self,
        suffix: str,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.adapter_model_prefix = (
            "base_model.model.model.layers.0.mlp.experts.{expert}." + suffix
        )
        self.A_T = self._parameter(a_shape)
        self.B_T = self._parameter(b_shape)
        self._slot_modules = torch.nn.ModuleDict(
            {"checkpoint": _LoraSlot(a_shape, b_shape)}
        )

    @staticmethod
    def _parameter(shape: tuple[int, ...]) -> torch.nn.Parameter:
        parameter = torch.nn.Parameter(torch.ones(shape))
        parameter.grad = torch.ones_like(parameter)
        setattr(parameter, "main_grad", torch.ones_like(parameter))
        setattr(parameter, "lora_tp_sharded", False)
        setattr(parameter, "lora_shard_domain", "expert_tp")
        return parameter


class _LoraSlot(torch.nn.Module):
    def __init__(self, a_shape: tuple[int, ...], b_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.A_T = _Lora._parameter(a_shape)
        self.B_T = _Lora._parameter(b_shape)


class _Chunk(torch.nn.Module):
    def __init__(
        self,
        config: dict[str, int],
        gate_shapes: tuple[tuple[int, ...], tuple[int, ...]],
        down_shapes: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(**config)
        self.gate_up = _Lora("gate_up_proj", *gate_shapes)
        self.down = _Lora("down_proj", *down_shapes)


_GEMMA_CONFIG = {
    "num_moe_experts": 2,
    "moe_ffn_hidden_size": 128,
    _GEMMA4_LOGICAL_MOE_FFN_ATTR: 4,
}
_GPT_OSS_CONFIG = {
    "num_moe_experts": 2,
    "hidden_size": 4,
    "moe_ffn_hidden_size": 6,
    _GPT_OSS_LOGICAL_HIDDEN_ATTR: 4,
    _GPT_OSS_INTERNAL_HIDDEN_ATTR: 128,
    _GPT_OSS_LOGICAL_MOE_FFN_ATTR: 6,
    _GPT_OSS_INTERNAL_MOE_FFN_ATTR: 128,
}


@pytest.mark.parametrize(
    "handler,chunk,padding",
    [
        pytest.param(
            GEMMA4_MOE_HANDLER,
            _Chunk(_GEMMA_CONFIG, ((2, 3, 2), (2, 2, 256)), ((2, 128, 2), (2, 2, 5))),
            (
                ("gate_up", "B_T", -1, ((4, 128), (132, 256))),
                ("down", "A_T", -2, ((4, 128),)),
            ),
            id="gemma4",
        ),
        pytest.param(
            GPT_OSS_MOE_HANDLER,
            _Chunk(
                _GPT_OSS_CONFIG, ((2, 128, 2), (2, 2, 256)), ((2, 128, 2), (2, 2, 128))
            ),
            (
                ("gate_up", "A_T", -2, ((4, 128),)),
                ("gate_up", "B_T", -1, ((6, 128), (134, 256))),
                ("down", "A_T", -2, ((6, 128),)),
                ("down", "B_T", -1, ((4, 128),)),
            ),
            id="gpt_oss",
        ),
    ],
)
def test_internal_padding_is_zeroed(
    handler: Any,
    chunk: _Chunk,
    padding: tuple[tuple[str, str, int, tuple[tuple[int, int], ...]], ...],
) -> None:
    handler.zero_internal_padding_grads([chunk])
    handler.zero_internal_padding_params([chunk])

    for module_name, parameter_name, dim, ranges in padding:
        module = getattr(chunk, module_name)
        parameters = (
            getattr(module, parameter_name),
            getattr(module._slot_modules["checkpoint"], parameter_name),
        )
        for parameter in parameters:
            for tensor in (parameter, parameter.grad, parameter.main_grad):
                assert torch.count_nonzero(tensor) > 0
                for start, end in ranges:
                    assert (
                        torch.count_nonzero(tensor.narrow(dim, start, end - start)) == 0
                    )
