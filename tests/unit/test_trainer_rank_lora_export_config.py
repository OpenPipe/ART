"""TrainerRank publication must prepare model-owned adapter metadata."""

from copy import deepcopy
from typing import Any

import torch

from art.megatron.glm5next.lora_export import META, to_vllm_lora_tensors
from art.megatron.model_support.handlers.glm5next import GLM5NEXT_HANDLER
from art.trainer_rank._lora_export import _prepare_vllm_lora_publish
from tests.unit.test_glm5next_lora import TINY, adapter_config, native_state


def test_trainer_publish_restores_native_glm_contract_after_serving_load():
    contract = adapter_config(8, dimensions=TINY)
    tensors = native_state()
    _, serving_config = to_vllm_lora_tensors(tensors, adapter_config=contract)
    original = deepcopy(serving_config)
    model: Any = torch.nn.Linear(1, 1)
    model._art_glm5next_adapter_config = contract
    plan = _prepare_vllm_lora_publish(
        model=[model],
        adapter_dtypes={},
        handler=GLM5NEXT_HANDLER,
        adapter_config=serving_config,
        rank=0,
        world_size=1,
        runtime=(0, torch.device("cpu")),
    )
    assert plan.adapter_config[META]["format"] == "art"
    assert plan.adapter_config[META] == contract[META]
    assert serving_config == original
    # The publisher's input tensors are native, even if the loaded config was
    # vLLM. This conversion rejected the old TrainerRank preparation result.
    _, exported = GLM5NEXT_HANDLER.to_vllm_lora_tensors(
        tensors, adapter_config=plan.adapter_config
    )
    assert exported[META]["format"] == "vllm"
    assert exported["base_model_name_or_path"] == "zai-org/GLM-5.3-Flash"
