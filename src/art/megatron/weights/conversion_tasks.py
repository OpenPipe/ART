from itertools import chain
from typing import Any, cast

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.param_mapping import (
    extract_expert_number_from_param,
)
import torch

from art.megatron.expert_parallel import get_expert_parallel_layout
from art.megatron.runtime.bridge_runtime import (
    _logical_hf_param,
)
from art.megatron.training.model_chunks import ModelChunks, as_megatron_api_chunks
from art.megatron.weights.param_name_canonicalization import (
    canonical_art_param_name,
    is_art_adapter_param_name,
)


def _hf_param_names(hf_param: Any) -> list[str]:
    if isinstance(hf_param, str):
        return [hf_param]
    return list(hf_param.values())


def _checkpoint_hf_param_names(mapping: Any, model_config: Any) -> list[str]:
    layout = get_expert_parallel_layout(model_config)
    if layout is None or not bool(getattr(mapping, "is_expert", False)):
        return _hf_param_names(mapping.hf_param)
    physical_expert = extract_expert_number_from_param(mapping.megatron_param)
    logical_expert = layout.logical_expert(physical_expert)
    if logical_expert is None:
        return []
    return _hf_param_names(
        _logical_hf_param(
            mapping.hf_param,
            physical_expert=physical_expert,
            logical_expert=logical_expert,
        )
    )


def build_art_conversion_tasks(*, bridge: AutoBridge, model: ModelChunks) -> list[Any]:
    from megatron.bridge.models.conversion.model_bridge import (
        WeightConversionTask,
        _megatron_local_name_to_global,
    )
    from megatron.bridge.models.conversion.utils import (
        get_module_and_param_from_name,
        persistent_buffers,
    )

    mapping_registry = bridge._model_bridge.mapping_registry()
    hf_source = bridge.hf_pretrained.state.source
    hf_keys = set(hf_source.get_all_keys())
    megatron_models = as_megatron_api_chunks(model)
    model_config = getattr(model[0], "config")
    tasks: list[Any] = []
    for vp_stage, chunk in enumerate(model):
        for local_name, _ in chain(
            chunk.named_parameters(),
            persistent_buffers(chunk),
        ):
            if "_extra_state" in local_name or is_art_adapter_param_name(local_name):
                continue
            global_name = _megatron_local_name_to_global(
                megatron_models,
                model_config,
                canonical_art_param_name(local_name),
                vp_stage,
            )
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if mapping is None:
                raise RuntimeError(
                    f"Missing HF conversion mapping for Megatron param {global_name}"
                )
            hf_params = _checkpoint_hf_param_names(mapping, model_config)
            missing_hf_params = sorted(set(hf_params) - hf_keys)
            if missing_hf_params and not getattr(
                mapping,
                "allow_hf_name_mismatch",
                False,
            ):
                raise RuntimeError(
                    f"Missing HF checkpoint weights for Megatron param {global_name}: "
                    f"{missing_hf_params}"
                )
            local_module, local_weights = cast(
                tuple[Any, torch.Tensor],
                get_module_and_param_from_name(
                    megatron_models,
                    local_name,
                    vp_stage,
                ),
            )
            if local_module is not None and not hasattr(local_module, "config"):
                setattr(local_module, "config", model_config)
            tasks.append(
                WeightConversionTask(
                    pp_rank=0,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )
            )
    return tasks
