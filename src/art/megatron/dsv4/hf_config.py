from __future__ import annotations

import sys
from typing import Any

import transformers

_TORCHVISION_LIB: Any | None = None


class DeepseekV4ForCausalLM:
    """Bridge-dispatch marker used before native HF modeling is imported."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError(
            "DeepSeek-V4 requires native Transformers DeepseekV4ForCausalLM."
        )


def _add_marker_to_transformers_module(module: Any, model_class: type) -> None:
    if module is None:
        return
    objects = getattr(module, "_objects", None)
    if isinstance(objects, dict):
        objects["DeepseekV4ForCausalLM"] = model_class
    setattr(module, "DeepseekV4ForCausalLM", model_class)


def _ensure_transformers_marker(model_class: type | None = None) -> None:
    model_class = model_class or DeepseekV4ForCausalLM
    _add_marker_to_transformers_module(transformers, model_class)
    auto_bridge = sys.modules.get("megatron.bridge.models.conversion.auto_bridge")
    if auto_bridge is not None:
        _add_marker_to_transformers_module(
            getattr(auto_bridge, "transformers", None), model_class
        )


def _ensure_torchvision_nms_schema() -> None:
    global _TORCHVISION_LIB
    if _TORCHVISION_LIB is not None:
        return
    import torch

    try:
        _TORCHVISION_LIB = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_LIB.define(
            "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
        )
    except RuntimeError as exc:
        if "Only a single TORCH_LIBRARY" not in str(exc) and "already" not in str(exc):
            raise
        _TORCHVISION_LIB = torch.library.Library("torchvision", "FRAGMENT")
        try:
            _TORCHVISION_LIB.define(
                "nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor"
            )
        except RuntimeError as define_exc:
            if "already" not in str(define_exc):
                raise


def _native_dsv4_config_class() -> type:
    try:
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
            DeepseekV4Config,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DeepSeek-V4 requires transformers with native deepseek_v4 support."
        ) from exc
    return DeepseekV4Config


def _native_dsv4_model_class() -> type:
    _ensure_torchvision_nms_schema()
    try:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4ForCausalLM as NativeDeepseekV4ForCausalLM,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DeepSeek-V4 requires transformers with native DeepseekV4ForCausalLM."
        ) from exc
    return NativeDeepseekV4ForCausalLM


def ensure_dsv4_hf_config_registered() -> None:
    _native_dsv4_config_class()
    _ensure_transformers_marker()


def ensure_dsv4_hf_model_registered() -> None:
    _native_dsv4_config_class()
    _ensure_transformers_marker(_native_dsv4_model_class())


__all__ = [
    "DeepseekV4ForCausalLM",
    "ensure_dsv4_hf_config_registered",
    "ensure_dsv4_hf_model_registered",
]
