from __future__ import annotations

from typing import Any

import transformers
from transformers import AutoConfig, PretrainedConfig


class Glm5NextForConditionalGeneration:
    """Dispatch marker for the pinned Bridge; does not implement HF inference."""


class Glm5NextTextOracle:
    """Dispatch marker for M1's text-only fixture checkpoint architecture."""


class Glm5NextTextConfig(PretrainedConfig):
    model_type = "glm5_next_text"

    def __init__(self, **kwargs: Any) -> None:
        # 5.12's validator predates HF's indexed_attention spelling. Keep the
        # original pattern for ART; only the HF/Bridge view uses its valid alias.
        original = kwargs.pop("art_layer_types", kwargs.get("layer_types"))
        if original is not None:
            normalized = [
                "full_attention" if kind == "indexed_attention" else kind
                for kind in original
            ]
            if kwargs.get("layer_types") not in (list(original), normalized):
                raise ValueError("Conflicting GLM attention patterns in config.")
            kwargs["layer_types"] = normalized
        # Bridge validates architecture suffixes before dispatch. M1's oracle
        # saves a custom class name, but uses the same text tensor namespace.
        # Normalize only the in-memory config; the fixture stays untouched.
        if kwargs.get("architectures") == ["Glm5NextTextOracle"]:
            kwargs["architectures"] = ["Glm5NextForConditionalGeneration"]
        super().__init__(**kwargs)
        if original is not None:
            self.art_layer_types = list(original)


class Glm5NextConfig(PretrainedConfig):
    """Config-only fallback for ART's Transformers 5.12 runtime.

    Attention reference/oracle execution requires native HF glm5_next. This shim
    only enables conversion and the ART-native provider without importing vision.
    """

    model_type = "glm5_next"
    is_composition = True

    def __init__(self, text_config: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text_config = (
            Glm5NextTextConfig(**text_config)
            if isinstance(text_config, dict)
            else text_config
        )


def ensure_glm5next_config_registered() -> None:
    def marker(name: str, model_class: type) -> None:
        objects = getattr(transformers, "_objects", None)
        if isinstance(objects, dict):
            objects[name] = model_class
        setattr(transformers, name, model_class)

    marker("Glm5NextTextOracle", Glm5NextTextOracle)
    try:
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextConfig as NativeConfig,
        )
    except ImportError:
        AutoConfig.register("glm5_next", Glm5NextConfig, exist_ok=True)
        AutoConfig.register("glm5_next_text", Glm5NextTextConfig, exist_ok=True)
        # Same dispatch-marker convention as ART's DSV4 handler. Avoid loading
        # a nonexistent HF implementation on the pinned Transformers runtime.
        marker("Glm5NextForConditionalGeneration", Glm5NextForConditionalGeneration)
    else:
        del NativeConfig
