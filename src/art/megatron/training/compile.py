from __future__ import annotations

import os
from typing import Any, cast

from megatron.core.transformer.transformer_layer import TransformerLayer
import torch
from torch._dynamo import config as dynamo_config

from art.megatron.compile_workarounds import install_torch_compile_workarounds
from art.megatron.provider import ProviderBundle
from art.megatron.training.model_chunks import ModelChunks

_DYNAMO_CONFIG = cast(Any, dynamo_config)


def _configure_dynamo() -> None:
    """Set the process-wide Dynamo policy required by dynamic LoRA slots."""
    # Dynamic checkpoint slots register differently shaped projection parameters
    # behind one LoRA.forward code object. Let automatic dynamic shapes generalize
    # those parameter dimensions instead of compiling once per projection site.
    _DYNAMO_CONFIG.force_parameter_static_shapes = False


def compile_enabled() -> bool:
    return os.environ.get("ART_DISABLE_MEGATRON_COMPILE", "0") in {
        "0",
        "false",
        "False",
    }


def _set_child_module(
    parent: torch.nn.Module,
    name: str,
    child: torch.nn.Module,
) -> None:
    if isinstance(parent, torch.nn.ModuleList | torch.nn.Sequential):
        parent[int(name)] = child
        return
    setattr(parent, name, child)


def _compile_transformer_layers(module: torch.nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, TransformerLayer):
            physical_forward = getattr(child, "_art_gdn_island_physical_forward", None)
            if callable(physical_forward):
                setattr(
                    child,
                    "_art_gdn_island_physical_forward",
                    torch.compile(physical_forward),
                )
                continue
            compiled_child = cast(torch.nn.Module, torch.compile(child))
            _set_child_module(parent=module, name=name, child=compiled_child)
            continue
        _compile_transformer_layers(child)


def configure_training_compile(
    *,
    model: ModelChunks,
    provider: Any,
    provider_bundle: ProviderBundle,
) -> bool:
    compile_workaround_config = provider_bundle.handler.compile_workaround_config(
        provider
    )
    enabled = compile_enabled()
    flags = (
        compile_workaround_config.flags
        if enabled and not compile_workaround_config.disable_compile
        else compile_workaround_config.unconditional_flags
    )
    if flags:
        install_torch_compile_workarounds(
            compile_workaround_config.model_copy(update={"flags": flags})
        )
    transformer_layers_compiled = (
        enabled and not compile_workaround_config.disable_compile
    )
    if transformer_layers_compiled:
        _configure_dynamo()
        for chunk in model:
            _compile_transformer_layers(chunk)
    return transformer_layers_compiled
