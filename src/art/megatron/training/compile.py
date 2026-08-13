from __future__ import annotations

import os
from typing import Any, cast

from megatron.core.transformer.transformer_layer import TransformerLayer
from pydantic import BaseModel, ConfigDict
import torch

from art.megatron.compile_workarounds import (
    install_torch_compile_workarounds,
    resolve_torch_compile_workaround_flags,
)
from art.megatron.model_support.spec import (
    CompileWorkaroundConfig,
    SharedExpertCompileState,
)
from art.megatron.provider import ProviderBundle, provider_runtime_env_identity
from art.megatron.training.model_chunks import ModelChunks


class TrainingCompilePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transformer_layers_compiled: bool
    workaround_flags: tuple[str, ...]
    shared_expert_state: SharedExpertCompileState
    provider_runtime: dict[str, Any]
    handler_identity: dict[str, Any]


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


def _compile_transformer_layers(module: torch.nn.Module, path: str) -> None:
    for name, child in list(module.named_children()):
        child_path = f"{path}.{name}"
        if isinstance(child, TransformerLayer):
            setattr(child, "_art_compile_cache_namespace", child_path)
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
        _compile_transformer_layers(child, child_path)


def resolve_training_compile_plan(
    *,
    provider: Any,
    provider_bundle: ProviderBundle,
) -> TrainingCompilePlan:
    workaround = provider_bundle.handler.compile_workaround_config(provider)
    transformer_layers_compiled = compile_enabled() and not workaround.disable_compile
    base_flags = (
        workaround.flags
        if transformer_layers_compiled
        else workaround.unconditional_flags
    )
    selected = resolve_torch_compile_workaround_flags(
        workaround.model_copy(update={"flags": base_flags})
    )
    return TrainingCompilePlan(
        transformer_layers_compiled=transformer_layers_compiled,
        workaround_flags=selected,
        shared_expert_state=workaround.shared_expert_state,
        provider_runtime=provider_runtime_env_identity(),
        handler_identity=provider_bundle.handler.compile_cache_identity(provider),
    )


def configure_training_compile(
    *,
    model: ModelChunks,
    provider: Any,
    provider_bundle: ProviderBundle,
    plan: TrainingCompilePlan | None = None,
) -> bool:
    plan = plan or resolve_training_compile_plan(
        provider=provider,
        provider_bundle=provider_bundle,
    )
    if plan.workaround_flags:
        install_torch_compile_workarounds(
            CompileWorkaroundConfig(
                flags=plan.workaround_flags,
                shared_expert_state=plan.shared_expert_state,
            ),
            resolved_flags=plan.workaround_flags,
        )
    if plan.transformer_layers_compiled:
        for chunk_index, chunk in enumerate(model):
            _compile_transformer_layers(chunk, f"chunk_{chunk_index}")
    return plan.transformer_layers_compiled
