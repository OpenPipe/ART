from collections import Counter

import torch

from art.megatron.model_support.spec import ArchitectureReport, LayerFamilyInstance
from art.megatron.provider import get_provider_bundle


def summarize_layer_families(
    layer_families: list[LayerFamilyInstance],
) -> list[LayerFamilyInstance]:
    counts = Counter(family.key for family in layer_families)
    exemplar_by_key: dict[str, LayerFamilyInstance] = {}
    for family in layer_families:
        exemplar_by_key.setdefault(family.key, family)
    return [
        LayerFamilyInstance(
            key=key,
            count=count,
            layer_index=exemplar_by_key[key].layer_index,
            module_path=exemplar_by_key[key].module_path,
            module_type=exemplar_by_key[key].module_type,
        )
        for key, count in sorted(counts.items())
    ]


def recommended_min_layers(
    layer_families: list[LayerFamilyInstance],
) -> int:
    indexed_layers = [
        family.layer_index
        for family in layer_families
        if family.layer_index is not None
    ]
    if indexed_layers:
        return max(indexed_layers) + 1
    return max(len(layer_families), 1)


def inspect_architecture(
    base_model: str,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    allow_unvalidated_arch: bool = False,
) -> ArchitectureReport:
    provider_bundle = get_provider_bundle(
        base_model,
        torch_dtype=torch_dtype,
        load_weights=False,
        allow_unvalidated_arch=allow_unvalidated_arch,
    )
    discovered = provider_bundle.handler.collect_layer_families(
        provider_bundle.provider
    )
    summarized = summarize_layer_families(discovered)
    unresolved_risks: list[str] = []
    if not summarized:
        unresolved_risks.append(
            "handler did not report any layer families; codex review is required"
        )
    if any(family.layer_index is None for family in summarized):
        unresolved_risks.append(
            "handler did not report representative layer indices for every family; "
            "codex review is required"
        )
    return ArchitectureReport(
        base_model=base_model,
        model_key=provider_bundle.spec.key,
        handler_key=provider_bundle.handler.key,
        bridge_type=type(provider_bundle.bridge._model_bridge).__name__,
        provider_type=type(provider_bundle.provider).__name__,
        layer_families=summarized,
        recommended_min_layers=recommended_min_layers(summarized),
        unresolved_risks=unresolved_risks,
    )
