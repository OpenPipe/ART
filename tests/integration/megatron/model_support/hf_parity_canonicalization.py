from __future__ import annotations

import re

import torch

_FUSED_MOE_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>.*\.mlp\.experts)\.(?P<param>gate_up_proj|down_proj)(?:\.weight)?$"
)
_FUSED_NEMOTRON_H_EXPERT_PATTERN = re.compile(
    r"^(?P<prefix>backbone\.layers\.\d+\.mixer\.experts)\."
    r"(?P<param>up_proj|down_proj)(?:\.weight)?$"
)


def _strip_language_model_prefix(key: str) -> str:
    if key.startswith("model.language_model."):
        return f"model.{key.removeprefix('model.language_model.')}"
    return key


def _expected_unfused_experts_for_prefix(
    expected_keys: set[str],
    prefix: str,
    *,
    param: str,
) -> bool:
    simplified_expected_keys = {
        _strip_language_model_prefix(key) for key in expected_keys
    }
    if param == "gate_up_proj":
        return (
            f"{prefix}.0.gate_proj.weight" in simplified_expected_keys
            or f"{prefix}.0.up_proj.weight" in simplified_expected_keys
        )
    if param == "down_proj":
        return f"{prefix}.0.down_proj.weight" in simplified_expected_keys
    return False


def hf_tensor_map_to_art_canonical(
    hf_tensor_map: dict[str, torch.Tensor],
    *,
    expected_keys: set[str],
) -> dict[str, torch.Tensor]:
    canonical: dict[str, torch.Tensor] = {}
    uses_backbone_prefix = any(
        expected.startswith("backbone.") for expected in expected_keys
    )
    for key, value in hf_tensor_map.items():
        if key.startswith("model.") and uses_backbone_prefix:
            key = f"backbone.{key.removeprefix('model.')}"
        match = _FUSED_MOE_EXPERT_PATTERN.match(key)
        nemotron_h_match = _FUSED_NEMOTRON_H_EXPERT_PATTERN.match(key)
        if match is None and nemotron_h_match is not None:
            prefix = nemotron_h_match.group("prefix")
            param = nemotron_h_match.group("param")
            if value.ndim == 3 and f"{prefix}.0.{param}.weight" in expected_keys:
                for expert in range(int(value.shape[0])):
                    canonical[f"{prefix}.{expert}.{param}.weight"] = value[expert]
                continue
        if match is None:
            canonical[key] = value
            continue

        prefix = match.group("prefix")
        param = match.group("param")
        if value.ndim != 3 or not _expected_unfused_experts_for_prefix(
            expected_keys,
            prefix,
            param=param,
        ):
            canonical[key] = value
            continue

        num_experts = int(value.shape[0])
        if param == "gate_up_proj":
            if value.shape[1] % 2 != 0:
                canonical[key] = value
                continue
            gate_proj, up_proj = value.chunk(2, dim=1)
            for expert in range(num_experts):
                canonical[f"{prefix}.{expert}.gate_proj.weight"] = gate_proj[expert]
                canonical[f"{prefix}.{expert}.up_proj.weight"] = up_proj[expert]
            continue

        for expert in range(num_experts):
            canonical[f"{prefix}.{expert}.down_proj.weight"] = value[expert]

    return canonical
