from __future__ import annotations

import re
from typing import Any

import torch

from art.megatron.model_support.spec import HfWeightSource


def hf_weight_source(name: str) -> HfWeightSource:
    """Support both released checkpoint names and the native HF oracle layout."""
    alternatives: list[tuple[str, ...]] = [(name,)]
    for hf, raw in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        for field in ("fn", "base", "scale"):
            if name.endswith(f".{hf}.{field}"):
                alternatives.append(
                    (name.removesuffix(f".{hf}.{field}") + f".{raw}_{field}",)
                )
            if name.endswith(f".{raw}_{field}"):
                alternatives.append(
                    (name.removesuffix(f".{raw}_{field}") + f".{hf}.{field}",)
                )
    if ".self_attn.forget_gate." in name:
        alternatives.append((name.replace(".self_attn.forget_gate.", ".self_attn."),))
    elif ".self_attn." in name:
        prefix, tail = name.rsplit(".self_attn.", 1)
        if tail in {"f_a_proj.weight", "f_b_proj.weight", "A_log", "dt_bias"}:
            alternatives.append((f"{prefix}.self_attn.forget_gate.{tail}",))
    for part in ("q", "k", "v"):
        if name.endswith(f".self_attn.{part}_conv1d.weight"):
            alternatives.append(
                (name.removesuffix(f"{part}_conv1d.weight") + "conv1d.weight",)
            )
    if name.endswith(".self_attn.conv1d.weight"):
        prefix = name.removesuffix("conv1d.weight")
        alternatives.append(
            tuple(f"{prefix}{part}_conv1d.weight" for part in ("q", "k", "v"))
        )
    match = re.fullmatch(
        r"(.+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name
    )
    if match:
        prefix, _, projection = match.groups()
        packed = "down_proj" if projection == "down_proj" else "gate_up_proj"
        alternatives.append((f"{prefix}.{packed}",))
    return HfWeightSource(
        logical_key=name,
        physical_key_options=tuple(alternatives),
        kind="bridge_materialized",
    )


def materialize_hf_weight(name: str, state: Any) -> torch.Tensor:
    source = hf_weight_source(name)
    for option in source.physical_key_options:
        if not all(key in state for key in option):
            continue
        if option == (name,):
            return state[name]
        if len(option) == 3:
            return torch.cat([state[key] for key in option], dim=0)
        tensor = state[option[0]]
        for index, part in enumerate(("q", "k", "v")):
            if name.endswith(f".self_attn.{part}_conv1d.weight"):
                return tensor.chunk(3, dim=0)[index]
        match = re.fullmatch(
            r"(.+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name
        )
        if match:
            _, expert, projection = match.groups()
            tensor = tensor[int(expert)]
            if projection != "down_proj":
                tensor = tensor.chunk(2, dim=0)[int(projection == "up_proj")]
        return tensor
    raise KeyError(
        f"No BF16 source for GLM tensor {name!r}: {source.physical_key_options}"
    )
