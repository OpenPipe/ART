"""Opt-in real-weight CPU gate: F4 HF merged weights versus low-rank branches.

Run with ART_GLM5NEXT_F4=1 in M1's Transformers 5.20.2 environment. Before the
oracle branch is integrated, ART_GLM5NEXT_ORACLE_PATH can name its hf_oracle.py.
"""

import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from art.megatron.glm5next.lora import ART_PREFIX, LoraDimensions, lora_targets
from art.megatron.glm5next.lora_export import adapter_config, to_vllm_lora_tensors


def _oracle():
    if path := os.environ.get("ART_GLM5NEXT_ORACLE_PATH"):
        spec = importlib.util.spec_from_file_location("glm_m1_oracle", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module("art.megatron.glm5next.hf_oracle")


class _ExpertBranches(torch.nn.Module):
    def __init__(self, base: Any, factors: dict[str, Any], scale: float):
        super().__init__()
        self.base = base
        self.factors = factors
        self.scale = scale
        self.calls = 0

    def delta(self, x, name, expert):
        a = self.factors[name + ".lora_A.weight"]
        b = self.factors[name + ".lora_B.weight"]
        a, b = a[0 if a.shape[0] == 1 else expert], b[0 if b.shape[0] == 1 else expert]
        return F.linear(F.linear(x, a), b) * self.scale

    def forward(self, hidden_states, top_k_index, top_k_weights):
        self.calls += 1
        base = self.base
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=base.num_experts + 1).permute(2, 1, 0)
        for index in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            expert = int(index[0])
            if expert == base.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert])
            x = hidden_states[token_idx]
            gate_up = F.linear(x, base.gate_up_proj[expert])
            gate_up = gate_up + torch.cat(
                (self.delta(x, "w1", expert), self.delta(x, "w3", expert)), dim=-1
            )
            current = base._apply_gate(gate_up)
            current = F.linear(current, base.down_proj[expert]) + self.delta(
                current, "w2", expert
            )
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def test_f4_bf16_logits_merged_against_adapter():
    if os.environ.get("ART_GLM5NEXT_F4") != "1":
        pytest.skip("Opt-in 19 GB real-weight F4 CPU gate")
    oracle = _oracle()
    torch.set_num_threads(8)
    model = oracle.load_fixture("F4", dtype=torch.bfloat16)
    model.requires_grad_(False)
    dims = LoraDimensions.from_config(model.config)
    assert dims.layers == 4 and dims.experts == 288 and dims.dense_layers == 3
    rng = torch.Generator().manual_seed(20261105)
    state = {}
    targets = lora_targets(dims)
    rank, scale = 8, 4.0
    for t in targets:
        pairs = [
            torch.randn(s, generator=rng, dtype=torch.bfloat16) * 0.01
            for s in t.shapes(rank)
        ]
        for e in range(t.experts or 1):
            for factor, tensor in zip(("A", "B"), pairs, strict=True):
                value = tensor[0 if t.shared == factor else e] if t.experts else tensor
                state[t.name.format(expert=e) + f".lora_{factor}.weight"] = value
    serving, _ = to_vllm_lora_tensors(
        state, adapter_config=adapter_config(rank, dimensions=dims)
    )
    hooks = []
    experts = []
    for t in targets:
        if t.experts:
            continue
        path = t.name.replace(ART_PREFIX, "model.language_model.layers.").removeprefix(
            "base_model.model."
        )
        module = model.get_submodule(path)
        a, b = (state[t.name + f".lora_{factor}.weight"] for factor in ("A", "B"))

        def hook(_module, args, output, a=a, b=b):
            return output + scale * F.linear(F.linear(args[0], a), b)

        hooks.append(module.register_forward_hook(hook))
    for layer in range(dims.dense_layers, dims.layers):
        mlp = model.model.language_model.layers[layer].mlp
        prefix = f"base_model.model.language_model.model.layers.{layer}.mlp.experts."
        factors = {
            k.removeprefix(prefix): v
            for k, v in serving.items()
            if k.startswith(prefix)
        }
        wrapper = _ExpertBranches(mlp.experts, factors, scale)
        mlp.experts = wrapper
        experts.append((mlp, wrapper))
    ids = torch.tensor([[1, 42, 17, 233]])
    print("F4 rank-8 random A/B: low-rank forward", flush=True)
    with torch.inference_mode(), oracle.cpu_reference_kernels():
        adapted = model(ids, use_cache=False).logits.clone()
    assert all(wrapper.calls == 1 for _, wrapper in experts)
    for handle in hooks:
        handle.remove()
    for mlp, wrapper in experts:
        mlp.experts = wrapper.base
    print("F4: merging all targets including all 288 experts", flush=True)
    with torch.no_grad():
        for t in targets:
            if not t.experts:
                path = t.name.replace(
                    ART_PREFIX, "model.language_model.layers."
                ).removeprefix("base_model.model.")
                weight = model.get_submodule(path).weight
                a, b = (
                    state[t.name + f".lora_{factor}.weight"] for factor in ("A", "B")
                )
                weight.copy_(
                    (weight.float() + scale * b.float() @ a.float()).to(weight.dtype)
                )
                continue
            prefix, projection = t.name.split(".{expert}.")
            base = model.get_submodule(
                prefix.replace(ART_PREFIX, "model.language_model.layers.")
            )
            for e in range(t.experts):
                a, b = (
                    state[t.name.format(expert=e) + f".lora_{factor}.weight"]
                    for factor in ("A", "B")
                )
                if projection == "down_proj":
                    weight = base.down_proj[e]
                else:
                    row = 0 if projection == "gate_proj" else dims.expert_width
                    weight = base.gate_up_proj[e, row : row + dims.expert_width]
                weight.copy_(
                    (weight.float() + scale * b.float() @ a.float()).to(weight.dtype)
                )
    with torch.inference_mode(), oracle.cpu_reference_kernels():
        merged = model(ids, use_cache=False).logits
    metrics = oracle.tensor_metrics(adapted, merged)
    print(
        json.dumps(
            {
                "fixture": "F4",
                "rank": rank,
                "alpha": 32,
                "seed": 20261105,
                "input_ids": ids.tolist(),
                "dtype": "bfloat16",
                **metrics,
            }
        ),
        flush=True,
    )
    if report := os.environ.get("ART_GLM5NEXT_F4_REPORT"):
        Path(report).write_text(json.dumps(metrics, indent=2) + "\n")
    assert metrics["rel_l2"] <= 1e-2, metrics
