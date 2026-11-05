"""Opt-in CPU checks against an actual vLLM v0.32.0 source checkout.

ART_GLM5NEXT_VLLM_ROOT points to that checkout. Execute only the real loader
weight classes; CUDA attention/kernel imports are unnecessary for these checks.
"""

import ast
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def source_root():
    path = os.environ.get("ART_GLM5NEXT_VLLM_ROOT")
    if not path:
        pytest.skip("Set ART_GLM5NEXT_VLLM_ROOT to the pinned vLLM source checkout")
    return Path(path)


def assignment_call(root, file, name):
    tree = ast.parse((root / file).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == name for t in node.targets
        ):
            if isinstance(node.value, ast.Call):
                return node.value
    raise AssertionError(f"No constructor for {name}")


def test_real_vllm_constructor_shapes_and_packing(source_root):
    root = source_root
    self = SimpleNamespace(
        hidden_size=4096,
        num_heads=64,
        head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
    )
    context = {"self": self, "projection_size": 8192, "proj_input_size": 4096}

    def value(node):
        return eval(
            compile(ast.Expression(node), "vllm-constructor", "eval"),
            {"__builtins__": {}},
            context,
        )

    kda = assignment_call(
        root, "vllm/models/glm5next/common/kda.py", "in_proj_qkvbfg_a"
    )
    assert value(kda.args[0]) == 4096
    assert value(kda.args[1]) == [8192, 8192, 8192, 64, 128, 128]
    assert value(
        next(k.value for k in kda.keywords if k.arg == "replicated_shard_ids")
    ) == (4, 5)
    mla = assignment_call(
        root, "vllm/models/glm5next/common/attention.py", "fused_qkv_a_proj"
    )
    assert value(mla.args[0]) == 4096
    assert value(mla.args[1]) == [1536, 512]
    head = assignment_call(root, "vllm/models/glm5next/common/model.py", "lm_head")
    self.config = SimpleNamespace(vocab_size=154880, hidden_size=4096)
    assert [value(v) for v in head.args[:2]] == [154880, 4096]
    utils = (root / "vllm/lora/utils.py").read_text()
    assert (
        '"experts.w1"' in utils and '"experts.w2"' in utils and '"experts.w3"' in utils
    )
    model = (root / "vllm/models/glm5next/common/model.py").read_text()
    assert 'prefix=maybe_prefix(prefix, "language_model")' in model


def test_real_vllm_stacked_expert_loader_scale(source_root):
    tree = ast.parse((source_root / "vllm/lora/lora_weights.py").read_text())
    classes: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    namespace: dict[str, Any] = {
        "torch": torch,
        "GenericSequence": Sequence,
        "dataclass": dataclass,
        "PEFTHelper": object,
        "PIN_MEMORY": False,
    }
    exec(
        compile(
            ast.Module(body=classes, type_ignores=[]),
            "vllm/lora/lora_weights.py",
            "exec",
        ),
        namespace,
    )
    layer = namespace["LoRALayerWeights"]
    packed = namespace["PackedLoRALayerWeights"]
    rng = torch.Generator().manual_seed(118)
    pairs = [
        (torch.randn(1, 8, 16, generator=rng), torch.randn(4, 12, 8, generator=rng)),
        (torch.randn(4, 8, 12, generator=rng), torch.randn(1, 16, 8, generator=rng)),
        (torch.randn(1, 8, 16, generator=rng), torch.randn(4, 12, 8, generator=rng)),
    ]
    loras = [
        layer(f"experts.w{i + 1}", 8, 32, a.clone(), b.clone())
        for i, (a, b) in enumerate(pairs)
    ]
    result = packed.pack_moe_stacked(loras, "language_model.model.layers.3.mlp.experts")
    result.optimize()
    for i, (a, b) in enumerate(pairs):
        assert result.lora_a[i].shape == a.shape
        assert result.lora_b[i].shape == b.shape
        for expert in range(4):
            aa, bb = (
                a[0 if a.shape[0] == 1 else expert],
                b[0 if b.shape[0] == 1 else expert],
            )
            actual = (
                result.lora_b[i][0 if b.shape[0] == 1 else expert]
                @ result.lora_a[i][0 if a.shape[0] == 1 else expert]
            )
            torch.testing.assert_close(actual, (32 / 8) * bb @ aa)
    once = [v.clone() for v in result.lora_b]
    result.optimize()
    assert all(torch.equal(a, b) for a, b in zip(once, result.lora_b, strict=True))
