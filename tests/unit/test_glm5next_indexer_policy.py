"""Serving-equation controls plus opt-in retained actual-image selections."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.nn import functional as F

from art.megatron.glm5next import indexer_policy
from art.megatron.glm5next.kpool_indexer import KPoolIndexer, select_pools


def test_fwht_quantization_and_fp32_pooling() -> None:
    torch.manual_seed(928)
    keys, gates = [torch.randn(40, 128).bfloat16() for _ in range(2)]
    ape = torch.randn(4, 128)
    expected = (
        (
            (gates.float().reshape(10, 4, 128) + ape).softmax(1)
            * keys.float().reshape(10, 4, 128)
        )
        .sum(1)
        .bfloat16()
    )
    pools = indexer_policy.compress(keys, gates, ape, torch.arange(10) * 4)
    torch.testing.assert_close(pools, expected, rtol=0, atol=0)
    matrix = torch.ones(1, 1)
    for _ in range(7):
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    rotated = (expected.float() @ matrix / 128**0.5).bfloat16().float()
    scale = torch.exp2(
        torch.ceil(
            torch.log2(rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    quantized, actual_scale = indexer_policy.quantize(pools)
    torch.testing.assert_close(actual_scale, scale, rtol=0, atol=0)
    torch.testing.assert_close(
        quantized.float(),
        (rotated / scale).to(torch.float8_e4m3fn).float(),
        rtol=0,
        atol=0,
    )
    zero, zero_scale = indexer_policy.quantize(torch.zeros(1, 128))
    assert not zero.float().any() and zero_scale.item() > 0


def test_projection_precision_survives_real_mp_wrapper() -> None:
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    from art.megatron.runtime.bridge_runtime import _wrap_with_mp_wrapper

    torch.manual_seed(582)
    indexer = KPoolIndexer(hidden_size=16, q_lora_rank=8, selection_policy="vllm_fp8")
    # Deliberately non-BF16 FP32 checkpoint values test preservation, not dtype alone.
    with torch.no_grad():
        indexer.index_kpool_compress_ape.normal_()
        indexer.k_norm.get_parameter("weight").normal_()
        indexer.k_norm.get_parameter("bias").normal_()
    preserved = {
        k: p.clone()
        for k, p in indexer.named_parameters()
        if k
        in (
            "index_kpool_compress_ape",
            "k_norm.weight",
            "k_norm.bias",
            "linear_weights_proj.weight",
        )
    }
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    _wrap_with_mp_wrapper(cast(Any, [indexer]), config, Float16Module)
    for name, expected in preserved.items():
        torch.testing.assert_close(
            indexer.get_parameter(name), expected, rtol=0, atol=0
        )
        assert indexer.get_parameter(name).dtype == torch.float32
    x, latent = torch.randn(9, 1, 16).bfloat16(), torch.randn(9, 1, 8).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, keys, weights, _ = indexer.project(x, latent)
    projected = F.linear(x.float(), indexer.linear_weights_proj.get_parameter("weight"))
    torch.testing.assert_close(weights, projected.transpose(0, 1), rtol=0, atol=0)
    assert weights.dtype == torch.float32
    # This control would fail if the FP32 projection were accidentally autocast.
    assert (weights != projected.bfloat16().float().transpose(0, 1)).any()
    raw_keys = F.linear(x, indexer.linear_wk.get_parameter("weight"))
    expected_keys = (
        F.layer_norm(
            raw_keys.float(),
            (128,),
            indexer.k_norm.get_parameter("weight"),
            indexer.k_norm.get_parameter("bias"),
            1e-6,
        )
        .bfloat16()
        .transpose(0, 1)
    )
    torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
    assert all(not p.requires_grad for p in indexer.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("length", [3, 32, 81])
def test_serving_policy_keeps_budget_causality_and_tail(length: int) -> None:
    torch.manual_seed(981)
    q = torch.randn(length, 2, 128).bfloat16()
    keys, gates = [torch.randn(length + 4, 128).bfloat16() for _ in range(2)]
    weights, ape = torch.randn(length, 2), torch.randn(4, 128)
    query_ids = torch.arange(length) + 4
    got = select_pools(
        q,
        keys,
        weights,
        gates,
        ape,
        pool_starts=torch.arange(length // 4) * 4 + 4,
        query_ids=query_ids,
        tail_origin=4,
        topk=16,
        selection_policy="vllm_fp8",
        query_chunk_size=3,
    )
    for row, query in zip(got, query_ids.tolist(), strict=True):
        valid = row[row >= 0].tolist()
        tail_count = (query - 4 + 1) % 4
        assert len(valid) == min((query - 4 + 1) // 4, 4) * 4 + tail_count
        assert len(valid) == len(set(valid))
        assert all(4 <= token <= query for token in valid)
        assert set(row[16:][row[16:] >= 0].tolist()) == set(
            range(query + 1 - tail_count, query + 1)
        )


def test_saved_96_long_context_sets_equal_actual_image() -> None:
    root_value = os.environ.get("GLM_INDEXER_SAVED_ROOT")
    if not root_value:
        pytest.skip("Set GLM_INDEXER_SAVED_ROOT to the retained lane artifact root")
    from safetensors.torch import load_file

    from art.megatron.glm5next.hf_oracle import sha256_file

    root = Path(root_value)
    seal = json.loads((root / "layer3/SEAL-LAYER3.json").read_text())
    total, replacements = 0, []
    for path in seal["probe_inputs"]:
        data = load_file(path)
        name = Path(path).stem
        saved = load_file(
            str(
                root
                / "layer3-execution-20261105T131408Z/gpu2"
                / (name + "-scores.safetensors")
            )
        )
        positions = data["positions"]
        got = select_pools(
            data["hf.q"],
            data["hf.keys"],
            data["sampler.weights"],
            data["hf.gates"],
            data["ape"],
            pool_starts=torch.arange(data["hf.keys"].shape[0] // 4) * 4,
            query_ids=positions,
            tail_origin=0,
            selection_policy="vllm_fp8",
        )
        expected = saved["sampler.kernel_pool_ids"]
        for i, position in enumerate(positions.tolist()):
            ids = got[i]
            pools = set((ids[:2048].long() // 4).tolist())
            replacements.append(len(set(expected[i].tolist()) - pools))
            assert pools == set(expected[i].tolist())
            tail = set(range(position + 1 - (position + 1) % 4, position + 1))
            assert set(ids[2048:][ids[2048:] >= 0].tolist()) == tail
            assert (
                set(ids[ids >= 0].tolist())
                == {p * 4 + slot for p in pools for slot in range(4)} | tail
            )
        total += len(positions)
        print(
            json.dumps(
                {
                    "input": path,
                    "sha256": sha256_file(Path(path)),
                    "exact_pool_sets": len(positions),
                    "replacements": replacements[-len(positions) :],
                }
            )
        )
    assert total == 96 and replacements == [0] * 96
    assert not torch.cuda.is_initialized()
