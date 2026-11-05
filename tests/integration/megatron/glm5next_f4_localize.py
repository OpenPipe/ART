"""CPU component replay on original F4 HF inputs; never qualifies CUDA kernels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch import Tensor
from torch.nn import functional as F


class Weights:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    def get(self, key: str) -> Tensor:
        for site in ("attn", "ffn"):
            for field in ("fn", "base", "scale"):
                key = key.replace(f".{site}_hc.{field}", f".hc_{site}_{field}")
        with safe_open(self.root / self.index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def metrics(actual: Tensor, expected: Tensor) -> dict[str, float]:
    from art.megatron.glm5next.hf_oracle import tensor_metrics

    return tensor_metrics(actual, expected)


def norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    xf = x.float()
    return weight * (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).to(
        x.dtype
    )


def swiglu(x: Tensor, limit: float = 10.0) -> Tensor:
    gate, up = x.chunk(2, -1)
    return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def sparse_reference(
    q: Tensor, kv: Tensor, indices: Tensor, *, scale: float, **_: Any
) -> Tensor:
    """Replace only TileLang: FP32 scores/softmax/value sum, BF16 output."""
    n = q.shape[1]
    mask = torch.zeros((n, kv.shape[1]), dtype=torch.bool)
    for i in range(n):
        selected = indices[0, i].long()
        selected = selected[selected >= 0]
        mask[i, selected] = True
    scores = torch.einsum("bshd,btd->bhst", q.float(), kv.float()) * scale
    scores.masked_fill_(~mask[None, None], -torch.inf)
    return torch.einsum("bhst,btd->bshd", scores.softmax(-1), kv[..., :512].float()).to(
        q.dtype
    )


@torch.no_grad()
def replay(root: Path, dump: Path, native_dir: Path | None = None) -> dict[str, Any]:
    from unittest.mock import patch

    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextAttention,
    )

    from art.megatron.dsv4.hyper_connection import DeepSeekV4HyperConnectionUtil
    from art.megatron.glm5next.dsa import Glm5NextSelfAttention, Glm5NextSparseCore
    from art.megatron.glm5next.kpool_indexer import KPoolIndexer, pooled_indexer_topk
    from art.megatron.glm5next.layer import Glm5NextRouter

    w = Weights(root)
    cfg = Glm5NextTextConfig.from_pretrained(root, local_files_only=True)
    cfg._attn_implementation = "sdpa"
    data = load_file(dump)
    prefix = "model.language_model.layers.3."
    result: dict[str, Any] = {
        "dump": str(dump),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "substitutions": [
            "TE linear/grouped GEMMs -> torch CPU F.linear with identical operand dtypes",
            "TileLang sparse MLA -> FP32 score/softmax/value reduction, BF16 output",
            "MoE dispatch/combine -> per-expert CPU loop; BF16 sequential HF and FP32 sums reported",
        ],
    }

    x = data["layers.3.mlp_input"].reshape(-1, cfg.hidden_size)
    router = Glm5NextRouter.__new__(Glm5NextRouter)
    torch.nn.Module.__init__(router)
    router.weight = torch.nn.Parameter(w.get(prefix + "mlp.gate.weight").float())
    router.bias = None
    logits = router.gating(x)
    probs, ids = topk_routing_with_score_function(
        logits,
        cfg.num_experts_per_tok,
        use_pre_softmax=True,
        score_function="sigmoid",
        expert_bias=w.get(prefix + "mlp.gate.e_score_correction_bias").float(),
        scaling_factor=cfg.routed_scaling_factor,
        dense_output=True,
    )
    expected_ids = data["layers.3.router.indices"].long()
    expected_probs = data["layers.3.router.weights"]
    sorted_ids, order = ids.sort(-1)
    expected_sorted, expected_order = expected_ids.sort(-1)
    result["router"] = {
        "set_mismatch_tokens": int((sorted_ids != expected_sorted).any(-1).sum()),
        "tokens": len(ids),
        "weights_sorted_by_id": metrics(
            probs.gather(1, order), expected_probs.gather(1, expected_order)
        ),
        "logits_dtype": str(logits.dtype),
    }

    # Use dumped IDs/weights for expert replay to isolate expert arithmetic.
    hf_sum = torch.zeros_like(x)
    native_sum = torch.zeros_like(x, dtype=torch.float32)
    native_sum_bf16 = torch.zeros_like(x)
    native_act: Any = SimpleNamespace(
        activation_func=F.silu,
        config=SimpleNamespace(
            use_te_activation_func=False,
            bias_activation_fusion=False,
            gated_linear_unit=True,
            activation_func_clamp_value=cfg.swiglu_limit,
            activation_func=F.silu,
            glu_linear_offset=0.0,
        ),
    )
    for expert in expected_ids.unique().tolist():
        tokens, slots = torch.where(expected_ids == expert)
        ep = prefix + f"mlp.experts.{expert}."
        gu = torch.cat([w.get(ep + "gate_proj.weight"), w.get(ep + "up_proj.weight")])
        down = w.get(ep + "down_proj.weight")
        activation = swiglu(F.linear(x[tokens], gu), cfg.swiglu_limit)
        p = expected_probs[tokens, slots, None]
        hf_sum.index_add_(0, tokens, (F.linear(activation, down) * p).to(x.dtype))
        native_activation = TEGroupedMLP.bias_act_func(
            native_act, F.linear(x[tokens], gu), None, p
        )
        expert_out = F.linear(native_activation, down)
        native_sum.index_add_(0, tokens, expert_out.float())
        native_sum_bf16.index_add_(0, tokens, expert_out)

    shared_prefix = prefix + "mlp.shared_experts."
    shared_gu = torch.cat(
        [
            w.get(shared_prefix + "gate_proj.weight"),
            w.get(shared_prefix + "up_proj.weight"),
        ]
    )
    shared = F.linear(
        swiglu(F.linear(x, shared_gu), cfg.swiglu_limit),
        w.get(shared_prefix + "down_proj.weight"),
    )
    target_mlp = data["layers.3.mlp"]
    result["mlp"] = {
        "HF_operation_order": metrics(
            (hf_sum + shared).view_as(target_mlp), target_mlp
        ),
        "native_weight_before_fc2_FP32_combine": metrics(
            (native_sum.to(x.dtype) + shared).view_as(target_mlp), target_mlp
        ),
        "native_weight_before_fc2_BF16_combine": metrics(
            (native_sum_bf16 + shared).view_as(target_mlp), target_mlp
        ),
    }

    inp = data["layers.3.attention_input"]
    with torch.device("meta"):
        hf = Glm5NextTextAttention(cfg, 3)
    hf.load_state_dict(
        {
            k.removeprefix(prefix + "self_attn."): w.get(k)
            for k in w.index
            if k.startswith(prefix + "self_attn.")
        },
        assign=True,
    )
    hf.eval()
    qr = hf.q_a_layernorm(hf.q_a_proj(inp))
    indexer = KPoolIndexer(
        hidden_size=cfg.hidden_size,
        q_lora_rank=cfg.q_lora_rank,
        num_heads=cfg.index_n_heads,
        head_dim=cfg.index_head_dim,
        topk=cfg.index_topk,
    ).to(torch.bfloat16)
    mapping = {
        "linear_wq_b.weight": "wq_b.weight",
        "linear_wk.weight": "wk.weight",
        "linear_weights_proj.weight": "weights_proj.weight",
    }
    indexer.load_state_dict(
        {
            k: w.get(prefix + "self_attn.indexer." + mapping.get(k, k))
            for k in indexer.state_dict()
        },
        assign=True,
    )
    qi, ki, wi, gi = indexer.project(inp.transpose(0, 1), qr.transpose(0, 1))
    selections = pooled_indexer_topk(
        qi[0], ki[0], wi[0], gi[0], indexer.index_kpool_compress_ape, [0, inp.shape[1]]
    )
    expected_selections = data["layers.3.indexer.indices"]
    mismatches = 0
    for actual, expected in zip(selections[0], expected_selections[0], strict=True):
        mismatches += set(actual[actual >= 0].tolist()) != set(
            expected[expected >= 0].tolist()
        )
    result["indexer"] = {"set_mismatch_tokens": mismatches, "tokens": inp.shape[1]}
    target_attn = data["layers.3.attention"]
    hf_attention = hf(inp, attention_mask=torch.ones(inp.shape[:2], dtype=torch.bool))[
        0
    ]

    class Linear(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.weight = cast(Tensor, module.weight)

        def forward(self, value: Tensor) -> tuple[Tensor, None]:
            return F.linear(value, self.weight), None

    native = Glm5NextSelfAttention.__new__(Glm5NextSelfAttention)
    torch.nn.Module.__init__(native)
    native.config = SimpleNamespace(
        qk_head_dim=cfg.qk_nope_head_dim,
        qk_pos_emb_head_dim=0,
        kv_lora_rank=cfg.kv_lora_rank,
        v_head_dim=cfg.v_head_dim,
        layernorm_epsilon=cfg.rms_norm_eps,
        sequence_parallel=False,
    )
    native.num_attention_heads_per_partition = cfg.num_attention_heads
    native.tp_group = None
    native.softmax_scale = cfg.qk_nope_head_dim**-0.5
    native.q_a_lora = native.q_b_lora = native.kv_a_lora = None
    for name, module in (
        ("linear_q_down_proj", hf.q_a_proj),
        ("linear_q_up_proj", hf.q_b_proj),
        ("linear_kv_down_proj", hf.kv_a_proj_with_mqa),
        ("linear_kv_up_proj", hf.kv_b_proj),
        ("linear_proj", hf.o_proj),
    ):
        setattr(native, name, Linear(module))
    native.q_layernorm = hf.q_a_layernorm
    native.kv_layernorm = hf.kv_a_layernorm
    core = Glm5NextSparseCore.__new__(Glm5NextSparseCore)
    torch.nn.Module.__init__(core)
    core.topk = lambda *args: selections
    native.core_attention = core
    with (
        patch.object(native, "_attention_state", return_value=None),
        patch("art.megatron.glm52.attention.sparse_mla", sparse_reference),
    ):
        native_attention = native(inp.transpose(0, 1))[0].transpose(0, 1)
    result["attention"] = {
        "HF_unabsorbed_CPU_SDPA": metrics(hf_attention, target_attn),
        "native_absorbed_CPU_reference": metrics(native_attention, target_attn),
        "absorbed_vs_unabsorbed_same_CPU": metrics(native_attention, hf_attention),
    }

    hc = DeepSeekV4HyperConnectionUtil(
        cast(
            Any,
            SimpleNamespace(
                layernorm_epsilon=cfg.rms_norm_eps,
                dsv4_hc_mult=cfg.hc_mult,
                dsv4_hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                dsv4_hc_eps=cfg.hc_eps,
            ),
        )
    )
    residual = data["layers.3.input_streams"]
    result["mhc"] = {}
    for site, output in (("attn", target_attn), ("ffn", target_mlp)):
        hp = prefix + site + "_hc."
        collapsed, post, comb = hc.hc_pre_raw(
            residual, w.get(hp + "fn"), w.get(hp + "scale"), w.get(hp + "base")
        )
        for name, actual in (("collapsed", collapsed), ("post", post), ("comb", comb)):
            result["mhc"][site + "." + name] = metrics(
                actual, data[f"layers.3.{site}_hc.{name}"]
            )
        residual = hc.hc_post_raw(output, residual, post, comb)
    result["mhc"]["layer_output_with_HF_sublayers"] = metrics(
        residual, data["layers.3.streams"]
    )
    if native_dir is not None:
        candidate = load_file(
            native_dir / dump.name.replace(".safetensors", "-rank0.safetensors")
        )
        stream = candidate["layers.2.streams"].transpose(0, 1).to(torch.bfloat16)
        n = stream.shape[1]

        def pre(value: Tensor, site: str) -> tuple[Tensor, Tensor, Tensor]:
            hp = prefix + site + "_hc."
            collapsed, post, comb = hc.hc_pre_raw(
                value, w.get(hp + "fn"), w.get(hp + "scale"), w.get(hp + "base")
            )
            nk = "input_layernorm" if site == "attn" else "post_attention_layernorm"
            return (
                norm(collapsed, w.get(prefix + nk + ".weight"), cfg.rms_norm_eps),
                post,
                comb,
            )

        native_input, post, comb = pre(stream, "attn")
        with (
            patch.object(native, "_attention_state", return_value=None),
            patch.object(core, "topk", return_value=selections[:, :n]),
            patch("art.megatron.glm52.attention.sparse_mla", sparse_reference),
        ):
            pred_attention = native(native_input.transpose(0, 1))[0].transpose(0, 1)
        mid_stream = hc.hc_post_raw(pred_attention, stream, post, comb)
        native_mlp_input, post, comb = pre(mid_stream, "ffn")
        logits = router.gating(native_mlp_input.reshape(n, -1))
        bias = w.get(prefix + "mlp.gate.e_score_correction_bias").float()
        np, ni = topk_routing_with_score_function(
            logits,
            cfg.num_experts_per_tok,
            score_function="sigmoid",
            expert_bias=bias,
            scaling_factor=cfg.routed_scaling_factor,
            dense_output=True,
        )
        discrepancies = []
        scores = logits.sigmoid() + bias
        for token in range(n):
            missing = set(expected_ids[token].tolist()) - set(ni[token].tolist())
            extra = set(ni[token].tolist()) - set(expected_ids[token].tolist())
            if missing or extra:
                discrepancies.append(
                    {
                        "token": token,
                        "missing": sorted(missing),
                        "extra": sorted(extra),
                        "scores_at_predicted_input": {
                            str(e): float(scores[token, e])
                            for e in sorted(missing | extra)
                        },
                    }
                )

        def expert_replay(route_ids: Tensor, route_probs: Tensor) -> Tensor:
            v = native_mlp_input.reshape(n, -1)
            acc = torch.zeros_like(v, dtype=torch.float32)
            for expert in route_ids.unique().tolist():
                tokens, slots = torch.where(route_ids == expert)
                ep = prefix + f"mlp.experts.{expert}."
                gu = torch.cat(
                    [w.get(ep + "gate_proj.weight"), w.get(ep + "up_proj.weight")]
                )
                activation = TEGroupedMLP.bias_act_func(
                    native_act,
                    F.linear(v[tokens], gu),
                    None,
                    route_probs[tokens, slots, None],
                )
                out = F.linear(activation, w.get(ep + "down_proj.weight"))
                acc.index_add_(0, tokens, out.float())
            shared = F.linear(
                swiglu(F.linear(v, shared_gu)),
                w.get(shared_prefix + "down_proj.weight"),
            )
            return (acc.to(v.dtype) + shared).view_as(native_mlp_input)

        predicted_mlp = expert_replay(ni, np)
        predicted_stream = hc.hc_post_raw(predicted_mlp, mid_stream, post, comb)
        fixed_mlp = expert_replay(expected_ids[:n], expected_probs[:n])
        fixed_stream = hc.hc_post_raw(fixed_mlp, mid_stream, post, comb)
        rounded_probs, rounded_ids = topk_routing_with_score_function(
            logits,
            cfg.num_experts_per_tok,
            score_function="sigmoid",
            expert_bias=bias.to(torch.bfloat16).float(),
            scaling_factor=cfg.routed_scaling_factor,
            dense_output=True,
        )
        rounded_mlp = expert_replay(rounded_ids, rounded_probs)
        rounded_stream = hc.hc_post_raw(rounded_mlp, mid_stream, post, comb)
        result["propagation_from_measured_native_layer2"] = {
            "input": metrics(native_input, inp[:, :n]),
            "attention": metrics(pred_attention, target_attn[:, :n]),
            "mlp_input": metrics(native_mlp_input, data["layers.3.mlp_input"][:, :n]),
            "router_set_differences": discrepancies,
            "mlp": metrics(predicted_mlp, target_mlp[:, :n]),
            "streams_vs_HF": metrics(predicted_stream, data["layers.3.streams"][:, :n]),
            "streams_vs_measured_native": metrics(
                predicted_stream.transpose(0, 1).float(), candidate["layers.3.streams"]
            ),
            "streams_with_HF_routes_vs_HF": metrics(
                fixed_stream, data["layers.3.streams"][:, :n]
            ),
            "rounded_bias_router_indices": rounded_ids.tolist(),
            "FP32_bias_router_indices": ni.tolist(),
            "rounded_bias_streams_vs_measured_native": metrics(
                rounded_stream.transpose(0, 1).float(), candidate["layers.3.streams"]
            ),
            "rounded_bias_streams_vs_HF": metrics(
                rounded_stream, data["layers.3.streams"][:, :n]
            ),
        }
    assert not torch.cuda.is_initialized()
    return result


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only replay requires CUDA_VISIBLE_DEVICES=''")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path, default=Path("/tmp/stark-gpu/glm-fixtures/F4")
    )
    parser.add_argument("--dump", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    results = []
    for dump in args.dump:
        result = replay(args.fixture, dump, args.native_dir)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
