"""CPU-only classification of saved GLM layer-3 selections and FP32 routing.

This reads prior captures; it never runs a model or qualifies GPU parity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from safetensors import safe_open
import torch
from torch.nn import functional as F

from tests.integration.megatron.glm5next_f4_localize import Weights, norm


def compiled_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Arithmetic in run-3's saved Inductor RMSNorm: one final BF16 cast."""
    value = x.float()
    return (
        weight.float()
        * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5))
    ).to(x.dtype)


def sampler_hadamard128(value: torch.Tensor) -> torch.Tensor:
    """FP32 butterflies from the audited vLLM FWHT, rounded once to BF16."""
    if value.shape[-1] != 128:
        raise ValueError("The sampler's GLM indexer requires dimension 128")
    result = value.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        groups = result.reshape(*value.shape[:-1], -1, 2, stride)
        left, right = groups.unbind(-2)
        result = torch.stack((left + right, left - right), -2).flatten(-3)
    return (result * 128**-0.5).bfloat16()


def sampler_fp8(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-128 E4M3FN, ceil-power-of-two scale, absmax floor 1e-4."""
    rotated = sampler_hadamard128(value).float()
    scale = torch.exp2(
        torch.ceil(
            torch.log2(rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448)
        )
    )
    return (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def sampler_pool_scores(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    gates: torch.Tensor,
    ape: torch.Tensor,
) -> torch.Tensor:
    """CPU equation oracle for image 83f9988e, not a DeepGEMM emulator.

    Unlike HF, pool probabilities/products stay FP32 until the pooled key is
    rounded. The query and pooled key are then rotated and FP8-quantized.
    weights must be the sampler's FP32 projection, before either positive scale.
    """
    n = keys.shape[0] // 4
    grouped = keys[: n * 4].float().view(n, 4, 128)
    logits = gates[: n * 4].float().view(n, 4, 128) + ape.float()
    pools = (logits.softmax(1) * grouped).sum(1).bfloat16()
    qq, qs = sampler_fp8(q)
    kk, ks = sampler_fp8(pools)
    dots = torch.matmul(qq.float(), kk.float().T)
    dots = dots * qs * ks.T
    return (dots.relu() * weights[..., None] * (q.shape[1] * 128) ** -0.5).sum(1)


def indexer_rows(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Compare valid sets independently of width, order, and repeated sentinels."""
    counts: Counter = Counter()
    details = []
    for position, (a, b) in enumerate(zip(actual, expected, strict=True)):
        left, right = a.tolist(), b.tolist()
        sa, sb = {x for x in left if x >= 0}, {x for x in right if x >= 0}
        budget = "all_candidates" if (position + 1) // 4 <= 512 else "selection"
        counts[budget + ".positions"] += 1
        counts[budget + ".set_disagreements"] += sa != sb
        counts["positions_below_2048"] += position < 2048
        counts["disagreements_below_2048"] += position < 2048 and sa != sb
        for label, seq, selected in (("native", left, sa), ("hf", right, sb)):
            valid = [x for x in seq if x >= 0]
            counts[label + ".duplicate_rows"] += len(valid) != len(selected)
            counts[label + ".noncausal_rows"] += any(x > position for x in selected)
            tail = set(range(position + 1 - (position + 1) % 4, position + 1))
            counts[label + ".tail_errors"] += not tail <= selected
            complete = selected - tail
            pools = {x // 4 for x in complete}
            counts[label + ".partial_pool_rows"] += len(complete) != 4 * len(pools)
            counts[label + ".count_errors"] += len(selected) != min(
                (position + 1) // 4, 512
            ) * 4 + len(tail)
        counts["same_set_different_order"] += sa == sb and [
            x for x in left if x >= 0
        ] != [x for x in right if x >= 0]
        if sa != sb:
            details.append(
                {
                    "position": position,
                    "native_only_pools": sorted({x // 4 for x in sa - sb}),
                    "hf_only_pools": sorted({x // 4 for x in sb - sa}),
                }
            )
    return {"counts": dict(counts), "disagreements": details}


def quantiles(values: torch.Tensor) -> dict:
    return (
        dict(
            zip(
                ("min", "p50", "p90", "p99", "max"),
                torch.quantile(
                    values.double(), torch.tensor([0, 0.5, 0.9, 0.99, 1.0]).double()
                ).tolist(),
                strict=True,
            )
        )
        if values.numel()
        else {}
    )


def router_rows(
    native_input: torch.Tensor,
    hf_input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    native_ids: torch.Tensor,
    hf_ids: torch.Tensor,
) -> dict:
    """Recompute sigmoid+bias in FP32 on each side's own saved inputs."""
    a = F.linear(native_input.float(), weight.float()).sigmoid() + bias.float()
    b = F.linear(hf_input.float(), weight.float()).sigmoid() + bias.float()
    ta, tb = a.topk(9), b.topk(9)
    aid, bid = native_ids.long().sort(-1).values, hf_ids.long().sort(-1).values
    disagree = (aid != bid).any(-1)
    margins = {
        "native": ta.values[:, 7] - ta.values[:, 8],
        "hf": tb.values[:, 7] - tb.values[:, 8],
    }
    # If both sorted lists move by at most eps, a gap >2eps cannot swap.
    drift = (a - b).abs().amax(-1)
    residuals = []
    for label, scores, captured, predicted in (
        ("native", a, aid, ta.indices[:, :8].sort(-1).values),
        ("hf", b, bid, tb.indices[:, :8].sort(-1).values),
    ):
        for p in (captured != predicted).any(-1).nonzero().flatten().tolist():
            unselected = scores[p].clone()
            unselected[captured[p]] = -torch.inf
            residuals.append(
                {
                    "side": label,
                    "position": p,
                    "capture_min_selected_minus_max_unselected": float(
                        scores[p, captured[p]].min() - unselected.max()
                    ),
                }
            )
    details = []
    for p in disagree.nonzero().flatten().tolist():
        sa, sb = set(aid[p].tolist()), set(bid[p].tolist())
        details.append(
            {
                "position": p,
                "native_only": sorted(sa - sb),
                "hf_only": sorted(sb - sa),
                "native_margin_fp32": float(margins["native"][p]),
                "hf_margin_fp32": float(margins["hf"][p]),
                "score_drift_linf": float(drift[p]),
                "native_capture_matches_cpu": bool(
                    (aid[p] == ta.indices[p, :8].sort().values).all()
                ),
                "hf_capture_matches_cpu": bool(
                    (bid[p] == tb.indices[p, :8].sort().values).all()
                ),
                "native_scores_on_union": {
                    str(i): float(a[p, i]) for i in sorted(sa | sb)
                },
                "hf_scores_on_union": {str(i): float(b[p, i]) for i in sorted(sa | sb)},
            }
        )
    return {
        "positions": len(a),
        "set_disagreements": int(disagree.sum()),
        "native_capture_cpu_disagreements": int(
            (aid != ta.indices[:, :8].sort(-1).values).any(-1).sum()
        ),
        "hf_capture_cpu_disagreements": int(
            (bid != tb.indices[:, :8].sort(-1).values).any(-1).sum()
        ),
        "disagreement_margins": {k: quantiles(v[disagree]) for k, v in margins.items()},
        "disagreement_margin_le_1e6": {
            k: int((v[disagree] <= 1e-6).sum()) for k, v in margins.items()
        },
        "disagreement_margin_le_twice_input_score_drift": {
            k: int((v[disagree] <= 2 * drift[disagree]).sum())
            for k, v in margins.items()
        },
        "disagreement_score_drift": quantiles(drift[disagree]),
        "cpu_reconstruction_residuals": residuals,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for this CPU analysis")
    torch.set_num_threads(4)
    w = Weights(args.fixtures / "F4")
    prefix = "model.language_model.layers.3."
    weight, bias, nw = [
        w.get(prefix + k)
        for k in (
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
            "post_attention_layernorm.weight",
        )
    ]
    rows = []
    work = args.run / "execution/work/results"
    with torch.no_grad():
        for path in sorted((work / "F4-diagnostics").glob("*-rank0.safetensors")):
            with (
                safe_open(path, framework="pt", device="cpu") as a,
                safe_open(
                    args.fixtures / "dumps/F4" / path.name.replace("-rank0", ""),
                    framework="pt",
                    device="cpu",
                ) as b,
            ):
                count = a.get_slice("layers.3.streams").get_shape()[1]

                def at(f, key):
                    return f.get_tensor("layers.3." + key)

                indexer = indexer_rows(
                    at(a, "indexer.indices")[0], at(b, "indexer.indices")[0, :count]
                )
                router = router_rows(
                    compiled_norm(at(a, "ffn_hc.collapsed")[0], nw),
                    at(b, "mlp_input")[0, :count],
                    weight,
                    bias,
                    at(a, "router.indices"),
                    at(b, "router.indices")[:count],
                )
                eager_ids = (
                    F.linear(
                        norm(at(a, "ffn_hc.collapsed")[0], nw, 1e-5).float(),
                        weight.float(),
                    )
                    .sigmoid()
                    .add(bias.float())
                    .topk(8)
                    .indices.sort(-1)
                    .values
                )
                router["native_capture_eager_norm_disagreements"] = int(
                    (eager_ids != at(a, "router.indices").long().sort(-1).values)
                    .any(-1)
                    .sum()
                )
                rows.append(
                    {
                        "document": path.name,
                        "positions": count,
                        "indexer": indexer,
                        "router": router,
                    }
                )
                print(
                    json.dumps(
                        {
                            "document": path.name,
                            "indexer": indexer["counts"],
                            "router": {
                                k: v for k, v in router.items() if k != "details"
                            },
                        }
                    ),
                    flush=True,
                )
            args.output.write_text(
                json.dumps(
                    {"rows": rows, "cuda_initialized": torch.cuda.is_initialized()},
                    indent=2,
                )
                + "\n"
            )
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
