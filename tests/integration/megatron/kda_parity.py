"""Bounded M2 parity worker; run directly or with torchrun (CP2/CP8).

CPU: --device cpu --fixture R2. GPU (only inside a lane grant):
  torchrun --standalone --nproc-per-node=8 kda_parity.py --device cuda \
      --fixture F4 --output /tmp/kda-cp8.json

CPU CP rehearsals exercise ART exchanges and autograd collectives using the
FP32 recurrence. They do not qualify FLA kernels or convolution halos.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
from datetime import timedelta
import inspect
from itertools import combinations
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from safetensors import safe_open
import torch
from torch import Tensor, nn
import torch.distributed as dist

from art.megatron.context_parallel.layout_index import TokenLayoutIndex
from art.megatron.gdn.gdn_prefix_tree import parse_gdn_prefix_tree_segments
from art.megatron.glm5next.kda import KimiDeltaAttention

UNPACKED_SHAPES = ((0, 1), (7, 1), (14, 2), (21, 3))


@contextmanager
def hf_torch_kernels():
    from transformers.models.glm5_next import modeling_glm5_next as hf

    names = ("causal_conv1d_fn", "chunk_kimi_delta_attention")
    saved = {name: getattr(hf, name) for name in names}
    for name, value in saved.items():
        setattr(hf, name, inspect.unwrap(value))
    try:
        yield hf
    finally:
        for name, value in saved.items():
            setattr(hf, name, value)


def load_weights(root: Path, layer: int) -> dict[str, Tensor]:
    prefix = f"model.language_model.layers.{layer}.self_attn."
    index = root / "model.safetensors.index.json"
    files = json.loads(index.read_text())["weight_map"] if index.exists() else None
    result = {}
    paths = (
        sorted({files[key] for key in files if key.startswith(prefix)})
        if files
        else ["model.safetensors"]
    )
    for name in paths:
        with safe_open(root / name, framework="pt") as stream:
            for key in stream.keys():
                if key.startswith(prefix):
                    result[key.removeprefix(prefix)] = stream.get_tensor(key)
    if not result:
        raise ValueError(f"No KDA weights for layer {layer}")
    return result


def hf_weights(state: dict[str, Tensor]) -> dict[str, Tensor]:
    result = dict(state)
    for key in ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"):
        result["forget_gate." + key] = result.pop(key)
    result["conv1d.weight"] = torch.cat(
        [result.pop(name + ".weight") for name in ("q_conv1d", "k_conv1d", "v_conv1d")]
    )
    return result


class Adapter(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4) -> None:
        super().__init__()
        self.base = base
        # Nonzero B checks both factors; an all-zero B would hide A-gradient bugs.
        self.A = nn.Parameter(
            torch.randn(rank, base.in_features, device=base.weight.device) * 0.01
        )
        self.B = nn.Parameter(
            torch.randn(base.out_features, rank, device=base.weight.device) * 0.01
        )

    def forward(self, x: Tensor) -> Tensor:  # ty: ignore[invalid-method-override]
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T).to(x.dtype)


def adapt(module: Any, *, source: Any = None) -> None:
    module.requires_grad_(False)
    for name in KimiDeltaAttention.lora_targets:
        wrapper = Adapter(getattr(module, name))
        if source is not None:
            wrapper.load_state_dict(getattr(source, name).state_dict())
        setattr(module, name, wrapper)


def gradients(module: nn.Module, output: Tensor, probe: Tensor) -> dict[str, Tensor]:
    named = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
    values = torch.autograd.grad((output.float() * probe).sum(), [p for _, p in named])
    return {
        name: grad.detach().float()
        for (name, _), grad in zip(named, values, strict=True)
    }


def metrics(actual: Tensor, expected: Tensor) -> dict[str, float]:
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    if not a.isfinite().all() or not b.isfinite().all():
        raise AssertionError("Nonfinite parity tensors")
    error = torch.linalg.vector_norm(a - b)
    denom = torch.linalg.vector_norm(b)
    return {
        "rel_l2": float(error / denom.clamp_min(1e-30)),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "max_abs": float((a - b).abs().max()),
    }


def packed(cu: Tensor) -> Any:
    return SimpleNamespace(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)


def forward_checks(module: Any, reference: Any, x: Tensor, cu: Tensor) -> dict:
    with torch.no_grad():
        actual = module(x, packed_seq_params=packed(cu))[0]
        reference_output = torch.cat(
            [
                reference(part.transpose(0, 1)).transpose(0, 1)
                for part in x.split((cu[1:] - cu[:-1]).tolist())
            ]
        )
        base = metrics(actual, reference_output)
        changed = x.clone()
        changed[8:] += 1
        causal = metrics(module(x)[0][:8], module(changed)[0][:8])
    return {
        "forward": base,
        "causality": causal,
        "forward_pass": base["cosine"] >= 0.9999 and base["rel_l2"] <= 1e-2,
        "causality_pass": causal["max_abs"] == 0,
    }


def shape_variants(target: Tensor) -> list[Tensor]:
    """Four unpacked GEMM shapes; target is always [:L, :1], loss excludes rest."""
    length, _, width = target.shape
    variants = []
    for extra, batch in UNPACKED_SHAPES:
        value = torch.randn(
            length + extra, batch, width, device=target.device, dtype=target.dtype
        )
        value[:length, :1] = target
        variants.append(value)
    return variants


def pairwise_envelope(values: list[Tensor]) -> dict[str, Any]:
    """Per-tensor E, with both relative-L2 denominators and no tolerance floor."""
    if len(values) < 4:
        raise ValueError("Cross-shape envelope requires K >= 4")
    pairs: list[dict[str, Any]] = []
    for first, second in combinations(range(len(values)), 2):
        pairs.append(
            {
                "variants": [first, second],
                "forward": metrics(values[first], values[second]),
                "reverse": metrics(values[second], values[first]),
            }
        )
    return {
        key: max(
            pair[direction][key]
            for pair in pairs
            for direction in ("forward", "reverse")
        )
        for key in ("max_abs", "rel_l2")
    } | {"pairs": pairs}


def envelope_comparison(
    actual: Tensor, expected: Tensor, envelope: dict[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = metrics(actual, expected)
    comparison["limits"] = {key: 2 * envelope[key] for key in ("max_abs", "rel_l2")}
    comparison["pass"] = all(
        comparison[key] <= comparison["limits"][key] for key in ("max_abs", "rel_l2")
    )
    comparison["hard_leakage_failure"] = any(
        comparison[key] > 10 * envelope[key] for key in ("max_abs", "rel_l2")
    )
    return comparison


def perturb_pack(
    parts: list[Tensor], target: int, delta: int
) -> tuple[Tensor, Tensor, slice]:
    """Replace every neighbor, change its length, and relocate the intact target."""
    changed = [
        part
        if index == target
        else torch.randn(
            part.shape[0] + delta,
            *part.shape[1:],
            device=part.device,
            dtype=part.dtype,
        )
        for index, part in enumerate(parts)
    ]
    boundaries = [0]
    for part in changed:
        boundaries.append(boundaries[-1] + part.shape[0])
    cu = torch.tensor(boundaries, device=parts[0].device, dtype=torch.int32)
    return torch.cat(changed), cu, slice(boundaries[target], boundaries[target + 1])


def packing_checks(module: Any, x: Tensor, cu: Tensor, probe: Tensor) -> dict[str, Any]:
    """Window-2 gates; every comparison has its own target-only output/LoRA E."""
    lengths = (cu[1:] - cu[:-1]).tolist()
    parts = list(x.split(lengths))
    targets = []
    for index, (target, target_probe) in enumerate(
        zip(parts, probe.split(lengths), strict=True)
    ):
        variants = shape_variants(target)
        outputs, grads = [], []
        for value in variants:
            output = module(value)[0][: target.shape[0], :1]
            grads.append(gradients(module, output, target_probe))
            outputs.append(output.detach().clone())
        output_e = pairwise_envelope(outputs)
        gradient_e = {
            name: pairwise_envelope([value[name] for value in grads])
            for name in grads[0]
        }
        target_slice = slice(int(cu[index]), int(cu[index + 1]))
        packed_output = module(x, packed_seq_params=packed(cu))[0][target_slice]
        packed_grads = gradients(module, packed_output, target_probe)
        packed_output = packed_output.detach().clone()
        output_comparison = envelope_comparison(packed_output, outputs[0], output_e)
        gradient_comparisons = {
            name: envelope_comparison(value, grads[0][name], gradient_e[name])
            for name, value in packed_grads.items()
        }
        leakage = {}
        # Keep training dispatch identical: no_grad could select FlashKDA instead.
        for delta in (0, 7, -7):
            changed, changed_cu, changed_slice = perturb_pack(parts, index, delta)
            changed_output = module(changed, packed_seq_params=packed(changed_cu))[0][
                changed_slice
            ].detach()
            leakage[str(delta)] = {
                "lengths": (changed_cu[1:] - changed_cu[:-1]).tolist(),
                **envelope_comparison(changed_output, packed_output, output_e),
            }
        targets.append(
            {
                "target": index,
                "length": target.shape[0],
                "shapes": [list(value.shape) for value in variants],
                "gemm_rows": [value.shape[0] * value.shape[1] for value in variants],
                "output_envelope": output_e,
                "gradient_envelopes": gradient_e,
                "packed_output": output_comparison,
                "packed_gradients": gradient_comparisons,
                "leakage": leakage,
                "packed_pass": output_comparison["pass"]
                and all(value["pass"] for value in gradient_comparisons.values()),
                "leakage_pass": all(value["pass"] for value in leakage.values()),
                "hard_leakage_failure": any(
                    value["hard_leakage_failure"] for value in leakage.values()
                ),
            }
        )
    return {
        "gate": "window2-cross-shape-2E",
        "targets": targets,
        "packed_pass": all(target["packed_pass"] for target in targets),
        "leakage_pass": all(target["leakage_pass"] for target in targets),
        "hard_leakage_failure": any(
            target["hard_leakage_failure"] for target in targets
        ),
    }


def cp_checks(module: Any, x: Tensor, cu: Tensor, group: Any) -> dict:
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    module.cp_group = None
    full = copy.deepcopy(module)
    module.cp_group = group
    adapt(module)
    adapt(full, source=module)
    local = x.chunk(world)[rank].clone().requires_grad_()
    probe = torch.randn_like(x, dtype=torch.float32)
    actual = module(local, packed_seq_params=packed(cu))[0]
    expected = full(x, packed_seq_params=packed(cu))[0]
    output_metrics = metrics(actual, expected.chunk(world)[rank])
    local_grads = gradients(module, actual, probe.chunk(world)[rank])
    expected_grads = gradients(full, expected, probe)
    comparisons = {}
    for name, grad in local_grads.items():
        dist.all_reduce(grad, group=group)
        comparisons[name] = metrics(grad, expected_grads[name])
    # Exercise ART's arbitrary attention layout, branching, and conv histories.
    length = x.shape[0]
    root, child = length // 4, length // 2
    groups = torch.tensor([[0] * root + [1] * child + [2] * (length - root - child)])
    parents = torch.tensor([[-1] * root + [0] * (length - root)])
    spec = parse_gdn_prefix_tree_segments(groups, parents)
    ranges = tuple(
        (((world - 1 - r) * (length // world), (world - r) * (length // world), 0),)
        for r in range(world)
    )
    state = SimpleNamespace(
        group_ids=groups,
        parent_ids=parents,
        gdn_execution_spec=spec,
        kda_token_layout=TokenLayoutIndex(ranges, (length // world,) * world),
        model_state={},
    )
    tree_local = x.chunk(world)[world - 1 - rank].clone().requires_grad_()
    tree_probe = probe.chunk(world)[world - 1 - rank]
    tree_output = module(tree_local, attention_bias=state)[0]
    tree_reference = full(
        x,
        attention_bias=SimpleNamespace(
            group_ids=groups, parent_ids=parents, model_state={}
        ),
    )[0]
    tree_metrics = metrics(tree_output, tree_reference.chunk(world)[world - 1 - rank])
    tree_grads = gradients(module, tree_output, tree_probe)
    tree_expected = gradients(full, tree_reference, probe)
    tree_comparisons = {}
    for name, grad in tree_grads.items():
        dist.all_reduce(grad, group=group)
        tree_comparisons[name] = metrics(grad, tree_expected[name])
    return {
        "gate": "rel-l2-1e-3-no-envelope-fallback",
        "output_policy": "blocking",
        "gradient_policy": "report-only; >1e-2 rel-L2 is a probable bug",
        "world_size": world,
        "rank": rank,
        "output": output_metrics,
        "gradients": comparisons,
        "tree_output": tree_metrics,
        "tree_gradients": tree_comparisons,
    }


def apply_gate_policy(record: dict[str, Any], world: int) -> None:
    """Keep numeric FAILs intact; only correctness controls continuation."""
    if world > 1:
        record["output_pass"] = all(
            record[key]["rel_l2"] <= 1e-3 for key in ("output", "tree_output")
        )
        record["gradients_pass"] = all(
            value["rel_l2"] <= 1e-3
            for key in ("gradients", "tree_gradients")
            for value in record[key].values()
        )
        record["probable_gradient_bugs"] = [
            {"case": key, "tensor": name, **value}
            for key in ("gradients", "tree_gradients")
            for name, value in record[key].items()
            if value["rel_l2"] > 1e-2
        ]
        record["blocking_pass"] = record["output_pass"]
        record["report_only_pass"] = record["gradients_pass"]
    else:
        record["blocking_pass"] = (
            all(
                record[key]
                for key in ("forward_pass", "causality_pass", "leakage_pass")
            )
            and not record["packing"]["hard_leakage_failure"]
        )
        record["report_only_pass"] = record["packed_pass"] and record["gradients_pass"]
    record["pass"] = record["blocking_pass"] and record["report_only_pass"]


def run(args: Any) -> dict:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        torch.cuda.set_device(rank)
        # This controls HF's torch oracle. KDA's CP launch shim separately
        # supplies TF32x3 to Triton; torch's flag does not control tl.dot.
        torch.backends.cuda.matmul.allow_tf32 = False
    device = (
        torch.device(args.device, rank)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if world > 1:
        dist.init_process_group(
            "nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=600)
        )
    torch.manual_seed(20261105)
    root = args.fixtures / args.fixture
    values = json.loads((root / "config.json").read_text())
    cfg: Any = SimpleNamespace(**values)
    if world == 1:
        from transformers.models.glm5_next.configuration_glm5_next import (
            Glm5NextTextConfig,
        )

        cfg = Glm5NextTextConfig(**values)
    assert cfg.layer_types is not None
    layers = [
        i for i, kind in enumerate(values["layer_types"]) if kind == "linear_attention"
    ][:3]
    records = {}
    try:
        with hf_torch_kernels() if world == 1 else nullcontext() as hf:
            for layer in layers:
                torch.manual_seed(20261105 + layer)
                state = load_weights(root, layer)
                dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
                module = KimiDeltaAttention(
                    cfg,
                    backend="fla" if args.device == "cuda" else "reference",
                    device=device,
                    dtype=dtype,
                )
                module.load_state_dict(state, strict=True)
                # Always keep the checkpoint strict-FP32 parameters in FP32.
                module.to(device=device, dtype=dtype)
                x = torch.randn(
                    args.tokens, 1, cfg.hidden_size, device=device, dtype=dtype
                )
                cu = torch.tensor(
                    [0, args.tokens // 4 + 1, args.tokens // 2 + 3, args.tokens],
                    device=device,
                    dtype=torch.int32,
                )
                if world > 1:
                    module.cp_group = dist.group.WORLD
                    records[str(layer)] = cp_checks(module, x, cu, dist.group.WORLD)
                else:
                    assert hf is not None
                    reference = hf.Glm5NextTextLinearAttention(cfg, layer_idx=layer).to(
                        device=device, dtype=dtype
                    )
                    reference.load_state_dict(hf_weights(state), strict=True)
                    reference.conv1d.float()
                    reference.forget_gate.A_log.data = (
                        reference.forget_gate.A_log.data.float()
                    )
                    reference.forget_gate.dt_bias.data = (
                        reference.forget_gate.dt_bias.data.float()
                    )
                    records[str(layer)] = forward_checks(module, reference, x, cu)
                    adapt(module)
                    # Independent FP32 autograd HF layer, identical adapter state.
                    reference.float()
                    adapt(reference, source=module)
                    expected = reference(x.float().transpose(0, 1)).transpose(0, 1)
                    actual = module(x)[0]
                    probe = torch.randn_like(actual, dtype=torch.float32)
                    actual_grads = gradients(module, actual, probe)
                    expected_grads = gradients(reference, expected, probe)
                    comparisons = {
                        name: metrics(value, expected_grads[name])
                        for name, value in actual_grads.items()
                    }
                    records[str(layer)]["gradients"] = comparisons
                    records[str(layer)]["gradients_pass"] = all(
                        value["rel_l2"] <= 2e-2 for value in comparisons.values()
                    )
                    packing = packing_checks(module, x, cu, probe)
                    records[str(layer)]["packing"] = packing
                    records[str(layer)]["packed_pass"] = packing["packed_pass"]
                    records[str(layer)]["leakage_pass"] = packing["leakage_pass"]
                apply_gate_policy(records[str(layer)], world)
                print(
                    json.dumps(
                        {"layer": layer, "rank": rank, "results": records[str(layer)]}
                    ),
                    flush=True,
                )
                del module
    finally:
        if world > 1:
            dist.destroy_process_group()
    return {
        "device": str(device),
        "fixture": args.fixture,
        "world_size": world,
        "kernel_evidence": args.device == "cuda",
        "seed": 20261105,
        "layer_seed": "seed + layer index",
        "gate_policy": "Updates 13/27: numeric tolerances report-only",
        **{
            key: bool(records) and all(record[key] for record in records.values())
            for key in ("pass", "blocking_pass", "report_only_pass")
        },
        "layers": records,
    }


def schema_only(args: Any) -> dict:
    """Exercise real F4 tensor volumes on meta, without any CUDA admission."""
    root = args.fixtures / args.fixture
    cfg = SimpleNamespace(**json.loads((root / "config.json").read_text()))
    records = {}
    for index, kind in enumerate(cfg.layer_types):
        if kind != "linear_attention":
            continue
        module = KimiDeltaAttention(cfg, device="meta", dtype=torch.bfloat16)
        weights = load_weights(root, index)
        expected = module.state_dict()
        assert set(weights) == set(expected)
        for name, value in weights.items():
            assert value.shape == expected[name].shape, (
                name,
                value.shape,
                expected[name].shape,
            )
        boundaries = [0, args.tokens // 4 + 1, args.tokens // 2 + 3, args.tokens]
        target_shapes = [
            [length + extra, batch, cfg.hidden_size]
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
            for length in [end - start]
            for extra, batch in UNPACKED_SHAPES
        ]
        records[str(index)] = {
            "checkpoint_bytes": sum(
                t.numel() * t.element_size() for t in weights.values()
            ),
            "parameter_elements": sum(t.numel() for t in weights.values()),
            "bf16_input_bytes": args.tokens * cfg.hidden_size * 2,
            "unpacked_target_shapes": target_shapes,
            "unpacked_target_bf16_input_bytes": [
                sequence * batch * width * 2 for sequence, batch, width in target_shapes
            ],
            "largest_perturbed_pack_bf16_input_bytes": (args.tokens + 14)
            * cfg.hidden_size
            * 2,
            "fp32_recurrent_state_bytes": cfg.linear_num_heads
            * cfg.linear_head_dim**2
            * 4,
            "lora_elements_rank4": 4
            * sum(
                sum(getattr(module, name).weight.shape) for name in module.lora_targets
            ),
        }
    return {
        "pass": True,
        "schema_only": True,
        "fixture": args.fixture,
        "layers": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--fixture", choices=("R2", "F4"), default="R2")
    parser.add_argument(
        "--fixtures", type=Path, default=Path("/tmp/stark-gpu/glm-fixtures")
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schema-only", action="store_true")
    args = parser.parse_args()
    if args.tokens < 32 or args.tokens % int(os.environ.get("WORLD_SIZE", "1")):
        parser.error("tokens must be >=32 and divisible by world size")
    result = schema_only(args) if args.schema_only else run(args)
    path = args.output.with_name(
        args.output.stem + f"-rank{os.environ.get('LOCAL_RANK', '0')}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    if not result.get("blocking_pass", result["pass"]):
        raise SystemExit("KDA blocking correctness gate failed; see receipt")


if __name__ == "__main__":
    main()
