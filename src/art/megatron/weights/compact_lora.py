"""Versioned, rank-local compact LoRA snapshots and lossless factor converters.

This is an internal format, not a claim of Tinker training-state compatibility.
All tensors remain in ART A_T/B_T order. Expert IDs and scale are explicit; the
caller owns version fencing, distributed assembly, and snapshot synchronization.
"""

import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

FORMAT = "art.shared_outer.v1"


def pack_serving_tensors(tensors, *, out=None):
    """Pack canonical BF16 factors into the serving harness's sorted byte arena.

    Run after snapshot capture, before publication. A caller-owned arena may be
    reused; its lifetime must extend through transfer completion. No expert
    expansion, CPU copy, or intermediate flattened concatenation is needed.
    """
    if not tensors:
        raise ValueError("canonical arena requires factors")
    device = next(iter(tensors.values())).device
    if any(t.dtype != torch.bfloat16 or t.device != device for t in tensors.values()):
        raise ValueError("canonical factors must be colocated BF16")
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    if out is None:
        out = torch.empty(nbytes, dtype=torch.uint8, device=device)
    if (
        out.shape != (nbytes,)
        or out.dtype != torch.uint8
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError("canonical arena geometry differs")
    offset, entries = 0, []
    for name, tensor in sorted(tensors.items()):
        if not tensor.is_contiguous():
            raise ValueError("canonical factors must be contiguous")
        size = tensor.numel() * tensor.element_size()
        out[offset : offset + size].copy_(tensor.reshape(-1).view(torch.uint8))
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "offset": offset,
                "nbytes": size,
            }
        )
        offset += size
    return out, {
        "format": "art.shared_outer.wire.v1",
        "nbytes": nbytes,
        "tensors": entries,
    }


def fold_serving_scale_(tensors, adapter_config):
    """Fold alpha/rank into caller-owned captured B factors on their device.

    Serving-only: native weights/Adam state keep their original parameterization.
    Never pass live trainable factors. Standard PEFT alpha records the new scale
    exactly once; the source alpha is retained as explicit wire provenance.
    """
    if "stark_scale_folded_from" in adapter_config:
        raise ValueError("serving scale is already folded")
    rank = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    if rank < 1 or not math.isfinite(alpha):
        raise ValueError("invalid serving rank/alpha")
    scale = alpha / rank
    factors = [t for k, t in tensors.items() if k.endswith(".lora_B.weight")]
    if not factors or any(
        t.dtype != torch.bfloat16 or t.shape[-1] != rank or t.requires_grad
        for t in factors
    ):
        raise ValueError("expected captured BF16 B factors at the configured rank")
    if scale != 1:
        for tensor in factors:
            tensor.mul_(scale)
    return {
        **adapter_config,
        "lora_alpha": rank,
        "stark_scale_folded_from": adapter_config["lora_alpha"],
    }


def per_expert_to_shared(a, b, *, shared_factor):
    """Collapse exact replicas; never discard differing trained factors.

    Gauge-equivalent factors with differing bytes need a separately qualified
    refactorization. Zero-function initialization is not silently rewritten.
    """
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("expected expert-major A_T and B_T")
    if shared_factor not in ("A", "B"):
        raise ValueError("shared_factor must be A or B")
    if a.shape[-1] != b.shape[1] or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("expert factor rank, dtype or device mismatch")
    outer = a if shared_factor == "A" else b
    first = outer[0].contiguous().view(torch.uint8)
    if not all(torch.equal(first, t.contiguous().view(torch.uint8)) for t in outer[1:]):
        raise ValueError("independent factors cannot be collapsed losslessly")
    return (a[:1].clone(), b) if shared_factor == "A" else (a, b[:1].clone())


def shared_to_per_expert(a, b, *, experts):
    if experts < 1 or a.ndim != 3 or b.ndim != 3:
        raise ValueError("invalid expert geometry")
    if a.shape[0] not in (1, experts) or b.shape[0] not in (1, experts):
        raise ValueError("expert count mismatch")
    return a.expand(experts, -1, -1).clone(), b.expand(experts, -1, -1).clone()


def refactor_per_expert_to_shared(
    a,
    b,
    *,
    shared_factor,
    rank,
    source_scale=1.0,
    target_scale=1.0,
    rtol=1e-6,
    atol=1e-7,
):
    """Function-preserving numerical conversion if the joint update space fits.

    This opt-in SVD converter can handle differing factor gauges. It preserves
    updates within explicit tolerances, not factor bytes or Adam state. It
    materializes dense updates and is intended for offline conversion only.
    Pass the stored source/target alpha-over-rank scales when changing rank;
    defaults preserve unscaled factor products. Tolerances apply to scaled updates.
    A general independent rank-r adapter is not representable at shared rank r.
    """
    if (
        shared_factor not in ("A", "B")
        or rank < 1
        or not math.isfinite(source_scale)
        or not math.isfinite(target_scale)
        or target_scale == 0
    ):
        raise ValueError("invalid shared-factor conversion")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("expected expert-major factors")
    original = (a.double() @ b.double()) * source_scale
    updates = original / target_scale
    if shared_factor == "A":
        joined = updates.permute(1, 0, 2).flatten(1)
        u, s, vh = torch.linalg.svd(joined, full_matrices=False)
        size = min(rank, s.numel())
        outer = u[:, :size]
        inner = (
            (s[:size, None] * vh[:size])
            .reshape(size, b.shape[0], b.shape[-1])
            .permute(1, 0, 2)
        )
        converted = outer.unsqueeze(0), inner
    else:
        joined = updates.flatten(0, 1)
        u, s, vh = torch.linalg.svd(joined, full_matrices=False)
        size = min(rank, s.numel())
        converted = (
            (u[:, :size] * s[:size]).reshape(a.shape[0], a.shape[1], size),
            vh[:size].unsqueeze(0),
        )
    # Pad a lower-dimensional common space to the requested LoRA rank.
    ca = torch.nn.functional.pad(converted[0], (0, rank - size)).to(a.dtype)
    cb = torch.nn.functional.pad(converted[1], (0, 0, 0, rank - size)).to(b.dtype)
    if not torch.allclose(
        (ca.double() @ cb.double()) * target_scale, original, rtol=rtol, atol=atol
    ):
        raise ValueError(
            "joint expert update space exceeds requested rank or tolerance"
        )
    return ca, cb


def capture_compact(
    modules, *, slot_ref=None, dtype=torch.bfloat16, device="cpu", gradients=False
):
    """Copy each stored factor exactly once, including singleton expert factors."""
    tensors, specs = {}, []
    for index, module in enumerate(modules):
        params = tuple(module._lora_params(slot_ref))
        if not params:
            continue
        slot = module._slot(slot_ref) if slot_ref is not None else module
        names = []
        for suffix, param in params:
            name = f"m{index}.{suffix}"
            value = param.grad if gradients else param.detach()
            if value is None:
                value = torch.zeros_like(param)
            tensors[name] = value.to(device=device, dtype=dtype, copy=True).contiguous()
            names.append(name)
        specs.append(
            {
                "prefix": module.adapter_model_prefix,
                "keys": names,
                "experts": list(module.expert_ids) if module.is_expert else None,
                "shared_factor": module.shared_factor,
                "scale": slot.scale,
            }
        )
    return tensors, {"format": FORMAT, "modules": specs, "gradients": gradients}


def pack_compact_gradients(modules, *, slot_ref=None, out=None):
    """Write FP32 gradients directly into one caller-owned root arena.

    Shared factors occupy their stored singleton extent. An owner can pass a
    row of an N-root staging slab as ``out``; no per-parameter FP32 copies or
    expanded expert gradients are allocated. Local BF16 gradients are widened,
    not recomputed with FP32 accumulation.
    """
    entries = [
        (module.adapter_model_prefix + "." + suffix, param)
        for module in modules
        for suffix, param in module._lora_params(slot_ref)
    ]
    if not entries:
        raise ValueError("gradient arena requires at least one stored factor")
    numel = sum(param.numel() for _key, param in entries)
    device = entries[0][1].device
    if out is None:
        out = torch.empty(numel, device=device, dtype=torch.float32)
    if (
        out.shape != (numel,)
        or out.dtype != torch.float32
        or out.device != device
        or not out.is_contiguous()
    ):
        raise ValueError("gradient arena geometry differs from stored factors")
    offset, metadata = 0, []
    for key, param in entries:
        target = out[offset : offset + param.numel()].view(param.shape)
        if param.grad is None:
            target.zero_()
        else:
            target.copy_(param.grad.detach())
        metadata.append(
            {
                "key": key,
                "offset": offset,
                "shape": list(param.shape),
                "numel": param.numel(),
            }
        )
        offset += param.numel()
    return out, {
        "format": FORMAT,
        "gradients": True,
        "dtype": "float32",
        "entries": metadata,
    }


def save_compact(path, tensors, manifest):
    if manifest.get("format") != FORMAT:
        raise ValueError("unknown compact LoRA format")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path / "adapter_model.safetensors")
    (path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )


def load_compact(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("unknown compact LoRA format")
    tensors = load_file(path / "adapter_model.safetensors")
    expected = {key for spec in manifest["modules"] for key in spec["keys"]}
    if set(tensors) != expected:
        raise ValueError("compact tensor manifest mismatch")
    return tensors, manifest


def tinker_expert_tensors(prefix, gate_a, gate_b, down_a, down_b):
    """Public-cookbook expert namespace; factors use ART transposed order.

    Tinker w1=gate, w2=down, w3=up. Our fused gate/up A is emitted twice;
    this is a representable subset of Tinker's independent gate/up factors.
    """
    if gate_b.shape[-1] % 2:
        raise ValueError("gate/up output dimension must be even")
    gate, up = gate_b.chunk(2, dim=-1)
    result = {}
    for name, a, b in (
        ("w1", gate_a, gate),
        ("w3", gate_a, up),
        ("w2", down_a, down_b),
    ):
        result[f"{prefix}.{name}.lora_A.weight"] = (
            a.transpose(1, 2).contiguous().clone()
        )
        result[f"{prefix}.{name}.lora_B.weight"] = (
            b.transpose(1, 2).contiguous().clone()
        )
    return result


def compact_to_vllm(tensors, manifest, *, handler, adapter_config):
    """Qwen EP1 compact snapshot -> vLLM's public pre-stacked shared layout.

    Dense conversions reuse the model handler, including gated-Q row ordering.
    Gate/up and Q/K/V remain tied where ART ties them; this does not introduce
    the extra independently trainable factors in Tinker's default parameter set.
    Neither hidden-side factor is expanded across experts.
    """
    if manifest.get("format") != FORMAT or manifest.get("gradients"):
        raise ValueError("expected compact adapter weights")
    dense, expert_groups = {}, {}
    for spec in manifest["modules"]:
        prefix = spec["prefix"]
        a, b = (tensors[key] for key in spec["keys"])
        if spec["experts"] is None:
            dense[prefix + ".lora_A.weight"] = a.T
            dense[prefix + ".lora_B.weight"] = b.T
            continue
        experts = spec["experts"]
        if experts != list(range(len(experts))):
            raise ValueError("compact export requires a complete EP1 expert range")
        group, projection = prefix.split(".{expert}.")
        expected = "B" if projection == "down_proj" else "A"
        if spec["shared_factor"] != expected:
            raise ValueError("shared vLLM export refuses independent expert factors")
        if a.ndim != 3 or b.ndim != 3 or a.shape[-1] != b.shape[1]:
            raise ValueError("invalid expert factor dimensions")
        expected_a = len(experts) if expected == "B" else 1
        expected_b = len(experts) if expected == "A" else 1
        if a.shape[0] != expected_a or b.shape[0] != expected_b:
            raise ValueError("expert factors disagree with manifest expert count")
        expert_groups.setdefault(group, {})[projection] = (a, b)
    result, config = handler.to_vllm_lora_tensors(
        dense, adapter_config=dict(adapter_config)
    )
    for prefix, projections in expert_groups.items():
        from art.megatron.model_support.handlers.qwen3_5 import _to_vllm_key

        public_prefix = _to_vllm_key(prefix)
        if set(projections) == {"gate_up_proj", "down_proj"}:
            ga, gb = projections["gate_up_proj"]
            da, db = projections["down_proj"]
            result.update(tinker_expert_tensors(public_prefix, ga, gb, da, db))
        elif set(projections) == {"gate_proj", "up_proj", "down_proj"}:
            for projection, public in (
                ("gate_proj", "w1"),
                ("up_proj", "w3"),
                ("down_proj", "w2"),
            ):
                a, b = projections[projection]
                result[f"{public_prefix}.{public}.lora_A.weight"] = a.transpose(
                    1, 2
                ).contiguous()
                result[f"{public_prefix}.{public}.lora_B.weight"] = b.transpose(
                    1, 2
                ).contiguous()
        else:
            raise ValueError("expected gate/up (fused or independent) and down")
    config["stark_moe_layout"] = "shared_outer_v1"
    # Public target names follow independently stored GDN factors.
    if any(".in_proj_q." in key for key in result):
        targets = set(config["target_modules"]) - {"in_proj_qkv"}
        config["target_modules"] = sorted(
            targets | {"in_proj_q", "in_proj_k", "in_proj_v"}
        )
    if expert_groups:
        config["target_modules"] = sorted(set(config["target_modules"]) | {"experts"})
    return {key: tensor.contiguous() for key, tensor in result.items()}, config
