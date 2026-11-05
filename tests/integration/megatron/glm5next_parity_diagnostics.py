"""Report-only, all-position diagnostics for the workstation parity window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True).contiguous()


class RestoreAttribute:
    def __init__(self, obj: Any, name: str, replacement: Any):
        self.obj, self.name = obj, name
        self.original = getattr(obj, name)
        setattr(obj, name, replacement)

    def remove(self) -> None:
        setattr(self.obj, self.name, self.original)


def capture_sublayers(layer: Any, values: dict) -> list[Any]:
    """Observe actual operator returns; never recompute native routing/math."""
    prefix = f"layers.{layer.layer_number - 1}"
    handles = []

    def hook(key: str):
        def capture(_module: Any, _args: Any, result: Any) -> None:
            value = result[0] if isinstance(result, tuple) else result
            values[prefix + key] = cpu(value.transpose(0, 1))

        return capture

    for name, module in (("attention", layer.self_attention), ("mlp", layer.mlp)):
        handles.append(module.register_forward_hook(hook("." + name)))

    util = layer.hc_util
    original_pre, original_post = util.layer_pre, util.layer_post
    site = ["attn_hc"]

    def pre(hidden: Any, fn: Any, scale: Any, base: Any) -> Any:
        site[0] = "attn_hc" if fn is layer.hc_attn_fn else "ffn_hc"
        result = original_pre(hidden, fn, scale, base)
        for field, value in zip(("collapsed", "post", "comb"), result, strict=True):
            values[f"{prefix}.{site[0]}.{field}"] = cpu(
                value.transpose(0, 1) if field == "collapsed" else value
            )
        return result

    def post(*args: Any, **kwargs: Any) -> Any:
        result = original_post(*args, **kwargs)
        values[f"{prefix}.{site[0]}.streams"] = cpu(result.transpose(0, 1))
        return result

    handles.extend(
        (
            RestoreAttribute(util, "layer_pre", pre),
            RestoreAttribute(util, "layer_post", post),
        )
    )
    router = getattr(layer.mlp, "router", None)
    if router is not None:

        def routing(module: Any, _args: Any, result: Any) -> None:
            probabilities, routing_map = result
            # MCore returns a dense routing map. Read selected IDs from that map,
            # including zero-weight selections; compare sets, not top-k ordering.
            ids = routing_map.int().topk(module.topk, dim=-1).indices
            values[prefix + ".router.indices"] = cpu(ids.to(torch.int32))
            values[prefix + ".router.weights"] = cpu(probabilities.gather(-1, ids))

        handles.append(router.register_forward_hook(routing))
    core = getattr(layer.self_attention, "core_attention", None)
    indexer = getattr(core, "indexer", None)
    if indexer is not None:

        def selection(_module: Any, _args: Any, result: Any) -> None:
            # This window is CP1; native's invalid sentinel is padded S, HF's -1.
            values[prefix + ".indexer.indices"] = cpu(
                result.masked_fill(result >= result.shape[1], -1)
            )

        handles.append(indexer.register_forward_hook(selection))
    return handles


def trim_captures(values: dict, positions: int) -> dict:
    return {
        key: (
            value[:, :positions] if value.ndim >= 3 else value[:positions]
        ).contiguous()
        for key, value in values.items()
    }


def validate_captures(values: dict, layers: int, positions: int) -> None:
    required = set()
    for layer in range(layers):
        prefix = f"layers.{layer}."
        required.update(prefix + field for field in ("attention", "mlp", "streams"))
        required.update(
            prefix + site + "." + field
            for site in ("attn_hc", "ffn_hc")
            for field in ("collapsed", "post", "comb", "streams")
        )
        if layer >= 3:
            required.update((prefix + "router.indices", prefix + "router.weights"))
        if layer % 4 == 3:
            required.add(prefix + "indexer.indices")
    if missing := required - values.keys():
        raise ValueError(f"Missing native diagnostic captures: {sorted(missing)}")
    for key in required:
        value = values[key]
        observed = value.shape[1] if value.ndim >= 3 else value.shape[0]
        if observed != positions:
            raise ValueError(
                f"Incomplete position capture: {key}: {observed} != {positions}"
            )


def reference_hc_streams(reference: dict) -> None:
    """Recover the two HF residual-site outputs from M1's exact operands."""
    for key in list(reference):
        if not key.endswith(".input_streams"):
            continue
        prefix = key.removesuffix(".input_streams")
        residual = reference[key]
        post = reference[prefix + ".attn_hc.post"].to(residual.dtype)
        comb = reference[prefix + ".attn_hc.comb"].to(residual.dtype)
        attention = reference[prefix + ".attention"]
        reference[prefix + ".attn_hc.streams"] = post.unsqueeze(
            -1
        ) * attention.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual)
        reference[prefix + ".ffn_hc.streams"] = reference[prefix + ".streams"]


def selection_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    from art.megatron.glm5next.hf_oracle import topk_set_agreement

    a = actual.reshape(-1, actual.shape[-1])
    b = expected.reshape(-1, expected.shape[-1])
    disagreements = [
        i
        for i, (left, right) in enumerate(zip(a.tolist(), b.tolist(), strict=True))
        if {x for x in left if x >= 0} != {x for x in right if x >= 0}
    ]
    return {
        "positions": len(a),
        "exact_set_positions": len(a) - len(disagreements),
        "set_agreement": topk_set_agreement(a, b),
        "disagreement_positions": disagreements,
    }


def sublayer_metrics(actual: dict, reference: dict, positions: int) -> dict:
    from art.megatron.glm5next.hf_oracle import tensor_metrics

    reference_hc_streams(reference)
    report = {}
    for key, value in actual.items():
        if not key.startswith("layers."):
            continue
        if key not in reference:
            raise ValueError(f"Missing diagnostic reference: {key}")
        expected = reference[key]
        expected = (
            expected[:, :positions] if expected.ndim >= 3 else expected[:positions]
        )
        if key.endswith(".indices"):
            report[key] = selection_metrics(value, expected)
        elif key.endswith(".router.weights"):
            ids = actual[key.replace("weights", "indices")].reshape(positions, -1)
            ref_ids = reference[key.replace("weights", "indices")].reshape(
                -1, ids.shape[-1]
            )[:positions]
            a = value.reshape(positions, -1)
            b = expected.reshape(positions, -1)
            # Compare weights by expert ID. Also record full sparse-vector error
            # (missing experts have weight zero) to expose route substitutions.
            width = int(torch.cat((ids.flatten(), ref_ids.flatten())).max()) + 1
            a_dense = a.new_zeros((positions, width)).scatter_(1, ids.long(), a)
            b_dense = b.new_zeros((positions, width)).scatter_(1, ref_ids.long(), b)
            report[key] = {"positions": positions, **tensor_metrics(a_dense, b_dense)}
        else:
            # All full captures use HF's B,S,... orientation.
            report[key] = {"positions": positions, **tensor_metrics(value, expected)}
    return report


def top1_diagnostics(actual: dict, reference: dict) -> dict:
    a = actual["top1"].reshape(-1)
    count = a.numel()
    ref_ids = reference.get("top1", reference.get("top64.indices"))
    assert ref_ids is not None
    b = (
        ref_ids.reshape(-1)
        if "top1" in reference
        else ref_ids.reshape(-1, 64)[:count, 0]
    )
    disagreements = (a != b).nonzero().flatten().tolist()
    records = []
    for position in disagreements:
        item = {
            "position": position,
            "native_id": int(a[position]),
            "hf_id": int(b[position]),
        }
        for label, tensors in (("native", actual), ("hf", reference)):
            if "top2.values" in tensors:
                logits = tensors["top2.values"].reshape(count, 2)[position]
            elif "top64.values" in tensors:
                logits = tensors["top64.values"].reshape(-1, 64)[position, :2]
            else:
                logits = tensors["logits"].reshape(count, -1)[position].topk(2).values
            item[label + "_top1_top2_logit_margin"] = float(logits[0] - logits[1])
        records.append(item)
    return {
        "positions": count,
        "agreements": count - len(records),
        "disagreements": len(records),
        "disagreement_details": records,
    }


def supplemental_reference(fixtures: Path, output: Path, *, tiny: bool) -> None:
    """M1 omitted F8 sublayers. Capture them without modifying shared fixtures."""
    import json
    import os

    from tests.integration.megatron.model_support.glm5next_oracle import dump

    name = "R2" if tiny else "F8"
    output.mkdir(parents=True, exist_ok=True)
    local = output / name
    local.mkdir()
    for path in (fixtures / name).iterdir():
        if path.is_file():
            os.link(path, local / path.name)
    manifest = json.loads((fixtures / "MANIFEST.json").read_text())
    manifest["fixtures"] = {name: manifest["fixtures"][name]}
    manifest["dumps"] = {}
    (output / "MANIFEST.json").write_text(json.dumps(manifest))
    dump(
        name,
        output,
        "cpu" if tiny else "cuda",
        lengths=(128,) if tiny else (128, 512, 2048, 4096, 8192, 16384),
        all_layer_outputs=True,
        gradients=False,
    )
