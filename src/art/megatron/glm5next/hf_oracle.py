"""Text-only native HF GLM-5.3 oracle and portable dump comparisons.

Load with Transformers 5.20.2; its checkpoint converters are essential. This
module deliberately has no Megatron dependency. Large fixtures live outside Git.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

HF_REPO = "zai-org/GLM-5.3-Flash-BF16"
HF_REVISION = "f9fed3bbc82291073373d67452e9f25317ef7577"
DEFAULT_FIXTURES = Path("/tmp/stark-gpu/glm-fixtures")
TRANSFORMERS_VERSION = "5.20.2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_records(root: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Artifact escapes manifest root: {path}")
        if (
            path.stat().st_size != record["size"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Artifact integrity failure: {path}")


def hf_model_class() -> Any:
    """Keep HF's text modules and original checkpoint prefix, omitting vision."""
    import transformers
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextPreTrainedModel,
        Glm5NextTextModel,
    )

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Oracle requires transformers=={TRANSFORMERS_VERSION}")

    class Glm5NextTextOracle(Glm5NextPreTrainedModel):
        config_class = Glm5NextTextConfig
        _tied_weights_keys: dict[str, str] = {}

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.model: Any = nn.Module()
            self.model.language_model = Glm5NextTextModel(config)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.post_init()

        def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
            output = self.model.language_model(input_ids=input_ids, **kwargs)
            return CausalLMOutputWithPast(
                logits=self.lm_head(output.last_hidden_state),
                past_key_values=output.past_key_values,
            )

    return Glm5NextTextOracle


@contextmanager
def cpu_reference_kernels():
    """Use HF's own torch fallbacks on CPU even when FLA is installed."""
    from transformers.models.glm5_next import modeling_glm5_next as modeling

    originals = {}
    for name in (
        "chunk_kimi_delta_attention",
        "recurrent_kimi_delta_attention",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ):
        original = getattr(modeling, name)
        originals[name] = original
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__
        setattr(modeling, name, original)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(modeling, name, original)


def load_fixture(
    name: str = "R2",
    *,
    root: Path = DEFAULT_FIXTURES,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    verify: bool = True,
) -> Any:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    entry = manifest["fixtures"][name]
    if entry["status"] != "ready":
        raise ValueError(f"Fixture {name} is not ready: {entry['status']}")
    if verify:
        verify_records(root, entry["files"])
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "attn_implementation": "sdpa",
        "dtype": "auto",
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    model, info = hf_model_class().from_pretrained(
        root / name, output_loading_info=True, **kwargs
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(f"Incomplete fixture load: {info}")
    return model.eval().to(device)


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    if actual.shape != expected.shape or actual.numel() == 0:
        raise ValueError(
            f"Nonempty equal shapes required: {actual.shape}, {expected.shape}"
        )
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite comparison input")
    delta = a - b
    a_norm, b_norm = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    rel_l2 = (
        float(torch.linalg.vector_norm(delta) / b_norm)
        if b_norm
        else (0.0 if not a_norm else float("inf"))
    )
    cosine = (
        float(torch.dot(a, b) / (a_norm * b_norm))
        if a_norm and b_norm
        else float(bool(not a_norm and not b_norm))
    )
    return {"max_abs": float(delta.abs().max()), "rel_l2": rel_l2, "cosine": cosine}


def logprob_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    metrics = tensor_metrics(actual, expected)
    delta = (actual.detach().double() - expected.detach().double()).abs().flatten()
    return {
        **metrics,
        "logprob_abs_mean": float(delta.mean()),
        "logprob_abs_p99": float(torch.quantile(delta, 0.99)),
    }


def topk_set_agreement(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Mean Jaccard similarity per row, ignoring -1 sentinels and duplicates."""
    if actual.shape[:-1] != expected.shape[:-1] or actual.ndim == 0:
        raise ValueError("Top-k leading dimensions must match")
    a = actual.detach().cpu().reshape(-1, actual.shape[-1]).tolist()
    b = expected.detach().cpu().reshape(-1, expected.shape[-1]).tolist()
    scores = []
    for left, right in zip(a, b, strict=True):
        x, y = {i for i in left if i >= 0}, {i for i in right if i >= 0}
        scores.append(len(x & y) / len(x | y) if x | y else 1.0)
    if not scores:
        raise ValueError("Empty selection comparison")
    return sum(scores) / len(scores)


def compare_dump(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, Any]:
    if actual.keys() != expected.keys():
        raise ValueError(
            f"Dump keys differ: missing={expected.keys() - actual.keys()}, extra={actual.keys() - expected.keys()}"
        )
    report: dict[str, Any] = {}
    for name, reference in expected.items():
        candidate = actual[name]
        if reference.is_floating_point():
            report[name] = (
                logprob_metrics(candidate, reference)
                if "logprob" in name
                else tensor_metrics(candidate, reference)
            )
        else:
            if candidate.shape != reference.shape:
                raise ValueError(f"Shape mismatch for {name}")
            report[name] = {"exact": bool(torch.equal(candidate, reference))}
            if "indices" in name or "topk" in name:
                report[name]["set_agreement"] = topk_set_agreement(candidate, reference)
    return report


def main() -> None:
    """Compare safetensors, or run a file exposing run_oracle(inputs) -> dict."""
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--actual", type=Path)
    source.add_argument("--module", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--key", action="append", help="Compare selected dump keys (repeatable)"
    )
    args = parser.parse_args()
    if args.module:
        if args.inputs is None:
            parser.error("--module requires --inputs")
        spec = importlib.util.spec_from_file_location("oracle_candidate", args.module)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load candidate module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actual = module.run_oracle(load_file(args.inputs))
    else:
        actual = load_file(args.actual)
    expected = load_file(args.expected)
    if args.key:
        expected = {key: expected[key] for key in args.key}
        actual = {key: actual[key] for key in args.key}
    report = json.dumps(compare_dump(actual, expected), indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
