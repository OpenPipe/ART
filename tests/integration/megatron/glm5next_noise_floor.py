"""CPU HF dtype floor; native comparisons consume only completed HF captures.

Run each fixture/dtype independently with M1's pinned interpreter. Repeated
documents are reused only when both saved input and position IDs are identical.
No original dump or historical qualification result is modified.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import time

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from tests.integration.megatron.glm5next_parity_diagnostics import margin_top1_gate

# M1's minimal interpreter intentionally has no ART application dependencies.
_spec = importlib.util.spec_from_file_location(
    "glm_noise_oracle",
    Path(__file__).resolve().parents[3] / "src/art/megatron/glm5next/hf_oracle.py",
)
assert _spec is not None and _spec.loader is not None
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)
cpu_reference_kernels = oracle.cpu_reference_kernels
load_fixture = oracle.load_fixture
logprob_metrics = oracle.logprob_metrics
sha256_file = oracle.sha256_file


@contextmanager
def bounded_cpu_kernels():
    """Same HF fallback with independent heads batched to bound its workspace.

    HF allocates [B,H,chunks,64,64,128] decay intermediates. Eight heads at a
    time and 512 tokens per call preserve every within-head reduction and all
    64-token chunk boundaries, passing the exact recurrent state between calls.
    """
    from transformers.models.glm5_next import modeling_glm5_next as modeling

    with cpu_reference_kernels():
        original = modeling.chunk_kimi_delta_attention

        def bounded(query, key, value, g, beta, **kwargs):
            if query.shape[1] <= 512:
                return original(query, key, value, g, beta, **kwargs)
            outputs, states = [], []
            initial = kwargs.pop("initial_state", None)
            final_state = kwargs.pop("output_final_state", False)
            for first in range(0, query.shape[2], 8):
                parts = []
                state = None if initial is None else initial[:, first : first + 8]
                for start in range(0, query.shape[1], 512):
                    interval = (
                        slice(None),
                        slice(start, start + 512),
                        slice(first, first + 8),
                    )
                    output, state = original(
                        query[interval],
                        key[interval],
                        value[interval],
                        g[interval],
                        beta[interval],
                        initial_state=state,
                        output_final_state=True,
                        **kwargs,
                    )
                    parts.append(output)
                outputs.append(torch.cat(parts, 1))
                if final_state:
                    states.append(state)
            return torch.cat(outputs, 2), torch.cat(states, 1) if states else None

        modeling.chunk_kimi_delta_attention = bounded
        try:
            yield
        finally:
            modeling.chunk_kimi_delta_attention = original


def cpu_model(args, dtype):
    if args.fixture != "F8" or dtype != torch.float32:
        return load_fixture(args.fixture, root=args.fixtures, dtype=dtype, verify=False)
    # The full FP32 F8 weights alone exceed the authorized 128 GiB per-Python
    # RSS ceiling. Accelerate's stock hooks stage frozen weights from local disk.
    model, info = oracle.hf_model_class().from_pretrained(
        args.fixtures / args.fixture,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
        output_loading_info=True,
        device_map={"model.language_model": "disk", "lm_head": "cpu"},
        offload_folder=str(args.output / "F8-fp32-offload"),
        offload_state_dict=True,
    )
    assert not any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ), info
    # Check staged tensors after Accelerate has materialized them, rather than
    # inferring the active dtype from meta placeholders or model.config.
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        original = hook.pre_forward

        def checked(module, *a, _original=original, **kw):
            result = _original(module, *a, **kw)
            for value in (
                *module.parameters(recurse=False),
                *module.buffers(recurse=False),
            ):
                if value.is_floating_point():
                    assert value.dtype == torch.float32, (
                        "Offloaded FP32 model staged a lower-precision weight"
                    )
            return result

        hook.pre_forward = checked
    return model.eval()


def capture(model, ids: torch.Tensor, positions: torch.Tensor) -> dict:
    with torch.inference_mode(), bounded_cpu_kernels():
        hidden = model.model.language_model(
            input_ids=ids, position_ids=positions, use_cache=False
        ).last_hidden_state
        probabilities, values, indices = [], [], []
        for first in range(0, ids.shape[1] - 1, 64):
            last = min(first + 64, ids.shape[1] - 1)
            logits = model.lm_head(hidden[:, first:last]).float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Nonfinite HF logits")
            probabilities.append(
                logits.gather(-1, ids[:, first + 1 : last + 1, None]).squeeze(-1)
                - logits.logsumexp(-1)
            )
            top = logits.topk(2)
            values.append(top.values)
            indices.append(top.indices[..., 0])
    return {
        "input_ids": ids,
        "position_ids": positions,
        "target_logprobs": torch.cat(probabilities, 1).contiguous(),
        "top2.logits": torch.cat(values, 1).contiguous(),
        "top1": torch.cat(indices, 1).contiguous(),
    }


def measure(args) -> None:
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out = args.output / args.fixture / args.dtype
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted(
        (args.fixtures / "dumps" / args.fixture).glob("packed-*-doc*.safetensors")
    )
    if args.document:
        sources = [path for path in sources if path.name in args.document]
        if len(sources) != len(set(args.document)):
            raise ValueError("Requested document not in the exact M1 fixture")
    rows, cache = [], {}
    model = None
    for path in sources:
        with safe_open(path, framework="pt", device="cpu") as stream:
            ids, positions = (
                stream.get_tensor(k) for k in ("input_ids", "position_ids")
            )
        digest = hashlib.sha256(
            ids.numpy().tobytes() + positions.numpy().tobytes()
        ).hexdigest()
        destination = out / path.name
        started = time.monotonic()
        if not destination.exists():
            if digest in cache:
                os.link(cache[digest], destination)
            else:
                if model is None:
                    print(f"Loading {args.fixture} {args.dtype}", flush=True)
                    model = cpu_model(args, dtype)
                    model.requires_grad_(False)
                    assert all(
                        p.dtype == dtype
                        or (dtype == torch.bfloat16 and p.dtype == torch.float32)
                        for p in model.parameters()
                    ), "Not a requested-dtype model"
                    print(
                        f"Loaded {sum(p.numel() for p in model.parameters())} parameters",
                        flush=True,
                    )
                save_file(capture(model, ids, positions), str(destination))
        cache.setdefault(digest, destination)
        saved = load_file(str(destination))
        assert torch.equal(saved["input_ids"], ids) and torch.equal(
            saved["position_ids"], positions
        )
        assert all(bool(torch.isfinite(v).all()) for v in saved.values())
        row = {
            "document": path.name,
            "tokens": ids.numel(),
            "input_sha256": digest,
            "seconds": time.monotonic() - started,
            "capture_sha256": sha256_file(destination),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        (out / "receipt.json").write_text(
            json.dumps(
                {
                    "fixture": args.fixture,
                    "dtype": args.dtype,
                    "torch": torch.__version__,
                    "cpu_kda_head_batch": 8,
                    "cpu_kda_token_batch": 512,
                    "threads": args.threads,
                    "peak_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    / 2**20,
                    "fp32_F8_disk_offload": args.fixture == "F8"
                    and args.dtype == "fp32",
                    "rows": rows,
                    "cuda_initialized": torch.cuda.is_initialized(),
                },
                indent=2,
            )
            + "\n"
        )


def compare(args) -> None:
    rows = []
    for fixture in ("F4", "F8"):
        for source in sorted(
            (args.fixtures / "dumps" / fixture).glob("packed-*-doc*.safetensors")
        ):
            path = args.output / fixture / "fp32" / source.name
            if (
                not path.exists()
                or not (args.output / fixture / "bf16" / source.name).exists()
            ):
                rows.append(
                    {"fixture": fixture, "document": source.name, "status": "NOT_RUN"}
                )
                continue
            bf16 = load_file(str(args.output / fixture / "bf16" / path.name))
            fp32 = load_file(str(path))
            native_path = (
                args.native
                / fixture
                / path.name.replace(".safetensors", "-rank0.safetensors")
            )
            with safe_open(native_path, framework="pt", device="cpu") as stream:
                native = {
                    k: stream.get_tensor(k)
                    for k in ("target_logprobs", "top1", "top2.logits")
                }
            native_finite = {}
            for rank in (0, 1):
                rank_path = native_path.with_name(
                    native_path.name.replace("rank0", f"rank{rank}")
                )
                with safe_open(rank_path, framework="pt", device="cpu") as stream:
                    native_finite[str(rank)] = all(
                        bool(torch.isfinite(stream.get_tensor(k)).all())
                        for k in stream.keys()
                    )
            assert torch.equal(bf16["input_ids"], fp32["input_ids"])
            count = fp32["target_logprobs"].numel()
            baseline = logprob_metrics(
                bf16["target_logprobs"].reshape(-1), fp32["target_logprobs"].reshape(-1)
            )
            measured = logprob_metrics(
                native["target_logprobs"].reshape(-1),
                fp32["target_logprobs"].reshape(-1),
            )
            finite = all(native_finite.values()) and all(
                bool(torch.isfinite(v).all())
                for data in (bf16, fp32, native)
                for v in data.values()
            )
            ratios = {
                k: measured[k] / baseline[k]
                if baseline[k]
                else (0 if measured[k] == 0 else None)
                for k in ("logprob_abs_mean", "logprob_abs_p99")
            }
            rows.append(
                {
                    "fixture": fixture,
                    "status": "MEASURED",
                    "document": path.name,
                    "positions": count,
                    "baseline": baseline,
                    "baseline_top1": margin_top1_gate(bf16, fp32),
                    "native": measured,
                    "native_top1": margin_top1_gate(native, fp32),
                    "ratios": ratios,
                    "finite": finite,
                    "native_finite_by_rank": native_finite,
                    "report_only_new_gate": finite
                    and all(measured[k] <= 2 * baseline[k] for k in ratios),
                }
            )
    (args.output / "comparison.json").write_text(
        json.dumps(
            {
                "multiplier": 2.0,
                "report_only": True,
                "rows": rows,
                "cuda_initialized": torch.cuda.is_initialized(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Compared {len(rows)} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "compare"))
    parser.add_argument("--fixture", choices=("F4", "F8"), default="F4")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--fixtures", type=Path, default=Path("/tmp/stark-gpu/glm-fixtures")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--document",
        action="append",
        help="Exact saved M1 document filename (repeatable)",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Explicit CUDA_VISIBLE_DEVICES='' required")
    torch.set_num_threads(args.threads)
    (measure if args.mode == "measure" else compare)(args)
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
