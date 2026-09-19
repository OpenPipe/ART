"""Fixed-workload native throughput and learning canary, including delayed graphs.

Run baseline on unchanged source, then gpu/cpu/replay with its tensor artifact as
--reference. Delayed cpu/replay instead use delayed gpu as their schedule oracle.
Use identical model, seed, lengths and compiler settings for all compared arms.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import weakref

from dotenv import load_dotenv
import torch
import torch.distributed as dist
from trainer_rank_support import load_random_checkpoints
from trainer_v1_acceptance import _cached_backward, _leaves

from art.trainer_rank import AdamParams, ForwardInput, TrainerRank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "gpu", "cpu", "replay"])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", type=int, default=0, help="0 keeps full model")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--leaves", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--explicit-cotangents", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("--delayed", action="store_true")
    parser.add_argument("--output-device", choices=["model", "cpu"], default="model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.delayed and args.mode == "baseline":
        parser.error("Delayed graphs require v1; use gpu as the delayed oracle")
    load_dotenv(".env")
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        import art.megatron.flex_attn.compiled as flex

        flex._FORCED_FLEX_BACKEND = "TRITON"
        flex._FORCED_FLEX_KERNEL_OPTIONS = {"BACKEND": "TRITON"}
        flex.dense_compiled_flex_attention = flex.triton_dense_compiled_flex_attention
        flex.sparse_compiled_flex_attention = flex.triton_sparse_compiled_flex_attention
    for axis in ("TENSOR_MODEL", "CONTEXT", "DATA", "PIPELINE_MODEL"):
        os.environ[f"ART_MEGATRON_{axis}_PARALLEL_SIZE"] = "1"
    os.environ["ART_TRAINER_RANK_TEST_HOOKS"] = "1"
    os.environ["ART_TRAINER_RANK_TEST_ANCHOR"] = "no_sharing"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 1:
            raise ValueError("This paired benchmark requires one physical rank")
        from art.megatron.train import build_training_runtime

        torch.manual_seed(90217)
        runtime = build_training_runtime(
            model_identifier=args.model,
            provider_configure=(lambda p: setattr(p, "num_layers", args.layers))
            if args.layers
            else None,
            print_env=False,
        )
        for chunk in runtime.model:
            chunk.eval()
        rank = TrainerRank(runtime)
        (checkpoint,) = load_random_checkpoints(
            runtime, rank, 1, base_model=args.model, lora_rank=2
        )
        forward = getattr(rank, "forward", None) or rank.dp_rank_forward
        options = None
        if args.mode != "baseline":
            from art.trainer_rank import ForwardOptions

            options = ForwardOptions(
                backward_state=args.mode,
                output_device=args.output_device,
                stale_gradient_corrections=(),
            )

        def inputs(offset=0, no_grad=False):
            leaves = []
            for index in range(args.leaves):
                tokens = (torch.arange(args.tokens) * 17 + index * 103 + offset) % 30000
                leaves.append(
                    ForwardInput(
                        input_tokens=tokens,
                        target_tokens=(tokens * 7 + 3) % 30000,
                        hidden_states=True,
                        checkpoint=checkpoint,
                        no_grad=no_grad,
                        **({"options": options} if options is not None else {}),
                    )
                )
            return [leaves[:2], leaves[2:]]

        boundary_outputs = {}

        def loss(outputs):
            tensors = [value.target_logprobs for value in _leaves(outputs)]
            value = -torch.cat(tensors).mean()
            if args.explicit_cotangents and value.requires_grad:
                boundary_outputs[id(value)] = tensors
            return value

        def backward(value):
            if args.mode == "baseline":
                if args.explicit_cotangents:
                    tensors = boundary_outputs.pop(id(value))
                    cotangents = torch.autograd.grad(value, tensors, retain_graph=True)
                    torch.autograd.backward(tensors, cotangents)
                else:
                    value.backward()
            else:
                _cached_backward(rank, value)

        parameters = rank._checkpoint_slots[checkpoint].params
        counters = {"physical_forwards": 0}
        record_refs, version_refs = [], []

        native_forward = rank._forward_packed

        def count_forward(*call_args, **kwargs):
            counters["physical_forwards"] += 1
            return native_forward(*call_args, **kwargs)

        rank._forward_packed = count_forward

        def cache_metrics():
            if args.mode == "baseline":
                return {"states": [], "historical_lora_bytes": 0}
            cache = rank._forward_graph_cache()
            record_refs.extend(
                weakref.ref(record) for record in cache._records.values()
            )
            storages = {}
            for version in rank._version_state().lora.values():
                version_refs.append(weakref.ref(version))
                for slot in version.slots.values():
                    for parameter in slot.parameters():
                        storage = parameter.untyped_storage()
                        storages[(parameter.device, storage.data_ptr())] = (
                            storage.nbytes()
                        )
            return {
                "states": [asdict(cache.state(handle)) for handle in cache.handles()],
                "historical_lora_bytes": sum(storages.values()),
            }

        optimizer = AdamParams(learning_rate=args.learning_rate, grad_clip_norm=1)
        first_step_gradients = None
        optimizer_metrics = []

        def step(value):
            nonlocal first_step_gradients
            backward(value)
            if first_step_gradients is None:
                first_step_gradients = [
                    None if p.grad is None else p.grad.detach().float().cpu()
                    for p in parameters
                ]
            if args.no_update:
                rank.zero_grad()
                return
            metrics = rank.optim_step(params=optimizer, checkpoints=[checkpoint])
            optimizer_metrics.append(metrics)
            if metrics["update_successful"] != 1:
                raise AssertionError(f"Failed optimizer update: {metrics}")

        # Warm every selected path without altering the learning initial state.
        for _ in range(args.warmups):
            value = loss(forward(inputs()))
            backward(value)
            rank.zero_grad()
        with torch.no_grad():
            initial_eval = loss(forward(inputs(no_grad=True))).item()
        rows, losses = [], []
        for iteration in range(args.rounds):
            rank.zero_grad()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            counters["physical_forwards"] = 0
            started = time.perf_counter()
            old = loss(forward(inputs()))
            torch.cuda.synchronize()
            forward_seconds = time.perf_counter() - started
            diagnostic_started = time.perf_counter()
            capture = cache_metrics()
            losses.append(old.detach().item())
            diagnostic_seconds = time.perf_counter() - diagnostic_started
            started = time.perf_counter()
            if args.delayed:
                fresh = loss(forward(inputs(offset=19)))
                step(fresh)
                if args.delay_ms:
                    time.sleep(args.delay_ms / 1000)
            step(old)
            torch.cuda.synchronize()
            elapsed = forward_seconds + time.perf_counter() - started
            tokens = args.tokens * args.leaves * (2 if args.delayed else 1)
            row = {
                "iteration": iteration,
                "loss": losses[-1],
                "elapsed_seconds": elapsed,
                "excluded_diagnostic_seconds": diagnostic_seconds,
                "logical_tokens": tokens,
                "tokens_per_second": tokens / elapsed,
                "physical_forwards": counters["physical_forwards"],
                "active_graphs_after_backward": len(
                    rank._forward_graph_cache().handles()
                )
                if args.mode != "baseline"
                else 0,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "gpu_resident_bytes": torch.cuda.memory_allocated(),
                "process_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                **capture,
            }
            rows.append(row)
            print("BENCHMARK=" + json.dumps(row), flush=True)
        with torch.no_grad():
            final_eval = loss(forward(inputs(no_grad=True))).item()
        rank._forward_packed = native_forward
        del old, value
        if args.delayed:
            del fresh
        gc.collect()
        after_gc = cache_metrics()
        after_gc["gpu_resident_bytes"] = torch.cuda.memory_allocated()
        after_gc["live_captured_records"] = sum(
            ref() is not None for ref in record_refs
        )
        after_gc["live_captured_versions"] = sum(
            ref() is not None for ref in version_refs
        )
        artifact = {
            "losses": losses,
            "weights": [p.detach().float().cpu() for p in parameters],
            "first_step_gradients": first_step_gradients,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.output)
        comparison = None
        if args.reference:
            reference = torch.load(args.reference, weights_only=True)
            numerator = denominator = 0.0
            for value, expected in zip(
                artifact["weights"], reference["weights"], strict=True
            ):
                numerator += (value.double() - expected.double()).square().sum().item()
                denominator += expected.double().square().sum().item()
            comparison = {
                "weight_relative_l2": math.sqrt(numerator / max(denominator, 1e-24))
            }
        metadata = {
            "arguments": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(sys.modules["art.trainer_rank"].__file__).resolve().parents[3],
                text=True,
            ).strip(),
            "harness_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
                text=True,
            ).strip(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "layers": runtime.provider.num_layers,
            "dtype": str(next(runtime.model[0].parameters()).dtype),
            "initial_fixed_objective": initial_eval,
            "final_fixed_objective": final_eval,
            "median_tokens_per_second": statistics.median(
                row["tokens_per_second"] for row in rows
            ),
            "comparison": comparison,
            "optimizer_metrics": optimizer_metrics,
            "after_gc": after_gc,
            "rows": rows,
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps(metadata), flush=True)
        if args.reference:
            torch.testing.assert_close(
                torch.tensor(losses),
                torch.tensor(reference["losses"]),
                atol=0.03,
                rtol=0.003,
            )
            if comparison["weight_relative_l2"] > 0.005:
                raise AssertionError(f"Learning trajectory changed: {comparison}")
        if not math.isfinite(final_eval) or (
            not args.no_update and final_eval >= initial_eval
        ):
            raise AssertionError(
                f"Fixed objective did not improve: {initial_eval} -> {final_eval}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
