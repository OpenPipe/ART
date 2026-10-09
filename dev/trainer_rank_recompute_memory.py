"""Measure cold unsplit admission and native CUDA peaks for one recompute mode.

Run with torchrun; use a fresh process per mode/topology. Random weights keep
the native model geometry and kernels without downloading a checkpoint. This
measures memory, not pretrained-model correctness. Like public admission, try
the minimum-memory layout before refusing an unsplit request; never split or
bypass the budget. Each repetition clears the learned memory profile, while
sample 0 includes cold compilation/autotuning. Output is one JSONL row per rank.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from dotenv import load_dotenv
import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--mode", choices=("full", "selective", "none"), required=True)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--reported-pair", action="store_true")
    parser.add_argument(
        "--pairs", action="store_true", help="Two sequences at each token length"
    )
    parser.add_argument(
        "--sequences", type=int, default=0, help="Override sequence count per batch"
    )
    parser.add_argument("--prefix-fraction", type=float, default=0.3)
    parser.add_argument("--modules", nargs="+", default=["core_attn"])
    parser.add_argument(
        "--components", action="store_true", help="Attribute eager layer allocations"
    )
    parser.add_argument(
        "--concentrate-routing",
        action="store_true",
        help="Concentrate 31/32 of tokens on the first expert rank",
    )
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.layers < 0 or any(n < 1 for n in args.tokens):
        parser.error("repeat/tokens must be positive and layers nonnegative")
    if args.sequences < 0:
        parser.error("sequences must be nonnegative")
    if not 0 <= args.prefix_fraction < 1:
        parser.error("prefix-fraction must be in [0, 1)")
    load_dotenv(".env")
    from trainer_rank_support import load_random_checkpoints

    from art.megatron import train
    from art.trainer_rank import ForwardInput, TrainerRank

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

    def configure(provider):
        if args.layers:
            provider.num_layers = args.layers
        provider.recompute_granularity = None if args.mode == "none" else args.mode
        provider.recompute_method = "uniform" if args.mode == "full" else None
        provider.recompute_num_layers = 1 if args.mode == "full" else None
        provider.recompute_modules = args.modules if args.mode == "selective" else []

    def emit(row):
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, {**facts, **row, "rank": dist.get_rank()})
        if dist.get_rank() == 0:
            with args.evidence.open("a") as stream:
                for item in gathered:
                    stream.write(json.dumps(item, default=str, sort_keys=True) + "\n")
            print("MEMORY_CALIBRATION " + json.dumps(gathered, default=str), flush=True)

    try:
        torch.manual_seed(913)
        runtime = train.build_training_runtime(
            model_identifier=args.model,
            model_initialization="random",
            provider_configure=configure,
            print_env=False,
        )
        if args.concentrate_routing:
            # Stress dispatch imbalance without replacing native routing/experts.
            for module in runtime.model[0].modules():
                if type(module).__name__ == "TopKRouter":

                    def gating(inputs, original=module.gating):
                        logits = original(inputs)
                        experts = logits.shape[-1]
                        ep = runtime.provider.expert_model_parallel_size
                        rows = torch.arange(
                            logits.numel() // experts, device=logits.device
                        ).reshape(*logits.shape[:-1], 1)
                        # Fully empty EP peers crash TE's grouped GEMM. Leave
                        # 1/32 of tokens on peers while stressing near-max load.
                        owner = (
                            torch.where(
                                rows % 32 == 0, 1 + (rows // 32) % max(1, ep - 1), 0
                            )
                            if ep > 1
                            else torch.zeros_like(rows)
                        )
                        expert = torch.arange(experts, device=logits.device)
                        selected = (expert >= owner * (experts // ep)) & (
                            expert
                            < owner * (experts // ep) + runtime.provider.moe_router_topk
                        )
                        return logits + selected * 10000

                    module.gating = gating
        for chunk in runtime.model:
            chunk.train()
        if args.components and runtime.transformer_layers_compiled:
            parser.error("--components requires ART_DISABLE_MEGATRON_COMPILE=1")
        rank = TrainerRank(runtime)
        [slot] = load_random_checkpoints(
            runtime, rank, 1, base_model=args.model, lora_rank=1
        )
        facts = {
            "schema": "art.dev.recompute_memory.v1",
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha": os.environ.get("ART_CALIBRATION_SOURCE_SHA")
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "model": args.model,
            "initialization": "random",
            "concentrated_routing": args.concentrate_routing,
            "lora_rank": 1,
            "mode": rank._recompute_granularity,
            "recompute_method": runtime.provider.recompute_method,
            "recompute_num_layers": runtime.provider.recompute_num_layers,
            "recompute_modules": runtime.provider.recompute_modules,
            "geometry": rank._geometry.as_dict(),
            "layers": rank._num_layers,
            "gdn_layers": rank._gdn_layers,
            "sequence_parallel": rank._sequence_parallel,
            "topology_dp_tp_cp_pp": rank._topology_key(),
            "parallel_shape": asdict(rank._parallel_shape),
            "dtype": str(next(runtime.model[0].parameters()).dtype),
            "device": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformer_layers_compiled": runtime.transformer_layers_compiled,
            "activation_config": {
                name: str(getattr(runtime.provider, name, None))
                for name in (
                    "bias_activation_fusion",
                    "use_te_activation_func",
                    "gated_linear_unit",
                    "activation_func",
                    "attention_output_gate",
                    "qk_layernorm",
                )
            },
        }
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        count = args.sequences or (2 if args.pairs else 1)
        workloads = [
            ([length] * count, int(length * args.prefix_fraction) if count > 1 else 0)
            for length in args.tokens
        ]
        if args.reported_pair:
            # Same logical/fully-shared packed counts as #913; synthetic IDs.
            workloads.append(([19221, 19222], 5733))
        generator = torch.Generator().manual_seed(913)
        for lengths, prefix in workloads:
            tokens = [
                torch.randint(100, 10000, (n,), generator=generator) for n in lengths
            ]
            for item in tokens[1:]:
                item[:prefix] = tokens[0][:prefix]
            requests = [
                ForwardInput(input_tokens=item, hidden_states=True) for item in tokens
            ]
            plan = rank._plan_flat_forward(requests, checkpoint=slot)
            memory_minimal = False
            for sample in range(args.repeat):
                rank.zero_grad()
                rank._memory_profiles.clear()
                gc.collect()
                torch.cuda.empty_cache()
                dist.barrier()
                torch.cuda.synchronize()
                check = rank._memory_check(plan)
                if not check.fits and not memory_minimal:
                    plan = rank._plan_flat_forward(
                        requests, checkpoint=slot, memory_minimal=True
                    )
                    memory_minimal = True
                    check = rank._memory_check(plan)
                row = {
                    "lengths": lengths,
                    "shared_prefix": prefix,
                    "logical_tokens": plan.logical_tokens,
                    "packed_tokens": plan.packed_tokens,
                    "grad_segment_count": plan.grad_segment_count,
                    "retained_tokens": rank._plan_retained_tokens(plan),
                    "output_bytes": plan.output_bytes,
                    "selected_max_depth": plan.selected_max_depth,
                    "memory_minimal": memory_minimal,
                    "sample": sample,
                    "admission": asdict(check),
                }
                if not check.fits:
                    emit({**row, "status": "refused"})
                    break
                baseline = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                components, handles, entries = [], [], {}
                if args.components:
                    from megatron.core.transformer.transformer_layer import (
                        TransformerLayer,
                    )

                    def enter(module, inputs):
                        entries[id(module)] = torch.cuda.memory_allocated()

                    def leave(name):
                        def record(module, inputs, output):
                            components.append(
                                {
                                    "name": name,
                                    "type": type(module).__name__,
                                    "input_shape": list(inputs[0].shape)
                                    if inputs and isinstance(inputs[0], torch.Tensor)
                                    else None,
                                    "retained_delta_bytes": torch.cuda.memory_allocated()
                                    - entries[id(module)],
                                }
                            )

                        return record

                    for name, layer in runtime.model[0].named_modules():
                        if isinstance(layer, TransformerLayer):
                            for part, module in (
                                ("layer", layer),
                                ("attention", layer.self_attention),
                                ("mlp", layer.mlp),
                            ):
                                handles.extend(
                                    (
                                        module.register_forward_pre_hook(enter),
                                        module.register_forward_hook(
                                            leave(f"{name}.{part}")
                                        ),
                                    )
                                )
                # Direct execution isolates the selected unsplit plan. It uses
                # the same native forward as public admission, with no split.
                outputs = rank._execute_flat_plan(plan)
                for handle in handles:
                    handle.remove()
                torch.cuda.synchronize()
                forward_peak = torch.cuda.max_memory_allocated()
                retained = torch.cuda.memory_allocated()
                forward_seconds = time.monotonic() - started
                terms = [
                    output.hidden_states.float().square().mean()
                    for output in outputs
                    if output.hidden_states is not None
                ]
                assert len(terms) == len(requests)
                loss = torch.stack(terms).sum()
                loss.backward()
                torch.cuda.synchronize()
                backward_peak = torch.cuda.max_memory_allocated()
                backward_seconds = time.monotonic() - started
                gradients = [
                    p.grad
                    for p in rank._checkpoint_slots[slot].params
                    if p.grad is not None
                ]
                emit(
                    {
                        **row,
                        "status": "measured",
                        "components": components,
                        "baseline_allocated_bytes": baseline,
                        "baseline_reserved_bytes": reserved,
                        "forward_peak_delta_bytes": forward_peak - baseline,
                        "retained_delta_bytes": retained - baseline,
                        "forward_backward_peak_delta_bytes": backward_peak - baseline,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "forward_seconds": forward_seconds,
                        "forward_backward_seconds": backward_seconds,
                        "finite_loss": bool(torch.isfinite(loss).item()),
                        "gradient_tensors": len(gradients),
                        "finite_gradients": bool(gradients)
                        and all(
                            bool(torch.isfinite(g).all().item()) for g in gradients
                        ),
                        "estimate_covers_forward": check.estimated_required_bytes
                        >= forward_peak - baseline,
                    }
                )
                del outputs, loss, terms, gradients
                rank.zero_grad()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
