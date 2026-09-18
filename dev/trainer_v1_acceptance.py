"""Native model acceptance against a global analytical-loss cotangent oracle.

Run ``oracle`` at DP=TP=CP=1, then ``zero`` on each target topology, passing the
oracle's .pt file as --reference. The oracle bypasses the callback tensor bridge
and differentiates native model outputs with hand-derived global cotangents.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import torch
import torch.distributed as dist
from trainer_rank_diag import rank0_checked
from trainer_rank_support import load_random_checkpoints

from art.trainer_rank import ForwardInput, TrainerRank


def _requests(checkpoint, offset, lengths, options=None):
    leaves = []
    for index, length in enumerate(lengths):
        tokens = (torch.arange(length) * 17 + offset + index * 103) % 30_000
        leaves.append(
            ForwardInput(
                input_tokens=tokens,
                target_tokens=(tokens * 7 + 3) % 30_000,
                checkpoint=checkpoint,
                hidden_states=True,
                **({"options": options} if options is not None else {}),
            )
        )
    # Nested complete roots, odd lengths, and an unused differentiable output.
    return [[leaves[0], *leaves[1:2]], leaves[2:]] if len(leaves) > 1 else leaves


def _leaves(tree):
    if isinstance(tree, (list, tuple)):
        return [leaf for item in tree for leaf in _leaves(item)]
    return [tree]


def _loss_and_cotangents(first, second):
    a = torch.cat([leaf.target_logprobs for leaf in _leaves(first)])
    b = torch.cat([leaf.target_logprobs for leaf in _leaves(second)])
    difference = a.mean() - 0.4 * b.mean()
    loss = difference.square() + 0.03 * a.square().mean() + 0.02 * b.square().mean()
    da = (2 * difference + 0.06 * a.detach()) / a.numel()
    db = (-0.8 * difference + 0.04 * b.detach()) / b.numel()
    tensors = [leaf.target_logprobs for leaf in _leaves(first) + _leaves(second)]
    gradients = list(
        da.detach().split([x.target_logprobs.numel() for x in _leaves(first)])
    )
    gradients += list(
        db.detach().split([x.target_logprobs.numel() for x in _leaves(second)])
    )
    return loss, tensors, gradients


def _cached_backward(rank, loss):
    with rank._gradient_transaction():
        packets = rank._forward_cotangent_collector().backward(loss)
        rank._forward_graph_cache().backward_many(
            [(packet.handle, packet.gradients) for packet in packets]
        )


def _canonical_gradients(rank, checkpoint):
    """Reduce once, then gather canonical LoRA shards without altering weights."""
    from art.megatron.lora import LoRA
    from art.megatron.weights.lora_publish import _merge_manifest_entries

    parameters = rank._checkpoint_slots[checkpoint].params
    reduced = rank._reduce_dynamic_grads(parameters, scale_grads=1.0)
    by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(parameters, reduced, strict=True)
    }
    local = {}
    for chunk in rank.runtime.model:
        for module in chunk.modules():
            if not isinstance(module, LoRA):
                continue
            for key, parameter, expert in module._export_items(
                rank._slot_ref(checkpoint)
            ):
                value = by_id[id(parameter)]
                value = value if expert is None else value[expert]
                local[key] = (
                    module._manifest_for_param(parameter),
                    value.T.float().cpu(),
                )
    if dist.get_rank() == 0:
        for name, custom in rank._checkpoint_slots[checkpoint].custom.items():
            for key, parameter in custom.value.named_parameters():
                local[f"custom.{name}.{key}"] = (
                    {"sharded": False, "shard_world_size": 1},
                    by_id[id(parameter)].float().cpu(),
                )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if dist.get_rank() != 0:
        return None
    groups = defaultdict(list)
    for shard in gathered:
        for key, entry in shard.items():
            groups[key].append(entry)
    return {
        key: _merge_manifest_entries(key, entries) for key, entries in groups.items()
    }


def _compare(actual, reference):
    if actual["gradients"].keys() != reference["gradients"].keys():
        raise AssertionError("Canonical gradient keys differ")
    torch.testing.assert_close(
        torch.tensor(actual["loss"]),
        torch.tensor(reference["loss"]),
        atol=0.03,
        rtol=0.003,
    )
    for output, expected in zip(actual["outputs"], reference["outputs"], strict=True):
        torch.testing.assert_close(output, expected, atol=0.03, rtol=0.003)
    rows, numerator, denominator = [], 0.0, 0.0
    for key, value in actual["gradients"].items():
        expected = reference["gradients"][key]
        error = (value.double() - expected.double()).square().sum().item()
        scale = expected.double().square().sum().item()
        relative = (error / max(scale, 1e-24)) ** 0.5
        rows.append({"key": key, "relative_l2": relative})
        numerator += error
        denominator += scale
        if scale > 1e-12 and relative > 0.06:
            raise AssertionError(
                f"{key}: gradient relative L2 {relative:.6f} exceeds 0.06"
            )
    relative = (numerator / max(denominator, 1e-24)) ** 0.5
    if denominator <= 1e-12 or relative > 0.03:
        raise AssertionError(
            f"global gradient relative L2 {relative:.6f}, reference norm² {denominator}"
        )
    return {"gradient_relative_l2": relative, "per_parameter": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["oracle", "control", "zero", "rank", "retained"]
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--retention", choices=["gpu", "cpu", "replay"], default="gpu")
    parser.add_argument("--output-device", choices=["model", "cpu"], default="model")
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Physical control uses replicated-root batching",
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Include a registered linear-head analytical oracle",
    )
    parser.add_argument("--cuda-values", action="store_true")
    parser.add_argument("--reverse-devices", action="store_true")
    parser.add_argument("--split-head-backward", action="store_true")
    args = parser.parse_args()
    if args.split_head_backward and (
        not args.head or args.mode not in ("zero", "rank")
    ):
        parser.error("Split head backward requires --head and zero/rank mode")
    for axis in ("TENSOR_MODEL", "CONTEXT", "PIPELINE_MODEL"):
        os.environ.setdefault(f"ART_MEGATRON_{axis}_PARALLEL_SIZE", "1")
    os.environ["ART_TRAINER_RANK_TEST_HOOKS"] = "1"
    os.environ["ART_TRAINER_RANK_TEST_ANCHOR"] = "no_sharing"
    device = int(os.environ["LOCAL_RANK"])
    if args.reverse_devices:
        device = int(os.environ["LOCAL_WORLD_SIZE"]) - 1 - device
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        from megatron.core import parallel_state as ps

        from art.megatron.train import build_training_runtime

        if args.mode in ("oracle", "retained") and dist.get_world_size() != 1:
            raise ValueError(
                "The mathematical global-loss oracle requires one physical rank"
            )
        torch.manual_seed(90217)
        runtime = build_training_runtime(
            model_identifier=args.model,
            provider_configure=(lambda p: setattr(p, "num_layers", args.layers))
            if args.layers
            else None,
            print_env=dist.get_rank() == 0,
        )
        for chunk in runtime.model:
            chunk.eval()
        physical = TrainerRank(runtime)
        if args.mode == "control" and ps.get_data_parallel_world_size() != 1:
            raise ValueError("A physical topology control requires DP=1")
        (checkpoint,) = load_random_checkpoints(
            runtime, physical, 1, base_model=args.model, lora_rank=2
        )
        options = None
        if args.mode == "retained":
            from art.trainer_rank import ForwardOptions

            options = ForwardOptions(
                backward_state=args.retention,
                output_device=args.output_device,
                stale_gradient_corrections=(),
            )
        first = _requests(checkpoint, 29, [17, 23, 31], options)
        # One complete root forces an empty DP partition when DP > 1.
        second = _requests(checkpoint, 113, [19], options)
        callbacks = 0
        cache_before_backward = None
        hidden_size = runtime.provider.hidden_size

        def head_factory():
            head = torch.nn.Linear(
                hidden_size,
                1,
                bias=False,
                device=physical.device if args.cuda_values else None,
            )
            with torch.no_grad():
                head.weight.copy_(
                    torch.linspace(-1, 1, hidden_size).reshape(1, -1) / hidden_size**0.5
                )
            return head

        def callback(rank):
            nonlocal callbacks, cache_before_backward
            callbacks += 1
            if args.cuda_values:
                for request in _leaves(first) + _leaves(second):
                    request.input_tokens = request.input_tokens.to(rank.device)
                    assert request.target_tokens is not None
                    request.target_tokens = request.target_tokens.to(rank.device)
            head = (
                rank.module("validation_head", head_factory, checkpoint=checkpoint)
                if args.head
                else None
            )
            # The original main branch names the physical operation differently;
            # keeping it usable as an oracle permits before/after comparisons.
            forward = getattr(rank, "forward", None) or rank.dp_rank_forward
            if args.batched:
                batches = (
                    getattr(rank, "forward_batches", None) or rank.forward_micro_batches
                )

                def forward(roots):
                    return [
                        root
                        for batch in batches(roots, yield_empty=True)
                        for root in batch.outputs
                    ]

            one, two = forward(first), forward(second)
            loss, outputs, cotangents = _loss_and_cotangents(one, two)
            model_loss = loss
            split_losses = None
            report_outputs = list(outputs)
            if head is not None:
                hidden = [leaf.hidden_states for leaf in _leaves(one) + _leaves(two)]
                scale = 0.01 / len(hidden)
                if args.mode in ("oracle", "control"):
                    # Explicit dH and dW avoid the live-head/snapshot autograd
                    # machinery in the unchanged-source mathematical reference.
                    weight = (
                        rank._checkpoint_slots[checkpoint]
                        .custom["validation_head"]
                        .value.weight
                    )
                    scores = [value.float() @ weight.detach().T for value in hidden]
                    derivatives = [
                        2 * scale * score.detach() / score.numel() for score in scores
                    ]
                    outputs.extend(hidden)
                    cotangents.extend(
                        [
                            (gradient @ weight.detach()).to(value.dtype)
                            for value, gradient in zip(hidden, derivatives, strict=True)
                        ]
                    )
                    outputs.append(weight)
                    cotangents.append(
                        sum(
                            gradient.T @ value.detach().float()
                            for value, gradient in zip(hidden, derivatives, strict=True)
                        )
                    )
                else:
                    scores = [head(value.float()) for value in hidden]
                loss = loss + scale * sum(score.square().mean() for score in scores)
                if args.split_head_backward:
                    # Distinct head captures publish twice to the same targets;
                    # their sum retains the independent one-copy dH/dW oracle.
                    repeated = [head(value.float()) for value in hidden]
                    split_losses = (
                        loss / 2,
                        (model_loss + scale * sum(x.square().mean() for x in repeated))
                        / 2,
                    )
                report_outputs.extend(scores)
            if args.mode in ("oracle", "control"):
                # No callback packet/autograd bridge or loss autograd contributes
                # to the reference gradients.
                torch.autograd.backward(outputs, cotangents)
            elif args.mode == "retained":
                from art.trainer_rank import AdamParams

                fresh = forward(first)
                update_loss = -torch.cat(
                    [leaf.target_logprobs for leaf in _leaves(fresh)]
                ).mean()
                _cached_backward(rank, update_loss)
                metrics = rank.optim_step(
                    params=AdamParams(learning_rate=0.01, grad_clip_norm=0),
                    checkpoints=[checkpoint],
                )
                if metrics["update_successful"] != 1:
                    raise AssertionError(
                        f"Intervening optimizer update failed: {metrics}"
                    )
                # Replay must use captured tokens and historical adapter tensors.
                for request in _leaves(first) + _leaves(second):
                    request.input_tokens.fill_(999)
                cache = rank._forward_graph_cache()
                cache_before_backward = [
                    asdict(cache.state(handle)) for handle in cache.handles()
                ]
                rng = torch.cuda.get_rng_state()
                _cached_backward(rank, loss)
                if not torch.equal(rng, torch.cuda.get_rng_state()):
                    raise AssertionError(
                        "Retained/replayed backward changed ambient CUDA RNG"
                    )
                if cache.handles():
                    raise AssertionError("Consumed model graphs remained resident")
            elif args.mode == "rank":
                # Each logical DP rank owns the same complete local workload.
                # Sum its scaled loss/gradients to recover the one-copy oracle;
                # TP/CP replicas must not multiply either reduction.
                size = ps.get_data_parallel_world_size()
                if split_losses is None:
                    rank.backward(loss / size)
                else:
                    rank.backward(split_losses[0] / size, retain_graph=True)
                    rank.backward(split_losses[1] / size)
                loss = loss.detach().clone() / size
                rank.reduce(loss)
            else:
                if split_losses is None:
                    rank.backward(loss)
                else:
                    rank.backward(split_losses[0], retain_graph=True)
                    rank.backward(split_losses[1])
            return {
                "loss": loss.item(),
                "outputs": [value.detach().float().cpu() for value in report_outputs],
            }

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if args.mode in ("oracle", "control", "retained"):
            result = callback(physical)
        else:
            from art.trainer_rank import run_rank_callback

            callback_error = None
            try:
                wrapped = asyncio.run(
                    run_rank_callback(
                        physical,
                        callback,
                        mode="rank" if args.mode == "rank" else "zero",
                    )
                )
            except BaseException:
                callback_error = traceback.format_exc()
                print(callback_error, file=sys.stderr, flush=True)
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, callback_error)
            if any(errors):
                raise RuntimeError(
                    "Callback oracle failed before result collection:\n"
                    + "\n".join(error for error in errors if error is not None)
                )
            result = wrapped.value
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        counts = [None] * dist.get_world_size()
        dist.all_gather_object(counts, callbacks)
        devices = [None] * dist.get_world_size()
        dist.all_gather_object(devices, str(physical.device))
        if args.reverse_devices:
            local_size = int(os.environ["LOCAL_WORLD_SIZE"])
            expected_devices = [
                f"cuda:{local_size - 1 - rank % local_size}"
                for rank in range(dist.get_world_size())
            ]
            if devices != expected_devices:
                raise AssertionError(f"Device mapping {devices} != {expected_devices}")
        expected_counts = (
            [1] * dist.get_world_size()
            if args.mode == "control"
            else [1] + [0] * (dist.get_world_size() - 1)
        )
        if args.mode == "rank":
            is_leader = int(
                ps.get_tensor_model_parallel_rank() == 0
                and ps.get_context_parallel_rank() == 0
                and ps.get_pipeline_model_parallel_rank() == 0
            )
            dist.all_gather_object(expected_counts, is_leader)
        if counts != expected_counts:
            raise AssertionError(f"User callback counts are {counts}")
        gradients = _canonical_gradients(physical, checkpoint)
        measurements = [None] * dist.get_world_size()
        dist.all_gather_object(
            measurements,
            {
                "elapsed_seconds": elapsed,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                "process_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            },
        )

        def finish():
            result["gradients"] = gradients
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, args.output)
            comparison = None
            if args.reference:
                comparison = _compare(
                    result, torch.load(args.reference, weights_only=True)
                )
            metadata = {
                "mode": args.mode,
                "model": args.model,
                "layers": args.layers,
                "callback_counts": counts,
                "physical_devices": devices,
                "loss": result["loss"],
                "topology": {
                    "dp": ps.get_data_parallel_world_size(),
                    "tp": ps.get_tensor_model_parallel_world_size(),
                    "cp": ps.get_context_parallel_world_size(),
                },
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(sys.modules["art.trainer_rank"].__file__)
                    .resolve()
                    .parents[3],
                    text=True,
                ).strip(),
                "harness_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parent.parent,
                    text=True,
                ).strip(),
                "measurements": measurements,
                "comparison": comparison,
                "retention": args.retention if args.mode == "retained" else None,
                "output_device": args.output_device,
                "cache_before_backward": cache_before_backward,
                "batched_control": args.batched,
                "registered_head": args.head,
                "cuda_values": args.cuda_values,
                "reverse_devices": args.reverse_devices,
                "split_head_backward": args.split_head_backward,
            }
            args.output.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            print(json.dumps(metadata), flush=True)

        rank0_checked("trainer v1 global-loss acceptance", finish)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
