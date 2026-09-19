"""Paired raw/logical facade forward-backward timings on identical frozen weights."""

import argparse
import asyncio
from collections import defaultdict
import cProfile
from functools import wraps
import json
import os
from pathlib import Path
import pstats
import statistics
import time

from dotenv import load_dotenv
import torch
import torch.distributed as dist
from trainer_rank_support import load_random_checkpoints

from art.trainer_rank import ForwardInput, ForwardOptions, TrainerRank, _commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    load_dotenv(".env")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        world = dist.get_world_size()
        for axis in ("TENSOR_MODEL", "CONTEXT", "DATA", "PIPELINE_MODEL"):
            os.environ[f"ART_MEGATRON_{axis}_PARALLEL_SIZE"] = str(
                world if axis == "DATA" else 1
            )
        os.environ["ART_TRAINER_RANK_TEST_HOOKS"] = "1"
        os.environ["ART_TRAINER_RANK_TEST_ANCHOR"] = "no_sharing"
        from art.megatron.train import build_training_runtime

        torch.manual_seed(90217)
        runtime = build_training_runtime(
            model_identifier="Qwen/Qwen3-0.6B",
            provider_configure=lambda p: setattr(p, "num_layers", args.layers),
            print_env=False,
        )
        for chunk in runtime.model:
            chunk.eval()
        physical = TrainerRank(runtime)
        (checkpoint,) = load_random_checkpoints(
            runtime, physical, 1, base_model="Qwen/Qwen3-0.6B", lora_rank=2
        )
        options = ForwardOptions(
            backward_state="gpu", output_device="model", stale_gradient_corrections=()
        )
        metrics = defaultdict(float)
        profiles = defaultdict(cProfile.Profile)
        for cls, name in (
            (_commands._Executor, "_packet"),
            (_commands._Executor, "_gather_outputs"),
            (_commands._RankView, "_place_outputs"),
            (_commands._RankView, "_attach"),
            (type(physical), "_capture_forward_options"),
            (type(physical), "_plan_admissible_forward"),
            (type(physical), "_execute_graph_group"),
            (type(physical), "_run_flat_plan_with_memory_tracking"),
        ):
            original = getattr(cls, name)

            @wraps(original)
            def measured(*a, _original=original, _name=name, **kw):
                start = time.perf_counter()
                try:
                    return _original(*a, **kw)
                finally:
                    metrics[_name] += time.perf_counter() - start

            setattr(cls, name, measured)

        rows, references = [], {}
        for kind in ("logprobs", "hidden"):
            requests = []
            for index in range(4):
                tokens = (torch.arange(args.tokens) * 17 + index * 103) % 30000
                requests.append(
                    ForwardInput(
                        input_tokens=tokens,
                        target_tokens=(tokens * 7 + 3) % 30000
                        if kind == "logprobs"
                        else None,
                        hidden_states=kind == "hidden",
                        checkpoint=checkpoint,
                        options=options,
                    )
                )
            inputs = [requests[:2], requests[2:]]

            local_inputs = inputs if world == 1 else [inputs[dist.get_rank()]]

            def run(view, mode, repetitions):
                for repetition in repetitions:
                    view.zero_grad()
                    metrics.clear()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    profile = (
                        profiles[kind, mode]
                        if args.profile and repetition >= 2
                        else None
                    )
                    if profile is not None:
                        profile.enable()
                    try:
                        outputs = view.forward(
                            inputs if mode == "zero" else local_inputs
                        )
                    finally:
                        if profile is not None:
                            profile.disable()
                    torch.cuda.synchronize()
                    forward_seconds = time.perf_counter() - started
                    values = [
                        output.target_logprobs
                        if kind == "logprobs"
                        else output.hidden_states
                        for root in outputs
                        for output in root
                    ]
                    loss = sum(
                        value.float().square().mean()
                        if kind == "hidden"
                        else value.float().mean()
                        for value in values
                    )
                    started = time.perf_counter()
                    view.backward(loss)
                    torch.cuda.synchronize()
                    backward_seconds = time.perf_counter() - started
                    row = dict(
                        physical_rank=dist.get_rank(),
                        mode=mode,
                        output=kind,
                        iteration=repetition - 2,
                        forward_seconds=forward_seconds,
                        backward_seconds=backward_seconds,
                        total_seconds=forward_seconds + backward_seconds,
                        output_bytes=sum(v.numel() * v.element_size() for v in values),
                        loss=loss.item(),
                        stages=dict(metrics),
                        telemetry=physical.last_forward_telemetry(),
                    )
                    if repetition == 1:
                        gradient = torch.cat(
                            [
                                p.grad.detach().float().reshape(-1)
                                for p in physical._checkpoint_slots[checkpoint].params
                                if p.grad is not None
                            ]
                        )
                        if mode == "native":
                            references[kind] = gradient.clone()
                        row["gradient_relative_l2"] = (
                            (gradient - references[kind]).norm()
                            / references[kind].norm().clamp_min(1e-20)
                        ).item()
                    if repetition >= 1:
                        rows.append(row)
                        print("FACADE=" + json.dumps(row), flush=True)
                    del outputs, values, loss
                if physical._forward_graph_cache().handles():
                    raise AssertionError("Unconsumed physical graph")

            def measure(mode, repetitions):
                if mode == "native":
                    run(physical, mode, repetitions)
                else:
                    asyncio.run(
                        _commands.run_rank_callback(
                            physical,
                            lambda view: run(view, mode, repetitions),
                            mode=mode,
                        )
                    )
                dist.barrier()

            modes = ("native", "rank", "zero")
            for mode in modes:
                measure(mode, range(2))
            for repetition in range(args.rounds):
                offset = repetition % len(modes)
                for mode in modes[offset:] + modes[:offset]:
                    measure(mode, (repetition + 2,))
        all_rows = [None] * world
        dist.all_gather_object(all_rows, rows)
        if dist.get_rank() != 0:
            return
        rows = [row for peer in all_rows for row in peer]
        summary = {
            f"{kind}/{mode}": {
                key: statistics.median(
                    max(
                        row[key]
                        for row in rows
                        if row["output"] == kind
                        and row["mode"] == mode
                        and row["iteration"] == iteration
                    )
                    for iteration in range(args.rounds)
                )
                for key in ("forward_seconds", "backward_seconds", "total_seconds")
            }
            for kind in ("logprobs", "hidden")
            for mode in ("native", "rank", "zero")
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(dict(dp=world, summary=summary, rows=rows), indent=2)
        )
        for (kind, mode), profile in profiles.items():
            prefix = args.output.with_suffix(f".{kind}-{mode}")
            profile.dump_stats(str(prefix) + ".pstats")
            with open(str(prefix) + ".txt", "w") as stream:
                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats(100)
        print("FACADE_SUMMARY=" + json.dumps(summary), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
