"""Native complete-root admission, measured peaks and paired headroom timings."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import torch
import torch.distributed as dist
from trainer_v1_support import (
    _cached_backward,
    _leaves,
    build_rank,
    deterministic_kernels,
    load_checkpoint,
    nccl_group,
)

from art.trainer_rank import (
    ForwardInput,
    ForwardOptions,
    TrainerRankMemoryError,
)
from art.trainer_rank._impl import _SplitForwardPlan
from art.trainer_rank._memory_policy import host_memory_budget, placement_cost


def _backward(rank, outputs):
    loss = sum(
        output.hidden_states.float().mean() for output in _leaves(outputs)
    ).square()
    _cached_backward(rank, loss)
    return float(loss.detach().cpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(".env")
    if args.deterministic:
        deterministic_kernels()
    with nccl_group(int(os.environ.get("LOCAL_RANK", "0"))):
        if dist.get_world_size() != 1:
            raise ValueError("This admission oracle uses one physical rank")
        os.environ["ART_TRAINER_RANK_TEST_HOOKS"] = "1"
        os.environ["ART_TRAINER_RANK_TEST_ANCHOR"] = "no_sharing"

        def configure(provider):
            provider.num_layers = args.layers
            provider.recompute_granularity = None
            provider.recompute_method = None
            provider.recompute_num_layers = None
            provider.recompute_modules = []

        rank = build_rank(
            args.model,
            seed=913,
            model_initialization="random",
            provider_configure=configure,
        )
        checkpoint = load_checkpoint(rank, args.model)
        forward = getattr(rank, "forward", None) or rank.dp_rank_forward
        batches = getattr(rank, "forward_batches", None) or rank.forward_micro_batches
        options = lambda state, device: ForwardOptions(
            backward_state=state, output_device=device, stale_gradient_corrections=()
        )

        def root(state, device):
            leaves = [
                ForwardInput(
                    input_tokens=(torch.arange(args.tokens) * 17 + 103 * i) % 30000,
                    hidden_states=True,
                    checkpoint=checkpoint,
                    options=options(state, device),
                )
                for i in range(4)
            ]
            return [leaves[:2], leaves[2:]]

        # Learn both forward retention and caller backward peak with the existing
        # profiler, at the same physical child size later used by the split root.
        for _ in range(2):
            rank.zero_grad()
            for batch in batches([root("gpu", "model")[0][0]]):
                _backward(rank, batch.outputs)
        del batch

        rows = []
        gradients_by_label = {}
        child_peaks = []
        run_child = rank._run_flat_plan_with_memory_tracking

        def track(*call_args, **kwargs):
            result = run_child(*call_args, **kwargs)
            child_peaks.append(torch.cuda.max_memory_allocated())
            return result

        rank._run_flat_plan_with_memory_tracking = track

        def run(
            state,
            device,
            label,
            *,
            cap=None,
            chunks=None,
            compare=None,
            legacy_admission=False,
        ):
            rank.zero_grad()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            if cap is None:
                os.environ.pop("ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES", None)
            else:
                os.environ["ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES"] = str(
                    baseline + cap
                )
            child_peaks.clear()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            find = rank._find_admissible_forward
            enabled = rank._graph_memory_policy_enabled

            def fixed_split(requests, *, checkpoint, **_kwargs):
                plan, check = rank._admit_split_rung(
                    chunks,
                    requests,
                    [request.input_tokens for request in requests],
                    checkpoint=checkpoint,
                )
                assert plan is not None and check.fits
                return plan, check

            if chunks is not None:
                rank._find_admissible_forward = fixed_split
            if legacy_admission:
                # Isolate admission overhead while retaining identical graph
                # execution, kernels and the existing calibrated GPU profiler.
                rank._graph_memory_policy_enabled = lambda: False
            try:
                outputs = forward(root(state, device))
            finally:
                rank._find_admissible_forward = find
                rank._graph_memory_policy_enabled = enabled
            torch.cuda.synchronize()
            retained = torch.cuda.memory_allocated() - baseline
            assert [len(part) for part in outputs] == [2, 2]
            assert all(
                tuple(value.hidden_states.shape) == (args.tokens, rank._hidden_size)
                for value in _leaves(outputs)
            )
            cache = rank._forward_graph_cache()
            states = [asdict(cache.state(handle)) for handle in cache.handles()]
            loss = _backward(rank, outputs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = max([torch.cuda.max_memory_allocated(), *child_peaks]) - baseline
            gradients = [
                None if p.grad is None else p.grad.detach().float().cpu()
                for p in rank._checkpoint_slots[checkpoint].params
            ]
            gradients_by_label[label] = gradients
            if cache.handles():
                raise AssertionError("Consumed forward graph remained cached")
            row = dict(
                label=label,
                state=state,
                output_device=device,
                legacy_admission=legacy_admission,
                loss=loss,
                elapsed_seconds=elapsed,
                logical_tokens=4 * args.tokens,
                gpu_retained_bytes=retained,
                gpu_peak_bytes=peak,
                budget_bytes=cap,
                graph_states=states,
                telemetry=rank.last_forward_telemetry(),
                transfer_stats=asdict(cache.transfer_stats)
                if hasattr(cache, "transfer_stats")
                else None,
            )
            rows.append(row)
            print("NATIVE_MEMORY=" + json.dumps(row, default=str), flush=True)
            # Preserve measured evidence even when a correctness gate fails.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"rows": rows}, indent=2, default=str))
            torch.save(gradients_by_label, args.output.with_suffix(".gradients.pt"))
            if compare is not None:
                for actual, expected in zip(
                    gradients, gradients_by_label[compare], strict=True
                ):
                    if (actual is None) != (expected is None):
                        raise AssertionError(
                            "Used parameter set changed across policies"
                        )
                    if actual is not None:
                        torch.testing.assert_close(
                            actual, expected, atol=5e-5, rtol=0.02
                        )
            return row

        run("gpu", "model", "reference")
        # The cap is derived from the same measured/calibrated child costs used
        # by admission. It limits available memory, never replaces a model cost.
        leaves = _leaves(root("replay", "cpu"))
        children = tuple(rank._plan_flat_forward([leaf]) for leaf in leaves)
        split = _SplitForwardPlan(children, tuple((i,) for i in range(4)), 4)
        placements = [
            placement_cost((cost,), backward_state="replay", output_device="cpu")
            for _, _, cost, _ in rank._graph_memory_units(split)
        ]
        cap = sum(
            p.gpu_retained_bytes + p.gpu_backward_bytes for p in placements
        ) + max(
            p.gpu_required_bytes - p.gpu_retained_bytes - p.gpu_backward_bytes
            for p in placements
        )
        cap = int(cap * 1.05)
        try:
            run("gpu", "model", "must_refuse", cap=cap)
        except TrainerRankMemoryError:
            pass
        else:
            raise AssertionError("Constrained retained-GPU root was not refused")
        accepted = run("replay", "cpu", "constrained_replay", cap=cap)
        if accepted["telemetry"]["subforward_count"] <= 1:
            raise AssertionError(
                "Constrained complete root did not execute as split physical forwards"
            )
        if accepted["gpu_peak_bytes"] > cap:
            raise AssertionError("Native measured peak exceeded admission cap")
        # BF16 packed and split matmuls can differ numerically. Compare replay
        # against the identical admitted physical partition, with CPU reduction
        # on both arms; retain the packed reference as a separate diagnostic.
        run(
            "gpu",
            "cpu",
            "same_split_gpu",
            chunks=accepted["telemetry"]["subforward_request_indices"],
            compare="constrained_replay",
        )
        run(
            "cpu",
            "cpu",
            "same_split_cpu_offload",
            chunks=accepted["telemetry"]["subforward_request_indices"],
            compare="constrained_replay",
        )
        adaptive = run(
            "auto",
            "cpu",
            "constrained_measured_auto",
            cap=cap,
            chunks=accepted["telemetry"]["subforward_request_indices"],
            compare="constrained_replay",
        )
        evidence = adaptive["telemetry"]["fallback_costs"]
        if evidence["source"] != "measured_forward_and_transfers":
            raise AssertionError("Matching measured fallback evidence was not used")
        if {state["retention"] for state in adaptive["graph_states"]} != {
            evidence["preferred"]
        }:
            raise AssertionError("Admitted fallback did not follow measured costs")
        if adaptive["gpu_peak_bytes"] > cap:
            raise AssertionError(
                "Measured adaptive fallback peak exceeded admission cap"
            )
        # Paired alternating warmed rounds compare the default choice with the
        # forced retained path under headroom, using identical computation.
        for repetition in range(args.rounds):
            order = ("auto", "gpu", "legacy")
            for state in order if repetition % 2 == 0 else tuple(reversed(order)):
                run(
                    "gpu" if state == "legacy" else state,
                    "model",
                    f"headroom_{state}",
                    compare="reference",
                    legacy_admission=state == "legacy",
                )
        # An older, larger replay must retain its restore reservation even when
        # the next root is small enough to fit by itself.
        os.environ.pop("ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES", None)
        rank.zero_grad()
        old_outputs = forward(root("auto", "cpu"))
        cache = rank._forward_graph_cache()
        for handle in cache.handles():
            cache.evict(handle)
        gc.collect()
        torch.cuda.empty_cache()
        workspace = max(
            cache.state(handle).restore_workspace_bytes for handle in cache.handles()
        )
        small = root("replay", "cpu")[0][0]
        small_cost = next(rank._graph_memory_units(rank._plan_flat_forward([small])))[2]
        small_required = placement_cost(
            (small_cost,), backward_state="replay", output_device="cpu"
        ).gpu_required_bytes
        cap = workspace - 1024**2
        assert 0 < small_required < cap
        baseline = torch.cuda.memory_allocated()
        os.environ["ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES"] = str(baseline + cap)
        try:
            try:
                forward([small])
            except TrainerRankMemoryError:
                pass
            else:
                raise AssertionError("A new root consumed the older replay reservation")
        finally:
            os.environ.pop("ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES", None)
        torch.cuda.reset_peak_memory_stats()
        _backward(rank, old_outputs)
        torch.cuda.synchronize()
        restored_peak = torch.cuda.max_memory_allocated() - baseline
        assert restored_peak <= workspace
        assert not cache.handles()
        row = dict(
            label="prior_replay_reservation",
            old_restore_workspace_bytes=workspace,
            small_root_required_bytes=small_required,
            budget_bytes=cap,
            measured_restore_peak_bytes=restored_peak,
        )
        rows.append(row)
        print("NATIVE_MEMORY=" + json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "layers": args.layers,
                    "tokens_per_child": args.tokens,
                    "torch": torch.__version__,
                    "deterministic": args.deterministic,
                    "device": torch.cuda.get_device_name(),
                    "host_budget": asdict(host_memory_budget(local_world_size=1)),
                    "rows": rows,
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
