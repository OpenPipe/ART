"""Reserved-GPU oracle for registered-head admission and streamed cotangents."""

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import StorageWeakRef
from trainer_rank_support import load_random_checkpoints

from art.trainer_rank import (
    ForwardInput,
    ForwardOptions,
    TrainerRank,
    TrainerRankMemoryError,
    run_rank_callback,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(".env")
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    try:
        from art.megatron.train import build_training_runtime

        os.environ["ART_TRAINER_RANK_TEST_HOOKS"] = "1"
        torch.manual_seed(716)
        runtime = build_training_runtime(
            model_identifier="Qwen/Qwen3-0.6B",
            model_initialization="random",
            provider_configure=lambda provider: setattr(provider, "num_layers", 2),
            print_env=False,
        )
        rank = TrainerRank(runtime)
        (checkpoint,) = load_random_checkpoints(
            runtime, rank, 1, base_model="Qwen/Qwen3-0.6B", lora_rank=2
        )
        request = ForwardInput(
            input_tokens=torch.arange(32),
            hidden_states=True,
            checkpoint=checkpoint,
            options=ForwardOptions(
                backward_state="replay",
                output_device="cpu",
                stale_gradient_corrections=(),
            ),
        )
        # Warm native kernels before allocator measurements.
        rank.backward(rank.forward(request).hidden_states.float().sum())
        rank.zero_grad()
        incoming = []
        commit = rank._commit_versioned_gradients

        def record(gradients):
            incoming.extend(
                StorageWeakRef(gradient.untyped_storage())
                for _, _, _, gradient in gradients
            )
            return commit(gradients)

        rank._commit_versioned_gradients = record
        state = rank._version_state()
        publish = state._publish
        publications = []

        def check(prepared):
            assert all(reference.expired() for reference in incoming), (
                "GPU head cotangent survived until publication"
            )
            publications.append(len(incoming))
            publish(prepared)

        state._publish = check
        rows = []

        def callback(view):
            head = view.module(
                "head",
                lambda: torch.nn.Linear(rank.hidden_size, 4096, bias=False),
                checkpoint=checkpoint,
            )
            head.cpu()  # Match driver/client head placement; isolate worker staging.
            parameter = rank._checkpoint_slots[checkpoint].custom["head"].value.weight
            target_bytes = parameter.numel() * parameter.element_size()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            os.environ["ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES"] = str(
                torch.cuda.memory_allocated() + 2 * target_bytes
            )
            try:
                try:
                    view.forward(request)
                except TrainerRankMemoryError:
                    pass
                else:
                    raise AssertionError(
                        "Known registered head staging was not reserved: "
                        + json.dumps(rank.last_forward_telemetry())
                    )
            finally:
                os.environ.pop("ART_TRAINER_RANK_TEST_MEMORY_LIMIT_BYTES", None)
            # Two complete outstanding model roots, each using the same head
            # eight times. Packet source tensors stay on the caller's CPU.
            outputs = [view.forward(request) for _ in range(2)]
            losses = [
                sum(
                    head(output.hidden_states.float().mean(0)).sum() * scale
                    for scale in range(1, 9)
                )
                for output in outputs
            ]
            reserved = rank._lora_gradient_staging_bytes(rank._slot_ref(checkpoint))
            workspace = max(
                rank._forward_graph_cache().state(handle).restore_workspace_bytes
                for handle in rank._forward_graph_cache().handles()
            )
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for loss in losses:
                view.backward(loss)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - baseline
            print(
                "HEAD_PEAK="
                + json.dumps(
                    dict(
                        peak=peak,
                        reserved=reserved,
                        workspace=workspace,
                        publications=publications,
                    )
                ),
                flush=True,
            )
            assert publications == [8, 16]
            assert peak <= reserved + workspace
            expected = 36 * sum(
                output.hidden_states.float().mean(0).detach().cpu()
                for output in outputs
            )
            torch.testing.assert_close(
                parameter.grad.cpu(),
                expected.expand(parameter.shape),
                rtol=1e-5,
                atol=1e-5,
            )
            rows.append(
                dict(
                    target_bytes=target_bytes,
                    staging_reserved_bytes=reserved,
                    restore_workspace_bytes=workspace,
                    backward_peak_bytes=peak,
                    publications=publications,
                    outstanding_roots=2,
                    captures_per_root=8,
                )
            )

        asyncio.run(run_rank_callback(rank, callback, mode="zero"))
        assert not rank._forward_graph_cache().handles()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
        print("HEAD_MEMORY=" + json.dumps(rows), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
