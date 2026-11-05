"""Full-model TrainerRank CP1/EP1 versus CP8/EP8, without optimizer updates."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist

from tests.integration.megatron.glm5next_qualification import sha, write


class LayerStream:
    """B0 only: CPU checkpoint, CUDA adapters/head and one frozen decoder layer.

    Copies preserve Parameter objects, dtypes and values. No math is replaced.
    B0 has no autograd graph and needs no DDP gradient buffers. This harness-only
    loader placement hook runs before any full-model CUDA materialization.
    """

    def __init__(self, chunks: Any, device: torch.device):
        self.device = device
        self.handles: list[Any] = []
        self.visits: list[int] = []
        self.layer_bytes: list[int] = []
        for wrapped in chunks:
            chunk = wrapped
            while hasattr(chunk, "module"):
                chunk = chunk.module
            layers = list(chunk.decoder.layers)
            frozen = {
                id(p)
                for layer in layers
                for p in layer.parameters()
                if not p.requires_grad
            }
            layer_buffers = {id(b) for layer in layers for b in layer.buffers()}
            for p in chunk.parameters():
                if id(p) not in frozen:
                    p.data = p.data.to(device)
            for module in chunk.modules():
                for name, b in module.named_buffers(recurse=False):
                    if id(b) not in layer_buffers:
                        module._buffers[name] = b.to(device)
            for index, layer in enumerate(layers):
                self.layer_bytes.append(
                    sum(
                        p.numel() * p.element_size()
                        for p in layer.parameters()
                        if not p.requires_grad
                    )
                )

                def pre(module: Any, args: Any, index: int = index) -> None:
                    assert not torch.is_grad_enabled(), (
                        "streaming reference is forward-only"
                    )
                    self.move(module, self.device)
                    self.visits.append(index)

                def post(module: Any, args: Any, result: Any) -> None:
                    self.move(module, torch.device("cpu"))

                self.handles.extend(
                    (
                        layer.register_forward_pre_hook(pre),
                        layer.register_forward_hook(post),
                    )
                )

    @staticmethod
    def move(module: Any, device: torch.device) -> None:
        for p in module.parameters():
            if not p.requires_grad:
                p.data = p.data.to(device)
        for child in module.modules():
            for name, b in child.named_buffers(recurse=False):
                child._buffers[name] = b.to(device)


def install_capture(trainer: Any, path: Path) -> None:
    """Capture the first document's first pass, with exact token UID joins.

    Hooks do not replace/disable compiled arithmetic. Captures are on the very
    pass supplying the comparison, avoiding a diagnostic-repeat confound.
    """
    original = trainer._decoder_hidden
    used = False

    def decoder(prepared: Any) -> Any:
        nonlocal used
        if used:
            return original(prepared)
        used = True
        records = {
            "token_uids": prepared.token_uids.detach().cpu(),
            "position_ids": prepared.position_ids.detach().cpu(),
        }
        handles = []

        def capture(module: Any, args: Any, result: Any, name: str) -> None:
            value = result[0] if isinstance(result, tuple) else result
            state = prepared.attention_state
            plan = getattr(state, "gdn_execution_plan", None)
            if plan is not None and plan.cp_size > 1:
                from art.megatron.gdn import operator

                layout = operator._gdn_runtime_state(state).hidden_layout
                uids = operator._local_layout_token_uids(
                    plan, layout, hidden_states=value, gdn=None
                )
            else:
                uids = prepared.token_uids.reshape(-1)
            # mHC can retain an explicit four-stream axis. It belongs to the
            # channel vector, never to token identity.
            records[name] = value.detach().cpu().clone().reshape(uids.numel(), -1)
            records[name + ".uids"] = uids.detach().cpu().reshape(-1)

        model = trainer.runtime.model[0]
        while hasattr(model, "module"):
            model = model.module
        for i, layer in enumerate(model.decoder.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda m, a, o, name=f"layer-{i}": capture(m, a, o, name)
                )
            )
        try:
            return original(prepared)
        finally:
            for handle in handles:
                handle.remove()
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(records, path)

    trainer._decoder_hidden = decoder


def metrics(
    reference: torch.Tensor, candidate: torch.Tensor, lengths: list[int]
) -> dict:
    assert reference.shape == candidate.shape and sum(lengths) == reference.numel()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    delta = (reference.double() - candidate.double()).abs()
    onsets, offset, documents = [], 0, []
    for length in lengths:
        part = delta[offset : offset + length]
        bad = (part > 1).nonzero().reshape(-1)
        onsets.append(int(bad[0]) if len(bad) else None)
        documents.append(
            {
                "length": length,
                "first_gt1": onsets[-1],
                "first_gt1_mod64": int(bad[0]) % 64 if len(bad) else None,
                "first_192_delta": part[:192].tolist() if finite else None,
                "mean_abs": float(part.mean()) if finite else None,
                "frac_gt1": float((part > 1).double().mean()),
                "packed_logprob_offset": offset,
            }
        )
        offset += length
    # M0 signature: retain the near-zero population, >1-nat population and
    # document onsets. A bimodal flag is report-only, never a tolerance gate.
    frac_small = float((delta < 0.01).double().mean())
    frac_large = float((delta > 1).double().mean())
    return {
        "finite": finite,
        "tokens": delta.numel(),
        "mean_abs": float(delta.mean()) if finite else None,
        "p50_abs": float(delta.quantile(0.5)) if finite else None,
        "p99_abs": float(delta.quantile(0.99)) if finite else None,
        "max_abs": float(delta.max()) if finite else None,
        "frac_gt1": frac_large,
        "frac_lt0_01": frac_small,
        "first_bad_position_per_document": onsets,
        "documents": documents,
        "position_basis": "zero-based input position predicting token position+1; bad means abs delta>1 nat",
        "histogram_edges_nats": [0, 0.001, 0.01, 0.1, 1, 10, 100],
        "histogram_counts": torch.histogram(
            delta,
            bins=torch.tensor([0, 0.001, 0.01, 0.1, 1, 10, 100], dtype=torch.double),
        )[0].tolist()
        if finite
        else None,
        "m0_bimodal_signature": finite and frac_small > 0.5 and frac_large > 0.1,
        "signature_definition_report_only": "majority abs delta<0.01 nat and >10% abs delta>1 nat; report all document onsets alongside M0's historical98..147",
        "hard_bug": not finite or bool(delta.mean() > 0.1),
    }


def worker(args: Any) -> None:
    from megatron.core import parallel_state as ps

    from art.megatron.lora import LoraLayout
    from art.trainer_rank import ForwardInput, TrainerRank
    from tests.integration.megatron.glm5next_qualification import (
        configure_qualification_provider,
    )

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    assert world == (1 if args.variant == "B0" else 8)
    if args.cpu:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    else:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "gloo" if args.cpu else "nccl", timeout=timedelta(seconds=300)
    )
    stream = None
    with ExitStack() as stack:
        if args.cpu:
            from tests.integration.megatron.glm5next_trainer_cpu import (
                build_trainer,
                cpu_kernels,
            )

            stack.enter_context(cpu_kernels())
            ps.initialize_model_parallel(
                context_parallel_size=world,
                expert_model_parallel_size=world,
                expert_tensor_parallel_size=1,
            )
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            cfg = json.loads(
                (
                    Path(__file__).parents[2]
                    / "unit/fixtures/glm5next_configs/R2/config.json"
                ).read_text()
            )
            cfg["n_routed_experts"] = 8
            (directory / "config.json").write_text(json.dumps(cfg))
            trainer = build_trainer(world, directory / "config.json")
            # Native build_trainer initializes B to zero via the contract.
            from tests.integration.megatron.glm5next_qualification import identity

            if rank == 0:
                identity(directory, args.output / "initial")
            dist.barrier()
            trainer.load_checkpoint(str(args.output / "initial"))
            stack.enter_context(
                patch.object(
                    trainer,
                    "_configure_hybridep",
                    lambda batches, *, topology: trainer._validate_hybridep_topology(
                        topology
                    ),
                )
            )
            if world == 1:
                stream = LayerStream(trainer.runtime.model, torch.device("cpu"))
        else:
            from art.megatron.train import build_training_runtime

            def configure(provider: Any) -> None:
                configure_qualification_provider(provider)
                provider.context_parallel_size = world
                provider.expert_model_parallel_size = world

            if world == 1:
                from megatron.bridge.models import model_provider

                from art.megatron.runtime import bridge_runtime

                def place(chunks: Any, *a: Any, **kw: Any) -> Any:
                    nonlocal stream
                    stream = LayerStream(chunks, torch.device("cuda", 0))
                    return chunks

                stack.enter_context(
                    patch.object(
                        bridge_runtime,
                        "_materialization_device",
                        return_value=torch.device("cpu"),
                    )
                )
                stack.enter_context(patch.object(model_provider, "_ddp_wrap", place))
            runtime = build_training_runtime(
                model_identifier=str(args.model),
                model_support_key="glm5next",
                allow_unvalidated_arch=True,
                build_optimizer=False,
                lora_layout=LoraLayout(expert="shared_outer"),
                provider_configure=configure,
            )
            if stream is not None:
                # Install on the final compiled wrappers, so transfers sit
                # outside compiled layer arithmetic.
                for handle in stream.handles:
                    handle.remove()
                stream = LayerStream(runtime.model, torch.device("cuda", 0))
            trainer = TrainerRank(runtime)
            trainer.load_checkpoint(str(args.identity))
        assert ps.get_context_parallel_world_size() == world
        assert ps.get_expert_model_parallel_world_size() == world
        install_capture(trainer, args.output / f"capture-rank{rank}.pt")
        docs = json.loads(args.documents.read_text())["documents"]
        if args.cpu:
            # Actual native token IDs are retained in the seal; only CPU compute
            # geometry/length/vocabulary scale down, explicitly in the receipt.
            docs = [
                {**d, "tokens": [t % cfg["vocab_size"] for t in d["tokens"][:257]]}
                for d in docs
            ]
        before = [
            p.detach().clone()
            for chunk in trainer.runtime.model
            for p in chunk.parameters()
            if p.requires_grad
        ]
        modes = [("no-grad", False)] + (
            [("forward_backward", True)] if world == 8 else []
        )
        for mode, backward in modes:
            vectors, losses, lengths = [], [], []
            for doc in docs:
                ids = doc["tokens"]
                trainer.zero_grad()
                request = ForwardInput(
                    input_tokens=torch.tensor(ids[:-1]),
                    target_tokens=torch.tensor(ids[1:]),
                )
                for mb in trainer.forward_micro_batches(
                    [request], no_grad=not backward
                ):
                    lp = mb.outputs[0].target_logprobs
                    assert lp is not None and lp.numel() == len(ids) - 1
                    if backward:
                        trainer.backward([-lp.mean()])
                        gradients = [
                            getattr(p, "main_grad", None)
                            if getattr(p, "main_grad", None) is not None
                            else p.grad
                            for slot in trainer._checkpoint_slots.values()
                            for p in slot.params
                        ]
                        gradients = [g for g in gradients if g is not None]
                        assert gradients and all(
                            torch.isfinite(g).all() for g in gradients
                        )
                        assert any(bool(g.abs().sum() > 0) for g in gradients)
                    vectors.append(lp.detach().float().cpu().reshape(-1))
                    losses.append(float(-lp.detach().mean()))
                    lengths.append(len(ids) - 1)
            vector = torch.cat(vectors)
            torch.save(vector, args.output / f"{mode}-rank{rank}.pt")
            write(
                args.output / f"{mode}-rank{rank}.json",
                {
                    "losses": losses,
                    "lengths": lengths,
                    "finite": bool(torch.isfinite(vector).all()),
                },
            )
        after = [
            p
            for chunk in trainer.runtime.model
            for p in chunk.parameters()
            if p.requires_grad
        ]
        assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True)), (
            "unexpected adapter update"
        )
        indexers = [
            m
            for c in trainer.runtime.model
            for m in c.modules()
            if type(m).__name__ == "KPoolIndexer"
        ]
        assert indexers and all(
            not p.requires_grad and p.grad is None
            for m in indexers
            for p in m.parameters()
        )
        write(
            args.output / f"rank{rank}.json",
            {
                "status": "pass",
                "rank": rank,
                "pid": os.getpid(),
                "world": world,
                "cp": world,
                "ep": world,
                "optimizer_steps": 0,
                "adapter_unchanged": True,
                "frozen_indexers": len(indexers),
                "documents_sha256": sha(args.documents),
                "cpu_scaled": args.cpu,
                "cuda_initialized": torch.cuda.is_initialized(),
                "streamed_layers": None if stream is None else stream.visits,
                "max_frozen_layer_bytes": None
                if stream is None
                else max(stream.layer_bytes),
                "peak_allocated_bytes": 0
                if args.cpu
                else torch.cuda.max_memory_allocated(),
            },
        )
    ps.destroy_model_parallel()
    dist.destroy_process_group()


def compare(root: Path, output: Path) -> dict:
    reference = torch.load(root / "B0/no-grad-rank0.pt", weights_only=True)
    lengths = json.loads((root / "B0/no-grad-rank0.json").read_text())["lengths"]
    rows = {}
    for mode in ("no-grad", "forward_backward"):
        for rank in range(8):
            rows[f"{mode}-rank{rank}"] = metrics(
                reference,
                torch.load(root / f"B1/{mode}-rank{rank}.pt", weights_only=True),
                lengths,
            )
    layer_rows = []
    baseline = torch.load(root / "B0/capture-rank0.pt", weights_only=True)
    shards = [
        torch.load(root / f"B1/capture-rank{rank}.pt", weights_only=True)
        for rank in range(8)
    ]
    for key in baseline:
        if not key.startswith("layer-") or key.endswith(".uids"):
            continue
        buids = baseline[key + ".uids"].reshape(-1)
        bvalues = baseline[key]
        assert bvalues.shape[0] == buids.numel()
        valid = (buids >= 0) & (buids < lengths[0])
        buids, bvalues = buids[valid], bvalues[valid]
        pairs = {}
        for shard in shards:
            values, uids = shard[key], shard[key + ".uids"]
            assert values.shape[0] == uids.numel()
            for i, uid in enumerate(uids.tolist()):
                if not 0 <= uid < lengths[0]:
                    continue
                assert uid not in pairs, f"duplicate captured UID {uid}"
                pairs[uid] = values[i]
        assert set(buids.tolist()) == set(pairs), f"incomplete capture {key}"
        delta = (
            bvalues.float() - torch.stack([pairs[u] for u in buids.tolist()]).float()
        ).abs()
        layer_rows.append(
            {
                "layer": key,
                "finite": bool(torch.isfinite(delta).all()),
                "mean_abs": float(delta.mean()),
                "max_abs": float(delta.max()),
                "report_only_diverged": bool(
                    (delta > 1e-3 + 1e-3 * bvalues.float().abs()).any()
                ),
            }
        )
    assert layer_rows
    result = {
        "status": "fail" if any(r["hard_bug"] for r in rows.values()) else "pass",
        "comparisons": rows,
        "layers": layer_rows,
        "first_divergent_layer_report_only": next(
            (r for r in layer_rows if r["report_only_diverged"]), None
        ),
        "capture_complete": True,
    }
    write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("B0", "B1"), required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker(args)


if __name__ == "__main__":
    main()
