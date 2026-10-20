"""Full-token, full-height CP capacity/parity probe; synthetic data, no service calls.

Run one context per fresh torchrun process. An r8 adapter includes FP32 master,
accumulator and Adam moments in backward mode. Never use proxy-layer results as
advertised service capability. Snapshot tensors use safetensors, never pickle.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("ART_DISABLE_MEGATRON_COMPILE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--weight-path")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--layers", type=int, default=0, help="0 means the complete model"
    )
    parser.add_argument("--mode", choices=("forward", "backward"), default="backward")
    parser.add_argument("--root-closed", action="store_true")
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--checkpoint-shards", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument(
        "--identity", help="Write complete frozen-base and adapter byte hashes"
    )
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument(
        "--adapter-init", choices=("stress", "production", "zero_b"), default="stress"
    )
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    cp = 1 if args.checkpoint_shards else world
    if args.canonical and cp != 1:
        parser.error("canonical split-root CP is not qualified")
    if args.checkpoint_shards and not args.canonical:
        parser.error("checkpoint shards require the canonical CP1 reference")
    os.environ["ART_MEGATRON_CONTEXT_PARALLEL_SIZE"] = str(cp)
    os.environ["ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE"] = "1"
    os.environ["ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE"] = "1"
    os.environ["ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE"] = "1"
    from trainer_rank_support import load_random_checkpoints

    from art.megatron import train
    from art.megatron.context_parallel.root_closed import execution
    from art.megatron.model_support.registry import get_model_support_spec
    from art.trainer_rank import ForwardInput, TrainerRank
    from art.trainer_rank._impl import _language_model

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    facts = dict(
        model=args.model,
        weight_path=args.weight_path,
        cp=cp,
        checkpoint_shards=world if args.checkpoint_shards else 1,
        canonical=args.canonical,
        tp=1,
        ep=1,
        layers_override=args.layers,
        mode=args.mode,
        root_closed=args.root_closed,
        tokens=args.tokens,
        head_dtype="float32",
        lora_rank=args.lora_rank,
        base_only=args.base_only,
        adapter_init=args.adapter_init,
        gdn_parameterization=os.environ.get(
            "ART_MEGATRON_QWEN_LORA_PARAMETERIZATION", "native"
        ),
        expert_lora_layout=os.environ.get("ART_MEGATRON_EXPERT_LORA_LAYOUT"),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
    )

    def emit(event, **data):
        if dist.get_rank() == 0:
            row = dict(**facts, event=event, **data)
            print(json.dumps(row, sort_keys=True), flush=True)
            with target.open("a") as output:
                output.write(json.dumps(row, sort_keys=True) + "\n")

    @contextmanager
    def memory_guard(phase):
        try:
            yield
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                emit(
                    "out_of_memory",
                    phase=phase,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    device_free_bytes=torch.cuda.mem_get_info()[0],
                )
            raise

    def configure(provider):
        if args.layers:
            provider.num_layers = args.layers

    torch.manual_seed(1234)
    started = time.perf_counter()
    runtime = train.build_training_runtime(
        model_identifier=args.weight_path or args.model,
        model_support_key=get_model_support_spec(args.model).key,
        model_initialization="pretrained",
        provider_configure=configure,
        build_optimizer=False,
        print_env=False,
    )
    for chunk in runtime.model:
        chunk.train()
    trainer = TrainerRank(runtime)
    [slot] = load_random_checkpoints(
        runtime, trainer, 1, base_model=args.model, lora_rank=args.lora_rank
    )
    parameters = trainer._checkpoint_slots[slot].params
    if args.adapter_init != "stress":
        from art.megatron.lora import LoRA, LoRASlot

        generator = torch.Generator(device=trainer.device).manual_seed(32)
        ref = trainer._slot_ref(slot)
        with torch.no_grad():
            modules = [
                module
                for chunk in runtime.model
                for module in chunk.modules()
                if isinstance(module, LoRA)
            ]
            for module in sorted(modules, key=lambda m: m.adapter_model_prefix):
                if ref not in module._slot_keys:
                    continue
                adapter = module._slot_modules[module._slot_keys[ref]]
                assert isinstance(adapter, LoRASlot)
                adapter.A_T.uniform_(
                    -1 / math.sqrt(module.in_features),
                    1 / math.sqrt(module.in_features),
                    generator=generator,
                )
                if args.adapter_init == "zero_b":
                    adapter.B_T.zero_()
                else:
                    adapter.B_T.normal_(std=1e-3, generator=generator)
    if args.identity:

        def fingerprint(tensor):
            raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
            return dict(
                shape=list(tensor.shape),
                dtype=str(tensor.dtype),
                sha256=hashlib.sha256(raw).hexdigest(),
            )

        identity = dict(
            frozen={
                f"{i}:{name}": fingerprint(p)
                for i, chunk in enumerate(runtime.model)
                for name, p in chunk.named_parameters()
                if not p.requires_grad
            },
            adapter={str(i): fingerprint(p) for i, p in enumerate(parameters)},
        )
        Path(f"{args.identity}.rank{dist.get_rank()}.json").write_text(
            json.dumps(identity, sort_keys=True)
        )
    model = _language_model(runtime.model[0])
    if args.canonical:
        from art.megatron.canonical import RootLayout, install

        install(model)
        canonical_layout = RootLayout((args.tokens,))
    head_weight = (
        model.shared_embedding_or_output_weight()
        if model.share_embeddings_and_output_weights
        else model.output_layer.weight
    )
    assert head_weight is not None
    head_weight = head_weight.float()
    optimizer_state = (
        [torch.zeros_like(p, dtype=torch.float32) for p in parameters for _ in range(4)]
        if args.mode == "backward"
        else []
    )
    torch.cuda.synchronize()
    emit(
        "loaded",
        elapsed_s=time.perf_counter() - started,
        layers=runtime.provider.num_layers,
        trainable_tensors=len(parameters),
        trainable_elements=sum(parameter.numel() for parameter in parameters),
        optimizer_state_bytes=sum(
            t.numel() * t.element_size() for t in optimizer_state
        ),
    )
    physical = args.tokens + (-args.tokens % (128 * cp))
    tokens = torch.arange(physical + 1, dtype=torch.long) % 1000 + 100
    placements = ((0, physical, 0, cp),)
    for iteration in range(args.repeat):
        trainer.zero_grad()
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        started = time.perf_counter()
        with ExitStack() as stack:
            if args.root_closed:
                stack.enter_context(execution(placements))
            if args.canonical:
                from art.megatron.canonical import execution as canonical_execution

                stack.enter_context(canonical_execution(canonical_layout))
            sharding = nullcontext(None)
            if args.checkpoint_shards:
                from art.megatron.context_parallel.checkpoint import (
                    shard_saved_activations,
                )

                for chunk in runtime.model:
                    no_sync = getattr(chunk, "no_sync", None)
                    if callable(no_sync):
                        stack.enter_context(no_sync())
                sharding = shard_saved_activations(
                    dist.group.WORLD,
                    shape=(physical, 1, runtime.provider.hidden_size),
                    key=f"probe/{args.model}/{args.tokens}/{args.lora_rank}/{iteration}",
                )
            with memory_guard("forward"), sharding as shard_stats:
                [result] = trainer.dp_rank_forward(
                    [ForwardInput(input_tokens=tokens[:-1], hidden_states=True)],
                    checkpoint=None if args.base_only else slot,
                    no_grad=args.mode == "forward",
                )
            assert result.hidden_states is not None
            hidden = result.hidden_states

            def head(h, labels):
                return -F.cross_entropy(
                    F.linear(h.float(), head_weight), labels, reduction="none"
                )

            pieces = []
            for start in range(0, physical, 128):
                h = hidden[start : start + 128]
                labels = tokens[1 + start : 1 + start + 128].to(h.device)
                pieces.append(
                    checkpoint(head, h, labels, use_reentrant=False)
                    if h.requires_grad
                    else head(h, labels)
                )
            logprobs = torch.cat(pieces)[: args.tokens]
            loss = -logprobs.mean()
            if args.mode == "backward":
                with memory_guard("backward"):
                    loss.backward()
        torch.cuda.synchronize()
        seconds = torch.tensor(time.perf_counter() - started, device="cuda")
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        peak = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda")
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        reserved = torch.tensor(torch.cuda.max_memory_reserved(), device="cuda")
        dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
        finite = all(
            p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters
        )
        nonzero = any(
            p.grad is not None and bool(p.grad.count_nonzero()) for p in parameters
        )
        assert bool(torch.isfinite(logprobs).all()) and finite
        if args.mode == "backward":
            assert nonzero, "no adapter gradients"
        emit(
            "passed",
            iteration=iteration,
            physical_tokens=physical,
            seconds=float(seconds),
            loss=float(loss.detach()),
            peak_allocated_bytes=int(peak),
            peak_reserved_bytes=int(reserved),
            gradients_finite=finite,
            nonzero_gradient=nonzero,
            checkpoint_storage=None if shard_stats is None else vars(shard_stats),
        )
        if args.checkpoint_shards and args.mode == "backward":
            digest = hashlib.sha256()
            for parameter in parameters:
                if parameter.grad is None:
                    digest.update(b"none")
                else:
                    digest.update(
                        parameter.grad.detach()
                        .contiguous()
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                    )
            digests = [None] * world
            dist.all_gather_object(digests, digest.hexdigest())
            assert len(set(digests)) == 1, "replicated canonical gradients differ"
            emit("replicated_gradients", sha256=digests[0], byte_exact=True)
        if args.snapshot and iteration == args.repeat - 1:
            from safetensors.torch import save_file

            tensors = {"logprobs": logprobs.detach().cpu().contiguous()}
            if args.mode == "backward":
                gradients = trainer._reduce_dynamic_grads(
                    parameters,
                    scale_grads=1.0 / world if args.checkpoint_shards else 1.0,
                )
                tensors.update(
                    {
                        f"gradient_{i}": g.cpu().contiguous()
                        for i, g in enumerate(gradients)
                    }
                )
            if dist.get_rank() == 0:
                save_file(
                    tensors,
                    args.snapshot,
                    metadata={k: str(v) for k, v in facts.items()},
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
