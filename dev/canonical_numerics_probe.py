"""Full dense canonical repeat/packing/chunking regression, run with torchrun.

Pretrained Qwen3.8 by default, TP1/CP1. Evidence binds the actual policy sources
and records byte equality, full per-parameter gradient hashes and warm timings.
Native policy and arbitrary within-root sequence splits are not qualified here.
"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Any, cast

import torch
import torch.distributed as dist

from art.megatron.canonical import RootLayout, execution, install, target_logprobs


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument(
        "--initialization", choices=("random", "pretrained"), default="pretrained"
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[257, 513, 129])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("this qualification requires a one-GPU world")
    os.environ["ART_DISABLE_MEGATRON_COMPILE"] = "1"
    os.environ["ART_MEGATRON_LORA_RANK"] = "8"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    torch.manual_seed(731)
    from art.megatron.lora import LoRA
    from art.megatron.prefix_tree_packing import prefix_tree_pack
    from art.megatron.prefix_tree_state import create_prefix_tree_state
    from art.megatron.train import _placeholder_attention_mask, build_training_runtime
    from art.megatron.training.microbatches import _gdn_planner_config_for_provider

    def configure(provider):
        provider.hidden_dropout = provider.attention_dropout = 0.0
        if args.layers:
            provider.num_layers = args.layers
            if isinstance(getattr(provider, "linear_attention_freq", None), list):
                provider.linear_attention_freq = provider.linear_attention_freq[
                    : args.layers
                ]
        provider.recompute_granularity = "full"
        provider.recompute_method = "uniform"
        provider.recompute_num_layers = 1

    runtime = build_training_runtime(
        model_identifier=args.model,
        model_initialization=args.initialization,
        provider_configure=configure,
        build_optimizer=False,
        print_env=False,
    )
    model: Any = runtime.model[0]
    while hasattr(model, "module"):
        model = model.module
    model = getattr(model, "language_model", model)
    model.train()
    install(model)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRA):
                module.B_T.normal_(std=1e-3)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    generator = torch.Generator().manual_seed(42)
    sequences = [
        torch.randint(100, 10000, (n,), generator=generator) for n in args.lengths
    ]
    weight = (
        model.shared_embedding_or_output_weight()
        if model.share_embeddings_and_output_weights
        else model.output_layer.weight
    )
    result = {
        "args": {**vars(args), "output": str(args.output)},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in (
                "transformer-engine",
                "megatron-core",
                "megatron-bridge",
                "flash-linear-attention",
            )
        },
        "policy_sources": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("src/art/megatron/canonical")
            .glob("*.py")
        },
        "rows": [],
    }
    reference_logps = reference_grads = None
    variants = [
        list(range(len(sequences))),
        list(reversed(range(len(sequences)))),
        [0],
        [0, len(sequences) - 1],
        list(range(len(sequences))),
    ]
    for repetition in range(args.repeat):
        for roots in variants:
            layout = RootLayout(tuple(args.lengths[r] for r in roots))
            physical = [
                RootLayout((args.lengths[r],)).pad([sequences[r]]) for r in roots
            ]
            pack = prefix_tree_pack(physical, max_depth=0)
            state = create_prefix_tree_state(
                pack.group_ids,
                pack.parent_ids,
                input_pos=pack.position_ids,
                target_device=torch.device("cuda"),
                build_gdn_execution_spec=runtime.model_support_handler.build_gdn_execution_spec,
                model_support_handler=runtime.model_support_handler,
                attention_head_dim=runtime.provider.kv_channels,
                attention_value_head_dim=runtime.provider.kv_channels,
                gdn_planner_config=_gdn_planner_config_for_provider(
                    runtime.provider, runtime.model_support_handler
                ),
            )
            tokens, positions = pack.tokens.cuda(), pack.position_ids.cuda()
            labels = layout.pad(
                [sequences[r].roll(-1) for r in roots], value=-100
            ).cuda()
            for typed_chunk in runtime.model:
                chunk = cast(Any, typed_chunk)
                if hasattr(chunk, "zero_grad_buffer"):
                    chunk.zero_grad_buffer()
                    for group in (
                        chunk.bucket_groups + chunk.expert_parallel_bucket_groups
                    ):
                        group.is_last_microbatch = False
            for param in params.values():
                param.grad = None
            for backward in (False, True):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with execution(layout), nullcontext() if backward else torch.no_grad():
                    pre = model._preprocess(
                        input_ids=tokens, position_ids=positions, packed_seq_params=None
                    )
                    hidden, rot, cos, sin, offset, padding = pre[:6]
                    output = model.decoder(
                        hidden_states=hidden,
                        attention_mask=_placeholder_attention_mask(
                            torch.device("cuda")
                        ),
                        rotary_pos_emb=rot,
                        rotary_pos_cos=cos,
                        rotary_pos_sin=sin,
                        rotary_pos_cos_sin=pre[6] if len(pre) == 7 else None,
                        packed_seq_params=None,
                        sequence_len_offset=offset,
                        padding_mask=padding,
                        attention_bias=state,
                    )
                    logps = target_logprobs(
                        output.squeeze(1), weight, labels, layout=layout
                    )[roots.index(0)]
                    loss = -logps.sum() / args.lengths[0]
                    if backward:
                        loss.backward()
                torch.cuda.synchronize()
                actual = logps.detach().cpu()
                if reference_logps is None:
                    reference_logps = actual.clone()
                row = {
                    "repeat": repetition,
                    "roots": roots,
                    "backward": backward,
                    "seconds": time.perf_counter() - start,
                    "loss": loss.item(),
                    "logps_sha256": digest(actual),
                    "logps_exact": torch.equal(reference_logps, actual),
                }
                if backward:
                    gradients = {}
                    for name, param in params.items():
                        grad = getattr(param, "main_grad", None)
                        if grad is None:
                            grad = param.grad
                        if grad is not None:
                            gradients[name] = grad.detach().float().cpu().clone()
                    if reference_grads is None:
                        reference_grads = gradients
                    row["gradient_hashes"] = {
                        n: digest(g) for n, g in gradients.items()
                    }
                    row["gradients_exact"] = (
                        gradients.keys() == reference_grads.keys()
                        and all(
                            torch.equal(g, reference_grads[n])
                            for n, g in gradients.items()
                        )
                    )
                    row["gradient_max_abs"] = max(
                        (g - reference_grads[n]).abs().max().item()
                        for n, g in gradients.items()
                    )
                result["rows"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2))
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k != "gradient_hashes"}
                    ),
                    flush=True,
                )
                if not row["logps_exact"] or not row.get("gradients_exact", True):
                    raise AssertionError(
                        "canonical byte comparison failed; see evidence"
                    )
                del output, hidden, logps, loss
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
