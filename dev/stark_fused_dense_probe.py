"""Compare independent dense slots with one canonical mixed-adapter backward.

Run with one-rank torchrun; full pretrained Qwen3.8 by default. One complete
root per adapter tests simultaneous gradients without cross-root BF16 sums.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist

from art.megatron.canonical import RootLayout, execution, install, target_logprobs
from art.megatron.lora import LoRA, LoRASlot, LoRASlotRef, use_lora_slot
from art.megatron.segmented_lora import (
    LoRASegments,
    slice_lora_segments,
    use_lora_segments,
)


def digest(value):
    return hashlib.sha256(
        value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--lengths", nargs="+", type=int, default=[257, 129])
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument(
        "--initialization", default="pretrained", choices=["pretrained", "random"]
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["ART_DISABLE_MEGATRON_COMPILE"] = "1"
    os.environ["ART_MEGATRON_LORA_RANK"] = "8"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    torch.manual_seed(731)
    from art.megatron.prefix_tree_packing import prefix_tree_pack
    from art.megatron.prefix_tree_state import create_prefix_tree_state
    from art.megatron.train import _placeholder_attention_mask, build_training_runtime
    from art.megatron.training.microbatches import _gdn_planner_config_for_provider

    def configure(provider):
        provider.hidden_dropout = provider.attention_dropout = 0.0
        provider.gradient_accumulation_fusion = False
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
        model_support_key="qwen3_5_dense",
        build_optimizer=False,
        print_env=False,
    )
    model: Any = runtime.model[0]
    while hasattr(model, "module"):
        model = model.module
    model = getattr(model, "language_model", model)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    refs = tuple(
        LoRASlotRef("checkpoint", f"fused-{i}") for i in range(len(args.lengths))
    )
    params = {ref: {} for ref in refs}
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            if not isinstance(module, LoRA):
                continue
            for index, ref in enumerate(refs):
                slot = LoRASlot(
                    ref=ref,
                    a_t=torch.empty_like(module.A_T).normal_(
                        std=module.in_features**-0.5
                    ),
                    b_t=torch.empty_like(module.B_T).normal_(std=1e-3),
                    alpha=module.alpha,
                    a_template=module.A_T,
                    b_template=module.B_T,
                    requires_grad=True,
                )
                key = f"fused_{index}"
                module._slot_keys[ref] = key
                module._slot_modules[key] = slot
                params[ref][name + ".A"] = slot.A_T
                params[ref][name + ".B"] = slot.B_T
    install(model, row_context=slice_lora_segments)
    generator = torch.Generator().manual_seed(42)
    roots = [torch.randint(100, 10000, (n,), generator=generator) for n in args.lengths]
    weight = (
        model.shared_embedding_or_output_weight()
        if model.share_embeddings_and_output_weights
        else model.output_layer.weight
    )
    reference = {}
    evidence = {
        "args": {**vars(args), "output": str(args.output)},
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "rows": [],
        "sources": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in [
                Path(__file__).resolve(),
                *Path(__file__)
                .resolve()
                .parents[1]
                .joinpath("src/art/megatron/canonical")
                .glob("*.py"),
                Path(__file__).resolve().parents[1] / "src/art/megatron/lora.py",
                Path(__file__).resolve().parents[1]
                / "src/art/megatron/segmented_lora.py",
            ]
        },
    }

    def execute(indices, fused, active=None):
        layout = RootLayout(tuple(args.lengths[i] for i in indices))
        physical = [RootLayout((args.lengths[i],)).pad([roots[i]]) for i in indices]
        pack = prefix_tree_pack(physical, max_depth=0)
        if not torch.equal(pack.tokens.flatten(), torch.cat(physical).flatten()):
            raise AssertionError("prefix packing changed immutable adapter row order")
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
        for owned in params.values():
            for parameter in owned.values():
                parameter.grad = None
        tokens, positions = pack.tokens.cuda(), pack.position_ids.cuda()
        labels = layout.pad([roots[i].roll(-1) for i in indices], value=-100).cuda()
        context = (
            use_lora_segments(
                LoRASegments(tuple(refs[i] for i in indices), layout.padded_lengths)
            )
            if fused
            else use_lora_slot(refs[indices[0]])
        )
        with execution(layout), context:
            pre = model._preprocess(
                input_ids=tokens, position_ids=positions, packed_seq_params=None
            )
            hidden, rot, cos, sin, offset, padding = pre[:6]
            output = model.decoder(
                hidden_states=hidden,
                attention_mask=_placeholder_attention_mask(torch.device("cuda")),
                rotary_pos_emb=rot,
                rotary_pos_cos=cos,
                rotary_pos_sin=sin,
                rotary_pos_cos_sin=pre[6] if len(pre) == 7 else None,
                packed_seq_params=None,
                sequence_len_offset=offset,
                padding_mask=padding,
                attention_bias=state,
            )
            logps = target_logprobs(output.squeeze(1), weight, labels, layout=layout)
            losses = [
                -value.sum() / args.lengths[index]
                for value, index in zip(logps, indices, strict=True)
            ]
            torch.stack(
                [
                    value
                    for value, index in zip(losses, indices, strict=True)
                    if active is None or index == active
                ]
            ).sum().backward()
        result = {}
        for position, index in enumerate(indices):
            gradients = {
                name: cast(torch.Tensor, parameter.grad).detach().float().cpu().clone()
                for name, parameter in params[refs[index]].items()
                if parameter.grad is not None
            }
            result[index] = (
                logps[position].detach().cpu(),
                losses[position].detach().cpu(),
                gradients,
            )
        return result

    for index in range(len(refs)):
        reference.update(execute([index], False))
    for indices, active in [
        (list(range(len(refs))), None),
        (list(reversed(range(len(refs)))), None),
        (list(range(len(refs))), 0),
    ]:
        results = execute(indices, True, active)
        for index, (logps, loss, gradients) in results.items():
            expected_logps, expected_loss, expected_grads = reference[index]
            masked = active is not None and index != active
            exact = all(
                not value.any() if masked else torch.equal(value, expected_grads[name])
                for name, value in gradients.items()
            )
            row = {
                "order": indices,
                "active": active,
                "adapter": index,
                "logps_exact": torch.equal(logps, expected_logps),
                "loss_exact": torch.equal(loss, expected_loss),
                "gradients_exact": exact and gradients.keys() == expected_grads.keys(),
                "gradient_max_abs": max(
                    float(
                        value.abs().max()
                        if masked
                        else (value - expected_grads[name]).abs().max()
                    )
                    for name, value in gradients.items()
                ),
                "gradient_hashes": {
                    name: digest(value) for name, value in gradients.items()
                },
            }
            evidence["rows"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, indent=2))
            print(
                json.dumps(
                    {
                        name: value
                        for name, value in row.items()
                        if name != "gradient_hashes"
                    }
                ),
                flush=True,
            )
            if not all(
                row[key] for key in ("logps_exact", "loss_exact", "gradients_exact")
            ):
                raise AssertionError("fused sequential canonical comparison failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
