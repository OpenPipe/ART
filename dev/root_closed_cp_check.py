"""Native CP GDN output/gradient check against unsharded FLA, including empty peers."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import torch
import torch.distributed as dist
import torch.nn.functional as F

from art.megatron.gdn.fla_cp import chunk_gated_delta_rule_native_cp


def metric(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return dict(
        relative_l2=float((a - b).norm() / b.norm().clamp_min(1e-30)),
        cosine=float(F.cosine_similarity(a, b, dim=0)),
        max_abs=float((a - b).abs().max()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--cases", default="sharded,trailing,leading,middle,disjoint_roots"
    )
    parser.add_argument("--trace", type=int, default=0)
    parser.add_argument("--shard-tokens", type=int, default=64)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, cp = dist.get_rank(), dist.get_world_size()
    assert cp in (2, 4, 8)
    active = max(1, cp // 2)
    chunk = args.shard_tokens
    assert chunk > 0 and chunk % 64 == 0
    cases = {
        "sharded": [[chunk] for _ in range(cp)],
        "trailing": [[chunk] if r < active else [0] for r in range(cp)],
        "leading": [[0] if r < cp - active else [chunk] for r in range(cp)],
        "middle": [
            [chunk] if (cp - active) // 2 <= r < (cp + active) // 2 else [0]
            for r in range(cp)
        ],
        "disjoint_roots": [[chunk, 0] if r < active else [0, chunk] for r in range(cp)],
    }
    for case_i, (name, lengths) in enumerate(cases.items()):
        if name not in args.cases.split(","):
            continue
        if args.trace:
            import art.megatron.gdn.fla_cp as cp_module

            for function_name in (
                "_fwd_summary",
                "_prefix_summary_exclusive",
                "_scan_fwd_initial_state",
                "_broadcast_chain_final_state",
            ):
                original = getattr(cp_module, function_name)

                def trace(*a, _fn=original, _name=function_name, **kw):
                    value = _fn(*a, **kw)
                    print(
                        "STAGE",
                        rank,
                        name,
                        _name,
                        None
                        if value is None
                        else dict(
                            shape=list(value.shape),
                            finite=bool(torch.isfinite(value).all()),
                            maximum=float(value.abs().max()),
                        ),
                        flush=True,
                    )
                    return value

                setattr(cp_module, function_name, trace)
        torch.manual_seed(772 + case_i)
        root_tokens = sum(row[0] for row in lengths)
        n = len(lengths[0])
        total = n * root_tokens
        shape = (1, total, 2, 32)
        data = [
            F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16(),
            F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16(),
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1,
            -torch.rand(1, total, 2, device="cuda") * 0.05,
            torch.rand(1, total, 2, device="cuda", dtype=torch.bfloat16),
        ]
        state = torch.randn(n, 2, 32, 32, device="cuda") * 0.01
        reference_inputs = [x.detach().clone().requires_grad_() for x in data]
        reference_state = state.detach().clone().requires_grad_()
        global_cu_cpu = torch.arange(n + 1, dtype=torch.int32) * root_tokens
        reference_out, reference_final = chunk_gated_delta_rule(
            *reference_inputs[:3],
            g=reference_inputs[3],
            beta=reference_inputs[4],
            initial_state=reference_state,
            output_final_state=True,
            cu_seqlens=global_cu_cpu.cuda(),
            cu_seqlens_cpu=global_cu_cpu,
            use_qk_l2norm_in_kernel=False,
        )
        reference_loss = (
            reference_out.float().square().sum() / total
            + 0.1 * reference_final.square().sum()
        )
        if args.trace:
            print(
                "REFERENCE",
                rank,
                name,
                bool(torch.isfinite(reference_loss)),
                flush=True,
            )
        reference_grads = torch.autograd.grad(
            reference_loss, [*reference_inputs, reference_state]
        )
        positions = []
        for root in range(n):
            start = root * root_tokens + sum(lengths[r][root] for r in range(rank))
            positions.extend(range(start, start + lengths[rank][root]))
        idx = torch.tensor(positions, device="cuda", dtype=torch.long)
        local_inputs = [x.index_select(1, idx).detach().requires_grad_() for x in data]
        local_state = state.detach().clone().requires_grad_()
        cu_cpu = torch.tensor([0, *lengths[rank]], dtype=torch.int32).cumsum(0).int()
        local_out, local_final = chunk_gated_delta_rule_native_cp(
            *local_inputs[:3],
            g=local_inputs[3],
            beta=local_inputs[4],
            initial_state=local_state,
            group=dist.group.WORLD,
            output_final_state=True,
            cu_seqlens=cu_cpu.cuda(),
            cu_seqlens_cpu=cu_cpu,
            lengths_by_rank_cpu=torch.tensor(lengths),
        )
        assert local_final is not None
        finite = torch.tensor(
            int(torch.isfinite(local_out).all() and torch.isfinite(local_final).all()),
            device="cuda",
        )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        assert bool(finite), "nonfinite native forward"
        loss = (
            local_out.float().square().sum() / total
            + (0.1 / cp) * local_final.square().sum()
        )
        grads = torch.autograd.grad(loss, [*local_inputs, local_state])
        grad_metrics = []
        for value, local_grad, expected in zip(data, grads[:-1], reference_grads[:-1]):
            assembled = torch.zeros_like(value, dtype=torch.float32)
            assembled.index_copy_(1, idx, local_grad.float())
            dist.all_reduce(assembled)
            grad_metrics.append(metric(assembled, expected))
        initial_metrics: list[Any] = [None] * cp
        dist.all_gather_object(initial_metrics, metric(grads[-1], reference_grads[-1]))
        assembled_out = torch.zeros_like(reference_out, dtype=torch.float32)
        assembled_out.index_copy_(1, idx, local_out.float())
        dist.all_reduce(assembled_out)
        loss_value = loss.detach().clone()
        dist.all_reduce(loss_value)
        row: dict[str, Any] = dict(
            case=name,
            cp=cp,
            lengths=lengths,
            input_gradients=grad_metrics,
            initial_gradients_by_rank=initial_metrics,
            output=metric(assembled_out, reference_out),
            final_state=metric(local_final, reference_final),
            loss_relative=float(
                ((loss_value - reference_loss).abs() / reference_loss).detach()
            ),
        )
        if rank == 0:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print("EMPTY_CHAIN", json.dumps(row, sort_keys=True), flush=True)
        assert all(
            m["relative_l2"] < 0.02 and m["cosine"] > 0.9995
            for m in [
                *grad_metrics,
                *initial_metrics,
                row["output"],
                row["final_state"],
            ]
        ), row
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
