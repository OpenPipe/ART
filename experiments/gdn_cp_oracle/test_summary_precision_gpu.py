"""A slow-decay layer detects reduced-precision rank-transition products.

Run with torchrun and GDN_SUMMARY_REGRESSION=1. The beta=0 recurrence has an
independent closed FP64 form for outputs, final state and initial-state grad.
Both ordinary and packed chains retain the original fixed 2x CP1 error gate.
"""

from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from .gates import compare_errors
from .reference import error_metrics


@pytest.mark.parametrize("packed", [False, True])
def test_slow_decay_summary_layer(packed):
    if os.environ.get("GDN_SUMMARY_REGRESSION") != "1":
        pytest.skip("requires granted GPUs and torchrun")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    from art.megatron.gdn.fla_cp import chunk_gated_delta_rule_native_cp

    rank, world = dist.get_rank(), dist.get_world_size()
    length, heads, width, sequences = 4096, 48, 128, 2 if packed else 1
    generator = torch.Generator().manual_seed(2101)
    q, k, v = [
        torch.randn(1, length * sequences, heads, width, generator=generator)
        .bfloat16()
        .cuda()
        * 0.05
        for _ in range(3)
    ]
    g = torch.full((1, length * sequences, heads), -1e-5, device="cuda")
    beta = torch.zeros_like(g, dtype=torch.bfloat16)
    h0 = (
        torch.randn(sequences, heads, width, width, generator=generator) * 0.05
    ).cuda()
    do = (torch.randn(q.shape, generator=generator) * 0.05).bfloat16().cuda()
    dht = (torch.randn(h0.shape, generator=generator) * 0.05).cuda()
    decay = g[:, :length].double().cumsum(1).exp()
    expected_out, expected_ht, expected_dh0 = [], [], []
    for sequence in range(sequences):
        start, end = length * sequence, length * (sequence + 1)
        query, cotangent = q[:, start:end].double(), do[:, start:end].double()
        state = h0[sequence : sequence + 1].double()
        expected_out.append(
            torch.einsum("bthk,bhkv->bthv", query, state)
            * decay[..., None]
            * width**-0.5
        )
        expected_ht.append(state * decay[:, -1, :, None, None])
        expected_dh0.append(
            torch.einsum("bthk,bthv,bth->bhkv", query, cotangent, decay) * width**-0.5
            + dht[sequence : sequence + 1].double() * decay[:, -1, :, None, None]
        )
    references = (
        torch.cat(expected_out, 1),
        torch.cat(expected_ht),
        torch.cat(expected_dh0),
    )
    cu_cpu = torch.arange(sequences + 1, dtype=torch.long) * length if packed else None
    public_h0 = h0.clone().requires_grad_()
    public_out, public_ht = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=public_h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None if cu_cpu is None else cu_cpu.cuda(),
        cu_seqlens_cpu=cu_cpu,
    )
    public_dh0 = torch.autograd.grad((public_out, public_ht), public_h0, (do, dht))[0]
    per_rank = length // world
    uids = torch.cat(
        [
            torch.arange(
                sequence * length + rank * per_rank,
                sequence * length + (rank + 1) * per_rank,
            )
            for sequence in range(sequences)
        ]
    )
    local_h0 = h0.clone().requires_grad_()
    local_cu_cpu = (
        torch.arange(sequences + 1, dtype=torch.long) * per_rank if packed else None
    )
    output, final = chunk_gated_delta_rule_native_cp(
        q[:, uids].contiguous(),
        k[:, uids].contiguous(),
        v[:, uids].contiguous(),
        g=g[:, uids].contiguous(),
        beta=beta[:, uids].contiguous(),
        initial_state=local_h0,
        output_final_state=True,
        group=dist.group.WORLD,
        cu_seqlens=None if local_cu_cpu is None else local_cu_cpu.cuda(),
        cu_seqlens_cpu=local_cu_cpu,
        lengths_by_rank_cpu=torch.full((world, sequences), per_rank),
    )
    assert final is not None
    dh0 = torch.autograd.grad(
        (output, final), local_h0, (do[:, uids].contiguous(), dht / world)
    )[0]
    outputs = [torch.empty_like(output) for _ in range(world)]
    dist.all_gather(outputs, output.detach())
    global_uids = torch.cat(
        [
            torch.cat(
                [
                    torch.arange(
                        sequence * length + r * per_rank,
                        sequence * length + (r + 1) * per_rank,
                    )
                    for sequence in range(sequences)
                ]
            )
            for r in range(world)
        ]
    )
    full_output = torch.cat(outputs, 1).index_select(1, global_uids.argsort().cuda())
    report = {}
    for name, actual, baseline, reference in zip(
        ("output", "final", "dh0"),
        (full_output, final, dh0),
        (public_out, public_ht, public_dh0),
        references,
        strict=True,
    ):
        assert all(
            bool(torch.isfinite(value).all()) for value in (actual, baseline, reference)
        )
        report[name] = compare_errors(
            error_metrics(baseline, reference), error_metrics(actual, reference)
        )
    destination = os.environ.get("GDN_SUMMARY_RESULTS")
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"packed{packed}.rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    assert all(check["pass"] for row in report.values() for check in row.values()), (
        report
    )


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    failed = False
    try:
        for packed_case in (False, True):
            try:
                test_slow_decay_summary_layer(packed_case)
                print("PASS", packed_case, dist.get_rank(), flush=True)
            except AssertionError:
                failed = True
                print("FAIL", packed_case, dist.get_rank(), flush=True)
    finally:
        dist.destroy_process_group()
    raise SystemExit(int(failed))
