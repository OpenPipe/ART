"""Root-closed FLA CP output, state, and gradient parity with empty peers."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from art.megatron.gdn.fla_cp import chunk_gated_delta_rule_native_cp

from .distributed_init import file_init_method

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize(
    "case", ["sharded", "leading", "middle", "trailing", "disjoint_roots"]
)
def test_root_closed_native_fla_cp(cp_size: int, case: str, tmp_path: Path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
        pytest.skip(f"Requires {cp_size} CUDA devices")
    if case == "middle" and cp_size == 2:
        pytest.skip("An interior empty peer requires CP4")
    pytest.importorskip("fla")
    mp.spawn(
        _root_closed_worker,
        args=(cp_size, case, file_init_method(tmp_path, "root_closed")),
        nprocs=cp_size,
        join=True,
    )


def _assert_parity(actual, expected):
    actual, expected = (
        actual.detach().double().flatten(),
        expected.detach().double().flatten(),
    )
    assert torch.isfinite(actual).all()
    assert (actual - expected).norm() / expected.norm().clamp_min(1e-30) < 0.02
    assert F.cosine_similarity(actual, expected, dim=0) > 0.9995


def _root_closed_worker(rank: int, cp: int, case: str, init_method: str):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=cp,
        init_method=init_method,
        timeout=timedelta(seconds=120),
    )
    try:
        active, chunk = max(1, cp // 2), 64
        lengths = {
            "sharded": [[chunk] for _ in range(cp)],
            "trailing": [[chunk] if r < active else [0] for r in range(cp)],
            "leading": [[0] if r < cp - active else [chunk] for r in range(cp)],
            "middle": [[chunk] if r in (0, cp - 1) else [0] for r in range(cp)],
            "disjoint_roots": [
                [chunk, 0] if r < active else [0, chunk] for r in range(cp)
            ],
        }[case]
        torch.manual_seed(772)
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
            allow_empty_shards=True,
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
        for value, local_grad, expected in zip(data, grads[:-1], reference_grads[:-1]):
            assembled = torch.zeros_like(value, dtype=torch.float32)
            assembled.index_copy_(1, idx, local_grad.float())
            dist.all_reduce(assembled)
            _assert_parity(assembled, expected)
        _assert_parity(grads[-1], reference_grads[-1])
        assembled_out = torch.zeros_like(reference_out, dtype=torch.float32)
        assembled_out.index_copy_(1, idx, local_out.float())
        dist.all_reduce(assembled_out)
        _assert_parity(assembled_out, reference_out)
        _assert_parity(local_final, reference_final)
        loss_value = loss.detach().clone()
        dist.all_reduce(loss_value)
        torch.testing.assert_close(loss_value, reference_loss, rtol=0.02, atol=0)
    finally:
        dist.destroy_process_group()
