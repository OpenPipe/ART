from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla.ops.gated_delta_rule")

from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from art.megatron.gdn.fla_cp import (  # noqa: E402
    _apply_summary,
    _fwd_summary,
    chunk_gated_delta_rule_native_cp,
)

from .distributed_init import file_init_method  # noqa: E402
from .metrics import GDN_CORRECTNESS_DTYPE, assert_mean_abs_pct  # noqa: E402

_CP_SIZES = (
    2,
    4,
    pytest.param(
        8,
        marks=pytest.mark.skipif(
            torch.cuda.device_count() < 8,
            reason="At least eight CUDA devices are required for CP8 coverage.",
        ),
    ),
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="At least four CUDA devices are required for native FLA CP coverage.",
)
@pytest.mark.parametrize("cp_size", _CP_SIZES)
def test_native_fla_cp_recurrent_matches_single_rank(
    cp_size: int, tmp_path: Path
) -> None:
    init_method = file_init_method(tmp_path, f"native_fla_recurrent_cp{cp_size}")
    mp.spawn(
        _native_fla_cp_worker,
        args=(cp_size, init_method, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"rank_{rank}.ok").read_text() == "ok\n"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="At least four CUDA devices are required for native FLA CP coverage.",
)
@pytest.mark.parametrize("cp_size", (2, 4))
def test_native_fla_cp_recurrent_varlen_multichain_matches_single_rank(
    cp_size: int, tmp_path: Path
) -> None:
    init_method = file_init_method(tmp_path, f"native_fla_varlen_cp{cp_size}")
    mp.spawn(
        _native_fla_cp_varlen_multichain_worker,
        args=(cp_size, init_method, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"varlen_rank_{rank}.ok").read_text() == "ok\n"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for FLA summary kernels.",
)
def test_native_fla_summary_affine_debug_matches_final_state() -> None:
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.utils import chunk_local_cumsum, solve_tril

    chunk_local_cumsum = cast(Any, chunk_local_cumsum)
    chunk_scaled_dot_kkt_fwd = cast(Any, chunk_scaled_dot_kkt_fwd)
    recompute_w_u_fwd = cast(Any, recompute_w_u_fwd)
    solve_tril = cast(Any, solve_tril)
    q, k, v, g, beta, h0, _, _ = _case_tensors_without_dist(cp_size=1)
    g_cumsum = chunk_local_cumsum(g, chunk_size=64)
    a = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_cumsum,
        beta=beta,
        output_dtype=GDN_CORRECTNESS_DTYPE,
    )
    a = solve_tril(A=a, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a, g=g_cumsum)
    summary = _fwd_summary(k=k, w=w, u=u, g=g_cumsum)

    _, ref_ht = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    assert ref_ht is not None
    assert_mean_abs_pct(
        ref_ht,
        _apply_summary(summary, h0[0]).unsqueeze(0),
        "summary_final_state",
    )


def _native_fla_cp_worker(
    rank: int,
    cp_size: int,
    init_method: str,
    output_dir: str,
) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=cp_size,
    )
    try:
        q, k, v, g, beta, h0, output_grad, ht_grad = _case_tensors(cp_size)
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)
        g_ref = g.clone().detach().requires_grad_(True)
        beta_ref = beta.clone().detach().requires_grad_(True)
        h0_ref = h0.clone().detach().requires_grad_(True)

        ref_out, ref_ht = chunk_gated_delta_rule(
            q_ref,
            k_ref,
            v_ref,
            g=g_ref,
            beta=beta_ref,
            initial_state=h0_ref,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        assert ref_ht is not None
        ref_loss = (ref_out * output_grad).sum() + (ref_ht * ht_grad).sum()
        ref_loss.backward()

        start = (q.shape[1] * rank) // cp_size
        end = (q.shape[1] * (rank + 1)) // cp_size
        local_grad = output_grad[:, start:end].contiguous()
        q_local = q[:, start:end].clone().detach().requires_grad_(True)
        k_local = k[:, start:end].clone().detach().requires_grad_(True)
        v_local = v[:, start:end].clone().detach().requires_grad_(True)
        g_local = g[:, start:end].clone().detach().requires_grad_(True)
        beta_local = beta[:, start:end].clone().detach().requires_grad_(True)
        h0_local = h0.clone().detach().requires_grad_(True)

        cp_out, cp_ht = chunk_gated_delta_rule_native_cp(
            q_local,
            k_local,
            v_local,
            g=g_local,
            beta=beta_local,
            initial_state=h0_local,
            group=torch.distributed.group.WORLD,
            output_final_state=True,
            lengths_by_rank_cpu=torch.full(
                (cp_size, 1), int(q_local.shape[1]), dtype=torch.long
            ),
        )
        assert cp_ht is not None
        cp_loss = (cp_out * local_grad).sum() + (cp_ht * (ht_grad / cp_size)).sum()
        cp_loss.backward()

        assert_mean_abs_pct(ref_out[:, start:end], cp_out, "output")
        assert_mean_abs_pct(ref_ht, cp_ht, "final_state")
        assert q_ref.grad is not None
        assert k_ref.grad is not None
        assert v_ref.grad is not None
        assert g_ref.grad is not None
        assert beta_ref.grad is not None
        assert h0_ref.grad is not None
        _assert_grad_close(q_local, q_ref.grad[:, start:end], "q")
        _assert_grad_close(k_local, k_ref.grad[:, start:end], "k")
        _assert_grad_close(v_local, v_ref.grad[:, start:end], "v")
        _assert_grad_close(g_local, g_ref.grad[:, start:end], "g")
        _assert_grad_close(beta_local, beta_ref.grad[:, start:end], "beta")
        _assert_grad_close(h0_local, h0_ref.grad, "h0")
        Path(output_dir, f"rank_{rank}.ok").write_text("ok\n")
    finally:
        destroy_process_group()


def _native_fla_cp_varlen_multichain_worker(
    rank: int,
    cp_size: int,
    init_method: str,
    output_dir: str,
) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=cp_size,
    )
    try:
        q, k, v, g, beta, h0, output_grad, ht_grad, cu = _varlen_case_tensors(cp_size)
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)
        g_ref = g.clone().detach().requires_grad_(True)
        beta_ref = beta.clone().detach().requires_grad_(True)
        h0_ref = h0.clone().detach().requires_grad_(True)

        ref_out, ref_ht = chunk_gated_delta_rule(
            q_ref,
            k_ref,
            v_ref,
            g=g_ref,
            beta=beta_ref,
            initial_state=h0_ref,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu,
        )
        assert ref_ht is not None
        ref_loss = (ref_out * output_grad).sum() + (ref_ht * ht_grad).sum()
        ref_loss.backward()

        local_slices = _rank_varlen_slices(cu, rank=rank, cp_size=cp_size)
        q_local = (
            _cat_varlen_slices(q, local_slices).clone().detach().requires_grad_(True)
        )
        k_local = (
            _cat_varlen_slices(k, local_slices).clone().detach().requires_grad_(True)
        )
        v_local = (
            _cat_varlen_slices(v, local_slices).clone().detach().requires_grad_(True)
        )
        g_local = (
            _cat_varlen_slices(g, local_slices).clone().detach().requires_grad_(True)
        )
        beta_local = (
            _cat_varlen_slices(beta, local_slices).clone().detach().requires_grad_(True)
        )
        h0_local = h0.clone().detach().requires_grad_(True)
        local_grad = _cat_varlen_slices(output_grad, local_slices).contiguous()
        local_cu_cpu = _local_cu_seqlens_cpu(local_slices)
        local_cu = local_cu_cpu.to(device=q.device)

        cp_out, cp_ht = chunk_gated_delta_rule_native_cp(
            q_local,
            k_local,
            v_local,
            g=g_local,
            beta=beta_local,
            initial_state=h0_local,
            cu_seqlens=local_cu,
            cu_seqlens_cpu=local_cu_cpu,
            lengths_by_rank_cpu=_lengths_by_rank_cpu(cu, cp_size=cp_size),
            group=torch.distributed.group.WORLD,
            output_final_state=True,
        )
        assert cp_ht is not None
        cp_loss = (cp_out * local_grad).sum() + (cp_ht * (ht_grad / cp_size)).sum()
        cp_loss.backward()

        assert_mean_abs_pct(
            _cat_varlen_slices(ref_out, local_slices),
            cp_out,
            "varlen_output",
        )
        assert_mean_abs_pct(ref_ht, cp_ht, "varlen_final_state")
        assert q_ref.grad is not None
        assert k_ref.grad is not None
        assert v_ref.grad is not None
        assert g_ref.grad is not None
        assert beta_ref.grad is not None
        assert h0_ref.grad is not None
        _assert_grad_close(q_local, _cat_varlen_slices(q_ref.grad, local_slices), "q")
        _assert_grad_close(k_local, _cat_varlen_slices(k_ref.grad, local_slices), "k")
        _assert_grad_close(v_local, _cat_varlen_slices(v_ref.grad, local_slices), "v")
        _assert_grad_close(g_local, _cat_varlen_slices(g_ref.grad, local_slices), "g")
        _assert_grad_close(
            beta_local,
            _cat_varlen_slices(beta_ref.grad, local_slices),
            "beta",
        )
        _assert_grad_close(h0_local, h0_ref.grad, "h0")
        Path(output_dir, f"varlen_rank_{rank}.ok").write_text("ok\n")
    finally:
        destroy_process_group()


def _case_tensors(cp_size: int) -> tuple[torch.Tensor, ...]:
    tensors = _case_tensors_without_dist(cp_size=cp_size)
    for tensor in tensors:
        torch.distributed.broadcast(tensor, src=0)
    return tensors


def _case_tensors_without_dist(cp_size: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20450426 + cp_size)
    token_count = 64 * max(cp_size, 1)
    q = F.normalize(
        torch.randn(
            1,
            token_count,
            2,
            8,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=generator,
        ),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(
            1,
            token_count,
            2,
            8,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=generator,
        ),
        p=2,
        dim=-1,
    )
    v = torch.randn(
        1,
        token_count,
        2,
        8,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    g = -torch.rand(
        1,
        token_count,
        2,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    beta = torch.rand(
        1,
        token_count,
        2,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    ).sigmoid()
    h0 = torch.randn(
        1, 2, 8, 8, device=device, dtype=GDN_CORRECTNESS_DTYPE, generator=generator
    )
    output_grad = torch.randn(
        1,
        token_count,
        2,
        8,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    ht_grad = torch.randn(
        1, 2, 8, 8, device=device, dtype=GDN_CORRECTNESS_DTYPE, generator=generator
    )
    return q, k, v, g, beta, h0, output_grad, ht_grad


def _varlen_case_tensors(cp_size: int) -> tuple[torch.Tensor, ...]:
    tensors = _varlen_case_tensors_without_dist(cp_size=cp_size)
    for tensor in tensors:
        torch.distributed.broadcast(tensor, src=0)
    return tensors


def _varlen_case_tensors_without_dist(cp_size: int) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20480426 + cp_size)
    lengths = (128 * cp_size, 192 * cp_size, 256 * cp_size)
    token_count = sum(lengths)
    q = F.normalize(
        torch.randn(
            1,
            token_count,
            2,
            8,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=generator,
        ),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(
            1,
            token_count,
            2,
            8,
            device=device,
            dtype=GDN_CORRECTNESS_DTYPE,
            generator=generator,
        ),
        p=2,
        dim=-1,
    )
    v = torch.randn(
        1,
        token_count,
        2,
        8,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    g = -torch.rand(
        1,
        token_count,
        2,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    beta = torch.rand(
        1,
        token_count,
        2,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    ).sigmoid()
    h0 = torch.randn(
        len(lengths),
        2,
        8,
        8,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    output_grad = torch.randn(
        1,
        token_count,
        2,
        8,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    ht_grad = torch.randn(
        len(lengths),
        2,
        8,
        8,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    cu = torch.tensor(
        [0, *torch.cumsum(torch.tensor(lengths), dim=0).tolist()],
        device=device,
        dtype=torch.long,
    )
    return q, k, v, g, beta, h0, output_grad, ht_grad, cu


def _rank_varlen_slices(
    cu_seqlens: torch.Tensor, *, rank: int, cp_size: int
) -> tuple[tuple[int, int], ...]:
    offsets = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    slices = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        length = end - start
        shard_start = start + (length * rank) // cp_size
        shard_end = start + (length * (rank + 1)) // cp_size
        if shard_start >= shard_end:
            raise ValueError("test varlen chain unexpectedly produced an empty shard")
        slices.append((shard_start, shard_end))
    return tuple(slices)


def _local_cu_seqlens_cpu(slices: tuple[tuple[int, int], ...]) -> torch.Tensor:
    lengths = [end - start for start, end in slices]
    return torch.tensor(
        [0, *torch.cumsum(torch.tensor(lengths), dim=0).tolist()],
        dtype=torch.long,
    )


def _lengths_by_rank_cpu(cu_seqlens: torch.Tensor, *, cp_size: int) -> torch.Tensor:
    rows = []
    for rank in range(cp_size):
        rank_slices = _rank_varlen_slices(cu_seqlens, rank=rank, cp_size=cp_size)
        rows.append([end - start for start, end in rank_slices])
    return torch.tensor(rows, dtype=torch.long)


def _cat_varlen_slices(
    tensor: torch.Tensor,
    slices: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    return torch.cat([tensor[:, start:end] for start, end in slices], dim=1)


def _assert_grad_close(left: torch.Tensor, right_grad: torch.Tensor, name: str) -> None:
    assert left.grad is not None, name
    assert_mean_abs_pct(right_grad, left.grad, name)
