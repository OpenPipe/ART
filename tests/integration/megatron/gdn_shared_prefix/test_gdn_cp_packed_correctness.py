from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.core import parallel_state as ps  # noqa: E402
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from art.megatron.gdn.gdn_shared_prefix import (  # noqa: E402
    GdnPlannerConfig,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.gdn.operator import run_gdn_layer  # noqa: E402
from art.megatron.shared_prefix_packing import pack_shared_prefixes  # noqa: E402

from .cases import (  # noqa: E402
    GdnFamilyShape,
    GdnPackedRowShape,
    GdnPhase0Case,
    default_phase0_cases,
)
from .distributed_grad import all_reduce_parameter_grads_coalesced  # noqa: E402
from .distributed_init import file_init_method  # noqa: E402
from .metrics import (  # noqa: E402
    GDN_CORRECTNESS_DTYPE,
    REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD,
    REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD,
    assert_mean_abs_pct,
    assert_scalar_loss_close,
    parameter_grad_mean_abs_pct_with_name,
    stable_output_mse_loss,
)
from .packed_layout import build_phase0_packed_tensors  # noqa: E402
from .real_gdn_oracle import zero_parameter_grads  # noqa: E402
from .test_real_gdn_native_fla_cp import _make_matching_gdn_pair  # noqa: E402

_CP_SIZES = (2, 4, 8)


@pytest.mark.parametrize("cp_size", _CP_SIZES)
def test_gdn_cp_packed_matches_cp1_oracle_all_edge_cases(
    cp_size: int, tmp_path: Path
) -> None:
    _skip_without_gpus(cp_size)
    init_method = file_init_method(tmp_path, f"cp1_oracle_cp{cp_size}")
    mp.spawn(
        _cp1_oracle_worker,
        args=(cp_size, init_method, str(tmp_path), False),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"cp1_oracle_rank_{rank}.ok").read_text() == "ok\n"


@pytest.mark.parametrize("cp_size", _CP_SIZES)
def test_gdn_cp_packed_sibling_order_matches_cp1_oracle(
    cp_size: int, tmp_path: Path
) -> None:
    _skip_without_gpus(cp_size)
    init_method = file_init_method(tmp_path, f"cp1_oracle_sibling_cp{cp_size}")
    mp.spawn(
        _cp1_oracle_worker,
        args=(cp_size, init_method, str(tmp_path), True),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"cp1_oracle_sibling_rank_{rank}.ok").read_text() == "ok\n"


@pytest.mark.parametrize("cp_size", (2, 4))
def test_gdn_cp_tree_chain_matches_cp1_oracle(cp_size: int, tmp_path: Path) -> None:
    _skip_without_gpus(cp_size)
    init_method = file_init_method(tmp_path, f"tree_chain_cp{cp_size}")
    mp.spawn(
        _tree_chain_oracle_worker,
        args=(cp_size, init_method, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"tree_chain_rank_{rank}.ok").read_text() == "ok\n"


def test_gdn_cp_tree_fuzz_matches_cp1_oracle(tmp_path: Path) -> None:
    cp_size = 4
    _skip_without_gpus(cp_size)
    init_method = file_init_method(tmp_path, "tree_fuzz_cp4")
    mp.spawn(
        _tree_fuzz_oracle_worker,
        args=(cp_size, init_method, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"tree_fuzz_rank_{rank}.ok").read_text() == "ok\n"


def _cp1_oracle_worker(
    rank: int,
    cp_size: int,
    init_method: str,
    output_dir: str,
    sibling_only: bool,
) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=cp_size,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        ref_gdn, cp_gdn = _make_matching_gdn_pair(cp_size=cp_size)
        if sibling_only:
            _assert_sibling_order_matches_cp1(
                ref_gdn,
                cp_gdn,
                rank=rank,
                cp_size=cp_size,
            )
            Path(output_dir, f"cp1_oracle_sibling_rank_{rank}.ok").write_text("ok\n")
            return
        for case_index, case in enumerate(_packed_correctness_cases()):
            _assert_case_matches_cp1(
                ref_gdn,
                cp_gdn,
                case,
                rank=rank,
                cp_size=cp_size,
                seed=20510426 + 1000 * cp_size + case_index,
                planner_config=_planner_config_for_case(case),
            )
            torch.distributed.barrier()
        Path(output_dir, f"cp1_oracle_rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _tree_chain_oracle_worker(
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
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        ref_gdn, cp_gdn = _make_matching_gdn_pair(cp_size=cp_size)
        _assert_tree_pack_matches_cp1(
            "tree_chain",
            ref_gdn,
            cp_gdn,
            _tree_chain_pack(),
            rank=rank,
            cp_size=cp_size,
            seed=9090,
            planner_config=_tree_chain_planner_config(),
            require_chain=True,
        )
        Path(output_dir, f"tree_chain_rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _tree_fuzz_oracle_worker(
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
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        ref_gdn, cp_gdn = _make_matching_gdn_pair(cp_size=cp_size)
        for case_index, (name, pack) in enumerate(_tree_fuzz_packs()):
            _assert_tree_pack_matches_cp1(
                name,
                ref_gdn,
                cp_gdn,
                pack,
                rank=rank,
                cp_size=cp_size,
                seed=9190 + case_index,
                planner_config=_tree_fuzz_planner_config(),
                require_chain=False,
            )
            torch.distributed.barrier()
        Path(output_dir, f"tree_fuzz_rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _assert_case_matches_cp1(
    ref_gdn: torch.nn.Module,
    cp_gdn: torch.nn.Module,
    case: GdnPhase0Case,
    *,
    rank: int,
    cp_size: int,
    seed: int,
    planner_config: GdnPlannerConfig | None,
) -> None:
    zero_parameter_grads(ref_gdn)
    zero_parameter_grads(cp_gdn)
    tensors = build_phase0_packed_tensors(case)
    group_ids = tensors["group_ids"].cuda()
    parent_ids = tensors["parent_ids"].cuda()
    spec = parse_gdn_shared_prefix_segments(group_ids, parent_ids)
    plan = build_gdn_rank_execution_plan(
        spec,
        device=group_ids.device,
        cp_rank=rank,
        cp_size=cp_size,
        planner_config=planner_config or GdnPlannerConfig(),
    )
    hidden, output_grad = _hidden_and_grad(case, seed=seed)
    real_mask = (group_ids != -1).transpose(0, 1).unsqueeze(-1)
    output_grad = output_grad * real_mask
    loss_denominator = real_mask.expand_as(output_grad).sum()
    ref_hidden = hidden.clone().detach().requires_grad_(True)
    ref_out, _ = run_gdn_layer(
        ref_gdn,
        ref_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    ref_loss = stable_output_mse_loss(
        ref_out,
        output_grad,
        mask=real_mask,
        denominator=loss_denominator,
    )
    ref_loss.backward()

    flat_hidden = hidden.transpose(0, 1).reshape(-1, hidden.shape[-1])
    flat_grad = output_grad.transpose(0, 1).reshape(-1, output_grad.shape[-1])
    local_index = torch.tensor(
        plan.attention_token_indices, device=hidden.device, dtype=torch.long
    )
    local_hidden = (
        flat_hidden.index_select(0, local_index)
        .unsqueeze(1)
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    local_output_grad = flat_grad.index_select(0, local_index).unsqueeze(1).contiguous()
    cp_out, _ = run_gdn_layer(
        cp_gdn,
        local_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
        execution_spec=spec,
        execution_plan=plan,
        cp_group=torch.distributed.group.WORLD,
    )
    cp_loss = stable_output_mse_loss(
        cp_out,
        local_output_grad,
        denominator=loss_denominator,
    )
    cp_loss.backward()
    _assert_cp_matches_reference(
        case.name,
        ref_gdn,
        cp_gdn,
        ref_hidden,
        ref_out,
        ref_loss.detach(),
        local_hidden,
        cp_out,
        cp_loss.detach(),
        local_index,
    )


def _assert_tree_pack_matches_cp1(
    name: str,
    ref_gdn: torch.nn.Module,
    cp_gdn: torch.nn.Module,
    pack: Any,
    *,
    rank: int,
    cp_size: int,
    seed: int,
    planner_config: GdnPlannerConfig,
    require_chain: bool,
) -> None:
    zero_parameter_grads(ref_gdn)
    zero_parameter_grads(cp_gdn)
    group_ids = pack.group_ids.cuda()
    parent_ids = pack.parent_ids.cuda()
    spec = parse_gdn_shared_prefix_segments(group_ids, parent_ids)
    plan = build_gdn_rank_execution_plan(
        spec,
        device=group_ids.device,
        cp_rank=rank,
        cp_size=cp_size,
        planner_config=planner_config,
    )
    if require_chain:
        assert any(plan.tree_chain_buckets_by_depth)
    hidden, output_grad = _tree_hidden_and_grad(spec.real_token_count, seed=seed)
    ref_hidden = hidden.clone().detach().requires_grad_(True)
    ref_out, _ = run_gdn_layer(
        ref_gdn,
        ref_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    ref_loss = (ref_out * output_grad).sum()
    ref_loss.backward()

    flat_hidden = hidden.transpose(0, 1).reshape(-1, hidden.shape[-1])
    flat_grad = output_grad.transpose(0, 1).reshape(-1, output_grad.shape[-1])
    local_index = torch.tensor(
        plan.attention_token_indices, device=hidden.device, dtype=torch.long
    )
    local_hidden = (
        flat_hidden.index_select(0, local_index)
        .unsqueeze(1)
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    local_output_grad = flat_grad.index_select(0, local_index).unsqueeze(1).contiguous()
    cp_out, _ = run_gdn_layer(
        cp_gdn,
        local_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
        execution_spec=spec,
        execution_plan=plan,
        cp_group=torch.distributed.group.WORLD,
    )
    cp_loss = (cp_out * local_output_grad).sum()
    cp_loss.backward()
    _assert_cp_matches_reference(
        name,
        ref_gdn,
        cp_gdn,
        ref_hidden,
        ref_out,
        ref_loss.detach(),
        local_hidden,
        cp_out,
        cp_loss.detach(),
        local_index,
    )


def _assert_sibling_order_matches_cp1(
    ref_gdn: torch.nn.Module,
    cp_gdn: torch.nn.Module,
    *,
    rank: int,
    cp_size: int,
) -> None:
    case = _sibling_case()
    zero_parameter_grads(ref_gdn)
    zero_parameter_grads(cp_gdn)
    tensors = build_phase0_packed_tensors(case)
    group_ids = tensors["group_ids"].cuda()
    parent_ids = tensors["parent_ids"].cuda()
    swapped_group_ids = torch.full_like(group_ids, -1)
    swapped_parent_ids = torch.full_like(parent_ids, -1)
    swapped_group_ids[0, :5] = 0
    swapped_parent_ids[0, :5] = 0
    swapped_group_ids[0, 5:9] = 1
    swapped_parent_ids[0, 5:9] = 0
    swapped_group_ids[0, 9:12] = 2
    swapped_parent_ids[0, 9:12] = 0
    spec = parse_gdn_shared_prefix_segments(swapped_group_ids, swapped_parent_ids)
    plan = build_gdn_rank_execution_plan(
        spec,
        device=group_ids.device,
        cp_rank=rank,
        cp_size=cp_size,
        planner_config=GdnPlannerConfig(),
    )
    hidden, output_grad = _hidden_and_grad(case, seed=20520426 + cp_size)
    real_mask = (group_ids != -1).transpose(0, 1).unsqueeze(-1)
    output_grad = output_grad * real_mask
    loss_denominator = real_mask.expand_as(output_grad).sum()
    swapped_hidden = _swap_siblings(hidden)
    swapped_grad = _swap_siblings(output_grad)

    ref_hidden = hidden.clone().detach().requires_grad_(True)
    ref_out, _ = run_gdn_layer(
        ref_gdn,
        ref_hidden,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    ref_loss = stable_output_mse_loss(
        ref_out,
        output_grad,
        mask=real_mask,
        denominator=loss_denominator,
    )
    ref_loss.backward()

    flat_hidden = swapped_hidden.transpose(0, 1).reshape(-1, hidden.shape[-1])
    flat_grad = swapped_grad.transpose(0, 1).reshape(-1, output_grad.shape[-1])
    local_index = torch.tensor(
        plan.attention_token_indices, device=hidden.device, dtype=torch.long
    )
    local_hidden = (
        flat_hidden.index_select(0, local_index)
        .unsqueeze(1)
        .contiguous()
        .detach()
        .requires_grad_(True)
    )
    local_output_grad = flat_grad.index_select(0, local_index).unsqueeze(1).contiguous()
    cp_out, _ = run_gdn_layer(
        cp_gdn,
        local_hidden,
        group_ids=swapped_group_ids,
        parent_ids=swapped_parent_ids,
        execution_spec=spec,
        execution_plan=plan,
        cp_group=torch.distributed.group.WORLD,
    )
    cp_loss = stable_output_mse_loss(
        cp_out,
        local_output_grad,
        denominator=loss_denominator,
    )
    cp_loss.backward()
    expected_out = _swap_siblings(ref_out)
    assert ref_hidden.grad is not None
    expected_grad = _swap_siblings(ref_hidden.grad)
    _assert_cp_matches_reference(
        case.name,
        ref_gdn,
        cp_gdn,
        _TensorGradView(expected_grad),
        expected_out,
        ref_loss.detach(),
        local_hidden,
        cp_out,
        cp_loss.detach(),
        local_index,
    )


def _assert_cp_matches_reference(
    name: str,
    ref_gdn: torch.nn.Module,
    cp_gdn: torch.nn.Module,
    ref_hidden: Any,
    ref_out: torch.Tensor,
    ref_loss: torch.Tensor,
    local_hidden: torch.Tensor,
    cp_out: torch.Tensor,
    cp_loss: torch.Tensor,
    local_index: torch.Tensor,
) -> None:
    torch.distributed.all_reduce(cp_loss, op=torch.distributed.ReduceOp.SUM)
    all_reduce_parameter_grads_coalesced(cp_gdn)
    torch.cuda.synchronize()
    flat_ref_out = ref_out.detach().transpose(0, 1).reshape(-1, ref_out.shape[-1])
    assert_scalar_loss_close(ref_loss, cp_loss, f"{name}:loss")
    if int(local_index.numel()) != 0:
        assert_mean_abs_pct(
            flat_ref_out.index_select(0, local_index),
            cp_out.detach().squeeze(1),
            f"{name}:output",
            threshold=REAL_GDN_OUTPUT_MEAN_ABS_PCT_THRESHOLD,
        )
        assert local_hidden.grad is not None
        flat_ref_grad = ref_hidden.grad.transpose(0, 1).reshape(
            -1, local_hidden.shape[-1]
        )
        assert_mean_abs_pct(
            flat_ref_grad.index_select(0, local_index),
            local_hidden.grad.squeeze(1),
            f"{name}:hidden_grad",
            threshold=REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD,
        )
    param_name, param_pct = parameter_grad_mean_abs_pct_with_name(ref_gdn, cp_gdn)
    assert param_pct <= REAL_GDN_GRAD_MEAN_ABS_PCT_THRESHOLD, f"{name}:{param_name}"
    torch.cuda.synchronize()


class _TensorGradView:
    def __init__(self, grad: torch.Tensor) -> None:
        self.grad = grad


def _hidden_and_grad(
    case: GdnPhase0Case, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        case.sequence_length,
        len(case.rows),
        64,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    grad = torch.randn(
        hidden.shape,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    torch.distributed.broadcast(hidden, src=0)
    torch.distributed.broadcast(grad, src=0)
    return hidden, grad


def _tree_hidden_and_grad(
    sequence_length: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        sequence_length,
        1,
        64,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    grad = torch.randn(
        hidden.shape,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    torch.distributed.broadcast(hidden, src=0)
    torch.distributed.broadcast(grad, src=0)
    return hidden, grad


def _tree_chain_pack():
    long_root = torch.arange(11, 267)
    short_root = torch.arange(1001, 1097)
    long_mid = torch.arange(2001, 2641)
    other_mid = torch.arange(3001, 3065)
    return pack_shared_prefixes(
        (
            torch.cat((long_root, torch.tensor([301]))),
            torch.cat((long_root, torch.tensor([302]))),
            torch.cat((short_root, long_mid, torch.tensor([401]))),
            torch.cat((short_root, long_mid, torch.tensor([402]))),
            torch.cat((short_root, other_mid, torch.tensor([403]))),
        ),
        max_depth=2,
    )


def _tree_chain_planner_config() -> GdnPlannerConfig:
    return GdnPlannerConfig(
        cp_chain_min_tokens_per_rank=16,
        cp_chain_min_total_tokens=128,
        cp_chain_min_prefix_only_tokens=128,
        max_padding_ratio=4.0,
    )


def _tree_fuzz_planner_config() -> GdnPlannerConfig:
    return GdnPlannerConfig(
        cp_chain_min_tokens_per_rank=1,
        cp_chain_min_total_tokens=64,
        cp_chain_min_prefix_only_tokens=64,
        cp_tree_chain_min_total_tokens=64,
        cp_tree_chain_min_prefix_only_tokens=64,
        max_padding_ratio=4.0,
    )


def _tree_fuzz_packs() -> tuple[tuple[str, Any], ...]:
    return (
        (
            "tree_fuzz_duplicates",
            pack_shared_prefixes(_duplicate_tree_sequences(), max_depth=4),
        ),
        (
            "tree_fuzz_ragged_depth4",
            pack_shared_prefixes(_random_tree_sequences(13, max_depth=4), max_depth=4),
        ),
        (
            "tree_fuzz_mixed_tiny_long",
            pack_shared_prefixes(_random_tree_sequences(29, max_depth=5), max_depth=5),
        ),
    )


def _duplicate_tree_sequences() -> tuple[torch.Tensor, ...]:
    root = torch.arange(11, 331)
    mid_a = torch.arange(1001, 1261)
    mid_b = torch.arange(2001, 2065)
    leaf_a = torch.arange(3001, 3013)
    leaf_b = torch.arange(4001, 4017)
    first = torch.cat((root, mid_a, leaf_a))
    second = torch.cat((root, mid_a, leaf_b))
    third = torch.cat((root, mid_b, torch.tensor([91, 92, 93])))
    return (first, first, second, third, third)


def _random_tree_sequences(seed: int, *, max_depth: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    next_token = 1

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, high + 1, (), generator=generator).item())

    def tokens(length: int) -> torch.Tensor:
        nonlocal next_token
        out = torch.arange(next_token, next_token + length)
        next_token += length + 997
        return out

    def segment_length(depth: int) -> int:
        choices = (1, 3, 17, 64, 129, 257, 384 if depth == 0 else 96)
        return choices[randint(0, len(choices) - 1)]

    def walk(prefix: torch.Tensor, depth: int) -> list[torch.Tensor]:
        here = torch.cat((prefix, tokens(segment_length(depth))))
        if depth + 1 >= max_depth:
            return [
                torch.cat((here, tokens(randint(1, 17)))) for _ in range(randint(2, 4))
            ]
        leaves: list[torch.Tensor] = []
        for _ in range(randint(2, 3)):
            leaves.extend(walk(here, depth + 1))
        return leaves

    return tuple(walk(torch.empty(0, dtype=torch.long), 0))


def _packed_correctness_cases() -> tuple[GdnPhase0Case, ...]:
    return (
        *default_phase0_cases(conv_width=2),
        _mixed_local_chain_case(),
        _local_prefix_chain_completion_case(),
    )


def _planner_config_for_case(case: GdnPhase0Case) -> GdnPlannerConfig | None:
    if case.name != "mixed_local_chain_edge":
        return None
    return GdnPlannerConfig(
        cp_chain_min_tokens_per_rank=16,
        cp_chain_min_total_tokens=128,
        cp_chain_min_prefix_only_tokens=128,
    )


def _mixed_local_chain_case() -> GdnPhase0Case:
    return GdnPhase0Case(
        name="mixed_local_chain_edge",
        sequence_length=960,
        rows=(
            GdnPackedRowShape(
                families=(
                    GdnFamilyShape(prefix_length=256, suffix_lengths=(320, 64)),
                    GdnFamilyShape(prefix_length=12, suffix_lengths=(7, 5, 9)),
                    GdnFamilyShape(prefix_length=128, suffix_lengths=(80, 32)),
                )
            ),
        ),
        seed=67,
        description="One row mixing long native CP-chain work and short local-fork siblings.",
    )


def _local_prefix_chain_completion_case() -> GdnPhase0Case:
    return GdnPhase0Case(
        name="local_prefix_chain_completion_edge",
        sequence_length=768,
        rows=(
            GdnPackedRowShape(
                families=(GdnFamilyShape(prefix_length=96, suffix_lengths=(640, 17)),)
            ),
        ),
        seed=71,
        description="Short local prefix feeding a long native CP-chain completion.",
    )


def _sibling_case() -> GdnPhase0Case:
    return GdnPhase0Case(
        name="sibling_order_edge",
        sequence_length=16,
        rows=(
            GdnPackedRowShape(
                families=(GdnFamilyShape(prefix_length=5, suffix_lengths=(3, 4)),)
            ),
        ),
        seed=59,
    )


def _swap_siblings(tensor: torch.Tensor) -> torch.Tensor:
    swapped = tensor.clone()
    swapped[5:9] = tensor[8:12]
    swapped[9:12] = tensor[5:8]
    return swapped


def _skip_without_gpus(cp_size: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < cp_size:
        pytest.skip(f"Need {cp_size} CUDA devices for CP{cp_size} packed GDN.")
