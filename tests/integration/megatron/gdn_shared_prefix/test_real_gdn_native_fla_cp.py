from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (  # noqa: E402
    Qwen3_5MoeVisionConfig,
    Qwen35VLMoEModelProvider,
)
from megatron.core import parallel_state as ps  # noqa: E402
from megatron.core.ssm.gated_delta_net import GatedDeltaNet  # noqa: E402
from megatron.core.tensor_parallel.random import (  # noqa: E402
    model_parallel_cuda_manual_seed,
)
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from art.megatron.gdn.gdn_shared_prefix import (  # noqa: E402
    GdnPlannerConfig,
    GdnSegmentBucketPlan,
    build_gdn_rank_execution_plan,
    parse_gdn_shared_prefix_segments,
)
from art.megatron.gdn.operator import (  # noqa: E402
    _project_gdn_inputs,
    _zero_conv_state,
    _zero_recurrent_state,
    run_gdn_bucket,
    run_gdn_layer,
)

from .cases import GdnFamilyShape, GdnPackedRowShape, GdnPhase0Case  # noqa: E402
from .distributed_init import file_init_method  # noqa: E402
from .metrics import (  # noqa: E402
    GDN_CORRECTNESS_DTYPE,
    MEAN_ABS_PCT_THRESHOLD,
    assert_mean_abs_pct,
    parameter_grad_mean_abs_pct_with_name,
)
from .packed_layout import build_phase0_packed_tensors  # noqa: E402
from .real_gdn_oracle import (  # noqa: E402
    attach_main_grads,
    zero_parameter_grads,
)

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
    reason="At least four CUDA devices are required for native FLA CP GDN coverage.",
)
@pytest.mark.parametrize("cp_size", _CP_SIZES)
def test_real_qwen35_gdn_native_fla_cp_prepared_varlen_batch_matches_single_rank(
    cp_size: int, tmp_path: Path
) -> None:
    init_method = file_init_method(tmp_path, f"native_gdn_prepared_cp{cp_size}")
    mp.spawn(
        _native_gdn_cp_prepared_varlen_worker,
        args=(cp_size, init_method, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"prepared_varlen_rank_{rank}.ok").read_text() == "ok\n"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="At least four CUDA devices are required for native packed CP GDN coverage.",
)
@pytest.mark.parametrize("cp_size", _CP_SIZES)
def test_real_qwen35_gdn_native_cp_packed_layer_matches_cp1(
    cp_size: int, tmp_path: Path
) -> None:
    init_method = file_init_method(tmp_path, f"native_gdn_packed_cp{cp_size}")
    mp.spawn(
        _native_gdn_cp_packed_layer_worker,
        args=(cp_size, init_method, str(tmp_path)),
        nprocs=cp_size,
        join=True,
    )
    for rank in range(cp_size):
        assert (tmp_path / f"packed_layer_rank_{rank}.ok").read_text() == "ok\n"


def _native_gdn_cp_packed_layer_worker(
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
        zero_parameter_grads(ref_gdn)
        zero_parameter_grads(cp_gdn)
        case = _packed_native_cp_case()
        tensors = build_phase0_packed_tensors(case)
        group_ids = tensors["group_ids"].cuda()
        parent_ids = tensors["parent_ids"].cuda()
        spec = parse_gdn_shared_prefix_segments(group_ids, parent_ids)
        plan = build_gdn_rank_execution_plan(
            spec,
            device=group_ids.device,
            cp_rank=rank,
            cp_size=cp_size,
            planner_config=GdnPlannerConfig(
                cp_chain_min_tokens_per_rank=16,
                cp_chain_min_total_tokens=128,
                cp_chain_min_prefix_only_tokens=128,
            ),
        )
        assert any(plan.tree_chain_buckets_by_depth)
        hidden, output_grad = _packed_hidden_and_grad(case, cp_size)
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
        local_output_grad = (
            flat_grad.index_select(0, local_index).unsqueeze(1).contiguous()
        )
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
        _all_reduce_parameter_grads(cp_gdn)

        flat_ref_out = ref_out.detach().transpose(0, 1).reshape(-1, ref_out.shape[-1])
        assert_mean_abs_pct(
            flat_ref_out.index_select(0, local_index),
            cp_out.detach().squeeze(1),
            "packed_output",
        )
        assert local_hidden.grad is not None
        assert ref_hidden.grad is not None
        flat_ref_grad = ref_hidden.grad.transpose(0, 1).reshape(-1, hidden.shape[-1])
        assert_mean_abs_pct(
            flat_ref_grad.index_select(0, local_index),
            local_hidden.grad.squeeze(1),
            "packed_hidden_grad",
        )
        param_name, param_pct = parameter_grad_mean_abs_pct_with_name(ref_gdn, cp_gdn)
        assert param_pct <= MEAN_ABS_PCT_THRESHOLD, param_name
        Path(output_dir, f"packed_layer_rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _native_gdn_cp_prepared_varlen_worker(
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
        zero_parameter_grads(ref_gdn)
        zero_parameter_grads(cp_gdn)
        hidden, lengths = _varlen_hidden_and_lengths(cp_size)
        with torch.no_grad():
            qkv_full, _, beta_full, recurrent_g_full = _project_gdn_inputs(
                ref_gdn, hidden
            )
        bucket = _varlen_bucket(lengths, device=hidden.device)
        conv0_ref = _zero_conv_state(
            ref_gdn, hidden, batch_size=int(lengths.numel())
        ).requires_grad_(True)
        rec0_ref = _zero_recurrent_state(
            ref_gdn, hidden, batch_size=int(lengths.numel())
        ).requires_grad_(True)
        conv_grad = torch.randn_like(conv0_ref)
        rec_grad = torch.randn_like(rec0_ref)
        output_grad = torch.randn(
            1,
            int(lengths.sum().item()),
            cast(int, cp_gdn.num_value_heads) // cast(int, cp_gdn.tp_size),
            cast(int, cp_gdn.value_head_dim),
            device=hidden.device,
            dtype=GDN_CORRECTNESS_DTYPE,
        )
        torch.distributed.broadcast(conv_grad, src=0)
        torch.distributed.broadcast(rec_grad, src=0)
        torch.distributed.broadcast(output_grad, src=0)

        full_offsets = tuple((0, int(length.item())) for length in lengths)
        ref_qkv = _cat_time_slices(qkv_full, full_offsets).requires_grad_(True)
        ref_beta = _cat_time_slices(beta_full, full_offsets).requires_grad_(True)
        ref_g = _cat_time_slices(recurrent_g_full, full_offsets).requires_grad_(True)
        ref_out, ref_conv, ref_rec = run_gdn_bucket(
            bucket,
            (ref_qkv, ref_beta, ref_g),
            (conv0_ref, rec0_ref),
            gdn=ref_gdn,
            output_final_state=True,
        )
        assert ref_conv is not None
        assert ref_rec is not None
        ref_loss = (
            (ref_out * output_grad).sum()
            + (ref_conv * conv_grad).sum()
            + (ref_rec * rec_grad).sum()
        )
        ref_loss.backward()

        local_offsets = _rank_varlen_offsets(lengths, rank=rank, cp_size=cp_size)
        local_lengths = torch.tensor(
            [end - start for start, end in local_offsets],
            device=hidden.device,
            dtype=torch.long,
        )
        local_bucket = _varlen_bucket(
            local_lengths,
            device=hidden.device,
            lengths_by_rank_cpu=_varlen_lengths_by_rank_cpu(
                lengths,
                cp_size=cp_size,
            ),
        )
        local_qkv = _cat_time_slices(qkv_full, local_offsets).requires_grad_(True)
        local_beta = _cat_time_slices(beta_full, local_offsets).requires_grad_(True)
        local_g = _cat_time_slices(recurrent_g_full, local_offsets).requires_grad_(True)
        conv0_cp = conv0_ref.detach().clone().requires_grad_(True)
        rec0_cp = rec0_ref.detach().clone().requires_grad_(True)
        cp_out, cp_conv, cp_rec = run_gdn_bucket(
            local_bucket,
            (local_qkv, local_beta, local_g),
            (conv0_cp, rec0_cp),
            gdn=cp_gdn,
            group=torch.distributed.group.WORLD,
            recurrent_cp=True,
            output_final_state=True,
        )
        assert cp_conv is not None
        assert cp_rec is not None
        local_output_grad = _cat_flat_slices(
            output_grad, bucket.cu_seqlens, local_offsets
        )
        cp_loss = (
            (cp_out * local_output_grad).sum()
            + (cp_conv * (conv_grad / cp_size)).sum()
            + (cp_rec * (rec_grad / cp_size)).sum()
        )
        cp_loss.backward()
        _all_reduce_parameter_grads(cp_gdn)

        assert_mean_abs_pct(
            _cat_flat_slices(ref_out, bucket.cu_seqlens, local_offsets),
            cp_out,
            "prepared_varlen_output",
        )
        assert_mean_abs_pct(ref_conv, cp_conv, "prepared_varlen_conv_final")
        assert_mean_abs_pct(ref_rec, cp_rec, "prepared_varlen_recurrent_final")
        assert ref_qkv.grad is not None
        assert ref_beta.grad is not None
        assert ref_g.grad is not None
        _assert_compact_grad_slices(
            local_qkv, ref_qkv.grad, bucket.cu_seqlens, local_offsets, "qkv"
        )
        _assert_compact_grad_slices(
            local_beta, ref_beta.grad, bucket.cu_seqlens, local_offsets, "beta"
        )
        _assert_compact_grad_slices(
            local_g, ref_g.grad, bucket.cu_seqlens, local_offsets, "g"
        )
        assert conv0_cp.grad is not None
        assert conv0_ref.grad is not None
        assert rec0_cp.grad is not None
        assert rec0_ref.grad is not None
        assert_mean_abs_pct(conv0_ref.grad, conv0_cp.grad, "prepared_conv_grad")
        assert_mean_abs_pct(rec0_ref.grad, rec0_cp.grad, "prepared_rec_grad")
        param_name, param_pct = parameter_grad_mean_abs_pct_with_name(ref_gdn, cp_gdn)
        assert param_pct <= MEAN_ABS_PCT_THRESHOLD, param_name
        Path(output_dir, f"prepared_varlen_rank_{rank}.ok").write_text("ok\n")
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _make_matching_gdn_pair(
    *, cp_size: int, params_dtype: torch.dtype = GDN_CORRECTNESS_DTYPE
) -> tuple[GatedDeltaNet, GatedDeltaNet]:
    model_parallel_cuda_manual_seed(1234)
    ref_model = _make_model(cp_size=cp_size, params_dtype=params_dtype)
    model_parallel_cuda_manual_seed(5678)
    cp_model = _make_model(cp_size=cp_size, params_dtype=params_dtype)
    ref_gdn = _first_gdn(ref_model)
    cp_gdn = _first_gdn(cp_model)
    cp_gdn.load_state_dict(ref_gdn.state_dict())
    attach_main_grads(ref_gdn)
    attach_main_grads(cp_gdn)
    return ref_gdn, cp_gdn


def _make_model(
    *, cp_size: int, params_dtype: torch.dtype = GDN_CORRECTNESS_DTYPE
) -> torch.nn.Module:
    assert Qwen3_5MoeVisionConfig is not None
    provider = Qwen35VLMoEModelProvider(
        num_layers=4,
        hidden_size=64,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=32,
        moe_shared_expert_intermediate_size=16,
        num_attention_heads=4,
        num_query_groups=1,
        kv_channels=16,
        linear_key_head_dim=8,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_output_gate=True,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=2,
        vocab_size=128,
        seq_length=128,
        position_embedding_type="mrope",
        vision_config=Qwen3_5MoeVisionConfig(),
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        # Megatron's stock GDN config still rejects CP. This test owns CP at the
        # ART wrapper boundary and uses the distributed WORLD group explicitly.
        context_parallel_size=1,
        params_dtype=params_dtype,
    )
    provider.finalize()
    return provider.provide_language_model(pre_process=True, post_process=True).cuda()


def _first_gdn(model: torch.nn.Module) -> GatedDeltaNet:
    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            return module
    raise AssertionError("expected Qwen3.5 provider to build a GDN layer")


def _packed_native_cp_case() -> GdnPhase0Case:
    return GdnPhase0Case(
        name="native_cp_packed_varying",
        sequence_length=3072,
        rows=(
            GdnPackedRowShape(
                families=(
                    GdnFamilyShape(prefix_length=1024, suffix_lengths=(512, 512)),
                    GdnFamilyShape(prefix_length=512, suffix_lengths=(512,)),
                )
            ),
        ),
        seed=67,
        description="Mixed long CP-chain and short local-fork GDN segments.",
    )


def _packed_hidden_and_grad(
    case: GdnPhase0Case, cp_size: int, *, dtype: torch.dtype = GDN_CORRECTNESS_DTYPE
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20490426 + cp_size)
    hidden = torch.randn(
        case.sequence_length,
        len(case.rows),
        64,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    output_grad = torch.randn(
        hidden.shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    torch.distributed.broadcast(hidden, src=0)
    torch.distributed.broadcast(output_grad, src=0)
    return hidden, output_grad


def _varlen_hidden_and_lengths(cp_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    lengths = torch.tensor((512, 1024, 1536), device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(20480426 + cp_size)
    hidden = torch.randn(
        int(lengths.max().item()),
        int(lengths.numel()),
        64,
        device=device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    torch.distributed.broadcast(hidden, src=0)
    return hidden, lengths


def _varlen_bucket(
    lengths: torch.Tensor,
    *,
    device: torch.device,
    lengths_by_rank_cpu: torch.Tensor | None = None,
) -> GdnSegmentBucketPlan:
    max_len = int(lengths.max().item())
    lengths_cpu = lengths.detach().cpu()
    cu_seqlens_cpu = torch.cat(
        [lengths_cpu.new_zeros(1), torch.cumsum(lengths_cpu, dim=0)]
    )
    offsets = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(1)
    real_mask = offsets < lengths.unsqueeze(0)
    return GdnSegmentBucketPlan(
        length=max_len,
        lengths=lengths,
        lengths_cpu=lengths_cpu,
        lengths_by_rank_cpu=lengths_by_rank_cpu,
        real_mask=real_mask,
        cu_seqlens=cu_seqlens_cpu.to(device=device),
        cu_seqlens_cpu=cu_seqlens_cpu,
        row_indices=torch.arange(int(lengths.numel()), device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(max_len, -1)
        .contiguous(),
        position_indices=offsets.expand(-1, int(lengths.numel())).contiguous(),
        family_indices=torch.arange(
            int(lengths.numel()), device=device, dtype=torch.long
        ),
        real_token_count_static=int(lengths.sum().item()),
    )


def _rank_varlen_offsets(
    lengths: torch.Tensor, *, rank: int, cp_size: int
) -> tuple[tuple[int, int], ...]:
    offsets = []
    for length in (int(value) for value in lengths.detach().cpu().tolist()):
        start = (length * rank) // cp_size
        end = (length * (rank + 1)) // cp_size
        if start >= end:
            raise ValueError("test varlen chain unexpectedly produced an empty shard")
        offsets.append((start, end))
    return tuple(offsets)


def _varlen_lengths_by_rank_cpu(lengths: torch.Tensor, *, cp_size: int) -> torch.Tensor:
    return torch.tensor(
        [
            [
                end - start
                for start, end in _rank_varlen_offsets(
                    lengths,
                    rank=rank,
                    cp_size=cp_size,
                )
            ]
            for rank in range(cp_size)
        ],
        dtype=torch.long,
    )


def _cat_time_slices(
    tensor: torch.Tensor, offsets: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    return torch.cat(
        [tensor[index, start:end] for index, (start, end) in enumerate(offsets)],
        dim=0,
    ).contiguous()


def _cat_flat_slices(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    offsets: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    pieces = []
    for chain, (start, end) in enumerate(offsets):
        base = int(cu_seqlens[chain].item())
        pieces.append(tensor[:, base + start : base + end])
    return torch.cat(pieces, dim=1).contiguous()


def _assert_compact_grad_slices(
    local: torch.Tensor,
    reference_grad: torch.Tensor,
    cu_seqlens: torch.Tensor,
    offsets: tuple[tuple[int, int], ...],
    name: str,
) -> None:
    assert local.grad is not None, name
    expected = _cat_flat_slices(
        reference_grad.unsqueeze(0), cu_seqlens, offsets
    ).squeeze(0)
    assert_mean_abs_pct(expected, local.grad, name)


def _all_reduce_parameter_grads(module: torch.nn.Module) -> None:
    world_size = torch.distributed.get_world_size()
    for parameter in module.parameters():
        has_grad = torch.tensor(
            1 if parameter.grad is not None else 0,
            device=parameter.device,
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(has_grad)
        grad_ranks = int(has_grad.item())
        if grad_ranks == world_size:
            assert parameter.grad is not None
            torch.distributed.all_reduce(parameter.grad)
        elif grad_ranks:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            torch.distributed.all_reduce(parameter.grad)
        main_grad = getattr(parameter, "main_grad", None)
        if main_grad is not None:
            torch.distributed.all_reduce(main_grad)
