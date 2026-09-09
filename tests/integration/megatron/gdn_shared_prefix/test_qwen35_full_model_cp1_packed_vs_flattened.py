from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen3_5MoeVisionConfig,
    Qwen35VLMoEModelProvider,
)
from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from torch.distributed import destroy_process_group, init_process_group, is_initialized
import torch.nn.functional as F

from art.loss import shift_tensor
from art.megatron.model_support.handlers.qwen3_5 import QWEN3_5_MOE_HANDLER
from art.megatron.prefix_tree_state import create_prefix_tree_state

from ..model_support.oracle_harness import TEST_DEFAULT_FLEX_BACKEND
from ..model_support.oracle_worker import (
    _apply_requested_flex_backend_patch,
    _apply_test_attention_full_fp32_patch,
    _apply_test_flex_inner_fp32_patch,
)
from .cases import default_phase0_cases
from .distributed_init import file_init_method
from .metrics import (
    GDN_CORRECTNESS_DTYPE,
    MEAN_ABS_PCT_THRESHOLD,
    assert_mean_abs_pct,
    mean_abs_pct,
    parameter_grad_mean_abs_pct_with_name,
    stable_output_mse_loss,
)
from .packed_layout import build_phase0_packed_tensors
from .parser_import import parse_gdn_prefix_tree_segments
from .real_gdn_oracle import (
    attach_main_grads,
    zero_parameter_grads,
)


@pytest.fixture(autouse=True)
def _fp32_test_flex_backend() -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            _apply_requested_flex_backend_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        stack.enter_context(
            _apply_test_flex_inner_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        stack.enter_context(
            _apply_test_attention_full_fp32_patch(TEST_DEFAULT_FLEX_BACKEND)
        )
        yield


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Qwen3.5 full-model prefix-tree oracle coverage.",
)
def test_qwen35_full_model_cp1_matches_flattened_grad_accumulation() -> None:
    with _single_rank_model_parallel():
        packed_model, flat_model = _make_matching_models()
        case = next(
            item
            for item in default_phase0_cases(conv_width=2)
            if item.name == "ragged_family_mix"
        )
        tensors = build_phase0_packed_tensors(case)
        device = torch.device("cuda")
        tokens = tensors["tokens"].remainder(128).to(device)
        input_pos = tensors["input_pos"].to(device)
        group_ids = tensors["group_ids"].to(device)
        parent_ids = tensors["parent_ids"].to(device)
        assistant_mask = tensors["assistant_mask"].to(device)

        zero_parameter_grads(packed_model)
        zero_parameter_grads(flat_model)
        packed_logits, packed_loss = _run_model_loss(
            packed_model,
            tokens=tokens,
            input_pos=input_pos,
            group_ids=group_ids,
            parent_ids=parent_ids,
            assistant_mask=assistant_mask,
        )
        packed_loss.backward()

        flat_loss_sum: torch.Tensor | None = None
        logits_mean_abs_pct = 0.0
        spec = parse_gdn_prefix_tree_segments(group_ids.cpu(), parent_ids.cpu())
        for segment_index, completion in enumerate(spec.tree_segments):
            if spec.tree_parent_indices[segment_index] < 0:
                continue
            row = completion.row_index
            path = _segment_path(spec, segment_index)
            completion_offset = sum(segment.length for segment in path[:-1])
            ref_tokens = torch.cat(
                [
                    tokens[row : row + 1, segment.start : segment.end]
                    for segment in path
                ],
                dim=1,
            )
            ref_pos = torch.cat(
                [
                    input_pos[row : row + 1, segment.start : segment.end]
                    for segment in path
                ],
                dim=1,
            )
            ref_assistant_mask = torch.cat(
                [
                    torch.zeros(
                        (1, completion_offset),
                        dtype=torch.bool,
                        device=device,
                    ),
                    assistant_mask[row : row + 1, completion.start : completion.end],
                ],
                dim=1,
            )
            ref_group_ids = torch.zeros_like(ref_tokens)
            ref_parent_ids = torch.zeros_like(ref_tokens)
            ref_logits, ref_loss = _run_model_loss(
                flat_model,
                tokens=ref_tokens,
                input_pos=ref_pos,
                group_ids=ref_group_ids,
                parent_ids=ref_parent_ids,
                assistant_mask=ref_assistant_mask,
            )
            ref_loss.backward()
            flat_loss_sum = (
                ref_loss.detach()
                if flat_loss_sum is None
                else flat_loss_sum + ref_loss.detach()
            )

            if completion.length > 1:
                packed_slice = packed_logits[
                    row : row + 1, completion.start : completion.end - 1
                ]
                ref_slice = ref_logits[
                    :, completion_offset : completion_offset + completion.length - 1
                ]
                logits_mean_abs_pct = max(
                    logits_mean_abs_pct,
                    mean_abs_pct(ref_slice, packed_slice),
                )

        assert flat_loss_sum is not None
        grad_name, grad_pct = parameter_grad_mean_abs_pct_with_name(
            flat_model, packed_model
        )
        assert_mean_abs_pct(flat_loss_sum, packed_loss.detach(), "loss")
        assert logits_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD
        assert grad_pct <= MEAN_ABS_PCT_THRESHOLD, grad_name

        _assert_logits_vjp_equivalence(
            packed_model=packed_model,
            flat_model=flat_model,
            tokens=tokens,
            input_pos=input_pos,
            group_ids=group_ids,
            parent_ids=parent_ids,
            assistant_mask=assistant_mask,
        )


def _assert_logits_vjp_equivalence(
    *,
    packed_model: torch.nn.Module,
    flat_model: torch.nn.Module,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> None:
    zero_parameter_grads(packed_model)
    zero_parameter_grads(flat_model)
    packed_logits = _run_model_logits(
        packed_model,
        tokens=tokens,
        input_pos=input_pos,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    shifted_assistant_mask = shift_tensor(assistant_mask, False)
    output_grad = torch.randn(
        packed_logits.shape,
        device=packed_logits.device,
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=torch.Generator(device=packed_logits.device).manual_seed(20280425),
    )
    output_grad = output_grad * shifted_assistant_mask.unsqueeze(-1) * 0.1
    loss_denominator = shifted_assistant_mask.unsqueeze(-1).expand_as(output_grad).sum()
    packed_loss = stable_output_mse_loss(
        packed_logits,
        output_grad,
        mask=shifted_assistant_mask.unsqueeze(-1),
        denominator=loss_denominator,
    )
    packed_loss.backward()

    flat_loss_sum: torch.Tensor | None = None
    logits_mean_abs_pct = 0.0
    spec = parse_gdn_prefix_tree_segments(group_ids.cpu(), parent_ids.cpu())
    for segment_index, completion in enumerate(spec.tree_segments):
        if spec.tree_parent_indices[segment_index] < 0:
            continue
        row = completion.row_index
        path = _segment_path(spec, segment_index)
        completion_offset = sum(segment.length for segment in path[:-1])
        ref_tokens = torch.cat(
            [tokens[row : row + 1, segment.start : segment.end] for segment in path],
            dim=1,
        )
        ref_pos = torch.cat(
            [input_pos[row : row + 1, segment.start : segment.end] for segment in path],
            dim=1,
        )
        ref_logits = _run_model_logits(
            flat_model,
            tokens=ref_tokens,
            input_pos=ref_pos,
            group_ids=torch.zeros_like(ref_tokens),
            parent_ids=torch.zeros_like(ref_tokens),
        )
        ref_output_grad = torch.zeros_like(ref_logits)
        ref_output_mask = torch.zeros(
            ref_logits.shape[:2],
            device=ref_logits.device,
            dtype=torch.bool,
        )
        if completion.length > 1:
            ref_output_grad[
                :, completion_offset : completion_offset + completion.length - 1
            ] = output_grad[row : row + 1, completion.start : completion.end - 1]
            ref_output_mask[
                :, completion_offset : completion_offset + completion.length - 1
            ] = True
        ref_loss = stable_output_mse_loss(
            ref_logits,
            ref_output_grad,
            mask=ref_output_mask.unsqueeze(-1),
            denominator=loss_denominator,
        )
        ref_loss.backward()
        flat_loss_sum = (
            ref_loss.detach()
            if flat_loss_sum is None
            else flat_loss_sum + ref_loss.detach()
        )
        if completion.length > 1:
            packed_slice = packed_logits[
                row : row + 1, completion.start : completion.end - 1
            ]
            ref_slice = ref_logits[
                :, completion_offset : completion_offset + completion.length - 1
            ]
            logits_mean_abs_pct = max(
                logits_mean_abs_pct,
                mean_abs_pct(ref_slice, packed_slice),
            )

    assert flat_loss_sum is not None
    grad_name, grad_pct = parameter_grad_mean_abs_pct_with_name(
        flat_model, packed_model
    )
    assert_mean_abs_pct(flat_loss_sum, packed_loss.detach(), "stable_loss")
    assert logits_mean_abs_pct <= MEAN_ABS_PCT_THRESHOLD
    assert grad_pct <= MEAN_ABS_PCT_THRESHOLD, grad_name


def _run_model_loss(
    model: torch.nn.Module,
    *,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _run_model_logits(
        model,
        tokens=tokens,
        input_pos=input_pos,
        group_ids=group_ids,
        parent_ids=parent_ids,
    )
    attention_state = create_prefix_tree_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
        build_gdn_execution_spec=True,
    )
    forward_kwargs = QWEN3_5_MOE_HANDLER.get_forward_kwargs(
        model,
        attention_bias=attention_state,
    )
    attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=tokens.device)
    shifted_labels = shift_tensor(tokens, -100)
    shifted_mask = shift_tensor(assistant_mask, False)
    shifted_labels = torch.where(
        shifted_mask,
        shifted_labels,
        torch.full_like(shifted_labels, -100),
    )
    per_token_loss = model(
        input_ids=tokens,
        position_ids=input_pos,
        attention_mask=attention_mask,
        labels=shifted_labels,
        **forward_kwargs,
    )
    return logits.detach(), per_token_loss[shifted_mask].sum()


def _run_model_logits(
    model: torch.nn.Module,
    *,
    tokens: torch.Tensor,
    input_pos: torch.Tensor,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
) -> torch.Tensor:
    attention_state = create_prefix_tree_state(
        group_ids=group_ids,
        parent_ids=parent_ids,
        build_gdn_execution_spec=True,
    )
    forward_kwargs = QWEN3_5_MOE_HANDLER.get_forward_kwargs(
        model,
        attention_bias=attention_state,
    )
    attention_mask = torch.zeros((1, 1, 1, 1), dtype=torch.bool, device=tokens.device)
    logits = model(
        input_ids=tokens,
        position_ids=input_pos,
        attention_mask=attention_mask,
        labels=None,
        **forward_kwargs,
    )
    return logits


def _segment_path(spec: Any, segment_index: int) -> tuple[Any, ...]:
    indices = []
    cursor = segment_index
    while cursor >= 0:
        indices.append(cursor)
        cursor = spec.tree_parent_indices[cursor]
    return tuple(spec.tree_segments[index] for index in reversed(indices))


def _make_matching_models() -> tuple[torch.nn.Module, torch.nn.Module]:
    model_parallel_cuda_manual_seed(1234)
    packed = _make_model()
    model_parallel_cuda_manual_seed(5678)
    flat = _make_model()
    flat.load_state_dict(packed.state_dict())
    attach_main_grads(packed)
    attach_main_grads(flat)
    return packed, flat


def _make_model() -> torch.nn.Module:
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
        moe_aux_loss_coeff=0.0,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=F.silu,
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
        mrope_section=[1, 1, 0],
        vision_config=Qwen3_5MoeVisionConfig(),
        tensor_model_parallel_size=1,
        expert_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        params_dtype=GDN_CORRECTNESS_DTYPE,
    )
    QWEN3_5_MOE_HANDLER.configure_provider_for_runtime(provider)
    QWEN3_5_MOE_HANDLER.patch_provider(provider, None)
    provider.finalize()
    model = provider.provide_language_model(pre_process=True, post_process=True).cuda()
    QWEN3_5_MOE_HANDLER.install_preprocess_patch([model])
    return model


@contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    if is_initialized():
        pytest.skip("torch.distributed is already initialized in this process.")
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix="art_dist_") as tmp:
        init_process_group(
            backend="nccl",
            init_method=file_init_method(Path(tmp), "qwen35_full_model_cp1"),
            rank=0,
            world_size=1,
        )
        try:
            ps.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
            )
            yield
        finally:
            if getattr(ps, "model_parallel_is_initialized", lambda: False)():
                ps.destroy_model_parallel()
            if is_initialized():
                destroy_process_group()
