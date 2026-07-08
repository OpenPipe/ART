from __future__ import annotations

from contextlib import ExitStack
import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from art.megatron.flex_attn.attention import FlexAttentionWrapper
from art.megatron.prefix_tree_state import create_prefix_tree_state
from tests.integration.megatron.gdn_shared_prefix.cases import default_phase0_cases
from tests.integration.megatron.gdn_shared_prefix.metrics import (
    GDN_CORRECTNESS_DTYPE,
    MEAN_ABS_PCT_MISMATCH_THRESHOLD,
    MEAN_ABS_PCT_THRESHOLD,
    assert_mean_abs_pct,
    mean_abs_pct,
)
from tests.integration.megatron.gdn_shared_prefix.packed_layout import (
    build_phase0_packed_tensors,
)
from tests.integration.megatron.gdn_shared_prefix.parser_import import (
    parse_gdn_prefix_tree_segments,
)
from tests.integration.megatron.model_support.oracle_harness import (
    TEST_DEFAULT_FLEX_BACKEND,
)
from tests.integration.megatron.model_support.oracle_worker import (
    _apply_requested_flex_backend_patch,
    _apply_test_attention_full_fp32_patch,
    _apply_test_flex_inner_fp32_patch,
)


@pytest.fixture(autouse=True)
def _fp32_test_flex_backend():
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
    reason="CUDA is required for compiled flex-attention prefix-tree coverage.",
)
def test_prefix_tree_attention_matches_flattened_grad_accumulation() -> None:
    case = next(
        item for item in default_phase0_cases() if item.name == "multi_family_repeated"
    )
    tensors = build_phase0_packed_tensors(case)
    group_ids = tensors["group_ids"].cuda()
    parent_ids = tensors["parent_ids"].cuda()
    spec = parse_gdn_prefix_tree_segments(group_ids.cpu(), parent_ids.cpu())
    q, k, v = _attention_inputs(group_ids.shape, seed=20260425)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    output_grad = _packed_output_grad(spec, q.shape, seed=20260426)

    attention_state = create_prefix_tree_state(group_ids, parent_ids)
    packed_out = FlexAttentionWrapper()(
        q,
        k,
        v,
        block_mask=attention_state.block_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=False,
    )
    (packed_out * output_grad).sum().backward()

    ref_out = torch.zeros_like(packed_out)
    ref_loss = q_ref.new_zeros(())
    for segment_index, segment in enumerate(spec.tree_segments):
        indices, output_slice = _segment_context_positions(spec, segment_index)
        index_tensor = torch.tensor(indices, device=q.device, dtype=torch.long)
        row = segment.row_index
        q_slice = q_ref[row : row + 1].index_select(2, index_tensor)
        k_slice = k_ref[row : row + 1].index_select(2, index_tensor)
        v_slice = v_ref[row : row + 1].index_select(2, index_tensor)
        flat_out = _dense_causal_attention(q_slice, k_slice, v_slice)

        ref_out[row, :, segment.start : segment.end] = flat_out[0, :, output_slice]
        flat_grad = torch.zeros_like(flat_out)
        flat_grad[0, :, output_slice] = output_grad[row, :, segment.start : segment.end]
        ref_loss = ref_loss + (flat_out * flat_grad).sum()
    ref_loss.backward()

    real_mask = _real_token_mask(spec, q.shape, device=q.device)
    assert_mean_abs_pct(ref_out[real_mask], packed_out[real_mask], "attention_output")
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q_ref.grad is not None
    assert k_ref.grad is not None
    assert v_ref.grad is not None
    assert_mean_abs_pct(q_ref.grad, q.grad, "q_grad")
    assert_mean_abs_pct(k_ref.grad, k.grad, "k_grad")
    assert_mean_abs_pct(v_ref.grad, v.grad, "v_grad")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for compiled flex-attention prefix-tree coverage.",
)
def test_physical_causal_attention_leaks_across_siblings() -> None:
    case = next(
        item for item in default_phase0_cases() if item.name == "multi_family_repeated"
    )
    tensors = build_phase0_packed_tensors(case)
    group_ids = tensors["group_ids"].cuda()
    parent_ids = tensors["parent_ids"].cuda()
    spec = parse_gdn_prefix_tree_segments(group_ids.cpu(), parent_ids.cpu())
    q, k, v = _attention_inputs(group_ids.shape, seed=20260427)
    attention_state = create_prefix_tree_state(group_ids, parent_ids)
    packed_out = FlexAttentionWrapper()(
        q,
        k,
        v,
        block_mask=attention_state.block_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        enable_gqa=False,
    )
    physical_out = _dense_causal_attention(q, k, v)
    completion_mask = _completion_token_mask(spec, q.shape, device=q.device)
    assert (
        mean_abs_pct(
            packed_out[completion_mask],
            physical_out[completion_mask],
        )
        > MEAN_ABS_PCT_MISMATCH_THRESHOLD
    )


def _attention_inputs(
    shape: torch.Size, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, sequence_length = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch_size,
        2,
        sequence_length,
        16,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
        requires_grad=True,
    )
    k = torch.randn(
        q.shape, device="cuda", dtype=GDN_CORRECTNESS_DTYPE, generator=generator
    )
    v = torch.randn(
        q.shape, device="cuda", dtype=GDN_CORRECTNESS_DTYPE, generator=generator
    )
    return q, k.requires_grad_(True), v.requires_grad_(True)


def _dense_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    length = int(q.shape[-2])
    causal_mask = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _packed_output_grad(spec: Any, shape: torch.Size, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    grad = torch.randn(
        shape,
        device="cuda",
        dtype=GDN_CORRECTNESS_DTYPE,
        generator=generator,
    )
    return grad * _real_token_mask(spec, shape, device=grad.device) * 0.1


def _real_token_mask(
    spec: Any, shape: torch.Size, *, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    for row_index, valid_length in enumerate(spec.valid_lengths):
        mask[row_index, :, :valid_length] = True
    return mask


def _completion_token_mask(
    spec: Any, shape: torch.Size, *, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    for index, segment in enumerate(spec.tree_segments):
        if spec.tree_parent_indices[index] >= 0:
            mask[segment.row_index, :, segment.start : segment.end] = True
    return mask


def _segment_context_positions(
    spec: Any, segment_index: int
) -> tuple[list[int], slice]:
    path = []
    cursor = segment_index
    while cursor >= 0:
        path.append(cursor)
        cursor = spec.tree_parent_indices[cursor]
    path.reverse()
    positions = [
        position
        for index in path
        for position in range(
            spec.tree_segments[index].start, spec.tree_segments[index].end
        )
    ]
    segment_length = spec.tree_segments[segment_index].length
    return positions, slice(len(positions) - segment_length, len(positions))
