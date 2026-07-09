from __future__ import annotations

import pytest

from art.megatron.gdn.gdn_prefix_tree import GdnPlannerConfig


def test_gdn_planner_runtime_model_preserves_qwen35_reference_shape() -> None:
    reference = GdnPlannerConfig.from_model_shape(
        hidden_size=2048,
        tensor_model_parallel_size=1,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        dtype_bytes=2,
    )
    default = GdnPlannerConfig()
    assert (
        reference.runtime_hidden_bytes_per_token
        == default.runtime_hidden_bytes_per_token
    )
    assert (
        reference.runtime_cp_summary_bytes_per_segment
        == default.runtime_cp_summary_bytes_per_segment
    )
    assert reference.runtime_local_recurrent_tokens_per_ms == pytest.approx(
        default.runtime_local_recurrent_tokens_per_ms
    )
    assert reference.runtime_chain_recurrent_tokens_per_ms == pytest.approx(
        default.runtime_chain_recurrent_tokens_per_ms
    )
    assert (
        reference.runtime_parent_state_bytes_per_exchange
        / reference.runtime_parent_state_bandwidth_bytes_per_ms
    ) == pytest.approx(
        default.runtime_parent_state_bytes_per_exchange
        / default.runtime_parent_state_bandwidth_bytes_per_ms
    )


def test_gdn_planner_runtime_model_scales_with_state_shape() -> None:
    reference = GdnPlannerConfig.from_model_shape(
        hidden_size=2048,
        tensor_model_parallel_size=1,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    small = GdnPlannerConfig.from_model_shape(
        hidden_size=1024,
        tensor_model_parallel_size=1,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
    )
    large = GdnPlannerConfig.from_model_shape(
        hidden_size=7168,
        tensor_model_parallel_size=1,
        linear_num_key_heads=56,
        linear_num_value_heads=112,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    assert (
        small.runtime_hidden_bytes_per_token < reference.runtime_hidden_bytes_per_token
    )
    assert small.runtime_cp_summary_bytes_per_segment < (
        reference.runtime_cp_summary_bytes_per_segment
    )
    assert small.runtime_local_recurrent_tokens_per_ms > (
        reference.runtime_local_recurrent_tokens_per_ms
    )
    assert (
        large.runtime_hidden_bytes_per_token > reference.runtime_hidden_bytes_per_token
    )
    assert large.runtime_cp_summary_bytes_per_segment > (
        reference.runtime_cp_summary_bytes_per_segment
    )
    assert large.runtime_local_recurrent_tokens_per_ms < (
        reference.runtime_local_recurrent_tokens_per_ms
    )


def test_gdn_planner_runtime_model_tracks_qwen35_shape_axes() -> None:
    qwen35_35b = GdnPlannerConfig.from_model_shape(
        hidden_size=2048,
        tensor_model_parallel_size=1,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    qwen35_9b = GdnPlannerConfig.from_model_shape(
        hidden_size=4096,
        tensor_model_parallel_size=1,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    qwen35_397b = GdnPlannerConfig.from_model_shape(
        hidden_size=4096,
        tensor_model_parallel_size=1,
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )

    assert qwen35_9b.runtime_hidden_bytes_per_token == 2 * (
        qwen35_35b.runtime_hidden_bytes_per_token
    )
    assert qwen35_9b.runtime_local_recurrent_tokens_per_ms == pytest.approx(
        qwen35_35b.runtime_local_recurrent_tokens_per_ms
    )
    assert qwen35_9b.runtime_cp_summary_bytes_per_segment == (
        qwen35_35b.runtime_cp_summary_bytes_per_segment
    )

    assert qwen35_397b.runtime_hidden_bytes_per_token == 2 * (
        qwen35_35b.runtime_hidden_bytes_per_token
    )
    assert qwen35_397b.runtime_cp_summary_bytes_per_segment == 2 * (
        qwen35_35b.runtime_cp_summary_bytes_per_segment
    )
    assert qwen35_397b.runtime_local_recurrent_tokens_per_ms == pytest.approx(
        (2**-0.75) * qwen35_35b.runtime_local_recurrent_tokens_per_ms
    )
    assert qwen35_397b.runtime_chain_recurrent_tokens_per_ms == pytest.approx(
        (2**-0.75) * qwen35_35b.runtime_chain_recurrent_tokens_per_ms
    )
