from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    Qwen3_5MoeVisionConfig,
    Qwen35VLMoEModelProvider,
)
from megatron.core import parallel_state as ps
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from torch.distributed import (
    destroy_process_group,
    init_process_group,
    is_initialized,
)

from .cases import default_phase0_cases
from .distributed_init import file_init_method
from .metrics import (
    GDN_CORRECTNESS_DTYPE,
    MEAN_ABS_PCT_MISMATCH_THRESHOLD,
    assert_real_gdn_metrics,
    mean_abs_pct,
)
from .packed_layout import build_phase0_packed_tensors
from .real_gdn_oracle import (
    attach_main_grads,
    compare_real_gdn_cp1_to_flattened,
    compare_real_gdn_cp1_to_flattened_with_output_grad,
    run_real_gdn_flattened_reference,
    run_real_gdn_physical_stream,
    zero_parameter_grads,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real Megatron/FLA GDN oracle coverage.",
)
def test_real_qwen35_gdn_cp1_matches_flattened_and_rejects_physical() -> None:
    with _single_rank_model_parallel():
        packed_gdn, flat_gdn = _make_matching_qwen35_gdn_pair()
        device = torch.device("cuda")
        for case_index, case in enumerate(default_phase0_cases(conv_width=2)):
            zero_parameter_grads(packed_gdn)
            zero_parameter_grads(flat_gdn)
            tensors = build_phase0_packed_tensors(case)
            group_ids = tensors["group_ids"].to(device)
            parent_ids = tensors["parent_ids"].to(device)
            assistant_mask = tensors["assistant_mask"].to(device)
            hidden_states = torch.randn(
                case.sequence_length,
                len(case.rows),
                64,
                device=device,
                dtype=GDN_CORRECTNESS_DTYPE,
                generator=torch.Generator(device=device).manual_seed(
                    20260424 + case_index
                ),
            )

            metrics = compare_real_gdn_cp1_to_flattened(
                packed_gdn=packed_gdn,
                flat_gdn=flat_gdn,
                hidden_states=hidden_states,
                group_ids=group_ids,
                parent_ids=parent_ids,
                assistant_mask=assistant_mask,
            )

            assert_real_gdn_metrics(metrics, case.name)

            real_token_mask = (group_ids != -1).transpose(0, 1).unsqueeze(-1)
            output_grads = {
                "random_all_real_tokens": (
                    torch.randn(
                        hidden_states.shape,
                        device=device,
                        dtype=GDN_CORRECTNESS_DTYPE,
                        generator=torch.Generator(device=device).manual_seed(
                            20270424 + case_index
                        ),
                    )
                    * real_token_mask
                )
            }
            if case.name == "ragged_family_mix":
                output_grads.update(
                    {
                        "prefix_only": _expanded_output_mask(
                            group_ids == parent_ids, hidden_states.shape[-1]
                        ),
                        "suffix_only": _expanded_output_mask(
                            group_ids != parent_ids, hidden_states.shape[-1]
                        ),
                        "single_token_channel": _single_token_channel_grad(
                            hidden_states, group_ids != -1
                        ),
                    }
                )
            for name, output_grad in output_grads.items():
                zero_parameter_grads(packed_gdn)
                zero_parameter_grads(flat_gdn)
                upstream_metrics = compare_real_gdn_cp1_to_flattened_with_output_grad(
                    packed_gdn=packed_gdn,
                    flat_gdn=flat_gdn,
                    hidden_states=hidden_states,
                    group_ids=group_ids,
                    parent_ids=parent_ids,
                    output_grad=output_grad,
                )

                assert_real_gdn_metrics(upstream_metrics, f"{case.name}:{name}")

            if case.name == "ragged_family_mix":
                with torch.no_grad():
                    flattened = run_real_gdn_flattened_reference(
                        flat_gdn,
                        hidden_states,
                        group_ids=group_ids,
                        parent_ids=parent_ids,
                    )
                    physical = run_real_gdn_physical_stream(
                        flat_gdn,
                        hidden_states,
                        group_ids=group_ids,
                    )
                assert (
                    mean_abs_pct(
                        flattened.transpose(0, 1)[assistant_mask],
                        physical.transpose(0, 1)[assistant_mask],
                    )
                    > MEAN_ABS_PCT_MISMATCH_THRESHOLD
                ), case.name


def _make_matching_qwen35_gdn_pair(
    *, params_dtype: torch.dtype = GDN_CORRECTNESS_DTYPE
) -> tuple[GatedDeltaNet, GatedDeltaNet]:
    model_parallel_cuda_manual_seed(1234)
    packed_model = _make_qwen35_language_model(params_dtype=params_dtype)
    model_parallel_cuda_manual_seed(5678)
    flat_model = _make_qwen35_language_model(params_dtype=params_dtype)
    packed_gdn = _first_gdn(packed_model)
    flat_gdn = _first_gdn(flat_model)
    flat_gdn.load_state_dict(packed_gdn.state_dict())
    attach_main_grads(packed_gdn)
    attach_main_grads(flat_gdn)
    return packed_gdn, flat_gdn


def _make_qwen35_language_model(
    *, params_dtype: torch.dtype = GDN_CORRECTNESS_DTYPE
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
        context_parallel_size=1,
        params_dtype=params_dtype,
    )
    provider.finalize()
    return provider.provide_language_model(pre_process=True, post_process=True).cuda()


def _first_gdn(model: torch.nn.Module) -> GatedDeltaNet:
    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            return module
    raise AssertionError("expected Qwen3.5 provider to build at least one GDN layer")


def _expanded_output_mask(mask: torch.Tensor, hidden_size: int) -> torch.Tensor:
    return (
        mask.transpose(0, 1)
        .unsqueeze(-1)
        .expand(mask.shape[1], mask.shape[0], hidden_size)
        .to(dtype=GDN_CORRECTNESS_DTYPE)
    )


def _single_token_channel_grad(
    hidden_states: torch.Tensor, real_mask: torch.Tensor
) -> torch.Tensor:
    row, position = real_mask.nonzero()[real_mask.sum() // 2].tolist()
    output_grad = torch.zeros_like(hidden_states)
    output_grad[position, row, 0] = 1.0
    return output_grad


@contextmanager
def _single_rank_model_parallel() -> Iterator[None]:
    if is_initialized():
        pytest.skip("torch.distributed is already initialized in this process.")
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory(prefix="art_dist_") as tmp:
        init_process_group(
            backend="nccl",
            init_method=file_init_method(Path(tmp), "single_rank"),
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
