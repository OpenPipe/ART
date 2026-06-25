from typing import Any

import pytest
import torch

from .forward_trace import ForwardTraceCapture, _extract_router_topk
from .oracle_harness import (
    CP_ATTENTION_SENSITIVITY_MUTATIONS,
    DENSE_CP_ATTENTION_SENSITIVITY_TOPOLOGY,
    DENSE_DP_SENSITIVITY_TOPOLOGY,
    DENSE_ORACLE_TOPOLOGY,
    DENSE_TOPOLOGIES,
    FORWARD_EXPERT_LORA_TRACE_NOISE_REASON,
    FORWARD_EXPERT_LORA_TRACE_NOISE_RELATIVE_L2_LIMIT,
    ORACLE_DEFAULT_MEAN_ABS_PCT_LIMIT,
    ORACLE_TOPOLOGY,
    ROUTER_SCORE_MEAN_ABS_PCT_LIMIT,
    TEST_DEFAULT_FLEX_BACKEND,
    TOPOLOGIES,
    DiffAccumulator,
    MetricRow,
    MetricThresholdRule,
    PackedTensorConfig,
    Topology,
    VariantRunner,
    _default_phase_pass_fns,
    _resolve_test_flex_backend,
    _suite_variants,
    case_config,
    selected_sensitivity_mutations_for_objective,
    sensitivity_topology_for_mutation,
)


def _metric_row(
    *,
    phase: str,
    param: str,
    pass_signal: bool,
    step_index: int = 0,
    mean_abs_pct: float = 0.0,
    relative_l2: float = 0.0,
    topk_mismatch_fraction: float | None = None,
    top1_mismatch_fraction: float | None = None,
) -> MetricRow:
    return MetricRow(
        case_id="case",
        variant="variant",
        topology="candidate",
        oracle_topology="oracle",
        step_index=step_index,
        phase=phase,
        param=param,
        numel=1.0,
        mean_abs_diff=0.0,
        relative_l2=relative_l2,
        typical_abs_scale=1.0,
        mean_abs_pct=mean_abs_pct,
        topk_mismatch_fraction=topk_mismatch_fraction,
        top1_mismatch_fraction=top1_mismatch_fraction,
        pass_signal=pass_signal,
        failure_reasons=[] if pass_signal else ["mean_abs_pct=2>1"],
    )


def _expert_trace_call(
    *,
    ep_rank: int,
    etp_rank: int,
    values: torch.Tensor,
    uids: torch.Tensor,
    hint: dict[str, object],
) -> dict[str, object]:
    return {
        "micro_call_index": 0,
        "micro_order": 0,
        "micro_sample_index": 0,
        "module_type": "SyntheticExpert",
        "primary_output": values,
        "row_token_uids": uids,
        "merge_hints": {"primary_output": hint},
        "rank_meta": {
            "global_rank": ep_rank * 2 + etp_rank,
            "world_size": 4,
            "tp_rank": etp_rank,
            "tp_world_size": 2,
            "cp_rank": 0,
            "cp_world_size": 1,
            "ep_rank": ep_rank,
            "ep_world_size": 2,
            "etp_rank": etp_rank,
            "etp_world_size": 2,
            "dp_rank": 0,
            "dp_world_size": 1,
            "expert_dp_rank": 0,
            "expert_dp_world_size": 1,
        },
    }


def test_metric_threshold_rule_can_require_strictly_positive_values() -> None:
    rule = MetricThresholdRule(minimums={"candidate_abs_scale": 0.0})

    summary = {"candidate_abs_scale": 0.0}

    assert not rule(summary)
    assert rule.failure_reasons(summary) == ["candidate_abs_scale=0<=0"]


def test_diff_accumulator_summary_uses_aggregate_mean_abs_pct() -> None:
    accumulator = DiffAccumulator()

    accumulator.update(
        torch.tensor([1.0, -2.0], dtype=torch.float32),
        torch.tensor([0.5, 0.0], dtype=torch.float32),
    )

    summary = accumulator.as_summary()

    assert summary["typical_abs_scale"] == 1.5
    assert summary["candidate_abs_scale"] == 0.25
    assert summary["mean_abs_diff"] == 1.25
    assert summary["mean_abs_pct"] == pytest.approx((1.25 / 1.5) * 100.0)


def test_context_parallel_accumulator_dtype_matches_dense_fp32_oracle() -> None:
    from art.megatron.context_parallel.executor import _accum_output_dtype

    assert _accum_output_dtype(torch.float32) is torch.float32
    assert _accum_output_dtype(torch.bfloat16) is torch.float32
    assert _accum_output_dtype(torch.float16) is torch.float32


def test_context_parallel_seeded_accumulator_can_own_stage_storage() -> None:
    from art.megatron.context_parallel.executor import _seed_stage_accumulators

    stage_out = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    stage_lse = torch.tensor([3.0], dtype=torch.float32)

    accum_out, accum_lse = _seed_stage_accumulators(
        stage_out=stage_out,
        stage_lse=stage_lse,
        target_dtype=torch.float32,
        needs_owned_storage=True,
    )
    accum_out.add_(1.0)
    accum_lse.add_(1.0)

    assert stage_out.tolist() == [[1.0, 2.0]]
    assert stage_lse.tolist() == [3.0]


def test_fp32_oracle_defaults_to_test_triton_backend() -> None:
    config = case_config().model_copy(update={"precision": "fp32"})

    assert _resolve_test_flex_backend(config, None) == TEST_DEFAULT_FLEX_BACKEND
    assert _resolve_test_flex_backend(config, "FLASH") == "FLASH"


def test_bf16_oracle_preserves_production_flex_default() -> None:
    config = case_config().model_copy(update={"precision": "bf16"})

    assert _resolve_test_flex_backend(config, None) is None


def test_production_compiled_flex_default_stays_flash() -> None:
    from art.megatron.flex_attn import compiled as compiled_flex_attention

    assert compiled_flex_attention._FORCED_FLEX_BACKEND == "FLASH"
    assert compiled_flex_attention._FORCED_FLEX_KERNEL_OPTIONS == {"BACKEND": "FLASH"}


def test_forward_trace_reads_row_uids_from_output_tensor() -> None:
    output = torch.zeros((2, 1), dtype=torch.float32)
    setattr(output, "_art_trace_row_token_uids", torch.tensor([4, 7]))

    row_uids, uid_span = ForwardTraceCapture._row_token_uids_for_trace(
        inputs=(),
        output=output,
        module=object(),
    )

    assert uid_span is None
    assert row_uids is not None
    assert torch.equal(row_uids, torch.tensor([4, 7]))


def test_forward_trace_prefers_local_tensor_uids_over_module_fallback() -> None:
    module = type("ModuleWithGenericTraceUids", (), {})()
    inputs = torch.zeros((2, 1), dtype=torch.float32)
    setattr(module, "_art_trace_row_token_uids", torch.tensor([10, 11]))
    setattr(inputs, "_art_trace_row_token_uids", torch.tensor([4, 7]))

    row_uids, _uid_span = ForwardTraceCapture._row_token_uids_for_trace(
        inputs=(inputs,),
        module=module,
    )

    assert row_uids is not None
    assert torch.equal(row_uids, torch.tensor([4, 7]))


def test_forward_trace_extracts_empty_router_topk_with_config_hint() -> None:
    topk = _extract_router_topk(
        (
            torch.empty((0, 8), dtype=torch.float32),
            torch.empty((0, 8), dtype=torch.bool),
        ),
        topk_hint=2,
    )
    assert topk is not None
    ids, scores = topk

    assert ids.shape == (0, 2)
    assert scores.shape == (0, 2)


def test_megatron_empty_swiglu_patch_preserves_known_output_width() -> None:
    from art.megatron.runtime.bridge_runtime import install_art_bridge_runtime_patches

    install_art_bridge_runtime_patches()
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl

    x = torch.empty((0, 1, 8), dtype=torch.float32, requires_grad=True)
    bias = torch.randn((8,), dtype=torch.float32, requires_grad=True)
    y = bias_swiglu_impl(x, bias)

    assert y.shape == (0, 1, 4)
    y.add_(torch.empty_like(y))
    y.sum().backward()
    assert x.grad is not None
    assert bias.grad is not None
    assert torch.equal(bias.grad, torch.zeros_like(bias.grad))


def test_megatron_empty_unpermute_patch_allows_view_then_inplace_add() -> None:
    from art.megatron.runtime.bridge_runtime import install_art_bridge_runtime_patches

    install_art_bridge_runtime_patches()
    from megatron.core.transformer.moe.token_dispatcher import unpermute

    tokens = torch.empty((0, 8), dtype=torch.float32, requires_grad=True)
    sorted_indices = torch.empty((0,), dtype=torch.long)
    output = unpermute(
        tokens,
        sorted_indices,
        torch.Size((0, 8)),
        fused=True,
    )
    output = output.view(0, 1, 8)
    output.add_(torch.empty_like(output))
    output.sum().backward()

    assert tokens.grad is not None


def test_forward_trace_splits_expert_rows_with_input_uid_span() -> None:
    module_name = "chunk0.module.decoder.layers.0.mlp.experts.linear_fc1"
    module = type("SyntheticExpertModule", (), {})()
    inputs = torch.zeros((4, 1), dtype=torch.float32)
    setattr(inputs, "_art_trace_row_token_uids", torch.tensor([0, 1, 10, 11]))
    setattr(inputs, "_art_trace_uid_span", 10)
    trace_item = {
        "micro_sample_index": None,
        "primary_output": torch.tensor([[0.0], [1.0], [2.0], [3.0]]),
    }

    split_items = ForwardTraceCapture._split_expert_trace_items(
        module_name=module_name,
        module=module,
        inputs=(inputs,),
        trace_item=trace_item,
    )

    assert [item["micro_sample_index"] for item in split_items] == [0, 1]
    assert torch.equal(split_items[0]["row_token_uids"], torch.tensor([0, 1]))
    assert torch.equal(split_items[1]["row_token_uids"], torch.tensor([10, 11]))
    assert torch.equal(split_items[0]["primary_output"], torch.tensor([[0.0], [1.0]]))
    assert torch.equal(split_items[1]["primary_output"], torch.tensor([[2.0], [3.0]]))
    assert split_items[0]["row_uid_span"] == 10
    assert split_items[1]["row_uid_span"] == 10


def test_forward_trace_canonicalizes_row_outputs_by_token_uid() -> None:
    trace: dict[str, list[dict[str, Any]]] = {
        "chunk0.module.decoder.layers.0": [
            {
                "primary_output": torch.tensor([[30.0], [10.0], [20.0]]),
                "router_topk_scores": torch.tensor([[0.3], [0.1], [0.2]]),
                "router_topk_ids": torch.tensor([[3], [1], [2]]),
                "output": {
                    "probs": torch.tensor([[3.0], [1.0], [2.0]]),
                    "routing_map": torch.tensor([[True], [False], [True]]),
                },
                "row_token_uids": torch.tensor([3, 1, 2]),
            }
        ]
    }

    ForwardTraceCapture.canonicalize_trace(trace)

    call = trace["chunk0.module.decoder.layers.0"][0]
    assert torch.equal(call["row_token_uids"], torch.tensor([1, 2, 3]))
    assert torch.equal(
        call["primary_output"],
        torch.tensor([[10.0], [20.0], [30.0]]),
    )
    assert torch.equal(call["router_topk_scores"], torch.tensor([[0.1], [0.2], [0.3]]))
    assert torch.equal(call["router_topk_ids"], torch.tensor([[1], [2], [3]]))
    assert torch.equal(call["output"]["probs"], torch.tensor([[1.0], [2.0], [3.0]]))
    assert torch.equal(
        call["output"]["routing_map"],
        torch.tensor([[False], [True], [True]]),
    )


def test_forward_trace_drops_exact_zero_padding_rows() -> None:
    trace: dict[str, list[dict[str, Any]]] = {
        "chunk0.module.decoder.layers.0.self_attention.out_proj": [
            {
                "primary_output": torch.tensor(
                    [[0.0, 0.0], [30.0, 31.0], [10.0, 11.0], [20.0, 21.0]]
                ),
                "output": {
                    "hidden": torch.tensor(
                        [[0.0, 0.0], [3.0, 3.1], [1.0, 1.1], [2.0, 2.1]]
                    )
                },
                "row_token_uids": torch.tensor([-1, 3, 1, 2]),
            }
        ]
    }

    ForwardTraceCapture.canonicalize_trace(trace)

    call = trace["chunk0.module.decoder.layers.0.self_attention.out_proj"][0]
    assert torch.equal(call["row_token_uids"], torch.tensor([1, 2, 3]))
    assert torch.equal(
        call["primary_output"],
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]),
    )
    assert torch.equal(
        call["output"]["hidden"],
        torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]),
    )


def test_forward_trace_expands_attention_output_uids_for_out_norm_heads() -> None:
    trace: dict[str, list[dict[str, Any]]] = {
        "chunk0.module.decoder.layers.0.self_attention": [
            {
                "micro_order": 0,
                "micro_sample_index": 0,
                "primary_output": torch.zeros((3, 1, 8)),
                "row_token_uids": torch.tensor([0, -1, 2]),
            }
        ],
        "chunk0.module.decoder.layers.0.self_attention.out_norm": [
            {
                "micro_order": 0,
                "micro_sample_index": 0,
                "primary_output": torch.arange(24, dtype=torch.float32).reshape(6, 4),
                "merge_hints": {
                    "primary_output": {
                        "op": "concat",
                        "dim": 0,
                        "layout": "rank_blocked_token_heads",
                        "local_heads": 2,
                        "world_size_key": "tp_world_size",
                    }
                },
                "rank_meta": {"tp_world_size": 1},
            }
        ],
    }

    ForwardTraceCapture.canonicalize_trace(trace)

    call = trace["chunk0.module.decoder.layers.0.self_attention.out_norm"][0]
    assert torch.equal(call["row_token_uids"], torch.tensor([-1, -1, 0, 0, 2, 2]))
    assert torch.equal(
        call["primary_output"],
        torch.tensor(
            [
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ]
        ),
    )


def test_forward_trace_merges_expert_tp_feature_shards_inside_ep_groups() -> None:
    module_name = "chunk0.module.decoder.layers.0.mlp.experts.linear_fc1.gate_lora"
    rank_traces = [
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=0,
                    etp_rank=0,
                    values=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                    uids=torch.tensor([10, 20]),
                    hint={"op": "concat", "dim": -1},
                )
            ]
        },
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=0,
                    etp_rank=1,
                    values=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                    uids=torch.tensor([10, 20]),
                    hint={"op": "concat", "dim": -1},
                )
            ]
        },
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=1,
                    etp_rank=0,
                    values=torch.tensor([[9.0, 10.0]]),
                    uids=torch.tensor([30]),
                    hint={"op": "concat", "dim": -1},
                )
            ]
        },
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=1,
                    etp_rank=1,
                    values=torch.tensor([[11.0, 12.0]]),
                    uids=torch.tensor([30]),
                    hint={"op": "concat", "dim": -1},
                )
            ]
        },
    ]

    merged = ForwardTraceCapture.canonicalize_trace(
        ForwardTraceCapture._merge_rank_traces(rank_traces)
    )
    call = merged[module_name][0]

    assert torch.equal(call["row_token_uids"], torch.tensor([10, 20, 30]))
    assert torch.equal(
        call["primary_output"],
        torch.tensor(
            [
                [1.0, 2.0, 5.0, 6.0],
                [3.0, 4.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        ),
    )


def test_forward_trace_sums_expert_tp_row_shards_inside_ep_groups() -> None:
    module_name = "chunk0.module.decoder.layers.0.mlp.experts.linear_fc2"
    rank_traces = [
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=0,
                    etp_rank=0,
                    values=torch.tensor([[1.0, 2.0]]),
                    uids=torch.tensor([10]),
                    hint={"op": "sum"},
                )
            ]
        },
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=0,
                    etp_rank=1,
                    values=torch.tensor([[10.0, 20.0]]),
                    uids=torch.tensor([10]),
                    hint={"op": "sum"},
                )
            ]
        },
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=1,
                    etp_rank=0,
                    values=torch.tensor([[3.0, 4.0]]),
                    uids=torch.tensor([20]),
                    hint={"op": "sum"},
                )
            ]
        },
        {
            module_name: [
                _expert_trace_call(
                    ep_rank=1,
                    etp_rank=1,
                    values=torch.tensor([[30.0, 40.0]]),
                    uids=torch.tensor([20]),
                    hint={"op": "sum"},
                )
            ]
        },
    ]

    merged = ForwardTraceCapture.canonicalize_trace(
        ForwardTraceCapture._merge_rank_traces(rank_traces)
    )
    call = merged[module_name][0]

    assert torch.equal(call["row_token_uids"], torch.tensor([10, 20]))
    assert torch.equal(
        call["primary_output"],
        torch.tensor([[11.0, 22.0], [33.0, 44.0]]),
    )


def test_gate_up_rank_interleaved_trace_layout_canonicalizes_dense_tp() -> None:
    canonical = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    gate0, gate1, up0, up1 = canonical.chunk(4, dim=-1)
    rank_concat = torch.cat((gate0, up0, gate1, up1), dim=-1)

    actual = ForwardTraceCapture._canonicalize_primary_output_tensor(
        module_name="chunk0.module.decoder.layers.0.mlp.linear_fc1",
        tensor=rank_concat,
        call={
            "merge_hints": {
                "primary_output": {
                    "layout": "gate_up_rank_interleaved",
                    "world_size_key": "tp_world_size",
                }
            },
            "rank_meta": [{"tp_world_size": 2}, {"tp_world_size": 2}],
        },
    )

    assert torch.equal(actual, canonical)


def test_forward_trace_canonicalizes_cp_tp_rank_blocked_heads_with_row_uids() -> None:
    module_name = "chunk0.module.decoder.layers.0.self_attention.out_norm"
    rank_traces = []
    rank_specs = [
        (0, 0, [10, 10, 20, 20], [[100.0], [101.0], [200.0], [201.0]]),
        (0, 1, [10, 10, 20, 20], [[110.0], [111.0], [210.0], [211.0]]),
        (1, 0, [5, 5], [[50.0], [51.0]]),
        (1, 1, [5, 5], [[60.0], [61.0]]),
    ]
    for global_rank, (cp_rank, tp_rank, uids, values) in enumerate(rank_specs):
        rank_traces.append(
            {
                module_name: [
                    {
                        "micro_call_index": 0,
                        "micro_order": 0,
                        "micro_sample_index": 0,
                        "module_type": "RMSNorm",
                        "primary_output": torch.tensor(values),
                        "row_token_uids": torch.tensor(uids),
                        "merge_hints": {
                            "primary_output": {
                                "op": "concat",
                                "dim": 0,
                                "layout": "rank_blocked_token_heads",
                                "local_heads": 2,
                                "world_size_key": "tp_world_size",
                            }
                        },
                        "rank_meta": {
                            "global_rank": global_rank,
                            "world_size": 4,
                            "tp_rank": tp_rank,
                            "tp_world_size": 2,
                            "cp_rank": cp_rank,
                            "cp_world_size": 2,
                        },
                    }
                ]
            }
        )

    merged = ForwardTraceCapture.canonicalize_trace(
        ForwardTraceCapture._merge_rank_traces(rank_traces)
    )
    call = merged[module_name][0]

    assert torch.equal(
        call["row_token_uids"],
        torch.tensor([5, 5, 5, 5, 10, 10, 10, 10, 20, 20, 20, 20]),
    )
    assert torch.equal(
        call["primary_output"].flatten(),
        torch.tensor(
            [
                50.0,
                51.0,
                60.0,
                61.0,
                100.0,
                101.0,
                110.0,
                111.0,
                200.0,
                201.0,
                210.0,
                211.0,
            ]
        ),
    )


def test_default_phase_rules_require_non_zero_forward_outputs_grads_and_deltas() -> (
    None
):
    phase_pass = _default_phase_pass_fns()
    zero_signal_summary = {
        "relative_l2": 0.0,
        "mean_abs_pct": 0.0,
        "typical_abs_scale": 0.0,
        "candidate_abs_scale": 0.0,
    }

    assert not phase_pass["forward"](zero_signal_summary)
    assert not phase_pass["outputs"](zero_signal_summary)
    assert not phase_pass["grads"](zero_signal_summary)
    assert not phase_pass["deltas"](zero_signal_summary)
    assert phase_pass["losses"](zero_signal_summary)


def test_default_phase_rules_use_default_mean_abs_pct_limit() -> None:
    phase_pass = _default_phase_pass_fns()
    passing_summary = {
        "relative_l2": 0.0,
        "mean_abs_pct": ORACLE_DEFAULT_MEAN_ABS_PCT_LIMIT,
        "typical_abs_scale": 1.0,
        "candidate_abs_scale": 1.0,
    }
    failing_summary = {
        **passing_summary,
        "mean_abs_pct": ORACLE_DEFAULT_MEAN_ABS_PCT_LIMIT + 1e-6,
    }

    assert phase_pass["forward"](passing_summary)
    assert phase_pass["outputs"](passing_summary)
    assert phase_pass["grads"](passing_summary)
    assert phase_pass["deltas"](passing_summary)
    assert phase_pass["losses"](passing_summary)
    assert not phase_pass["forward"](failing_summary)
    assert not phase_pass["outputs"](failing_summary)
    assert not phase_pass["grads"](failing_summary)
    assert not phase_pass["deltas"](failing_summary)
    assert not phase_pass["losses"](failing_summary)


def test_router_score_rule_uses_tight_dedicated_limit() -> None:
    phase_pass = _default_phase_pass_fns()
    assert phase_pass["router_scores"](
        {"relative_l2": 1.0, "mean_abs_pct": ROUTER_SCORE_MEAN_ABS_PCT_LIMIT}
    )
    assert not phase_pass["router_scores"](
        {"relative_l2": 0.0, "mean_abs_pct": ROUTER_SCORE_MEAN_ABS_PCT_LIMIT + 1e-8}
    )


def test_forward_expert_lora_noise_pass_requires_clean_step_gates() -> None:
    noisy_row = _metric_row(
        phase="forward",
        param="chunk0.module.decoder.layers.__layer_avg__.mlp.experts.linear_fc2.lora.call_3",
        pass_signal=False,
        mean_abs_pct=2.0,
        relative_l2=FORWARD_EXPERT_LORA_TRACE_NOISE_RELATIVE_L2_LIMIT,
    )
    rows = [
        noisy_row,
        _metric_row(phase="outputs", param="logprobs.micro_000", pass_signal=True),
        _metric_row(
            phase="router_scores",
            param="chunk0.module.decoder.layers.__layer_avg__.mlp.router.call_3",
            pass_signal=True,
            mean_abs_pct=0.0,
            relative_l2=0.0,
        ),
        _metric_row(
            phase="router_topk_ids",
            param="chunk0.module.decoder.layers.__layer_avg__.mlp.router.call_3",
            pass_signal=True,
            topk_mismatch_fraction=0.0,
            top1_mismatch_fraction=0.0,
        ),
    ]

    VariantRunner._apply_forward_expert_lora_trace_noise_passes(rows)

    assert noisy_row.pass_signal
    assert noisy_row.failure_reasons == [FORWARD_EXPERT_LORA_TRACE_NOISE_REASON]


def test_forward_expert_lora_noise_pass_rejects_broad_escape_hatches() -> None:
    def _candidate(param: str, *, relative_l2: float = 1e-4) -> MetricRow:
        return _metric_row(
            phase="forward",
            param=param,
            pass_signal=False,
            mean_abs_pct=2.0,
            relative_l2=relative_l2,
        )

    def _gates(
        *, output_pass: bool = True, router_exact: bool = True
    ) -> list[MetricRow]:
        return [
            _metric_row(
                phase="outputs", param="logprobs.micro_000", pass_signal=output_pass
            ),
            _metric_row(
                phase="router_scores",
                param="chunk0.module.decoder.layers.__layer_avg__.mlp.router.call_3",
                pass_signal=router_exact,
                mean_abs_pct=0.0 if router_exact else 1e-9,
                relative_l2=0.0 if router_exact else 1e-9,
            ),
            _metric_row(
                phase="router_topk_ids",
                param="chunk0.module.decoder.layers.__layer_avg__.mlp.router.call_3",
                pass_signal=True,
                topk_mismatch_fraction=0.0,
                top1_mismatch_fraction=0.0,
            ),
        ]

    non_expert = _candidate(
        "chunk0.module.decoder.layers.__layer_avg__.self_attention.out_proj.lora.call_3"
    )
    VariantRunner._apply_forward_expert_lora_trace_noise_passes([non_expert, *_gates()])
    assert not non_expert.pass_signal

    too_large = _candidate(
        "chunk0.module.decoder.layers.__layer_avg__.mlp.experts.linear_fc2.lora.call_3",
        relative_l2=FORWARD_EXPERT_LORA_TRACE_NOISE_RELATIVE_L2_LIMIT + 1e-9,
    )
    VariantRunner._apply_forward_expert_lora_trace_noise_passes([too_large, *_gates()])
    assert not too_large.pass_signal

    fc1_gate = _candidate(
        "chunk0.module.decoder.layers.__layer_avg__.mlp.experts.linear_fc1.gate_lora.call_3"
    )
    VariantRunner._apply_forward_expert_lora_trace_noise_passes([fc1_gate, *_gates()])
    assert fc1_gate.pass_signal

    fc1_up = _candidate(
        "chunk0.module.decoder.layers.__layer_avg__.mlp.experts.linear_fc1.up_lora.call_3"
    )
    VariantRunner._apply_forward_expert_lora_trace_noise_passes([fc1_up, *_gates()])
    assert fc1_up.pass_signal

    output_failed = _candidate(
        "chunk0.module.decoder.layers.__layer_avg__.mlp.experts.linear_fc2.lora.call_3"
    )
    VariantRunner._apply_forward_expert_lora_trace_noise_passes(
        [output_failed, *_gates(output_pass=False)]
    )
    assert not output_failed.pass_signal

    router_not_exact = _candidate(
        "chunk0.module.decoder.layers.__layer_avg__.mlp.experts.linear_fc2.lora.call_3"
    )
    VariantRunner._apply_forward_expert_lora_trace_noise_passes(
        [router_not_exact, *_gates(router_exact=False)]
    )
    assert not router_not_exact.pass_signal


def test_suite_variants_skip_duplicate_oracle_replay_variant() -> None:
    variants = _suite_variants("rl")

    assert variants
    assert all(variant.topology != ORACLE_TOPOLOGY for variant in variants)
    assert all("oracle_replay" not in variant.name for variant in variants)


def test_dense_suite_variants_preserve_dense_and_cp_topologies() -> None:
    variants = _suite_variants("rl", is_moe=False)

    assert variants
    assert all(variant.topology != DENSE_ORACLE_TOPOLOGY for variant in variants)
    assert any(
        variant.topology.tp == 2
        and variant.topology.dp == 2
        and variant.topology.cp == 1
        for variant in variants
    )
    assert any(
        variant.topology.tp == 2
        and variant.topology.dp == 2
        and variant.topology.cp == 2
        for variant in variants
    )


def test_max_world_size_arg_filters_dense_variants() -> None:
    variants = _suite_variants("rl", is_moe=False, max_world_size=2)

    assert variants
    assert all(variant.topology.world_size() <= 2 for variant in variants)
    assert not any(
        variant.topology.tp == 2 and variant.topology.dp == 2 for variant in variants
    )


def test_oracle_topologies_are_the_compact_cp_validation_matrix() -> None:
    assert TOPOLOGIES == [
        Topology(tp=1, ep=1, etp=1, dp=1, sp=False),
        Topology(tp=1, ep=2, etp=1, dp=1, cp=2, sp=False),
        Topology(tp=2, ep=2, etp=1, dp=1, cp=2, sp=True),
        Topology(tp=2, ep=4, etp=2, dp=2, cp=2, sp=True),
    ]
    assert [topology.world_size() for topology in TOPOLOGIES] == [1, 2, 4, 8]


def test_dense_topologies_include_vllm_separation_and_cp_coverage() -> None:
    assert DENSE_TOPOLOGIES == [
        Topology(tp=1, ep=1, etp=1, dp=1, sp=False),
        Topology(tp=2, ep=1, etp=1, dp=1, sp=True),
        Topology(tp=1, ep=1, etp=1, dp=2, sp=False),
        Topology(tp=2, ep=1, etp=1, dp=2, sp=True),
        Topology(tp=1, ep=1, etp=1, dp=1, cp=2, sp=False),
        Topology(tp=2, ep=1, etp=1, dp=1, cp=2, sp=True),
        Topology(tp=2, ep=1, etp=1, dp=2, cp=2, sp=True),
    ]
    assert [topology.world_size() for topology in DENSE_TOPOLOGIES] == [
        1,
        2,
        2,
        4,
        2,
        4,
        8,
    ]


def test_dense_sensitivity_keeps_dp_and_cp_attention_cases() -> None:
    mutations = selected_sensitivity_mutations_for_objective(
        "rl",
        [
            "skip_finalize",
            "dp_local_token_normalization",
            *CP_ATTENTION_SENSITIVITY_MUTATIONS,
        ],
        is_moe=False,
    )

    assert mutations == [
        "skip_finalize",
        "dp_local_token_normalization",
        *CP_ATTENTION_SENSITIVITY_MUTATIONS,
    ]
    assert sensitivity_topology_for_mutation("skip_finalize", is_moe=False) == Topology(
        tp=2, ep=1, etp=1, dp=1, sp=True
    )
    assert (
        sensitivity_topology_for_mutation(
            "dp_local_token_normalization",
            is_moe=False,
        )
        == DENSE_DP_SENSITIVITY_TOPOLOGY
    )
    assert (
        sensitivity_topology_for_mutation(
            CP_ATTENTION_SENSITIVITY_MUTATIONS[0],
            is_moe=False,
        )
        == DENSE_CP_ATTENTION_SENSITIVITY_TOPOLOGY
    )


def test_case_config_base_model_can_be_overridden_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ART_ORACLE_BASE_MODEL", "Qwen/Qwen3.5-35B-A3B")

    assert case_config().base_model == "Qwen/Qwen3.5-35B-A3B"
    assert case_config(base_model="custom/model").base_model == "custom/model"


def test_packed_tensor_defaults_match_main_rebase_oracle_tokens() -> None:
    config = PackedTensorConfig()

    assert config.num_sequences == 4
    assert config.sequence_length == 1024
    assert config.prefill_tokens == 256
    assert config.completion_branches_per_prefix == 2
    assert config.decode_tokens == 128
    assert config.decode_tokens_jitter == 32
    assert config.packing_mode == "stop_early"
    assert config.vocab_high == 8192
