from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.megatron.model_support.handlers.dsv4 import DSV4_HANDLER
from art.megatron.routing_replay import ParallelTopology as ReplayParallelTopology

from ..artifacts import GitRepoState
from . import hf_parity as hf_parity_module
from . import hf_parity_worker as hf_parity_worker_module
from .base_megatron_session import initialize_single_rank_process_group
from .hf_parity import (
    HF_PARITY_OUTPUT_DIRNAME,
    HF_PARITY_PACKED_TENSORS,
    HF_PARITY_REPORT_FILENAME,
    HfParityReport,
    HfParityRunRequest,
    build_parity_sample_indices,
    build_tensor_map_metric_rows,
    run_hf_parity,
    set_hf_config_num_layers,
)
from .hf_parity_worker import (
    _build_megatron_runtime,
    _drop_gemma4_reparameterized_norm_grads,
    _filter_language_only_tensor_map,
    _focus_derivative_tensor_map,
    _hf_moe_router_key,
    _hf_prefix_tree_forward_inputs,
    _hf_router_num_experts,
    _HfMoeRoutingCapture,
    _is_language_hf_param_name,
    _mapping_supports_derivative_parity,
    _maybe_modify_converted_hf_grad,
    _normalize_hf_grads_for_bridge,
    _normalize_hf_tensor_map_for_bridge,
    _validate_distributed_process_env,
)
from .oracle_harness import DiskPackedTensorsSpec, OracleCaseConfig
from .validation_spec import MinimalLayerCoverageReport


def _git_state() -> GitRepoState:
    return GitRepoState(path="/repo", commit="a" * 40, dirty=False)


def test_build_parity_sample_indices_pads_with_none() -> None:
    assert build_parity_sample_indices(
        num_sequences=2,
        global_grad_accumulation_sequences=4,
    ) == [0, 1, None, None]


def test_hf_prefix_tree_inputs_block_siblings_and_repeat_positions() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=2,
        )
    )
    micro = {
        "group_ids": torch.tensor([0, 0, 1, 1, 2, 2]),
        "parent_ids": torch.tensor([0, 0, 0, 0, 0, 0]),
        "position_ids": torch.tensor([0, 1, 2, 3, 2, 3]),
    }

    attention_mask, position_ids = _hf_prefix_tree_forward_inputs(
        model,
        micro,
        actual_len=6,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert isinstance(attention_mask, dict)
    masks = cast(dict[str, torch.Tensor], attention_mask)
    full_allowed = masks["full_attention"][0, 0] == 0
    sliding_allowed = masks["sliding_attention"][0, 0] == 0
    assert position_ids.tolist() == [[0, 1, 2, 3, 2, 3]]
    assert full_allowed[4, 1]
    assert not full_allowed[4, 2]
    assert not sliding_allowed[4, 0]
    assert sliding_allowed[4, 1]


def test_hf_parity_uses_train_inf_mismatch_settings() -> None:
    assert HF_PARITY_PACKED_TENSORS.sequence_length == 256
    assert HF_PARITY_PACKED_TENSORS.prefill_tokens == 64
    assert HF_PARITY_PACKED_TENSORS.decode_tokens == 64

    phase_pass = hf_parity_module._hf_parity_phase_pass_fns()
    assert cast(Any, phase_pass["outputs"]).limits == {
        "relative_l2": 1e-2,
        "mean_abs_pct": 1.0,
    }
    assert cast(Any, phase_pass["losses"]).limits == {
        "relative_l2": 2e-2,
        "mean_abs_pct": 2.0,
    }
    assert cast(Any, phase_pass["grads"]).limits == {"mean_abs_pct": 3.0}


def test_set_hf_config_num_layers_updates_supported_field() -> None:
    config = SimpleNamespace(num_hidden_layers=28)

    field = set_hf_config_num_layers(config, 4)

    assert field == "num_hidden_layers"
    assert config.num_hidden_layers == 4


def test_set_hf_config_num_layers_updates_nested_text_config() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=40,
        layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
        mlp_only_layers=[1, 4, 7],
    )
    config = SimpleNamespace(text_config=text_config)

    field = set_hf_config_num_layers(config, 4)

    assert field == "text_config.num_hidden_layers"
    assert text_config.num_hidden_layers == 4
    assert text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]
    assert text_config.mlp_only_layers == [1]


def test_run_hf_parity_rejects_uncovered_toy_model(monkeypatch) -> None:
    monkeypatch.setattr(
        hf_parity_module,
        "assess_minimal_layer_coverage",
        lambda **_: SimpleNamespace(
            covered=False,
            missing_layer_families=["standard_attention"],
            unresolved_risks=[],
        ),
    )

    with pytest.raises(
        AssertionError,
        match="HF parity toy model does not cover required layer families",
    ):
        run_hf_parity(
            case_config=OracleCaseConfig(
                base_model="Qwen/Qwen3.5-35B-A3B",
                num_layers=2,
            )
        )


def test_run_hf_parity_always_reruns_existing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    coverage = MinimalLayerCoverageReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        requested_num_layers=4,
        recommended_min_layers=4,
        covered=True,
    )
    case_dir = tmp_path / "case"
    output_dir = case_dir / HF_PARITY_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True)
    stale_report = HfParityReport(
        git=_git_state(),
        case_id="stale",
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        requested_num_layers=4,
        coverage=coverage,
        signal="pass",
        pass_count=99,
        fail_count=0,
    )
    (output_dir / HF_PARITY_REPORT_FILENAME).write_text(
        stale_report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        hf_parity_module,
        "assess_minimal_layer_coverage",
        lambda **_: coverage,
    )
    monkeypatch.setattr(hf_parity_module, "pinned_git_state", lambda _: _git_state())
    monkeypatch.setattr(
        hf_parity_module,
        "ensure_case_artifacts",
        lambda _: SimpleNamespace(
            case_id="fresh-case",
            case_dir=str(case_dir),
            packed_tensors=DiskPackedTensorsSpec(
                dir=str(case_dir / "packed"),
                num_sequences=4,
                sequence_length=8,
            ),
        ),
    )
    calls: list[str] = []

    def _fake_subprocess(request, run_output_dir):
        calls.append(request.case_id)
        fresh_report = HfParityReport(
            git=request.git,
            case_id=request.case_id,
            base_model=request.case_config.base_model,
            model_key=request.coverage.model_key,
            requested_num_layers=request.case_config.num_layers,
            coverage=request.coverage,
            signal="pass",
            pass_count=1,
            fail_count=0,
        )
        (run_output_dir / HF_PARITY_REPORT_FILENAME).write_text(
            fresh_report.model_dump_json(indent=2),
            encoding="utf-8",
        )

    monkeypatch.setattr(hf_parity_module, "run_hf_parity_subprocess", _fake_subprocess)

    report = run_hf_parity(
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B")
    )

    assert calls == ["fresh-case"]
    assert report.case_id == "fresh-case"
    assert report.pass_count == 1


def test_run_hf_parity_subprocess_does_not_override_recompute(
    monkeypatch, tmp_path
) -> None:
    request = HfParityRunRequest(
        git=_git_state(),
        case_id="case-id",
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B"),
        packed_tensors=DiskPackedTensorsSpec(
            dir=str(tmp_path / "packed"),
            num_sequences=4,
            sequence_length=8,
        ),
        output_dir=str(tmp_path),
        coverage=MinimalLayerCoverageReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            requested_num_layers=4,
            recommended_min_layers=4,
            covered=True,
        ),
    )
    captured: dict[str, Any] = {}

    def _fake_run(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        hf_parity_module,
        "subprocess",
        SimpleNamespace(run=_fake_run),
    )

    hf_parity_module.run_hf_parity_subprocess(request, tmp_path)

    env = cast(dict[str, str], captured["env"])
    assert "ART_MEGATRON_RECOMPUTE_GRANULARITY" not in env
    assert "ART_MEGATRON_RECOMPUTE_METHOD" not in env
    assert "ART_MEGATRON_RECOMPUTE_NUM_LAYERS" not in env
    assert "ART_MEGATRON_RECOMPUTE_MODULES" not in env


def test_run_hf_parity_subprocess_does_not_allocate_tcp_rendezvous(
    monkeypatch, tmp_path
) -> None:
    request = HfParityRunRequest(
        git=_git_state(),
        case_id="case-id",
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B"),
        packed_tensors=DiskPackedTensorsSpec(
            dir=str(tmp_path / "packed"),
            num_sequences=4,
            sequence_length=8,
        ),
        output_dir=str(tmp_path),
        coverage=MinimalLayerCoverageReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            requested_num_layers=4,
            recommended_min_layers=4,
            covered=True,
        ),
    )
    environments: list[dict[str, str]] = []
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    def _fake_run(*args, **kwargs):
        del args
        environments.append(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        hf_parity_module,
        "subprocess",
        SimpleNamespace(run=_fake_run),
    )

    hf_parity_module.run_hf_parity_subprocess(request, tmp_path)

    for env in environments:
        assert "MASTER_ADDR" not in env
        assert "MASTER_PORT" not in env
        assert env["RANK"] == "0"
        assert env["WORLD_SIZE"] == "1"
        assert env["LOCAL_RANK"] == "0"
        assert env["LOCAL_WORLD_SIZE"] == "1"


def test_hf_parity_worker_requires_explicit_distributed_env(monkeypatch) -> None:
    distributed_env = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
    }
    for name in distributed_env:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="explicit distributed environment"):
        _validate_distributed_process_env()

    for name, value in distributed_env.items():
        monkeypatch.setenv(name, value)
    _validate_distributed_process_env()


def test_single_rank_process_group_uses_an_in_process_store(monkeypatch) -> None:
    store = object()
    init_kwargs: dict[str, object] = {}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "HashStore", lambda: store)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: init_kwargs.update(kwargs),
    )

    initialize_single_rank_process_group()

    assert init_kwargs == {
        "backend": "nccl",
        "store": store,
        "rank": 0,
        "world_size": 1,
    }


def test_normalize_hf_tensor_map_for_bridge_adds_language_model_prefix() -> None:
    normalized = _normalize_hf_tensor_map_for_bridge(
        {
            "model.layers.0.input_layernorm.weight": torch.ones(1),
            "lm_head.weight": torch.ones(1),
        },
        {
            "model.language_model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        },
    )

    assert set(normalized) == {
        "model.language_model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    }


def test_build_tensor_map_metric_rows_rejects_tensor_set_mismatch() -> None:
    rows = build_tensor_map_metric_rows(
        phase="grads",
        reference={"a": torch.ones(1)},
        candidate={"b": torch.ones(1)},
    )

    assert len(rows) == 1
    assert rows[0].param == "__tensor_set__"
    assert rows[0].pass_signal is False
    assert "missing=['a'] extra=['b']" in rows[0].failure_reasons[0]


def test_build_tensor_map_metric_rows_enforces_nonzero_per_tensor() -> None:
    rows = build_tensor_map_metric_rows(
        phase="grads",
        reference={"all_zero": torch.zeros(2), "active": torch.ones(2)},
        candidate={"all_zero": torch.zeros(2), "active": torch.ones(2)},
    )
    by_param = {row.param: row for row in rows}

    assert by_param["all_zero"].pass_signal is False
    assert by_param["active"].pass_signal is True


def test_grouped_tensor_map_rows_keep_individual_nonzero_gates() -> None:
    rows = build_tensor_map_metric_rows(
        phase="grads",
        reference={"large": torch.ones(1000), "small": torch.ones(1)},
        candidate={"large": torch.full((1000,), 1.01), "small": torch.full((1,), 2.0)},
        group_by=lambda _: "joint_update",
    )
    by_param = {row.param: row for row in rows}

    assert by_param["joint_update"].phase == "grads"
    assert by_param["joint_update"].mean_abs_pct < 3.0
    assert by_param["joint_update"].pass_signal is True
    assert by_param["small"].phase == "grads_diagnostic"
    assert by_param["small"].mean_abs_pct == 100.0
    assert by_param["small"].pass_signal is True

    zero_rows = build_tensor_map_metric_rows(
        phase="grads",
        reference={"active": torch.ones(2), "zero": torch.zeros(1)},
        candidate={"active": torch.ones(2), "zero": torch.zeros(1)},
        group_by=lambda _: "joint_update",
    )
    assert {row.param: row for row in zero_rows}["zero"].pass_signal is False


@pytest.mark.parametrize(
    ("param", "group"),
    (
        ("model.embed_tokens.weight", "embedding"),
        ("lm_head.weight", "final_envelope"),
        ("model.hc_head.hc_scale", "final_envelope"),
        ("model.norm.weight", "final_envelope"),
        ("model.layers.2.attn_hc.base", "model.layers.2.attention"),
        (
            "model.layers.2.self_attn.compressor.kv_proj.weight",
            "model.layers.2.attention",
        ),
        ("model.layers.2.ffn_hc.fn", "model.layers.2.ffn"),
        ("model.layers.2.mlp.experts.0.up_proj.weight", "model.layers.2.ffn"),
        ("model.layers.2.input_layernorm.weight", "model.layers.2.input_norm"),
        (
            "model.layers.2.post_attention_layernorm.weight",
            "model.layers.2.post_attention_norm",
        ),
    ),
)
def test_dsv4_hf_parity_gradient_groups(param: str, group: str) -> None:
    assert DSV4_HANDLER.hf_parity_gradient_group(param) == group


def test_dsv4_hf_parity_gradient_groups_reject_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Unmapped DSV4 HF-parity gradient"):
        DSV4_HANDLER.hf_parity_gradient_group("model.layers.0.unknown.weight")


def test_language_hf_param_filter_keeps_text_and_drops_visual() -> None:
    assert _is_language_hf_param_name("model.layers.0.self_attn.q_proj.weight") is True
    assert _is_language_hf_param_name("model.visual.blocks.0.attn.qkv.weight") is False
    filtered = _filter_language_only_tensor_map(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(1),
            "model.visual.blocks.0.attn.qkv.weight": torch.ones(1),
        }
    )
    assert set(filtered) == {"model.layers.0.self_attn.q_proj.weight"}
    assert torch.equal(
        filtered["model.layers.0.self_attn.q_proj.weight"],
        torch.ones(1),
    )


def test_normalize_hf_grads_for_bridge_keeps_expected_key_set() -> None:
    normalized = _normalize_hf_grads_for_bridge(
        {
            "model.layers.0.input_layernorm.weight": torch.ones(1),
            "lm_head.weight": torch.ones(1),
            "model.visual.blocks.0.attn.qkv.weight": torch.ones(1),
        },
        expected_grad_keys={
            "model.language_model.layers.0.input_layernorm.weight",
            "lm_head.weight",
        },
    )

    assert set(normalized) == {
        "model.language_model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    }


def test_hf_moe_routing_capture_recognizes_supported_router_names() -> None:
    assert (
        _hf_moe_router_key("model.layers.3.mlp.gate")
        == "chunk_00.layer_0003.mlp.router"
    )
    assert (
        _hf_moe_router_key("model.language_model.layers.5.router")
        == "chunk_00.layer_0005.mlp.router"
    )
    assert _hf_moe_router_key("model.layers.7.router") == (
        "chunk_00.layer_0007.mlp.router"
    )
    assert _hf_moe_router_key("backbone.layers.6.mixer.gate") == (
        "chunk_00.layer_0006.mlp.router"
    )
    assert _hf_moe_router_key("model.backbone.layers.8.mixer.gate") == (
        "chunk_00.layer_0008.mlp.router"
    )
    assert _hf_moe_router_key("model.language_model.layers.5.mlp.gate") is None


def _nemotron_router_model(
    output: tuple[torch.Tensor, torch.Tensor],
    *,
    n_routed_experts: int | None = 4,
    top_k: int = 2,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    class NemotronHTopkRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.top_k = top_k
            if n_routed_experts is not None:
                self.n_routed_experts = n_routed_experts

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return output

    model = torch.nn.Module()
    model.backbone = torch.nn.Module()
    model.backbone.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.backbone.layers[0].mixer = torch.nn.Module()
    router = NemotronHTopkRouter()
    model.backbone.layers[0].mixer.gate = router
    return model, router


@pytest.mark.parametrize("weights_dtype", (torch.float32, torch.bfloat16))
def test_hf_moe_routing_capture_replays_nemotron_router_output(
    weights_dtype: torch.dtype,
) -> None:
    model, router = _nemotron_router_model(
        (
            torch.tensor([[0, 3], [2, 1]], dtype=torch.int64),
            torch.tensor([[1.25, 0.75], [1.5, 0.5]], dtype=weights_dtype),
        )
    )
    capture = _HfMoeRoutingCapture(model, required=True)
    capture.set_active_micro(sample_index=7, micro_slot=0)
    router(torch.ones(2, 4))
    replay = capture.build_replay_bundle(topology=ReplayParallelTopology(tp=1, ep=1))
    capture.close()

    assert replay is not None
    route = replay.steps[0].routers["chunk_00.layer_0000.mlp.router"].calls[0]
    assert route.num_experts == 4
    assert route.expert_indices.device.type == "cpu"
    assert route.expert_indices.dtype == torch.int32
    assert route.expert_indices.is_contiguous()
    assert route.expert_indices.tolist() == [[0, 3], [2, 1]]
    assert route.expert_probs is not None
    assert route.expert_probs.device.type == "cpu"
    assert route.expert_probs.dtype == torch.float32
    assert route.expert_probs.is_contiguous()
    assert route.expert_probs.tolist() == [[1.25, 0.75], [1.5, 0.5]]


def test_hf_moe_routing_capture_preserves_existing_three_tuple_output() -> None:
    class ExistingRouter(torch.nn.Module):
        num_experts = 4

        def forward(self, hidden_states: torch.Tensor) -> tuple[Any, ...]:
            return (
                None,
                torch.tensor([[0.75, 0.25]], dtype=torch.float32),
                torch.tensor([[3, 1]], dtype=torch.int64),
            )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    model.model.layers[0].mlp.gate = ExistingRouter()
    capture = _HfMoeRoutingCapture(model, required=True)
    capture.set_active_micro(sample_index=2, micro_slot=0)
    model.model.layers[0].mlp.gate(torch.ones(1, 4))
    replay = capture.build_replay_bundle(topology=ReplayParallelTopology(tp=1, ep=1))
    capture.close()

    assert replay is not None
    route = replay.steps[0].routers["chunk_00.layer_0000.mlp.router"].calls[0]
    assert route.expert_indices.tolist() == [[3, 1]]
    assert route.expert_probs is not None
    assert route.expert_probs.tolist() == [[0.75, 0.25]]


@pytest.mark.parametrize(
    ("output", "message"),
    (
        (
            (
                torch.ones(2, 2, dtype=torch.float32),
                torch.ones(2, 2, dtype=torch.int64),
            ),
            "indices dtype",
        ),
        (
            (
                torch.ones(2, 2, dtype=torch.int64),
                torch.ones(2, 2, dtype=torch.int64),
            ),
            "weights dtype",
        ),
        (
            (
                torch.ones(2, dtype=torch.int64),
                torch.ones(2, dtype=torch.float32),
            ),
            "rank-2",
        ),
        (
            (
                torch.ones(2, 2, dtype=torch.int64),
                torch.ones(2, 1, dtype=torch.float32),
            ),
            "shape mismatch",
        ),
        (
            (
                torch.ones(2, 1, dtype=torch.int64),
                torch.ones(2, 1, dtype=torch.float32),
            ),
            "top-k width",
        ),
    ),
)
def test_hf_moe_routing_capture_rejects_invalid_nemotron_output(
    output: tuple[torch.Tensor, torch.Tensor], message: str
) -> None:
    model, router = _nemotron_router_model(output)
    capture = _HfMoeRoutingCapture(model, required=True)
    capture.set_active_micro(sample_index=7, micro_slot=0)
    with pytest.raises(RuntimeError, match=message):
        router(torch.ones(2, 4))
    capture.close()


def test_hf_router_num_experts_uses_nested_config() -> None:
    module = SimpleNamespace(config=SimpleNamespace(num_experts=128))
    assert _hf_router_num_experts(module, torch.ones(2, 8)) == 128


def test_hf_router_num_experts_uses_n_routed_experts() -> None:
    module = SimpleNamespace(n_routed_experts=4)
    assert _hf_router_num_experts(module, torch.ones(2, 2)) == 4
    nested = SimpleNamespace(config=SimpleNamespace(n_routed_experts=8))
    assert _hf_router_num_experts(nested, torch.ones(2, 2)) == 8
    with pytest.raises(RuntimeError, match="conflicting expert counts"):
        _hf_router_num_experts(
            SimpleNamespace(num_experts=4, n_routed_experts=8), torch.ones(2, 2)
        )
    with pytest.raises(RuntimeError, match="expert count"):
        _hf_router_num_experts(SimpleNamespace(), torch.ones(2, 2))


def test_hf_moe_routing_capture_is_required_only_for_moe() -> None:
    model = torch.nn.Module()
    with pytest.raises(RuntimeError, match="requires HF MoE routing capture"):
        _HfMoeRoutingCapture(model, required=True)
    capture = _HfMoeRoutingCapture(model)
    assert (
        capture.build_replay_bundle(topology=ReplayParallelTopology(tp=1, ep=1)) is None
    )


def test_hf_moe_routing_capture_requires_complete_nemotron_pattern() -> None:
    model, _router = _nemotron_router_model(
        (
            torch.ones(2, 2, dtype=torch.int64),
            torch.ones(2, 2, dtype=torch.float32),
        )
    )
    setattr(model, "config", SimpleNamespace(hybrid_override_pattern="EE"))
    with pytest.raises(RuntimeError, match="capture is incomplete.*layers.1"):
        _HfMoeRoutingCapture(model, required=True)


def test_focus_derivative_tensor_map_handles_nemotron_moe_names() -> None:
    focused = _focus_derivative_tensor_map(
        {
            "backbone.embeddings.weight": torch.arange(15).view(5, 3),
            "backbone.layers.1.mixer.gate.weight": torch.arange(12).view(4, 3),
            "backbone.layers.1.mixer.experts.0.up_proj.weight": torch.ones(2),
            "backbone.layers.1.mixer.experts.1.up_proj.weight": torch.ones(2),
        },
        active_embedding_rows=torch.tensor([1, 4]),
        active_router_rows={1: torch.tensor([0, 2])},
        last_layer_index=3,
        loss_active_last_layer_experts=set(),
    )

    assert set(focused) == {
        "backbone.embeddings.weight",
        "backbone.layers.1.mixer.gate.weight",
        "backbone.layers.1.mixer.experts.0.up_proj.weight",
    }
    assert focused["backbone.embeddings.weight"].tolist() == [
        [3, 4, 5],
        [12, 13, 14],
    ]
    assert focused["backbone.layers.1.mixer.gate.weight"].tolist() == [
        [0, 1, 2],
        [6, 7, 8],
    ]


def test_build_megatron_runtime_uses_training_provider_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    configured_bundles: list[tuple[Any, bool]] = []
    runtime = SimpleNamespace(provider="provider", model=["model"])

    monkeypatch.setattr(
        hf_parity_worker_module.megatron_train,
        "build_training_runtime",
        lambda **kwargs: calls.append(kwargs) or runtime,
    )
    monkeypatch.setattr(
        hf_parity_worker_module,
        "_configure_hf_parity_provider_bundle",
        lambda provider_bundle, *, use_hf_reference_state: configured_bundles.append(
            (provider_bundle, use_hf_reference_state)
        ),
    )

    request = HfParityRunRequest(
        git=_git_state(),
        case_id="case",
        case_config=OracleCaseConfig(base_model="Qwen/Qwen3.5-35B-A3B"),
        packed_tensors=DiskPackedTensorsSpec(
            dir="/tmp", num_sequences=4, sequence_length=8
        ),
        output_dir="/tmp/out",
        coverage=MinimalLayerCoverageReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            requested_num_layers=4,
            recommended_min_layers=4,
            covered=True,
        ),
    )

    built_runtime = _build_megatron_runtime(request)

    assert built_runtime is runtime
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model_identifier"] == "Qwen/Qwen3.5-35B-A3B"
    assert kwargs["provider_torch_dtype"] == torch.float32
    provider_bundle = SimpleNamespace()
    kwargs["provider_bundle_configure"](provider_bundle)
    assert configured_bundles == [(provider_bundle, False)]
    assert kwargs["print_env"] is False
    assert kwargs["trainable_parameter_mode"] == "base_model"
    configured_provider = SimpleNamespace(_art_model_support_handler=SimpleNamespace())
    kwargs["provider_configure"](configured_provider)
    optimizer_config = kwargs["optimizer_config"]
    assert configured_provider.num_layers == request.case_config.num_layers
    assert optimizer_config.params_dtype == torch.float32


def test_mapping_supports_derivative_parity_rejects_affine_weight_exports() -> None:
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        RMSNorm2ZeroCenteredRMSNormMapping,
    )

    assert _mapping_supports_derivative_parity(AutoMapping("a", "b")) is True
    assert (
        _mapping_supports_derivative_parity(
            RMSNorm2ZeroCenteredRMSNormMapping("a", "b")
        )
        is False
    )


class Gemma4BridgeForTest:
    pass


def test_gemma4_router_grad_export_applies_chain_rule() -> None:
    key = "model.language_model.layers.0.router.proj.weight"
    prefix = "model.language_model.layers.0."
    grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    hf_weight = torch.tensor([[5.0, 7.0], [11.0, 13.0]])
    scale = torch.tensor([3.0, 5.0])
    ln2 = torch.tensor([2.0, 4.0])

    converted, additive = _maybe_modify_converted_hf_grad(
        Gemma4BridgeForTest(),
        SimpleNamespace(),
        {key: grad},
        {
            key: hf_weight,
            f"{prefix}router.scale": scale,
            f"{prefix}pre_feedforward_layernorm_2.weight": ln2,
        },
        model_is_moe=True,
    )

    root = grad.shape[-1] ** -0.5
    factor = scale * root / ln2
    assert torch.allclose(converted[key], grad * factor)
    assert torch.allclose(
        converted[f"{prefix}pre_feedforward_layernorm_2.weight"],
        (grad * hf_weight * (-scale * root / ln2.square()).unsqueeze(0)).sum(dim=0),
    )
    assert additive == {f"{prefix}pre_feedforward_layernorm_2.weight"}


def test_gemma4_absent_v_grad_export_adds_to_k() -> None:
    prefix = "model.language_model.layers.5.self_attn."
    k_key = f"{prefix}k_proj.weight"
    v_key = f"{prefix}v_proj.weight"
    k_grad = torch.tensor([[1.0, 2.0]])
    v_grad = torch.tensor([[3.0, 4.0]])

    converted, additive = _maybe_modify_converted_hf_grad(
        Gemma4BridgeForTest(),
        SimpleNamespace(),
        {k_key: k_grad, v_key: v_grad},
        {k_key: torch.ones_like(k_grad)},
        model_is_moe=False,
    )

    assert torch.equal(converted[k_key], k_grad + v_grad)
    assert additive == {k_key}


def test_drop_gemma4_reparameterized_norm_grads_is_exact() -> None:
    kept_key = "model.language_model.layers.0.self_attn.q_norm.weight"
    dropped_key = "model.language_model.layers.0.pre_feedforward_layernorm_2.weight"
    filtered = _drop_gemma4_reparameterized_norm_grads(
        {
            kept_key: torch.ones(1),
            dropped_key: torch.ones(1),
        }
    )

    assert set(filtered) == {kept_key}


def test_gemma4_shared_expert_grad_export_applies_chain_rule() -> None:
    prefix = "model.language_model.layers.0."
    gate_key = f"{prefix}mlp.gate_proj.weight"
    up_key = f"{prefix}mlp.up_proj.weight"
    gate_grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    up_grad = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    gate_weight = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    up_weight = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
    pffl = torch.tensor([3.0, 5.0])
    ln2 = torch.tensor([2.0, 4.0])

    converted, additive = _maybe_modify_converted_hf_grad(
        Gemma4BridgeForTest(),
        SimpleNamespace(),
        {gate_key: gate_grad, up_key: up_grad},
        {
            gate_key: gate_weight,
            up_key: up_weight,
            f"{prefix}pre_feedforward_layernorm.weight": pffl,
            f"{prefix}pre_feedforward_layernorm_2.weight": ln2,
        },
        model_is_moe=True,
    )

    factor = pffl / ln2
    assert torch.allclose(converted[gate_key], gate_grad * factor)
    assert torch.allclose(converted[up_key], up_grad * factor)
    expected_ln2 = (gate_grad * gate_weight * (-pffl / ln2.square()).unsqueeze(0)).sum(
        dim=0
    ) + (up_grad * up_weight * (-pffl / ln2.square()).unsqueeze(0)).sum(dim=0)
    assert torch.allclose(
        converted[f"{prefix}pre_feedforward_layernorm_2.weight"],
        expected_ln2,
    )
    assert additive == {f"{prefix}pre_feedforward_layernorm_2.weight"}
