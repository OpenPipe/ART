import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from art.megatron.model_support.spec import (
    ArchitectureReport,
    LayerFamilyInstance,
)
from art.pipeline_tuner import PipelineTuneSettings

from .validation_spec import ValidationStageResult
from .workflow import (
    INCLUDE_FLASH_SENSITIVITY_ENV,
    MANDATORY_VALIDATION_STAGES,
    WORKFLOW_STAGE_DIR_ENV,
    _inspect_architecture_for_workflow,
    _prune_runtime_artifacts,
    build_validation_report,
    build_validation_stage_names,
    run_lora_coverage_stage,
    validated_architecture_representative_models,
)
from .workflow_fixtures import (
    FIXTURE_PATH_ENV,
    WorkflowFixture,
    _fixture_paths,
    _validate_tokenizer_compatible_fixture,
)
from .workflow_resources import (
    HANDLER_WORKFLOW_RESOURCES,
    ThroughputThresholds,
    ThroughputWorkflowConfig,
    handler_workflow_resources_for_base_model,
    resolve_stage_resources_for_current_host,
    resolve_stage_resources_for_visible_gpus,
)
from .workflow_throughput import (
    PolicyActivationEvent,
    ThroughputFixture,
    _canonical_production_config_sha256,
    _classify_acceptance_failures,
    _collect_measurements,
    _current_pipeline_settings,
    _digest,
    _freeze_pipeline_settings_from_step,
    _packed_batch_fingerprint,
    _phase_evidence,
    _run_throughput_attempts,
    _settled_execution_decision_suffix,
    _sized_prompt,
    _synthetic_sampling_constraints,
    acceptance_failures,
)


@pytest.fixture(autouse=True)
def _stub_workflow_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(INCLUDE_FLASH_SENSITIVITY_ENV, raising=False)
    fixture_path = tmp_path / "correctness_fixture"
    tokenizer_compatible_path = tmp_path / "tokenizer_compatible_fixture"
    stage_path = tmp_path / "stage"
    fixture_path.mkdir()
    tokenizer_compatible_path.mkdir()
    stage_path.mkdir()
    monkeypatch.setenv(FIXTURE_PATH_ENV, str(fixture_path))
    monkeypatch.setenv(WORKFLOW_STAGE_DIR_ENV, str(stage_path))
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.pinned_git_state",
        lambda suite_name: SimpleNamespace(
            model_dump=lambda mode="json": {
                "path": "/tmp/art",
                "commit": "test",
                "dirty": False,
                "status": [],
            }
        ),
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.ensure_workflow_fixture",
        lambda base_model, allow_unvalidated_arch=False, required_stages=frozenset(): (
            WorkflowFixture(
                canonical_model=base_model,
                model_key="qwen3_5_moe",
                source_revision="test",
                path=str(fixture_path),
                hf_home=str(tmp_path / "hf_home"),
                manifest={"version": 15},
                tokenizer_compatible_path=str(tokenizer_compatible_path),
                tokenizer_compatible_hf_home=str(tmp_path / "tokenizer_hf_home"),
                tokenizer_compatible_manifest={"version": 1},
                functional_path=str(fixture_path),
                functional_hf_home=str(tmp_path / "hf_home"),
                functional_manifest={"version": 1, "num_layers": 8},
                canonical_path=str(fixture_path),
                canonical_hf_home=str(tmp_path / "hf_home"),
            )
        ),
    )


def _fixture(tmp_path: Path, model_key: str) -> WorkflowFixture:
    return WorkflowFixture(
        canonical_model=model_key,
        model_key=model_key,
        source_revision="pinned",
        path=str(tmp_path / "compact"),
        hf_home=str(tmp_path / "compact_cache"),
        manifest={"version": 15},
        tokenizer_compatible_path=str(tmp_path / "tokenizer"),
        tokenizer_compatible_hf_home=str(tmp_path / "tokenizer_cache"),
        functional_path=str(tmp_path / "functional"),
        functional_hf_home=str(tmp_path / "functional_cache"),
        functional_manifest={"version": 1, "num_layers": 8},
        canonical_path=str(tmp_path / "canonical"),
        canonical_hf_home=str(tmp_path / "canonical_cache"),
    )


def test_canonical_production_config_digest_uses_resolved_config(tmp_path: Path) -> None:
    import torch
    from transformers import AutoConfig, LlamaConfig

    from art.runtime_attestation import canonical_config_sha256

    production_config = tmp_path / "production_config"
    LlamaConfig(num_hidden_layers=2).save_pretrained(production_config)
    source = json.loads((production_config / "config.json").read_text())
    resolved = AutoConfig.from_pretrained(
        production_config,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )

    digest = _canonical_production_config_sha256(production_config / "config.json")
    assert digest == canonical_config_sha256(resolved)
    assert digest != _digest(source)


def test_shared_fixture_root_rehomes_node_local_paths(tmp_path: Path) -> None:
    assert all(path.parent == tmp_path for path in _fixture_paths(str(tmp_path)))


def test_fixture_stage_contracts(tmp_path: Path) -> None:
    # fmt: off
    cases = {
        ("gemma4_dense", "canonical"): ("hf_parity", "packing_invariance", "length_trainability"),
        ("gemma4_dense", "compact"): ("lora_coverage",),
        ("gemma4_dense", "functional"): ("train_inf_mismatch",),
        ("llama3_dense", "compact"): ("hf_parity",),
        ("llama3_dense", "functional"): ("train_inf_mismatch",),
        ("llama3_dense", "canonical"): ("length_trainability",),
        ("qwen3_5_moe", "canonical"): ("length_trainability",),
        ("gpt_oss_moe", "functional"): ("train_inf_mismatch",),
        ("gpt_oss_moe", "canonical"): ("length_trainability",),
        ("glm52", "functional"): ("train_inf_mismatch",),
        ("glm52", "compact"): ("length_trainability",),
        ("dsv4", "functional"): ("train_inf_mismatch",),
        ("dsv4", "canonical"): ("length_trainability",),
    }
    # fmt: on
    for (model_key, selected), stages in cases.items():
        for stage in stages:
            environment = _fixture(tmp_path, model_key).environment(stage)
            assert environment[FIXTURE_PATH_ENV] == str(tmp_path / selected)
            assert environment["ART_ORACLE_BASE_MODEL"] == str(tmp_path / selected)
            if selected == "functional":
                assert environment["ART_MODEL_SUPPORT_FUNCTIONAL_NUM_LAYERS"] == "8"


def test_fixture_stage_contracts_require_available_assets(tmp_path: Path) -> None:
    for stage, missing, contract in (
        ("hf_parity", "canonical_path", "canonical weights"),
        (
            "train_inf_mismatch",
            "functional_path",
            "pretrained production-width functional weights",
        ),
    ):
        fixture = _fixture(tmp_path, "gemma4_dense").model_copy(update={missing: None})
        with pytest.raises(RuntimeError, match=f"requires {contract}"):
            fixture.environment(stage)


def test_reduced_trainability_preserves_validated_token_contract(
    tmp_path: Path,
) -> None:
    for model_key, stage, expected in (
        ("glm52", "length_trainability", "154820,38069"),
    ):
        key = f"ART_MODEL_SUPPORT_{stage.removesuffix('_trainability').upper()}_ALLOWED_TOKEN_IDS"
        assert _fixture(tmp_path, model_key).environment(stage)[key] == expected


@pytest.mark.parametrize(
    ("vocab_size", "registered_max", "encoded_max", "error"),
    [
        (8_192, 9_000, 3, "registered tokenizer ID 9000"),
        (128_256, 128_255, 128_009, None),
    ],
)
def test_tokenizer_compatible_fixture_preflight(
    monkeypatch: pytest.MonkeyPatch,
    vocab_size: int,
    registered_max: int,
    encoded_max: int,
    error: str | None,
) -> None:
    class Tokenizer:
        chat_template = "template"

        def get_vocab(self):
            return {"ordinary": 1, "highest": registered_max}

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [1, encoded_max]}

        apply_chat_template = __call__

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: Tokenizer(),
    )
    manifest: dict[str, object] = {"config_vocab_size": vocab_size}
    if error:
        with pytest.raises(RuntimeError, match=error):
            _validate_tokenizer_compatible_fixture(Path("/tmp/provider"), manifest)
    else:
        _validate_tokenizer_compatible_fixture(Path("/tmp/provider"), manifest)
        assert manifest["representative_max_token_id"] == encoded_max
        assert manifest["tokenizer_max_id"] == registered_max


def test_throughput_runtime_keeps_canonical_handler_separate_from_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.megatron.runtime import local as local_runtime

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        local_runtime,
        "get_megatron_runtime_config",
        lambda: SimpleNamespace(
            topology={"tp": 1, "ep": 1, "etp": 1, "cp": 1, "pp": 1}
        ),
    )
    topology = local_runtime.compile_local_runtime_topology(
        cast(
            Any,
            {
                "trainer_gpu_ids": [0],
                "init_args": {"model_name": "/tmp/production-width-provider"},
            },
        ),
        model_name="validation",
        base_model="meta-llama/Llama-3.2-1B-Instruct",
        artifact_root="/tmp/art",
        visible_gpu_count=1,
    )

    assert topology.model_services[0].base_model == "/tmp/production-width-provider"


def test_throughput_measurements_use_runtime_rows_and_activation_timestamps(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "step": step,
            "data/step_num_groups_trainable": 8,
            "data/step_num_groups_submitted": 24,
            "data/step_packed_sequences": 1,
            "data/step_nonpadding_logical_tokens": 1_000,
            "train/prefix_tree/logical_tokens": 4_000,
            "data/step_loss_bearing_tokens": 500,
            "data/step_trainable_assistant_tokens": 500,
            "data/step_executed_token_equivalents": 1_000,
            "data/step_dummy_executed_token_equivalents": 0,
            "data/step_nominal_schedule_capacity_tokens": 131_072,
            "data/step_dummy_schedule_capacity_tokens": 0,
            "data/step_unused_packed_capacity_tokens": 130_072,
            "data/step_num_gradient_steps": 1,
            "pipeline/global_real_microbatches": 1,
            "pipeline/global_dummy_microbatches": 0,
            "pipeline/packed_sequence_length": 131_072,
            "pipeline_settings/num_rollout_workers": 16,
            "pipeline_settings/min_batch_size": 8,
            "pipeline_settings/max_batch_size": 32,
            "pipeline_settings/queue_maxsize": 48,
            "pipeline_settings/target_groups_per_step": 24,
            "queue/packing_policy_lag_steps": 1,
            "time/step_train_s": 1.5,
            "time/step_wall_s": 2.0,
            "time/step_collect_batch_s": 0.001068115234375,
            "queue/packed_get_wait_s": 0.1 if step >= 7 else 0.001,
            "queue/packed_queue_depth": 0.0 if step == 6 else 1.0,
            "time/inter_forward_backward_gpu_gap_rank_0_s": (
                1.0 if step >= 6 else 0.1 + (step - 2) * 0.01
            ),
            "time/inter_forward_backward_gpu_gap_rank_1_s": (
                2.0 if step >= 6 else 0.11 + (step - 2) * 0.01
            ),
            "offpolicy/token_weighted_policy_age_steps": 1.0,
            "offpolicy/token_weighted_policy_age_p95_steps": 2.0,
            "sample_efficiency/freshness_discount": 0.8,
            "discarded/step/stale_groups": 0,
            "discarded/step/zero_variance_groups": 0,
        }
        for step in range(1, 10)
    ]
    history_path = tmp_path / "history.jsonl"
    history_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    measured_settings = PipelineTuneSettings(
        num_rollout_workers=16,
        min_batch_size=8,
        max_batch_size=32,
        queue_maxsize=48,
        target_groups_per_step=24,
    )
    future_settings = measured_settings.model_copy(update={"num_rollout_workers": 14})
    profile = SimpleNamespace(
        config=SimpleNamespace(mode="online", window_steps=2),
        decisions=[
            SimpleNamespace(
                action="hold",
                previous=measured_settings,
                updated=measured_settings,
                stats=SimpleNamespace(
                    start_step=2,
                    end_step=3,
                    window_start_s=-4.0,
                    window_end_s=0.0,
                    vllm_pressure=0.6,
                    vllm_waiting_capacity_request_s=6.0,
                    vllm_running_request_s=10.0,
                    trainer_underfeed_score=0.07,
                    actual_stale_frac=0.0,
                ),
            ),
            SimpleNamespace(
                action="decrease_workers",
                previous=measured_settings,
                updated=future_settings,
                stats=SimpleNamespace(
                    start_step=4,
                    end_step=5,
                    window_start_s=0.0,
                    window_end_s=4.0,
                    vllm_pressure=0.45,
                    vllm_waiting_capacity_request_s=4.5,
                    vllm_running_request_s=10.0,
                    trainer_underfeed_score=0.10,
                    actual_stale_frac=0.0,
                ),
            ),
            SimpleNamespace(
                action="hold",
                previous=future_settings,
                updated=future_settings,
                stats=SimpleNamespace(
                    start_step=6,
                    end_step=7,
                    window_start_s=4.0,
                    window_end_s=8.0,
                    vllm_pressure=0.65,
                    vllm_waiting_capacity_request_s=19.5,
                    vllm_running_request_s=30.0,
                    trainer_underfeed_score=0.04,
                    actual_stale_frac=0.0,
                ),
            ),
            SimpleNamespace(
                action="decrease_workers",
                previous=future_settings,
                updated=future_settings.model_copy(update={"num_rollout_workers": 12}),
                stats=SimpleNamespace(
                    start_step=8,
                    end_step=9,
                    window_start_s=8.0,
                    window_end_s=12.0,
                    vllm_pressure=0.6,
                    vllm_waiting_capacity_request_s=6.0,
                    vllm_running_request_s=10.0,
                    trainer_underfeed_score=0.5,
                    actual_stale_frac=0.0,
                ),
            ),
        ],
        policy_age_limit_steps=4,
    )
    events = [
        PolicyActivationEvent(1, -4.25, -4.0),
        PolicyActivationEvent(2, -3.5, -3.25),
        PolicyActivationEvent(3, -1.5, -1.25),
        PolicyActivationEvent(4, 0.5, 0.75),
        PolicyActivationEvent(5, 2.5, 2.75),
        PolicyActivationEvent(6, 4.5, 4.75),
        PolicyActivationEvent(7, 6.5, 7.75),
        PolicyActivationEvent(8, 8.5, 8.75),
        PolicyActivationEvent(9, 10.5, 10.75),
    ]
    config = ThroughputWorkflowConfig(num_layers=2, completion_tokens=128, max_steps=7)
    fixture = ThroughputFixture(
        model_key="llama3_dense",
        path="/tmp/llama-throughput",
        num_layers=2,
        width_fingerprint={"hidden_size": 2048},
        manifest={"initialization": "deterministic_random_v1"},
    )

    def phase(kind: str, packed: str, steps: tuple[int, ...]):
        phase_rows = [dict(rows[-1]) for _ in range(3)]
        phase_rows[-1]["data/step_nonpadding_logical_tokens"] += 1
        phase_rows[-1]["data/step_unused_packed_capacity_tokens"] -= 1
        return _phase_evidence(
            phase=cast(Any, kind),
            runtime_fingerprint="runtime-a",
            trajectory_input_fingerprint="trajectory-a",
            packed_input_fingerprint=packed,
            samples=list(zip(phase_rows, steps, strict=True)),
        )

    e2e_phase, isolated_phase = (
        phase("e2e", "input-a", (5, 6, 7)),
        phase("isolated", "input-a", (9, 10, 11)),
    )

    def collect(isolated):
        return _collect_measurements(
            fixture=fixture,
            config=config,
            hardware="b300",
            model_output_dir=tmp_path,
            profile=profile,
            events=events,
            isolated=isolated,
            e2e=e2e_phase,
            capture_settings=measured_settings.model_dump(mode="json"),
            calibration_fingerprint="a" * 64,
        )

    measurements = collect(isolated_phase)

    expected = {
        "original_trajectory_tokens": 24_000,
        "nonpadding_logical_tokens": 6_000,
        "loss_bearing_tokens": 3_000,
        "accepted_train_tokens": 3_000,
        "isolated_train_tok_s": 1_000 / 1.5,
        "matched_e2e_core_train_tok_s": 1_000 / 1.5,
        "matched_core_to_isolated_ratio": 1.0,
        "e2e_core_train_tok_s": 8_000 / 12.0,
        "e2e_train_tok_s": 500.0,
        "accepted_train_tok_s": 250.0,
        "unused_and_dummy_ratio": 1.0 - 1_000 / 131_072,
        "queue_ready_inter_forward_backward_gap_rank_zero_mean_s": 0.115,
        "queue_ready_inter_forward_backward_gap_rank_zero_p50_s": 0.115,
        "queue_ready_inter_forward_backward_gap_rank_zero_p95_s": 0.1285,
        "queue_ready_inter_forward_backward_gap_rank_zero_max_s": 0.13,
        "queue_ready_inter_forward_backward_gap_rank_zero_count": 4,
        "queue_ready_inter_forward_backward_gap_worst_rank": 1,
        "queue_ready_inter_forward_backward_gap_worst_rank_mean_s": 0.125,
        "queue_ready_inter_forward_backward_gap_worst_rank_p50_s": 0.125,
        "queue_ready_inter_forward_backward_gap_worst_rank_p95_s": 0.1385,
        "queue_ready_inter_forward_backward_gap_worst_rank_max_s": 0.14,
        "queue_ready_inter_forward_backward_gap_worst_rank_count": 4,
        "mean_train_gap_s": 0.5,
        "stable_vllm_pressure": 0.6,
        "stable_trainer_underfeed": 0.07,
        "post_warmup_policy_activation_count": 6,
        "mean_policy_activation_lag_s": 2.5 / 6.0,
        "p50_policy_activation_lag_s": 0.25,
        "p95_policy_activation_lag_s": 1.0,
        "max_policy_activation_lag_s": 1.25,
        "mean_policy_activation_interval_s": 11.75 / 6.0,
        "p50_policy_activation_interval_s": 2.0,
        "p95_policy_activation_interval_s": 2.75,
        "second_max_policy_activation_interval_s": 2.0,
        "max_policy_activation_interval_s": 3.0,
    }
    assert {key: measurements[key] for key in expected} == pytest.approx(expected)
    thresholds = ThroughputThresholds(
        calibration_basis="measured",
        calibration_fingerprint="a" * 64,
        min_isolated_train_tok_s=1.0,
        min_e2e_train_tok_s=1.0,
        min_accepted_train_tok_s=1.0,
        min_e2e_to_isolated_ratio=0.5,
        min_matched_core_to_isolated_ratio=0.95,
        max_mean_policy_activation_lag_s=1.5,
        max_policy_activation_lag_s=2.0,
        max_repeated_policy_activation_interval_s=1.5,
    )
    assert acceptance_failures(measurements, config, thresholds) == [
        "unused_and_dummy_ratio",
        "repeated_policy_activation_cadence_s",
    ]
    robust = {
        **measurements,
        "queue_ready_inter_forward_backward_gap_worst_rank_max_s": 0.5,
    }
    assert "queue_ready_inter_forward_backward_gap_p50_s" not in acceptance_failures(
        robust, config, thresholds
    )
    assert "queue_ready_inter_forward_backward_gap_p50_s" not in acceptance_failures(
        {
            **measurements,
            "queue_ready_inter_forward_backward_gap_worst_rank_p50_s": 0.225,
        },
        config,
        thresholds,
    )
    assert "queue_ready_inter_forward_backward_gap_max_s" in acceptance_failures(
        {
            **measurements,
            "queue_ready_inter_forward_backward_gap_worst_rank_max_s": 1.01,
        },
        config,
        thresholds,
    )
    sparse = {
        **measurements,
        "queue_ready_inter_forward_backward_gap_worst_rank_count": 2,
    }
    assert "queue_ready_inter_forward_backward_gap_count" in acceptance_failures(
        sparse, config, thresholds
    )
    with pytest.raises(ValueError):
        ThroughputThresholds.model_validate(
            {
                **thresholds.model_dump(),
                "max_queue_ready_inter_forward_backward_gap_p50_s": 0.231,
            }
        )
    assert acceptance_failures(
        {
            **measurements,
            "stable_vllm_pressure": 0.49,
            "stable_trainer_underfeed": 0.09,
        },
        config,
        thresholds,
    ) == [
        "stable_min_vllm_pressure",
        "stable_trainer_underfeed",
        "unused_and_dummy_ratio",
        "repeated_policy_activation_cadence_s",
    ]
    estimated = ThroughputThresholds(
        calibration_basis="estimated",
        min_isolated_train_tok_s=1.0,
        min_e2e_train_tok_s=1.0,
        min_accepted_train_tok_s=1.0,
        min_e2e_to_isolated_ratio=0.5,
        min_matched_core_to_isolated_ratio=0.95,
        max_mean_policy_activation_lag_s=1.5,
        max_policy_activation_lag_s=2.0,
        max_repeated_policy_activation_interval_s=1.5,
    )
    assert acceptance_failures(measurements, config, estimated) == [
        "unused_and_dummy_ratio",
        "repeated_policy_activation_cadence_s",
        "calibration_basis",
    ]
    lag_failures = acceptance_failures(
        measurements,
        config,
        thresholds.model_copy(
            update={
                "max_mean_policy_activation_lag_s": 0.35,
                "max_policy_activation_lag_s": 1.0,
                "max_repeated_policy_activation_interval_s": 3.5,
            }
        ),
    )
    assert lag_failures == [
        "unused_and_dummy_ratio",
        "mean_policy_activation_lag_s",
        "max_policy_activation_lag_s",
    ]
    measurements["matched_core_to_isolated_ratio"] *= 1.1
    assert "matched_core_to_isolated_ratio_max" in acceptance_failures(
        measurements, config, thresholds
    )
    inconsistent = [dict(row) for row in rows]
    next(row for row in inconsistent if row["step"] == config.max_steps)[
        "pipeline_settings/num_rollout_workers"
    ] = 14
    history_path.write_text("".join(json.dumps(row) + "\n" for row in inconsistent))
    with pytest.raises(RuntimeError, match="two trailing settled execution"):
        collect(isolated_phase)
    history_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    capture_settings = measured_settings.model_dump(mode="json")
    capture_settings["num_rollout_workers"] = 14
    with pytest.raises(
        RuntimeError, match="did not use the measured pipeline settings"
    ):
        _collect_measurements(
            fixture=fixture,
            config=config,
            hardware="b300",
            model_output_dir=tmp_path,
            profile=profile,
            events=events,
            isolated=isolated_phase,
            e2e=e2e_phase,
            capture_settings=capture_settings,
            calibration_fingerprint="a" * 64,
        )
    fractional = [dict(row) for row in rows]
    next(row for row in fractional if row["step"] == 2)[
        "data/step_nonpadding_logical_tokens"
    ] = 999.5
    history_path.write_text("".join(json.dumps(row) + "\n" for row in fractional))
    with pytest.raises(RuntimeError, match="must be a nonnegative integer"):
        collect(isolated_phase)
    history_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="same packed input"):
        collect(phase("isolated", "input-b", (9, 10, 11)))


def test_throughput_classification_is_fail_closed() -> None:
    assert _classify_acceptance_failures([])["acceptance_status"] == "accepted"
    load = _classify_acceptance_failures(
        ["stable_trainer_underfeed", "accepted_train_tok_s"]
    )
    assert load["acceptance_status"] == "load_inconclusive"
    assert load["load_failures"] == ["stable_trainer_underfeed"]
    assert load["performance_failures"] == ["accepted_train_tok_s"]
    for hard_failure in (
        "calibration_fingerprint",
        "unused_and_dummy_ratio",
        "window_4_5_policy_age_p95",
    ):
        classified = _classify_acceptance_failures(
            ["stable_min_vllm_pressure", hard_failure]
        )
        assert classified["acceptance_status"] == "rejected"
        assert classified["hard_failures"] == [hard_failure]
    future = _classify_acceptance_failures(
        ["stable_min_vllm_pressure", "future_acceptance_gate"]
    )
    assert future["acceptance_status"] == "rejected"
    assert future["unclassified_failures"] == ["future_acceptance_gate"]


def test_throughput_retry_is_bounded_and_preserves_attempts(
    tmp_path: Path,
) -> None:
    plans = [
        ([[]], "accepted"),
        ([["e2e_train_tok_s"], []], "accepted"),
        ([["e2e_train_tok_s"], ["e2e_train_tok_s"]], "rejected"),
        ([["stable_min_vllm_pressure", "calibration_basis"]], "rejected"),
        ([["future_acceptance_gate"]], "rejected"),
        ([["stable_min_vllm_pressure"], []], "accepted"),
        (
            [["stable_min_vllm_pressure"], ["stable_trainer_underfeed"]],
            "load_inconclusive",
        ),
    ]
    for case, (failures_by_attempt, expected_status) in enumerate(plans):
        stage_dir = tmp_path / str(case)
        calls: list[int] = []

        def run_attempt(attempt: int, artifact_dir: Path) -> ValidationStageResult:
            calls.append(attempt)
            (artifact_dir / "complete.txt").write_text(str(attempt))
            runtime_dir = artifact_dir / "optimizer_states"
            runtime_dir.mkdir()
            (runtime_dir / "state.pt").write_bytes(b"state")
            classification = _classify_acceptance_failures(
                failures_by_attempt[attempt - 1]
            )
            return ValidationStageResult(
                name="e2e_throughput",
                passed=classification["acceptance_status"] == "accepted",
                metrics={"selected_metric": attempt, **classification},
                artifact_dir=str(artifact_dir),
            )

        result = _run_throughput_attempts(stage_dir, run_attempt)
        expected_calls = len(failures_by_attempt)
        assert calls == list(range(1, expected_calls + 1))
        assert result.metrics["acceptance_status"] == expected_status
        assert result.metrics["selected_metric"] == expected_calls
        assert result.metrics["throughput_attempt_count"] == expected_calls
        assert all(
            (stage_dir / f"attempt_{attempt}" / "complete.txt").is_file()
            for attempt in calls
        )
        assert not any(stage_dir.rglob("optimizer_states"))


def test_throughput_measurement_freezes_actual_settings() -> None:
    measured = PipelineTuneSettings(
        num_rollout_workers=16,
        min_batch_size=8,
        max_batch_size=32,
        queue_maxsize=48,
        target_groups_per_step=24,
    )
    future = measured.model_copy(update={"num_rollout_workers": 14})
    trainer = SimpleNamespace(
        state=SimpleNamespace(next_training_step=18),
        **measured.model_dump(mode="python"),
    )

    def apply(settings: PipelineTuneSettings) -> None:
        for name, value in settings.model_dump(mode="python").items():
            setattr(trainer, name, value)

    trainer.apply_pipeline_settings = apply
    original = trainer.apply_pipeline_settings

    with _freeze_pipeline_settings_from_step(trainer, 19):
        trainer.apply_pipeline_settings(measured)
        trainer.state.next_training_step = 19
        trainer.apply_pipeline_settings(future)
        trainer.state.next_training_step = 20
        trainer.apply_pipeline_settings(future)
        trainer.state.next_training_step = 21
        trainer.apply_pipeline_settings(future)
        assert _current_pipeline_settings(trainer) == measured.model_dump(mode="json")

    trainer.apply_pipeline_settings(future)
    assert _current_pipeline_settings(trainer) == future.model_dump(mode="json")
    assert trainer.apply_pipeline_settings == original


def test_throughput_measurement_uses_settled_execution_suffix() -> None:
    measured = PipelineTuneSettings(
        num_rollout_workers=16,
        min_batch_size=24,
        max_batch_size=36,
        queue_maxsize=48,
        target_groups_per_step=31,
    )
    previous = measured.model_copy(
        update={
            "num_rollout_workers": 14,
            "min_batch_size": 27,
            "max_batch_size": 27,
            "target_groups_per_step": 27,
        }
    )

    def decision(start_step: int) -> SimpleNamespace:
        return SimpleNamespace(
            stats=SimpleNamespace(
                start_step=start_step,
                end_step=start_step + 1,
                window_start_s=float(start_step),
                window_end_s=float(start_step + 2),
            )
        )

    def row(
        settings: PipelineTuneSettings,
        step: int,
        packed_length: int,
        *,
        submitted: int | None = None,
    ) -> dict[str, int | float]:
        return {
            **{
                f"pipeline_settings/{name}": value
                for name, value in settings.model_dump(mode="json").items()
            },
            "queue/packing_policy_lag_steps": 1,
            "data/step_num_groups_submitted": (
                settings.target_groups_per_step if submitted is None else submitted
            ),
            "data/step_packed_sequences": 1,
            "data/step_num_gradient_steps": 1,
            "pipeline/global_real_microbatches": 1,
            "pipeline/global_dummy_microbatches": 0,
            "pipeline/packed_sequence_length": packed_length,
            "time/step_train_s": float(step),
        }

    by_step = {
        **{step: row(previous, step, 56_832) for step in range(3, 6)},
        6: row(measured, 6, 56_832, submitted=27),
        **{step: row(measured, step, 64_512) for step in range(7, 12)},
    }
    selected = _settled_execution_decision_suffix(
        [decision(4), decision(6), decision(8), decision(10)],
        by_step,
    )

    assert [item.stats.start_step for item in selected] == [8, 10]

    alternating = dict(by_step)
    for step, packed_length in zip(range(6, 12), (60_000, 64_000) * 3, strict=True):
        alternating[step] = row(measured, step, packed_length)
    assert [
        item.stats.start_step
        for item in _settled_execution_decision_suffix(
            [decision(4), decision(6), decision(8), decision(10)], alternating
        )
    ] == [8, 10]

    timing_changed = {
        step: {**values, "time/step_train_s": 1000.0 - step}
        for step, values in alternating.items()
    }
    assert [
        item.stats.start_step
        for item in _settled_execution_decision_suffix(
            [decision(4), decision(6), decision(8), decision(10)], timing_changed
        )
    ] == [8, 10]

    unique = dict(alternating)
    for step in range(8, 12):
        unique[step] = row(measured, step, 70_000 + step)
    with pytest.raises(RuntimeError, match="two trailing settled execution"):
        _settled_execution_decision_suffix(
            [decision(4), decision(6), decision(8), decision(10)], unique
        )


def test_throughput_packed_input_fingerprint_hashes_data_plane_bytes() -> None:
    from array import array
    from multiprocessing import shared_memory

    from art.pipeline_tuner.config import PackedGroupShape, PackingLeafShape

    shm = shared_memory.SharedMemory(create=True, size=4)
    try:
        buffer = shm.buf
        assert buffer is not None
        buffer[:] = b"abcd"
        tensor = SimpleNamespace(offset=0, byte_count=4)
        ref = SimpleNamespace(
            shared_memory_name=shm.name,
            owner_process_id=os.getpid(),
            tensors=(tensor,),
            model_dump=lambda **kwargs: {
                "tensors": [{"name": "tokens", "shape": [4], "dtype": "int8"}]
            },
        )
        packed = SimpleNamespace(
            leases=SimpleNamespace(ref=ref),
            packed_group_shapes=(
                PackedGroupShape(
                    leaves=(
                        PackingLeafShape(
                            token_ids=array("I", [1, 2, 3]), shareable_length=2
                        ),
                    )
                ),
            ),
        )
        batch = SimpleNamespace(
            payload=SimpleNamespace(packed=packed),
            model_dump=lambda **kwargs: {"sequence_length": 4},
        )
        prepared = SimpleNamespace(
            batch=batch,
            packing_config=SimpleNamespace(
                model_dump=lambda **kwargs: {"packed_sequence_length": 4}
            ),
        )
        before = _packed_batch_fingerprint(prepared)
        buffer[0] = ord("z")
        changed_bytes = _packed_batch_fingerprint(prepared)
        buffer[0] = ord("a")
        packed.packed_group_shapes = (
            PackedGroupShape(
                leaves=(
                    PackingLeafShape(
                        token_ids=array("I", [1, 2, 4]), shareable_length=2
                    ),
                )
            ),
        )
        changed_shape = _packed_batch_fingerprint(prepared)

        assert before != changed_bytes
        assert before != changed_shape
    finally:
        del buffer
        shm.close()
        shm.unlink()


def _without_stage_duration(stage: ValidationStageResult) -> dict[str, object]:
    metrics = dict(stage.metrics)
    assert float(metrics.pop("workflow_stage_duration_s")) >= 0.0
    metrics.pop("fixture_provisioning_s", None)
    metrics.pop("workflow_pruned_runtime_artifact_dirs", None)
    metrics.pop("workflow_pruned_runtime_artifact_bytes", None)
    metrics.pop("workflow_pruned_runtime_filesystem_bytes", None)
    return metrics


def test_runtime_pruning_retains_vllm_log(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "attempt" / "megatron_runtime"
    log_path = runtime_dir / "logs" / "vllm-runtime.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("runtime failure\n")

    metrics = _prune_runtime_artifacts(tmp_path)

    assert metrics["workflow_pruned_runtime_artifact_dirs"] == 1
    assert metrics["workflow_pruned_runtime_artifact_bytes"] > 0
    assert isinstance(metrics["workflow_pruned_runtime_filesystem_bytes"], int)
    assert not runtime_dir.exists()
    assert (
        tmp_path
        / "retained_runtime_logs"
        / "attempt"
        / "megatron_runtime"
        / "logs"
        / "vllm-runtime.log"
    ).read_text() == "runtime failure\n"


def test_build_validation_stage_names_has_fixed_order() -> None:
    assert build_validation_stage_names() == list(MANDATORY_VALIDATION_STAGES)


def test_throughput_prompt_bounds_shared_context_before_scenario_identity() -> None:
    tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **_kwargs: list(
            range(len(messages[0]["content"].split()))
        )
    )

    prompt = _sized_prompt(tokenizer, target_tokens=64)

    first_context = prompt.find("measured context item")
    scenario = prompt.find("Throughput scenario 00000000.")
    last_context = prompt.rfind("measured context item")

    assert 0 <= first_context < scenario < last_context


def test_gpt_oss_random_fixture_cannot_open_a_harmony_message() -> None:
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _token: 200008)

    assert _synthetic_sampling_constraints(tokenizer, model_key="gpt_oss_moe") == {
        "logit_bias": {"200008": -100.0}
    }
    assert _synthetic_sampling_constraints(tokenizer, model_key="qwen3_moe") == {}


def test_validated_architecture_representative_models_are_fixed() -> None:
    assert validated_architecture_representative_models() == [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-27B",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
        "deepseek-ai/DeepSeek-V4-Flash",
        "zai-org/GLM-5.2",
        "openai/gpt-oss-20b",
    ]


def test_qwen38_uses_its_measured_throughput_fingerprint() -> None:
    qwen35 = handler_workflow_resources_for_base_model("Qwen/Qwen3.5-27B")
    qwen38 = handler_workflow_resources_for_base_model("Qwen/Qwen3.8-27B")
    assert qwen35 is not None and qwen35.e2e_throughput is not None
    assert qwen38 is not None and qwen38.e2e_throughput is not None
    qwen35_config = qwen35.e2e_throughput.throughput
    qwen38_config = qwen38.e2e_throughput.throughput
    assert qwen35_config is not None and qwen38_config is not None
    assert qwen35_config.thresholds["b300"].calibration_fingerprint == (
        "89adaa0867956c739fbc4d3ced5a8589db9d4b14d6528caccc57c586fcd089aa"
    )
    assert qwen38_config.thresholds["b300"].calibration_fingerprint == (
        "b07ee7ec6338ec021463a43a90fc96c5c5a036b4a04d90b80e1d22c1eef86774"
    )


def test_dsv4_runtime_stages_use_full_model_resources() -> None:
    resources = handler_workflow_resources_for_base_model(
        "deepseek-ai/DeepSeek-V4-Flash"
    )
    assert resources is not None
    for stage in (
        resources.train_inf_mismatch,
        resources.length_trainability,
    ):
        assert stage is not None
        assert stage.required_world_size == 8
        assert stage.requires_external_vllm is True
        assert stage.megatron is not None
        assert stage.megatron.gpu_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert stage.megatron.topology.tp == 2
        assert stage.megatron.topology.ep == 8
        assert stage.megatron.topology.cp == 1
        assert stage.vllm is not None
        assert stage.vllm.gpu_ids == [4, 5, 6, 7]
        engine_args = stage.vllm.engine_args()
        assert "hf_overrides" not in engine_args
        assert engine_args.get("load_format") != "dummy"
        assert engine_args["moe_backend"] == "auto"
        assert engine_args["kv_cache_dtype"] == "fp8"
        assert stage.streaming_weight_offload is True
        assert stage.megatron_env == {}


@pytest.mark.parametrize(
    ("stage_name", "trainer_gpu_ids", "trainer_ep", "trainer_dp"),
    [
        ("train_inf_mismatch", [0, 1, 2, 3], 4, 2),
        ("length_trainability", [0, 1, 2, 3], 4, 2),
    ],
)
def test_dsv4_resources_remap_to_four_high_vram_gpus(
    monkeypatch,
    stage_name: str,
    trainer_gpu_ids: list[int],
    trainer_ep: int,
    trainer_dp: int,
) -> None:
    resources = handler_workflow_resources_for_base_model(
        "deepseek-ai/DeepSeek-V4-Flash"
    )
    assert resources is not None
    stage_resources = getattr(resources, stage_name)
    assert stage_resources is not None
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow_resources."
        "_visible_h200_equivalent_gpus",
        lambda *, visible_gpu_count: 8,
    )

    stage = resolve_stage_resources_for_visible_gpus(
        stage_name,
        stage_resources,
        visible_gpu_count=4,
    )

    assert stage.megatron is not None
    assert stage.vllm is not None
    assert stage.megatron.gpu_ids == trainer_gpu_ids
    assert stage.megatron.topology.tp == 2
    assert stage.megatron.topology.ep == trainer_ep
    assert stage.megatron.topology.dp == trainer_dp
    assert stage.vllm.gpu_ids == [2, 3]
    assert stage.vllm.tensor_parallel_size == 2
    assert stage.vllm.engine_args()["moe_backend"] == "auto"
    assert stage.vllm.engine_args()["kv_cache_dtype"] == "fp8"


def test_glm52_reduced_workflow_uses_portable_serving_backends() -> None:
    resources = handler_workflow_resources_for_base_model("zai-org/GLM-5.2")
    assert resources is not None
    joint_stages = (
        resources.train_inf_mismatch,
        resources.length_trainability,
    )
    for stage in joint_stages:
        assert stage is not None
        assert stage.required_world_size == 2
        assert stage.megatron is not None
        assert stage.megatron.gpu_ids == [0]
    for stage in joint_stages:
        assert stage is not None
        assert stage.vllm is not None
        assert stage.vllm.gpu_ids == [1]
        engine_args = stage.vllm.engine_args()
        assert engine_args["attention_backend"] == "FLASHMLA_SPARSE"
        assert engine_args["max_model_len"] == 1024
        assert engine_args["moe_backend"] == "triton"


@pytest.mark.parametrize("handler_key", sorted(HANDLER_WORKFLOW_RESOURCES))
def test_throughput_requires_four_distinct_physical_gpus(
    handler_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = HANDLER_WORKFLOW_RESOURCES[handler_key].e2e_throughput
    assert stage is not None
    megatron, vllm = stage.megatron, stage.vllm
    assert megatron is not None and vllm is not None
    assert (stage.required_world_size, stage.required_physical_gpus) == (4, 4)
    assert (megatron.gpu_ids, vllm.gpu_ids) == ([0, 1], [2, 3])
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow_resources."
        "_visible_h200_equivalent_gpus",
        lambda *, visible_gpu_count: visible_gpu_count * 2,
    )

    with pytest.raises(RuntimeError, match="Need 4 physical GPUs"):
        resolve_stage_resources_for_visible_gpus(
            "e2e_throughput",
            stage,
            visible_gpu_count=2,
        )

    assert (
        resolve_stage_resources_for_visible_gpus(
            "e2e_throughput", stage, visible_gpu_count=4
        )
        == stage
    )


def test_backend_resources_stay_logical_until_topology_compilation(monkeypatch) -> None:
    from art.megatron.runtime import local as local_runtime

    stage = HANDLER_WORKFLOW_RESOURCES["llama3_dense"].e2e_throughput
    assert stage is not None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow_resources."
        "_current_visible_gpu_count",
        lambda: 4,
    )

    resolved = resolve_stage_resources_for_current_host("e2e_throughput", stage)

    megatron = resolved.megatron
    vllm = resolved.vllm
    assert megatron is not None
    assert vllm is not None
    assert megatron.gpu_ids == [0, 1]
    assert vllm.gpu_ids == [2, 3]
    monkeypatch.setattr(
        local_runtime,
        "get_megatron_runtime_config",
        lambda: SimpleNamespace(topology=megatron.topology.to_megatron_config()),
    )
    topology = local_runtime.compile_local_runtime_topology(
        cast(
            Any,
            {
                "trainer_gpu_ids": megatron.gpu_ids,
                "inference_gpu_ids": vllm.gpu_ids,
                "engine_args": vllm.engine_args(),
            },
        ),
        model_name="throughput",
        base_model="/tmp/provider",
        artifact_root="/tmp/art",
        visible_gpu_count=4,
    )

    assert topology.trainer is not None
    assert [rank.gpu_id for rank in topology.trainer.ranks] == [4, 5]
    assert topology.model_services[0].members[0].gpu_ids == (6, 7)
    assert topology.cluster.hosts[0].gpu_ids == (4, 5, 6, 7)


def test_inspect_architecture_for_workflow_uses_minimal_topology(monkeypatch) -> None:
    seen_env: dict[str, str | None] = {}

    def _inspect_architecture(base_model: str, **kwargs) -> ArchitectureReport:
        del kwargs
        seen_env.update(
            {
                "tp": os.environ.get("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE"),
                "cp": os.environ.get("ART_MEGATRON_CONTEXT_PARALLEL_SIZE"),
                "ep": os.environ.get("ART_MEGATRON_EXPERT_MODEL_PARALLEL_SIZE"),
                "etp": os.environ.get("ART_MEGATRON_EXPERT_TENSOR_PARALLEL_SIZE"),
            }
        )
        return ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_dense",
            handler_key="qwen3_dense",
            layer_families=[LayerFamilyInstance(key="standard_attention", count=1)],
            recommended_min_layers=1,
        )

    monkeypatch.setattr(
        "art.megatron.model_support.discovery.inspect_architecture",
        _inspect_architecture,
    )

    _inspect_architecture_for_workflow(
        "Qwen/Qwen3-32B",
        allow_unvalidated_arch=True,
    )

    assert seen_env == {"tp": "1", "cp": "1", "ep": "1", "etp": "1"}


def test_build_validation_report_captures_hf_parity_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "art.megatron.model_support.discovery.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[],
            recommended_min_layers=4,
        ),
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.detect_dependency_versions",
        lambda: {},
    )

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._run_stage_in_subprocess",
        lambda *, stage_name, base_model, architecture, allow_unvalidated_arch=False: (
            ValidationStageResult(
                name="hf_parity",
                passed=False,
                metrics={"error": "AssertionError: parity failed"},
            )
            if stage_name == "hf_parity"
            else ValidationStageResult(
                name=stage_name,
                passed=True,
                metrics={},
            )
        ),
    )

    report = build_validation_report(base_model="Qwen/Qwen3.5-35B-A3B")

    hf_parity_stage = next(
        stage for stage in report.stages if stage.name == "hf_parity"
    )
    assert hf_parity_stage.passed is False
    assert _without_stage_duration(hf_parity_stage) == {
        "error": "AssertionError: parity failed"
    }
    assert hf_parity_stage.artifact_dir is None


def test_run_lora_coverage_stage_reports_missing_targets(monkeypatch) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        layer_families=[LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0)],
        recommended_min_layers=4,
    )
    oracle_module = SimpleNamespace(
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    coverage_report = SimpleNamespace(
        missing_wrapped_target_modules=["in_proj_z"],
        missing_exported_target_modules=[],
        unexpected_trainable_parameter_names=[],
        model_dump=lambda mode="json": {
            "base_model": "Qwen/Qwen3.5-35B-A3B",
            "missing_wrapped_target_modules": ["in_proj_z"],
        },
    )
    coverage_module = SimpleNamespace(
        run_lora_coverage=lambda case_config: coverage_report
    )

    def _import_integration_module(name: str):
        if name == "integration.megatron.model_support.oracle_harness":
            return oracle_module
        if name == "integration.megatron.model_support.lora_coverage":
            return coverage_module
        raise AssertionError(name)

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        _import_integration_module,
    )

    stage = run_lora_coverage_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=architecture,
    )

    assert stage.name == "lora_coverage"
    assert stage.passed is False
    assert stage.metrics == {
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "missing_wrapped_target_modules": ["in_proj_z"],
    }
