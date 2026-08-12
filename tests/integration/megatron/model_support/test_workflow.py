import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from art.megatron.model_support.spec import (
    ArchitectureReport,
    LayerFamilyInstance,
)
from art.pipeline_tuner import PipelineTuneSettings
from tests.integration.megatron.train_inf_mismatch import (
    workflow_stage as mismatch_workflow_stage,
)

from .validation_spec import ValidationReport, ValidationStageResult
from .workflow import (
    _RUNTIME_ARTIFACT_DIR_NAMES,
    INCLUDE_FLASH_SENSITIVITY_ENV,
    KEEP_TOPOLOGY_ARTIFACTS_ENV,
    MANDATORY_VALIDATION_STAGES,
    NATIVE_VLLM_LORA_STAGE,
    SKIP_SENSITIVITY_ENV,
    WORKFLOW_STAGE_DIR_ENV,
    _inspect_architecture_for_workflow,
    _prune_runtime_artifacts,
    assess_minimal_layer_coverage,
    build_all_architectures_validation_report,
    build_validation_report,
    build_validation_stage_names,
    run_chat_template_rollout_stage,
    run_correctness_sensitivity_stage,
    run_length_trainability_stage,
    run_lora_coverage_stage,
    run_merged_vllm_serving_stage,
    run_native_vllm_lora_stage,
    run_packing_invariance_stage,
    run_train_inf_mismatch_stage,
    run_yes_no_trainability_stage,
    validated_architecture_representative_models,
)
from .workflow_fixtures import (
    FIXTURE_PATH_ENV,
    WorkflowFixture,
    _validate_tokenizer_compatible_fixture,
)
from .workflow_resources import (
    _THROUGHPUT_CONFIGS,
    HANDLER_WORKFLOW_RESOURCES,
    ThroughputThresholds,
    ThroughputWorkflowConfig,
    _h200_equivalent_slots_for_total_gib,
    handler_workflow_resources_for_base_model,
    resolve_stage_resources_for_current_host,
    resolve_stage_resources_for_visible_gpus,
)
from .workflow_throughput import (
    PolicyActivationEvent,
    ThroughputFixture,
    _collect_matched_packing_shapes,
    _collect_measurements,
    _current_pipeline_settings,
    _environment_provenance,
    _freeze_pipeline_settings_from_step,
    _packed_input_fingerprint,
    _phase_evidence,
    _reduced_config,
    _same_setting_decision_suffix,
    _throughput_config_for_hardware,
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


def test_fixture_stage_contracts(tmp_path: Path) -> None:
    # fmt: off
    cases = {
        ("gemma4_dense", "canonical"): ("hf_parity", "packing_invariance"),
        ("gemma4_dense", "compact"): ("lora_coverage",),
        ("gemma4_dense", "functional"): ("train_inf_mismatch", "merged_vllm_serving", "native_vllm_lora", "length_trainability"),
        ("gemma4_dense", "tokenizer"): ("yes_no_trainability",),
        ("llama3_dense", "compact"): ("hf_parity",),
        ("llama3_dense", "functional"): ("train_inf_mismatch", "merged_vllm_serving", "native_vllm_lora", "length_trainability"),
        ("llama3_dense", "canonical"): ("yes_no_trainability",),
        ("gpt_oss_moe", "functional"): ("train_inf_mismatch", "merged_vllm_serving", "native_vllm_lora", "length_trainability"),
        ("glm52", "functional"): ("train_inf_mismatch", "merged_vllm_serving", "native_vllm_lora", "length_trainability"),
        ("glm52", "compact"): ("yes_no_trainability",),
        ("dsv4", "functional"): ("train_inf_mismatch", "merged_vllm_serving", "native_vllm_lora", "length_trainability"),
        ("dsv4", "canonical"): ("yes_no_trainability",),
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
        ("glm52", "yes_no_trainability", "9829,902,36569"),
        ("gemma4_dense", "yes_no_trainability", "4443,951,7463"),
        ("gemma4_moe", "yes_no_trainability", "4443,951,7463"),
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
            "pipeline_settings/num_rollout_workers": 16,
            "pipeline_settings/min_batch_size": 8,
            "pipeline_settings/max_batch_size": 32,
            "pipeline_settings/queue_maxsize": 48,
            "pipeline_settings/target_groups_per_step": 24,
            "time/step_train_s": 1.5,
            "time/step_wall_s": 2.0,
            "time/step_collect_batch_s": 0.001068115234375,
            "queue/packed_get_wait_s": 0.1 if step >= 7 else 0.001,
            "queue/packed_queue_depth": 0.0 if step == 6 else 1.0,
            "time/inter_forward_backward_gap_rank_0_s": (
                1.0 if step >= 6 else 0.1 + (step - 2) * 0.01
            ),
            "time/inter_forward_backward_gap_rank_1_s": (
                2.0 if step >= 6 else 0.11 + (step - 2) * 0.01
            ),
            "offpolicy/token_weighted_policy_age_steps": 1.0,
            "offpolicy/token_weighted_policy_age_p95_steps": 2.0,
            "sample_efficiency/freshness_discount": 0.8,
            "discarded/step/stale_groups": 0,
            "discarded/step/zero_variance_groups": 0,
        }
        for step in range(2, 10)
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
    config = ThroughputWorkflowConfig(num_layers=2, completion_tokens=128)
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
        phase("e2e", "input-a", (7, 8, 9)),
        phase("isolated", "input-a", (11, 12, 13)),
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
        "original_trajectory_tokens": 32_000,
        "nonpadding_logical_tokens": 8_000,
        "loss_bearing_tokens": 4_000,
        "accepted_train_tokens": 4_000,
        "isolated_train_tok_s": 1_000 / 1.5,
        "matched_e2e_core_train_tok_s": 1_000 / 1.5,
        "matched_core_to_isolated_ratio": 1.0,
        "e2e_core_train_tok_s": 8_000 / 12.0,
        "e2e_train_tok_s": 500.0,
        "accepted_train_tok_s": 250.0,
        "queue_ready_inter_forward_backward_gap_rank_zero_p50_s": 0.115,
        "queue_ready_inter_forward_backward_gap_rank_zero_p95_s": 0.1285,
        "queue_ready_inter_forward_backward_gap_rank_zero_max_s": 0.13,
        "queue_ready_inter_forward_backward_gap_rank_zero_count": 4,
        "queue_ready_inter_forward_backward_gap_worst_rank": 1,
        "queue_ready_inter_forward_backward_gap_worst_rank_p50_s": 0.125,
        "queue_ready_inter_forward_backward_gap_worst_rank_p95_s": 0.1385,
        "queue_ready_inter_forward_backward_gap_worst_rank_max_s": 0.14,
        "queue_ready_inter_forward_backward_gap_worst_rank_count": 4,
        "mean_train_gap_s": 0.5,
        "stable_vllm_pressure": 0.6,
        "stable_trainer_underfeed": 0.07,
        "post_warmup_policy_activation_count": 8,
        "mean_policy_activation_lag_s": 3.0 / 8.0,
        "p50_policy_activation_lag_s": 0.25,
        "p95_policy_activation_lag_s": 0.9,
        "max_policy_activation_lag_s": 1.25,
        "mean_policy_activation_interval_s": 14.75 / 8.0,
        "p50_policy_activation_interval_s": 2.0,
        "p95_policy_activation_interval_s": 2.65,
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
        "repeated_policy_activation_cadence_s"
    ]
    robust = {
        **measurements,
        "queue_ready_inter_forward_backward_gap_worst_rank_max_s": 0.5,
    }
    assert "queue_ready_inter_forward_backward_gap_p95_s" not in acceptance_failures(
        robust, config, thresholds
    )
    sparse = {
        **measurements,
        "queue_ready_inter_forward_backward_gap_worst_rank_count": 3,
    }
    assert "queue_ready_inter_forward_backward_gap_count" in acceptance_failures(
        sparse, config, thresholds
    )
    with pytest.raises(ValueError):
        ThroughputThresholds.model_validate(
            {
                **thresholds.model_dump(),
                "max_queue_ready_inter_forward_backward_gap_p95_s": 0.201,
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
        "mean_policy_activation_lag_s",
        "max_policy_activation_lag_s",
    ]
    measurements["matched_core_to_isolated_ratio"] *= 1.1
    assert "matched_core_to_isolated_ratio_max" in acceptance_failures(
        measurements, config, thresholds
    )
    inconsistent = [dict(row) for row in rows]
    inconsistent[-1]["pipeline_settings/num_rollout_workers"] = 14
    history_path.write_text("".join(json.dumps(row) + "\n" for row in inconsistent))
    with pytest.raises(RuntimeError, match="two trailing same-setting"):
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
    fractional[0]["data/step_nonpadding_logical_tokens"] = 999.5
    history_path.write_text("".join(json.dumps(row) + "\n" for row in fractional))
    with pytest.raises(RuntimeError, match="must be a nonnegative integer"):
        collect(isolated_phase)
    history_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(RuntimeError, match="same packed input"):
        collect(phase("isolated", "input-b", (9, 10, 11)))


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


def test_throughput_measurement_uses_full_same_setting_suffix() -> None:
    measured = PipelineTuneSettings(
        num_rollout_workers=16,
        min_batch_size=8,
        max_batch_size=32,
        queue_maxsize=48,
        target_groups_per_step=24,
    )
    previous = measured.model_copy(update={"num_rollout_workers": 14})

    def decision(start_step: int) -> SimpleNamespace:
        return SimpleNamespace(
            stats=SimpleNamespace(
                start_step=start_step,
                end_step=start_step + 1,
                window_start_s=float(start_step),
                window_end_s=float(start_step + 2),
            )
        )

    def rows(settings: PipelineTuneSettings, *steps: int) -> dict[int, dict[str, int]]:
        values = settings.model_dump(mode="json")
        return {
            step: {f"pipeline_settings/{name}": value for name, value in values.items()}
            for step in steps
        }

    by_step = {
        **rows(previous, 10, 11),
        **rows(measured, 12, 13, 14, 15, 16, 17),
    }
    selected = _same_setting_decision_suffix(
        [decision(10), decision(12), decision(14), decision(16)],
        by_step,
    )

    assert [item.stats.start_step for item in selected] == [12, 14, 16]


def test_throughput_provenance_ignores_local_editable_paths() -> None:
    distributions = _environment_provenance(
        Path(sys.executable), ("openpipe-art", "pydantic")
    )["distributions"]

    assert set(distributions["openpipe-art"]) == {"version", "metadata_sha256"}
    assert {"direct_url_sha256", "record_sha256"} <= set(distributions["pydantic"])


def test_dsv4_throughput_reduction_preserves_hash_moe_prefix() -> None:
    source = {
        "num_hidden_layers": 10,
        "hidden_size": 2048,
        "mlp_layer_types": ["hash_moe"] * 3 + ["moe"] * 7,
    }

    reduced, _ = _reduced_config(source, model_key="dsv4", num_layers=8)

    assert "num_hash_layers" not in reduced
    assert reduced["mlp_layer_types"] == ["hash_moe"] * 3 + ["moe"] * 5


def test_matched_batch_always_collects_packing_shapes() -> None:
    groups = [SimpleNamespace(_collect_packing_shape=False) for _ in range(3)]

    _collect_matched_packing_shapes(groups)

    assert all(group._collect_packing_shape for group in groups)


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
        groups = [SimpleNamespace(_prepared_training_batch=prepared)]

        before = _packed_input_fingerprint(groups)
        buffer[0] = ord("z")
        changed_bytes = _packed_input_fingerprint(groups)
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
        changed_shape = _packed_input_fingerprint(groups)

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
    return metrics


def test_build_validation_stage_names_has_fixed_order() -> None:
    assert build_validation_stage_names() == list(MANDATORY_VALIDATION_STAGES)
    assert build_validation_stage_names(include_native_vllm_lora=True) == [
        *MANDATORY_VALIDATION_STAGES,
        NATIVE_VLLM_LORA_STAGE,
    ]
    assert build_validation_stage_names(native_vllm_lora_status="wip") == [
        *MANDATORY_VALIDATION_STAGES,
        NATIVE_VLLM_LORA_STAGE,
    ]
    assert build_validation_stage_names(include_yes_no_trainability=True) == [
        *MANDATORY_VALIDATION_STAGES,
        "yes_no_trainability",
    ]


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


def test_dsv4_runtime_stages_use_full_model_resources() -> None:
    resources = handler_workflow_resources_for_base_model(
        "deepseek-ai/DeepSeek-V4-Flash"
    )
    assert resources is not None
    for stage in (
        resources.train_inf_mismatch,
        resources.yes_no_trainability,
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
        assert engine_args["moe_backend"] == "triton"
        assert engine_args["kv_cache_dtype"] == "fp8"
        assert stage.streaming_weight_offload is True
        assert stage.megatron_env == {}

    for stage in (resources.merged_vllm_serving, resources.native_vllm_lora):
        assert stage is not None
        assert stage.vllm is not None
        engine_args = stage.vllm.engine_args()
        assert engine_args.get("load_format") != "dummy"
        assert "hf_overrides" not in engine_args
        assert engine_args["max_model_len"] == 1024
    assert resources.merged_vllm_serving is not None
    assert resources.merged_vllm_serving.required_world_size == 8
    assert resources.merged_vllm_serving.megatron is not None
    assert resources.merged_vllm_serving.vllm is not None
    assert resources.merged_vllm_serving.megatron.gpu_ids == [0, 1, 2, 3]
    assert resources.merged_vllm_serving.megatron.topology.tp == 2
    assert resources.merged_vllm_serving.megatron.topology.ep == 4
    assert resources.merged_vllm_serving.megatron.topology.dp == 2
    assert resources.merged_vllm_serving.vllm.gpu_ids == [4, 5, 6, 7]
    assert not (
        set(resources.merged_vllm_serving.megatron.gpu_ids)
        & set(resources.merged_vllm_serving.vllm.gpu_ids)
    )
    assert resources.merged_vllm_serving.vllm.engine_args()["kv_cache_dtype"] == "fp8"
    assert resources.native_vllm_lora is not None
    assert resources.native_vllm_lora.vllm is not None
    assert resources.native_vllm_lora.vllm.engine_args().get("max_loras", 2) == 2


@pytest.mark.parametrize(
    ("stage_name", "trainer_gpu_ids", "trainer_ep", "trainer_dp"),
    [
        ("train_inf_mismatch", [0, 1, 2, 3], 4, 2),
        ("yes_no_trainability", [0, 1, 2, 3], 4, 2),
        ("length_trainability", [0, 1, 2, 3], 4, 2),
        ("merged_vllm_serving", [0, 1], 2, 1),
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
    assert stage.vllm.engine_args()["moe_backend"] == "triton"
    assert stage.vllm.engine_args()["kv_cache_dtype"] == "fp8"


def test_glm52_reduced_workflow_uses_portable_serving_backends() -> None:
    resources = handler_workflow_resources_for_base_model("zai-org/GLM-5.2")
    assert resources is not None
    joint_stages = (
        resources.train_inf_mismatch,
        resources.merged_vllm_serving,
        resources.yes_no_trainability,
        resources.length_trainability,
    )
    for stage in joint_stages:
        assert stage is not None
        assert stage.required_world_size == 2
        assert stage.megatron is not None
        assert stage.megatron.gpu_ids == [0]
    assert resources.native_vllm_lora is not None
    assert resources.native_vllm_lora.required_world_size == 2
    assert resources.native_vllm_lora.megatron is None
    for stage in (*joint_stages, resources.native_vllm_lora):
        assert stage is not None
        assert stage.vllm is not None
        assert stage.vllm.gpu_ids == [1]
        engine_args = stage.vllm.engine_args()
        assert engine_args["attention_backend"] == "FLASHMLA_SPARSE"
        assert engine_args["max_model_len"] == 1024
        assert engine_args["moe_backend"] == "triton"
    assert resources.yes_no_trainability_variant == "megatron_dedicated"


def test_h200_equivalent_slots_tolerate_reported_gb300_vram() -> None:
    assert _h200_equivalent_slots_for_total_gib(80.0) == 0
    assert _h200_equivalent_slots_for_total_gib(139.0) == 1
    assert _h200_equivalent_slots_for_total_gib(267.69) == 2
    assert _h200_equivalent_slots_for_total_gib(276.6) == 2


def test_h200_throughput_depth_only_reduces_memory_bound_handlers() -> None:
    config = ThroughputWorkflowConfig(num_layers=12)
    assert _throughput_config_for_hardware("glm52", config, "h200").num_layers == 6
    assert _throughput_config_for_hardware("dsv4", config, "h200").num_layers == 4
    assert _throughput_config_for_hardware("glm52", config, "b300") is config
    assert _throughput_config_for_hardware("llama3_dense", config, "h200") is config


@pytest.mark.parametrize("hardware", ("b300", "h200"))
def test_dsv4_uses_model_specific_activation_lag_limit(hardware: str) -> None:
    for handler_key, config in _THROUGHPUT_CONFIGS.items():
        thresholds = config.thresholds[hardware]
        assert thresholds.max_mean_policy_activation_lag_s == (
            2.25 if handler_key == "dsv4" else 1.5
        )
        assert thresholds.max_policy_activation_lag_s == 3.5


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


@pytest.mark.parametrize(
    ("outcomes", "passed", "attempts", "retryable"),
    [
        (
            (
                (
                    2,
                    {
                        "outcome": "error",
                        "comparison_completed": False,
                        "exception_type": "ConnectionRefusedError",
                    },
                ),
                (0, {"outcome": "passed", "comparison_completed": True}),
            ),
            True,
            2,
            True,
        ),
        (((1, {"outcome": "failed", "comparison_completed": True}),), False, 1, False),
        (
            (
                (
                    2,
                    {
                        "outcome": "error",
                        "comparison_completed": False,
                        "exception_type": "ValueError",
                    },
                ),
            ),
            False,
            1,
            False,
        ),
        (
            ((0, {"outcome": "skipped", "comparison_completed": False}),),
            False,
            1,
            False,
        ),
        (((3, None),), False, 1, False),
    ],
)
def test_mismatch_workflow_retries_only_executed_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcomes: tuple[tuple[int, dict[str, object] | None], ...],
    passed: bool,
    attempts: int,
    retryable: bool,
) -> None:
    results = iter(outcomes)

    def run_attempt(command, **_kwargs):
        returncode, payload = next(results)
        if payload is not None:
            Path(command[-1]).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    monkeypatch.setenv("ART_TRAIN_INF_MISMATCH_ATTEMPTS", "3")
    monkeypatch.setattr(
        mismatch_workflow_stage, "create_artifact_dir", lambda _nodeid: tmp_path
    )
    monkeypatch.setattr(mismatch_workflow_stage, "_run_attempt", run_attempt)
    report = mismatch_workflow_stage.run_train_inf_mismatch(
        base_model="Qwen/Qwen3.5-35B-A3B"
    )
    assert (report.passed, report.attempt_count, report.attempts[0].retryable) == (
        passed,
        attempts,
        retryable,
    )
    assert report.duration_s >= report.attempts[0].duration_s


@pytest.mark.parametrize("handler_key", ["qwen3_moe", "qwen3_5_moe"])
def test_qwen_moe_reduced_serving_uses_plain_expert_storage(handler_key: str) -> None:
    resources = HANDLER_WORKFLOW_RESOURCES[handler_key]
    for stage in (resources.merged_vllm_serving, resources.native_vllm_lora):
        assert stage is not None
        assert stage.vllm is not None
        assert stage.vllm.gpu_ids == [1]
        engine_args = stage.vllm.engine_args()
        assert engine_args["enforce_eager"] is True
        assert engine_args["max_model_len"] == 1024
        assert engine_args["moe_backend"] == "triton"


def test_gpt_oss_reduced_serving_has_bounded_context() -> None:
    resources = HANDLER_WORKFLOW_RESOURCES["gpt_oss_moe"]
    for stage in (resources.merged_vllm_serving, resources.native_vllm_lora):
        assert stage is not None
        assert stage.vllm is not None
        assert stage.vllm.engine_args()["max_model_len"] == 1024


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


def test_build_all_architectures_validation_report_stops_on_failure(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []

    def _build_validation_report(
        *,
        base_model,
        include_yes_no_trainability=False,
        include_sensitivity=None,
        output_json=None,
        skip_stages=None,
        only_stage=None,
        stop_on_failure=False,
        allow_unvalidated_arch=False,
    ):
        del include_yes_no_trainability
        del include_sensitivity
        del output_json
        del skip_stages
        del only_stage
        del stop_on_failure
        del allow_unvalidated_arch
        calls.append(base_model)
        passed = base_model != "Qwen/Qwen3-32B"
        return ValidationReport(
            git={},
            base_model=base_model,
            model_key="qwen3_dense",
            passed=passed,
            stages=[
                ValidationStageResult(
                    name="train_inf_mismatch",
                    passed=passed,
                )
            ],
        )

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.build_validation_report",
        _build_validation_report,
    )

    report = build_all_architectures_validation_report(
        output_json=tmp_path / "all_architectures.json",
        stop_on_failure=True,
    )

    assert calls == [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
    ]
    assert report.passed is False
    assert [item.base_model for item in report.reports] == calls


def test_build_validation_report_populates_architecture_stage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "art.megatron.model_support.discovery.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[LayerFamilyInstance(key="standard_attention", count=2)],
            recommended_min_layers=1,
        ),
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.detect_dependency_versions",
        lambda: {"transformers": "5.2.0"},
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._run_stage_in_subprocess",
        lambda *, stage_name, base_model, architecture, allow_unvalidated_arch=False: {
            "hf_parity": ValidationStageResult(
                name="hf_parity",
                passed=True,
                metrics={"signal": "pass", "requested_num_layers": 1},
                artifact_dir="/tmp/hf_parity",
            ),
            "lora_coverage": ValidationStageResult(
                name="lora_coverage",
                passed=True,
                metrics={"wrapped_adapter_prefix_count": 12},
            ),
            "train_inf_mismatch": ValidationStageResult(
                name="train_inf_mismatch",
                passed=True,
                metrics={"passed_count": 1, "failed_count": 0},
                artifact_dir="/tmp/train-inf-mismatch",
            ),
            "merged_vllm_serving": ValidationStageResult(
                name="merged_vllm_serving",
                passed=True,
                metrics={"served_model_name": "validation@0"},
                artifact_dir="/tmp/merged-serving",
            ),
            "correctness_sensitivity": ValidationStageResult(
                name="correctness_sensitivity",
                passed=True,
                metrics={
                    "correctness_variant_count": 4,
                    "sensitivity_variant_count": 9,
                },
                artifact_dir="/tmp/correctness",
            ),
            "chat_template_rollout": ValidationStageResult(
                name="chat_template_rollout",
                passed=True,
                metrics={
                    "passed": True,
                    "scenario_count": 6,
                    "failed_scenarios": [],
                },
                artifact_dir="/tmp/chat-template",
            ),
            "packing_invariance": ValidationStageResult(
                name="packing_invariance",
                passed=True,
                metrics={
                    "num_layers": 4,
                    "scenarios": [
                        {
                            "name": "stop_early",
                            "matched": True,
                            "checked_token_count": 40,
                        }
                    ],
                },
                artifact_dir="/tmp/packing-invariance",
            ),
            "length_trainability": ValidationStageResult(
                name="length_trainability",
                passed=True,
                metrics={
                    "latest_step": 4,
                    "best_train_abs_error": 1.0,
                },
                artifact_dir="/tmp/length-trainability",
            ),
            "e2e_throughput": ValidationStageResult(
                name="e2e_throughput",
                passed=True,
                metrics={"accepted_train_tok_s": 1234.0},
                artifact_dir="/tmp/e2e-throughput",
            ),
            "native_vllm_lora": ValidationStageResult(
                name="native_vllm_lora",
                passed=True,
                metrics={
                    "rollout_weights_mode": "lora",
                    "step0_name": "validation@0",
                    "step1_name": "validation@1",
                    "model_ids_before": ["validation@0"],
                    "model_ids_after": ["validation@0", "validation@1"],
                    "step0_served": True,
                    "step1_served": True,
                },
                artifact_dir="/tmp/native-vllm-lora",
            ),
        }[stage_name],
    )

    report = build_validation_report(base_model="Qwen/Qwen3.5-35B-A3B")

    assert report.base_model == "Qwen/Qwen3.5-35B-A3B"
    assert report.model_key == "qwen3_5_moe"
    assert report.dependency_versions == {"transformers": "5.2.0"}
    dependency_stage = next(
        stage for stage in report.stages if stage.name == "dependency_resolution"
    )
    assert dependency_stage.passed is True
    assert _without_stage_duration(dependency_stage) == {"transformers": "5.2.0"}
    architecture_stage = next(
        stage for stage in report.stages if stage.name == "architecture_discovery"
    )
    assert architecture_stage.passed is True
    assert _without_stage_duration(architecture_stage) == {
        "recommended_min_layers": 1,
        "layer_families": [
            {
                "key": "standard_attention",
                "count": 2,
                "layer_index": None,
                "module_path": None,
                "module_type": None,
            }
        ],
        "unresolved_risks": [],
    }
    hf_parity_stage = next(
        stage for stage in report.stages if stage.name == "hf_parity"
    )
    assert hf_parity_stage.passed is True
    assert _without_stage_duration(hf_parity_stage) == {
        "signal": "pass",
        "requested_num_layers": 1,
    }
    assert hf_parity_stage.artifact_dir == "/tmp/hf_parity"
    lora_coverage_stage = next(
        stage for stage in report.stages if stage.name == "lora_coverage"
    )
    assert lora_coverage_stage.passed is True
    assert _without_stage_duration(lora_coverage_stage) == {
        "wrapped_adapter_prefix_count": 12
    }
    mismatch_stage = next(
        stage for stage in report.stages if stage.name == "train_inf_mismatch"
    )
    assert mismatch_stage.passed is True
    assert _without_stage_duration(mismatch_stage) == {
        "passed_count": 1,
        "failed_count": 0,
    }
    assert mismatch_stage.artifact_dir == "/tmp/train-inf-mismatch"
    correctness_stage = next(
        stage for stage in report.stages if stage.name == "correctness_sensitivity"
    )
    assert correctness_stage.passed is True
    assert _without_stage_duration(correctness_stage) == {
        "correctness_variant_count": 4,
        "sensitivity_variant_count": 9,
    }
    assert correctness_stage.artifact_dir == "/tmp/correctness"
    merged_stage = next(
        stage for stage in report.stages if stage.name == "merged_vllm_serving"
    )
    assert merged_stage.passed is True
    assert _without_stage_duration(merged_stage) == {
        "served_model_name": "validation@0"
    }
    assert merged_stage.artifact_dir == "/tmp/merged-serving"
    chat_template_stage = next(
        stage for stage in report.stages if stage.name == "chat_template_rollout"
    )
    assert chat_template_stage.passed is True
    assert _without_stage_duration(chat_template_stage) == {
        "passed": True,
        "scenario_count": 6,
        "failed_scenarios": [],
    }
    assert chat_template_stage.artifact_dir == "/tmp/chat-template"
    packing_invariance_stage = next(
        stage for stage in report.stages if stage.name == "packing_invariance"
    )
    assert packing_invariance_stage.passed is True
    assert _without_stage_duration(packing_invariance_stage) == {
        "num_layers": 4,
        "scenarios": [
            {
                "name": "stop_early",
                "matched": True,
                "checked_token_count": 40,
            }
        ],
    }
    assert packing_invariance_stage.artifact_dir == "/tmp/packing-invariance"
    trainability_stage = next(
        stage for stage in report.stages if stage.name == "length_trainability"
    )
    assert trainability_stage.passed is True
    assert _without_stage_duration(trainability_stage) == {
        "latest_step": 4,
        "best_train_abs_error": 1.0,
    }
    assert trainability_stage.artifact_dir == "/tmp/length-trainability"
    throughput_stage = next(
        stage for stage in report.stages if stage.name == "e2e_throughput"
    )
    assert throughput_stage.passed is True
    throughput_metrics = _without_stage_duration(throughput_stage)
    assert throughput_metrics["accepted_train_tok_s"] == 1234.0
    assert throughput_stage.artifact_dir == "/tmp/e2e-throughput"
    fixture_durations = [
        cast(float, stage.metrics["fixture_provisioning_s"])
        for stage in report.stages
        if "fixture_provisioning_s" in stage.metrics
    ]
    assert len(fixture_durations) == 1 and fixture_durations[0] >= 0.0
    assert all(stage.name != "yes_no_trainability" for stage in report.stages)
    native_vllm_lora_stage = next(
        stage for stage in report.stages if stage.name == "native_vllm_lora"
    )
    assert native_vllm_lora_stage.passed is True
    assert _without_stage_duration(native_vllm_lora_stage) == {
        "rollout_weights_mode": "lora",
        "step0_name": "validation@0",
        "step1_name": "validation@1",
        "model_ids_before": ["validation@0"],
        "model_ids_after": ["validation@0", "validation@1"],
        "step0_served": True,
        "step1_served": True,
    }
    assert native_vllm_lora_stage.artifact_dir == "/tmp/native-vllm-lora"


def test_build_validation_report_success_cleanup_does_not_implicitly_keep_traces(
    monkeypatch,
) -> None:
    seen_keep_env: list[str | None] = []

    monkeypatch.delenv(KEEP_TOPOLOGY_ARTIFACTS_ENV, raising=False)

    monkeypatch.setattr(
        "art.megatron.model_support.discovery.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[LayerFamilyInstance(key="standard_attention", count=1)],
            recommended_min_layers=1,
        ),
    )

    def _run_stage_in_subprocess(
        *,
        stage_name,
        base_model,
        architecture,
        allow_unvalidated_arch=False,
    ) -> ValidationStageResult:
        del base_model, architecture, allow_unvalidated_arch
        if stage_name == "correctness_sensitivity":
            seen_keep_env.append(os.environ.get(KEEP_TOPOLOGY_ARTIFACTS_ENV))
        return ValidationStageResult(name=stage_name, passed=True, metrics={})

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._run_stage_in_subprocess",
        _run_stage_in_subprocess,
    )

    build_validation_report(
        base_model="Qwen/Qwen3.5-35B-A3B",
        include_sensitivity=True,
    )

    assert seen_keep_env == [None]
    assert os.environ.get(KEEP_TOPOLOGY_ARTIFACTS_ENV) is None


def test_runtime_artifact_cleanup_preserves_evidence(tmp_path: Path) -> None:
    stage_dir = tmp_path / "e2e_throughput"
    model_dir = stage_dir / "art" / "models" / "run"
    report = stage_dir / "throughput_measurements.json"
    matched_input = stage_dir / "matched_packed_input.msgpack"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    matched_input.write_bytes(b"input")
    removed_bytes = 0
    for name in (
        "checkpoints",
        "megatron_runtime",
        "optimizer_states",
        "trajectories",
    ):
        path = model_dir / name
        path.mkdir(parents=True)
        payload = path / "payload"
        payload.write_bytes(name.encode())
        removed_bytes += payload.stat().st_size

    metrics = _prune_runtime_artifacts(stage_dir)

    assert metrics == {
        "workflow_pruned_runtime_artifact_dirs": 4,
        "workflow_pruned_runtime_artifact_bytes": removed_bytes,
    }
    assert report.read_text() == "{}"
    assert matched_input.read_bytes() == b"input"
    assert not any((model_dir / name).exists() for name in _RUNTIME_ARTIFACT_DIR_NAMES)


def test_build_validation_report_only_stage_skips_other_stages(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "art.megatron.model_support.discovery.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[],
            recommended_min_layers=1,
        ),
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.detect_dependency_versions",
        lambda: {},
    )

    def _run_stage_in_subprocess(**kwargs) -> ValidationStageResult:
        stage_name = kwargs["stage_name"]
        calls.append(stage_name)
        return ValidationStageResult(name=stage_name, passed=True)

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._run_stage_in_subprocess",
        _run_stage_in_subprocess,
    )

    report = build_validation_report(
        base_model="Qwen/Qwen3.5-35B-A3B",
        only_stage="length_trainability",
    )

    skipped = next(stage for stage in report.stages if stage.name == "hf_parity")
    assert calls == ["length_trainability"]
    assert _without_stage_duration(skipped) == {
        "skipped": True,
        "reason": "--only-stage=length_trainability",
    }


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


def test_build_validation_report_captures_lora_coverage_failure(monkeypatch) -> None:
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
                name="lora_coverage",
                passed=False,
                metrics={"error": "RuntimeError: missing wrapped targets"},
            )
            if stage_name == "lora_coverage"
            else ValidationStageResult(
                name=stage_name,
                passed=True,
                metrics={},
            )
        ),
    )

    report = build_validation_report(base_model="Qwen/Qwen3.5-35B-A3B")

    lora_coverage_stage = next(
        stage for stage in report.stages if stage.name == "lora_coverage"
    )
    assert lora_coverage_stage.passed is False
    assert _without_stage_duration(lora_coverage_stage) == {
        "error": "RuntimeError: missing wrapped targets"
    }


def test_build_validation_report_writes_incremental_output_and_stops(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "art.megatron.model_support.discovery.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[],
            recommended_min_layers=1,
        ),
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.detect_dependency_versions",
        lambda: {},
    )

    def _run_stage_in_subprocess(
        *,
        stage_name,
        base_model,
        architecture,
        allow_unvalidated_arch=False,
    ):
        calls.append(stage_name)
        return ValidationStageResult(
            name=stage_name,
            passed=stage_name != "lora_coverage",
            metrics={"stage": stage_name},
        )

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._run_stage_in_subprocess",
        _run_stage_in_subprocess,
    )
    output_json = tmp_path / "workflow_report.json"

    report = build_validation_report(
        base_model="Qwen/Qwen3.5-35B-A3B",
        output_json=output_json,
        stop_on_failure=True,
    )

    assert calls == ["hf_parity", "lora_coverage"]
    assert output_json.exists()
    saved = ValidationReport.model_validate_json(output_json.read_text())
    assert saved == report
    failed_stage = next(
        stage for stage in saved.stages if stage.name == "lora_coverage"
    )
    skipped_stage = next(
        stage for stage in saved.stages if stage.name == "train_inf_mismatch"
    )
    assert failed_stage.passed is False
    assert _without_stage_duration(skipped_stage) == {
        "skipped": True,
        "reason": "stopped after lora_coverage failed",
    }


def test_assess_minimal_layer_coverage_reports_missing_families(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "art.megatron.model_support.discovery.inspect_architecture",
        lambda base_model: ArchitectureReport(
            base_model=base_model,
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            layer_families=[
                LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
                LayerFamilyInstance(key="standard_attention", layer_index=3),
                LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
                LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
            ],
            recommended_min_layers=4,
        ),
    )

    coverage = assess_minimal_layer_coverage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        num_layers=2,
    )

    assert coverage.covered is False
    assert coverage.requested_num_layers == 2
    assert coverage.recommended_min_layers == 4
    assert coverage.missing_layer_families == ["standard_attention"]
    assert coverage.unresolved_risks == []


def test_run_chat_template_rollout_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(
            run_chat_template_rollout=lambda *, base_model: SimpleNamespace(
                passed=True,
                scenario_count=6,
                failed_scenarios=[],
                output_dir="/tmp/chat-template",
                model_dump=lambda mode="json": {
                    "passed": True,
                    "scenario_count": 6,
                    "failed_scenarios": [],
                },
            )
        ),
    )

    result = run_chat_template_rollout_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
        ),
    )

    assert result.passed is True
    assert result.artifact_dir == "/tmp/chat-template"


def test_run_correctness_sensitivity_stage_runs_dense_models(monkeypatch) -> None:
    case_configs: list[SimpleNamespace] = []
    oracle_module = SimpleNamespace(
        ORACLE_OBJECTIVE_ENV="ART_ORACLE_OBJECTIVE",
        SUPPORTED_ORACLE_OBJECTIVES=("sft",),
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        selected_suite_topologies=lambda *, is_moe, cp_supported=True: [
            SimpleNamespace(world_size=lambda: 1, slug=lambda: "tp1", pp=1, vpp=1),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "tp2", pp=1, vpp=1),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "dp2", pp=1, vpp=1),
            SimpleNamespace(world_size=lambda: 4, slug=lambda: "tp2_dp2", pp=1, vpp=1),
        ],
        oracle_topology=lambda *, is_moe: SimpleNamespace(world_size=lambda: 1),
        selected_oracle_objectives=lambda: ["sft"],
        supported_sensitivity_mutations_for_objective=lambda objective, *, is_moe: (
            ["skip_finalize"] if objective == "sft" and not is_moe else []
        ),
        sensitivity_topology_for_mutation=lambda mutation, *, is_moe: SimpleNamespace(
            world_size=lambda: 2
        ),
        available_gpu_count=lambda: 4,
        run_suite=lambda case_config, max_world_size, cp_supported=True, **kwargs: (
            case_configs.append(case_config)
            or [
                SimpleNamespace(
                    variant="sft_topology_tp2_dp2",
                    topology="tp2_dp2",
                    signal="pass",
                    fail_count=0,
                )
            ]
        ),
        run_sensitivity_suite=lambda case_config, mutations, max_world_size: [
            SimpleNamespace(
                variant="sft_sensitivity_skip_finalize",
                topology="tp2",
                signal="fail",
                expected_signal="fail",
                fail_count=1,
            )
        ],
        ensure_case_artifacts=lambda case_config: SimpleNamespace(
            case_dir="/tmp/oracle"
        ),
        keep_topology_artifacts=lambda: False,
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: oracle_module,
    )
    monkeypatch.delenv(SKIP_SENSITIVITY_ENV, raising=False)

    result = run_correctness_sensitivity_stage(
        base_model="Qwen/Qwen3.5-4B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-4B",
            model_key="qwen3_5_dense",
            handler_key="qwen3_5_dense",
            layer_families=[
                LayerFamilyInstance(key="dense_mlp", layer_index=0),
                LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
                LayerFamilyInstance(key="standard_attention", layer_index=3),
            ],
            recommended_min_layers=4,
        ),
    )

    assert result.passed is True
    assert result.metrics["is_moe"] is False
    assert result.metrics["available_gpu_count"] == 4
    assert result.metrics["max_world_size"] == 4
    assert result.metrics["required_gpu_count"] == 4
    assert result.metrics["correctness_variant_count"] == 1
    assert result.metrics["correctness_excluded_topologies"] == []
    assert result.metrics["sensitivity_mutations"] == ["skip_finalize"]
    assert result.metrics["default_excluded_sensitivity_mutations"] == [
        "attn_skip_flash_lse_normalize"
    ]
    assert case_configs[0].is_moe is False


def test_run_yes_no_trainability_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(
            run_yes_no_trainability=lambda *, base_model, artifact_root, allow_unvalidated_arch=False: (
                SimpleNamespace(
                    latest_step=2,
                    initial_eval_reward=0.4,
                    final_eval_reward=0.95,
                    reward_threshold=0.95,
                    saturated_step=2,
                    output_dir="/tmp/trainability",
                    model_dump=lambda mode="json": {
                        "latest_step": 2,
                        "initial_eval_reward": 0.4,
                        "final_eval_reward": 0.95,
                        "reward_threshold": 0.95,
                        "saturated_step": 2,
                    },
                )
            ),
            yes_no_trainability_passed=lambda report: (
                report.final_eval_reward >= report.reward_threshold
            ),
        ),
    )

    result = run_yes_no_trainability_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
        ),
    )

    assert result.passed is True
    assert result.artifact_dir == "/tmp/trainability"


def test_run_length_trainability_stage(monkeypatch) -> None:
    workspace = (
        Path(os.environ[WORKFLOW_STAGE_DIR_ENV])
        / "artifacts"
        / "megatron_dedicated_workspace"
    )
    workspace.mkdir(parents=True)
    (workspace / "optimizer.pt").write_bytes(b"large")
    report = SimpleNamespace(
        summary_log_path="/tmp/length-trainability/length_trainability.log",
        model_dump=lambda mode="json": {
            "latest_step": 3,
            "initial_train_abs_error": 12.0,
            "best_train_abs_error": 1.0,
        },
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(
            run_length_trainability=lambda *, base_model, allow_unvalidated_arch=False: (
                report
            ),
            length_trainability_passed=lambda candidate: candidate is report,
        ),
    )

    result = run_length_trainability_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
        ),
    )

    assert result.name == "length_trainability"
    assert result.passed is True
    assert result.artifact_dir == "/tmp/length-trainability"
    assert not workspace.exists()


def test_run_length_trainability_stage_cleans_workspace_on_failure(monkeypatch) -> None:
    workspace = (
        Path(os.environ[WORKFLOW_STAGE_DIR_ENV])
        / "artifacts"
        / "megatron_dedicated_workspace"
    )
    workspace.mkdir(parents=True)

    def fail(**kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(run_length_trainability=fail),
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_length_trainability_stage(
            base_model="google/gemma-4-31B-it",
            architecture=ArchitectureReport(
                base_model="google/gemma-4-31B-it",
                model_key="gemma4_dense",
                handler_key="gemma4_dense",
            ),
        )

    assert not workspace.exists()


def test_run_train_inf_mismatch_stage(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _run_train_inf_mismatch(
        *,
        base_model: str,
        allow_unvalidated_arch: bool,
    ) -> SimpleNamespace:
        seen["allow_unvalidated_arch"] = allow_unvalidated_arch
        return SimpleNamespace(
            passed=True,
            artifact_dir="/tmp/train-inf-mismatch",
            model_dump=lambda mode="json": {
                "base_model": base_model,
                "passed": True,
                "passed_count": 1,
                "failed_count": 0,
            },
        )

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(
            run_train_inf_mismatch=_run_train_inf_mismatch,
        ),
    )

    result = run_train_inf_mismatch_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
        ),
        allow_unvalidated_arch=True,
    )

    assert result.name == "train_inf_mismatch"
    assert result.passed is True
    assert result.artifact_dir == "/tmp/train-inf-mismatch"
    assert seen == {"allow_unvalidated_arch": True}
    assert result.metrics == {
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "passed": True,
        "passed_count": 1,
        "failed_count": 0,
    }


def test_run_native_vllm_lora_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: (
            SimpleNamespace(
                OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            )
            if name == "integration.megatron.model_support.oracle_harness"
            else SimpleNamespace(
                run_native_vllm_lora=lambda case_config: SimpleNamespace(
                    rollout_weights_mode="lora",
                    step0_name="validation@0",
                    step1_name="validation@1",
                    model_ids_before=["validation@0"],
                    model_ids_after=["validation@0", "validation@1"],
                    step0_served=True,
                    step1_served=True,
                    output_dir="/tmp/native-vllm-lora",
                    model_dump=lambda mode="json": {
                        "rollout_weights_mode": "lora",
                        "step0_name": "validation@0",
                        "step1_name": "validation@1",
                        "model_ids_before": ["validation@0"],
                        "model_ids_after": ["validation@0", "validation@1"],
                        "step0_served": True,
                        "step1_served": True,
                    },
                )
            )
        ),
    )

    result = run_native_vllm_lora_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
        ),
    )

    assert result.name == "native_vllm_lora"
    assert result.passed is True
    assert result.artifact_dir == "/tmp/native-vllm-lora"


def test_run_packing_invariance_stage(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(
            run_packing_invariance=lambda *, base_model, num_layers, allow_unvalidated_arch=False, in_process=False: (
                calls.append(in_process)
                or SimpleNamespace(
                    output_dir="/tmp/packing-invariance",
                    model_dump=lambda mode="json": {
                        "base_model": base_model,
                        "num_layers": num_layers,
                        "scenarios": [
                            {
                                "name": "stop_early",
                                "matched": True,
                                "checked_token_count": 40,
                            },
                            {
                                "name": "truncate",
                                "matched": True,
                                "checked_token_count": 44,
                            },
                        ],
                    },
                )
            )
        ),
    )

    result = run_packing_invariance_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            recommended_min_layers=4,
        ),
    )

    assert result.passed is True
    assert result.artifact_dir == "/tmp/packing-invariance"
    assert calls == [True]


def test_assess_minimal_layer_coverage_passes_when_prefix_covers_all_families(
    monkeypatch,
) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        layer_families=[
            LayerFamilyInstance(key="gated_delta_net_attention", layer_index=0),
            LayerFamilyInstance(key="standard_attention", layer_index=3),
            LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0),
            LayerFamilyInstance(key="shared_experts_mlp", layer_index=0),
        ],
        recommended_min_layers=4,
    )

    coverage = assess_minimal_layer_coverage(
        base_model=architecture.base_model,
        num_layers=4,
        architecture=architecture,
    )

    assert coverage.covered is True
    assert coverage.missing_layer_families == []


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


def test_run_correctness_sensitivity_stage_summarizes_reports(monkeypatch) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        layer_families=[LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0)],
        recommended_min_layers=1,
    )
    oracle_module = SimpleNamespace(
        ORACLE_OBJECTIVE_ENV="ART_ORACLE_OBJECTIVE",
        SUPPORTED_ORACLE_OBJECTIVES=("sft",),
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        selected_suite_topologies=lambda *, is_moe, cp_supported=True: [
            SimpleNamespace(world_size=lambda: 1, slug=lambda: "tp1", pp=1, vpp=1),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "tp2", pp=2, vpp=2),
        ],
        oracle_topology=lambda *, is_moe: SimpleNamespace(world_size=lambda: 1),
        selected_oracle_objectives=lambda: ["sft"],
        supported_sensitivity_mutations_for_objective=lambda objective, *, is_moe: (
            ["skip_finalize"] if objective == "sft" else []
        ),
        sensitivity_topology_for_mutation=lambda mutation, *, is_moe: SimpleNamespace(
            world_size=lambda: 2
        ),
        available_gpu_count=lambda: 2,
        run_suite=lambda case_config, max_world_size, cp_supported=True, **kwargs: [
            SimpleNamespace(
                variant="sft_topology_tp2",
                topology="tp2",
                signal="pass",
                fail_count=0,
            )
        ],
        run_sensitivity_suite=lambda case_config, mutations, max_world_size: [
            SimpleNamespace(
                variant="sft_sensitivity_skip_finalize",
                topology="tp2",
                signal="fail",
                expected_signal="fail",
                fail_count=1,
            )
        ],
        ensure_case_artifacts=lambda case_config: SimpleNamespace(
            case_dir="/tmp/oracle"
        ),
        keep_topology_artifacts=lambda: False,
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: oracle_module,
    )

    stage = run_correctness_sensitivity_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=architecture,
    )

    assert stage.name == "correctness_sensitivity"
    assert stage.passed is True
    assert stage.metrics["requested_num_layers"] == 4
    assert stage.metrics["is_moe"] is True
    assert stage.metrics["objectives"] == ["sft"]
    assert stage.metrics["sensitivity_mutations"] == ["skip_finalize"]
    assert stage.metrics["default_excluded_sensitivity_mutations"] == [
        "attn_skip_flash_lse_normalize"
    ]
    assert stage.metrics["available_gpu_count"] == 2
    assert stage.metrics["required_gpu_count"] == 2
    assert stage.metrics["correctness_variant_count"] == 1
    assert stage.metrics["sensitivity_skipped"] is False
    assert stage.metrics["sensitivity_skip_reason"] is None
    assert stage.metrics["sensitivity_variant_count"] == 1
    assert stage.artifact_dir == "/tmp/oracle"


def test_run_correctness_sensitivity_stage_uses_dsv4_real_path_config(
    monkeypatch,
) -> None:
    architecture = ArchitectureReport(
        base_model="deepseek-ai/DeepSeek-V4-Flash",
        model_key="dsv4",
        handler_key="dsv4",
        layer_families=[LayerFamilyInstance(key="dsv4_attention", layer_index=0)],
        recommended_min_layers=4,
    )
    captured: dict[str, object] = {}
    oracle_module = SimpleNamespace(
        ORACLE_OBJECTIVE_ENV="ART_ORACLE_OBJECTIVE",
        SUPPORTED_ORACLE_OBJECTIVES=("rl",),
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        MetricThresholdRule=lambda **kwargs: SimpleNamespace(**kwargs),
        selected_suite_topologies=lambda *, is_moe, cp_supported=True: [
            SimpleNamespace(world_size=lambda: 1, slug=lambda: "tp1", pp=1, vpp=1),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "tp2", pp=1, vpp=1),
        ],
        oracle_topology=lambda *, is_moe: SimpleNamespace(world_size=lambda: 1),
        selected_oracle_objectives=lambda: ["rl"],
        supported_sensitivity_mutations_for_objective=lambda objective, *, is_moe: [],
        sensitivity_topology_for_mutation=lambda mutation, *, is_moe: SimpleNamespace(
            world_size=lambda: 2
        ),
        available_gpu_count=lambda: 2,
        run_suite=lambda case_config, **kwargs: (
            captured.update(case_config=case_config, suite_kwargs=kwargs)
            or [
                SimpleNamespace(
                    variant="rl_topology_tp2",
                    topology="tp2",
                    signal="pass",
                    fail_count=0,
                )
            ]
        ),
        run_sensitivity_suite=lambda case_config, mutations, max_world_size: [],
        ensure_case_artifacts=lambda case_config: SimpleNamespace(
            case_dir="/tmp/oracle"
        ),
        keep_topology_artifacts=lambda: False,
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: oracle_module,
    )
    monkeypatch.setenv(SKIP_SENSITIVITY_ENV, "1")

    stage = run_correctness_sensitivity_stage(
        base_model="deepseek-ai/DeepSeek-V4-Flash",
        architecture=architecture,
    )

    case_config = captured["case_config"]
    suite_kwargs = cast(dict[str, object], captured["suite_kwargs"])
    phase_pass_fns = cast(dict[str, object], suite_kwargs["phase_pass_fns"])
    assert getattr(case_config, "precision") == "bf16"
    assert suite_kwargs["use_fp32_lora_reference"] is False
    assert getattr(phase_pass_fns["forward"], "limits") == {"mean_abs_pct": 3.0}
    assert getattr(phase_pass_fns["grads"], "limits") == {"mean_abs_pct": 5.0}
    assert stage.metrics["precision"] == "bf16"
    assert stage.metrics["use_fp32_lora_reference"] is False


def test_run_correctness_sensitivity_stage_can_skip_sensitivity_only(
    monkeypatch,
) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        layer_families=[LayerFamilyInstance(key="grouped_moe_mlp", layer_index=0)],
        recommended_min_layers=4,
    )
    oracle_module = SimpleNamespace(
        ORACLE_OBJECTIVE_ENV="ART_ORACLE_OBJECTIVE",
        SUPPORTED_ORACLE_OBJECTIVES=("sft",),
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        selected_suite_topologies=lambda *, is_moe, cp_supported=True: [
            SimpleNamespace(world_size=lambda: 1, slug=lambda: "tp1", pp=1, vpp=1),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "tp2", pp=1, vpp=1),
        ],
        oracle_topology=lambda *, is_moe: SimpleNamespace(world_size=lambda: 1),
        selected_oracle_objectives=lambda: ["sft"],
        supported_sensitivity_mutations_for_objective=lambda objective, *, is_moe: (
            ["skip_finalize"] if objective == "sft" else []
        ),
        sensitivity_topology_for_mutation=lambda mutation, *, is_moe: SimpleNamespace(
            world_size=lambda: 4
        ),
        available_gpu_count=lambda: 2,
        run_suite=lambda case_config, max_world_size, cp_supported=True, **kwargs: [
            SimpleNamespace(
                variant="sft_topology_tp2",
                topology="tp2",
                signal="pass",
                fail_count=0,
            )
        ],
        run_sensitivity_suite=lambda case_config, mutations, max_world_size: (
            _ for _ in ()
        ).throw(AssertionError("sensitivity suite should be skipped")),
        ensure_case_artifacts=lambda case_config: SimpleNamespace(
            case_dir="/tmp/oracle"
        ),
        keep_topology_artifacts=lambda: False,
    )
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: oracle_module,
    )
    monkeypatch.setenv(SKIP_SENSITIVITY_ENV, "1")

    stage = run_correctness_sensitivity_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=architecture,
    )

    assert stage.name == "correctness_sensitivity"
    assert stage.passed is True
    assert stage.metrics["required_gpu_count"] == 2
    assert stage.metrics["correctness_variant_count"] == 1
    assert stage.metrics["sensitivity_mutations"] == []
    assert stage.metrics["default_excluded_sensitivity_mutations"] == []
    assert stage.metrics["sensitivity_skipped"] is True
    assert stage.metrics["sensitivity_skip_reason"] == f"{SKIP_SENSITIVITY_ENV}=1"
    assert stage.metrics["sensitivity_variant_count"] == 0
    assert stage.metrics["sensitivity_variants"] == []


def test_run_merged_vllm_serving_stage_reports_served_model(monkeypatch) -> None:
    architecture = ArchitectureReport(
        base_model="Qwen/Qwen3.5-35B-A3B",
        model_key="qwen3_5_moe",
        handler_key="qwen3_5_moe",
        recommended_min_layers=4,
    )
    oracle_module = SimpleNamespace(
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs)
    )
    merged_module = SimpleNamespace(
        run_merged_vllm_serving=lambda case_config: SimpleNamespace(
            output_dir="/tmp/merged-serving",
            model_ids=["validation@0"],
            model_dump=lambda mode="json": {
                "base_model": "Qwen/Qwen3.5-35B-A3B",
                "served_model_name": "validation@0",
            },
        )
    )

    def _import_integration_module(name: str):
        if name == "integration.megatron.model_support.oracle_harness":
            return oracle_module
        if name == "integration.megatron.lora.merged_vllm_serving":
            return merged_module
        raise AssertionError(name)

    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        _import_integration_module,
    )

    stage = run_merged_vllm_serving_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=architecture,
    )

    assert stage.name == "merged_vllm_serving"
    assert stage.passed is True
    assert stage.metrics["base_model"] == "Qwen/Qwen3.5-35B-A3B"
    assert stage.metrics["served_model_name"] == "validation@0"
    assert "readable_summary" in stage.metrics
    assert stage.artifact_dir == "/tmp/merged-serving"
