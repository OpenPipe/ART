import os
from types import SimpleNamespace

import pytest

from art.megatron.model_support.spec import (
    ArchitectureReport,
    LayerFamilyInstance,
)

from .validation_spec import ValidationReport, ValidationStageResult
from .workflow import (
    MANDATORY_VALIDATION_STAGES,
    NATIVE_VLLM_LORA_STAGE,
    SKIP_SENSITIVITY_ENV,
    _inspect_architecture_for_workflow,
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
    run_packed_position_ids_stage,
    run_train_inf_mismatch_stage,
    run_yes_no_trainability_stage,
    validated_architecture_representative_models,
)


@pytest.fixture(autouse=True)
def _stub_pinned_git_state(monkeypatch) -> None:
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
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen/Qwen3.5-27B",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
        "openai/gpt-oss-20b",
    ]


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
        "tests.integration.megatron.model_support.workflow.inspect_architecture",
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
        return ValidationReport(
            git={},
            base_model=base_model,
            model_key="qwen3_dense",
            stages=[
                ValidationStageResult(
                    name="train_inf_mismatch",
                    passed=base_model != "Qwen/Qwen3-32B",
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

    assert calls == ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-32B"]
    assert report.passed is False
    assert [item.base_model for item in report.reports] == calls


def test_build_validation_report_populates_architecture_stage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.inspect_architecture",
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
            "packed_position_ids": ValidationStageResult(
                name="packed_position_ids",
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
                artifact_dir="/tmp/packed-position-ids",
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
    assert dependency_stage.metrics == {"transformers": "5.2.0"}
    architecture_stage = next(
        stage for stage in report.stages if stage.name == "architecture_discovery"
    )
    assert architecture_stage.passed is True
    assert architecture_stage.metrics == {
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
    assert hf_parity_stage.metrics == {"signal": "pass", "requested_num_layers": 1}
    assert hf_parity_stage.artifact_dir == "/tmp/hf_parity"
    lora_coverage_stage = next(
        stage for stage in report.stages if stage.name == "lora_coverage"
    )
    assert lora_coverage_stage.passed is True
    assert lora_coverage_stage.metrics == {"wrapped_adapter_prefix_count": 12}
    mismatch_stage = next(
        stage for stage in report.stages if stage.name == "train_inf_mismatch"
    )
    assert mismatch_stage.passed is True
    assert mismatch_stage.metrics == {"passed_count": 1, "failed_count": 0}
    assert mismatch_stage.artifact_dir == "/tmp/train-inf-mismatch"
    correctness_stage = next(
        stage for stage in report.stages if stage.name == "correctness_sensitivity"
    )
    assert correctness_stage.passed is True
    assert correctness_stage.metrics == {
        "correctness_variant_count": 4,
        "sensitivity_variant_count": 9,
    }
    assert correctness_stage.artifact_dir == "/tmp/correctness"
    merged_stage = next(
        stage for stage in report.stages if stage.name == "merged_vllm_serving"
    )
    assert merged_stage.passed is True
    assert merged_stage.metrics == {"served_model_name": "validation@0"}
    assert merged_stage.artifact_dir == "/tmp/merged-serving"
    chat_template_stage = next(
        stage for stage in report.stages if stage.name == "chat_template_rollout"
    )
    assert chat_template_stage.passed is True
    assert chat_template_stage.metrics == {
        "passed": True,
        "scenario_count": 6,
        "failed_scenarios": [],
    }
    assert chat_template_stage.artifact_dir == "/tmp/chat-template"
    position_id_stage = next(
        stage for stage in report.stages if stage.name == "packed_position_ids"
    )
    assert position_id_stage.passed is True
    assert position_id_stage.metrics == {
        "num_layers": 4,
        "scenarios": [
            {
                "name": "stop_early",
                "matched": True,
                "checked_token_count": 40,
            }
        ],
    }
    assert position_id_stage.artifact_dir == "/tmp/packed-position-ids"
    trainability_stage = next(
        stage for stage in report.stages if stage.name == "length_trainability"
    )
    assert trainability_stage.passed is True
    assert trainability_stage.metrics == {
        "latest_step": 4,
        "best_train_abs_error": 1.0,
    }
    assert trainability_stage.artifact_dir == "/tmp/length-trainability"
    assert all(stage.name != "yes_no_trainability" for stage in report.stages)
    native_vllm_lora_stage = next(
        stage for stage in report.stages if stage.name == "native_vllm_lora"
    )
    assert native_vllm_lora_stage.passed is True
    assert native_vllm_lora_stage.metrics == {
        "rollout_weights_mode": "lora",
        "step0_name": "validation@0",
        "step1_name": "validation@1",
        "model_ids_before": ["validation@0"],
        "model_ids_after": ["validation@0", "validation@1"],
        "step0_served": True,
        "step1_served": True,
    }
    assert native_vllm_lora_stage.artifact_dir == "/tmp/native-vllm-lora"


def test_build_validation_report_only_stage_skips_other_stages(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.inspect_architecture",
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
    assert skipped.metrics == {
        "skipped": True,
        "reason": "--only-stage=length_trainability",
    }


def test_build_validation_report_captures_hf_parity_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.inspect_architecture",
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
    assert hf_parity_stage.metrics == {"error": "AssertionError: parity failed"}
    assert hf_parity_stage.artifact_dir is None


def test_build_validation_report_captures_lora_coverage_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.inspect_architecture",
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
    assert lora_coverage_stage.metrics == {
        "error": "RuntimeError: missing wrapped targets"
    }


def test_build_validation_report_writes_incremental_output_and_stops(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.inspect_architecture",
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
    assert skipped_stage.metrics == {
        "skipped": True,
        "reason": "stopped after lora_coverage failed",
    }


def test_assess_minimal_layer_coverage_reports_missing_families(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow.inspect_architecture",
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
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        selected_suite_topologies=lambda *, is_moe: [
            SimpleNamespace(world_size=lambda: 1, slug=lambda: "tp1"),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "tp2"),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "dp2"),
            SimpleNamespace(world_size=lambda: 4, slug=lambda: "tp2_dp2"),
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
        run_suite=lambda case_config, max_world_size: (
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
    assert result.metrics["required_gpu_count"] == 1
    assert result.metrics["correctness_variant_count"] == 1
    assert result.metrics["correctness_excluded_topologies"] == []
    assert result.metrics["sensitivity_mutations"] == ["skip_finalize"]
    assert case_configs[0].is_moe is False


def test_run_yes_no_trainability_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(
            run_yes_no_trainability=lambda *, base_model, allow_unvalidated_arch=False: (
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
            )
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


def test_run_packed_position_ids_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        "tests.integration.megatron.model_support.workflow._import_integration_module",
        lambda name: SimpleNamespace(
            run_packed_position_ids=lambda *, base_model, num_layers, allow_unvalidated_arch=False: (
                SimpleNamespace(
                    output_dir="/tmp/packed-position-ids",
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

    result = run_packed_position_ids_stage(
        base_model="Qwen/Qwen3.5-35B-A3B",
        architecture=ArchitectureReport(
            base_model="Qwen/Qwen3.5-35B-A3B",
            model_key="qwen3_5_moe",
            handler_key="qwen3_5_moe",
            recommended_min_layers=4,
        ),
    )

    assert result.passed is True
    assert result.artifact_dir == "/tmp/packed-position-ids"


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
        recommended_min_layers=4,
    )
    oracle_module = SimpleNamespace(
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        selected_suite_topologies=lambda *, is_moe: [
            SimpleNamespace(world_size=lambda: 1, slug=lambda: "tp1"),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "tp2"),
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
        run_suite=lambda case_config, max_world_size: [
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
    assert stage.metrics["available_gpu_count"] == 2
    assert stage.metrics["required_gpu_count"] == 1
    assert stage.metrics["correctness_variant_count"] == 1
    assert stage.metrics["sensitivity_skipped"] is False
    assert stage.metrics["sensitivity_skip_reason"] is None
    assert stage.metrics["sensitivity_variant_count"] == 1
    assert stage.artifact_dir == "/tmp/oracle"


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
        OracleCaseConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        selected_suite_topologies=lambda *, is_moe: [
            SimpleNamespace(world_size=lambda: 1, slug=lambda: "tp1"),
            SimpleNamespace(world_size=lambda: 2, slug=lambda: "tp2"),
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
        run_suite=lambda case_config, max_world_size: [
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
    assert stage.metrics["required_gpu_count"] == 1
    assert stage.metrics["correctness_variant_count"] == 1
    assert stage.metrics["sensitivity_mutations"] == []
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
    assert stage.metrics == {
        "base_model": "Qwen/Qwen3.5-35B-A3B",
        "served_model_name": "validation@0",
    }
    assert stage.artifact_dir == "/tmp/merged-serving"
