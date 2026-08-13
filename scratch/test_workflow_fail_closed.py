from __future__ import annotations

from pathlib import Path

from integration.megatron.model_support import workflow_scheduler
from integration.megatron.model_support import workflow_stage_worker as worker
from integration.megatron.model_support.validation_spec import (
    ValidationReport,
    ValidationStageResult,
)
from integration.megatron.model_support.workflow_runtime import (
    WorkflowDevice,
    WorkflowOperation,
    WorkflowOperationFailed,
    WorkflowRuntimeKey,
    compile_workflow,
    execute_workflow,
)
from integration.megatron.model_support.workflow_stage_worker import (
    WorkflowStageWorkerItem,
    WorkflowStageWorkerSession,
)

from art.megatron.model_support.spec import ArchitectureReport


def _runtime(name: str, *, handler: str | None = None) -> WorkflowRuntimeKey:
    return WorkflowRuntimeKey(
        source_fingerprint="source",
        handler=handler or name,
        fixture="fixture",
        kind="cpu",
        mode=name,
    )


def _worker_request(tmp_path: Path) -> WorkflowStageWorkerSession:
    architecture_json = tmp_path / "architecture.json"
    architecture_json.write_text(
        ArchitectureReport(
            base_model="model",
            model_key="model",
            handler_key="model",
            recommended_min_layers=1,
        ).model_dump_json(),
        encoding="utf-8",
    )
    stages = ("hf_parity", "packing_invariance")
    for stage in stages:
        (tmp_path / stage).mkdir()
    return WorkflowStageWorkerSession(
        base_model="model",
        architecture_json=str(architecture_json),
        items=tuple(
            WorkflowStageWorkerItem(
                stage=stage,
                stage_dir=str(tmp_path / stage),
                output_json=str(tmp_path / stage / "stage_result.json"),
                environment={},
            )
            for stage in stages
        ),
    )


def test_stage_worker_stops_at_first_failed_result(monkeypatch, tmp_path: Path) -> None:
    request = _worker_request(tmp_path)
    called: list[str] = []

    def run(stage: str, *, passed: bool):
        def stage_runner(**_kwargs) -> ValidationStageResult:
            called.append(stage)
            return ValidationStageResult(
                name=stage,
                passed=passed,
                metrics={"error": "root failure"} if not passed else {},
            )

        return stage_runner

    monkeypatch.setitem(
        worker._STAGE_RUNNERS, "hf_parity", run("hf_parity", passed=False)
    )
    monkeypatch.setitem(
        worker._STAGE_RUNNERS,
        "packing_invariance",
        run("packing_invariance", passed=True),
    )

    worker._run_session(request)

    assert called == ["hf_parity"]
    assert Path(request.items[0].output_json).is_file()
    assert not Path(request.items[1].output_json).exists()


def test_executor_blocks_failed_dependency_transitively() -> None:
    operations = (
        WorkflowOperation(id="root", stage="root", runtime=_runtime("root")),
        WorkflowOperation(
            id="child",
            stage="child",
            runtime=_runtime("child"),
            dependencies=("root",),
        ),
        WorkflowOperation(
            id="grandchild",
            stage="grandchild",
            runtime=_runtime("grandchild"),
            dependencies=("child",),
        ),
        WorkflowOperation(
            id="independent", stage="independent", runtime=_runtime("independent")
        ),
    )
    called: list[str] = []

    def runner(session, _placement):
        operation_id = session.operations[0].id
        called.append(operation_id)
        if operation_id == "root":
            raise WorkflowOperationFailed(operation_id)
        return operation_id

    execution = execute_workflow(
        compile_workflow(operations),
        devices=[WorkflowDevice(host="local", gpu="0")],
        runner=runner,
    )

    assert set(called) == {"root", "independent"}
    assert execution.results["session_000"].failed_operation_id == "root"
    assert execution.blocked_by_failed_operations == {
        "session_001": ("root",),
        "session_002": ("root",),
    }


class _Fixture:
    def environment(self, _stage: str | None = None) -> dict[str, str]:
        return {"ART_MODEL_SUPPORT_FIXTURE_PATH": "/tmp/model"}


class _Prepared:
    def __init__(self, run_dir: Path) -> None:
        self.report = ValidationReport(
            git={"commit": "test"},
            base_model="model",
            model_key="model",
            stages=[
                ValidationStageResult(name=stage)
                for stage in ("hf_parity", "packing_invariance", "length_trainability")
            ],
        )
        self.architecture = ArchitectureReport(
            base_model="model",
            model_key="model",
            handler_key="model",
            recommended_min_layers=1,
        )
        self.fixture = _Fixture()
        self.run_dir = run_dir
        self.output_json = None
        self.allow_unvalidated_arch = False
        self.include_sensitivity = None

    def record(self, result: ValidationStageResult) -> None:
        stage = next(stage for stage in self.report.stages if stage.name == result.name)
        stage.passed = result.passed
        stage.skipped = result.skipped
        stage.metrics = dict(result.metrics)
        stage.artifact_dir = result.artifact_dir

    def record_fixture_metric(self, _metrics: dict[str, object]) -> None:
        pass


class _Forkservers:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def run(self, _host: str, *, request_json: Path, **_kwargs):
        request = WorkflowStageWorkerSession.model_validate_json(
            Path(request_json).read_text(encoding="utf-8")
        )
        stages = tuple(item.stage for item in request.items)
        self.calls.append(stages)
        for item in request.items:
            result = ValidationStageResult(name=item.stage, passed=True)
            if item.stage == "hf_parity":
                result = ValidationStageResult(
                    name=item.stage,
                    passed=False,
                    metrics={"error": "sentinel root failure"},
                )
            Path(item.output_json).write_text(
                result.model_dump_json(), encoding="utf-8"
            )
            if not result.passed:
                break
        return {"returncode": 0, "child_wall_s": 0.01}

    def metrics(self, _host: str) -> dict[str, float]:
        return {}


def test_scheduler_records_failure_and_skips_blocked_operations(
    monkeypatch, tmp_path: Path
) -> None:
    runtime = _runtime("shared", handler="model")
    operations = (
        WorkflowOperation(id="model:hf_parity", stage="hf_parity", runtime=runtime),
        WorkflowOperation(
            id="model:packing_invariance",
            stage="packing_invariance",
            runtime=runtime,
            dependencies=("model:hf_parity",),
        ),
        WorkflowOperation(
            id="model:length_trainability",
            stage="length_trainability",
            runtime=_runtime("dependent", handler="model"),
            dependencies=("model:packing_invariance",),
        ),
    )
    plan = compile_workflow(operations)
    prepared = _Prepared(tmp_path / "run")
    forkservers = _Forkservers()
    monkeypatch.setattr(
        workflow_scheduler, "compile_prepared_workflows", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(
        workflow_scheduler,
        "_visible_devices",
        lambda: [WorkflowDevice(host="local", gpu="0")],
    )

    reports = workflow_scheduler.run_prepared_workflows(
        [prepared], forkservers=forkservers
    )

    assert forkservers.calls == [("hf_parity", "packing_invariance")]
    stages = {stage.name: stage for stage in reports[0].stages}
    assert stages["hf_parity"].metrics["error"] == "sentinel root failure"
    for stage_name in ("packing_invariance", "length_trainability"):
        stage = stages[stage_name]
        assert stage.skipped is True
        assert stage.metrics["workflow_failed_dependencies"] == ["model:hf_parity"]
        evidence = stage.metrics["workflow_failure_evidence"]["model:hf_parity"]
        assert evidence["error"] == "sentinel root failure"
        assert Path(evidence["stage_result_json"]).is_file()
