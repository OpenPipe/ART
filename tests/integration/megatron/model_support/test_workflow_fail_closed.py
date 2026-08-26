from __future__ import annotations

from pathlib import Path
from typing import cast

from art.megatron.model_support.spec import ArchitectureReport

from . import workflow_scheduler
from .validation_spec import (
    ValidationReport,
    ValidationStageResult,
)
from .workflow import CORRECTNESS_REFERENCE_STAGE
from .workflow_forkserver import (
    WorkflowForkserverPool,
)
from .workflow_runtime import (
    WorkflowDevice,
    WorkflowOperation,
    WorkflowOperationFailed,
    WorkflowRuntimeKey,
    compile_workflow,
    execute_workflow,
)
from .workflow_scheduler import PreparedWorkflow
from .workflow_stage_worker import WorkflowStageWorkerSession


def _runtime(name: str, *, handler: str | None = None) -> WorkflowRuntimeKey:
    return WorkflowRuntimeKey(
        source_fingerprint="source",
        handler=handler or name,
        fixture="fixture",
        kind="cpu",
        mode=name,
    )


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
    def __init__(self, run_dir: Path, stages: tuple[str, ...] | None = None) -> None:
        stages = stages or (
            "hf_parity",
            "packing_invariance",
            "length_trainability",
        )
        self.report = ValidationReport(
            git={"commit": "test"},
            base_model="model",
            model_key="model",
            stages=[ValidationStageResult(name=stage) for stage in stages],
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
    def __init__(self, *, fail: str | None = "hf_parity", stop: bool = True) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail = fail
        self.stop = stop

    def run(self, _host: str, *, request_json: Path, **_kwargs):
        request = WorkflowStageWorkerSession.model_validate_json(
            Path(request_json).read_text(encoding="utf-8")
        )
        stages = tuple(item.stage for item in request.items)
        self.calls.append(stages)
        for item in request.items:
            result = ValidationStageResult(name=item.stage, passed=True)
            if item.stage == self.fail:
                result = ValidationStageResult(
                    name=item.stage,
                    passed=False,
                    metrics={"error": "sentinel root failure"},
                )
            Path(item.output_json).write_text(
                result.model_dump_json(), encoding="utf-8"
            )
            if not result.passed and self.stop:
                break
        return {"returncode": 0, "child_wall_s": 0.01}

    def metrics(self, _host: str) -> dict[str, float]:
        return {}


def _run(prepared: _Prepared, forkservers: _Forkservers) -> ValidationReport:
    return workflow_scheduler.run_prepared_workflows(
        [cast(PreparedWorkflow, prepared)],
        forkservers=cast(WorkflowForkserverPool, forkservers),
    )[0]


def test_hidden_correctness_failure_fails_visible_owner(
    monkeypatch, tmp_path: Path
) -> None:
    reference = WorkflowOperation(
        id=f"model:{CORRECTNESS_REFERENCE_STAGE}",
        stage=CORRECTNESS_REFERENCE_STAGE,
        runtime=_runtime("reference", handler="model"),
    )
    visible = WorkflowOperation(
        id="model:correctness_sensitivity",
        stage="correctness_sensitivity",
        runtime=_runtime("variants", handler="model"),
        dependencies=(reference.id,),
    )
    plan = compile_workflow((reference, visible))
    prepared = _Prepared(tmp_path / "run", ("correctness_sensitivity",))
    forkservers = _Forkservers(fail=CORRECTNESS_REFERENCE_STAGE)
    monkeypatch.setattr(
        workflow_scheduler, "compile_prepared_workflows", lambda *_args, **_kwargs: plan
    )
    monkeypatch.setattr(
        workflow_scheduler,
        "_visible_devices",
        lambda: [WorkflowDevice(host="local", gpu="0")],
    )

    report = _run(prepared, forkservers)

    owner = report.stages[0]
    assert owner.name == "correctness_sensitivity"
    assert owner.passed is False and owner.skipped is False
    assert owner.metrics["blocked"] is True
    assert owner.metrics["workflow_failed_dependencies"] == [reference.id]
    assert report.passed is False
