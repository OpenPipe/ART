import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Literal

from pydantic import BaseModel

from .artifacts import REPO_ROOT, TEST_ROOT, create_artifact_dir

DEFAULT_ATTEMPTS = 3
MAX_ATTEMPTS = 5
ATTEMPT_ASSERTION_EXIT_CODE = 1
ATTEMPT_ERROR_EXIT_CODE = 2

_TRANSIENT_STARTUP_ERRORS = (
    "address already in use",
    "brokenpipeerror",
    "connection refused",
    "connectionrefusederror",
    "connection reset by peer",
    "connectionreseterror",
    "distnetworkerror",
    "ncclremoteerror",
    "ncclsystemerror",
    "timed out waiting for",
    "timeouterror",
)


class TrainInfMismatchWorkerResult(BaseModel):
    outcome: Literal["passed", "failed", "error", "skipped"]
    artifact_dir: str | None = None
    comparison_completed: bool = False
    exception_type: str | None = None
    exception_message: str | None = None


class TrainInfMismatchAttemptReport(BaseModel):
    attempt: int
    returncode: int
    stdout_path: str
    stderr_path: str
    passed_count: int
    failed_count: int
    error_count: int
    skipped_count: int
    retryable: bool
    duration_s: float


class TrainInfMismatchReport(BaseModel):
    base_model: str
    passed: bool
    returncode: int
    artifact_dir: str
    test_root: str
    stdout_path: str
    stderr_path: str
    passed_count: int
    failed_count: int
    error_count: int
    skipped_count: int
    attempt_count: int
    max_attempts: int
    attempts: list[TrainInfMismatchAttemptReport]
    duration_s: float


def _attempt_counts(
    result: TrainInfMismatchWorkerResult | None,
    *,
    returncode: int,
) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    expected_returncode = (
        {
            "passed": 0,
            "failed": ATTEMPT_ASSERTION_EXIT_CODE,
            "error": ATTEMPT_ERROR_EXIT_CODE,
            "skipped": 0,
        }.get(result.outcome)
        if result is not None
        else None
    )
    if result is None or returncode != expected_returncode:
        counts["errors"] = 1
    elif result.outcome == "error":
        counts["errors"] = 1
    else:
        counts[f"{result.outcome}"] = 1
    return counts


def _retryable_attempt_failure(
    *,
    returncode: int,
    result: TrainInfMismatchWorkerResult | None,
    output: str,
) -> bool:
    if result is not None:
        if result.outcome == "failed":
            return result.comparison_completed
        if result.outcome != "error" or result.comparison_completed:
            return False
    if returncode in {-9, -15}:
        return True
    details = "\n".join(
        value
        for value in (
            result.exception_type if result is not None else None,
            result.exception_message if result is not None else None,
            output,
        )
        if value
    ).lower()
    return any(marker in details for marker in _TRANSIENT_STARTUP_ERRORS)


def _attempt_limit() -> int:
    raw = os.environ.get("ART_TRAIN_INF_MISMATCH_ATTEMPTS")
    attempts = DEFAULT_ATTEMPTS if raw is None else int(raw)
    if attempts < 1:
        raise ValueError("ART_TRAIN_INF_MISMATCH_ATTEMPTS must be positive")
    return min(attempts, MAX_ATTEMPTS)


def _run_attempt(
    command: list[str], *, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_train_inf_mismatch(
    *,
    base_model: str,
    allow_unvalidated_arch: bool = False,
) -> TrainInfMismatchReport:
    started = time.monotonic()
    artifact_dir = create_artifact_dir("workflow::train_inf_mismatch")
    max_attempts = _attempt_limit()
    env = os.environ.copy()
    env["BASE_MODEL"] = base_model
    env["ART_RUN_TRAIN_INF_MISMATCH_LIVE"] = "1"
    env["ART_TRAIN_INF_MISMATCH_BASE_MODEL"] = base_model
    env["ART_TRAIN_INF_MISMATCH_ALLOW_UNVALIDATED_ARCH"] = (
        "1" if allow_unvalidated_arch else "0"
    )
    env["ART_REAL_PATH_MAX_COMPLETION_TOKENS"] = "16"
    env.setdefault("ART_TRAIN_INF_MISMATCH_VLLM_GPU_MEMORY_UTILIZATION", "0.50")
    existing_pythonpath = env.get("PYTHONPATH")
    tests_dir = str(REPO_ROOT / "tests")
    env["PYTHONPATH"] = (
        tests_dir
        if not existing_pythonpath
        else f"{tests_dir}{os.pathsep}{existing_pythonpath}"
    )
    attempts: list[TrainInfMismatchAttemptReport] = []
    selected: TrainInfMismatchAttemptReport | None = None
    for attempt in range(1, max_attempts + 1):
        stdout_path = artifact_dir / f"attempt_{attempt}_pytest_stdout.txt"
        stderr_path = artifact_dir / f"attempt_{attempt}_pytest_stderr.txt"
        result_path = artifact_dir / f"attempt_{attempt}_result.json"
        attempt_started = time.monotonic()
        result = _run_attempt(
            [
                sys.executable,
                "-m",
                "integration.megatron.train_inf_mismatch."
                "test_live_real_path_output_parity",
                "--workflow-attempt-result",
                str(result_path),
            ],
            cwd=Path(REPO_ROOT),
            env=env,
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        try:
            worker_result = TrainInfMismatchWorkerResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            worker_result = None
        output = result.stdout + "\n" + result.stderr
        counts = _attempt_counts(worker_result, returncode=result.returncode)
        retryable = _retryable_attempt_failure(
            returncode=result.returncode,
            result=worker_result,
            output=output,
        )
        selected = TrainInfMismatchAttemptReport(
            attempt=attempt,
            returncode=result.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            passed_count=counts["passed"],
            failed_count=counts["failed"],
            error_count=counts["errors"],
            skipped_count=counts["skipped"],
            retryable=retryable,
            duration_s=time.monotonic() - attempt_started,
        )
        attempts.append(selected)
        if (
            result.returncode == 0
            and selected.passed_count > 0
            and selected.skipped_count == 0
        ):
            break
        if not retryable:
            break
    if selected is None:
        raise RuntimeError("train/inf mismatch retry loop did not run")
    passed = (
        selected.returncode == 0
        and selected.passed_count > 0
        and selected.failed_count == 0
        and selected.error_count == 0
        and selected.skipped_count == 0
    )
    return TrainInfMismatchReport(
        base_model=base_model,
        passed=passed,
        returncode=selected.returncode,
        artifact_dir=str(artifact_dir),
        test_root=str(TEST_ROOT),
        stdout_path=selected.stdout_path,
        stderr_path=selected.stderr_path,
        passed_count=selected.passed_count,
        failed_count=selected.failed_count,
        error_count=selected.error_count,
        skipped_count=selected.skipped_count,
        attempt_count=len(attempts),
        max_attempts=max_attempts,
        attempts=attempts,
        duration_s=time.monotonic() - started,
    )
