import os
from pathlib import Path
import re
import subprocess
import sys

from pydantic import BaseModel

from .artifacts import REPO_ROOT, TEST_ROOT, create_artifact_dir

DEFAULT_ATTEMPTS = 3
MAX_ATTEMPTS = 5


class TrainInfMismatchAttemptReport(BaseModel):
    attempt: int
    returncode: int
    stdout_path: str
    stderr_path: str
    passed_count: int
    failed_count: int
    skipped_count: int


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
    skipped_count: int
    attempt_count: int
    max_attempts: int
    attempts: list[TrainInfMismatchAttemptReport]


def _pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0}
    for line in reversed(output.splitlines()):
        matches = re.findall(r"(\d+) (passed|failed|skipped|error|errors)", line)
        if not matches:
            continue
        for count, kind in matches:
            if kind in {"error", "errors"}:
                counts["failed"] += int(count)
            else:
                counts[kind] += int(count)
        return counts
    return counts


def _attempt_limit() -> int:
    raw = os.environ.get("ART_TRAIN_INF_MISMATCH_ATTEMPTS")
    attempts = DEFAULT_ATTEMPTS if raw is None else int(raw)
    if attempts < 1:
        raise ValueError("ART_TRAIN_INF_MISMATCH_ATTEMPTS must be positive")
    return min(attempts, MAX_ATTEMPTS)


def run_train_inf_mismatch(
    *,
    base_model: str,
    allow_unvalidated_arch: bool = False,
) -> TrainInfMismatchReport:
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
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(TEST_ROOT / "test_live_real_path_output_parity.py"),
                "--tb=short",
            ],
            cwd=Path(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        counts = _pytest_counts(result.stdout + "\n" + result.stderr)
        selected = TrainInfMismatchAttemptReport(
            attempt=attempt,
            returncode=result.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            passed_count=counts["passed"],
            failed_count=counts["failed"],
            skipped_count=counts["skipped"],
        )
        attempts.append(selected)
        if result.returncode == 0:
            break
    if selected is None:
        raise RuntimeError("train/inf mismatch retry loop did not run")
    return TrainInfMismatchReport(
        base_model=base_model,
        passed=selected.returncode == 0,
        returncode=selected.returncode,
        artifact_dir=str(artifact_dir),
        test_root=str(TEST_ROOT),
        stdout_path=selected.stdout_path,
        stderr_path=selected.stderr_path,
        passed_count=selected.passed_count,
        failed_count=selected.failed_count,
        skipped_count=selected.skipped_count,
        attempt_count=len(attempts),
        max_attempts=max_attempts,
        attempts=attempts,
    )
