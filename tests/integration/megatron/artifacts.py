"""Shared helpers for integration tests that need durable per-run artifacts.

These helpers create a suite-owned artifacts/ directory keyed by test node id,
git commit, and run id, then write metadata that ties logs and JSON outputs back
to the exact committed code. They do not replace repo .local logs used by oracle
workflows that intentionally keep mutable local development output.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import uuid

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts"
SUITE_NAME = "Megatron integration tests"
LONGREP_MAX_CHARS = 12000
DEPLOYED_SOURCE_ROOT_ENV = "ART_DEPLOYED_SOURCE_ROOT"
DEPLOYED_SOURCE_COMMIT_ENV = "ART_DEPLOYED_SOURCE_COMMIT"
DEPLOYED_SOURCE_TREE_ENV = "ART_DEPLOYED_SOURCE_TREE"
DEPLOYED_SOURCE_BRANCH_ENV = "ART_DEPLOYED_SOURCE_BRANCH"


class GitRepoState(BaseModel):
    path: str
    commit: str
    tree: str | None = None
    dirty: bool
    status: tuple[str, ...] = ()


class ArtifactMetadata(BaseModel):
    commit: str
    branch: str
    test_nodeid: str
    created_at_utc: str
    python_executable: str
    artifact_dir: str


class PytestPhaseResult(BaseModel):
    when: str
    outcome: str
    duration: float
    location: tuple[str, int | None, str] | None = None
    longrepr: str | None = None


class PytestResult(BaseModel):
    commit: str
    branch: str
    test_nodeid: str
    created_at_utc: str
    phases: list[PytestPhaseResult]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sanitize_nodeid(nodeid: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid.strip())
    return collapsed.strip("._") or "unnamed_test"


def _short_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= LONGREP_MAX_CHARS:
        return text
    return text[-LONGREP_MAX_CHARS:]


def _native_worktree() -> bool:
    try:
        top_level = _git("rev-parse", "--show-toplevel")
    except subprocess.CalledProcessError:
        return False
    return Path(top_level).resolve() == REPO_ROOT.resolve()


def _deployed_source_state() -> GitRepoState:
    values = {
        "root": os.environ.get(DEPLOYED_SOURCE_ROOT_ENV),
        "commit": os.environ.get(DEPLOYED_SOURCE_COMMIT_ENV),
        "tree": os.environ.get(DEPLOYED_SOURCE_TREE_ENV),
    }
    if not all(values.values()):
        missing = sorted(name for name, value in values.items() if not value)
        raise RuntimeError(
            "source-only deployment requires a complete ART source attestation; "
            f"missing={missing}"
        )
    root = Path(str(values["root"])).resolve()
    if root != REPO_ROOT.resolve():
        raise RuntimeError(
            f"deployed ART source root differs: attested={root}, actual={REPO_ROOT.resolve()}"
        )
    commit, tree = str(values["commit"]), str(values["tree"])
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(
        r"[0-9a-f]{40}", tree
    ):
        raise RuntimeError(
            "deployed ART source commit and tree must be full Git object IDs"
        )
    return GitRepoState(path=str(root), commit=commit, tree=tree, dirty=False)


def _source_state(suite_name: str) -> GitRepoState:
    if not _native_worktree():
        return _deployed_source_state()
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if dirty:
        rendered = "\n".join(dirty)
        raise RuntimeError(
            f"{suite_name} require a fully committed worktree.\n"
            "Commit or remove these changes before running tests:\n"
            f"{rendered}"
        )
    return GitRepoState(
        path=str(REPO_ROOT),
        commit=_git("rev-parse", "HEAD"),
        tree=_git("rev-parse", "HEAD^{tree}"),
        dirty=False,
    )


def _source_branch() -> str:
    if _native_worktree():
        return _git("branch", "--show-current")
    return os.environ.get(DEPLOYED_SOURCE_BRANCH_ENV, "deployed")


def require_clean_git_state(suite_name: str) -> str:
    """Return the current commit after checking artifacts can be tied to clean code."""
    return _source_state(suite_name).commit


def git_state(path: Path) -> GitRepoState:
    if path.resolve() == REPO_ROOT.resolve() and not _native_worktree():
        return _deployed_source_state()
    status = tuple(
        line
        for line in _git("-C", str(path), "status", "--porcelain=v1").splitlines()
        if line
    )
    return GitRepoState(
        path=str(path),
        commit=_git("-C", str(path), "rev-parse", "HEAD"),
        tree=_git("-C", str(path), "rev-parse", "HEAD^{tree}"),
        dirty=bool(status),
        status=status,
    )


def pinned_git_state(suite_name: str) -> GitRepoState:
    return _source_state(suite_name)


def create_artifact_dir(
    test_nodeid: str,
    *,
    artifacts_root: Path,
    suite_name: str,
) -> Path:
    """Create a durable, git-addressed artifact directory for one test invocation."""
    commit = require_clean_git_state(suite_name)
    branch = _source_branch()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    artifact_dir = artifacts_root / _sanitize_nodeid(test_nodeid) / commit[:12] / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    metadata = ArtifactMetadata(
        commit=commit,
        branch=branch,
        test_nodeid=test_nodeid,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        python_executable=sys.executable,
        artifact_dir=str(artifact_dir),
    )
    (artifact_dir / "run_metadata.json").write_text(
        metadata.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_dir


def create_megatron_artifact_dir(test_nodeid: str) -> Path:
    return create_artifact_dir(
        test_nodeid,
        artifacts_root=ARTIFACTS_ROOT,
        suite_name=SUITE_NAME,
    )


def write_pytest_result(
    artifact_dir: Path,
    *,
    test_nodeid: str,
    reports: list[Any],
) -> Path:
    result = PytestResult(
        commit=require_clean_git_state(SUITE_NAME),
        branch=_source_branch(),
        test_nodeid=test_nodeid,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        phases=[
            PytestPhaseResult(
                when=str(report.when),
                outcome=str(report.outcome),
                duration=float(report.duration),
                location=(
                    str(report.location[0]),
                    int(report.location[1]) if report.location[1] is not None else None,
                    str(report.location[2]),
                )
                if getattr(report, "location", None) is not None
                else None,
                longrepr=_short_text(getattr(report, "longrepr", None)),
            )
            for report in reports
        ],
    )
    path = artifact_dir / "pytest_result.json"
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
