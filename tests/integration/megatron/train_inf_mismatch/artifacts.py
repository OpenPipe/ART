from __future__ import annotations

import os
from pathlib import Path

from ..artifacts import REPO_ROOT
from ..artifacts import create_artifact_dir as _create_artifact_dir
from ..artifacts import require_clean_git_state as _require_clean_git_state

TEST_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = TEST_ROOT / "artifacts"
ARTIFACTS_ROOT_ENV = "ART_TRAIN_INF_MISMATCH_ARTIFACTS_ROOT"
SUITE_NAME = "Megatron train/inf mismatch tests"


def require_clean_git_state() -> str:
    return _require_clean_git_state(SUITE_NAME)


def create_artifact_dir(
    test_nodeid: str,
    artifacts_root: Path | None = None,
) -> Path:
    if artifacts_root is None:
        raw_root = os.environ.get(ARTIFACTS_ROOT_ENV)
        artifacts_root = Path(raw_root) if raw_root is not None else ARTIFACTS_ROOT
    return _create_artifact_dir(
        test_nodeid,
        artifacts_root=artifacts_root,
        suite_name=SUITE_NAME,
    )
