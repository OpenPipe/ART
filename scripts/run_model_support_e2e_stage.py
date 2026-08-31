from __future__ import annotations

import argparse
import os
from pathlib import Path

from art.megatron.model_support.discovery import inspect_architecture
from art.megatron.model_support.spec import ArchitectureReport
from tests.integration.megatron.model_support import workflow, workflow_throughput
from tests.integration.megatron.model_support.workflow_fixtures import FIXTURE_PATH_ENV

_STAGE_DIR_ENV = "ART_MODEL_SUPPORT_WORKFLOW_STAGE_DIR"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one existing model-support E2E throughput stage.",
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--fixture-path", type=Path)
    parser.add_argument("--architecture-json", type=Path)
    parser.add_argument("--max-attempts", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        os.environ.setdefault(_STAGE_DIR_ENV, str(args.artifact_root / "stage"))
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    architecture_source = args.base_model
    if args.fixture_path is not None:
        fixture_path = args.fixture_path.resolve(strict=True)
        os.environ[FIXTURE_PATH_ENV] = str(fixture_path)
        architecture_source = str(fixture_path)

    architecture = (
        ArchitectureReport.model_validate_json(
            args.architecture_json.resolve(strict=True).read_text()
        )
        if args.architecture_json is not None
        else inspect_architecture(
            architecture_source,
            allow_unvalidated_arch=True,
        )
    )
    workflow_throughput._THROUGHPUT_MAX_ATTEMPTS = args.max_attempts
    result = workflow.run_e2e_throughput_stage(
        base_model=args.base_model,
        architecture=architecture,
        allow_unvalidated_arch=True,
    )
    (args.artifact_root / "result.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
