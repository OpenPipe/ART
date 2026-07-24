#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REQUIRED_FILES = {
    "manifest.json",
    "request-trace.jsonl",
    "tensor-manifest.json",
    "samples.jsonl",
    "summary.json",
    "stdout.log",
    "stderr.log",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=("idle", "fixed-load"), required=True)
    args = parser.parse_args()

    missing = sorted(
        name for name in REQUIRED_FILES if not (args.input / name).is_file()
    )
    if missing:
        raise RuntimeError(f"Refusing to upload incomplete evidence: {missing}")
    manifest = json.loads((args.input / "manifest.json").read_text())
    claimed = set(manifest.get("required_result_files", ()))
    if claimed != REQUIRED_FILES:
        raise RuntimeError(
            f"Result manifest claims {sorted(claimed)}; expected {sorted(REQUIRED_FILES)}"
        )
    summary = json.loads((args.input / "summary.json").read_text())
    acceptance = summary.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("passed") is not True:
        raise RuntimeError("Refusing to upload a result that failed acceptance gates")
    additional = set(manifest.get("additional_result_files", ()))
    missing_additional = sorted(
        name for name in additional if not (args.input / name).is_file()
    )
    if missing_additional:
        raise RuntimeError(
            f"Result manifest claims missing additional files: {missing_additional}"
        )

    import wandb

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wb-training"),
        project=os.environ.get("WANDB_PROJECT", "bench"),
        job_type="lora-update-replay",
        config={"mode": args.mode, "artifact_schema_version": 1},
    )
    assert run is not None
    artifact = wandb.Artifact(
        name=f"art-lora-update-replay-{args.mode}",
        type="lora-update-replay-result",
        metadata=summary,
    )
    for name in sorted(REQUIRED_FILES | additional):
        artifact.add_file(str(args.input / name), name=name)
    logged = run.log_artifact(artifact)
    logged.wait()
    artifact_ref = f"{logged.entity}/{logged.project}/{logged.name}"
    run.summary["result_artifact"] = artifact_ref
    run.summary["result_artifact_digest"] = logged.digest
    run.finish()
    print(f"WANDB_ARTIFACT={artifact_ref}")
    print(f"WANDB_ARTIFACT_DIGEST={logged.digest}")


if __name__ == "__main__":
    main()
