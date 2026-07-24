#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REQUIRED_FILES = {
    "manifest.json",
    "request-trace.json",
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
        metadata=json.loads((args.input / "summary.json").read_text()),
    )
    artifact.add_dir(str(args.input))
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
