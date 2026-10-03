"""Bind prospective manual validation to a base; verify it before PR reuse.

Runtime blobs identify source-controlled inputs, not the resolved image behind
the task's mutable container tag. Old manual artifacts have no base provenance.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import cast
import zipfile

WORKFLOW = ".github/workflows/trainer-rank-gpu.yml"
RUNTIME_INPUTS = (
    "scripts/ci/trainer-rank-gpu.sky.yaml",
    "scripts/ci/trainer-rank-gpu-tests.sh",
    "src/art/megatron/setup.sh",
    "megatron_runtime/pyproject.toml",
    "megatron_runtime/uv.lock",
    "pyproject.toml",
    "uv.lock",
)
JOBS = (
    "classify",
    "Qualify CI API enclosure (CPU only)",
    "Run on 2x H200",
    "trainer-rank-gpu-validation",
)


def require(condition: object, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def object_value(value: object) -> dict[str, object]:
    require(isinstance(value, dict), "Expected a JSON object")
    return cast(dict[str, object], value)


def sha(value: object) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
        "Expected an exact commit SHA",
    )
    return cast(str, value)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, timeout=20).strip()


def ensure_commit(commit: str) -> None:
    sha(commit)
    try:
        git("cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError:
        git("fetch", "--no-tags", "origin", commit)
        git("cat-file", "-e", f"{commit}^{{commit}}")


def runtime_inputs(head: str) -> dict[str, str]:
    return {path: git("rev-parse", f"{head}:{path}") for path in RUNTIME_INPUTS}


def record(root: Path) -> None:
    owner = object_value(json.loads((root / "owner.json").read_text()))
    base = sha(os.environ["REUSE_BASE_SHA"])
    head = sha(owner.get("head"))
    require(owner.get("event") == "workflow_dispatch", "Expected a manual run")
    require(
        git("rev-parse", "HEAD") == head == os.environ["GITHUB_WORKFLOW_SHA"],
        "Manual source and executed workflow must match",
    )
    ref = os.environ["GITHUB_REF"]
    require(ref.startswith("refs/heads/"), "Only branch dispatches can be reused")
    ensure_commit(base)
    # Capture the base before GPU work; never infer it from a later PR event.
    provenance = {
        "schema": 1,
        "base": base,
        "head_ref": ref.removeprefix("refs/heads/"),
        "workflow_sha": head,
        "workflow_blob": git("rev-parse", f"{head}:{WORKFLOW}"),
        "runtime_inputs": runtime_inputs(head),
        "owner": owner,
    }
    (root / "manual-reuse.json").write_text(json.dumps(provenance, indent=2) + "\n")


def verify(archive: Path, evidence: dict[str, object]) -> str:
    pull = object_value(evidence["pull"])
    source = object_value(pull["head"])
    head = sha(source["sha"])
    base = sha(object_value(pull["base"])["sha"])
    repository = object_value(source["repo"])["full_name"]
    require(
        repository == os.environ["GITHUB_REPOSITORY"],
        "Manual reuse requires the same repository",
    )
    run = object_value(evidence["run"])
    run_id, attempt = run["id"], run["run_attempt"]
    require(
        type(run_id) is int and run_id > 0 and type(attempt) is int and attempt > 0,
        "Invalid run attempt",
    )
    require(str(run_id) != os.environ["GITHUB_RUN_ID"], "Cannot reuse current run")
    require(
        run.get("event") == "workflow_dispatch"
        and run.get("status") == "completed"
        and run.get("conclusion") == "success"
        and run.get("path") == WORKFLOW
        and run.get("head_sha") == head
        and run.get("head_branch") == source["ref"]
        and object_value(run.get("head_repository")).get("full_name") == repository,
        "Manual run source or outcome mismatch",
    )
    jobs = evidence["jobs"]
    require(isinstance(jobs, list), "Missing native jobs")
    for name in JOBS:
        matches = [
            object_value(job)
            for job in cast(list[object], jobs)
            if object_value(job).get("name") == name
        ]
        require(len(matches) == 1, f"Missing or ambiguous job: {name}")
        job = matches[0]
        require(
            job.get("conclusion") == "success"
            and job.get("status") == "completed"
            and job.get("head_sha") == head
            and job.get("run_attempt") == attempt,
            f"Job did not pass on this source and attempt: {name}",
        )
    artifact = object_value(evidence["artifact"])
    require(
        artifact.get("expired") is False
        and artifact.get("name") == f"trainer-rank-result-{run_id}-{attempt}-manual",
        "Missing current manual artifact",
    )
    bound_run = object_value(artifact.get("workflow_run"))
    require(
        bound_run.get("id") == run_id
        and bound_run.get("head_sha") == head
        and bound_run.get("head_branch") == source["ref"],
        "Artifact belongs to a different run or source",
    )
    require(archive.stat().st_size <= 64 * 1024 * 1024, "Oversized artifact")
    require(
        artifact.get("digest")
        == "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
        "Artifact digest mismatch or unavailable",
    )
    with zipfile.ZipFile(archive) as bundle:
        require(
            len(bundle.namelist()) == len(set(bundle.namelist())),
            "Ambiguous artifact entries",
        )

        def read(name: str) -> dict[str, object]:
            require(bundle.getinfo(name).file_size <= 64 * 1024, "Oversized receipt")
            return object_value(json.loads(bundle.read(name)))

        owner = read("owner.json")
        require(
            all(
                owner.get(key) == value
                for key, value in {
                    "run_id": str(run_id),
                    "attempt": str(attempt),
                    "head": head,
                    "repository": repository,
                    "event": "workflow_dispatch",
                    "cluster": f"trainer-rank-gpu-{run_id}-{attempt}",
                }.items()
            ),
            "Native owner mismatch",
        )
        require(
            owner.get("infra") in {"k8s/cks-wb3", "k8s/ext-collab2"},
            "Unsupported producer context",
        )
        provenance = read("manual-reuse.json")
        workflow_sha = sha(os.environ["GITHUB_WORKFLOW_SHA"])
        ensure_commit(workflow_sha)
        workflow_blob = git("rev-parse", f"{head}:{WORKFLOW}")
        require(
            workflow_blob == git("rev-parse", f"{workflow_sha}:{WORKFLOW}"),
            "Producer and current executed workflows differ",
        )
        require(
            provenance
            == {
                "schema": 1,
                "base": base,
                "head_ref": source["ref"],
                "workflow_sha": head,
                "workflow_blob": workflow_blob,
                "runtime_inputs": runtime_inputs(head),
                "owner": owner,
            },
            "Manual base, workflow, runtime inputs or owner mismatch",
        )

        def bound(name: str) -> dict[str, object]:
            receipt = read(name)
            require(
                all(receipt.get(key) == value for key, value in owner.items()),
                f"Native receipt owner mismatch: {name}",
            )
            return receipt

        job = bound("job.json")["job_id"]
        result = bound("result.json")
        require(
            type(job) is int
            and job > 0
            and result.get("job_id") == job
            and result.get("status") == "SUCCEEDED",
            "Native result did not succeed",
        )
        cleanup = bound("cleanup.json")
        require(
            cleanup.get("operations_succeeded") is True
            and cleanup.get("physical_absence") == "ABSENT"
            and cleanup.get("creator_quiescence") == "RETIRED",
            "Native cleanup is unproven",
        )
        require(
            bound("api-retired.json").get("creator_quiescence") == "RETIRED",
            "Native API retirement is unproven",
        )
        resources = bound("resources-cleanup.json")
        require(
            resources.get("observed")
            and resources.get("remaining") == []
            and resources.get("exact_uid_absence") == "ABSENT",
            "Retained resource absence is unproven",
        )
    return f"Verified manual run {run_id}, attempt {attempt}, {owner['infra']}, base {base}."


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(Path(sys.argv[2]))
    elif sys.argv[1] == "verify":
        print(verify(Path(sys.argv[2]), object_value(json.load(sys.stdin))))
    else:
        raise ValueError("Expected record or verify")
