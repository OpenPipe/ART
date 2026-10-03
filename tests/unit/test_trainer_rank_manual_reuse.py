"""Exercise prospective recording, artifact verification and the real PR step."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/ci/trainer_rank_manual_reuse.py"
spec = importlib.util.spec_from_file_location("manual_reuse", SCRIPT)
assert spec and spec.loader
reuse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reuse)
WORKFLOW = yaml.safe_load((ROOT / reuse.WORKFLOW).read_text())


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True, timeout=5)
    for path in (*reuse.RUNTIME_INPUTS, reuse.WORKFLOW):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path + "\n")
    # Run the real helper from the workflow fixture's checked-out source.
    target = tmp_path / SCRIPT.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(SCRIPT.read_text())
    reuse.git("add", ".")
    reuse.git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-qm",
        "test: fixture source",
    )
    head = reuse.git("rev-parse", "HEAD")
    monkeypatch.setenv("GITHUB_REPOSITORY", "OpenPipe/ART")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    monkeypatch.setenv("GITHUB_WORKFLOW_SHA", head)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/fixture")
    monkeypatch.setenv("REUSE_BASE_SHA", head)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    owner = {
        "run_id": "42",
        "attempt": "1",
        "head": head,
        "repository": "OpenPipe/ART",
        "event": "workflow_dispatch",
        "cluster": "trainer-rank-gpu-42-1",
        "infra": "k8s/cks-wb3",
        "label": "a" * 32,
    }
    receipts = {
        "owner.json": owner,
        "job.json": {**owner, "job_id": 1},
        "result.json": {**owner, "job_id": 1, "status": "SUCCEEDED"},
        "cleanup.json": {
            **owner,
            "operations_succeeded": True,
            "physical_absence": "ABSENT",
            "creator_quiescence": "RETIRED",
        },
        "api-retired.json": {**owner, "creator_quiescence": "RETIRED"},
        "resources-cleanup.json": {
            **owner,
            "observed": [{"uid": "owned-pod"}],
            "remaining": [],
            "exact_uid_absence": "ABSENT",
        },
    }
    result = tmp_path / "result"
    result.mkdir()
    (result / "owner.json").write_text(json.dumps(owner))
    reuse.record(result)
    receipts["manual-reuse.json"] = json.loads(
        (result / "manual-reuse.json").read_text()
    )
    return {
        "pull": {
            "head": {
                "sha": head,
                "ref": "fixture",
                "repo": {"full_name": "OpenPipe/ART"},
            },
            "base": {"sha": head},
        },
        "run": {
            "id": 42,
            "run_attempt": 1,
            "event": "workflow_dispatch",
            "path": reuse.WORKFLOW,
            "head_sha": head,
            "head_branch": "fixture",
            "head_repository": {"full_name": "OpenPipe/ART"},
            "status": "completed",
            "conclusion": "success",
        },
        "jobs": [
            {
                "name": name,
                "head_sha": head,
                "run_attempt": 1,
                "status": "completed",
                "conclusion": "success",
            }
            for name in reuse.JOBS
        ],
        "artifact": {
            "id": 17,
            "name": "trainer-rank-result-42-1-manual",
            "expired": False,
            "workflow_run": {"id": 42, "head_sha": head, "head_branch": "fixture"},
        },
        "receipts": receipts,
    }


def archive(tmp_path, evidence):
    path = tmp_path / "evidence.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        for name, receipt in evidence["receipts"].items():
            bundle.writestr(name, json.dumps(receipt))
    evidence["artifact"].update(
        size_in_bytes=path.stat().st_size,
        digest="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return path


@pytest.mark.parametrize("infra", ["k8s/cks-wb3", "k8s/ext-collab2"])
def test_prospective_manual_validation_passes(tmp_path, evidence, infra):
    for receipt in evidence["receipts"].values():
        if "infra" in receipt:
            receipt["infra"] = infra
    evidence["receipts"]["manual-reuse.json"]["owner"]["infra"] = infra
    # The producer's supported context may differ from today's PR default.
    assert infra in reuse.verify(archive(tmp_path, evidence), evidence)


@pytest.mark.parametrize(
    "path,value",
    [
        (("pull", "base", "sha"), "b" * 40),
        (("pull", "head", "sha"), "b" * 40),
        (("pull", "head", "ref"), "different-branch"),
        (("pull", "head", "repo", "full_name"), "fork/ART"),
        (("run", "event"), "pull_request"),
        (("run", "path"), ".github/workflows/other.yml"),
        (("run", "head_sha"), "b" * 40),
        (("run", "head_branch"), "different-branch"),
        (("run", "run_attempt"), 2),
        (("run", "status"), "in_progress"),
        (("run", "conclusion"), "failure"),
        (("jobs", 0, "conclusion"), "failure"),
        (("jobs", 1, "conclusion"), "failure"),
        (("jobs", 2, "conclusion"), "skipped"),
        (("jobs", 3, "conclusion"), "cancelled"),
        (("jobs", 2, "run_attempt"), 2),
        (("artifact", "expired"), True),
        (("artifact", "workflow_run", "id"), 41),
        (("artifact", "name"), "trainer-rank-result-42-1-fabricated-base"),
        (("receipts", "manual-reuse.json", "workflow_sha"), "b" * 40),
        (
            (
                "receipts",
                "manual-reuse.json",
                "runtime_inputs",
                "megatron_runtime/uv.lock",
            ),
            "b" * 40,
        ),
        (("receipts", "owner.json", "infra"), "k8s/unapproved"),
        (("receipts", "result.json", "status"), "FAILED"),
        (("receipts", "result.json", "job_id"), 2),
        (("receipts", "result.json", "attempt"), "2"),
        (("receipts", "cleanup.json", "physical_absence"), "UNKNOWN"),
        (("receipts", "cleanup.json", "operations_succeeded"), False),
        (("receipts", "api-retired.json", "creator_quiescence"), "UNKNOWN"),
        (("receipts", "resources-cleanup.json", "remaining"), [{"uid": "owned-pod"}]),
    ],
)
def test_mismatched_stale_or_failed_evidence_refused(tmp_path, evidence, path, value):
    node = evidence
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        reuse.verify(archive(tmp_path, evidence), evidence)


@pytest.mark.parametrize(
    "missing", ["manual-reuse.json", "result.json", "cleanup.json"]
)
def test_old_or_incomplete_artifacts_refused(tmp_path, evidence, missing):
    del evidence["receipts"][missing]
    with pytest.raises(KeyError):
        reuse.verify(archive(tmp_path, evidence), evidence)


def test_changed_executed_workflow_refused(tmp_path, evidence, monkeypatch):
    (tmp_path / reuse.WORKFLOW).write_text("changed merge workflow\n")
    reuse.git("add", ".")
    reuse.git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "-qm",
        "test: changed merge workflow",
    )
    monkeypatch.setenv("GITHUB_WORKFLOW_SHA", reuse.git("rev-parse", "HEAD"))
    with pytest.raises(ValueError, match="workflow"):
        reuse.verify(archive(tmp_path, evidence), evidence)


def test_artifact_digest_checked(tmp_path, evidence):
    path = archive(tmp_path, evidence)
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        reuse.verify(path, evidence)


@pytest.mark.parametrize("base", ["", "main", "--help", "a" * 39])
def test_record_requires_explicit_exact_base(tmp_path, evidence, monkeypatch, base):
    monkeypatch.setenv("REUSE_BASE_SHA", base)
    with pytest.raises(ValueError, match="exact commit"):
        reuse.record(tmp_path / "result")


@pytest.mark.parametrize(
    "case",
    ["valid_manual", "prior_pr", "old_manual", "rerun", "bad_digest", "missing_job"],
)
def test_real_ready_step_verifies_artifact_before_reuse(tmp_path, evidence, case):
    accepted = case in {"valid_manual", "prior_pr"}
    if case == "prior_pr":
        evidence["run"]["event"] = "pull_request"
        evidence["artifact"]["name"] = (
            f"trainer-rank-result-42-1-{evidence['pull']['base']['sha']}"
        )
    if case == "old_manual":
        del evidence["receipts"]["manual-reuse.json"]
    if case == "missing_job":
        evidence["jobs"].pop()
    path = archive(tmp_path, evidence)
    if case == "bad_digest":
        evidence["artifact"]["digest"] = "sha256:" + "0" * 64
    current = copy.deepcopy(evidence["run"])
    if case == "rerun":
        current["run_attempt"] = 2
    fixture = tmp_path / "api.json"
    fixture.write_text(
        json.dumps({**evidence, "current": current, "archive": str(path)})
    )
    step = next(
        step
        for step in WORKFLOW["jobs"]["classify"]["steps"]
        if step.get("id") == "reuse"
    )
    harness = tmp_path / "harness.cjs"
    harness.write_text("""
const fs = require('fs');
const fixture = JSON.parse(fs.readFileSync(process.argv[2]));
const calls = [], outputs = {};
const actions = {
  listWorkflowRuns: async args => {
    calls.push(['runs', args.event]);
    return {data: {workflow_runs: args.event === fixture.run.event ? [fixture.run] : []}};
  },
  listJobsForWorkflowRun: async () => ({data: {jobs: fixture.jobs}}),
  listJobsForWorkflowRunAttempt: async args => {
    calls.push(['jobs', args.run_id, args.attempt_number]);
    return {data: {jobs: fixture.jobs}};
  },
  listWorkflowRunArtifacts: async () => ({data: {artifacts: [fixture.artifact]}}),
  downloadArtifact: async args => {
    calls.push(['download', args.artifact_id]);
    return {data: fs.readFileSync(fixture.archive)};
  },
  getWorkflowRun: async () => ({data: fixture.current}),
};
const context = {repo: {owner: 'OpenPipe', repo: 'ART'}, runId: 99,
                 payload: {pull_request: fixture.pull}};
const core = {setOutput: (key, value) => outputs[key] = value,
              info: () => {}, warning: () => {}};
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
new AsyncFunction('github', 'context', 'core', 'require', fs.readFileSync(process.argv[3], 'utf8'))(
  {rest: {actions}}, context, core, require
).then(() => process.stdout.write(JSON.stringify({outputs, calls})))
 .catch(error => {console.error(error); process.exit(1)});
""")
    source = tmp_path / "reuse.js"
    source.write_text(step["with"]["script"])
    result = subprocess.run(
        ["node", str(harness), str(fixture), str(source)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    observed = json.loads(result.stdout)
    if case == "prior_pr":
        assert observed["calls"] == [["runs", "pull_request"]]
    else:
        assert ["jobs", 42, 1] in observed["calls"]
        assert ["download", 17] in observed["calls"]
    assert observed["outputs"] == (
        {"reused": "true", "run_id": "42"} if accepted else {}
    )
    assert not list(tmp_path.glob("manual-reuse-*"))
    # Normal required-check shell executes and refuses a failed native fallback.
    gate = WORKFLOW["jobs"]["gate"]["steps"][0]["run"]
    outcome = subprocess.run(
        ["bash", "-c", gate],
        capture_output=True,
        text=True,
        timeout=5,
        env={
            **os.environ,
            "ENCLOSURE_RESULT": "success",
            "CLASSIFY_RESULT": "success",
            "REQUIRED": "true",
            "REUSED": observed["outputs"].get("reused", ""),
            "REUSED_RUN": observed["outputs"].get("run_id", ""),
            "RESULT": "skipped" if accepted else "failure",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        },
    )
    assert outcome.returncode == (0 if accepted else 1)
