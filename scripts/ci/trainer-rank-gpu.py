"""Run the CI job without making an attached log stream its result gate.

SkyPilot 0.12 returns an actual job ID from launch. Each SDK operation runs in a
bounded child; only that job's SUCCEEDED status passes. The workflow EXIT trap
still owns cluster teardown, including launch failures before a job ID is known.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

NONTERMINAL = {"INIT", "PENDING", "SETTING_UP", "RUNNING"}
EXIT_CODES = {
    "SUCCEEDED": 0,
    "FAILED": 100,
    "FAILED_SETUP": 100,
    "FAILED_DRIVER": 100,
    "CANCELLED": 103,
    None: 102,
}


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_bound(root, filename, owner):
    data = json.loads((root / filename).read_text())
    if any(data.get(key) != value for key, value in owner.items()):
        raise ValueError(f"Wrong CI identity in {filename}")
    return data


def worker(root, operation):
    import sky

    owner = json.loads((root / "owner.json").read_text())
    cluster = owner["cluster"]
    if operation == "launch":
        task = sky.Task.from_yaml("scripts/ci/trainer-rank-gpu.sky.yaml")
        task.set_resources_override({"infra": "k8s/cks-wb3"})
        request_id = sky.launch(task, cluster_name=cluster, retry_until_up=False)
        try:
            write_json(root / "request.json", {**owner, "request_id": request_id})
            job_id, handle = sky.get(request_id)
            if type(job_id) is not int or job_id < 1 or handle.cluster_name != cluster:
                raise ValueError(
                    "Sky launch returned a different cluster or invalid job ID"
                )
            write_json(root / "job.json", {**owner, "job_id": job_id})
        except BaseException:
            try:
                sky.get(sky.api_cancel(request_ids=[request_id]))
            except Exception as error:
                print(
                    f"Launch request cancellation also failed: {error}", file=sys.stderr
                )
            raise
        return
    if operation == "cancel_request":
        request = read_bound(root, "request.json", owner)["request_id"]
        sky.get(sky.api_cancel(request_ids=[request]))
        return
    job_id = read_bound(root, "job.json", owner)["job_id"]
    if operation == "status":
        statuses = sky.get(sky.job_status(cluster, job_ids=[job_id]))
        if set(statuses) != {job_id}:
            raise ValueError("Sky returned a different job's status")
        value = statuses[job_id]
        status = None if value is None else value.value
        if status not in NONTERMINAL and status not in EXIT_CODES:
            raise ValueError(f"Unknown Sky job status: {status}")
        write_json(root / "status.json", {**owner, "job_id": job_id, "status": status})
    elif operation == "cancel":
        sky.get(sky.cancel(cluster, job_ids=[job_id]))
    elif operation == "logs":
        # Diagnostic retrieval has a separate timeout and never follows the job.
        sys.exit(sky.tail_logs(cluster, job_id=job_id, follow=False))
    else:
        raise ValueError(operation)


def run_child(command, timeout, output):
    if timeout <= 0:
        raise TimeoutError("CI remote-result deadline reached")
    with output.open("ab") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # The unreaped direct child reserves this process-group identity.
            if process.returncode is None:
                try:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
                except Exception as cleanup_error:
                    print(
                        f"Child cleanup also failed: {cleanup_error}", file=sys.stderr
                    )
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def run_worker(root, operation, timeout):
    run_child(
        [sys.executable, str(Path(__file__).resolve()), "worker", str(root), operation],
        timeout,
        root / f"{operation}.log",
    )


def supervise(root, owner, timeout=35 * 60, poll_seconds=10):
    deadline = time.monotonic() + timeout
    remote_status = None
    terminal = False
    job = None
    errors = 0
    try:
        run_worker(root, "launch", deadline - time.monotonic())
        job = read_bound(root, "job.json", owner)
        while True:
            try:
                run_worker(root, "status", min(30, deadline - time.monotonic()))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                errors += 1
                print(f"Status query failed ({errors}/3): {error}", flush=True)
                if errors >= 3:
                    raise
            else:
                observed = read_bound(root, "status.json", job)
                remote_status = observed["status"]
                if time.monotonic() >= deadline:
                    raise TimeoutError("CI remote-result deadline reached")
                errors = 0
                print(f"Job {job['job_id']}: {remote_status}", flush=True)
                if remote_status in EXIT_CODES:
                    terminal = remote_status is not None
                    write_json(root / "result.json", observed)
                    return EXIT_CODES[remote_status]
                if remote_status not in NONTERMINAL:
                    raise ValueError(f"Unknown Sky job status: {remote_status}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("CI remote-result deadline reached")
            time.sleep(min(poll_seconds, remaining))
    except BaseException as error:
        try:
            write_json(
                root / "failure.json",
                {**owner, "remote_status": remote_status, "error": repr(error)},
            )
        except Exception as receipt_error:
            print(f"Failure receipt also failed: {receipt_error}", file=sys.stderr)
        raise
    finally:
        # Also recover the exact ID if a receipt read failed after launch.
        # No guessed/latest job is ever cancelled. The outer trap handles unknown ID.
        if job is None:
            try:
                job = read_bound(root, "job.json", owner)
            except Exception:
                pass
        operations = (["cancel"] if not terminal else []) + ["logs"] if job else []
        if job is None and (root / "request.json").exists():
            operations = ["cancel_request"]
        outcomes = {}
        for operation in operations:
            try:
                run_worker(root, operation, 30)
                outcomes[operation] = {"success": True}
            except Exception as error:
                outcomes[operation] = {"success": False, "error": repr(error)}
                print(f"::warning::{operation} failed: {error}", flush=True)
        try:
            write_json(root / "diagnostics.json", {**owner, "operations": outcomes})
        except Exception as error:
            print(f"::warning::Diagnostic receipt failed: {error}", flush=True)


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError("Expected numeric GitHub run ID and attempt")
    if os.environ["SKY_INFRA"] != "k8s/cks-wb3":
        raise ValueError("TrainerRank CI requires free cks-wb3")
    owner = {
        "run_id": run_id,
        "attempt": attempt,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cluster": f"trainer-rank-gpu-{run_id}-{attempt}",
        "repository": os.environ["GITHUB_REPOSITORY"],
        "event": os.environ["GITHUB_EVENT_NAME"],
    }
    if owner["head"] != os.environ["EXPECTED_HEAD_SHA"]:
        raise ValueError("Checkout differs from classified CI head")
    write_json(root / "owner.json", owner)

    def interrupted(signum, frame):
        raise InterruptedError(f"CI interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    return supervise(root, owner)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(Path(sys.argv[2]), sys.argv[3])
    else:
        sys.exit(main(Path(sys.argv[1])))
