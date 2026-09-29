"""Run the CI job without making an attached log stream its result gate.

SkyPilot returns an actual job ID from launch. Each SDK operation runs in a
bounded child; only that job's SUCCEEDED status passes. An always-run workflow
step cancels pending admission and verifies cleanup of owned resources.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

NONTERMINAL = {"INIT", "PENDING", "SETTING_UP", "RUNNING"}
INFRAS = {"k8s/cks-wb3", "k8s/ext-collab2"}
LABEL = "art-ci-execution"
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


def launch_task(root, owner):
    import sky
    from sky.provision.kubernetes import utils as kube_utils

    context = owner["infra"].split("/", 1)[1]
    write_json(
        root / "allocation.json",
        {**owner, "namespace": kube_utils.get_namespace(context=context)},
    )
    task = sky.Task.from_yaml("scripts/ci/trainer-rank-gpu.sky.yaml")
    task.set_resources_override(
        {
            "infra": owner["infra"],
            "_cluster_config_overrides": {
                "kubernetes": {"custom_metadata": {"labels": {LABEL: owner["label"]}}}
            },
        }
    )
    return task


def resources(root, owner, *, delete):
    """Observe/delete only this attempt's nonce-labelled Pods and Services."""
    from kubernetes import client, config

    allocation = read_bound(root, "allocation.json", owner)
    namespace = allocation["namespace"]
    connection = config.new_client_from_config(context=owner["infra"].split("/", 1)[1])
    api = client.CoreV1Api(connection)
    retained = []
    try:
        while True:
            found = []
            for kind in ("pod", "service"):
                items = getattr(api, f"list_namespaced_{kind}")(
                    namespace,
                    label_selector=f"{LABEL}={owner['label']}",
                    _request_timeout=(3, 10),
                ).items
                for item in items:
                    metadata = item.metadata
                    if (
                        metadata.labels.get(LABEL) != owner["label"]
                        or metadata.namespace != namespace
                        or not metadata.uid
                    ):
                        raise ValueError("Kubernetes returned an unowned resource")
                    identity = {
                        "kind": kind,
                        "name": metadata.name,
                        "uid": metadata.uid,
                    }
                    found.append(identity)
                    if identity not in retained:
                        retained.append(identity)
            write_json(
                root / ("resources-cleanup.json" if delete else "resources.json"),
                {**allocation, "observed": retained, "remaining": found},
            )
            if not delete or not found:
                return
            for item in found:
                try:
                    getattr(api, f"delete_namespaced_{item['kind']}")(
                        item["name"],
                        namespace,
                        body=client.V1DeleteOptions(
                            preconditions=client.V1Preconditions(uid=item["uid"])
                        ),
                        _request_timeout=(3, 10),
                    )
                except client.ApiException as error:
                    if error.status != 404:
                        raise
            time.sleep(1)
    finally:
        connection.close()


def worker(root, operation):
    import sky

    owner = json.loads((root / "owner.json").read_text())
    cluster = owner["cluster"]
    if operation == "launch":
        task = launch_task(root, owner)
        if time.time() >= min(owner["admission_deadline"], owner["work_deadline"]):
            raise TimeoutError("CI launch deadline reached")
        write_json(root / "launch-attempt.json", owner)
        request_id = sky.launch(task, cluster_name=cluster, retry_until_up=True)
        try:
            write_json(root / "request.json", {**owner, "request_id": request_id})
            job_id, handle = sky.get(request_id)
            if (
                type(job_id) is not int
                or job_id < 1
                or getattr(handle, "cluster_name", None) != cluster
            ):
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
        while True:
            records = sky.api_status(request_ids=[request])
            if len(records) != 1 or records[0].request_id != request:
                raise ValueError("Sky did not return the exact launch request")
            status = records[0].status
            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                write_json(
                    root / "request-terminal.json",
                    {**owner, "request_id": request, "status": status},
                )
                break
            time.sleep(1)
        return
    if operation == "down":
        clusters = sky.get(sky.status([cluster]))
        if any(item["name"] != cluster for item in clusters):
            raise ValueError("Sky returned a different cluster")
        if clusters:
            sky.get(sky.down(cluster))
        return
    if operation == "stop_api":
        # This workflow owns the entire ephemeral runner and its local API.
        # Stop its workers before the final physical census, including when a
        # cancelled launch has not yet unwound. Never stop a developer's API.
        if (
            os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
        ):
            raise ValueError("API cleanup requires an isolated GitHub-hosted runner")
        sky.api_stop()  # The maintained SDK rejects remote API endpoints.
        if sky.api_status():
            raise RuntimeError("The CI API still reports requests after stopping")
        return
    if operation in {"resources", "remove_resources"}:
        resources(root, owner, delete=operation == "remove_resources")
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


def finish_child(process):
    """Keep the unreaped Linux child as the group identity until cleanup finishes."""
    deadline = time.monotonic() + 5
    while True:
        # If another reaper consumed the child, fail without signaling a number
        # whose identity is no longer reserved by this parent.
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        alive = False
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = path.read_text().rsplit(") ", 1)[1].split()
            except FileNotFoundError:
                continue
            if int(fields[2]) == process.pid and fields[0] != "Z":
                alive = True
                break
        if not alive:
            return process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            raise TimeoutError("SDK worker process group did not stop")
        time.sleep(0.01)


def run_child(command, timeout, output):
    if timeout <= 0:
        raise TimeoutError("CI remote-result deadline reached")
    with output.open("ab") as stream:
        interruption = None

        def defer_interrupt(signum, frame):
            nonlocal interruption
            if interruption is None:
                interruption = InterruptedError(f"CI interrupted by signal {signum}")

        # Recording instead of raising also covers interruption *inside* Popen,
        # before its process handle is returned. Unlike masking, this does not
        # leave SIGINT/SIGTERM blocked in the exec'd SDK worker.
        handlers = {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            for sig in handlers:
                signal.signal(sig, defer_interrupt)
            if interruption is not None:
                raise interruption
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            original = None
            try:
                deadline = time.monotonic() + timeout
                # WNOWAIT observes exit without freeing the PID/PGID for reuse.
                while (
                    os.waitid(
                        os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                    )
                    is None
                ):
                    if interruption is not None:
                        raise interruption
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    time.sleep(min(0.05, remaining))
            except BaseException as error:
                original = error
            try:
                code = finish_child(process)
            except BaseException as cleanup_error:
                if original is not None:
                    raise original from cleanup_error
                raise
            if original is not None:
                raise original
            if interruption is not None:
                raise interruption
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            # Repeated interrupts cannot abort mandatory group retirement.
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


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
        run_worker(
            root,
            "launch",
            min(deadline - time.monotonic(), owner["admission_deadline"] - time.time()),
        )
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
            if job is None:
                write_json(
                    root / "result.json",
                    {
                        **owner,
                        "status": "UNCONFIRMED",
                        "reason": "launch_result_unavailable",
                    },
                )
        except Exception as receipt_error:
            print(f"Failure receipt also failed: {receipt_error}", file=sys.stderr)
        raise
    finally:
        # Also recover the exact ID if a receipt read failed after launch.
        # No guessed/latest job is cancelled. The workflow reconciles unknown IDs.
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


def admit(root):
    root.mkdir(parents=True, exist_ok=False)
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError("Expected numeric GitHub run ID and attempt")
    infra = os.environ["SKY_INFRA"]
    if infra not in INFRAS:
        raise ValueError("TrainerRank CI requires free cks-wb3 or ext-collab2")
    owner = {
        "run_id": run_id,
        "attempt": attempt,
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cluster": f"trainer-rank-gpu-{run_id}-{attempt}",
        "repository": os.environ["GITHUB_REPOSITORY"],
        "event": os.environ["GITHUB_EVENT_NAME"],
        "infra": infra,
    }
    if owner["head"] != os.environ["EXPECTED_HEAD_SHA"]:
        raise ValueError("Checkout differs from classified CI head")
    now = time.time()
    admission_deadline = int(os.environ["ADMISSION_DEADLINE"])
    owner.update(
        label=uuid.uuid4().hex,
        admitted_at=now,
        admission_deadline=admission_deadline,
        work_deadline=now + 35 * 60,
        cleanup_deadline=now + 40 * 60,
    )
    write_json(root / "owner.json", owner)
    if now >= admission_deadline:
        write_json(
            root / "result.json",
            {**owner, "status": "NOT_RUN", "reason": "admission_expired"},
        )
        print("::error::GPU validation did not run: its queue admission expired.")
        return 104
    return 0


def cleanup(root):
    if not (root / "launch-attempt.json").exists():
        return 0
    owner = json.loads((root / "owner.json").read_text())

    read_bound(root, "launch-attempt.json", owner)
    outcomes = {}
    # A failed/unknown cancellation must not prevent best-effort physical cleanup.
    for operation, limit in [
        ("cancel_request", 30),
        ("resources", 20),
        ("down", 90),
        ("stop_api", 30),
        ("remove_resources", 90),
    ]:
        try:
            run_worker(
                root, operation, min(limit, owner["cleanup_deadline"] - time.time())
            )
            outcomes[operation] = {"success": True}
        except BaseException as error:
            outcomes[operation] = {"success": False, "error": repr(error)}
    confirmed = all(outcome["success"] for outcome in outcomes.values())
    write_json(
        root / "cleanup.json", {**owner, "confirmed": confirmed, "operations": outcomes}
    )
    return 0 if confirmed else 105


def main(root):
    if not (root / "owner.json").exists():
        code = admit(root)
        if code:
            return code
    owner = json.loads((root / "owner.json").read_text())
    if time.time() >= owner["admission_deadline"]:
        write_json(
            root / "result.json",
            {**owner, "status": "NOT_RUN", "reason": "admission_expired"},
        )
        return 104

    def interrupted(signum, frame):
        raise InterruptedError(f"CI interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    return supervise(root, owner, timeout=max(0, owner["work_deadline"] - time.time()))


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(Path(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "admit":
        sys.exit(admit(Path(sys.argv[2])))
    elif sys.argv[1] == "cleanup":
        sys.exit(cleanup(Path(sys.argv[2])))
    else:
        sys.exit(main(Path(sys.argv[1])))
