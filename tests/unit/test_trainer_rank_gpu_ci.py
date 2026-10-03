"""No Sky service or GPU calls: exercise the real CI driver at its SDK boundary."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/ci/trainer-rank-gpu.py"
spec = importlib.util.spec_from_file_location("trainer_rank_gpu_ci", SCRIPT)
assert spec is not None and spec.loader is not None
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


@pytest.fixture
def owner(tmp_path):
    value = {
        "cluster": "trainer-rank-gpu-42-2",
        "head": "a" * 40,
        "attempt": "2",
        "infra": "k8s/cks-wb3",
        "label": "a" * 32,
        "admission_deadline": time.time() + 1800,
        "work_deadline": time.time() + 2100,
        "cleanup_deadline": time.time() + 2400,
    }
    ci.write_json(tmp_path / "owner.json", value)
    return value


@pytest.fixture
def sky(monkeypatch):
    task = Mock()
    api = NS(
        Task=NS(from_yaml=Mock(return_value=task)),
        launch=Mock(return_value="request-17"),
        get=Mock(),
        job_status=Mock(return_value="status-request"),
        cancel=Mock(return_value="cancel-request"),
        api_cancel=Mock(return_value="api-cancel-request"),
        api_status=Mock(return_value=[NS(request_id="request-17", status="CANCELLED")]),
        api_stop=Mock(),
        tail_logs=Mock(return_value=0),
    )
    monkeypatch.setitem(sys.modules, "sky", api)
    monkeypatch.setitem(
        sys.modules,
        "sky.provision.kubernetes",
        NS(utils=NS(get_namespace=lambda **_: "default")),
    )
    return api


@pytest.mark.parametrize("infra", ["k8s/cks-wb3", "k8s/ext-collab2"])
def test_submit_once_records_request_before_wait_and_actual_job(
    tmp_path, owner, sky, monkeypatch, infra
):
    owner["infra"] = infra
    ci.write_json(tmp_path / "owner.json", owner)
    monkeypatch.setenv("SKY_INFRA", "k8s/unapproved")  # Use recorded ownership.

    def launch(*args, **kwargs):
        receipt = ci.read_bound(tmp_path, "cleanup.json", owner)
        assert receipt["physical_absence"] == "UNKNOWN"
        assert receipt["confirmed"] is False
        return "request-17"

    sky.launch.side_effect = launch

    def get(request):
        assert request == "request-17"
        assert ci.read_bound(tmp_path, "request.json", owner)["request_id"] == request
        return 17, NS(
            cluster_name=owner["cluster"], cluster_name_on_cloud="physical-suffix"
        )

    sky.get.side_effect = get
    ci.worker(tmp_path, "launch")
    task = sky.Task.from_yaml.return_value
    assert task.set_resources_override.call_args.args[0] == {
        "infra": infra,
        "_cluster_config_overrides": {
            "kubernetes": {"custom_metadata": {"labels": {ci.LABEL: owner["label"]}}}
        },
    }
    sky.launch.assert_called_once_with(
        task, cluster_name=owner["cluster"], retry_until_up=True
    )
    assert ci.read_bound(tmp_path, "job.json", owner)["job_id"] == 17
    assert ci.read_bound(tmp_path, "physical.json", owner)["cloud_name"] == (
        "physical-suffix"
    )
    assert ci.read_bound(tmp_path, "cleanup.json", owner)["physical_absence"] == (
        "UNKNOWN"
    )
    sky.tail_logs.assert_not_called()


@pytest.mark.parametrize("job_id,cluster", [(True, None), (0, None), (17, "peer")])
def test_launch_rejects_wrong_identity(tmp_path, owner, sky, job_id, cluster):
    sky.get.return_value = job_id, NS(cluster_name=cluster or owner["cluster"])
    with pytest.raises(ValueError):
        ci.worker(tmp_path, "launch")
    assert not (tmp_path / "job.json").exists()


def test_launch_missing_handle_cancels_recorded_request(tmp_path, owner, sky):
    sky.get.return_value = 17, None
    with pytest.raises(ValueError):
        ci.worker(tmp_path, "launch")
    sky.api_cancel.assert_called_once_with(request_ids=["request-17"])
    assert not (tmp_path / "job.json").exists()


@pytest.mark.parametrize("status", [*ci.NONTERMINAL, *ci.EXIT_CODES])
def test_exact_sdk_status_mapping(tmp_path, owner, sky, status):
    ci.write_json(tmp_path / "job.json", {**owner, "job_id": 17})
    sky.get.return_value = {17: None if status is None else NS(value=status)}
    ci.worker(tmp_path, "status")
    sky.job_status.assert_called_once_with(owner["cluster"], job_ids=[17])
    assert ci.read_bound(tmp_path, "status.json", owner)["status"] == status


@pytest.mark.parametrize(
    "response", [{1: NS(value="SUCCEEDED")}, {}, {17: NS(value="NEW")}]
)
def test_status_rejects_wrong_job_or_unknown_state(tmp_path, owner, sky, response):
    ci.write_json(tmp_path / "job.json", {**owner, "job_id": 17})
    sky.get.return_value = response
    with pytest.raises(ValueError):
        ci.worker(tmp_path, "status")


def test_cancel_and_logs_use_exact_job_and_never_follow(tmp_path, owner, sky):
    ci.write_json(tmp_path / "job.json", {**owner, "job_id": 17})
    ci.worker(tmp_path, "cancel")
    sky.cancel.assert_called_once_with(owner["cluster"], job_ids=[17])
    with pytest.raises(SystemExit) as outcome:
        ci.worker(tmp_path, "logs")
    assert outcome.value.code == 0
    sky.tail_logs.assert_called_once_with(owner["cluster"], job_id=17, follow=False)


def scenario(monkeypatch, root, owner, statuses, *, failures=None):
    iterator = iter(statuses)
    calls = []
    failures = failures or {}

    def run(root, operation, timeout):
        calls.append(operation)
        if timeout <= 0:
            raise TimeoutError("expired")
        if operation in failures:
            raise failures[operation]
        if operation == "launch":
            ci.write_json(root / "job.json", {**owner, "job_id": 17})
        if operation == "status":
            status = next(iterator)
            if isinstance(status, BaseException):
                raise status
            ci.write_json(
                root / "status.json", {**owner, "job_id": 17, "status": status}
            )

    monkeypatch.setattr(ci, "run_worker", run)
    return calls


@pytest.mark.parametrize("status,code", list(ci.EXIT_CODES.items()))
def test_remote_terminal_result_controls_exit(
    tmp_path, owner, monkeypatch, status, code
):
    calls = scenario(monkeypatch, tmp_path, owner, ["SETTING_UP", "RUNNING", status])
    assert ci.supervise(tmp_path, owner, poll_seconds=0) == code
    assert calls.count("launch") == 1
    assert calls[-1] == "logs"
    assert ("cancel" in calls) is (status is None)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == status


def test_status_transport_loss_recovers_without_resubmitting(
    tmp_path, owner, monkeypatch
):
    calls = scenario(
        monkeypatch,
        tmp_path,
        owner,
        [subprocess.CalledProcessError(1, "status"), "RUNNING", "SUCCEEDED"],
    )
    assert ci.supervise(tmp_path, owner, poll_seconds=0) == 0
    assert calls.count("launch") == 1
    assert "cancel" not in calls


def test_log_failure_does_not_rewrite_known_remote_success(
    tmp_path, owner, monkeypatch
):
    scenario(
        monkeypatch, tmp_path, owner, ["SUCCEEDED"], failures={"logs": OSError("EOF")}
    )
    assert ci.supervise(tmp_path, owner, poll_seconds=0) == 0
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "SUCCEEDED"


@pytest.mark.parametrize(
    "error", [InterruptedError("signal"), TimeoutError("deadline")]
)
def test_interruption_attempts_exact_cancel_and_logs(
    tmp_path, owner, monkeypatch, error
):
    calls = scenario(monkeypatch, tmp_path, owner, [error])
    with pytest.raises(type(error)) as raised:
        ci.supervise(tmp_path, owner, poll_seconds=0)
    assert raised.value is error
    assert calls[-2:] == ["cancel", "logs"]


def test_repeated_query_failure_and_cleanup_errors_preserve_primary(
    tmp_path, owner, monkeypatch
):
    primary = subprocess.CalledProcessError(1, "status")
    calls = scenario(
        monkeypatch,
        tmp_path,
        owner,
        [primary] * 3,
        failures={"cancel": OSError("cancel failed"), "logs": OSError("EOF")},
    )
    with pytest.raises(subprocess.CalledProcessError) as raised:
        ci.supervise(tmp_path, owner, poll_seconds=0)
    assert raised.value is primary
    assert calls.count("status") == 3
    assert calls[-2:] == ["cancel", "logs"]


def test_receipt_failure_still_cancels_and_preserves_error(
    tmp_path, owner, monkeypatch
):
    calls = scenario(monkeypatch, tmp_path, owner, ["RUNNING"])
    read = ci.read_bound
    primary = OSError("receipt failed")

    def fail_read(root, filename, expected):
        if filename == "status.json":
            raise primary
        return read(root, filename, expected)

    monkeypatch.setattr(ci, "read_bound", fail_read)
    with pytest.raises(OSError) as raised:
        ci.supervise(tmp_path, owner, poll_seconds=0)
    assert raised.value is primary
    assert calls[-2:] == ["cancel", "logs"]


def test_late_success_does_not_bypass_deadline(tmp_path, owner, monkeypatch):
    calls = scenario(monkeypatch, tmp_path, owner, ["SUCCEEDED"])
    clock = iter([0, 0, 0, 0, 2])
    monkeypatch.setattr(ci.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError):
        ci.supervise(tmp_path, owner, timeout=1)
    assert calls[-2:] == ["cancel", "logs"]


def test_real_bounded_child_kills_waiting_fork_descendant(tmp_path):
    """The descendant ignores TERM; the timed-out SDK process group is killed."""
    receipt = tmp_path / "descendant"
    program = """
import os, pathlib, signal, sys, time
child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
"""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        ci.run_child(
            [sys.executable, "-c", program, str(receipt)], 0.5, tmp_path / "child.log"
        )
    assert time.monotonic() - started < 3
    pid = int(receipt.read_text())
    # An exited descendant may remain a zombie until init reaps it.
    stat = Path(f"/proc/{pid}/stat")
    for _ in range(100):
        if not stat.exists() or stat.read_text().split(") ", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.01)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("owned descendant survived command timeout")


def test_run_child_preserves_remote_failure_code(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as raised:
        ci.run_child(
            [sys.executable, "-c", "raise SystemExit(17)"], 2, tmp_path / "exit.log"
        )
    assert raised.value.returncode == 17


def test_launch_receipt_failure_cancels_request_and_preserves_error(
    tmp_path, owner, sky, monkeypatch
):
    primary = OSError("disk full")
    write = ci.write_json

    def fail_receipt(path, data):
        if path.name == "request.json":
            raise primary
        write(path, data)

    monkeypatch.setattr(ci, "write_json", fail_receipt)
    sky.api_cancel.side_effect = OSError("cancel failed")
    with pytest.raises(OSError) as raised:
        ci.worker(tmp_path, "launch")
    assert raised.value is primary
    sky.api_cancel.assert_called_once_with(request_ids=["request-17"])


def test_cancel_request_requires_own_receipt(tmp_path, owner, sky):
    ci.write_json(tmp_path / "request.json", {**owner, "request_id": "request-17"})
    ci.worker(tmp_path, "cancel_request")
    sky.api_cancel.assert_called_once_with(request_ids=["request-17"])
    sky.api_cancel.reset_mock()
    ci.write_json(
        tmp_path / "request.json",
        {**owner, "cluster": "peer", "request_id": "peer-request"},
    )
    with pytest.raises(ValueError):
        ci.worker(tmp_path, "cancel_request")
    sky.api_cancel.assert_not_called()


@pytest.mark.parametrize(
    "attempted,known_request", [(False, False), (True, False), (True, True)]
)
def test_launch_timeout_cancels_only_recorded_request(
    tmp_path, owner, monkeypatch, attempted, known_request
):
    if attempted:
        ci.write_json(tmp_path / "launch-attempt.json", owner)
    if known_request:
        ci.write_json(tmp_path / "request.json", {**owner, "request_id": "request-17"})
    error = TimeoutError("launch wait")
    calls = scenario(monkeypatch, tmp_path, owner, [], failures={"launch": error})
    with pytest.raises(TimeoutError) as raised:
        ci.supervise(tmp_path, owner)
    assert raised.value is error
    assert calls == (["launch", "cancel_request"] if known_request else ["launch"])
    # A lost launch reply does not prove the remote test never started.
    assert ci.read_bound(tmp_path, "result.json", owner)["status"] == (
        "UNCONFIRMED" if attempted else "NOT_RUN"
    )


def test_capacity_wait_uses_original_admission_expiry(tmp_path, owner, monkeypatch):
    owner["admission_deadline"] = 107
    monkeypatch.setattr(ci.time, "time", lambda: 100)
    calls = []

    def run(root, operation, timeout):
        calls.append((operation, timeout))
        raise TimeoutError("capacity wait expired")

    monkeypatch.setattr(ci, "run_worker", run)
    with pytest.raises(TimeoutError):
        ci.supervise(tmp_path, owner, timeout=2100)
    assert calls == [("launch", 7)]


@pytest.mark.parametrize(
    "key,value",
    [
        ("EXPECTED_HEAD_SHA", "invalid"),
        ("GITHUB_RUN_ID", "invalid"),
        ("SKY_INFRA", "invalid"),
        ("SKY_INFRA", "k8s/unapproved"),
        ("SKY_INFRA", "aws/us-east-1"),
    ],
)
def test_main_rejects_invalid_scope_before_launch(tmp_path, monkeypatch, key, value):
    environment = {
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "OpenPipe/ART",
        "GITHUB_EVENT_NAME": "pull_request",
        "EXPECTED_HEAD_SHA": "a" * 40,
        "SKY_INFRA": "k8s/cks-wb3",
    }
    environment[key] = value
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ci.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    run = Mock()
    monkeypatch.setattr(ci, "supervise", run)
    with pytest.raises(ValueError):
        ci.main(tmp_path / "evidence")
    run.assert_not_called()


@pytest.mark.parametrize("infra", ["k8s/cks-wb3", "k8s/ext-collab2"])
def test_real_worker_round_trip_with_fake_sdk(tmp_path, infra):
    """Exercise the actual direct Python parent/worker JSON transport, without Sky."""
    (tmp_path / "sky.py").write_text("""
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace as NS
def deny_network(*args, **kwargs): raise AssertionError("Fake SDK worker must stay offline")
socket.socket.connect = socket.getaddrinfo = deny_network
sys.modules["sky.provision.kubernetes"] = NS(utils=NS(get_namespace=lambda **_: "default"))
sys.modules["kubernetes"] = NS(
    client=NS(CoreV1Api=lambda _: NS(
        list_namespaced_pod=lambda *a, **kw: NS(items=[]),
        list_namespaced_service=lambda *a, **kw: NS(items=[]),
    )),
    config=NS(new_client_from_config=lambda **kw: NS(close=lambda: None)),
)
class Task:
    @classmethod
    def from_yaml(cls, path): return cls()
    def set_resources_override(self, options): assert options["infra"] == os.environ["SKY_INFRA"]
def launch(task, *, cluster_name, retry_until_up):
    assert retry_until_up is True
    Path(os.environ["FAKE_CLUSTER"]).write_text(cluster_name)
    return "launch-request"
def get(value):
    if value == "launch-request":
        return 17, NS(cluster_name=Path(os.environ["FAKE_CLUSTER"]).read_text())
    return value
def job_status(cluster, *, job_ids):
    assert job_ids == [17]
    return {17:NS(value="SUCCEEDED")}
def tail_logs(cluster, *, job_id, follow):
    assert job_id == 17 and follow is False
    print("fake complete log")
    return 0
""")
    repo = SCRIPT.parents[2]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "FAKE_CLUSTER": str(tmp_path / "cluster"),
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "OpenPipe/ART",
        "GITHUB_EVENT_NAME": "pull_request",
        "SKY_INFRA": infra,
        "EXPECTED_HEAD_SHA": head,
        "ADMISSION_DEADLINE": str(int(time.time()) + 1800),
    }
    root = tmp_path / "evidence"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((root / "result.json").read_text())["job_id"] == 17
    assert json.loads((root / "result.json").read_text())["infra"] == infra
    assert json.loads((root / "resources.json").read_text())["remaining"] == []
    assert "fake complete log" in (root / "logs.log").read_text()


@pytest.mark.parametrize("code", [0, 17])
def test_early_worker_exit_stops_descendant_before_reaping_leader(tmp_path, code):
    import ctypes

    libc = ctypes.CDLL(None)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    receipt = tmp_path / "child.json"
    program = """
import json,os,pathlib,signal,sys,time
child=os.fork()
if child==0:
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))
    time.sleep(60)
    os._exit(0)
while not pathlib.Path(sys.argv[1]).exists():time.sleep(.005)
os._exit(int(sys.argv[2]))
"""
    info = None
    try:
        command = [sys.executable, "-c", program, str(receipt), str(code)]
        if code:
            with pytest.raises(subprocess.CalledProcessError) as raised:
                ci.run_child(command, 1, tmp_path / "out.log")
            assert raised.value.returncode == code
        else:
            ci.run_child(command, 1, tmp_path / "out.log")
        info = json.loads(receipt.read_text())
        stat = Path(f"/proc/{info['pid']}/stat")
        assert (
            not stat.exists() or stat.read_text().rsplit(") ", 1)[1].split()[0] == "Z"
        )
        assert not Path(f"/proc/{info['pgid']}").exists()
    finally:
        if info is None and receipt.exists():
            info = json.loads(receipt.read_text())
        if info:
            try:
                os.kill(info["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(info["pid"], 0)
            except ChildProcessError:
                pass
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_lost_waitable_child_identity_never_signals_group(monkeypatch):
    monkeypatch.setattr(ci.os, "waitid", Mock(side_effect=ChildProcessError))
    kill = Mock()
    monkeypatch.setattr(ci.os, "killpg", kill)
    with pytest.raises(ChildProcessError):
        ci.finish_child(NS(pid=42))
    kill.assert_not_called()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("boundary", ["acquiring", "acquired", "cleanup", "repeated"])
def test_interrupt_boundaries_retire_worker_before_propagating(
    tmp_path, monkeypatch, signum, boundary
):
    """Real signals at acquisition/retirement boundaries, never a Sky call."""
    tree = ast.parse(SCRIPT.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    acquired_line = next(
        n.lineno
        for n in ast.walk(functions["run_child"])
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "original" for t in n.targets)
        and isinstance(n.value, ast.Constant)
        and n.value.value is None
    )
    cleanup_line = next(
        n.lineno
        for n in ast.walk(functions["finish_child"])
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "killpg"
    )
    children, delivered = [], []
    popen = subprocess.Popen

    def acquire(*args, **kwargs):
        process = popen(*args, **kwargs)
        children.append(process)
        if boundary == "acquiring":
            os.kill(os.getpid(), signum)
            delivered.append("acquiring")
        return process

    def trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == str(SCRIPT):
            if frame.f_lineno == acquired_line and boundary in {"acquired", "repeated"}:
                if "acquired" not in delivered:
                    os.kill(os.getpid(), signum)
                    delivered.append("acquired")
            if frame.f_lineno == cleanup_line and boundary in {"cleanup", "repeated"}:
                # Both signals must remain non-raising throughout group cleanup.
                if "cleanup" not in delivered:
                    os.kill(os.getpid(), signal.SIGTERM)
                    os.kill(os.getpid(), signal.SIGINT)
                    delivered.append("cleanup")
        return trace

    def interrupted(signum, frame):
        raise InterruptedError(f"CI interrupted by signal {signum}")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(ci.subprocess, "Popen", acquire)
    try:
        for sig in handlers:
            signal.signal(sig, interrupted)
        sys.settrace(trace)
        expected = (
            subprocess.TimeoutExpired if boundary == "cleanup" else InterruptedError
        )
        with pytest.raises(expected) as raised:
            ci.run_child(
                [sys.executable, "-c", "import time; time.sleep(20)"],
                0.1,
                tmp_path / "worker.log",
            )
        sys.settrace(None)
        assert raised.value.__cause__ is None
        assert len(children) == 1 and delivered
        assert not Path(f"/proc/{children[0].pid}").exists()
        assert all(signal.getsignal(sig) is interrupted for sig in handlers)
    finally:
        sys.settrace(None)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


def admission_environment(monkeypatch, *, deadline):
    for key, value in {
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "OpenPipe/ART",
        "GITHUB_EVENT_NAME": "pull_request",
        "EXPECTED_HEAD_SHA": "a" * 40,
        "SKY_INFRA": "k8s/cks-wb3",
        "ADMISSION_DEADLINE": str(deadline),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ci.subprocess, "check_output", lambda *a, **kw: "a" * 40)


def test_expired_queue_admission_cannot_later_launch(tmp_path, monkeypatch):
    admission_environment(monkeypatch, deadline=100)
    monkeypatch.setattr(ci.time, "time", lambda: 101)
    launch = Mock()
    monkeypatch.setattr(ci, "supervise", launch)
    root = tmp_path / "expired"
    assert ci.admit(root) == 104
    assert ci.main(root) == 104
    assert ci.cleanup(root) == 0
    launch.assert_not_called()
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "NOT_RUN" and result["reason"] == "admission_expired"


def test_setup_consumes_original_admitted_work_window(tmp_path, monkeypatch):
    admission_environment(monkeypatch, deadline=1300)
    now = [100]
    monkeypatch.setattr(ci.time, "time", lambda: now[0])
    root = tmp_path / "admitted"
    assert ci.admit(root) == 0
    now[0] = 400
    run = Mock(return_value=0)
    monkeypatch.setattr(ci, "supervise", run)
    assert ci.main(root) == 0
    assert run.call_args.kwargs["timeout"] == 1800
    assert json.loads((root / "owner.json").read_text())["cleanup_deadline"] == 2500


def test_admission_expiring_during_setup_does_not_launch(tmp_path, monkeypatch):
    admission_environment(monkeypatch, deadline=300)
    now = [100]
    monkeypatch.setattr(ci.time, "time", lambda: now[0])
    root = tmp_path / "admitted"
    assert ci.admit(root) == 0
    now[0] = 301
    run = Mock()
    monkeypatch.setattr(ci, "supervise", run)
    assert ci.main(root) == 104
    run.assert_not_called()
    assert json.loads((root / "result.json").read_text())["status"] == "NOT_RUN"


@pytest.mark.parametrize("deadline", ["admission_deadline", "work_deadline"])
def test_worker_does_not_submit_after_cutoff(
    tmp_path, owner, sky, monkeypatch, deadline
):
    monkeypatch.setattr(ci.time, "time", lambda: owner[deadline])
    with pytest.raises(TimeoutError, match="launch deadline"):
        ci.worker(tmp_path, "launch")
    sky.launch.assert_not_called()
    assert not (tmp_path / "launch-attempt.json").exists()


def test_request_cancellation_waits_for_exact_terminal_state(
    tmp_path, owner, sky, monkeypatch
):
    ci.write_json(tmp_path / "request.json", {**owner, "request_id": "request-17"})
    sky.api_status.side_effect = [
        [NS(request_id="request-17", status="PENDING")],
        [NS(request_id="request-17", status="RUNNING")],
        [NS(request_id="request-17", status="CANCELLED")],
    ]
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    ci.worker(tmp_path, "cancel_request")
    assert sky.api_status.call_count == 3
    assert (
        ci.read_bound(tmp_path, "request-terminal.json", owner)["status"] == "CANCELLED"
    )
    sky.launch.assert_not_called()


@pytest.mark.parametrize("records", [[], [NS(request_id="peer", status="CANCELLED")]])
def test_request_cancellation_does_not_claim_unknown_terminal(
    tmp_path, owner, sky, records
):
    ci.write_json(tmp_path / "request.json", {**owner, "request_id": "request-17"})
    sky.api_status.return_value = records
    with pytest.raises(ValueError, match="exact launch request"):
        ci.worker(tmp_path, "cancel_request")
    assert not (tmp_path / "request-terminal.json").exists()


def test_cleanup_is_finite_and_reports_unknown_cancel(tmp_path, owner, monkeypatch):
    ci.write_json(tmp_path / "launch-attempt.json", owner)
    now = [owner["cleanup_deadline"] - 5]
    monkeypatch.setattr(ci.time, "time", lambda: now[0])
    calls = []

    def run(root, operation, timeout):
        calls.append((operation, timeout))
        now[0] += 1
        if operation == "cancel_request":
            raise TimeoutError("request state unknown")

    monkeypatch.setattr(ci, "run_worker", run)
    assert ci.cleanup(tmp_path) == 105
    assert calls == [
        ("cancel_request", 5),
        ("resources", 4),
        ("down", 3),
        ("stop_api", 2),
        ("remove_resources", 1),
    ]
    receipt = ci.read_bound(tmp_path, "cleanup.json", owner)
    assert receipt["confirmed"] is False
    assert receipt["operations"]["remove_resources"]["success"] is True


def test_api_cleanup_is_limited_to_its_ephemeral_runner(
    tmp_path, owner, sky, monkeypatch
):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "self-hosted")
    with pytest.raises(ValueError, match="isolated GitHub-hosted"):
        ci.worker(tmp_path, "stop_api")
    sky.api_stop.assert_not_called()
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    sky.api_status.return_value = []
    ci.worker(tmp_path, "stop_api")
    sky.api_stop.assert_called_once_with()


def test_api_cleanup_detects_surviving_server_with_only_terminal_requests(
    tmp_path, owner, sky, monkeypatch
):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    sky.api_status.side_effect = lambda **options: (
        [NS(request_id="request-17", status="CANCELLED")]
        if options.get("all_status")
        else []
    )
    with pytest.raises(RuntimeError, match="still reports requests"):
        ci.worker(tmp_path, "stop_api")


@pytest.mark.parametrize("foreign", [False, True])
def test_cleanup_resources_records_uids_and_confines_deletion(
    tmp_path, owner, monkeypatch, foreign
):
    ci.write_json(tmp_path / "allocation.json", {**owner, "namespace": "ci"})
    pod = NS(
        metadata=NS(
            name="pod",
            uid="original-uid",
            namespace="ci",
            labels={ci.LABEL: "peer" if foreign else owner["label"]},
        )
    )
    api = NS(
        list_namespaced_pod=Mock(side_effect=[NS(items=[pod]), NS(items=[])]),
        list_namespaced_service=Mock(return_value=NS(items=[])),
        delete_namespaced_pod=Mock(),
        read_namespaced_pod=Mock(
            return_value=NS(metadata=NS(name="pod", namespace="ci", uid="replacement"))
        ),
    )
    connection = NS(close=Mock())

    class ApiException(Exception):
        pass

    client = NS(
        CoreV1Api=lambda c: api,
        V1DeleteOptions=NS,
        V1Preconditions=NS,
        ApiException=ApiException,
    )
    config = NS(new_client_from_config=Mock(return_value=connection))
    monkeypatch.setitem(sys.modules, "kubernetes", NS(client=client, config=config))
    monkeypatch.setattr(ci.time, "sleep", lambda _: None)
    if foreign:
        with pytest.raises(ValueError, match="unowned"):
            ci.resources(tmp_path, owner, delete=True)
        api.delete_namespaced_pod.assert_not_called()
    else:

        def delete(name, namespace, *, body, _request_timeout):
            assert (name, namespace, body.preconditions.uid) == (
                "pod",
                "ci",
                "original-uid",
            )
            recorded = ci.read_bound(tmp_path, "resources-cleanup.json", owner)
            assert recorded["remaining"][0]["uid"] == "original-uid"

        api.delete_namespaced_pod.side_effect = delete
        ci.resources(tmp_path, owner, delete=True)
        assert (
            ci.read_bound(tmp_path, "resources-cleanup.json", owner)["remaining"] == []
        )
        api.list_namespaced_pod.assert_called_with(
            "ci",
            label_selector=f"{ci.LABEL}={owner['label']}",
            _request_timeout=(3, 10),
        )
    connection.close.assert_called_once()


def test_handle_is_retained_before_job_receipt_failure(
    tmp_path, owner, sky, monkeypatch
):
    sky.get.return_value = (
        17,
        NS(cluster_name=owner["cluster"], cluster_name_on_cloud="actual-cloud-suffix"),
    )
    write = ci.write_json

    def fail_job(path, data):
        if path.name == "job.json":
            raise OSError("interrupted after handle")
        write(path, data)

    monkeypatch.setattr(ci, "write_json", fail_job)
    with pytest.raises(OSError, match="after handle"):
        ci.worker(tmp_path, "launch")
    physical = ci.read_bound(tmp_path, "physical.json", owner)
    assert physical["cloud_name"] == "actual-cloud-suffix"
    assert physical["namespace"] == "default"
    assert (
        ci.read_bound(tmp_path, "cleanup.json", owner)["physical_absence"] == "UNKNOWN"
    )


def test_initial_identity_capture_precedes_job_polling(tmp_path, owner, monkeypatch):
    calls = scenario(monkeypatch, tmp_path, owner, ["SUCCEEDED"])
    assert ci.supervise(tmp_path, owner, poll_seconds=0) == 0
    assert calls[:3] == ["launch", "resources", "status"]


def test_initial_identity_failure_preserves_job_result(tmp_path, owner, monkeypatch):
    scenario(
        monkeypatch,
        tmp_path,
        owner,
        ["SUCCEEDED"],
        failures={"resources": subprocess.TimeoutExpired("resources", 20)},
    )
    assert ci.supervise(tmp_path, owner, poll_seconds=0) == 0


def test_interrupted_provisioning_captures_ids_before_cancelling(
    tmp_path, owner, monkeypatch
):
    ci.write_json(tmp_path / "allocation.json", {**owner, "namespace": "ci"})
    ci.write_json(tmp_path / "launch-attempt.json", owner)
    ci.write_json(tmp_path / "request.json", {**owner, "request_id": "request-17"})
    calls = scenario(
        monkeypatch, tmp_path, owner, [], failures={"launch": TimeoutError("launch")}
    )
    with pytest.raises(TimeoutError):
        ci.supervise(tmp_path, owner)
    assert calls == ["launch", "resources", "cancel_request"]


@pytest.fixture
def kube_receipts(tmp_path, owner, monkeypatch):
    ci.write_json(tmp_path / "allocation.json", {**owner, "namespace": "ci"})
    pod = NS(
        metadata=NS(
            name="physical-head",
            uid="original-uid",
            namespace="ci",
            labels={
                ci.LABEL: owner["label"],
                "skypilot-cluster-name": "physical-suffix",
            },
        )
    )

    class ApiException(Exception):
        def __init__(self, status):
            self.status = status

    api = NS(
        list_namespaced_pod=Mock(return_value=NS(items=[pod])),
        list_namespaced_service=Mock(return_value=NS(items=[])),
        read_namespaced_pod=Mock(side_effect=ApiException(404)),
        delete_namespaced_pod=Mock(),
    )
    connection = NS(close=Mock())
    monkeypatch.setitem(
        sys.modules,
        "kubernetes",
        NS(
            client=NS(
                CoreV1Api=lambda _: api,
                ApiException=ApiException,
                V1DeleteOptions=NS,
                V1Preconditions=NS,
            ),
            config=NS(new_client_from_config=lambda **_: connection),
        ),
    )
    return api, pod, ApiException


def test_partial_resource_query_retains_exact_pod(tmp_path, owner, kube_receipts):
    api, pod, _ = kube_receipts
    api.list_namespaced_service.side_effect = OSError("service query unavailable")
    with pytest.raises(OSError):
        ci.resources(tmp_path, owner, delete=False)
    receipt = ci.read_bound(tmp_path, "resources.json", owner)
    assert receipt["observed"][0]["uid"] == pod.metadata.uid
    assert receipt["observed"][0]["cloud_name"] == "physical-suffix"
    assert receipt["remaining"] is None
    assert receipt["exact_uid_absence"] == "UNKNOWN"


@pytest.mark.parametrize("state", ["absent", "replaced", "relabelled", "forbidden"])
def test_exact_uid_absence_requires_named_query(tmp_path, owner, kube_receipts, state):
    api, pod, error = kube_receipts
    ci.resources(tmp_path, owner, delete=False)
    api.list_namespaced_pod.return_value = NS(items=[])
    # A later empty pre-cleanup query must not discard the live-job receipt.
    ci.resources(tmp_path, owner, delete=False)
    if state in {"replaced", "relabelled"}:
        api.read_namespaced_pod.side_effect = None
        api.read_namespaced_pod.return_value = NS(
            metadata=NS(
                name=pod.metadata.name,
                namespace="ci",
                uid="new-uid" if state == "replaced" else pod.metadata.uid,
            )
        )
    elif state == "forbidden":
        api.read_namespaced_pod.side_effect = error(403)
    if state in {"relabelled", "forbidden"}:
        with pytest.raises((ValueError, error)):
            ci.resources(tmp_path, owner, delete=True)
    else:
        ci.resources(tmp_path, owner, delete=True)
    receipt = ci.read_bound(tmp_path, "resources-cleanup.json", owner)
    assert receipt["exact_uid_absence"] == (
        "ABSENT" if state in {"absent", "replaced"} else "UNKNOWN"
    )
    api.read_namespaced_pod.assert_called_once_with(
        "physical-head", "ci", _request_timeout=(3, 10)
    )
    api.delete_namespaced_pod.assert_not_called()


def test_empty_resource_history_is_unknown(tmp_path, owner, kube_receipts):
    api, _, _ = kube_receipts
    api.list_namespaced_pod.return_value = NS(items=[])
    ci.resources(tmp_path, owner, delete=True)
    assert (
        ci.read_bound(tmp_path, "resources-cleanup.json", owner)["exact_uid_absence"]
        == "UNKNOWN"
    )


@pytest.mark.parametrize("stop_api", [True, False])
@pytest.mark.parametrize("receipt", [True, False])
def test_cleanup_absence_requires_stopped_creator_and_uid_receipt(
    tmp_path, owner, monkeypatch, stop_api, receipt
):
    ci.write_json(tmp_path / "launch-attempt.json", owner)
    if receipt:
        ci.write_json(
            tmp_path / "resources-cleanup.json",
            {
                **owner,
                "observed": [{"kind": "pod", "name": "pod", "uid": "original"}],
                "remaining": [],
                "exact_uid_absence": "ABSENT",
            },
        )

    def run(root, operation, timeout):
        if operation == "stop_api" and not stop_api:
            raise TimeoutError("creator still running")

    monkeypatch.setattr(ci, "run_worker", run)
    assert ci.cleanup(tmp_path) == (0 if stop_api else 105)
    assert ci.read_bound(tmp_path, "cleanup.json", owner)["physical_absence"] == (
        "ABSENT" if stop_api and receipt else "UNKNOWN"
    )
