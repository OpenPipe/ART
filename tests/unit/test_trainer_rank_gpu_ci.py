"""No Sky service or GPU calls: exercise the real CI driver at its SDK boundary."""

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
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


@pytest.fixture
def owner(tmp_path):
    value = {"cluster": "trainer-rank-gpu-42-2", "head": "a" * 40, "attempt": "2"}
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
        tail_logs=Mock(return_value=0),
    )
    monkeypatch.setitem(sys.modules, "sky", api)
    return api


def test_submit_once_records_request_before_wait_and_actual_job(tmp_path, owner, sky):
    def get(request):
        assert request == "request-17"
        assert ci.read_bound(tmp_path, "request.json", owner)["request_id"] == request
        return 17, NS(cluster_name=owner["cluster"])

    sky.get.side_effect = get
    ci.worker(tmp_path, "launch")
    task = sky.Task.from_yaml.return_value
    task.set_resources_override.assert_called_once_with({"infra": "k8s/cks-wb3"})
    sky.launch.assert_called_once_with(
        task, cluster_name=owner["cluster"], retry_until_up=False
    )
    assert ci.read_bound(tmp_path, "job.json", owner)["job_id"] == 17
    sky.tail_logs.assert_not_called()


@pytest.mark.parametrize("job_id,cluster", [(True, None), (0, None), (17, "peer")])
def test_launch_rejects_wrong_identity(tmp_path, owner, sky, job_id, cluster):
    sky.get.return_value = job_id, NS(cluster_name=cluster or owner["cluster"])
    with pytest.raises(ValueError):
        ci.worker(tmp_path, "launch")
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
    clock = iter([0, 0, 0, 2])
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
    monkeypatch.setattr(ci, "write_json", Mock(side_effect=primary))
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


@pytest.mark.parametrize("known_request", [False, True])
def test_launch_timeout_cancels_only_recorded_request(
    tmp_path, owner, monkeypatch, known_request
):
    if known_request:
        ci.write_json(tmp_path / "request.json", {**owner, "request_id": "request-17"})
    error = TimeoutError("launch wait")
    calls = scenario(monkeypatch, tmp_path, owner, [], failures={"launch": error})
    with pytest.raises(TimeoutError) as raised:
        ci.supervise(tmp_path, owner)
    assert raised.value is error
    assert calls == (["launch", "cancel_request"] if known_request else ["launch"])


@pytest.mark.parametrize("bad", ["head", "infra", "run_id"])
def test_main_rejects_invalid_scope_before_launch(tmp_path, monkeypatch, bad):
    environment = {
        "GITHUB_RUN_ID": "42",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_REPOSITORY": "OpenPipe/ART",
        "GITHUB_EVENT_NAME": "pull_request",
        "EXPECTED_HEAD_SHA": "a" * 40,
        "SKY_INFRA": "k8s/cks-wb3",
    }
    environment[
        {"head": "EXPECTED_HEAD_SHA", "infra": "SKY_INFRA", "run_id": "GITHUB_RUN_ID"}[
            bad
        ]
    ] = "invalid"
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ci.subprocess, "check_output", lambda *args, **kwargs: "a" * 40)
    run = Mock()
    monkeypatch.setattr(ci, "supervise", run)
    with pytest.raises(ValueError):
        ci.main(tmp_path / "evidence")
    run.assert_not_called()


def test_real_worker_round_trip_with_fake_sdk(tmp_path):
    """Exercise the actual direct Python parent/worker JSON transport, without Sky."""
    (tmp_path / "sky.py").write_text("""
import os
from pathlib import Path
from types import SimpleNamespace as NS
class Task:
    @classmethod
    def from_yaml(cls, path): return cls()
    def set_resources_override(self, options): assert options == {"infra":"k8s/cks-wb3"}
def launch(task, *, cluster_name, retry_until_up):
    assert retry_until_up is False
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
        "SKY_INFRA": "k8s/cks-wb3",
        "EXPECTED_HEAD_SHA": head,
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
    assert "fake complete log" in (root / "logs.log").read_text()
