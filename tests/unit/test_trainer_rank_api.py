"""File-backed systemd/cgroup controls only: never start a service or Sky API."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/ci/trainer_rank_api.py"
spec = importlib.util.spec_from_file_location("api_enclosure", SCRIPT)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


@pytest.fixture
def control(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    cgroups = tmp_path / "cgroups"
    cgroups.mkdir()
    (cgroups / "cgroup.controllers").touch()
    boot = tmp_path / "boot-id"
    boot.write_text("this-boot")
    monkeypatch.setattr(api, "CGROUPS", cgroups)
    monkeypatch.setattr(api, "BOOT_ID", boot)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    monkeypatch.setattr(api.os, "getuid", lambda: 1001)
    monkeypatch.setattr(api.os, "getgid", lambda: 1002)
    monkeypatch.setattr(api.time, "time", lambda: 100)
    owner = {"label": "a" * 32, "admission_deadline": 1000, "cleanup_deadline": 2000}
    unit = f"art-ci-{owner['label']}.service"
    plan = {
        **owner,
        "unit": unit,
        "port": 12345,
        "runtime_seconds": 1880,
        "probe": False,
    }
    receipt = {
        **plan,
        "cgroup": f"/system.slice/{unit}",
        "invocation": "b" * 32,
        "boot_id": "this-boot",
        "uid": 1001,
        "gid": 1002,
    }
    group = cgroups / receipt["cgroup"].lstrip("/")
    group.mkdir(parents=True)
    events = group / "cgroup.events"
    events.write_text("populated 1\nfrozen 0\n")
    api.write(root / "api-plan.json", plan)
    api.write(root / "api-service.json", receipt)
    values = {
        **api.PROPERTIES,
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": receipt["invocation"],
        "ControlGroup": receipt["cgroup"],
        "User": "1001",
        "Group": "1002",
        "MainPID": "42",
    }
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["systemctl", "show"]:
            assert command[2] == unit
            return NS(
                returncode=0, stdout="\n".join(f"{k}={v}" for k, v in values.items())
            )
        assert command == ["sudo", "-n", "systemctl", "stop", unit]
        events.write_text("populated 0\nfrozen 0\n")
        return NS(returncode=0)

    monkeypatch.setattr(api.subprocess, "run", run)
    return NS(
        root=root,
        owner=owner,
        receipt=receipt,
        group=group,
        events=events,
        values=values,
        calls=calls,
        run=run,
    )


def test_stop_proves_whole_cgroup_empty_and_is_idempotent(control):
    c = control
    api.stop(c.root, c.owner)
    api.require_retired(c.root, c.owner)
    proof = json.loads((c.root / "api-retired.json").read_text())
    assert proof["creator_quiescence"] == "RETIRED"
    assert proof["invocation"] == c.receipt["invocation"]
    assert c.calls[-1][0] == ["sudo", "-n", "systemctl", "stop", c.receipt["unit"]]
    assert c.calls[-1][1]["timeout"] == 15
    calls = list(c.calls)
    api.stop(c.root, c.owner)
    assert c.calls == calls
    c.events.write_text("populated 1\n")
    with pytest.raises(RuntimeError, match="populated"):
        api.require_retired(c.root, c.owner)


@pytest.mark.parametrize(
    "field,value",
    [
        ("InvocationID", "c" * 32),
        ("ControlGroup", "/system.slice/peer.service"),
        ("User", "1003"),
        ("Group", "1003"),
        ("KillMode", "process"),
        ("Restart", "always"),
        ("ExitType", "main"),
        ("ProtectControlGroups", "no"),
    ],
)
def test_changed_unit_never_receives_stop(control, field, value):
    c = control
    c.values[field] = value
    with pytest.raises(ValueError, match="changed identity"):
        api.stop(c.root, c.owner)
    assert all(command[0] == "systemctl" for command, _ in c.calls)
    assert not (c.root / "api-retired.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [("boot_id", "old-boot"), ("cgroup", "/peer"), ("uid", 123), ("invocation", "")],
)
def test_receipt_mismatch_never_controls_service(control, field, value):
    c = control
    api.write(c.root / "api-service.json", {**c.receipt, field: value})
    with pytest.raises(ValueError):
        api.stop(c.root, c.owner)
    assert c.calls == []


def test_stop_success_or_terminal_cancellation_does_not_prove_retirement(
    control, monkeypatch
):
    c = control

    def no_retirement(command, **kwargs):
        if command[0] == "systemctl":
            return c.run(command, **kwargs)
        return NS(returncode=0)  # Detached late executor still populates cgroup.

    monkeypatch.setattr(api.subprocess, "run", no_retirement)
    api.write(c.root / "request-terminal.json", {**c.owner, "status": "CANCELLED"})
    with pytest.raises(RuntimeError, match="retirement is unproven"):
        api.stop(c.root, c.owner)
    assert not (c.root / "api-retired.json").exists()


def test_missing_receipt_and_ambiguous_cgroup_are_unknown(control):
    c = control
    c.events.unlink()
    with pytest.raises(RuntimeError, match="population evidence"):
        api.empty(c.receipt)
    (c.root / "api-service.json").unlink()
    with pytest.raises(FileNotFoundError):
        api.stop(c.root, c.owner)
    assert c.calls == []


def test_collected_unit_requires_recorded_cgroup_to_have_disappeared(control):
    c = control
    c.values.clear()
    c.values["LoadState"] = "not-found"
    with pytest.raises(RuntimeError, match="unproven"):
        api.stop(c.root, c.owner)
    c.events.unlink()
    c.group.rmdir()
    api.stop(c.root, c.owner)
    api.require_retired(c.root, c.owner)
    assert all(command[0] == "systemctl" for command, _ in c.calls)


def test_stop_timeout_cannot_emit_retirement(control, monkeypatch):
    c = control

    def timeout(command, **kwargs):
        if command[0] == "systemctl":
            return c.run(command, **kwargs)
        raise subprocess.TimeoutExpired(command, 15)

    monkeypatch.setattr(api.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        api.stop(c.root, c.owner)
    assert not (c.root / "api-retired.json").exists()


def test_client_cannot_choose_another_endpoint_or_autostart(control, monkeypatch):
    c = control
    monkeypatch.setattr(api, "owns_listener", lambda _: True)
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://peer")
    monkeypatch.setenv("SKYPILOT_DISABLE_LOCAL_API_SERVER", "0")
    api.client_environment(c.root, c.owner)
    assert os.environ["SKYPILOT_API_SERVER_ENDPOINT"] == "http://127.0.0.1:12345"
    assert os.environ["SKYPILOT_DISABLE_LOCAL_API_SERVER"] == "1"
    c.values["ActiveState"] = "failed"
    with pytest.raises(RuntimeError, match="not running"):
        api.client_environment(c.root, c.owner)


def test_start_has_finite_runtime_and_receipt_gate_before_release(control, monkeypatch):
    c = control
    (c.root / "api-plan.json").unlink()
    (c.root / "api-service.json").unlink()
    reservation = Mock()
    reservation.getsockname.return_value = ("127.0.0.1", 12345)
    context = MagicMock()
    context.__enter__.return_value = reservation
    monkeypatch.setattr(api.socket, "socket", Mock(return_value=context))
    command_seen = []

    def start(command, **kwargs):
        if command[:3] == ["sudo", "-n", "systemd-run"]:
            assert not (c.root / "api-release.json").exists()
            command_seen.extend(command)
            api.write(c.root / "api-service.json", c.receipt)
            return NS(returncode=0)
        return c.run(command, **kwargs)

    monkeypatch.setattr(api.subprocess, "run", start)

    def listener(receipt):
        assert json.loads((c.root / "api-release.json").read_text()) == receipt
        return True

    monkeypatch.setattr(api, "owns_listener", listener)
    api.start(c.root, c.owner)
    assert "--property=RuntimeMaxSec=1880s" in command_seen
    assert "--property=TimeoutStopSec=5s" in command_seen
    assert "--property=KillMode=control-group" in command_seen
    assert "--property=ExitType=cgroup" in command_seen
    assert "--uid=1001" in command_seen and "--gid=1002" in command_seen
    with pytest.raises(ValueError, match="already attempted"):
        api.start(c.root, c.owner)


@pytest.mark.parametrize("host", ["self-hosted", ""])
def test_unsupported_host_is_rejected_without_service_control(
    control, monkeypatch, host
):
    c = control
    monkeypatch.setenv("RUNNER_ENVIRONMENT", host)
    with pytest.raises(ValueError, match="GitHub-hosted"):
        api.start(c.root, c.owner)
    assert c.calls == []


def test_listener_must_belong_to_exact_cgroup(control, monkeypatch, tmp_path):
    c = control
    proc = tmp_path / "proc"
    (proc / "net").mkdir(parents=True)
    (proc / "net/tcp").write_text(
        "header\n0: 0100007F:3039 00000000:0000 0A 0 0 0 0 0 123\n"
    )
    (proc / "42/fd").mkdir(parents=True)
    (proc / "42/fd/7").symlink_to("socket:[123]")
    (proc / "42/cgroup").write_text("0::/peer\n")
    (c.group / "cgroup.procs").write_text("42\n")
    monkeypatch.setattr(api, "PROC", proc)
    assert not api.owns_listener(c.receipt)
    (proc / "42/cgroup").write_text(f"0::{c.receipt['cgroup']}\n")
    assert api.owns_listener(c.receipt)
    assert not api.owns_listener({**c.receipt, "port": 12346})


def test_runtime_expiry_accepts_only_same_invocation_and_empty_cgroup(control):
    c = control
    c.values.update(ActiveState="failed", ControlGroup="")
    with pytest.raises(ValueError):
        api.stop(c.root, c.owner)
    c.events.write_text("populated 0\n")
    api.stop(c.root, c.owner)
    api.require_retired(c.root, c.owner)
    assert all(command[0] == "systemctl" for command, _ in c.calls)


def test_live_unit_without_owned_listener_cannot_receive_client_calls(
    control, monkeypatch
):
    monkeypatch.setattr(api, "owns_listener", lambda _: False)
    with pytest.raises(RuntimeError, match="listener is unavailable"):
        api.client_environment(control.root, control.owner)


@pytest.mark.parametrize("events", ["", "frozen 0\n", "populated unknown\n"])
def test_unknown_population_blocks_start_and_retirement(control, events, monkeypatch):
    c = control
    monkeypatch.setattr(
        api.subprocess,
        "run",
        lambda command, **kwargs: (
            c.run(command, **kwargs) if command[0] == "systemctl" else NS(returncode=0)
        ),
    )
    c.events.write_text(events)
    with pytest.raises(RuntimeError, match="population is unknown"):
        api.require_running(c.root, c.owner)
    with pytest.raises(RuntimeError, match="population is unknown"):
        api.stop(c.root, c.owner)
    assert not (c.root / "api-retired.json").exists()


def test_invalid_nonce_is_rejected_before_any_service_creation(control):
    c = control
    (c.root / "api-plan.json").unlink()
    with pytest.raises(ValueError, match="nonce"):
        api.start(c.root, {**c.owner, "label": "../peer"})
    assert c.calls == []


@pytest.mark.parametrize("released", [True, False])
def test_foreground_server_inherits_autostart_prohibition_only_after_release(
    control, monkeypatch, tmp_path, released
):
    c = control
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text(f"0::{c.receipt['cgroup']}\n")
    monkeypatch.setattr(api, "PROC", proc)
    monkeypatch.setenv("INVOCATION_ID", c.receipt["invocation"])
    monkeypatch.setenv("SKYPILOT_DISABLE_LOCAL_API_SERVER", "0")
    api.write(
        c.root / "api-release.json",
        c.receipt if released else {**c.receipt, "invocation": "peer"},
    )
    execute = Mock(side_effect=RuntimeError("exec boundary"))
    monkeypatch.setattr(api.os, "execv", execute)
    monkeypatch.setattr(api.os, "dup2", Mock())
    with pytest.raises(RuntimeError if released else ValueError):
        api.serve(c.root)
    if released:
        assert os.environ["SKYPILOT_DISABLE_LOCAL_API_SERVER"] == "1"
        assert os.environ["IS_SKYPILOT_SERVER"] == "true"
        assert execute.call_args.args[1][1:] == [
            "-m",
            "sky.server.server",
            "--host=127.0.0.1",
            "--port=12345",
        ]
    else:
        execute.assert_not_called()
    assert c.calls == []


def test_timed_out_start_cannot_release_a_later_service(control, monkeypatch, tmp_path):
    c = control
    (c.root / "api-plan.json").unlink()
    (c.root / "api-service.json").unlink()
    context = MagicMock()
    context.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
    monkeypatch.setattr(api.socket, "socket", Mock(return_value=context))
    start = Mock(side_effect=subprocess.TimeoutExpired("systemd-run", 10))
    monkeypatch.setattr(api.subprocess, "run", start)
    with pytest.raises(subprocess.TimeoutExpired):
        api.start(c.root, c.owner)
    assert "--property=RuntimeMaxSec=1880s" in start.call_args.args[0]
    assert not (c.root / "api-release.json").exists()
    assert not (c.root / "api-service.json").exists()

    # Simulate systemd accepting the timed-out request later. Its process has
    # no permission receipt, independently times out, and never execs Sky.
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text(f"0::{c.receipt['cgroup']}\n")
    monkeypatch.setattr(api, "PROC", proc)
    monkeypatch.setenv("INVOCATION_ID", c.receipt["invocation"])
    clock = iter([100, 116])
    monkeypatch.setattr(api.time, "time", lambda: next(clock))
    execute = Mock()
    monkeypatch.setattr(api.os, "execv", execute)
    with pytest.raises(TimeoutError, match="not released"):
        api.serve(c.root)
    assert (c.root / "api-service.json").exists()
    assert not (c.root / "api-release.json").exists()
    execute.assert_not_called()
