"""Private GitHub-hosted CI enclosure; never discover or stop another API server."""

import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

CGROUPS = Path("/sys/fs/cgroup")
PROC = Path("/proc")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
PROPERTIES = {
    "Transient": "yes",
    "Type": "exec",
    "ExitType": "cgroup",
    "KillMode": "control-group",
    "SendSIGKILL": "yes",
    "Restart": "no",
    "NoNewPrivileges": "yes",
    "ProtectControlGroups": "yes",
}


def write(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read(root, name, expected):
    data = json.loads((root / name).read_text())
    if any(data.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Wrong CI API identity in {name}")
    return data


def host():
    if (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
        or os.getuid() == 0
        or not (CGROUPS / "cgroup.controllers").is_file()
    ):
        raise ValueError("CI API requires an isolated GitHub-hosted cgroup-v2 runner")


def unit_name(owner):
    if not re.fullmatch(r"[0-9a-f]{32}", owner["label"]):
        raise ValueError("Invalid API nonce")
    return f"art-ci-{owner['label']}.service"


def identity(root, owner):
    host()
    plan = read(root, "api-plan.json", owner)
    unit = unit_name(owner)
    expected = {
        **plan,
        "unit": unit,
        "cgroup": f"/system.slice/{unit}",
        "boot_id": BOOT_ID.read_text().strip(),
        "uid": os.getuid(),
        "gid": os.getgid(),
    }
    receipt = read(root, "api-service.json", expected)
    if not re.fullmatch(r"[0-9a-f]{32}", receipt["invocation"]):
        raise ValueError("Invalid API invocation")
    return receipt


def show(receipt):
    result = subprocess.run(
        [
            "systemctl",
            "show",
            receipt["unit"],
            "--no-pager",
            "--property="
            + ",".join(
                [
                    *PROPERTIES,
                    "LoadState",
                    "ActiveState",
                    "SubState",
                    "InvocationID",
                    "ControlGroup",
                    "User",
                    "Group",
                    "MainPID",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    if result.returncode and not (
        result.returncode == 4 and values.get("LoadState") == "not-found"
    ):
        raise RuntimeError("Cannot inspect owned API unit")
    return values


def matching(receipt, values):
    expected = {
        **PROPERTIES,
        "InvocationID": receipt["invocation"],
        "ControlGroup": receipt["cgroup"],
        "User": str(receipt["uid"]),
        "Group": str(receipt["gid"]),
    }
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError("Owned API unit changed identity or containment")


def empty(receipt):
    path = CGROUPS / receipt["cgroup"].lstrip("/")
    try:
        events = dict(
            line.split() for line in (path / "cgroup.events").read_text().splitlines()
        )
    except FileNotFoundError:
        # The kernel only removes an empty cgroup. Permission/read failures do
        # not count, nor does a missing events file in a still-existing cgroup.
        try:
            path.stat()
        except FileNotFoundError:
            return True
        raise RuntimeError("Owned API cgroup lacks population evidence")
    if events.get("populated") not in {"0", "1"}:
        raise RuntimeError("Owned API cgroup population is unknown")
    return events["populated"] == "0"


def require_running(root, owner):
    receipt = identity(root, owner)
    values = show(receipt)
    matching(receipt, values)
    if values.get("ActiveState") != "active" or empty(receipt):
        raise RuntimeError("Owned API service is not running")
    return receipt


def client_environment(root, owner):
    receipt = require_running(root, owner)
    if not owns_listener(receipt):
        raise RuntimeError("Owned API listener is unavailable")
    os.environ["SKYPILOT_DISABLE_LOCAL_API_SERVER"] = "1"
    os.environ["SKYPILOT_API_SERVER_ENDPOINT"] = f"http://127.0.0.1:{receipt['port']}"
    os.environ["SKYPILOT_API_SERVER_LOCAL_PORT"] = str(receipt["port"])


def owns_listener(receipt):
    """Check only this cgroup's processes for the exact loopback listening socket."""
    address = f"0100007F:{receipt['port']:04X}"
    sockets = {
        f"socket:[{fields[9]}]"
        for line in (PROC / "net/tcp").read_text().splitlines()[1:]
        if (fields := line.split())[1] == address and fields[3] == "0A"
    }
    group = CGROUPS / receipt["cgroup"].lstrip("/")
    for members in group.rglob("cgroup.procs"):
        for pid in members.read_text().split():
            process = PROC / pid
            try:
                before = (process / "cgroup").read_text()
                owned = f"0::{receipt['cgroup']}\n"
                if before != owned:
                    continue
                found = any(
                    os.readlink(fd) in sockets for fd in (process / "fd").iterdir()
                )
                if found and (process / "cgroup").read_text() == before:
                    return True
            except FileNotFoundError:
                continue
    return False


def start(root, owner, *, probe=False):
    host()
    unit = unit_name(owner)
    # Do not retry/recreate an invocation, including an interrupted startup.
    if (root / "api-plan.json").exists():
        raise ValueError("CI API startup was already attempted")
    # Reserve 10s for systemd-run, 5s TERM grace, and 5s scheduling margin.
    # This is never renewed, even when launch/setup consumes the work window.
    remaining = math.floor(owner["cleanup_deadline"] - time.time()) - 20
    if remaining <= 0 or time.time() >= owner["admission_deadline"]:
        raise TimeoutError("CI API lifetime expired")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    plan = {
        **owner,
        "unit": unit,
        "port": port,
        "runtime_seconds": remaining,
        "probe": probe,
    }
    write(root / "api-plan.json", plan)
    properties = {
        **PROPERTIES,
        "TimeoutStartSec": "10s",
        "TimeoutStopSec": "5s",
        "RuntimeMaxSec": f"{remaining}s",
    }
    command = [
        "sudo",
        "-n",
        "systemd-run",
        f"--unit={plan['unit']}",
        f"--uid={os.getuid()}",
        f"--gid={os.getgid()}",
        f"--working-directory={Path.cwd()}",
    ]
    command += [
        f"--property={key}={value}"
        for key, value in properties.items()
        if key != "Transient"
    ]
    # Workflow auth/config files live under HOME; task context/labels travel
    # in the SDK payload. Retain the runner's executables and optional kubeconfig.
    command += [
        f"--setenv={key}={os.environ[key]}"
        for key in ("HOME", "PATH", "USER", "LOGNAME", "KUBECONFIG")
        if key in os.environ
    ]
    command += [
        sys.executable,
        str(Path(__file__).resolve()),
        "serve",
        str(root.resolve()),
    ]
    with (root / "api-start.log").open("ab") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=10
        )
    deadline = min(time.time() + 60, owner["admission_deadline"])
    while not (root / "api-service.json").exists():
        if time.time() >= deadline:
            raise TimeoutError("CI API did not record its cgroup identity")
        time.sleep(0.1)
    receipt = require_running(root, owner)
    if time.time() >= deadline:
        raise TimeoutError("CI API startup missed admission")
    write(root / "api-release.json", receipt)
    while True:
        require_running(root, owner)
        if probe:
            if (root / "api-probe.json").exists() and show(receipt).get(
                "MainPID"
            ) == "0":
                read(root, "api-probe.json", receipt)
                return
        elif owns_listener(receipt):
            return
        if time.time() >= deadline:
            raise TimeoutError("Owned API listener did not become ready")
        time.sleep(0.1)


def stop(root, owner):
    receipt = identity(root, owner)
    if (root / "api-retired.json").exists():
        require_retired(root, owner)
        return
    values = show(receipt)
    already_stopped = (
        values.get("ActiveState") in {"inactive", "failed"}
        and values.get("InvocationID") == receipt["invocation"]
        and values.get("ControlGroup") in {"", receipt["cgroup"]}
        and empty(receipt)
    )
    if values.get("LoadState") != "not-found" and not already_stopped:
        matching(receipt, values)
        # No asynchronous stop: systemd first sends TERM to the entire cgroup,
        # then KILL after five seconds, including helpers which called setsid().
        subprocess.run(
            ["sudo", "-n", "systemctl", "stop", receipt["unit"]], check=True, timeout=15
        )
    if not empty(receipt):
        raise RuntimeError("Owned API cgroup retirement is unproven")
    write(root / "api-retired.json", {**receipt, "creator_quiescence": "RETIRED"})


def require_retired(root, owner):
    receipt = identity(root, owner)
    read(root, "api-retired.json", {**receipt, "creator_quiescence": "RETIRED"})
    if not empty(receipt):
        raise RuntimeError("Owned API cgroup is populated after retirement")


def serve(root):
    plan = json.loads((root / "api-plan.json").read_text())
    group = f"/system.slice/{plan['unit']}"
    if (PROC / "self/cgroup").read_text() != f"0::{group}\n":
        raise ValueError("API service did not enter the planned cgroup")
    receipt = {
        **plan,
        "cgroup": group,
        "invocation": os.environ["INVOCATION_ID"],
        "boot_id": BOOT_ID.read_text().strip(),
        "uid": os.getuid(),
        "gid": os.getgid(),
    }
    write(root / "api-service.json", receipt)
    deadline = min(time.time() + 15, plan["admission_deadline"])
    while not (root / "api-release.json").exists():
        if time.time() >= deadline:
            raise TimeoutError("API service was not released by its owner")
        time.sleep(0.1)
    read(root, "api-release.json", receipt)
    if time.time() >= plan["admission_deadline"]:
        raise TimeoutError("CI API release missed admission")
    if plan["probe"]:
        # CPU-only hosted qualification: an ordinary detached child remains in
        # this service cgroup after its leader exits. No SDK/network/GPU import.
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(30)",
                str(root / "api-probe-ready"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.time() + 5
        while not (root / "api-probe-ready").exists():
            if time.time() >= deadline:
                raise TimeoutError("Detached probe did not start")
            time.sleep(0.05)
        write(root / "api-probe.json", receipt)
        return
    os.environ.update(
        SKYPILOT_DISABLE_LOCAL_API_SERVER="1",
        SKYPILOT_API_SERVER_ENDPOINT=f"http://127.0.0.1:{plan['port']}",
        SKYPILOT_API_SERVER_LOCAL_PORT=str(plan["port"]),
        IS_SKYPILOT_SERVER="true",
    )
    # This is the exec target of the pinned SDK's api_start(foreground=True).
    # Direct exec keeps the SDK autostart prohibition inherited by ALL children.
    log = os.open(
        root / "api-server.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "sky.server.server",
            "--host=127.0.0.1",
            f"--port={plan['port']}",
        ],
    )


def probe(root):
    root.mkdir(parents=True, exist_ok=False)
    now = time.time()
    owner = {
        "label": uuid.uuid4().hex,
        "admission_deadline": now + 30,
        "cleanup_deadline": now + 60,
    }
    try:
        start(root, owner, probe=True)
    finally:
        if (root / "api-service.json").exists():
            stop(root, owner)
    require_retired(root, owner)


if __name__ == "__main__":
    {"serve": serve, "probe": probe}[sys.argv[1]](Path(sys.argv[2]))
