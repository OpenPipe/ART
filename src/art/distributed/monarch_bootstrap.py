from __future__ import annotations

"""Provider-neutral bootstrap for pinned torchmonarch 0.2.

Both worker and controller endpoints must be reachable only on one trusted private
network. Monarch 0.2 requires ``trust_all_connections`` because authenticated
transport is unavailable in that release.
"""

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import hashlib
import ipaddress
import multiprocessing
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from .rollout import InstalledAsyncCallable

DEFAULT_MONARCH_PORT = 22222
DEFAULT_STARTUP_TIMEOUT_S = 600.0
_INVALID_IDENTIFIER = re.compile(r"\W")
_MAX_IDENTIFIER_LENGTH = 48
_MONARCH_TIMEOUT_ENV = (
    "HYPERACTOR_HOST_SPAWN_READY_TIMEOUT",
    "HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT",
    "HYPERACTOR_MESH_ACTOR_SPAWN_MAX_IDLE",
    "HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE",
)
_WORKER_CODE = """\
import sys
from monarch.actor import run_worker_loop_forever
run_worker_loop_forever(address=sys.argv[1], ca="trust_all_connections")
"""


class _BootstrapContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExplicitHostBootstrap(_BootstrapContract):
    worker_addresses: tuple[str, ...]
    controller_rank: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_workers(self) -> "ExplicitHostBootstrap":
        if not self.worker_addresses:
            raise ValueError("worker_addresses must not be empty")
        if len(set(self.worker_addresses)) != len(self.worker_addresses):
            raise ValueError("worker_addresses must be unique")
        if self.controller_rank >= len(self.worker_addresses):
            raise ValueError("controller_rank must identify a worker address")
        return self


class SkyPilotBootstrap(_BootstrapContract):
    node_rank: int = Field(ge=0)
    node_ips: tuple[str, ...]
    port: int = Field(default=DEFAULT_MONARCH_PORT, ge=1, le=65534)

    @classmethod
    def from_environ(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        port: int = DEFAULT_MONARCH_PORT,
    ) -> "SkyPilotBootstrap":
        environ = os.environ if environ is None else environ
        try:
            node_rank = int(environ["SKYPILOT_NODE_RANK"])
            node_ips = tuple(environ["SKYPILOT_NODE_IPS"].replace(",", "\n").split())
            declared_nodes = int(environ["SKYPILOT_NUM_NODES"])
        except KeyError as error:
            raise RuntimeError(
                f"missing SkyPilot environment variable {error.args[0]}"
            ) from None
        if declared_nodes != len(node_ips):
            raise ValueError(
                f"SKYPILOT_NUM_NODES={declared_nodes} but received {len(node_ips)} IPs"
            )
        return cls(node_rank=node_rank, node_ips=node_ips, port=port)

    @model_validator(mode="after")
    def _validate_rank(self) -> "SkyPilotBootstrap":
        if not self.node_ips or self.node_rank >= len(self.node_ips):
            raise ValueError("SkyPilot node rank must identify a node IP")
        if len(set(self.node_ips)) != len(self.node_ips):
            raise ValueError("SKYPILOT_NODE_IPS must be unique")
        for node_ip in self.node_ips:
            try:
                ipaddress.ip_address(node_ip)
            except ValueError:
                raise ValueError(
                    f"SKYPILOT_NODE_IPS contains invalid IP address {node_ip!r}"
                ) from None
        return self

    @property
    def worker_addresses(self) -> tuple[str, ...]:
        return tuple(_tcp_address(ip, self.port) for ip in self.node_ips)

    @property
    def lifecycle_port(self) -> int:
        return self.port + 1


class SshHost(_BootstrapContract):
    target: str = Field(min_length=1)
    worker_host: str = Field(min_length=1)


class SshBootstrap(_BootstrapContract):
    hosts: tuple[SshHost, ...]
    python_executable: str = Field(min_length=1)
    port: int = Field(default=DEFAULT_MONARCH_PORT, ge=1, le=65535)
    ssh_args: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_hosts(self) -> "SshBootstrap":
        if not self.hosts:
            raise ValueError("hosts must not be empty")
        if len({host.target for host in self.hosts}) != len(self.hosts):
            raise ValueError("SSH targets must be unique")
        if len({host.worker_host for host in self.hosts}) != len(self.hosts):
            raise ValueError("worker hosts must be unique")
        return self

    @property
    def worker_addresses(self) -> tuple[str, ...]:
        return tuple(_tcp_address(host.worker_host, self.port) for host in self.hosts)


def _tcp_address(host: str, port: int) -> str:
    host = host.removeprefix("[").removesuffix("]")
    return f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"


def _parse_ssh_host(value: str) -> SshHost:
    target, separator, worker_host = value.partition("=")
    target = target.strip()
    if not separator:
        worker_host = target.rsplit("@", 1)[-1]
    return SshHost(target=target, worker_host=worker_host.strip())


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def monarch_identifier(value: str) -> str:
    """Return a stable valid Monarch mesh, proc, or actor identifier."""

    identifier = _INVALID_IDENTIFIER.sub("_", value)
    if not identifier or identifier[0].isdigit():
        identifier = f"art_{identifier}"
    if identifier == value and len(identifier) <= _MAX_IDENTIFIER_LENGTH:
        return identifier
    suffix = hashlib.sha256(value.encode()).hexdigest()[:8]
    prefix_length = _MAX_IDENTIFIER_LENGTH - len(suffix) - 1
    return f"{identifier[:prefix_length]}_{suffix}"


def _prepare_child_environment(*, worker: bool = False) -> None:
    # Monarch's spawned interpreter may resolve outside the active uv venv. Make
    # the controller's import roots explicit for ART and installed user code.
    roots = [path for path in sys.path if path and os.path.isabs(path)]
    roots.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(filter(None, roots)))
    if os.path.isfile(os.path.join(sys.prefix, "pyvenv.cfg")):
        os.environ.setdefault("ART_VIRTUAL_ENV", sys.prefix)
    if worker:
        nvidia_libs = (
            str(path)
            for root in roots
            for path in (Path(root) / "nvidia").glob("*/lib")
            if path.is_dir()
        )
        inherited = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            dict.fromkeys((*nvidia_libs, *filter(None, inherited)))
        )
    for name in _MONARCH_TIMEOUT_ENV:
        os.environ.setdefault(name, "600s")


def activate_child_virtualenv() -> None:
    """Restore venv identity lost when Monarch resolves the Python executable."""

    if virtual_env := os.environ.get("ART_VIRTUAL_ENV"):
        sys.prefix = sys.exec_prefix = virtual_env


async def deployment_smoke(hosts: Any) -> None:
    """Verify that the bootstrap attached every allocated host."""

    print(
        f"ART attached to {hosts.region.slice().sizes[0]} Monarch host(s)", flush=True
    )


def run_worker(address: str) -> None:
    """Run a pinned Monarch 0.2 worker on a trusted private network.

    Monarch 0.2 does not implement transport authentication, so its required
    trust-all mode must never be exposed to an untrusted or public network.
    """

    _prepare_child_environment(worker=True)
    # Importing ``art`` initializes enough third-party state to break Monarch's
    # spawned interpreter bootstrap. Replace this process with a clean worker.
    os.execv(sys.executable, [sys.executable, "-c", _WORKER_CODE, address])


async def attach_controller(
    worker_addresses: Sequence[str],
    *,
    name: str = "art",
    startup_timeout_s: float | None = None,
) -> Any:
    """Attach a controller to already-started workers on a trusted network."""

    _prepare_child_environment()
    from monarch.actor import (  # ty: ignore[unresolved-import]
        attach_to_workers,
        enable_transport,
    )

    enable_transport("tcp")
    hosts = attach_to_workers(
        workers=list(worker_addresses),
        ca="trust_all_connections",
        name=monarch_identifier(name),
    )
    if startup_timeout_s is None:
        await hosts.initialized
    else:
        await asyncio.wait_for(hosts.initialized, startup_timeout_s)
    return hosts


async def run_explicit_controller(
    spec: ExplicitHostBootstrap,
    program: "InstalledAsyncCallable",
    *,
    startup_timeout_s: float | None = None,
) -> Any:
    hosts = await attach_controller(
        spec.worker_addresses, startup_timeout_s=startup_timeout_s
    )
    try:
        return await program.resolve()(hosts)
    finally:
        await hosts.shutdown()


def _start_worker(address: str) -> multiprocessing.Process:
    worker = multiprocessing.Process(target=run_worker, args=(address,), daemon=True)
    worker.start()
    return worker


def _stop_worker(worker: multiprocessing.Process) -> None:
    if worker.is_alive():
        worker.terminate()
    worker.join(timeout=10)
    if worker.is_alive():
        worker.kill()
        worker.join()


def _lifecycle_listener(spec: SkyPilotBootstrap) -> socket.socket:
    # Attached Monarch 0.2 workers outlive HostMesh.shutdown(); task parents use
    # this private channel to leave together when rank 0 completes or disconnects.
    family = (
        socket.AF_INET6
        if ipaddress.ip_address(spec.node_ips[0]).version == 6
        else socket.AF_INET
    )
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((spec.node_ips[0], spec.lifecycle_port))
        listener.listen(len(spec.node_ips) - 1)
        return listener
    except BaseException:
        listener.close()
        raise


def _accept_sky_peers(
    spec: SkyPilotBootstrap,
    listener: socket.socket,
    worker: multiprocessing.Process,
    startup_timeout_s: float,
) -> list[socket.socket]:
    peers: list[socket.socket] = []
    deadline = time.monotonic() + startup_timeout_s
    try:
        while len(peers) < len(spec.node_ips) - 1:
            if not worker.is_alive():
                raise RuntimeError(f"Monarch worker exited with code {worker.exitcode}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = len(spec.node_ips) - 1 - len(peers)
                raise TimeoutError(f"timed out waiting for {missing} SkyPilot rank(s)")
            listener.settimeout(min(1.0, remaining))
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            connection.settimeout(None)
            peers.append(connection)
        return peers
    except BaseException:
        for connection in peers:
            connection.close()
        raise


def _wait_for_sky_controller(
    spec: SkyPilotBootstrap,
    worker: multiprocessing.Process,
    startup_timeout_s: float,
) -> None:
    deadline = time.monotonic() + startup_timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        if not worker.is_alive():
            raise RuntimeError(f"Monarch worker exited with code {worker.exitcode}")
        try:
            connection = socket.create_connection(
                (spec.node_ips[0], spec.lifecycle_port), timeout=1
            )
            break
        except OSError as error:
            last_error = error
            time.sleep(0.2)
    else:
        raise TimeoutError("timed out connecting to SkyPilot rank 0") from last_error
    with connection:
        connection.settimeout(None)
        status = connection.recv(1)
    if status != b"\x00":
        detail = "failed" if status == b"\x01" else "disconnected"
        raise RuntimeError(f"SkyPilot rank-0 ART controller {detail}")


def _notify_sky_peers(peers: Sequence[socket.socket], success: bool) -> None:
    status = b"\x00" if success else b"\x01"
    for connection in peers:
        try:
            connection.sendall(status)
        except OSError:
            pass
        finally:
            connection.close()


def run_skypilot(
    program_module: str,
    program_qualname: str,
    *,
    port: int = DEFAULT_MONARCH_PORT,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> None:
    """Translate SkyPilot topology and own one worker process per task rank."""

    from .rollout import InstalledAsyncCallable

    spec = SkyPilotBootstrap.from_environ(port=port)
    program = InstalledAsyncCallable(module=program_module, qualname=program_qualname)
    worker = _start_worker(spec.worker_addresses[spec.node_rank])
    if spec.node_rank != 0:
        try:
            _wait_for_sky_controller(spec, worker, startup_timeout_s)
        finally:
            _stop_worker(worker)
        return

    peers: list[socket.socket] = []
    listener: socket.socket | None = None
    success = False
    try:
        if len(spec.node_ips) > 1:
            listener = _lifecycle_listener(spec)
            peers = _accept_sky_peers(spec, listener, worker, startup_timeout_s)
        asyncio.run(
            run_explicit_controller(
                ExplicitHostBootstrap(worker_addresses=spec.worker_addresses),
                program,
                startup_timeout_s=startup_timeout_s,
            )
        )
        success = True
    finally:
        _notify_sky_peers(peers, success)
        if listener is not None:
            listener.close()
        _stop_worker(worker)


def _start_ssh_workers(spec: SshBootstrap) -> list[subprocess.Popen[bytes]]:
    workers: list[subprocess.Popen[bytes]] = []
    try:
        for host, address in zip(spec.hosts, spec.worker_addresses, strict=True):
            command = shlex.join(
                (
                    spec.python_executable,
                    "-m",
                    "art.distributed.monarch_bootstrap",
                    "worker",
                    "--address",
                    address,
                )
            )
            workers.append(
                subprocess.Popen(
                    (
                        "ssh",
                        "-n",
                        "-o",
                        "BatchMode=yes",
                        *spec.ssh_args,
                        host.target,
                        command,
                    ),
                    stdin=subprocess.DEVNULL,
                )
            )
        return workers
    except BaseException:
        _stop_ssh_workers(workers)
        raise


def _stop_ssh_workers(workers: Sequence[subprocess.Popen[bytes]]) -> None:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    for worker in workers:
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()


def run_ssh(
    spec: SshBootstrap,
    program: "InstalledAsyncCallable",
    *,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> None:
    """Start workers on passwordless SSH hosts and own them for one ART run."""

    workers = _start_ssh_workers(spec)
    try:
        asyncio.run(
            run_explicit_controller(
                ExplicitHostBootstrap(worker_addresses=spec.worker_addresses),
                program,
                startup_timeout_s=startup_timeout_s,
            )
        )
    finally:
        _stop_ssh_workers(workers)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ART Monarch bootstrap (trusted private networks only)",
        epilog=(
            "torchmonarch 0.2 has no transport authentication; never expose worker "
            "addresses to a public or untrusted network."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--address", required=True)
    controller = subparsers.add_parser(
        "controller", help="attach to worker commands managed by the caller"
    )
    controller.add_argument("--worker", action="append", required=True)
    controller.add_argument("--program", required=True, help="module:qualname")
    controller.add_argument(
        "--startup-timeout", type=_positive_float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    sky = subparsers.add_parser(
        "skypilot", help="consume the nodes in one SkyPilot task"
    )
    sky.add_argument("--program", required=True, help="module:qualname")
    sky.add_argument("--port", type=int, default=DEFAULT_MONARCH_PORT)
    sky.add_argument(
        "--startup-timeout", type=_positive_float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    ssh = subparsers.add_parser(
        "ssh", help="start and own workers on preallocated SSH hosts"
    )
    ssh.add_argument(
        "--host",
        action="append",
        required=True,
        help="[USER@]SSH_TARGET[=WORKER_HOST]",
    )
    ssh.add_argument("--program", required=True, help="module:qualname")
    ssh.add_argument("--python", default=sys.executable, dest="python_executable")
    ssh.add_argument("--port", type=int, default=DEFAULT_MONARCH_PORT)
    ssh.add_argument(
        "--ssh-arg",
        action="append",
        default=[],
        help="argument passed to ssh; use --ssh-arg=VALUE",
    )
    ssh.add_argument(
        "--startup-timeout", type=_positive_float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    args = parser.parse_args(argv)
    if args.command == "worker":
        run_worker(args.address)
        return
    module, separator, qualname = args.program.partition(":")
    if not module or not separator or not qualname:
        parser.error("--program must use module:qualname")
    if args.command == "skypilot":
        run_skypilot(
            module,
            qualname,
            port=args.port,
            startup_timeout_s=args.startup_timeout,
        )
        return
    from .rollout import InstalledAsyncCallable

    program = InstalledAsyncCallable(module=module, qualname=qualname)
    if args.command == "ssh":
        run_ssh(
            SshBootstrap(
                hosts=tuple(_parse_ssh_host(host) for host in args.host),
                python_executable=args.python_executable,
                port=args.port,
                ssh_args=tuple(args.ssh_arg),
            ),
            program,
            startup_timeout_s=args.startup_timeout,
        )
    else:
        asyncio.run(
            run_explicit_controller(
                ExplicitHostBootstrap(worker_addresses=tuple(args.worker)),
                program,
                startup_timeout_s=args.startup_timeout,
            )
        )


if __name__ == "__main__":
    main()
