from __future__ import annotations

"""Provider-neutral bootstrap for pinned torchmonarch 0.6.

Both worker and controller endpoints must be reachable only on one trusted private
network because ART currently configures Monarch with ``trust_all_connections``.
"""

import argparse
import asyncio
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import ipaddress
import os
from pathlib import Path
import re
import select
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from .rollout import InstalledAsyncCallable

DEFAULT_MONARCH_PORT = 22222
DEFAULT_STARTUP_TIMEOUT_S = 600.0
_INVALID_IDENTIFIER = re.compile(r"\W")
_MAX_IDENTIFIER_LENGTH = 48
_SSH_LAUNCH_ID = re.compile(r"^[0-9a-f]{32}$")
_SSH_READY_PREFIX = b"ART_MONARCH_READY "
_PROGRAM_PYTHONPATH_ENV = "ART_MONARCH_PROGRAM_PYTHONPATH"
_MONARCH_TIMEOUT_ENV = (
    "HYPERACTOR_HOST_SPAWN_READY_TIMEOUT",
    "HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT",
    "HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT",
    "HYPERACTOR_MESH_ACTOR_SPAWN_MAX_IDLE",
    "HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE",
)
_MONARCH_SHUTDOWN_ENV = {
    "HYPERACTOR_PROCESS_EXIT_TIMEOUT": "2s",
    "HYPERACTOR_MESH_PROC_STOP_MAX_IDLE": "240s",
}
_WORKER_ADDRESS_LOCK = threading.Lock()
_USED_WORKER_ADDRESSES: set[str] = set()
_BROKEN_OUTPUT_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
_WORKER_CODE = """\
import ctypes
import os
import signal
import sys

if len(sys.argv) >= 4 and sys.argv[2] == "--parent-pid":
    expected_parent_pid = int(sys.argv[3])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0):
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)

from monarch.actor import run_worker_loop_forever
run_worker_loop_forever(address=sys.argv[1], ca="trust_all_connections")
"""
_LEGACY_OWNED_WORKER_CODE = """\
import sys
from monarch.actor import run_worker_loop_forever
run_worker_loop_forever(address=sys.argv[1], ca="trust_all_connections")
"""
_WORKER_LOCK_ROOT = Path("/tmp")
_OWNED_WORKER_SCHEMA = "art.monarch.owned-worker.v1"


class _BootstrapContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _OwnedWorkerMetadata(_BootstrapContract):
    schema_name: str = _OWNED_WORKER_SCHEMA
    address: str
    controller_pid: int = Field(gt=0)
    controller_start_time: int = Field(gt=0)
    worker_pid: int = Field(gt=0)
    worker_start_time: int = Field(gt=0)
    python_executable: str = Field(min_length=1)
    worker_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ownership_token: str = Field(pattern=r"^[0-9a-f]{32}$")

    @model_validator(mode="after")
    def _validate_schema(self) -> "_OwnedWorkerMetadata":
        if self.schema_name != _OWNED_WORKER_SCHEMA:
            raise ValueError("unsupported owned-worker metadata schema")
        return self


class _WorkerSession(_BootstrapContract):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    address: str
    process: subprocess.Popen[bytes]
    label: str
    graceful: bool = False
    launch_id: str | None = None
    lease: Any = None

    @property
    def exitcode(self) -> int | None:
        return self.process.poll()

    def is_alive(self) -> bool:
        return self.exitcode is None

    def release(self) -> None:
        if not self.is_alive():
            return
        if self.graceful:
            assert self.process.stdin is not None
            self.process.stdin.close()
        else:
            os.killpg(self.process.pid, signal.SIGTERM)

    def wait(self) -> None:
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if self.graceful:
                self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            raise RuntimeError(f"{self.label} did not stop in time") from None
        finally:
            if self.lease is not None:
                self.lease.close()
        if self.graceful and self.process.returncode:
            raise RuntimeError(f"{self.label} exited {self.process.returncode}")


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


def require_local_worker_address(worker_addresses: Sequence[str]) -> str:
    error = "local ART runtime requires exactly one loopback tcp worker address"
    if len(worker_addresses) != 1:
        raise ValueError(error)
    address = worker_addresses[0]
    try:
        parsed = urlsplit(address)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError(error) from None
    if (
        parsed.scheme != "tcp"
        or host is None
        or port is None
        or port < 0
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(error)
    try:
        is_loopback = (
            host.casefold() == "localhost" or ipaddress.ip_address(host).is_loopback
        )
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError(error)
    return address


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


def _program_reference(value: str) -> tuple[str, str]:
    target, separator, qualname = value.partition(":")
    if not target or not separator or not qualname:
        raise ValueError("--program must use module:qualname or path.py:qualname")
    if Path(target).suffix != ".py":
        return target, qualname

    script = Path(target).expanduser().resolve()
    if not script.is_file():
        raise ValueError(f"program script does not exist: {script}")
    module = script.stem
    if not module.isidentifier():
        raise ValueError(
            f"program script name must be a Python identifier: {script.name}"
        )

    root = str(script.parent)
    sys.path[:] = [root, *(path for path in sys.path if path != root)]
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys((root, *filter(None, inherited)))
    )
    os.environ[_PROGRAM_PYTHONPATH_ENV] = root
    importlib.invalidate_caches()
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != script:
        raise ValueError(f"program module {module!r} does not resolve to {script}")
    return module, qualname


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


def _prepare_child_environment(
    *,
    worker: bool = False,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    environ = os.environ if environ is None else environ
    # Monarch's spawned interpreter may resolve outside the active uv venv. Make
    # the controller's import roots explicit for ART and installed user code.
    roots = [path for path in sys.path if path and os.path.isabs(path)]
    roots.extend(environ.get("PYTHONPATH", "").split(os.pathsep))
    environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(filter(None, roots)))
    if os.path.isfile(os.path.join(sys.prefix, "pyvenv.cfg")):
        environ.setdefault("ART_VIRTUAL_ENV", sys.prefix)
    if worker:
        environ.pop("CUDA_VISIBLE_DEVICES", None)
        allocator_config = "expandable_segments:True"
        environ["PYTORCH_ALLOC_CONF"] = allocator_config
        environ["PYTORCH_CUDA_ALLOC_CONF"] = allocator_config
        nvidia_libs = (
            str(path)
            for root in roots
            for path in (Path(root) / "nvidia").glob("*/lib")
            if path.is_dir()
        )
        inherited = environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            dict.fromkeys((*nvidia_libs, *filter(None, inherited)))
        )
    for name in _MONARCH_TIMEOUT_ENV:
        environ.setdefault(name, "600s")
    for name, value in _MONARCH_SHUTDOWN_ENV.items():
        environ.setdefault(name, value)
    # INFO launch records include the inherited environment and may expose secrets.
    environ.setdefault("MONARCH_FILE_LOG", "warn")


def _stabilize_child_stdio() -> None:
    fds = (sys.stdout.fileno(), sys.stderr.fileno())
    poller = select.poll()
    for fd in fds:
        poller.register(fd, select.POLLOUT)
    if not any(flags & _BROKEN_OUTPUT_FLAGS for _, flags in poller.poll(0)):
        return
    log_dir = Path(
        os.environ.get("ART_MONARCH_CHILD_LOG_DIR") or "/tmp/art-monarch-child-logs"
    )
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (log_dir / f"{socket.gethostname()}-{os.getpid()}.log").open("ab") as log:
        for fd in fds:
            os.dup2(log.fileno(), fd)


def activate_child_virtualenv() -> None:
    """Restore venv identity lost when Monarch resolves the Python executable."""

    _stabilize_child_stdio()
    if virtual_env := os.environ.get("ART_VIRTUAL_ENV"):
        sys.prefix = sys.exec_prefix = virtual_env


def activate_trainer_child_virtualenv() -> None:
    threads = os.environ.get("MKL_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, threads)
    activate_child_virtualenv()


def activate_cuda_device(gpu_id: int | str) -> int:
    """Bind a clean trainer process to one physical ordinal or CUDA UUID."""

    if isinstance(gpu_id, str):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        return 0
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError(
            "physical GPU placement requires an unmasked Monarch worker process"
        )
    return gpu_id


def activate_cpu_child_virtualenv() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    activate_child_virtualenv()


def _owns_tcp_listener(pid: int, port: int) -> bool:
    socket_inodes = {
        target[8:-1]
        for descriptor in Path(f"/proc/{pid}/fd").iterdir()
        if (target := os.readlink(descriptor)).startswith("socket:[")
    }
    for table in ("tcp", "tcp6"):
        for line in Path(f"/proc/{pid}/net/{table}").read_text().splitlines()[1:]:
            fields = line.split()
            if (
                len(fields) > 9
                and fields[3] == "0A"
                and int(fields[1].rsplit(":", 1)[1], 16) == port
                and fields[9] in socket_inodes
            ):
                return True
    return False


def _stop_worker_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_ssh_worker_session(
    address: str,
    launch_id: str,
    startup_timeout_s: float,
) -> None:
    port = urlsplit(address).port
    if port is None or not _SSH_LAUNCH_ID.fullmatch(launch_id):
        raise ValueError("invalid SSH worker launch identity")
    worker = subprocess.Popen(
        [sys.executable, "-c", _WORKER_CODE, address, launch_id],
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        stderr=sys.stderr,
        start_new_session=True,
    )

    def terminate(_signum: int, _frame: Any) -> None:
        raise SystemExit

    previous = {
        signum: signal.signal(signum, terminate)
        for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline and worker.poll() is None:
            try:
                if _owns_tcp_listener(worker.pid, port):
                    print((_SSH_READY_PREFIX + launch_id.encode()).decode(), flush=True)
                    break
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        else:
            if worker.poll() is not None:
                raise RuntimeError(
                    f"Monarch worker exited {worker.returncode} before ready"
                )
            raise TimeoutError(f"Monarch worker did not own {address} in time")
        while worker.poll() is None:
            readable, _, _ = select.select((sys.stdin,), (), (), 0.1)
            if readable and not os.read(sys.stdin.fileno(), 1):
                return
        raise RuntimeError(f"Monarch worker exited {worker.returncode}")
    finally:
        _stop_worker_process(worker)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_worker(
    address: str,
    *,
    launch_id: str | None = None,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> None:
    """Run a pinned Monarch worker on a trusted private network.

    ART's trust-all mode must never be exposed to an untrusted or public network.
    """

    _prepare_child_environment(worker=True)
    if launch_id is not None:
        _run_ssh_worker_session(address, launch_id, startup_timeout_s)
        return
    # Importing ``art`` initializes enough third-party state to break Monarch's
    # spawned interpreter bootstrap. Replace this process with a clean worker.
    os.execv(sys.executable, [sys.executable, "-c", _WORKER_CODE, address])


async def attach_controller(
    worker_addresses: Sequence[str],
    *,
    name: str = "art",
    startup_timeout_s: float | None = None,
    owned_workers: Sequence[_WorkerSession] = (),
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
    initialized = asyncio.ensure_future(hosts.initialized)
    deadline = (
        None
        if startup_timeout_s is None
        else asyncio.get_running_loop().time() + startup_timeout_s
    )
    try:
        while not initialized.done():
            exited = [worker for worker in owned_workers if not worker.is_alive()]
            if exited:
                raise RuntimeError(
                    "owned Monarch worker exited during attach: "
                    + ", ".join(
                        f"{worker.address} code={worker.exitcode}" for worker in exited
                    )
                )
            timeout = 0.05
            if deadline is not None:
                timeout = min(
                    timeout, max(0.0, deadline - asyncio.get_running_loop().time())
                )
                if timeout == 0:
                    raise TimeoutError("timed out attaching to Monarch workers")
            await asyncio.wait((initialized,), timeout=timeout)
        await initialized
        exited = [worker for worker in owned_workers if not worker.is_alive()]
        if exited:
            raise RuntimeError("owned Monarch worker exited as attach completed")
    except BaseException as startup_error:
        initialized.cancel()
        try:
            await asyncio.wait_for(
                hosts.shutdown(),
                startup_timeout_s or DEFAULT_STARTUP_TIMEOUT_S,
            )
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "Monarch attach and cleanup failed",
                [startup_error, cleanup_error],
            ) from None
        raise
    return hosts


async def run_explicit_controller(
    spec: ExplicitHostBootstrap,
    program: "InstalledAsyncCallable",
    *,
    startup_timeout_s: float | None = None,
    owned_workers: Sequence[_WorkerSession] = (),
) -> Any:
    hosts = await attach_controller(
        spec.worker_addresses,
        startup_timeout_s=startup_timeout_s,
        owned_workers=owned_workers,
    )
    program_task = asyncio.ensure_future(program.resolve()(hosts))
    try:
        while not program_task.done():
            exited = [worker for worker in owned_workers if not worker.is_alive()]
            if exited:
                raise RuntimeError(
                    "owned Monarch worker exited during controller program: "
                    + ", ".join(
                        f"{worker.address} code={worker.exitcode}" for worker in exited
                    )
                )
            await asyncio.wait((program_task,), timeout=0.05)
        result = await program_task
        if any(not worker.is_alive() for worker in owned_workers):
            raise RuntimeError("owned Monarch worker exited as program completed")
    except BaseException as program_error:
        if not program_task.done():
            program_task.cancel()
            await asyncio.gather(program_task, return_exceptions=True)
        try:
            await asyncio.wait_for(
                hosts.shutdown(),
                startup_timeout_s or DEFAULT_STARTUP_TIMEOUT_S,
            )
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "Monarch program and controller cleanup failed",
                [program_error, cleanup_error],
            ) from None
        raise
    await asyncio.wait_for(
        hosts.shutdown(),
        startup_timeout_s or DEFAULT_STARTUP_TIMEOUT_S,
    )
    return result


def _require_bindable_worker_address(address: str) -> None:
    parsed = urlsplit(address)
    assert parsed.hostname is not None and parsed.port is not None
    error: OSError | None = None
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
        parsed.hostname, parsed.port, type=socket.SOCK_STREAM
    ):
        probe = socket.socket(family, socktype, proto)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(sockaddr)
            return
        except OSError as exc:
            error = exc
        finally:
            probe.close()
    raise RuntimeError(
        f"Monarch worker address is already in use: {address}"
    ) from error


def _resolve_ephemeral_worker_address(address: str) -> str:
    parsed = urlsplit(address)
    if parsed.port != 0:
        return address
    assert parsed.hostname is not None
    error: OSError | None = None
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
        parsed.hostname, 0, type=socket.SOCK_STREAM
    ):
        probe = socket.socket(family, socktype, proto)
        try:
            probe.bind(sockaddr)
            candidate = _tcp_address(parsed.hostname, probe.getsockname()[1])
            if candidate not in _USED_WORKER_ADDRESSES:
                return candidate
        except OSError as exc:
            error = exc
        finally:
            probe.close()
    raise RuntimeError("could not allocate a fresh local worker address") from error


def _worker_lock_path(address: str) -> Path:
    digest = hashlib.sha256(address.encode()).hexdigest()[:16]
    return _WORKER_LOCK_ROOT / f"art-monarch-worker-{digest}.lock"


def _process_identity(pid: int) -> tuple[int, int, int, str] | None:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except OSError:
        return None
    fields = stat.rsplit(")", 1)[1].split()
    return int(fields[19]), int(fields[1]), int(fields[3]), fields[0]


def _process_command(pid: int) -> tuple[str, ...] | None:
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return tuple(os.fsdecode(value) for value in command.rstrip(b"\0").split(b"\0"))


def _write_owned_worker_metadata(
    lease: Any,
    address: str,
    process: subprocess.Popen[bytes],
    ownership_token: str,
) -> None:
    controller = _process_identity(os.getpid())
    worker = _process_identity(process.pid)
    if controller is None or worker is None:
        raise RuntimeError("owned Monarch worker process identity disappeared")
    metadata = _OwnedWorkerMetadata(
        address=address,
        controller_pid=os.getpid(),
        controller_start_time=controller[0],
        worker_pid=process.pid,
        worker_start_time=worker[0],
        python_executable=os.path.realpath(sys.executable),
        worker_code_sha256=hashlib.sha256(_WORKER_CODE.encode()).hexdigest(),
        ownership_token=ownership_token,
    )
    lease.seek(0)
    lease.truncate()
    lease.write(metadata.model_dump_json().encode())
    lease.flush()
    os.fsync(lease.fileno())


def _metadata_owned_orphan(
    metadata: _OwnedWorkerMetadata, lock_path: Path
) -> tuple[int, int] | None:
    if _worker_lock_path(
        metadata.address
    ) != lock_path or metadata.python_executable != os.path.realpath(sys.executable):
        return None
    worker = _process_identity(metadata.worker_pid)
    if worker is None or worker[0] != metadata.worker_start_time or worker[3] == "Z":
        return None
    controller = _process_identity(metadata.controller_pid)
    if controller is not None and controller[0] == metadata.controller_start_time:
        return None
    command = _process_command(metadata.worker_pid)
    if command is None or len(command) != 8:
        return None
    expected_tail = (
        metadata.address,
        "--parent-pid",
        str(metadata.controller_pid),
        "--ownership-token",
        metadata.ownership_token,
    )
    if os.path.realpath(command[0]) != metadata.python_executable:
        return None
    if (
        command[1] != "-c"
        or hashlib.sha256(command[2].encode()).hexdigest()
        != metadata.worker_code_sha256
        or command[3:] != expected_tail
        or worker[2] != metadata.worker_pid
    ):
        return None
    return metadata.worker_pid, metadata.worker_start_time


def _legacy_owned_orphans() -> dict[Path, tuple[int, int]]:
    matches: dict[Path, tuple[int, int]] = {}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        identity = _process_identity(pid)
        command = _process_command(pid)
        if identity is None or command is None or len(command) != 4:
            continue
        start_time, parent_pid, session_id, state = identity
        if parent_pid != 1 or session_id != pid or state == "Z":
            continue
        if os.path.realpath(command[0]) != os.path.realpath(sys.executable):
            continue
        if command[1:3] != ("-c", _LEGACY_OWNED_WORKER_CODE):
            continue
        address = command[3]
        parsed = urlsplit(address)
        try:
            loopback = (
                parsed.hostname is not None
                and ipaddress.ip_address(parsed.hostname).is_loopback
            )
        except ValueError:
            loopback = False
        if not loopback or parsed.port in (None, 0):
            continue
        lock_path = _worker_lock_path(address)
        if lock_path in matches:
            raise RuntimeError(f"multiple legacy workers match owned lease {lock_path}")
        matches[lock_path] = (pid, start_time)
    return matches


def _terminate_owned_orphan(pid: int, start_time: int) -> None:
    try:
        pidfd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        identity = _process_identity(pid)
        if identity is None or identity[0] != start_time or identity[3] == "Z":
            return
        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        exited = select.poll()
        exited.register(pidfd, select.POLLIN)
        if not exited.poll(5000):
            raise RuntimeError(f"owned Monarch worker {pid} did not exit")
    finally:
        os.close(pidfd)


def _reconcile_orphaned_workers() -> None:
    legacy_owned: dict[Path, tuple[int, int]] | None = None
    for lock_path in sorted(_WORKER_LOCK_ROOT.glob("art-monarch-worker-*.lock")):
        try:
            lease = open(lock_path, "r+b")
        except FileNotFoundError:
            continue
        try:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            payload = lease.read().strip()
            owned: tuple[int, int] | None
            if payload:
                try:
                    metadata = _OwnedWorkerMetadata.model_validate_json(payload)
                except ValueError:
                    continue
                if _worker_lock_path(
                    metadata.address
                ) != lock_path or metadata.python_executable != os.path.realpath(
                    sys.executable
                ):
                    continue
                identity = _process_identity(metadata.worker_pid)
                if (
                    identity is None
                    or identity[0] != metadata.worker_start_time
                    or identity[3] == "Z"
                ):
                    lock_path.unlink(missing_ok=True)
                    continue
                owned = _metadata_owned_orphan(metadata, lock_path)
            else:
                if legacy_owned is None:
                    legacy_owned = _legacy_owned_orphans()
                owned = legacy_owned.get(lock_path)
            if owned is None:
                continue
            _terminate_owned_orphan(*owned)
            lock_path.unlink(missing_ok=True)
        finally:
            lease.close()


def _wait_for_worker_listener(
    process: subprocess.Popen[bytes], address: str, timeout_s: float
) -> None:
    port = urlsplit(address).port
    assert port is not None
    deadline = time.monotonic() + timeout_s
    while True:
        if (exitcode := process.poll()) is not None:
            raise RuntimeError(
                f"Monarch worker exited {exitcode} before listening on {address}"
            )
        try:
            if _owns_tcp_listener(process.pid, port):
                return
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Monarch worker did not listen on {address} in time")
        time.sleep(0.05)


def _start_worker(
    address: str, *, startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S
) -> _WorkerSession:
    with _WORKER_ADDRESS_LOCK:
        _reconcile_orphaned_workers()
        address = _resolve_ephemeral_worker_address(address)
        lease = open(_worker_lock_path(address), "a+b")
        process: subprocess.Popen[bytes] | None = None
        ownership_token = uuid.uuid4().hex
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if address in _USED_WORKER_ADDRESSES:
                raise RuntimeError(
                    "Monarch 0.5 requires a fresh owned-worker address per "
                    f"generation; use port 0 instead of reusing {address}"
                )
            _require_bindable_worker_address(address)
            environment = os.environ.copy()
            _prepare_child_environment(worker=True, environ=environment)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _WORKER_CODE,
                    address,
                    "--parent-pid",
                    str(os.getpid()),
                    "--ownership-token",
                    ownership_token,
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            _wait_for_worker_listener(process, address, startup_timeout_s)
            _write_owned_worker_metadata(lease, address, process, ownership_token)
            _USED_WORKER_ADDRESSES.add(address)
            return _WorkerSession(
                address=address,
                process=process,
                label=f"Monarch worker {address}",
                lease=lease,
            )
        except BaseException:
            if process is not None:
                _stop_worker_process(process)
            lease.close()
            raise


def _stop_worker_sessions(workers: Sequence[_WorkerSession]) -> None:
    failures: list[BaseException] = []
    for worker in workers:
        try:
            worker.release()
        except BaseException as error:
            failures.append(error)
    for worker in workers:
        try:
            worker.wait()
        except BaseException as error:
            failures.append(error)
    if failures:
        raise BaseExceptionGroup("Monarch worker cleanup failed", failures)


def _stop_worker(worker: _WorkerSession) -> None:
    _stop_worker_sessions((worker,))
    with _WORKER_ADDRESS_LOCK:
        _reconcile_orphaned_workers()


def run_local(
    program: "InstalledAsyncCallable",
    *,
    port: int = DEFAULT_MONARCH_PORT,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> None:
    """Own one clean loopback worker and controller for one local program."""

    address = require_local_worker_address((_tcp_address("127.0.0.1", port),))
    worker = _start_worker(address, startup_timeout_s=startup_timeout_s)
    try:
        asyncio.run(
            run_explicit_controller(
                ExplicitHostBootstrap(worker_addresses=(worker.address,)),
                program,
                startup_timeout_s=startup_timeout_s,
                owned_workers=(worker,),
            )
        )
    except BaseException as program_error:
        try:
            _stop_worker(worker)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "local controller and worker cleanup failed",
                [program_error, cleanup_error],
            ) from None
        raise
    _stop_worker(worker)


def _lifecycle_listener(spec: SkyPilotBootstrap) -> socket.socket:
    # Task parents use this channel to leave together independently of worker exit.
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
    worker: _WorkerSession,
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
    worker: _WorkerSession,
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

    spec = SkyPilotBootstrap.from_environ(port=port)
    worker = _start_worker(
        spec.worker_addresses[spec.node_rank], startup_timeout_s=startup_timeout_s
    )
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
        from .rollout import InstalledAsyncCallable

        program = InstalledAsyncCallable(
            module=program_module,
            qualname=program_qualname,
        )
        if len(spec.node_ips) > 1:
            listener = _lifecycle_listener(spec)
            peers = _accept_sky_peers(spec, listener, worker, startup_timeout_s)
        asyncio.run(
            run_explicit_controller(
                ExplicitHostBootstrap(worker_addresses=spec.worker_addresses),
                program,
                startup_timeout_s=startup_timeout_s,
                owned_workers=(worker,),
            )
        )
        success = True
    finally:
        _notify_sky_peers(peers, success)
        if listener is not None:
            listener.close()
        _stop_worker(worker)


def _require_unused_ssh_addresses(spec: SshBootstrap) -> None:
    for host in spec.hosts:
        try:
            connection = socket.create_connection(
                (host.worker_host.strip("[]"), spec.port),
                timeout=0.2,
            )
        except OSError:
            continue
        connection.close()
        raise RuntimeError(
            "refusing to reuse a pre-existing Monarch worker listener at "
            f"{host.worker_host}:{spec.port}"
        )


def _start_ssh_workers(
    spec: SshBootstrap,
    startup_timeout_s: float,
) -> list[_WorkerSession]:
    workers: list[_WorkerSession] = []
    environment = os.environ.copy()
    environment.pop("ART_VIRTUAL_ENV", None)
    environment.pop("PYTHONPATH", None)
    try:
        for host, address in zip(spec.hosts, spec.worker_addresses, strict=True):
            launch_id = uuid.uuid4().hex
            python_path = os.environ.get(_PROGRAM_PYTHONPATH_ENV)
            command_prefix = ("env", f"PYTHONPATH={python_path}") if python_path else ()
            command = "exec " + shlex.join(
                (
                    *command_prefix,
                    spec.python_executable,
                    "-m",
                    "art.distributed.monarch_bootstrap",
                    "worker",
                    "--address",
                    address,
                    "--launch-id",
                    launch_id,
                    "--startup-timeout",
                    str(startup_timeout_s),
                )
            )
            workers.append(
                _WorkerSession(
                    address=address,
                    process=subprocess.Popen(
                        (
                            "ssh",
                            "-o",
                            "BatchMode=yes",
                            *spec.ssh_args,
                            host.target,
                            command,
                        ),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        env=environment,
                    ),
                    label=f"SSH worker {host.target!r}",
                    graceful=True,
                    launch_id=launch_id,
                )
            )
        return workers
    except BaseException as startup_error:
        try:
            _stop_ssh_workers(spec, workers)
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "SSH worker startup and cleanup failed",
                [startup_error, cleanup_error],
            ) from None
        raise


def _wait_for_ssh_workers(
    spec: SshBootstrap, workers: Sequence[_WorkerSession], timeout_s: float
) -> None:
    pending = {
        host.target: (host, worker)
        for host, worker in zip(spec.hosts, workers, strict=True)
    }
    streams = {}
    for target, (_, worker) in pending.items():
        assert worker.process.stdout is not None and worker.launch_id is not None
        streams[worker.process.stdout] = target
    ready: set[str] = set()
    deadline = time.monotonic() + timeout_s
    while pending:
        for target, (_, worker) in tuple(pending.items()):
            if (code := worker.exitcode) is not None:
                raise RuntimeError(f"SSH worker {target!r} exited {code} before ready")
        wait = max(0.0, min(0.05, deadline - time.monotonic()))
        readable, _, _ = select.select(tuple(streams), (), (), wait)
        for stream in readable:
            target = streams.pop(stream)
            _, worker = pending[target]
            assert worker.launch_id is not None
            expected = _SSH_READY_PREFIX + worker.launch_id.encode() + b"\n"
            if stream.readline() != expected:
                raise RuntimeError(
                    f"SSH worker {target!r} did not prove launch identity"
                )
            ready.add(target)
        for target in tuple(ready):
            host, _ = pending[target]
            try:
                with socket.create_connection(
                    (host.worker_host.strip("[]"), spec.port),
                    timeout=0.2,
                ):
                    pending.pop(target)
                    ready.remove(target)
            except OSError:
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for SSH workers {tuple(pending)}")


def _stop_ssh_workers(
    _spec: SshBootstrap,
    workers: Sequence[_WorkerSession],
) -> None:
    _stop_worker_sessions(workers)


@contextmanager
def _ssh_termination_signals() -> Iterator[Callable[[], None]]:
    received = False

    def terminate(signum: int, _frame: Any) -> None:
        nonlocal received
        if not received:
            received = True
            raise SystemExit(128 + signum)

    managed = (signal.SIGTERM, signal.SIGHUP)
    previous = {signum: signal.signal(signum, terminate) for signum in managed}

    def ignore() -> None:
        for signum in managed:
            signal.signal(signum, signal.SIG_IGN)

    try:
        yield ignore
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_ssh(
    spec: SshBootstrap,
    program: "InstalledAsyncCallable",
    *,
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S,
) -> None:
    """Start workers on passwordless SSH hosts and own them for one ART run."""

    with _ssh_termination_signals() as ignore_termination:
        _require_unused_ssh_addresses(spec)
        workers = _start_ssh_workers(spec, startup_timeout_s)
        try:
            _wait_for_ssh_workers(spec, workers, startup_timeout_s)
            asyncio.run(
                run_explicit_controller(
                    ExplicitHostBootstrap(worker_addresses=spec.worker_addresses),
                    program,
                    startup_timeout_s=startup_timeout_s,
                    owned_workers=workers,
                )
            )
        except BaseException as program_error:
            ignore_termination()
            try:
                _stop_ssh_workers(spec, workers)
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "SSH controller and worker cleanup failed",
                    [program_error, cleanup_error],
                ) from None
            raise
        ignore_termination()
        _stop_ssh_workers(spec, workers)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ART Monarch bootstrap (trusted private networks only)",
        epilog=(
            "ART uses Monarch trust-all transport; never expose worker addresses "
            "to a public or untrusted network."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--address", required=True)
    worker.add_argument("--launch-id")
    worker.add_argument(
        "--startup-timeout", type=_positive_float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    controller = subparsers.add_parser(
        "controller", help="attach to worker commands managed by the caller"
    )
    controller.add_argument("--worker", action="append", required=True)
    program_help = "module:qualname or path.py:qualname"
    controller.add_argument("--program", required=True, help=program_help)
    controller.add_argument(
        "--startup-timeout", type=_positive_float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    local = subparsers.add_parser("local", help="own one loopback worker")
    local.add_argument("--program", required=True, help=program_help)
    local.add_argument("--port", type=int, default=DEFAULT_MONARCH_PORT)
    local.add_argument(
        "--startup-timeout", type=_positive_float, default=DEFAULT_STARTUP_TIMEOUT_S
    )
    sky = subparsers.add_parser(
        "skypilot", help="consume the nodes in one SkyPilot task"
    )
    sky.add_argument("--program", required=True, help=program_help)
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
    ssh.add_argument("--program", required=True, help=program_help)
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
        run_worker(
            args.address,
            launch_id=args.launch_id,
            startup_timeout_s=args.startup_timeout,
        )
        return
    try:
        module, qualname = _program_reference(args.program)
    except ValueError as error:
        parser.error(str(error))
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
    if args.command == "local":
        run_local(
            program,
            port=args.port,
            startup_timeout_s=args.startup_timeout,
        )
    elif args.command == "ssh":
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
