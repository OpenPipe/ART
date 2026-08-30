from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import selectors
import shlex
import signal
import socket
import subprocess
import sys
from threading import Lock, Thread
import time
import traceback
from typing import Any
import uuid

_PREFIX = "ART_WORKFLOW_FORKSERVER\t"
_MODULE = "integration.megatron.model_support.workflow_forkserver"
_TERMINATION_GRACE_S = 10.0


def _reply(payload: dict[str, Any]) -> None:
    print(_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _process_state() -> dict[str, Any]:
    import torch

    return {
        "pid": os.getpid(),
        "task_count": len(os.listdir("/proc/self/task")),
        "cuda_initialized": torch.cuda.is_initialized(),
        "distributed_initialized": (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ),
    }


def _assert_fork_safe() -> dict[str, Any]:
    state = _process_state()
    if (
        state["task_count"] != 1
        or state["cuda_initialized"]
        or state["distributed_initialized"]
    ):
        raise RuntimeError(f"unsafe workflow fork parent: {state}")
    return state


def _signal_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _raise_signal_exit(signum: int, _frame: Any) -> None:
    raise SystemExit(128 + signum)


def _run_child(request: dict[str, Any]) -> None:
    try:
        os.setsid()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, _raise_signal_exit)
        log_fd = os.open(
            request["log_path"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
        )
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        for name in os.listdir("/proc/self/fd"):
            fd = int(name)
            if fd > 2:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for key, value in request["environment"].items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        import torch

        from . import workflow_stage_worker

        torch.set_num_threads(int(request["torch_threads"]))
        workflow_stage_worker.run_session_json(request["request_json"])
    except BaseException:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)


def _launch_child(
    selector: selectors.BaseSelector,
    children: dict[int, dict[str, Any]],
    request: dict[str, Any],
) -> None:
    _assert_fork_safe()
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:
        _run_child(request)
    pid_fd = os.pidfd_open(pid)
    child = {
        "id": request["id"],
        "pid": pid,
        "pid_fd": pid_fd,
        "started": started,
        "deadline": started + float(request["timeout_s"]),
        "timed_out": False,
        "kill_deadline": None,
    }
    children[pid_fd] = child
    selector.register(pid_fd, selectors.EVENT_READ, "child")


def _finish_child(
    selector: selectors.BaseSelector,
    children: dict[int, dict[str, Any]],
    pid_fd: int,
) -> None:
    child = children.pop(pid_fd)
    selector.unregister(pid_fd)
    os.close(pid_fd)
    _pid, status = os.waitpid(child["pid"], 0)
    returncode = os.waitstatus_to_exitcode(status)
    if returncode != 0:
        _signal_group(child["pid"], signal.SIGTERM)
    _reply(
        {
            "id": child["id"],
            "ok": True,
            "returncode": None if child["timed_out"] else returncode,
            "actual_returncode": returncode,
            "timed_out": child["timed_out"],
            "child_wall_s": time.monotonic() - child["started"],
        }
    )


def _serve() -> None:
    started = time.monotonic()
    state = _assert_fork_safe()
    _reply(
        {
            "id": "ready",
            "ok": True,
            "preload_s": time.monotonic() - started,
            "state": state,
        }
    )
    selector = selectors.DefaultSelector()
    selector.register(sys.stdin.fileno(), selectors.EVENT_READ, "stdin")
    children: dict[int, dict[str, Any]] = {}
    buffer = b""
    stopping = False
    shutdown_id: str | None = None

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping or children:
        for key, _mask in selector.select(timeout=0.05):
            if key.data == "child":
                _finish_child(selector, children, key.fd)
                continue
            chunk = os.read(key.fd, 1 << 20)
            if not chunk:
                stopping = True
                selector.unregister(key.fd)
                continue
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                request = json.loads(raw)
                if request["command"] == "shutdown":
                    stopping = True
                    shutdown_id = request["id"]
                elif stopping:
                    _reply({"id": request["id"], "ok": False, "error": "stopping"})
                else:
                    _launch_child(selector, children, request)
        now = time.monotonic()
        for child in tuple(children.values()):
            if not child["timed_out"] and now >= child["deadline"]:
                child["timed_out"] = True
                child["kill_deadline"] = now + _TERMINATION_GRACE_S
                _signal_group(child["pid"], signal.SIGTERM)
            elif child["kill_deadline"] is not None and now >= child["kill_deadline"]:
                child["kill_deadline"] = None
                _signal_group(child["pid"], signal.SIGKILL)
        if stopping:
            for child in children.values():
                if not child["timed_out"]:
                    child["timed_out"] = True
                    child["kill_deadline"] = now + _TERMINATION_GRACE_S
                    _signal_group(child["pid"], signal.SIGTERM)
    selector.close()
    if shutdown_id is not None:
        _reply({"id": shutdown_id, "ok": True})


def _jemalloc_conf(value: str | None) -> str:
    options = [
        option
        for option in (value or "").split(",")
        if option and not option.startswith("background_thread:")
    ]
    return ",".join((*options, "background_thread:false"))


class _HostForkserver:
    def __init__(self, host: str, repo_root: Path, tests_dir: Path, log_dir: Path):
        self.host = host
        self.repo_root = repo_root
        self.tests_dir = tests_dir
        self.log_path = log_dir / f"{host.replace('/', '_')}.log"
        self.process: subprocess.Popen[str] | None = None
        self.preload_s = 0.0
        self.startup_s = 0.0
        self._pending: dict[str, Queue[dict[str, Any]]] = {}
        self._pending_lock = Lock()
        self._write_lock = Lock()
        self._reader: Thread | None = None
        self._log = None

    def start(self) -> None:
        started = time.monotonic()
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1",
                "_RJEM_MALLOC_CONF": _jemalloc_conf(
                    environment.get("_RJEM_MALLOC_CONF")
                ),
                "PYTHONPATH": os.pathsep.join(
                    filter(None, (str(self.tests_dir), environment.get("PYTHONPATH")))
                ),
                "WANDB_MODE": "disabled",
            }
        )
        command = [sys.executable, "-m", _MODULE]
        local_names = {socket.gethostname(), socket.getfqdn(), "localhost"}
        if self.host not in local_names:
            profile = Path(sys.prefix) / "art-megatron-env.sh"
            activate = (
                f"source {shlex.quote(str(profile))} && " if profile.is_file() else ""
            )
            remote = (
                activate
                + f"cd {shlex.quote(str(self.repo_root))} && exec "
                + shlex.join(
                    [
                        "env",
                        "CUDA_VISIBLE_DEVICES=",
                        "OMP_NUM_THREADS=1",
                        f"_RJEM_MALLOC_CONF={environment['_RJEM_MALLOC_CONF']}",
                        f"PYTHONPATH={environment['PYTHONPATH']}",
                        "WANDB_MODE=disabled",
                        *command,
                    ]
                )
            )
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                self.host,
                "/bin/bash",
                "--noprofile",
                "--norc",
                "-c",
                shlex.quote(remote),
            ]
        self._log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        ready = self._read_ready(timeout_s=120.0)
        self.preload_s = float(ready["preload_s"])
        self.startup_s = time.monotonic() - started
        self._reader = Thread(target=self._read_responses, daemon=True)
        self._reader.start()

    def _read_ready(self, *, timeout_s: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout.fileno(), selectors.EVENT_READ)
        deadline = time.monotonic() + timeout_s
        try:
            while self.process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(timeout=remaining):
                    raise TimeoutError(f"forkserver {self.host} did not become ready")
                line = self.process.stdout.readline()
                if line.startswith(_PREFIX):
                    result = json.loads(line.removeprefix(_PREFIX))
                    if result.get("id") == "ready" and result.get("ok"):
                        return result
                elif line and self._log is not None:
                    self._log.write(line)
            raise RuntimeError(
                f"forkserver {self.host} exited with {self.process.returncode}"
            )
        finally:
            selector.close()

    def _read_responses(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if not line.startswith(_PREFIX):
                    if self._log is not None:
                        self._log.write(line)
                    continue
                response = json.loads(line.removeprefix(_PREFIX))
                with self._pending_lock:
                    pending = self._pending.pop(response["id"], None)
                if pending is not None:
                    pending.put(response)
        except BaseException as exc:
            error = f"forkserver {self.host} protocol failed: {exc!r}"
        else:
            error = f"forkserver {self.host} closed unexpectedly"
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        for result in pending.values():
            result.put({"ok": False, "error": error})

    def _request(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdin is not None
        request_id = payload.setdefault("id", uuid.uuid4().hex)
        result: Queue[dict[str, Any]] = Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = result
        try:
            with self._write_lock:
                self.process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
                self.process.stdin.flush()
            response = result.get(timeout=timeout_s)
        except (BrokenPipeError, Empty) as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise RuntimeError(f"forkserver {self.host} request failed") from exc
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", response)))
        return response

    def run(
        self,
        *,
        request_json: Path,
        log_path: Path,
        environment: dict[str, str],
        session_environment: dict[str, str],
        torch_threads: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        child_environment: dict[str, str | None] = {
            key: environment.get(key)
            for key in ("OMP_NUM_THREADS", "_RJEM_MALLOC_CONF")
        }
        child_environment.update(
            {
                key: environment[key]
                for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "WANDB_MODE")
            }
        )
        child_environment.update(session_environment)
        return self._request(
            {
                "command": "run",
                "request_json": str(request_json),
                "log_path": str(log_path),
                "environment": child_environment,
                "torch_threads": torch_threads,
                "timeout_s": timeout_s,
            },
            timeout_s=timeout_s + _TERMINATION_GRACE_S + 30.0,
        )

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        error: Exception | None = None
        if process.poll() is None and self._reader is not None:
            try:
                self._request({"command": "shutdown"}, timeout_s=30.0)
            except Exception as exc:
                error = exc
        elif process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if self._log is not None:
            self._log.close()
        self.process = None
        if process.returncode != 0 and error is None:
            error = RuntimeError(
                f"forkserver {self.host} exited with {process.returncode}"
            )
        if error is not None:
            raise error


class WorkflowForkserverPool:
    def __init__(
        self, *, hosts: list[str], repo_root: Path, tests_dir: Path, log_dir: Path
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=False)
        clients = [
            _HostForkserver(host, repo_root, tests_dir, log_dir) for host in hosts
        ]
        try:
            with ThreadPoolExecutor(max_workers=len(clients)) as executor:
                list(executor.map(lambda client: client.start(), clients))
        except BaseException:
            for client in clients:
                try:
                    client.close()
                except Exception:
                    pass
            raise
        self._clients = {client.host: client for client in clients}

    def run(self, host: str, **kwargs: Any) -> dict[str, Any]:
        return self._clients[host].run(**kwargs)

    def metrics(self, host: str) -> dict[str, float]:
        client = self._clients[host]
        return {
            "workflow_forkserver_preload_s": client.preload_s,
            "workflow_forkserver_startup_s": client.startup_s,
        }

    def close(self) -> None:
        errors = []
        with ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
            futures = [
                executor.submit(client.close) for client in self._clients.values()
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup("workflow forkserver shutdown failed", errors)

    def __enter__(self) -> WorkflowForkserverPool:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            raise BaseExceptionGroup(
                "workflow execution and forkserver shutdown failed",
                [exc, cleanup_error],
            ) from None
        return False


if __name__ == "__main__":
    _serve()
