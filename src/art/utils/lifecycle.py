from __future__ import annotations

import asyncio
import atexit
from collections.abc import Awaitable, Callable, Sequence
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import FrameType
from typing import Any, TypeVar

PROCESS_SHUTDOWN_TIMEOUT_SECONDS = 20.0
_PROCESS_SHUTDOWN_LEVEL_STEP = 0.1
_PROCESS_SHUTDOWN_SWEEP_GRACE_FRACTION = 0.05
_T = TypeVar("_T")


async def complete_task(
    task: asyncio.Task[_T],
) -> tuple[_T, asyncio.CancelledError | None]:
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.cancelled():
                break
            cancelled = cancelled or error
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException as error:
        if cancelled is not None:
            cancelled.add_note(f"operation also failed: {error}")
            raise cancelled
        raise
    return result, cancelled


async def complete_to_thread(
    operation: Callable[[], _T],
) -> tuple[_T, asyncio.CancelledError | None]:
    return await complete_task(asyncio.create_task(asyncio.to_thread(operation)))


def process_shutdown_timeout(level: int) -> float:
    multiplier = max(0.1, 1.0 - _PROCESS_SHUTDOWN_LEVEL_STEP * level)
    return PROCESS_SHUTDOWN_TIMEOUT_SECONDS * multiplier


def process_shutdown_sweep_grace() -> float:
    return PROCESS_SHUTDOWN_TIMEOUT_SECONDS * _PROCESS_SHUTDOWN_SWEEP_GRACE_FRACTION


async def cleanup_after_failure(
    primary: BaseException,
    cleanup: Callable[[], Awaitable[None]],
    *,
    message: str,
) -> None:
    try:
        await cleanup()
    except BaseException as cleanup_failure:
        raise BaseExceptionGroup(message, [primary, cleanup_failure]) from None


def managed_process_cmd(
    command: Sequence[str], *, parent_pid: int | None = None
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().with_name("managed_process.py")),
        "--parent-pid",
        str(parent_pid or os.getpid()),
        "--child-timeout",
        str(process_shutdown_timeout(2)),
        "--sweep-grace",
        str(process_shutdown_sweep_grace()),
        "--",
        *command,
    ]


def kill_process_group(pid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except ProcessLookupError:
        pass


def terminate_popen_process_group(
    process: subprocess.Popen[Any],
    *,
    timeout: float = process_shutdown_timeout(1),
) -> None:
    if process.poll() is None:
        kill_process_group(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_group(process.pid, signal.SIGKILL)
        process.wait()


def terminate_asyncio_process_group(
    process: Any, *, timeout: float = process_shutdown_timeout(1)
) -> None:
    if process.returncode is None:
        kill_process_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            finished_pid, _ = os.waitpid(process.pid, os.WNOHANG)
        except ChildProcessError:
            return
        if finished_pid:
            return
        time.sleep(0.05)
    kill_process_group(process.pid, signal.SIGKILL)
    try:
        os.waitpid(process.pid, 0)
    except ChildProcessError:
        pass


class ChildProcessSupervisor:
    def __init__(self, on_unexpected_exit: Callable[[RuntimeError], None]) -> None:
        self._on_unexpected_exit = on_unexpected_exit
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._failure: RuntimeError | None = None
        self._closing = False

    def watch_popen(
        self,
        name: str,
        process: subprocess.Popen[Any],
        *,
        log_path: str,
    ) -> None:
        self._watch(name, lambda: self._wait_popen(process), log_path=log_path)

    def watch_asyncio_process(
        self,
        name: str,
        process: Any,
        *,
        log_path: str,
    ) -> None:
        self._watch(name, process.wait, log_path=log_path)

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def close(self) -> None:
        self._closing = True
        current = self._current_task()
        for task in self._tasks.values():
            if task is not current:
                task.cancel()
        self._tasks.clear()

    def _watch(
        self,
        name: str,
        wait: Callable[[], Awaitable[int]],
        *,
        log_path: str,
    ) -> None:
        previous = self._tasks.pop(name, None)
        if previous is not None:
            previous.cancel()
        self._tasks[name] = asyncio.create_task(
            self._watch_exit(name, wait, log_path=log_path)
        )

    async def _watch_exit(
        self,
        name: str,
        wait: Callable[[], Awaitable[int]],
        *,
        log_path: str,
    ) -> None:
        try:
            returncode = await wait()
        except asyncio.CancelledError:
            return
        if self._closing:
            return
        error = RuntimeError(
            f"{name} exited with code {returncode}. Check logs at {log_path}"
        )
        self._failure = error
        self._on_unexpected_exit(error)

    async def _wait_popen(self, process: subprocess.Popen[Any]) -> int:
        return int(await asyncio.to_thread(process.wait))

    def _current_task(self) -> asyncio.Task[Any] | None:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None


class ServiceLifecycle:
    def __init__(self) -> None:
        self.closing = False
        self._close_callback: Callable[[], None] | None = None
        self._previous_signal_handlers: dict[
            int, Callable[[int, FrameType | None], object] | int | None
        ] = {}

    def begin_close(self) -> bool:
        if self.closing:
            return False
        self.closing = True
        return True

    def install_parent_cleanup(self, close: Callable[[], None]) -> None:
        if self._close_callback is not None:
            return
        self._close_callback = close
        atexit.register(close)

        def _default_signal_exit(signum: int) -> None:
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(signum)
            self._previous_signal_handlers[signum] = previous

            def _handler(
                received_signum: int,
                frame: FrameType | None,
                *,
                _previous: Callable[[int, FrameType | None], object]
                | int
                | None = previous,
            ) -> None:
                close()
                if not isinstance(_previous, int) and _previous is not None:
                    _previous(received_signum, frame)
                    return
                if _previous == signal.SIG_IGN:
                    return
                _default_signal_exit(received_signum)

            signal.signal(signum, _handler)

    def restore_parent_cleanup(self) -> None:
        if self._close_callback is not None:
            try:
                atexit.unregister(self._close_callback)
            except ValueError:
                pass
            self._close_callback = None
        for signum, previous in self._previous_signal_handlers.items():
            signal.signal(signum, previous)
        self._previous_signal_handlers.clear()
