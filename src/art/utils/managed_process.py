from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ART-owned child process")
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command")
    return args


def set_parent_death_signal(parent_pid: int, sig: signal.Signals) -> None:
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(sig), 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() != parent_pid:
        os._exit(1)


def main() -> None:
    args = parse_args()
    if hasattr(os, "setsid") and os.getpgrp() != os.getpid():
        os.setsid()

    process: subprocess.Popen[bytes] | None = None
    child_pgid: int | None = None
    shutting_down = False
    requested_shutdown: tuple[signal.Signals, int] | None = None

    def signal_child_group(sig: signal.Signals) -> None:
        if child_pgid is None:
            return
        try:
            os.killpg(child_pgid, sig)
        except ProcessLookupError:
            pass

    def sweep_child_group() -> None:
        signal_child_group(signal.SIGTERM)
        time.sleep(float(os.environ.get("ART_MANAGED_PROCESS_SWEEP_GRACE", 0.5)))
        signal_child_group(signal.SIGKILL)

    def shutdown(sig: signal.Signals, exit_code: int) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        signal_child_group(sig)
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_child_group(signal.SIGKILL)
                process.wait()
        sweep_child_group()
        os._exit(exit_code)

    def handle_signal(signum: int, _frame: object | None) -> None:
        nonlocal requested_shutdown
        requested_shutdown = (signal.Signals(signum), 128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    wrapper_pid = os.getpid()
    process = subprocess.Popen(
        args.command,
        start_new_session=True,
        preexec_fn=lambda: set_parent_death_signal(wrapper_pid, signal.SIGTERM),
    )
    child_pgid = process.pid

    while True:
        if requested_shutdown is not None:
            shutdown(*requested_shutdown)
        if os.getppid() != args.parent_pid:
            shutdown(signal.SIGTERM, 1)
        return_code = process.poll()
        if return_code is not None:
            sweep_child_group()
            sys.exit(return_code)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
