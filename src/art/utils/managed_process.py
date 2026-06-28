from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time


def _proc_stat(pid: int) -> tuple[int, int] | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    try:
        return int(fields[1]), int(fields[19])
    except (IndexError, ValueError):
        return None


def _process_snapshot() -> dict[int, tuple[int, int]]:
    snapshot: dict[int, tuple[int, int]] = {}
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return snapshot
    for entry in entries:
        if entry.name.isdecimal():
            stat = _proc_stat(int(entry.name))
            if stat is not None:
                snapshot[int(entry.name)] = stat
    return snapshot


def _descendant_processes(root_pid: int) -> dict[int, int]:
    snapshot = _process_snapshot()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _start_time) in snapshot.items():
        children.setdefault(ppid, []).append(pid)
    found: dict[int, int] = {}
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        stat = snapshot.get(pid)
        if stat is None:
            continue
        found[pid] = stat[1]
        stack.extend(children.get(pid, ()))
    return found


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
    known_child_processes: dict[int, int] = {}
    shutting_down = False
    requested_shutdown: tuple[signal.Signals, int] | None = None

    def refresh_child_processes() -> None:
        if process is not None:
            known_child_processes.update(_descendant_processes(process.pid))

    def signal_known_children(sig: signal.Signals) -> None:
        refresh_child_processes()
        for pid, start_time in sorted(known_child_processes.items(), reverse=True):
            stat = _proc_stat(pid)
            if stat is None or stat[1] != start_time:
                continue
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def signal_child_group(sig: signal.Signals) -> None:
        if child_pgid is None:
            return
        try:
            os.killpg(child_pgid, sig)
        except ProcessLookupError:
            pass

    def sweep_child_group() -> None:
        signal_known_children(signal.SIGTERM)
        signal_child_group(signal.SIGTERM)
        time.sleep(float(os.environ.get("ART_MANAGED_PROCESS_SWEEP_GRACE", 0.5)))
        signal_known_children(signal.SIGKILL)
        signal_child_group(signal.SIGKILL)

    def shutdown(sig: signal.Signals, exit_code: int) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        signal_known_children(sig)
        signal_child_group(sig)
        if process is not None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_known_children(signal.SIGKILL)
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
        refresh_child_processes()
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
