from __future__ import annotations

import argparse
import asyncio
import json
import os
from threading import Event, Lock, Thread, get_ident
import time
import traceback
from typing import Any
import uuid

from monarch.actor import (
    Actor,
    Channel,
    Port,
    ValueMesh,
    endpoint,
    shutdown_context,
    this_host,
)


class SnapshotChannelProbeActor(Actor):
    def __init__(self) -> None:
        self._lock = Lock()
        self._states: dict[str, dict[str, Any]] = {}

    @endpoint(explicit_response_port=True)
    def background_snapshot(
        self,
        response_port: Port[dict[str, Any]],
        operation_id: str,
        event_port: Port[dict[str, Any]],
        ready_port: Port[dict[str, Any]],
    ) -> None:
        release = Event()
        state: dict[str, Any] = {
            "operation_id": operation_id,
            "actor_pid": os.getpid(),
            "endpoint_thread_id": get_ident(),
            "phase": "waiting_for_post_return_release",
            "release": release,
        }
        with self._lock:
            self._states[operation_id] = state

        def prepare() -> None:
            phase = "wait"
            try:
                release.wait()
                with self._lock:
                    state["worker_thread_id"] = get_ident()
                    state["send_started_ns"] = time.monotonic_ns()
                phase = "ready_send"
                ready_port.send(
                    {
                        "kind": "ready",
                        "operation_id": operation_id,
                        "sender": "background_thread",
                    }
                )
                with self._lock:
                    state["ready_send_returned"] = True
                phase = "event_send"
                event_port.send(
                    {
                        "kind": "mutation_fenced",
                        "operation_id": operation_id,
                        "sender": "background_thread",
                    }
                )
                with self._lock:
                    state["event_send_returned"] = True
                phase = "response_send"
                response_port.send(
                    {
                        "kind": "complete",
                        "operation_id": operation_id,
                        "sender": "background_thread",
                    }
                )
                with self._lock:
                    state["response_send_returned"] = True
                    state["phase"] = "complete"
            except BaseException as error:
                with self._lock:
                    state.update(
                        phase=f"failed_{phase}",
                        error_type=type(error).__name__,
                        error_message=str(error),
                        traceback=traceback.format_exc(),
                    )
                try:
                    response_port.exception(RuntimeError(f"{phase}: {error}"))
                    with self._lock:
                        state["exception_send_returned"] = True
                except BaseException as response_error:
                    with self._lock:
                        state["exception_send_error"] = (
                            f"{type(response_error).__name__}: {response_error}"
                        )
            finally:
                with self._lock:
                    state["worker_done"] = True

        Thread(
            target=prepare,
            name=f"snapshot-channel-probe-{operation_id}",
            daemon=True,
        ).start()

    @endpoint
    def release_after_endpoint_return(self, operation_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._states[operation_id]
            state["release_endpoint_thread_id"] = get_ident()
            state["release_endpoint_ns"] = time.monotonic_ns()
            state["phase"] = "released_after_endpoint_return"
            state["release"].set()
            return {
                "operation_id": operation_id,
                "actor_pid": os.getpid(),
                "actor_thread_id": get_ident(),
            }

    @endpoint
    def inspect(self, operation_id: str) -> dict[str, Any]:
        with self._lock:
            return {
                key: value
                for key, value in self._states[operation_id].items()
                if key != "release"
            }

    @endpoint(explicit_response_port=True)
    def actor_thread_control(
        self,
        response_port: Port[dict[str, Any]],
        operation_id: str,
        event_port: Port[dict[str, Any]],
        ready_port: Port[dict[str, Any]],
    ) -> None:
        payload = {
            "operation_id": operation_id,
            "sender": "actor_thread",
            "actor_pid": os.getpid(),
            "actor_thread_id": get_ident(),
        }
        ready_port.send({"kind": "ready", **payload})
        event_port.send({"kind": "mutation_fenced", **payload})
        response_port.send({"kind": "complete", **payload})


async def _receive(receiver: Any, timeout_s: float) -> dict[str, Any]:
    return await asyncio.wait_for(receiver.recv(), timeout_s)


async def _call(call: Any, timeout_s: float) -> Any:
    return await asyncio.wait_for(asyncio.shield(call), timeout_s)


async def _run_trial(
    actors: Any, *, mode: str, trial: int, timeout_s: float
) -> dict[str, Any]:
    operation_id = f"{mode}-{trial}-{uuid.uuid4().hex}"
    event_port, event_receiver = Channel[dict[str, Any]].open()
    ready_port, ready_receiver = Channel[dict[str, Any]].open()
    endpoint = (
        actors.background_snapshot
        if mode == "background_thread"
        else actors.actor_thread_control
    )
    rank_call = asyncio.ensure_future(
        endpoint.call(operation_id, event_port, ready_port)
    )
    await asyncio.sleep(0)
    release = None
    if mode == "background_thread":
        release = await _call(
            actors.release_after_endpoint_return.call_one(operation_id), timeout_s
        )

    async def capture(awaitable: Any) -> dict[str, Any]:
        try:
            value = await awaitable
            if isinstance(value, ValueMesh):
                value = list(value.values())
            return {"status": "received", "value": value}
        except BaseException as error:
            return {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

    ready, event, response = await asyncio.gather(
        capture(_receive(ready_receiver, timeout_s)),
        capture(_receive(event_receiver, timeout_s)),
        capture(_call(rank_call, timeout_s)),
    )
    state = None
    if mode == "background_thread":
        state = await _call(actors.inspect.call_one(operation_id), timeout_s)
    if not rank_call.done():
        rank_call.cancel()
        await asyncio.gather(rank_call, return_exceptions=True)
    return {
        "operation_id": operation_id,
        "release": release,
        "ready": ready,
        "event": event,
        "response": response,
        "state": state,
    }


def _validate(trial: dict[str, Any], mode: str) -> list[str]:
    errors = [
        f"{name}: {trial[name]}"
        for name in ("ready", "event", "response")
        if trial[name]["status"] != "received"
    ]
    if mode == "background_thread":
        state = trial["state"]
        if state.get("phase") != "complete":
            errors.append(f"worker state: {state}")
        if state.get("release_endpoint_ns", 0) > state.get("send_started_ns", -1):
            errors.append("background send began before the post-return release endpoint")
        if state.get("worker_thread_id") == state.get("endpoint_thread_id"):
            errors.append("background send unexpectedly ran on the actor endpoint thread")
    return errors


async def run(trials: int, timeout_s: float) -> dict[str, Any]:
    name = f"snapshot_channel_probe_{uuid.uuid4().hex}"
    proc_mesh = this_host().spawn_procs(
        per_host={"cpu_worker": 1}, name=f"{name}_proc"
    )
    actors = None
    results: dict[str, Any] = {}
    try:
        await proc_mesh.initialized
        actors = proc_mesh.spawn(f"{name}_actor", SnapshotChannelProbeActor)
        await actors.initialized
        for mode in ("actor_thread", "background_thread"):
            mode_trials = [
                await _run_trial(
                    actors, mode=mode, trial=trial, timeout_s=timeout_s
                )
                for trial in range(trials)
            ]
            errors = [error for trial in mode_trials for error in _validate(trial, mode)]
            results[mode] = {
                "trials": trials,
                "passed": trials - len([trial for trial in mode_trials if _validate(trial, mode)]),
                "errors": errors,
                "first": mode_trials[0],
                "last": mode_trials[-1],
            }
    finally:
        if actors is not None:
            await actors.stop()
        await proc_mesh.stop()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout_s <= 0:
        parser.error("--trials and --timeout-s must be positive")
    results = asyncio.run(run(args.trials, args.timeout_s))
    shutdown_context().get(timeout=args.timeout_s)
    print(json.dumps(results, indent=2, sort_keys=True))
    if any(result["errors"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
