from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
import json
import statistics
import time
from typing import Any

from art.distributed.trajectory_store import (
    TrajectoryGroupBundle,
    TrajectoryRouteSequence,
)
from art.serverless.client import _prepare_forward_submission
from art.serverless.contracts import RemoteForwardRequest, remote_request_fingerprint
from art.serverless.data_plane import encode_forward_submission, prepare_training_batch
from art.training.contracts import ForwardBackwardRequest, LossConfig, RlTrajectoryBatch

Result = tuple[int, str]
Submission = Callable[[ForwardBackwardRequest], Awaitable[Result]]


def _batch(
    *, groups: int, choices: int, prompt_tokens: int, completion_tokens: int
) -> RlTrajectoryBatch:
    bytes_per_token = 40 * 8
    bundles = []
    for group_index in range(groups):
        prompt = bytes([group_index % 251]) * (prompt_tokens * bytes_per_token)
        bundles.append(
            TrajectoryGroupBundle(
                header=f"group-{group_index}".encode(),
                records=tuple(
                    bytes([(group_index + choice) % 251]) * (32 << 10)
                    for choice in range(choices)
                ),
                route_sequences=tuple(
                    TrajectoryRouteSequence(
                        trajectory_index=choice,
                        scope="messages",
                        scope_index=0,
                        choice_index=0,
                        dtype="uint8",
                        shape=(prompt_tokens + completion_tokens, 40, 8),
                        token_ids=tuple(range(prompt_tokens))
                        + tuple(
                            prompt_tokens + choice * completion_tokens + index
                            for index in range(completion_tokens)
                        ),
                        data=(
                            prompt,
                            bytes([(group_index + choice + 1) % 251])
                            * (completion_tokens * bytes_per_token),
                        ),
                    )
                    for choice in range(choices)
                ),
            )
        )
    return RlTrajectoryBatch.from_group_bundles(
        bundles, min_source_version=0, max_source_version=0
    )


def _request(batch: RlTrajectoryBatch, index: int) -> ForwardBackwardRequest:
    return ForwardBackwardRequest(
        run_id=f"run-{index}",
        request_id=f"request-{index}",
        sequence_id=0,
        batch=batch,
        loss=LossConfig(name="cispo"),
        return_token_logprobs=False,
    )


async def _old_submission(request: ForwardBackwardRequest) -> Result:
    prepared = await asyncio.to_thread(prepare_training_batch, request.batch)
    remote = RemoteForwardRequest.from_command(request, prepared.remote)
    encoded = await asyncio.to_thread(encode_forward_submission, remote, prepared)
    return encoded.byte_count, remote_request_fingerprint(remote)


async def _new_submission(request: ForwardBackwardRequest) -> Result:
    _, encoded, fingerprint = await asyncio.to_thread(
        _prepare_forward_submission, request, None
    )
    return encoded.byte_count, fingerprint


async def _heartbeat(done: asyncio.Event, samples: list[float]) -> None:
    deadline = time.perf_counter()
    while not done.is_set():
        deadline += 0.005
        await asyncio.sleep(max(0, deadline - time.perf_counter()))
        samples.append(max(0, time.perf_counter() - deadline))


async def _timed(
    call: Submission, requests: tuple[ForwardBackwardRequest, ...]
) -> tuple[float, list[float], tuple[Result, ...]]:
    lags: list[float] = []
    done = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat(done, lags))
    started = time.perf_counter()
    values = tuple(await asyncio.gather(*(call(request) for request in requests)))
    elapsed = time.perf_counter() - started
    done.set()
    await heartbeat
    return elapsed, lags, values


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


def _metrics(
    name: str,
    times: list[float],
    lags: list[float],
    concurrency: int,
    wire_bytes: int,
) -> dict[str, Any]:
    wall = statistics.median(times)
    return {
        "name": name,
        "concurrency": concurrency,
        "wire_mib": wire_bytes / (1 << 20),
        "wall_p50_s": wall,
        "submissions_per_s": concurrency / wall,
        "event_loop_lag_p99_ms": 1000 * _percentile(lags, 0.99) if lags else 0,
    }


async def _measure(
    requests: tuple[ForwardBackwardRequest, ...], repeats: int
) -> dict[str, object]:
    times = {"old": [], "new": []}
    lags = {"old": [], "new": []}
    expected: tuple[Result, ...] | None = None
    for repeat in range(repeats):
        order = (
            (("old", _old_submission), ("new", _new_submission))
            if repeat % 2 == 0
            else (("new", _new_submission), ("old", _old_submission))
        )
        for name, call in order:
            elapsed, samples, values = await _timed(call, requests)
            times[name].append(elapsed)
            lags[name].extend(samples)
            if expected is None:
                expected = values
            elif values != expected:
                raise RuntimeError("old and new submissions differ")
    assert expected is not None
    old = _metrics(
        "old_two_stage",
        times["old"],
        lags["old"],
        len(requests),
        expected[0][0],
    )
    new = _metrics(
        "new_single_worker",
        times["new"],
        lags["new"],
        len(requests),
        expected[0][0],
    )
    return {
        "old": old,
        "new": new,
        "wall_speedup": old["wall_p50_s"] / new["wall_p50_s"],
        "event_loop_lag_reduction": (
            old["event_loop_lag_p99_ms"] / new["event_loop_lag_p99_ms"]
            if new["event_loop_lag_p99_ms"]
            else None
        ),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=14)
    parser.add_argument("--choices", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--completion-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    batch = _batch(
        groups=args.groups,
        choices=args.choices,
        prompt_tokens=args.prompt_tokens,
        completion_tokens=args.completion_tokens,
    )
    requests = tuple(_request(batch, index) for index in range(args.concurrency))
    await _old_submission(requests[0])
    await _new_submission(requests[0])
    print(json.dumps(await _measure(requests, args.repeats), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
