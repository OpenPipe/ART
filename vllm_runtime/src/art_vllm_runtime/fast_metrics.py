"""Process-isolated HTTP serving for ART's scalar vLLM metrics."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mmap
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
from typing import Mapping, cast
import zlib

FAST_METRIC_NAMES = (
    "prompt_tokens_total",
    "prompt_tokens_computed_total",
    "prompt_tokens_cached_total",
    "prompt_tokens_local_cache_hit_total",
    "prompt_tokens_external_kv_transfer_total",
    "generation_tokens_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "external_prefix_cache_queries_total",
    "external_prefix_cache_hits_total",
    "num_preempted_reqs_total",
    "policy_cache_salted_lora_requests_total",
    "policy_cache_unsalted_lora_requests_total",
    "policy_cache_waiting_requests_updated_total",
    "policy_cache_started_waiting_requests_skipped_total",
    "prefix_cache_hit_rate",
    "external_prefix_cache_hit_rate",
    "num_requests_running",
    "num_requests_waiting",
    "num_requests_waiting_capacity",
    "num_requests_waiting_deferred",
    "kv_cache_usage_perc",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_num_scheduled_tokens",
    "max_model_len",
    "world_size",
)

_CONTROL = struct.Struct("<Q")
_PAYLOAD = struct.Struct(f"<dQQ{len(FAST_METRIC_NAMES)}d")
_SLOT_HEADER = struct.Struct("<QI")
_SLOT_SIZE = _SLOT_HEADER.size + _PAYLOAD.size
_STATE_SIZE = _CONTROL.size + 2 * _SLOT_SIZE
_LIBC = ctypes.CDLL(None, use_errno=True)
_MEMFD_CREATE = _LIBC.memfd_create
_MEMFD_CREATE.argtypes = (ctypes.c_char_p, ctypes.c_uint)
_MEMFD_CREATE.restype = ctypes.c_int


def _memfd_create(name: str) -> int:
    function = ctypes.CDLL(None, use_errno=True).memfd_create
    function.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    fd = function(name.encode(), 1)  # MFD_CLOEXEC
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _slot_offset(sequence: int) -> int:
    return _CONTROL.size + (sequence & 1) * _SLOT_SIZE


def _memfd_create(name: str) -> int:
    fd = _MEMFD_CREATE(name.encode(), 1)  # MFD_CLOEXEC
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return fd


class FastMetricsSharedWriter:
    def __init__(self) -> None:
        self.fd = _memfd_create("art-fast-metrics")
        os.ftruncate(self.fd, _STATE_SIZE)
        self._mapping = mmap.mmap(self.fd, _STATE_SIZE)
        self._sequence = 0
        self._closed = False

    def publish(
        self,
        *,
        last_update_unix_s: float,
        record_count: int,
        engine_count: int,
        metrics: Mapping[str, float],
    ) -> None:
        values = tuple(float(metrics[name]) for name in FAST_METRIC_NAMES)
        if not math.isfinite(last_update_unix_s) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("fast metrics must be finite")
        payload = _PAYLOAD.pack(
            last_update_unix_s,
            record_count,
            engine_count,
            *values,
        )
        self._sequence += 1
        offset = _slot_offset(self._sequence)
        # Fill the inactive slot completely before publishing its sequence.
        slot = _SLOT_HEADER.pack(self._sequence, zlib.crc32(payload)) + payload
        self._mapping[offset : offset + _SLOT_SIZE] = slot
        _CONTROL.pack_into(self._mapping, 0, self._sequence)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._mapping.close()
        os.close(self.fd)


class _FastMetricsSharedReader:
    def __init__(self, fd: int) -> None:
        self._mapping = mmap.mmap(fd, _STATE_SIZE, access=mmap.ACCESS_READ)

    def read(self) -> tuple[int, tuple[float | int, ...]]:
        for _ in range(8):
            sequence = _CONTROL.unpack_from(self._mapping)[0]
            if sequence == 0:
                continue
            offset = _slot_offset(sequence)
            slot_sequence, checksum = _SLOT_HEADER.unpack_from(self._mapping, offset)
            payload = self._mapping[offset + _SLOT_HEADER.size : offset + _SLOT_SIZE]
            if slot_sequence == sequence and zlib.crc32(payload) == checksum:
                return sequence, _PAYLOAD.unpack(payload)
        raise RuntimeError("fast metrics shared snapshot changed during every read")

    def close(self) -> None:
        self._mapping.close()


class _FastMetricsHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(
        self,
        host: str,
        port: int,
        *,
        token_hashes: tuple[bytes, ...],
        reader: _FastMetricsSharedReader,
        process_uuid: str,
        generation: int,
    ) -> None:
        self._token_hashes = token_hashes
        self._reader = reader
        self._process_uuid = process_uuid
        self._generation = generation
        self._cache_lock = threading.Lock()
        self._cached_sequence = 0
        self._cached_body = b""
        super().__init__((host, port), _FastMetricsRequestHandler)

    def get_request(self) -> tuple[socket.socket, object]:
        request, address = super().get_request()
        request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return request, address

    def authorized(self, value: str | None) -> bool:
        if not self._token_hashes:
            return True
        scheme, _, token = (value or "").partition(" ")
        candidate = hashlib.sha256(token.encode()).digest()
        matches = False
        for expected in self._token_hashes:
            matches |= secrets.compare_digest(candidate, expected)
        return scheme.casefold() == "bearer" and matches

    def snapshot_body(self) -> bytes:
        sequence, values = self._reader.read()
        with self._cache_lock:
            if sequence > self._cached_sequence:
                last_update_unix_s, record_count, engine_count, *metrics = values
                content = {
                    "schema_version": 1,
                    "source": "art_vllm_runtime",
                    "last_update_unix_s": last_update_unix_s,
                    "record_count": record_count,
                    "engine_count": engine_count,
                    "metrics": dict(zip(FAST_METRIC_NAMES, metrics, strict=True)),
                    "process_uuid": self._process_uuid,
                    "generation": self._generation,
                }
                self._cached_body = json.dumps(
                    content, allow_nan=False, separators=(",", ":")
                ).encode()
                self._cached_sequence = sequence
            return self._cached_body


class _FastMetricsRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        server = cast(_FastMetricsHTTPServer, self.server)
        if self.path.partition("?")[0] != "/art/metrics":
            self._send_json(HTTPStatus.NOT_FOUND, b'{"error":"Not Found"}')
        elif not server.authorized(self.headers.get("Authorization")):
            self._send_json(HTTPStatus.UNAUTHORIZED, b'{"error":"Unauthorized"}')
        else:
            try:
                body = server.snapshot_body()
            except RuntimeError:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    b'{"error":"Metrics unavailable"}',
                )
            else:
                self._send_json(HTTPStatus.OK, body)

    def _send_json(self, status: HTTPStatus, body: bytes) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


class FastMetricsSidecar:
    def __init__(
        self,
        *,
        process: subprocess.Popen[bytes],
        writer: FastMetricsSharedWriter,
        lifetime_fd: int,
        port: int,
    ) -> None:
        self.process = process
        self.writer = writer
        self._lifetime_fd = lifetime_fd
        self.port = port
        self._closed = False

    @classmethod
    def start(
        cls,
        host: str,
        tokens: list[str],
        *,
        process_uuid: str,
        generation: int,
        port: int = 0,
        startup_timeout_s: float = 10.0,
    ) -> FastMetricsSidecar:
        writer = FastMetricsSharedWriter()
        ready_read, ready_write = os.pipe()
        lifetime_read, lifetime_write = os.pipe()
        token_hashes = [hashlib.sha256(token.encode()).hexdigest() for token in tokens]
        command = [
            sys.executable,
            "-E",
            "-S",
            str(Path(__file__).resolve()),
            "--serve",
            f"--host={host}",
            f"--port={port}",
            f"--state-fd={writer.fd}",
            f"--ready-fd={ready_write}",
            f"--lifetime-fd={lifetime_read}",
            f"--process-uuid={process_uuid}",
            f"--generation={generation}",
            *(f"--token-sha256={value}" for value in token_hashes),
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    pass_fds=(writer.fd, ready_write, lifetime_read),
                    env={"LC_ALL": "C"},
                )
            finally:
                os.close(ready_write)
                os.close(lifetime_read)
        except BaseException:
            os.close(ready_read)
            os.close(lifetime_write)
            writer.close()
            raise
        try:
            ready, _, _ = select.select([ready_read], [], [], startup_timeout_s)
            if not ready:
                raise TimeoutError("fast metrics sidecar did not become ready")
            message = os.read(ready_read, 64)
            if not message:
                returncode = None if process is None else process.poll()
                raise RuntimeError(
                    f"fast metrics sidecar exited before readiness: {returncode=}"
                )
            return cls(
                process=process,
                writer=writer,
                lifetime_fd=lifetime_write,
                port=int(message),
            )
        except BaseException:
            os.close(lifetime_write)
            writer.close()
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            raise
        finally:
            os.close(ready_read)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        os.close(self._lifetime_fd)
        try:
            returncode = self.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
            raise RuntimeError("fast metrics sidecar did not stop after parent release")
        finally:
            self.writer.close()
        # vLLM's shutdown manager SIGTERMs child processes before the API
        # server's finally block releases the sidecar lifetime pipe.
        if returncode not in {0, -signal.SIGTERM}:
            raise RuntimeError(f"fast metrics sidecar exited with status {returncode}")


def _serve(args: argparse.Namespace) -> None:
    reader = _FastMetricsSharedReader(args.state_fd)
    server = _FastMetricsHTTPServer(
        args.host,
        args.port,
        token_hashes=tuple(bytes.fromhex(value) for value in args.token_sha256),
        reader=reader,
        process_uuid=args.process_uuid,
        generation=args.generation,
    )

    try:
        os.write(args.ready_fd, str(server.server_port).encode())
        os.close(args.ready_fd)
        os.set_blocking(args.lifetime_fd, False)
        server.timeout = 0.05
        while True:
            try:
                if os.read(args.lifetime_fd, 1) == b"":
                    break
            except BlockingIOError:
                pass
            server.handle_request()
    finally:
        server.server_close()
        reader.close()
        os.close(args.lifetime_fd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-fd", type=int, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    parser.add_argument("--lifetime-fd", type=int, required=True)
    parser.add_argument("--process-uuid", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--token-sha256", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    _serve(_parse_args())
