from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid

_ROOT_ENV = "ART_VLLM_ROUTE_SHM_ROOT"
_DEFAULT_ROOT = Path("/dev/shm/art_vllm_routes")


class LocalRouteStore:
    """Process-scoped immutable route objects on a shared local shm mount."""

    def __init__(self, namespace: str) -> None:
        if not namespace or len(namespace) > 512:
            raise ValueError("local route namespace identity is invalid")
        base = Path(os.environ.get(_ROOT_ENV, str(_DEFAULT_ROOT))).resolve()
        digest = hashlib.sha256(namespace.encode()).hexdigest()
        self.root = base / digest
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def retain(self, request_identity: str, payload: bytes) -> dict[str, object]:
        if len(request_identity) != 64 or any(
            value not in "0123456789abcdef" for value in request_identity
        ):
            raise ValueError("local route request identity must be a SHA-256")
        if not payload:
            raise ValueError("local route payload must not be empty")
        sha256 = hashlib.sha256(payload).hexdigest()
        target = self.root / f"{request_identity}.routes"
        temporary = self.root / f".{request_identity}.{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("local route write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, target)
        except FileExistsError:
            self._verify(target, size_bytes=len(payload), sha256=sha256)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "store": "holder_local",
            "locator": str(target),
            "size_bytes": len(payload),
            "sha256": sha256,
        }

    def release(self, ref: dict[str, object]) -> None:
        target = self._target(ref)
        size_bytes = ref.get("size_bytes")
        sha256 = ref.get("sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or not isinstance(sha256, str)
        ):
            raise ValueError("local route object identity is invalid")
        if target.exists():
            self._verify(target, size_bytes=size_bytes, sha256=sha256)
            target.unlink()

    def discard(self, ref: dict[str, object]) -> None:
        """Remove a source object after its verified destination copy commits."""

        self._target(ref).unlink(missing_ok=True)

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _target(self, ref: dict[str, object]) -> Path:
        if ref.get("store") != "holder_local":
            raise ValueError("local route store received another object type")
        target = Path(str(ref.get("locator", ""))).resolve()
        if target.parent != self.root or target.suffix != ".routes":
            raise ValueError("local route object escaped its process namespace")
        return target

    @staticmethod
    def _verify(target: Path, *, size_bytes: int, sha256: str) -> None:
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size_bytes:
                raise RuntimeError("local route object changed size or type")
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1 << 20):
                digest.update(chunk)
            if digest.hexdigest() != sha256:
                raise RuntimeError("local route object changed digest")
        finally:
            os.close(descriptor)


def encode_route_object_header(ref: dict[str, object]) -> str:
    payload = json.dumps(ref, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")
