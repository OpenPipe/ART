from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen

from art.utils.cache_dirs import configure_model_cache_env

from .specs import EndpointSpec

ETCD_VERSION = "3.5.33"
ETCD_SHA256 = "5025b5b24d81a9616b6e284ccd439b9a3df055ef8fdcdc142af3ec8f6a3b3c95"
ETCD_URL = (
    "https://github.com/etcd-io/etcd/releases/download/"
    f"v{ETCD_VERSION}/etcd-v{ETCD_VERSION}-linux-amd64.tar.gz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_etcd() -> Path:
    root = configure_model_cache_env() / "native" / f"etcd-{ETCD_VERSION}"
    executable = root / "etcd"
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / f".{root.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if executable.is_file() and os.access(executable, os.X_OK):
            return executable
        if root.exists():
            raise RuntimeError(f"Refusing to replace invalid etcd cache: {root}")
        archive = root.parent / f"etcd-{ETCD_VERSION}.tar.gz"
        if not archive.is_file() or _sha256(archive) != ETCD_SHA256:
            with tempfile.NamedTemporaryFile(
                dir=root.parent, prefix=f".{archive.name}.", delete=False
            ) as output:
                partial = Path(output.name)
                try:
                    request = Request(ETCD_URL, headers={"User-Agent": "openpipe-art"})
                    with urlopen(request, timeout=60) as response:
                        shutil.copyfileobj(response, output)
                except BaseException:
                    partial.unlink(missing_ok=True)
                    raise
            if _sha256(partial) != ETCD_SHA256:
                partial.unlink(missing_ok=True)
                raise RuntimeError("Downloaded etcd archive failed checksum validation")
            partial.replace(archive)
        stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
        try:
            member_name = f"etcd-v{ETCD_VERSION}-linux-amd64/etcd"
            with tarfile.open(archive) as tar:
                member = tar.getmember(member_name)
                if not member.isfile() or member.size <= 0:
                    raise RuntimeError("Pinned etcd archive has an invalid executable")
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError("Pinned etcd archive is missing its executable")
                with (stage / "etcd").open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            (stage / "etcd").chmod(0o755)
            stage.rename(root)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return executable


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("", 0))
        return int(listener.getsockname()[1])


def _healthy(endpoint: EndpointSpec, timeout_s: float) -> bool:
    try:
        with urlopen(f"{endpoint.url}/health", timeout=timeout_s) as response:
            return json.loads(response.read()).get("health") in (True, "true")
    except (json.JSONDecodeError, OSError):
        return False


class ManagedEtcd:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        endpoint: EndpointSpec,
        data_dir: Path,
    ) -> None:
        self.process = process
        self.endpoint = endpoint
        self.data_dir = data_dir

    @classmethod
    def start(
        cls, *, advertise_host: str, runtime_id: str, timeout_s: float
    ) -> ManagedEtcd:
        executable = ensure_etcd()
        client_port, peer_port = _free_port(), _free_port()
        while peer_port == client_port:
            peer_port = _free_port()
        endpoint = EndpointSpec(host=advertise_host, port=client_port)
        data_dir = Path(tempfile.mkdtemp(prefix=f"art-etcd-{runtime_id[:12]}-"))
        name = f"art-{runtime_id[:24]}"
        peer_url = f"http://127.0.0.1:{peer_port}"
        try:
            with (data_dir / "etcd.log").open("wb") as log:
                process = subprocess.Popen(
                    [
                        str(executable),
                        "--name",
                        name,
                        "--data-dir",
                        str(data_dir / "data"),
                        "--listen-client-urls",
                        f"http://0.0.0.0:{client_port}",
                        "--advertise-client-urls",
                        endpoint.url,
                        "--listen-peer-urls",
                        peer_url,
                        "--initial-advertise-peer-urls",
                        peer_url,
                        "--initial-cluster",
                        f"{name}={peer_url}",
                        "--initial-cluster-state",
                        "new",
                        "--logger",
                        "zap",
                        "--log-level",
                        "warn",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except BaseException:
            shutil.rmtree(data_dir)
            raise
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = (data_dir / "etcd.log").read_text(errors="replace")[-4000:]
                shutil.rmtree(data_dir)
                raise RuntimeError(
                    f"managed etcd exited {process.returncode}:\n{detail}"
                )
            if _healthy(endpoint, min(0.2, max(0.01, deadline - time.monotonic()))):
                return cls(process, endpoint, data_dir)
            time.sleep(0.05)
        instance = cls(process, endpoint, data_dir)
        instance.close()
        raise TimeoutError(f"managed etcd did not become healthy at {endpoint.url}")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        shutil.rmtree(self.data_dir, ignore_errors=True)
