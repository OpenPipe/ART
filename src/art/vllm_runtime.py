import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

RUNTIME_SERVER = "art-vllm-runtime-server"
RUNTIME_PACKAGE = "art-vllm-runtime"
RUNTIME_PROTOCOL_VERSION = 1
RUNTIME_INSTALL_MARKER = "openpipe-art-vllm-runtime"


class VllmRuntimeLaunchConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_model: str
    port: int
    host: str = "127.0.0.1"
    cuda_visible_devices: str
    lora_path: str
    served_model_name: str
    rollout_weights_mode: Literal["lora", "merged"]
    engine_args: dict[str, object] = Field(default_factory=dict)
    server_args: dict[str, object] = Field(default_factory=dict)


class VllmRuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    art_package: str = "openpipe-art"
    art_version: str
    runtime_package: str = RUNTIME_PACKAGE
    runtime_version: str
    protocol_version: int = RUNTIME_PROTOCOL_VERSION
    python: str
    runtime_wheel: str
    runtime_wheel_sha256: str
    pyproject: str = "pyproject.toml"
    pyproject_sha256: str
    lockfile: str = "uv.lock"
    lockfile_sha256: str


class VllmRuntimeInstallMarker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    managed_by: str = RUNTIME_INSTALL_MARKER
    runtime_package: str = RUNTIME_PACKAGE
    runtime_version: str
    protocol_version: int = RUNTIME_PROTOCOL_VERSION
    manifest_hash: str
    runtime_wheel_sha256: str
    cache_root: str


def get_vllm_runtime_project_root() -> Path:
    override = os.environ.get("ART_VLLM_RUNTIME_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2] / "vllm_runtime"


def get_vllm_runtime_working_dir() -> Path:
    runtime_root = get_vllm_runtime_project_root()
    if runtime_root.exists():
        return runtime_root
    return Path.cwd()


def get_vllm_runtime_cache_root() -> Path:
    override = os.environ.get("ART_VLLM_RUNTIME_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "art" / "vllm_runtime"


def _bundled_runtime_dir() -> Path:
    return Path(__file__).resolve().parent / "_vllm_runtime"


def _source_runtime_bin() -> Path:
    return get_vllm_runtime_project_root() / ".venv" / "bin" / RUNTIME_SERVER


def _runtime_bin(runtime_dir: Path) -> Path:
    return runtime_dir / ".venv" / "bin" / RUNTIME_SERVER


def _runtime_python(runtime_dir: Path) -> Path:
    return runtime_dir / ".venv" / "bin" / "python"


def _runtime_dir_from_bin(runtime_bin: Path) -> Path | None:
    if (
        runtime_bin.name == RUNTIME_SERVER
        and runtime_bin.parent.name == "bin"
        and runtime_bin.parent.parent.name == ".venv"
    ):
        return runtime_bin.parent.parent.parent
    return None


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(manifest: VllmRuntimeManifest) -> str:
    payload = json.dumps(manifest.model_dump(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_bundled_manifest(bundle_dir: Path | None = None) -> VllmRuntimeManifest:
    bundle_dir = bundle_dir or _bundled_runtime_dir()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "ART vLLM runtime bundle is missing. Reinstall openpipe-art from a "
            "wheel built with scripts/build_package.py or set ART_VLLM_RUNTIME_BIN."
        )
    return VllmRuntimeManifest.model_validate_json(manifest_path.read_text())


def _run_install_command(command: list[str], *, cwd: Path | None = None) -> None:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "uv is required to install ART's managed vLLM runtime. Install uv or "
            "set ART_VLLM_RUNTIME_BIN to an existing runtime server."
        ) from exc
    if result.returncode == 0:
        return
    output = (result.stdout + result.stderr)[-4000:]
    raise RuntimeError(
        "Failed to install ART's managed vLLM runtime with command "
        f"{shlex.join(command)}.\n{output}"
    )


@contextmanager
def _runtime_install_lock(cache_root: Path):
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / ".install.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _install_marker_path(runtime_dir: Path) -> Path:
    return runtime_dir / "install.json"


def _read_install_marker(runtime_dir: Path) -> VllmRuntimeInstallMarker | None:
    marker_path = _install_marker_path(runtime_dir)
    if not marker_path.exists():
        return None
    try:
        return VllmRuntimeInstallMarker.model_validate_json(marker_path.read_text())
    except ValueError:
        return None


def _is_managed_runtime_dir(
    runtime_dir: Path,
    *,
    cache_root: Path,
    expected_hash: str | None = None,
) -> bool:
    if not runtime_dir.is_dir():
        return False
    if runtime_dir.resolve().parent != cache_root.resolve():
        return False
    if len(runtime_dir.name) != 64 or any(
        c not in "0123456789abcdef" for c in runtime_dir.name
    ):
        return False
    if expected_hash is not None and runtime_dir.name != expected_hash:
        return False
    marker = _read_install_marker(runtime_dir)
    if marker is None:
        return False
    if marker.managed_by != RUNTIME_INSTALL_MARKER:
        return False
    if marker.runtime_package != RUNTIME_PACKAGE:
        return False
    if marker.manifest_hash != runtime_dir.name:
        return False
    if marker.cache_root != str(cache_root.resolve()):
        return False
    if not (runtime_dir / ".venv" / "pyvenv.cfg").exists():
        return False
    return True


def _validate_managed_runtime(
    runtime_dir: Path,
    *,
    cache_root: Path,
    manifest: VllmRuntimeManifest,
    manifest_hash: str,
) -> Path | None:
    if not _is_managed_runtime_dir(
        runtime_dir, cache_root=cache_root, expected_hash=manifest_hash
    ):
        return None
    marker = _read_install_marker(runtime_dir)
    if marker is None:
        return None
    if marker.runtime_version != manifest.runtime_version:
        return None
    if marker.protocol_version != manifest.protocol_version:
        return None
    if marker.runtime_wheel_sha256 != manifest.runtime_wheel_sha256:
        return None
    runtime_bin = _runtime_bin(runtime_dir)
    if not _is_executable_file(runtime_bin):
        return None
    return runtime_bin


def _cleanup_old_managed_runtimes(cache_root: Path, *, keep_hash: str) -> None:
    if os.environ.get("ART_VLLM_RUNTIME_KEEP_OLD"):
        return
    if not cache_root.exists():
        return
    for child in cache_root.iterdir():
        if child.name == keep_hash:
            continue
        if not _is_managed_runtime_dir(child, cache_root=cache_root):
            continue
        shutil.rmtree(child)


def _install_managed_runtime(
    *,
    bundle_dir: Path,
    cache_root: Path,
    manifest: VllmRuntimeManifest,
    manifest_hash: str,
) -> Path:
    runtime_wheel = bundle_dir / manifest.runtime_wheel
    if _sha256_file(runtime_wheel) != manifest.runtime_wheel_sha256:
        raise RuntimeError(f"Bundled vLLM runtime wheel hash mismatch: {runtime_wheel}")

    cache_root.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{manifest_hash}.tmp-", dir=str(cache_root.resolve()))
    )
    runtime_dir = cache_root / manifest_hash
    promoted = False
    try:
        shutil.copy2(bundle_dir / manifest.pyproject, stage / "pyproject.toml")
        shutil.copy2(bundle_dir / manifest.lockfile, stage / "uv.lock")
        _run_install_command(
            [
                "uv",
                "sync",
                "--project",
                str(stage),
                "--frozen",
                "--no-install-project",
                "--no-dev",
            ]
        )
        if runtime_dir.exists():
            existing = _validate_managed_runtime(
                runtime_dir,
                cache_root=cache_root,
                manifest=manifest,
                manifest_hash=manifest_hash,
            )
            if existing is not None:
                shutil.rmtree(stage)
                return existing
            raise RuntimeError(
                f"Refusing to replace invalid vLLM runtime cache directory: {runtime_dir}"
            )
        stage.rename(runtime_dir)
        promoted = True
        runtime_python = _runtime_python(runtime_dir)
        _run_install_command(
            [
                "uv",
                "pip",
                "install",
                "--no-deps",
                "--python",
                str(runtime_python),
                str(runtime_wheel),
            ]
        )
        runtime_bin = _runtime_bin(runtime_dir)
        if not _is_executable_file(runtime_bin):
            raise RuntimeError(f"vLLM runtime server was not installed: {runtime_bin}")

        marker = VllmRuntimeInstallMarker(
            runtime_version=manifest.runtime_version,
            protocol_version=manifest.protocol_version,
            manifest_hash=manifest_hash,
            runtime_wheel_sha256=manifest.runtime_wheel_sha256,
            cache_root=str(cache_root.resolve()),
        )
        _install_marker_path(runtime_dir).write_text(
            json.dumps(marker.model_dump(), indent=2, sort_keys=True) + "\n"
        )
        _cleanup_old_managed_runtimes(cache_root, keep_hash=manifest_hash)
        return runtime_bin
    except Exception:
        shutil.rmtree(runtime_dir if promoted else stage, ignore_errors=True)
        raise


def ensure_vllm_runtime() -> Path:
    bundle_dir = _bundled_runtime_dir()
    manifest = _load_bundled_manifest(bundle_dir)
    manifest_hash = _manifest_hash(manifest)
    cache_root = get_vllm_runtime_cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root.resolve()
    runtime_dir = cache_root / manifest_hash

    with _runtime_install_lock(cache_root):
        existing = _validate_managed_runtime(
            runtime_dir,
            cache_root=cache_root,
            manifest=manifest,
            manifest_hash=manifest_hash,
        )
        if existing is not None:
            _cleanup_old_managed_runtimes(cache_root, keep_hash=manifest_hash)
            return existing
        return _install_managed_runtime(
            bundle_dir=bundle_dir,
            cache_root=cache_root,
            manifest=manifest,
            manifest_hash=manifest_hash,
        )


def _runtime_python_for_nccl_discovery() -> Path:
    override = os.environ.get("ART_VLLM_RUNTIME_BIN")
    if override:
        runtime_bin = Path(shlex.split(override)[0]).expanduser().resolve()
        runtime_dir = _runtime_dir_from_bin(runtime_bin)
        if runtime_dir is None:
            raise RuntimeError(
                "Cannot infer vLLM runtime Python from ART_VLLM_RUNTIME_BIN. "
                "Merged rollout weights require ART's source or managed vLLM runtime."
            )
        return _runtime_python(runtime_dir)

    source_runtime_bin = _source_runtime_bin()
    if source_runtime_bin.exists():
        runtime_dir = _runtime_dir_from_bin(source_runtime_bin)
        assert runtime_dir is not None
        return _runtime_python(runtime_dir)

    runtime_bin = ensure_vllm_runtime()
    runtime_dir = _runtime_dir_from_bin(runtime_bin)
    assert runtime_dir is not None
    return _runtime_python(runtime_dir)


def get_vllm_runtime_nccl_so_path() -> Path:
    runtime_python = _runtime_python_for_nccl_discovery()
    script = (
        "from pathlib import Path\n"
        "import importlib.util\n"
        "spec = importlib.util.find_spec('nvidia.nccl')\n"
        "if spec is None or spec.submodule_search_locations is None:\n"
        "    raise SystemExit('vLLM runtime is missing nvidia-nccl-cu12')\n"
        "package_dir = Path(next(iter(spec.submodule_search_locations)))\n"
        "path = package_dir / 'lib' / 'libnccl.so.2'\n"
        "if not path.exists():\n"
        "    raise SystemExit(f'vLLM runtime is missing {path}')\n"
        "print(path.resolve())\n"
    )
    result = subprocess.run(
        [str(runtime_python), "-c", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        output = (result.stdout + result.stderr)[-4000:]
        raise RuntimeError(
            "Failed to discover vLLM runtime NCCL library with "
            f"{runtime_python}.\n{output}"
        )
    nccl_so_path = Path(result.stdout.strip()).resolve()
    if not nccl_so_path.exists():
        raise RuntimeError(
            f"vLLM runtime reported a missing NCCL library: {nccl_so_path}"
        )
    return nccl_so_path


def _runtime_command_prefix() -> list[str]:
    override = os.environ.get("ART_VLLM_RUNTIME_BIN")
    if override:
        return shlex.split(override)
    runtime_bin = _source_runtime_bin()
    if runtime_bin.exists():
        return [str(runtime_bin)]
    runtime_root = get_vllm_runtime_project_root()
    if (
        runtime_root.exists()
        and not (_bundled_runtime_dir() / "manifest.json").exists()
    ):
        raise RuntimeError(
            "vLLM runtime env is not built. Run `uv sync` in "
            f"{runtime_root} or set ART_VLLM_RUNTIME_BIN."
        )
    return [str(ensure_vllm_runtime())]


def build_vllm_runtime_server_cmd(config: VllmRuntimeLaunchConfig) -> list[str]:
    return [
        *_runtime_command_prefix(),
        f"--model={config.base_model}",
        f"--port={config.port}",
        f"--host={config.host}",
        f"--cuda-visible-devices={config.cuda_visible_devices}",
        f"--lora-path={config.lora_path}",
        f"--served-model-name={config.served_model_name}",
        f"--rollout-weights-mode={config.rollout_weights_mode}",
        f"--engine-args-json={json.dumps(config.engine_args)}",
        f"--server-args-json={json.dumps(config.server_args)}",
    ]


async def wait_for_vllm_runtime(
    *,
    process: subprocess.Popen[Any],
    host: str,
    port: int,
    timeout: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    url = f"http://{host}:{port}/health"
    async with httpx.AsyncClient() as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM runtime exited with code {process.returncode}"
                )
            try:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"vLLM runtime did not become ready within {math.ceil(timeout)}s"
                )
            await asyncio.sleep(0.5)
