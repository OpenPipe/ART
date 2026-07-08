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
from typing import Any, Callable, Literal, Mapping, TypedDict
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .utils.lifecycle import (
    ChildProcessSupervisor,
    managed_process_cmd,
    terminate_popen_process_group,
)

RUNTIME_SERVER = "art-vllm-runtime-server"
RUNTIME_PACKAGE = "art-vllm-runtime"
RUNTIME_PROTOCOL_VERSION = 1
RUNTIME_INSTALL_MARKER = "openpipe-art-vllm-runtime"
_TILELANG_ENV_KEYS = (
    "PYTHONPATH",
    "TVM_IMPORT_PYTHON_PATH",
    "TVM_LIBRARY_PATH",
    "TL_CUTLASS_PATH",
    "TL_TEMPLATE_PATH",
    "TL_COMPOSABLE_KERNEL_PATH",
)
_TILELANG_PATH_MARKERS = ("/site-packages/tilelang/", "\\site-packages\\tilelang\\")
_FLASHINFER_WORKSPACE_ENV = "FLASHINFER_WORKSPACE_BASE"
_ART_FLASHINFER_WORKSPACE_ENV = "ART_VLLM_RUNTIME_FLASHINFER_WORKSPACE_BASE"
VLLM_RUNTIME_CLOSE_TIMEOUT = 15.0


class VllmRuntimeLaunchConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_model: str
    port: int
    host: str = "127.0.0.1"
    cuda_visible_devices: str
    lora_path: str | None = None
    served_model_name: str
    rollout_weights_mode: Literal["lora", "merged"]
    engine_args: dict[str, object] = Field(default_factory=dict)
    server_args: dict[str, object] = Field(default_factory=dict)


class ExternalVllmRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["external"]
    server_url: str
    api_key: str | None = None
    local_checkpoint_root: str | None = None
    server_checkpoint_root: str | None = None
    health_timeout_s: float = Field(default=120.0, gt=0)


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


class VllmRuntimeRequestKwargs(TypedDict, total=False):
    headers: dict[str, str]


def is_external_vllm_runtime(config: Mapping[str, Any]) -> bool:
    runtime_config = config.get("vllm_runtime")
    return (
        isinstance(runtime_config, Mapping) and runtime_config.get("mode") == "external"
    )


def get_external_vllm_runtime_config(
    config: Mapping[str, Any],
) -> ExternalVllmRuntimeConfig | None:
    runtime_config = config.get("vllm_runtime")
    if not isinstance(runtime_config, Mapping):
        return None
    if runtime_config.get("mode", "managed") != "external":
        return None
    return ExternalVllmRuntimeConfig.model_validate(runtime_config)


def normalize_vllm_server_url(server_url: str) -> str:
    normalized = server_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"External vLLM server_url must be an HTTP URL: {server_url!r}"
        )
    return normalized


def openai_base_url_from_vllm_server_url(server_url: str) -> str:
    return f"{normalize_vllm_server_url(server_url)}/v1"


def map_checkpoint_path_for_vllm(
    config: Mapping[str, Any],
    checkpoint_path: str,
) -> str:
    runtime_config = get_external_vllm_runtime_config(config)
    if runtime_config is None:
        return checkpoint_path
    local_root = runtime_config.local_checkpoint_root
    server_root = runtime_config.server_checkpoint_root
    if local_root is None and server_root is None:
        return checkpoint_path
    if not local_root or not server_root:
        raise ValueError(
            "Set both vllm_runtime.local_checkpoint_root and "
            "vllm_runtime.server_checkpoint_root, or neither."
        )
    checkpoint_abs = os.path.abspath(checkpoint_path)
    local_root_abs = os.path.abspath(local_root)
    rel_path = os.path.relpath(checkpoint_abs, local_root_abs)
    if rel_path == os.pardir or rel_path.startswith(os.pardir + os.sep):
        raise ValueError(
            f"Checkpoint path {checkpoint_path!r} is not under "
            f"vllm_runtime.local_checkpoint_root {local_root!r}"
        )
    return os.path.join(server_root, rel_path)


async def wait_for_vllm_http_runtime(
    *,
    base_url: str,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    url = f"{base_url.rstrip('/')}/health"
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(url, headers=headers, timeout=5.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"vLLM runtime did not become ready within {math.ceil(timeout)}s"
                )
            await asyncio.sleep(0.5)


def _drop_tilelang_env_paths(value: str | None) -> str | None:
    if value is None:
        return None
    kept = [
        part
        for part in value.split(os.pathsep)
        if not any(marker in part for marker in _TILELANG_PATH_MARKERS)
    ]
    return os.pathsep.join(kept) if kept else None


def _vllm_runtime_subprocess_env() -> dict[str, str]:
    """Build a child env isolated from runtime-specific JIT path leaks.

    TileLang mutates process env during import. If a vLLM runtime child inherits
    those paths, spawn workers can load two TVM FFI libraries and abort on
    duplicate global registration during model load.

    FlashInfer writes absolute source/include paths into generated build.ninja
    files. Sharing its default ~/.cache/flashinfer across source worktrees can
    make one runtime compile kernels from another runtime's venv.
    """
    env = os.environ.copy()
    for key in _TILELANG_ENV_KEYS:
        value = _drop_tilelang_env_paths(env.get(key))
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env[_FLASHINFER_WORKSPACE_ENV] = str(_vllm_runtime_flashinfer_workspace_base())
    return env


class ManagedVllmRuntime:
    def __init__(self, *, host: str = "127.0.0.1") -> None:
        self.host = host
        self.port = 0
        self.api_key: str | None = None
        self.nccl_so_path: str | None = None
        self.process: subprocess.Popen[Any] | None = None
        self.log_file: Any = None
        self.log_path: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def request_kwargs(self) -> VllmRuntimeRequestKwargs:
        if self.api_key is None:
            return {}
        return {"headers": {"Authorization": f"Bearer {self.api_key}"}}

    async def start(
        self,
        *,
        launch_config: VllmRuntimeLaunchConfig,
        output_dir: str,
        child_processes: ChildProcessSupervisor,
        install_parent_cleanup: Callable[[], None],
        cleanup_on_error: Callable[[], None] | None = None,
        timeout: float | None = None,
    ) -> tuple[str, int]:
        self.host = launch_config.host
        self.port = launch_config.port
        api_key = launch_config.server_args.get("api_key")
        self.api_key = api_key if isinstance(api_key, str) else None
        self.nccl_so_path = (
            str(get_vllm_runtime_nccl_so_path())
            if launch_config.rollout_weights_mode == "merged"
            else None
        )

        cmd = build_vllm_runtime_server_cmd(launch_config)
        install_parent_cleanup()
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "vllm-runtime.log")
        self.log_file = open(self.log_path, "w", buffering=1)
        self.process = subprocess.Popen(
            managed_process_cmd(cmd),
            cwd=str(get_vllm_runtime_working_dir()),
            env=_vllm_runtime_subprocess_env(),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            bufsize=1,
            start_new_session=True,
        )

        runtime_timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("ART_DEDICATED_VLLM_TIMEOUT", 1200))
        )
        async with httpx.AsyncClient() as client:
            try:
                await wait_for_vllm_runtime(
                    process=self.process,
                    host=self.host,
                    port=self.port,
                    timeout=runtime_timeout,
                )
            except TimeoutError as exc:
                log_path = self.log_path
                self._cleanup_after_start_error(cleanup_on_error)
                raise TimeoutError(
                    "vLLM subprocess did not become ready within "
                    f"{runtime_timeout}s. Check logs at {log_path}"
                ) from exc
            except RuntimeError as exc:
                returncode = self.process.returncode
                log_path = self.log_path
                self._cleanup_after_start_error(cleanup_on_error)
                raise RuntimeError(
                    f"vLLM subprocess exited with code {returncode}. "
                    f"Check logs at {log_path}"
                ) from exc

            try:
                response = await client.get(
                    f"{self.base_url}/v1/models",
                    **self.request_kwargs(),
                    timeout=5.0,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                log_path = self.log_path
                self._cleanup_after_start_error(cleanup_on_error)
                raise RuntimeError(
                    "vLLM passed /health but /v1/models was not reachable. "
                    f"Check logs at {log_path}"
                ) from exc

        assert self.process is not None
        assert self.log_path is not None
        child_processes.watch_popen(
            "vLLM runtime",
            self.process,
            log_path=self.log_path,
        )
        return self.host, self.port

    def close(self) -> None:
        if self.process is not None:
            terminate_popen_process_group(
                self.process,
                timeout=float(
                    os.environ.get(
                        "ART_VLLM_RUNTIME_CLOSE_TIMEOUT",
                        VLLM_RUNTIME_CLOSE_TIMEOUT,
                    )
                ),
            )
            self.process = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self.log_path = None
        self.api_key = None
        self.nccl_so_path = None
        self.port = 0

    def _cleanup_after_start_error(
        self, cleanup_on_error: Callable[[], None] | None
    ) -> None:
        if cleanup_on_error is None:
            self.close()
        else:
            cleanup_on_error()


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


def _vllm_runtime_flashinfer_workspace_base() -> Path:
    override = os.environ.get(_ART_FLASHINFER_WORKSPACE_ENV)
    if override:
        return Path(override).expanduser()
    runtime_root = get_vllm_runtime_project_root()
    if runtime_root.exists():
        return runtime_root.resolve().parent / "scratch" / "vllm_runtime_flashinfer"
    return get_vllm_runtime_cache_root().expanduser() / "flashinfer_workspace"


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
    command = [
        *_runtime_command_prefix(),
        f"--model={config.base_model}",
        f"--port={config.port}",
        f"--host={config.host}",
        f"--cuda-visible-devices={config.cuda_visible_devices}",
    ]
    if config.lora_path is not None:
        command.append(f"--lora-path={config.lora_path}")
    command.extend(
        [
            f"--served-model-name={config.served_model_name}",
            f"--rollout-weights-mode={config.rollout_weights_mode}",
            f"--engine-args-json={json.dumps(config.engine_args)}",
            f"--server-args-json={json.dumps(config.server_args)}",
        ]
    )
    return command


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
