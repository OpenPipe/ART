from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from art.distributed.host_admission import (
    RuntimeFingerprint,
    runtime_package_names,
)
from art.utils.cache_dirs import configure_model_cache_env

RUNTIME_INSTALL_MARKER = "openpipe-art-megatron-runtime"
RUNTIME_LAUNCHER = "art-megatron-python"
RUNTIME_PROTOCOL_VERSION = 1
RuntimeProfile = Literal["cuda12", "cuda13"]
RuntimeVariant = Literal["base", "hybrid_ep", "hybrid_ep_multinode"]


class MegatronRuntimeAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MegatronRuntimeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    art_package: str
    art_version: str
    runtime_package: str
    runtime_version: str
    protocol_version: int
    python: str
    pyproject: MegatronRuntimeAsset
    lockfile: MegatronRuntimeAsset
    source_archives: tuple[MegatronRuntimeAsset, ...] = ()


class MegatronRuntimeInstallMarker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    managed_by: Literal["openpipe-art-megatron-runtime"]
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: RuntimeProfile
    variant: RuntimeVariant
    cache_root: str


class MegatronRuntimeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str = Field(min_length=1)
    profile: RuntimeProfile
    variant: RuntimeVariant
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeFingerprint


def _runtime_profile() -> RuntimeProfile:
    if override := os.environ.get("ART_MEGATRON_RUNTIME_CUDA_PROFILE"):
        if override in ("cuda12", "cuda13"):
            return override
        raise ValueError(
            "ART_MEGATRON_RUNTIME_CUDA_PROFILE must be 'cuda12' or 'cuda13'"
        )
    torch_profile: RuntimeProfile | None = None
    try:
        import torch

        if str(torch.version.cuda).startswith("13."):
            torch_profile = "cuda13"
        elif str(torch.version.cuda).startswith("12."):
            torch_profile = "cuda12"
    except ImportError:
        pass
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    toolkit_profile: RuntimeProfile | None = None
    for command in ([str(cuda_home / "bin/nvcc"), "--version"], ["nvidia-smi"]):
        try:
            output = subprocess.run(
                command, capture_output=True, text=True, check=False
            ).stdout
        except FileNotFoundError:
            continue
        if "release 13." in output or "CUDA Version: 13." in output:
            toolkit_profile = "cuda13"
            break
        if "release 12." in output or "CUDA Version: 12." in output:
            toolkit_profile = "cuda12"
            break
    if torch_profile and toolkit_profile and torch_profile != toolkit_profile:
        raise RuntimeError(
            f"ART has {torch_profile} PyTorch but CUDA_HOME is {toolkit_profile}; "
            "install the matching megatron or megatron-cu130 profile"
        )
    return torch_profile or toolkit_profile or "cuda12"


def _bundled_runtime_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "_megatron_runtime"


def _source_runtime_python() -> Path:
    root = Path(__file__).resolve().parents[4]
    return root / "megatron_runtime" / ".venv" / "bin" / "python"


def _runtime_cache_root() -> Path:
    if override := os.environ.get("ART_MEGATRON_RUNTIME_CACHE_DIR"):
        return Path(override).expanduser()
    return configure_model_cache_env(os.environ.copy()) / "megatron_runtime"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(bundle: Path) -> MegatronRuntimeManifest:
    path = bundle / "manifest.json"
    if not path.is_file():
        raise RuntimeError(
            "ART's Megatron runtime bundle is missing; install a release wheel built "
            "with scripts/build_package.py"
        )
    manifest = MegatronRuntimeManifest.model_validate_json(path.read_text())
    if manifest.protocol_version != RUNTIME_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Unsupported Megatron runtime protocol {manifest.protocol_version}"
        )
    if (
        manifest.art_package != "openpipe-art"
        or metadata.version(manifest.art_package) != manifest.art_version
    ):
        raise RuntimeError("Megatron runtime bundle does not match the ART wheel")
    for asset in (manifest.pyproject, manifest.lockfile, *manifest.source_archives):
        if _sha256_file(bundle / asset.filename) != asset.sha256:
            raise RuntimeError(
                f"Bundled Megatron runtime asset is corrupt: {asset.filename}"
            )
    return manifest


def _manifest_hash(
    manifest: MegatronRuntimeManifest,
    profile: RuntimeProfile,
    variant: RuntimeVariant,
    art_build_sha256: str,
) -> str:
    payload = json.dumps(
        {
            "manifest": manifest.model_dump(mode="json"),
            "profile": profile,
            "variant": variant,
            "art_build_sha256": art_build_sha256,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stdout + result.stderr)[-8000:]
        raise RuntimeError(
            f"Megatron runtime command failed: {shlex.join(command)}\n{detail}"
        )
    return result.stdout


def _uv() -> str:
    candidates = (Path(sys.executable).parent / "uv", shutil.which("uv"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError(
        "The Megatron install profile requires uv; reinstall openpipe-art with "
        "the megatron or megatron-cu130 extra"
    )


@contextmanager
def _install_lock(cache_root: Path):
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / ".install.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _marker(runtime_dir: Path) -> MegatronRuntimeInstallMarker | None:
    try:
        return MegatronRuntimeInstallMarker.model_validate_json(
            (runtime_dir / "install.json").read_text()
        )
    except (OSError, ValueError):
        return None


def _runtime_python(runtime_dir: Path) -> Path:
    return runtime_dir / ".venv" / "bin" / "python"


def _runtime_launcher(runtime_dir: Path) -> Path:
    return runtime_dir / ".venv" / "bin" / RUNTIME_LAUNCHER


def _write_runtime_launcher(runtime_dir: Path) -> Path:
    launcher = _runtime_launcher(runtime_dir)
    python_directory = f"python{sys.version_info.major}.{sys.version_info.minor}"
    launcher.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'runtime_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)\n'
        f'site_packages="$runtime_root/lib/{python_directory}/site-packages"\n'
        "library_path=\n"
        'for directory in "$site_packages"/nvidia/*/lib; do\n'
        '    [ -d "$directory" ] || continue\n'
        '    library_path="${library_path}${library_path:+:}${directory}"\n'
        "done\n"
        'export LD_LIBRARY_PATH="${library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"\n'
        'if [ -n "${ART_MONARCH_PROGRAM_PYTHONPATH:-}" ]; then\n'
        '    export PYTHONPATH="$ART_MONARCH_PROGRAM_PYTHONPATH"\n'
        "else\n"
        "    unset PYTHONPATH\n"
        "fi\n"
        'exec "$runtime_root/bin/python" "$@"\n'
    )
    launcher.chmod(0o755)
    return launcher


def _valid_runtime(
    runtime_dir: Path,
    *,
    cache_root: Path,
    manifest_hash: str,
    profile: RuntimeProfile,
    variant: RuntimeVariant,
) -> Path | None:
    marker = _marker(runtime_dir)
    python = _runtime_python(runtime_dir)
    launcher = _runtime_launcher(runtime_dir)
    if (
        marker is None
        or marker.managed_by != RUNTIME_INSTALL_MARKER
        or marker.manifest_hash != manifest_hash
        or marker.profile != profile
        or marker.variant != variant
        or marker.cache_root != str(cache_root.resolve())
        or runtime_dir.resolve().parent != cache_root.resolve()
        or not os.access(python, os.X_OK)
        or not os.access(launcher, os.X_OK)
        or not (runtime_dir / ".venv" / "pyvenv.cfg").is_file()
    ):
        return None
    return launcher


def _site_packages(python: Path) -> Path:
    value = _run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ]
    ).strip()
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"Megatron runtime site-packages does not exist: {path}")
    return path


def _copy_art(runtime_python: Path) -> None:
    import art

    source = Path(art.__file__).resolve().parent
    destination = _site_packages(runtime_python)
    shutil.copytree(
        source,
        destination / "art",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_vllm_runtime"),
    )
    mp_actors = source.parent / "mp_actors"
    if mp_actors.is_dir():
        shutil.copytree(mp_actors, destination / "mp_actors")
    dist_info = tuple(source.parent.glob("openpipe_art-*.dist-info"))
    if len(dist_info) != 1:
        raise RuntimeError(
            f"Expected one installed openpipe-art dist-info, found {dist_info}"
        )
    shutil.copytree(dist_info[0], destination / dist_info[0].name)


def _install_runtime(
    bundle: Path,
    manifest: MegatronRuntimeManifest,
    profile: RuntimeProfile,
    variant: RuntimeVariant,
    cache_root: Path,
    manifest_hash: str,
) -> Path:
    stage = Path(tempfile.mkdtemp(prefix=f".{manifest_hash}.tmp-", dir=cache_root))
    runtime_dir = cache_root / manifest_hash
    promoted = False
    try:
        shutil.copy2(bundle / manifest.pyproject.filename, stage / "pyproject.toml")
        shutil.copy2(bundle / manifest.lockfile.filename, stage / "uv.lock")
        _run(
            [
                _uv(),
                "sync",
                "--project",
                str(stage),
                "--extra",
                profile,
                "--frozen",
                "--no-dev",
                "--no-install-project",
                "--python",
                sys.executable,
            ]
        )
        python = _runtime_python(stage)
        _copy_art(python)
        launcher = _write_runtime_launcher(stage)
        if variant != "base":
            _prepare_hybrid_ep(launcher, multinode=variant == "hybrid_ep_multinode")
        if runtime_dir.exists():
            if existing := _valid_runtime(
                runtime_dir,
                cache_root=cache_root,
                manifest_hash=manifest_hash,
                profile=profile,
                variant=variant,
            ):
                return existing
            raise RuntimeError(
                f"Refusing to replace invalid Megatron runtime directory: {runtime_dir}"
            )
        (stage / "install.json").write_text(
            MegatronRuntimeInstallMarker(
                managed_by=RUNTIME_INSTALL_MARKER,
                manifest_hash=manifest_hash,
                profile=profile,
                variant=variant,
                cache_root=str(cache_root.resolve()),
            ).model_dump_json(indent=2)
            + "\n"
        )
        stage.rename(runtime_dir)
        promoted = True
        return _runtime_launcher(runtime_dir)
    finally:
        if not promoted and stage.exists():
            shutil.rmtree(stage)


def _fingerprint(
    python: Path, profile: RuntimeProfile, *, hybrid_ep: bool
) -> RuntimeFingerprint:
    nixl_package = {"cuda12": "nixl-cu12", "cuda13": "nixl-cu13"}[profile]
    packages = [*runtime_package_names(trainer=True), nixl_package]
    if profile == "cuda12":
        packages.append("apex")
    if hybrid_ep:
        packages.append("art-deep-ep")
    script = (
        "import json; from art.distributed.host_admission import "
        "build_runtime_fingerprint; print(build_runtime_fingerprint("
        "json.loads(__import__('sys').argv[1])).model_dump_json())"
    )
    return RuntimeFingerprint.model_validate_json(
        _run([str(python), "-c", script, json.dumps(packages)]).strip()
    )


def _prepare_hybrid_ep(python: Path, *, multinode: bool) -> None:
    environment = os.environ.copy()
    environment.update(
        HYBRID_EP_MULTINODE="1" if multinode else "0",
        USE_NIXL="1" if multinode else "0",
    )
    result = subprocess.run(
        [str(python), "-m", "art.megatron.hybrid_ep_setup"],
        env=environment,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stdout + result.stderr)[-8000:]
        raise RuntimeError(f"HybridEP runtime preparation failed:\n{detail}")


def ensure_megatron_runtime(
    *,
    art_build_sha256: str,
    require_hybrid_ep: bool = False,
    multinode: bool = False,
) -> MegatronRuntimeInfo:
    if multinode and not require_hybrid_ep:
        raise ValueError("multi-node HybridEP requires require_hybrid_ep=True")
    profile = _runtime_profile()
    variant: RuntimeVariant = (
        "hybrid_ep_multinode"
        if multinode
        else "hybrid_ep"
        if require_hybrid_ep
        else "base"
    )
    managed = False
    if override := os.environ.get("ART_MEGATRON_RUNTIME_PYTHON"):
        python = Path(override).expanduser().resolve()
        identity = hashlib.sha256(
            f"{art_build_sha256}:{python}:{variant}".encode()
        ).hexdigest()
    else:
        bundle = _bundled_runtime_dir()
        if not (bundle / "manifest.json").is_file():
            source_python = _source_runtime_python()
            python = (
                _write_runtime_launcher(source_python.parent.parent)
                if source_python.is_file()
                else Path(sys.executable)
            )
            identity = hashlib.sha256(
                f"source:{art_build_sha256}:{python}:{platform.python_version()}:{variant}".encode()
            ).hexdigest()
        else:
            managed = True
            manifest = _load_manifest(bundle)
            identity = _manifest_hash(manifest, profile, variant, art_build_sha256)
            cache_root = _runtime_cache_root()
            runtime_dir = cache_root / identity
            with _install_lock(cache_root):
                python = _valid_runtime(
                    runtime_dir,
                    cache_root=cache_root,
                    manifest_hash=identity,
                    profile=profile,
                    variant=variant,
                ) or _install_runtime(
                    bundle,
                    manifest,
                    profile,
                    variant,
                    cache_root,
                    identity,
                )
    if not os.access(python, os.X_OK):
        raise RuntimeError(f"Megatron runtime Python is not executable: {python}")
    if require_hybrid_ep and not managed:
        _prepare_hybrid_ep(python, multinode=multinode)
    return MegatronRuntimeInfo(
        python=str(python),
        profile=profile,
        variant=variant,
        manifest_hash=identity,
        runtime=_fingerprint(python, profile, hybrid_ep=require_hybrid_ep),
    )
