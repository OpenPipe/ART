from __future__ import annotations

import fcntl
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen

import torch

PACKAGE = "art-deep-ep"
SOURCE = Path(__file__).with_name("_hybrid_ep")
NATIVE_ASSETS = Path(__file__).parent / "runtime" / "native_assets.json"


def _output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _cuda_home() -> Path:
    from torch.utils.cpp_extension import CUDA_HOME

    value = os.environ.get("CUDA_HOME") or CUDA_HOME
    if not value:
        raise RuntimeError("HybridEP setup requires CUDA_HOME")
    cuda_home = Path(value).resolve()
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"HybridEP setup requires nvcc under {cuda_home}")
    return cuda_home


def _arch_list() -> str:
    if configured := os.environ.get("TORCH_CUDA_ARCH_LIST"):
        architectures = {
            value.strip() for value in configured.split(";") if value.strip()
        }
        if len(architectures) != 1:
            raise RuntimeError(
                "HybridEP requires exactly one TORCH_CUDA_ARCH_LIST value"
            )
        return architectures.pop()
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("HybridEP setup requires nvidia-smi")
    capabilities = {
        value.strip()
        for value in _output(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ).splitlines()
        if value.strip()
    }
    if len(capabilities) != 1:
        raise RuntimeError("HybridEP requires host GPUs with one compute capability")
    return capabilities.pop()


def _source_hash() -> str:
    digest = sha256()
    for path in sorted(
        path
        for path in SOURCE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ):
        digest.update(str(path.relative_to(SOURCE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _native_archives() -> dict[str, tuple[Path, str]]:
    from art.megatron.runtime.managed import _bundled_runtime_dir, _load_manifest

    bundle = _bundled_runtime_dir()
    if (bundle / "manifest.json").is_file():
        manifest = _load_manifest(bundle)
        return {
            asset.filename: (bundle / asset.filename, asset.sha256)
            for asset in manifest.source_archives
        }

    assets = json.loads(NATIVE_ASSETS.read_text())
    root = _cache_root() / "native_archives"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for filename, asset in assets.items():
            path = root / filename
            if path.is_file() and _file_hash(path) == asset["sha256"]:
                continue
            partial = path.with_suffix(path.suffix + ".partial")
            try:
                with (
                    urlopen(
                        Request(asset["url"], headers={"User-Agent": "openpipe-art"})
                    ) as response,
                    partial.open("wb") as output,
                ):
                    shutil.copyfileobj(response, output)
                if _file_hash(partial) != asset["sha256"]:
                    raise RuntimeError(f"Native asset checksum mismatch: {filename}")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
    return {
        filename: (root / filename, asset["sha256"])
        for filename, asset in assets.items()
    }


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_nixl_build_environment() -> str:
    from art.distributed.nixl_runtime import configure_nixl_environment

    paths = configure_nixl_environment()
    runtime_identity = f"{paths.module}=={version(paths.module.replace('_', '-'))}"
    headers = ("NIXL_INCLUDE_DIR", "NIXL_GPU_INCLUDE_DIR", "UCX_INCLUDE_DIR")
    if all(name in os.environ and Path(os.environ[name]).is_dir() for name in headers):
        return (
            runtime_identity
            + ":"
            + sha256(
                "\0".join(os.environ[name] for name in headers).encode()
            ).hexdigest()
        )
    nixl_home = Path(os.environ.get("NIXL_HOME", ""))
    ucx_home = Path(os.environ.get("UCX_HOME", ""))
    if (nixl_home / "include" / "nixl.h").is_file() and (
        ucx_home / "include" / "ucp" / "api" / "device" / "ucp_device_impl.h"
    ).is_file():
        os.environ.update(
            NIXL_INCLUDE_DIR=str(nixl_home / "include"),
            NIXL_GPU_INCLUDE_DIR=str(nixl_home / "include" / "gpu" / "ucx"),
            UCX_INCLUDE_DIR=str(ucx_home / "include"),
        )
        return (
            runtime_identity
            + ":"
            + sha256(f"{nixl_home}\0{ucx_home}".encode()).hexdigest()
        )

    archives = _native_archives()
    required = {"nixl-de8115ca.tar.gz", "ucx-1.21.0.tar.gz"}
    if archives.keys() != required:
        raise RuntimeError(
            f"Megatron runtime has the wrong source archives: {archives}"
        )
    identity = sha256(
        "".join(archives[name][1] for name in sorted(archives)).encode()
    ).hexdigest()
    destination = _cache_root() / "native_sources" / identity
    lock_path = destination.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.is_dir():
            stage = Path(
                tempfile.mkdtemp(prefix=f".{identity}.tmp-", dir=lock_path.parent)
            )
            try:
                for name in sorted(required):
                    with tarfile.open(archives[name][0]) as archive:
                        archive.extractall(stage, filter="data")
                stage.rename(destination)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
    nixl_source = next(destination.glob("nixl-*"), None)
    ucx_source = next(destination.glob("ucx-*"), None)
    if nixl_source is None or ucx_source is None:
        raise RuntimeError(f"Native source archives are malformed: {destination}")
    os.environ.update(
        NIXL_INCLUDE_DIR=str(nixl_source / "src" / "api" / "cpp"),
        NIXL_GPU_INCLUDE_DIR=str(nixl_source / "src" / "api" / "gpu" / "ucx"),
        UCX_INCLUDE_DIR=str(ucx_source / "src"),
        NIXL_LIBRARY_DIR=str(paths.library_dir),
        NIXL_DEPENDENCY_LIBRARY_DIR=str(paths.dependency_library_dir),
    )
    return f"{runtime_identity}:{identity}"


def _cuda_dependency_versions(cuda_home: Path) -> tuple[str, str]:
    if torch.version.cuda and torch.version.cuda.startswith("12."):
        return version("nvidia-cuda-cccl-cu12"), version("nvidia-nvtx-cu12")
    if torch.version.cuda and torch.version.cuda.startswith("13."):
        major, minor = torch.version.cuda.split(".")[:2]
        cccl = _output(
            ["dpkg-query", "-W", "-f=${Version}", f"cuda-cccl-{major}-{minor}"]
        )
        return cccl, version("nvidia-nvtx")
    raise RuntimeError(f"HybridEP does not support torch CUDA {torch.version.cuda}")


def _build_identity(
    *, enable_multinode: bool | None = None, use_nixl: bool | None = None
) -> tuple[str, str]:
    cuda_home = _cuda_home()
    arch_list = _arch_list()
    digest = sha256()
    if enable_multinode is None:
        enable_multinode = os.environ.get("HYBRID_EP_MULTINODE", "0") == "1"
    if use_nixl is None:
        use_nixl = os.environ.get("USE_NIXL", "0") == "1"
    if use_nixl and not enable_multinode:
        raise ValueError("NIXL HybridEP requires multi-node support")
    nixl_runtime_identity = None
    if use_nixl:
        from art.distributed.nixl_runtime import validate_nixl_host

        validate_nixl_host()
        nixl_runtime_identity = _prepare_nixl_build_environment()
    cccl_version, nvtx_version = _cuda_dependency_versions(cuda_home)
    values = [
        _source_hash(),
        sys.implementation.cache_tag,
        platform.machine(),
        torch.__version__,
        str(torch.version.cuda),
        cccl_version,
        nvtx_version,
        _output([str(cuda_home / "bin" / "nvcc"), "--version"]),
        _output([os.environ.get("CXX", "c++"), "--version"]),
        arch_list,
        str(int(enable_multinode)),
        str(int(use_nixl)),
    ]
    if nixl_runtime_identity:
        values.append(nixl_runtime_identity)
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    base_version = (SOURCE / "VERSION").read_text().strip()
    return f"{base_version}+art.{digest.hexdigest()[:16]}", arch_list


def _installed_version() -> str | None:
    try:
        return version(PACKAGE)
    except PackageNotFoundError:
        return None


def _cache_root() -> Path:
    root = os.environ.get("ART_MEGATRON_CACHE_ROOT")
    if root:
        return Path(root) / "hybrid_ep"
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "art"
        / "hybrid_ep"
    )


def _uv() -> str:
    candidates = (Path(sys.executable).parent / "uv", shutil.which("uv"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("HybridEP setup requires uv")


def _environment_install_lock() -> Path:
    environment = Path(sys.prefix).resolve()
    environment.mkdir(parents=True, exist_ok=True)
    return environment / ".art-hybrid-ep.install.lock"


def _build_wheel(build_version: str, arch_list: str) -> Path:
    uv = _uv()
    cache = _cache_root() / build_version
    cache.mkdir(parents=True, exist_ok=True)
    lock_path = cache.with_name(f"{cache.name}.lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        wheels = list(cache.glob("art_deep_ep-*.whl"))
        if len(wheels) == 1:
            return wheels[0]
        if wheels:
            raise RuntimeError(f"Unexpected HybridEP build artifacts: {wheels}")
        with tempfile.TemporaryDirectory(dir=cache.parent) as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            dist = root / "dist"
            shutil.copytree(SOURCE, source)
            env = os.environ.copy()
            env["ART_HYBRID_EP_BUILD_VERSION"] = build_version
            env["CUDA_HOME"] = str(_cuda_home())
            env["TORCH_CUDA_ARCH_LIST"] = arch_list
            subprocess.run(
                [
                    uv,
                    "build",
                    "--wheel",
                    "--no-build-isolation",
                    "--python",
                    sys.executable,
                    "--out-dir",
                    str(dist),
                    str(source),
                ],
                env=env,
                check=True,
            )
            built = list(dist.glob("art_deep_ep-*.whl"))
            if len(built) != 1:
                raise RuntimeError(f"Expected one HybridEP wheel, found {built}")
            return Path(shutil.move(built[0], cache / built[0].name))


def setup_hybrid_ep() -> str:
    lock_path = _environment_install_lock()
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            build_version, arch_list = _build_identity()
            if _installed_version() != build_version:
                wheel = _build_wheel(build_version, arch_list)
                subprocess.run(
                    [
                        _uv(),
                        "pip",
                        "install",
                        "--python",
                        sys.executable,
                        "--reinstall",
                        "--no-deps",
                        str(wheel),
                    ],
                    check=True,
                )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import deep_ep, hybrid_ep_cpp; "
                    f"assert hybrid_ep_cpp.SM_ARCH == {arch_list!r}",
                ],
                check=True,
            )
            return build_version
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _validate_overlay(destination: Path, *, build_version: str, arch_list: str) -> None:
    environment = os.environ.copy()
    pythonpath = environment.get("ART_MONARCH_PROGRAM_PYTHONPATH") or environment.get(
        "PYTHONPATH"
    )
    overlay_path = str(destination)
    if pythonpath:
        overlay_path += os.pathsep + pythonpath
    environment["ART_MONARCH_PROGRAM_PYTHONPATH"] = overlay_path
    environment["PYTHONPATH"] = overlay_path
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.metadata import version; "
            f"assert version({PACKAGE!r}) == {build_version!r}; "
            "import deep_ep, hybrid_ep_cpp; "
            f"assert hybrid_ep_cpp.SM_ARCH == {arch_list!r}",
        ],
        env=environment,
        check=True,
    )


def prepare_hybrid_ep_overlay() -> tuple[str, Path]:
    """Build one immutable HybridEP package view without mutating its Python env."""

    build_version, arch_list = _build_identity()
    root = _cache_root() / "overlays"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / build_version
    lock_path = root / f".{build_version}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_dir():
            _validate_overlay(
                destination, build_version=build_version, arch_list=arch_list
            )
            return build_version, destination
        if destination.exists():
            raise RuntimeError(
                f"HybridEP overlay destination is not a directory: {destination}"
            )
        wheel = _build_wheel(build_version, arch_list)
        stage = Path(tempfile.mkdtemp(prefix=f".{build_version}.tmp-", dir=root))
        try:
            subprocess.run(
                [
                    _uv(),
                    "pip",
                    "install",
                    "--target",
                    str(stage),
                    "--reinstall",
                    "--no-deps",
                    str(wheel),
                ],
                check=True,
            )
            _validate_overlay(stage, build_version=build_version, arch_list=arch_list)
            stage.rename(destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return build_version, destination


def validate_hybrid_ep(*, require_multinode: bool = False) -> None:
    candidates = [_build_identity(enable_multinode=True, use_nixl=True)[0]]
    if not require_multinode:
        candidates.append(_build_identity(enable_multinode=False, use_nixl=False)[0])
    if (installed := _installed_version()) not in candidates:
        raise RuntimeError(
            "HybridEP is not built for this ART source and Megatron environment "
            f"(expected one of {candidates}, found {installed}). Run Megatron setup."
        )


if __name__ == "__main__":
    print(f"HybridEP {setup_hybrid_ep()} is ready")
