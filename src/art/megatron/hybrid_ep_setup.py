from __future__ import annotations

import fcntl
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import torch

PACKAGE = "art-deep-ep"
SOURCE = Path(__file__).with_name("_hybrid_ep")


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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "HybridEP setup requires a visible GPU or TORCH_CUDA_ARCH_LIST"
        )
    capabilities = {
        torch.cuda.get_device_capability(device)
        for device in range(torch.cuda.device_count())
    }
    if len(capabilities) != 1:
        raise RuntimeError("HybridEP requires visible GPUs with one compute capability")
    major, minor = capabilities.pop()
    return f"{major}.{minor}"


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
    if uv := shutil.which("uv"):
        return uv
    raise RuntimeError("HybridEP setup requires uv")


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
