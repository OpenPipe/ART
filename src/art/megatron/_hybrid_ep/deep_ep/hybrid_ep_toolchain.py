from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
import os
from pathlib import Path
import subprocess

import hybrid_ep_cpp
import torch


def _cuda_paths() -> tuple[Path, Path]:
    from torch.utils.cpp_extension import CUDA_HOME

    cuda_home = Path(os.environ.get("CUDA_HOME") or CUDA_HOME or "")
    if torch.version.cuda and torch.version.cuda.startswith("12."):
        return cuda_home, Path(str(files("nvidia.cuda_cccl") / "include"))
    if torch.version.cuda and torch.version.cuda.startswith("13."):
        for include in [cuda_home / "include", *cuda_home.glob("targets/*/include")]:
            if (include / "cccl/cuda/ptx").is_file():
                return cuda_home, include / "cccl"
    raise RuntimeError(f"HybridEP cannot find headers for torch CUDA {torch.version.cuda}")


def runtime_paths() -> tuple[str, str, str]:
    cuda_home, cccl_include = _cuda_paths()
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"HybridEP requires CUDA nvcc at {nvcc}")

    if not (cccl_include / "cuda" / "ptx").is_file():
        raise RuntimeError(f"HybridEP CCCL headers are missing from {cccl_include}")

    digest = sha256()
    digest.update(version("art-deep-ep").encode())
    digest.update((cccl_include / "cuda/std/__cccl/version.h").read_bytes())
    digest.update(str(hybrid_ep_cpp.SM_ARCH).encode())
    digest.update(subprocess.check_output([nvcc, "--version"]))
    digest.update(Path(hybrid_ep_cpp.__file__).read_bytes())
    backend = Path(__file__).with_name("backend")
    for path in sorted(
        (item for item in backend.rglob("*") if item.is_file()), key=str
    ):
        digest.update(str(path.relative_to(backend)).encode())
        digest.update(path.read_bytes())

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = cache_root / "art_deep_ep" / "hybrid_ep" / digest.hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cuda_home), str(cccl_include), str(cache_dir)
