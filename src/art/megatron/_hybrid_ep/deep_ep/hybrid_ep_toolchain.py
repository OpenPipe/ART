from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
import os
from pathlib import Path
import subprocess

import hybrid_ep_cpp


def runtime_paths() -> tuple[str, str, str]:
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8"))
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"HybridEP requires CUDA nvcc at {nvcc}")

    cccl_include = Path(str(files("nvidia.cuda_cccl") / "include"))
    if not (cccl_include / "cuda" / "ptx").is_file():
        raise RuntimeError(f"HybridEP CCCL headers are missing from {cccl_include}")

    digest = sha256()
    digest.update(version("art-deep-ep").encode())
    digest.update(version("nvidia-cuda-cccl-cu12").encode())
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
