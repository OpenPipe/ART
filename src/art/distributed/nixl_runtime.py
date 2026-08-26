from __future__ import annotations

from collections.abc import MutableMapping
from ctypes.util import find_library
import importlib.util
import os
from pathlib import Path
import platform
import shutil

from pydantic import BaseModel, ConfigDict


class NixlRuntimePaths(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str
    library_dir: Path
    dependency_library_dir: Path
    plugin_dir: Path
    ucx_module_dir: Path


def discover_nixl_runtime() -> NixlRuntimePaths:
    for module in ("nixl_cu13", "nixl_cu12", "nixl"):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.submodule_search_locations:
            continue
        site_packages = Path(next(iter(spec.submodule_search_locations))).parent
        core = site_packages / f".{module}.mesonpy.libs"
        dependencies = site_packages / f"{module}.libs"
        plugin = dependencies / "nixl"
        ucx = dependencies / "ucx"
        if not (core / "libnixl.so").is_file():
            raise RuntimeError(f"{module} is missing its bundled libnixl.so")
        if not (plugin / "libplugin_UCX.so").is_file():
            raise RuntimeError(f"{module} is missing its bundled UCX plugin")
        if not (ucx / "libuct_ib_mlx5_gda.so").is_file():
            raise RuntimeError(f"{module} is missing its UCX GDA transport")
        return NixlRuntimePaths(
            module=module,
            library_dir=core,
            dependency_library_dir=dependencies,
            plugin_dir=plugin,
            ucx_module_dir=ucx,
        )
    raise RuntimeError(
        "NIXL is unavailable; install ART with the megatron or megatron-cu130 extra"
    )


def configure_nixl_environment(
    environment: MutableMapping[str, str] | None = None,
) -> NixlRuntimePaths:
    environment = os.environ if environment is None else environment
    paths = discover_nixl_runtime()
    environment["NIXL_LIBRARY_DIR"] = str(paths.library_dir)
    environment["NIXL_DEPENDENCY_LIBRARY_DIR"] = str(paths.dependency_library_dir)
    environment["NIXL_PLUGIN_DIR"] = str(paths.plugin_dir)
    environment["UCX_MODULE_DIR"] = str(paths.ucx_module_dir)
    environment.setdefault("UCX_NET_DEVICES", "all")
    environment.setdefault("UCX_TLS", "rc,rc_gda,cuda_copy")
    environment.setdefault("UCX_IB_GDA_RETAIN_INACTIVE_CTX", "yes")
    libraries = (str(paths.library_dir), str(paths.dependency_library_dir))
    inherited = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys((*libraries, *filter(None, inherited)))
    )
    return paths


def validate_nixl_host() -> NixlRuntimePaths:
    """Fail before compilation when the image cannot support HybridEP GDA."""

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("multi-node HybridEP requires Linux x86_64")
    missing_commands = [
        command for command in ("c++", "gcc", "ninja") if shutil.which(command) is None
    ]
    if missing_commands:
        raise RuntimeError(
            f"HybridEP image is missing build tools: {', '.join(missing_commands)}"
        )
    required_files = (
        Path("/usr/include/infiniband/verbs.h"),
        Path("/dev/infiniband/rdma_cm"),
        Path("/dev/infiniband/uverbs0"),
    )
    if missing := [str(path) for path in required_files if not path.exists()]:
        raise RuntimeError(
            f"HybridEP image is missing RDMA/GDA capabilities: {missing}"
        )
    try:
        driver = Path("/proc/driver/nvidia/version").read_text()
        modules = Path("/proc/modules").read_text()
        parameters = Path("/proc/driver/nvidia/params").read_text()
    except OSError as error:
        raise RuntimeError(
            "HybridEP cannot inspect NVIDIA kernel capabilities"
        ) from error
    if "Open Kernel Module" not in driver:
        raise RuntimeError("HybridEP GDA requires the NVIDIA open kernel module")
    if not any(line.startswith("nvidia_peermem ") for line in modules.splitlines()):
        raise RuntimeError("HybridEP GDA requires loaded kernel module nvidia_peermem")
    for setting in ("EnableStreamMemOPs: 1", "PeerMappingOverride=1"):
        if setting not in parameters:
            raise RuntimeError(f"HybridEP GDA requires NVIDIA setting {setting}")
    missing_libraries = [
        library for library in ("ibverbs", "mlx5") if find_library(library) is None
    ]
    if missing_libraries:
        raise RuntimeError(
            f"HybridEP image is missing RDMA libraries: {', '.join(missing_libraries)}"
        )
    if not any(Path("/sys/class/infiniband").glob("*")):
        raise RuntimeError("HybridEP image exposes no InfiniBand device")
    return configure_nixl_environment()
