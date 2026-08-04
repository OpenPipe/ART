# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import subprocess
import setuptools
import importlib
import shutil
import re

import torch
from pathlib import Path
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class CleanBuildPy(build_py):
    def run(self):
        shutil.rmtree(self.build_lib, ignore_errors=True)
        super().run()


def package_dir(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError(f"Required build package {module!r} is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def cuda_includes() -> tuple[Path, Path]:
    if torch.version.cuda and torch.version.cuda.startswith("12."):
        return (
            package_dir("nvidia.cuda_cccl") / "include",
            package_dir("nvidia.nvtx") / "include",
        )
    if torch.version.cuda and torch.version.cuda.startswith("13."):
        cuda_home = Path(os.environ["CUDA_HOME"])
        for include in [cuda_home / "include", *cuda_home.glob("targets/*/include")]:
            if (include / "cccl/cuda/ptx").is_file() and (
                include / "nvtx3/nvToolsExt.h"
            ).is_file():
                return include / "cccl", include
    raise RuntimeError(f"HybridEP cannot find headers for torch CUDA {torch.version.cuda}")


def collect_package_files(package: str, relative_dir: str):
    base_path = Path(package) / relative_dir
    if not base_path.exists():
        return []
    return [
        str(path.relative_to(package))
        for path in base_path.rglob('*')
        if path.is_file()
    ]


def to_nvcc_gencode(s: str) -> str:
    flags = []
    for part in re.split(r'[,\s;]+', s.strip()):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)\.(\d+)([A-Za-z]?)', part)
        if not m:
            raise ValueError(f"Invalid entry: {part}")
        major, minor, suf = m.groups()
        arch = f"{int(major)}{int(minor)}{suf.lower()}"
        flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    return " ".join(flags)


def get_extension_hybrid_ep_cpp():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cccl_include, nvtx_include = cuda_includes()
    enable_multinode = os.getenv("HYBRID_EP_MULTINODE", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    # NIXL is opt-in and disabled by default; the DOCA/NCCL path is the default when multinode is enabled.
    use_nixl = os.getenv("USE_NIXL", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        raise RuntimeError("Megatron setup must select TORCH_CUDA_ARCH_LIST")

    # Basic compile arguments
    compile_args = {
        "nvcc": [
            "-std=c++17",
            "-Xcompiler",
            "-fPIC",
            "--expt-relaxed-constexpr",
            "-O3",
            "--shared",
        ],
    }

    sources = [
        "csrc/hybrid_ep/hybrid_ep.cu",
        "csrc/hybrid_ep/buffer/intranode.cu",
        "csrc/hybrid_ep/allocator/allocator.cu",
        "csrc/hybrid_ep/jit/compiler.cu",
        "csrc/hybrid_ep/executor/executor.cu",
        "csrc/hybrid_ep/extension/permute.cu",
        "csrc/hybrid_ep/extension/allgather.cu",
        "csrc/hybrid_ep/pybind_hybrid_ep.cu",
    ]
    include_dirs = [
        str(cccl_include),
        str(nvtx_include),
        os.path.join(current_dir, "csrc/hybrid_ep/"),
        os.path.join(current_dir, "csrc/hybrid_ep/backend/"),
    ]
    library_dirs = []
    libraries = ["cuda"]
    extra_objects = []
    runtime_library_dirs = []
    extra_link_args = []

    # Add dependency for jit
    compile_args["nvcc"].append(f'-DSM_ARCH="{os.environ["TORCH_CUDA_ARCH_LIST"]}"')
    # Copy only the current HybridEP backend into the wheel's JIT payload.
    generated_backend = os.path.join(current_dir, "deep_ep/backend/")
    shutil.rmtree(generated_backend, ignore_errors=True)
    shutil.copytree(
        os.path.join(current_dir, "csrc/hybrid_ep/backend/"),
        generated_backend,
    )
    # Copy the utils.cuh
    shutil.copy(
        os.path.join(current_dir, "csrc/hybrid_ep/utils.cuh"),
        os.path.join(current_dir, "deep_ep/backend/utils.cuh")
    )
    # Add inter-node dependency 
    if enable_multinode:
        compile_args["nvcc"].append("-DHYBRID_EP_BUILD_MULTINODE_ENABLE")
        print(f'Multinode enabled: use_nixl={use_nixl} (USE_NIXL={os.getenv("USE_NIXL", "0")})')
        if use_nixl:
            # NIXL path: use NIXL connector instead of DOCA
            print('  -> NIXL path: skipping NCCL/DOCA build')
            compile_args["nvcc"].append("-DUSE_NIXL")
            sources.extend([
                "csrc/hybrid_ep/buffer/internode_nixl.cu",
                "csrc/hybrid_ep/buffer/nixl_connector.cu",
            ])
            nixl_home = os.getenv("NIXL_HOME", "/usr/local/nixl")
            ucx_home = os.getenv("UCX_HOME", "/usr")
            nixl_include = os.path.join(nixl_home, "include")
            nixl_gpu_include = os.path.join(nixl_home, "include/gpu/ucx")
            import platform
            machine = platform.machine()
            if machine == "aarch64":
                nixl_lib_suffix = "lib/aarch64-linux-gnu"
            else:
                nixl_lib_suffix = "lib/x86_64-linux-gnu"
            nixl_lib = os.path.join(nixl_home, nixl_lib_suffix)
            include_dirs.extend([nixl_include, nixl_gpu_include, os.path.join(ucx_home, "include")])
            library_dirs.append(nixl_lib)
            runtime_library_dirs.append(nixl_lib)
            libraries.extend(["nixl", "nixl_build", "nixl_common"])
            extra_link_args.extend([f"-Wl,-rpath,{nixl_lib}"])
            extra_link_args.append("-l:libnvidia-ml.so.1")
            libraries.extend(["mlx5", "ibverbs"])
            doca_home = os.getenv("DOCA_HOME", "")
            if doca_home:
                include_dirs.append(os.path.join(doca_home, "include"))
            rdma_core_dir = os.getenv("RDMA_CORE_HOME", "")
            if rdma_core_dir:
                include_dirs.append(os.path.join(rdma_core_dir, "include"))
                library_dirs.append(os.path.join(rdma_core_dir, "lib"))
        else:
            # DOCA path: use RDMA coordinator (requires NCCL submodule + DOCA)
            print('  -> DOCA path: building NCCL/DOCA')
            sources.extend(["csrc/hybrid_ep/buffer/internode_doca.cu"])
            rdma_core_dir = os.getenv("RDMA_CORE_HOME", "")
            nccl_dir = os.path.join(current_dir, "third-party/nccl")
            compile_args["nvcc"].append(f"-DRDMA_CORE_HOME=\"{rdma_core_dir}\"")
            extra_link_args.append("-l:libnvidia-ml.so.1")

            subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=current_dir)
            subprocess.run(
                ["make", "-j", "src.build", f"NVCC_GENCODE={to_nvcc_gencode(os.environ['TORCH_CUDA_ARCH_LIST'])}"],
                cwd=nccl_dir,
                check=True,
            )
            include_dirs.append(os.path.join(nccl_dir, "src/transport/net_ib/gdaki/doca-gpunetio/include"))
            include_dirs.append(os.path.join(rdma_core_dir, "include"))
            library_dirs.append(os.path.join(rdma_core_dir, "lib"))
            runtime_library_dirs.append(os.path.join(rdma_core_dir, "lib"))
            libraries.append("mlx5")
            libraries.append("ibverbs")
            shutil.copytree(
                os.path.join(nccl_dir, "src/transport/net_ib/gdaki/doca-gpunetio/include"),
                os.path.join(current_dir, "deep_ep/backend/nccl/include"),
                dirs_exist_ok=True
            )
            shutil.copytree(
                os.path.join(nccl_dir, "build/obj/transport/net_ib/gdaki/doca-gpunetio"),
                os.path.join(current_dir, "deep_ep/backend/nccl/obj"),
                dirs_exist_ok=True
            )
            DOCA_OBJ_PATH = os.path.join(current_dir, "deep_ep/backend/nccl/obj")
            extra_objects = [
                os.path.join(DOCA_OBJ_PATH, "doca_gpunetio.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_high_level.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_cuda_wrapper.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_device_attr.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_ibv_wrapper.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_mlx5dv_wrapper.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_qp.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_cq.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_srq.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_uar.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_verbs_umem.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_gdrcopy.o"),
                os.path.join(DOCA_OBJ_PATH, "doca_gpunetio_log.o"),
            ]


    print('Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {libraries}')
    print(f' > Library dirs: {library_dirs}')
    print(f' > Extra link args: {extra_link_args}')
    print(f' > Compilation flags: {compile_args}')
    print(f' > Extra objects: {extra_objects}')
    print(f' > Runtime library dirs: {runtime_library_dirs}')
    print(f' > Arch list: {os.environ["TORCH_CUDA_ARCH_LIST"]}')
    print()

    extension_hybrid_ep_cpp = CUDAExtension(
        "hybrid_ep_cpp",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=compile_args,
        extra_objects=extra_objects,
        runtime_library_dirs=runtime_library_dirs,
        extra_link_args=extra_link_args,
    )

    return extension_hybrid_ep_cpp

if __name__ == '__main__':
    # noinspection PyBroadException
    extension = get_extension_hybrid_ep_cpp()
    setuptools.setup(
        name='art-deep-ep',
        version=os.environ.get(
            "ART_HYBRID_EP_BUILD_VERSION", Path("VERSION").read_text().strip()
        ),
        description="ART's production HybridEP communication runtime",
        url="https://github.com/OpenPipe/art",
        license='MIT',
        python_requires='>=3.12',
        packages=setuptools.find_namespace_packages(
            include=['deep_ep', 'deep_ep.*']
        ),
        install_requires=[
            'torch==2.11.0',
        ],
        ext_modules=[extension],
        cmdclass={
            'build_ext': BuildExtension,
            'build_py': CleanBuildPy,
        },
        package_data={
            'deep_ep': collect_package_files('deep_ep', 'backend'),
        },
        include_package_data=True,
        zip_safe=False,
    )
