import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
# Transformers initializes Torch Inductor's own cache default during import.
CACHE_ENV_VARS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TRITON_CACHE_DIR",
    "VLLM_CACHE_ROOT",
    "XDG_CACHE_HOME",
)


def _run(
    command: list[str],
    *,
    artifact_dir: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    (artifact_dir / "stdout.txt").write_text(result.stdout)
    (artifact_dir / "stderr.txt").write_text(result.stderr)
    return result


def _load_json_from_stdout(stdout: str) -> dict[str, object]:
    return json.loads(stdout.strip().splitlines()[-1])


def test_art_import_does_not_require_vllm_or_set_cache_defaults(
    artifact_dir: Path,
) -> None:
    env = dict(os.environ)
    for name in CACHE_ENV_VARS:
        env.pop(name, None)
    result = _run(
        [
            sys.executable,
            "-c",
            (
                f"names = {CACHE_ENV_VARS!r}; "
                "import importlib.util, json, os; "
                "import art; "
                "print(json.dumps({"
                "'cache_env': {name: os.environ.get(name) for name in names}, "
                "'has_vllm': importlib.util.find_spec('vllm') is not None"
                "}))"
            ),
        ],
        artifact_dir=artifact_dir,
        env=env,
    )
    payload = _load_json_from_stdout(result.stdout)
    assert payload["has_vllm"] is False
    assert payload["cache_env"] == dict.fromkeys(CACHE_ENV_VARS)


def test_base_import_does_not_require_torch(artifact_dir: Path) -> None:
    script = """
import importlib.util
import builtins
import json


real_find_spec = importlib.util.find_spec
real_import = builtins.__import__


def find_spec(fullname, package=None):
    if fullname.split(".")[0] == "torch":
        return None
    return real_find_spec(fullname, package)


def import_module(name, *args, **kwargs):
    if name.split(".")[0] == "torch":
        raise ImportError(f"blocked optional dependency: {name}")
    return real_import(name, *args, **kwargs)


importlib.util.find_spec = find_spec
builtins.__import__ = import_module
import art

print(json.dumps({"imported": art.__name__}))
"""
    result = _run(
        [sys.executable, "-c", script],
        artifact_dir=artifact_dir,
    )
    assert _load_json_from_stdout(result.stdout) == {"imported": "art"}


def test_service_modules_import_without_vllm(artifact_dir: Path) -> None:
    result = _run(
        [
            sys.executable,
            "-c",
            (
                "import importlib, json; "
                "modules = ["
                "'art.megatron.distributed_service', "
                "'art.megatron.weights.conversion_tasks'"
                "]; "
                "loaded = [importlib.import_module(name).__name__ for name in modules]; "
                "print(json.dumps({'loaded': loaded}))"
            ),
        ],
        artifact_dir=artifact_dir,
    )
    payload = _load_json_from_stdout(result.stdout)
    assert payload["loaded"] == [
        "art.megatron.distributed_service",
        "art.megatron.weights.conversion_tasks",
    ]


def test_runtime_env_preserves_build_arch_without_initializing_cuda(
    artifact_dir: Path,
) -> None:
    cache_root = artifact_dir / "cache"
    env = dict(os.environ)
    env.update(
        ART_MEGATRON_CACHE_ROOT=str(cache_root),
        CUDA_VISIBLE_DEVICES="",
        TORCH_CUDA_ARCH_LIST="10.3",
        XDG_CACHE_HOME=str(cache_root),
    )
    for name in (
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        env.pop(name, None)
    result = _run(
        [
            sys.executable,
            "-c",
            (
                "import json, os, torch; "
                "from art.megatron.runtime.runtime_env import "
                "configure_megatron_runtime_env; "
                "configure_megatron_runtime_env(); "
                "print(json.dumps({'arch': os.environ['TORCH_CUDA_ARCH_LIST'], "
                "'cuda_initialized': torch.cuda.is_initialized(), "
                "'inductor': os.environ['TORCHINDUCTOR_CACHE_DIR'], "
                "'triton': os.environ['TRITON_CACHE_DIR'], "
                "'flash': os.environ['FLASH_ATTENTION_CUTE_DSL_CACHE_DIR']}))"
            ),
        ],
        artifact_dir=artifact_dir,
        env=env,
    )
    assert _load_json_from_stdout(result.stdout) == {
        "arch": "10.3",
        "cuda_initialized": False,
        "inductor": str(cache_root / "compiled" / "10.3" / "torchinductor"),
        "triton": str(cache_root / "compiled" / "10.3" / "triton"),
        "flash": str(cache_root / "compiled" / "10.3" / "flash_attention_cute_dsl"),
    }
