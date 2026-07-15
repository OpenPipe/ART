import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]


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


def test_art_import_does_not_require_vllm_or_mutate_compile_threads(
    artifact_dir: Path,
) -> None:
    env = dict(os.environ)
    env.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
    result = _run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, json, os; "
                "before = os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'); "
                "import art; "
                "after = os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'); "
                "print(json.dumps({"
                "'before': before, "
                "'after': after, "
                "'has_vllm': importlib.util.find_spec('vllm') is not None"
                "}))"
            ),
        ],
        artifact_dir=artifact_dir,
        env=env,
    )
    payload = _load_json_from_stdout(result.stdout)
    assert payload["has_vllm"] is False
    assert payload["before"] is None
    assert payload["after"] is None


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
                "'art.unsloth.service', "
                "'art.megatron.service', "
                "'art.megatron.weights.merged_weight_export'"
                "]; "
                "loaded = [importlib.import_module(name).__name__ for name in modules]; "
                "print(json.dumps({'loaded': loaded}))"
            ),
        ],
        artifact_dir=artifact_dir,
    )
    payload = _load_json_from_stdout(result.stdout)
    assert payload["loaded"] == [
        "art.unsloth.service",
        "art.megatron.service",
        "art.megatron.weights.merged_weight_export",
    ]
