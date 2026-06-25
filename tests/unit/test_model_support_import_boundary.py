from pathlib import Path
import subprocess
import sys
import textwrap


def test_get_model_config_qwen3_metadata_does_not_import_megatron() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import builtins
                import tempfile

                real_import = builtins.__import__

                def blocked_import(
                    name, globals=None, locals=None, fromlist=(), level=0
                ):
                    if level == 0 and name.partition(".")[0] == "megatron":
                        raise ModuleNotFoundError("No module named 'megatron'")
                    return real_import(name, globals, locals, fromlist, level)

                builtins.__import__ = blocked_import

                from art.dev.get_model_config import get_model_config

                with tempfile.TemporaryDirectory() as tmpdir:
                    config = get_model_config(
                        "Qwen/Qwen3-30B-A3B-Instruct-2507",
                        tmpdir,
                        None,
                    )
                """
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
    )
