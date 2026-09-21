import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _run(script: str) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                (str(root / "src"), os.getenv("PYTHONPATH", ""))
            ),
            "PYTHON_DOTENV_DISABLED": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_client_import_preserves_native_asyncio() -> None:
    _run(
        """
        import _asyncio
        import asyncio
        import importlib
        import multiprocessing
        import sys

        def snapshot():
            policy = asyncio.get_event_loop_policy()
            loop = asyncio.new_event_loop()
            try:
                assert not getattr(policy, "_nest_patched", False)
                assert not getattr(loop, "_nest_patched", False)
                return (
                    asyncio.Task, asyncio.tasks.Task,
                    asyncio.Future, asyncio.futures.Future,
                    asyncio.run, asyncio.get_event_loop,
                    asyncio.events.get_event_loop, type(policy).get_event_loop,
                    type(loop).run_until_complete, type(loop).run_forever,
                    type(loop)._run_once,
                    multiprocessing.get_start_method(allow_none=True),
                )
            finally:
                loop.close()

        assert asyncio.Task is _asyncio.Task
        assert asyncio.Future is _asyncio.Future
        native = snapshot()
        for name in ("art", "art.tinker", "art.tinker.client"):
            importlib.import_module(name)
            assert snapshot() == native, name
            assert not any(m == "mp_actors" or m.startswith("mp_actors.")
                           for m in sys.modules), name
            assert {"art.tinker.backend", "art.tinker.server", "art.local.backend",
                    "nest_asyncio"}.isdisjoint(sys.modules), name
        """
    )


@pytest.mark.parametrize("name", ["TinkerBackend", "OpenAICompatibleTinkerServer"])
def test_public_exports_keep_original_classes(name: str) -> None:
    _run(
        f"""
        import importlib
        import art.tinker as package

        public = ["TinkerBackend", "get_renderer_name", "OpenAICompatibleTinkerServer"]
        assert package.__all__ == public
        assert set(public) <= set(dir(package))
        try:
            package.unknown_export
        except AttributeError as error:
            assert str(error) == "module 'art.tinker' has no attribute 'unknown_export'"
        else:
            raise AssertionError("unknown export did not raise AttributeError")

        first = getattr(package, {name!r})
        namespace = {{}}
        exec("from art.tinker import *", namespace)
        from art.tinker import TinkerBackend, OpenAICompatibleTinkerServer
        for export, module in (("TinkerBackend", "backend"),
                               ("OpenAICompatibleTinkerServer", "server"),
                               ("get_renderer_name", "renderers")):
            original = getattr(importlib.import_module("art.tinker." + module), export)
            assert getattr(package, export) is original
            assert vars(package)[export] is original
            assert namespace[export] is original
        assert first is namespace[{name!r}]
        assert TinkerBackend is namespace["TinkerBackend"]
        assert OpenAICompatibleTinkerServer is namespace["OpenAICompatibleTinkerServer"]
        """
    )
