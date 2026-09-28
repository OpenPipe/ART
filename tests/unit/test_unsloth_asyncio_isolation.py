from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def _run(source: str) -> None:
    root = Path(__file__).parents[2] / "src"
    # Optional trainer classes require vLLM; these checks exercise ART's actual
    # module import and queue bridge without loading any model/backend runtime.
    bootstrap = textwrap.dedent("""
        import sys
        from importlib.machinery import ModuleSpec
        from types import ModuleType
        trl = ModuleType('trl')
        trl.__spec__ = ModuleSpec('trl', loader=None)
        trl.GRPOConfig = object
        trl.GRPOTrainer = object
        sys.modules['trl'] = trl
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", bootstrap + textwrap.dedent(source)],
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_unsloth_preserves_event_loop_and_generator_cleanup() -> None:
    _run("""
        import asyncio
        import gc
        import sys

        loop = asyncio.new_event_loop()
        before = (asyncio.run, asyncio.Task, asyncio.Future,
                  type(loop).run_until_complete, type(loop).run_forever)
        import art.unsloth.train
        import art.local
        assert before == (asyncio.run, asyncio.Task, asyncio.Future,
                          type(loop).run_until_complete, type(loop).run_forever)
        closed = []
        async def values():
            try:
                yield 1
            finally:
                await asyncio.sleep(0)
                closed.append(True)
        async def consume():
            hooks = sys.get_asyncgen_hooks()
            assert hooks.firstiter is not None and hooks.finalizer is not None
            iterator = values()
            assert await anext(iterator) == 1
            del iterator
            gc.collect()
            for _ in range(4):
                await asyncio.sleep(0)
            assert closed == [True]
            assert not [t for t in asyncio.all_tasks()
                        if t is not asyncio.current_task() and not t.done()]
        try:
            loop.run_until_complete(consume())
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
    """)


def test_actual_trainer_callback_retains_nested_queue_and_error_behavior() -> None:
    _run("""
        import asyncio
        import sys
        from types import ModuleType, SimpleNamespace

        import art.unsloth.train as train
        original_run = asyncio.run

        # Only model/config/dataset construction is substituted. The installed
        # trainer callback, real queues and nested event loop execute unchanged.
        model = SimpleNamespace(peft_config={}, warnings_issued={})
        loader = SimpleNamespace(from_pretrained=lambda **kw: (model, object()))
        unsloth = ModuleType('unsloth')
        unsloth.FastLanguageModel = loader
        unsloth.FastModel = loader
        sys.modules['unsloth'] = unsloth
        train.GRPOTrainer = lambda **kw: SimpleNamespace(optimizer=object())
        train.GRPOConfig = lambda **kw: kw
        train.Dataset = SimpleNamespace(from_list=lambda rows: rows)
        train.range = lambda count: range(1) if count == 10_000_000 else range(count)
        ctx = train.create_unsloth_train_context(
            init_args={}, peft_args={}, trainer_args={})
        assert asyncio.run is original_run

        async def exercise():
            baseline = set(asyncio.all_tasks())
            payload = {'unchanged': object()}
            async def produce():
                await asyncio.sleep(0)
                ctx.inputs_queue.put_nowait(payload)
            producer = asyncio.create_task(produce())
            assert ctx.trainer._prepare_inputs() is payload
            await producer
            assert getattr(asyncio.get_running_loop(), '_nest_patched', False)

            ctx.inputs_queue.put_nowait(train._STOP_TRAIN_INPUT)
            try:
                ctx.trainer._prepare_inputs()
            except train.StopTrainingLoop:
                pass
            else:
                raise AssertionError('stop sentinel was lost')

            original_get = ctx.inputs_queue.get
            error = RuntimeError('queue failure')
            async def failing_get():
                raise error
            ctx.inputs_queue.get = failing_get
            try:
                ctx.trainer._prepare_inputs()
            except RuntimeError as caught:
                assert caught is error
            else:
                raise AssertionError('queue failure was lost')

            started = asyncio.Event()
            observed = []
            owner = []
            async def cancellable_get():
                owner.append(asyncio.current_task())
                started.set()
                try:
                    return await original_get()
                except asyncio.CancelledError as caught:
                    observed.append(caught)
                    raise
            async def cancel_get():
                await started.wait()
                owner[0].cancel('queue cancellation')
            ctx.inputs_queue.get = cancellable_get
            canceller = asyncio.create_task(cancel_get())
            try:
                ctx.trainer._prepare_inputs()
            except asyncio.CancelledError as caught:
                assert observed == [caught] and observed[0] is caught
            else:
                raise AssertionError('queue cancellation was lost')
            await canceller
            ctx.inputs_queue.get = original_get
            assert ctx.inputs_queue.empty()
            assert not [t for t in asyncio.all_tasks()
                        if t not in baseline and not t.done()]

        original_run(exercise())
    """)
