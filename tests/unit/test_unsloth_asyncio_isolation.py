from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


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


@pytest.mark.parametrize(
    "mode", ["payload", "stop", "error", "cancel", "handoff_cancel"]
)
def test_actual_trainer_callback_retains_nested_queue_and_error_behavior(
    mode: str,
) -> None:
    _run(
        "MODE = "
        + repr(mode)
        + "\n"
        + textwrap.dedent("""
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

        original_loss = lambda *args: None
        original_log = lambda *args: None
        ctx.trainer.compute_loss = original_loss
        ctx.trainer.log = original_log
        # Substitute GPU computation/log delivery only, retaining actual train()
        # setup/finally, installed input callback, queues, and stop handling.
        train.get_compute_loss_fn = lambda trainer: lambda *args: None
        train.get_log_fn = lambda trainer, queue: lambda *args: None

        async def exercise():
            baseline = set(asyncio.all_tasks())
            payload = {'unchanged': object()}
            called = []
            siblings = []
            helpers = []
            original_get = ctx.inputs_queue.get
            error = RuntimeError('queue failure')
            started = asyncio.Event()
            observed = []
            readers = []

            async def produce():
                await asyncio.sleep(0)
                ctx.inputs_queue.put_nowait(
                    train._STOP_TRAIN_INPUT if MODE == 'stop' else payload)
            async def failing_get():
                raise error
            async def cancellable_get():
                readers.append(asyncio.current_task())
                started.set()
                try:
                    return await original_get()
                except asyncio.CancelledError as caught:
                    observed.append(caught)
                    raise
            async def cancel_get():
                await started.wait()
                readers[0].cancel('queue cancellation')

            def synchronous_train():
                called.append(True)
                if MODE == 'error':
                    ctx.inputs_queue.get = failing_get
                elif MODE == 'cancel':
                    ctx.inputs_queue.get = cancellable_get
                    helpers.append(asyncio.create_task(cancel_get()))
                else:
                    helpers.append(asyncio.create_task(produce()))
                assert ctx.trainer._prepare_inputs() is payload
                assert MODE == 'payload', 'stop/error/cancellation was swallowed'

            ctx.trainer.train = synchronous_train
            owner = asyncio.create_task(train.train(ctx.trainer, ctx.results_queue))
            # First activation happens with sibling work already in the native
            # loop's fixed ready-count. Immediate nesting used to drain its deque.
            for i in range(5):
                asyncio.get_running_loop().call_soon(siblings.append, i)
            if MODE == 'handoff_cancel':
                asyncio.get_running_loop().call_soon(owner.cancel, 'handoff')
            try:
                await owner
            except RuntimeError as caught:
                assert MODE == 'error' and caught is error
            except asyncio.CancelledError as caught:
                if MODE == 'cancel':
                    assert observed == [caught] and observed[0] is caught
                else:
                    assert MODE == 'handoff_cancel' and caught.args == ('handoff',)
            else:
                assert MODE in ('payload', 'stop'), 'interruption was swallowed'
            await asyncio.gather(*helpers)
            assert called == ([] if MODE == 'handoff_cancel' else [True])
            assert siblings == list(range(5))
            assert getattr(asyncio.get_running_loop(), '_nest_patched', False)
            assert ctx.trainer.compute_loss is original_loss
            assert ctx.trainer.log is original_log
            ctx.inputs_queue.get = original_get
            assert ctx.inputs_queue.empty()
            assert not [t for t in asyncio.all_tasks()
                        if t not in baseline and not t.done()]

        original_run(exercise())
    """)
    )
