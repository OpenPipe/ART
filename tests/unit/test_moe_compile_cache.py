from functools import wraps
import inspect
from types import SimpleNamespace

import pytest
import torch

from art.megatron.compile_workarounds import _isolate_moe_compile_cache


def _decorator(function):
    @wraps(function)
    def wrapped_func(layer, *args, **kwargs):
        if layer.replay:
            return layer.replayed
        return function(layer, *args, **kwargs)

    return wrapped_func


def _layer_type():
    class Layer(torch.nn.Module):
        replay = False
        replayed = None

        @_decorator
        def route(self, x, padding=None):
            return x + 1

        @_decorator
        def preprocess(self, x, probs, routing):
            return x * probs + routing

        @_decorator
        def shared_experts_compute(self, x):
            return x.sin()

    return Layer


def test_methods_keep_independent_compiled_variants_at_default_limit():
    torch._dynamo.reset()
    Layer = _layer_type()
    _isolate_moe_compile_cache(Layer)
    graphs, executions = [], []

    def backend(graph, _inputs):
        index = len(graphs)
        graphs.append(graph)

        def run(*args):
            executions.append(index)
            return graph.forward(*args)

        return run

    functions = [Layer.route, Layer.preprocess, Layer.shared_experts_compute]
    try:
        with torch._dynamo.config.patch(recompile_limit=8):
            optimized = [torch.compile(fn, backend=backend) for fn in functions]
            layer = Layer()
            # A second instance must reuse the same per-method caches.
            for instance in (layer, layer, Layer()):
                hits = len(executions)
                for grad_enabled in (True, False):
                    for requires_grad in (False, True):
                        x = torch.ones(2, requires_grad=requires_grad)
                        for index, args in enumerate(((x, None), (x, x, x), (x,))):
                            with torch.set_grad_enabled(grad_enabled):
                                actual = optimized[index](instance, *args)
                                expected = functions[index](instance, *args)
                            torch.testing.assert_close(actual, expected)
                assert len(executions) - hits == 12
            assert len(graphs) == 12
    finally:
        torch._dynamo.reset()


def test_isolation_preserves_callable_and_is_idempotent():
    Layer = _layer_type()
    function = Layer.route
    before = SimpleNamespace(
        code=function.__code__,
        closure=function.__closure__,
        signature=inspect.signature(function),
        wrapped=function.__wrapped__,
        defaults=function.__defaults__,
        kwdefaults=function.__kwdefaults__,
        annotations=function.__annotations__,
    )
    _isolate_moe_compile_cache(Layer)
    code = function.__code__
    assert function is Layer.route and code is not before.code
    assert function.__closure__ is before.closure
    assert inspect.signature(function) == before.signature
    assert function.__wrapped__ is before.wrapped
    assert function.__defaults__ is before.defaults
    assert function.__kwdefaults__ is before.kwdefaults
    assert function.__annotations__ is before.annotations
    for field in (
        "co_code",
        "co_consts",
        "co_freevars",
        "co_filename",
        "co_firstlineno",
        "co_linetable",
        "co_exceptiontable",
    ):
        assert getattr(code, field) == getattr(before.code, field)
    layer = Layer()
    layer.replay, layer.replayed = True, object()
    assert layer.route(None) is layer.replayed
    _isolate_moe_compile_cache(Layer)
    assert Layer.route.__code__ is code
    assert (
        len(
            {
                id(getattr(Layer, name).__code__)
                for name in ("route", "preprocess", "shared_experts_compute")
            }
        )
        == 3
    )


def test_explicit_compile_disable_remains_disabled():
    Layer = _layer_type()
    original = Layer.preprocess
    disabled = Layer.preprocess = torch.compiler.disable(original)
    _isolate_moe_compile_cache(Layer)
    assert Layer.preprocess is disabled
    assert disabled._torchdynamo_orig_callable is original
    assert disabled._torchdynamo_disable
    assert original.__art_compile_cache_isolated__
    with pytest.raises(TypeError):
        Layer().preprocess(torch.ones(2))
