"""Opt-in cache memory oracle; run on the validation lane's reserved GPU."""

import gc
import os
from types import SimpleNamespace
from typing import Any, cast
import weakref

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from art.trainer_rank._graphs import GraphCache

pytestmark = pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1", reason="requires a reserved GPU"
)


@pytest.fixture
def cyclic_gc_disabled():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.mark.usefixtures("cyclic_gc_disabled")
@pytest.mark.parametrize("finish", ["backward", "release", "evict"])
@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
@pytest.mark.parametrize("multiple_hooks", [False, True])
def test_selective_checkpoint_recomputes_each_retained_backward_and_releases(
    finish, retention, multiple_hooks
):
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    from art.megatron.compile_workarounds import install_reusable_checkpoint_backward

    install_reusable_checkpoint_backward()
    inputs = torch.linspace(0.2, 0.8, 8, device="cuda")
    weight = torch.nn.Parameter(torch.linspace(0.1, 0.7, 8, device="cuda"))
    calls = []
    owners = []
    physical_outputs = []

    def block(x):
        calls.append(True)
        return x.sin()

    def execute(x):
        checkpoint = CheckpointWithoutOutput(fp8=None)
        # Native TransformerLayer keeps this controller on a live module.
        owners.append(checkpoint)
        hidden = checkpoint.checkpoint(block, x * weight)
        physical_outputs.append(weakref.ref(hidden))
        output = hidden.square()
        extra = hidden * 2 if multiple_hooks else None
        checkpoint.discard_output_and_register_recompute(output)
        if extra is not None:
            checkpoint.discard_output_and_register_recompute(extra)
            output = output + extra
        return (output,)

    cache = GraphCache()
    handle, (output,) = cache.run(
        execute,
        inputs,
        retention=retention,
        options=SimpleNamespace(
            allow_replay=retention == "replay" or finish == "evict"
        ),
    )
    z = inputs * weight.detach()
    expected = 2 * z.sin() * z.cos() * inputs
    if multiple_hooks:
        expected += 2 * z.cos() * inputs
    consume = finish == "backward"
    for retain in (True, True, False) if consume else (True,):
        weight.grad = None
        cache.backward(handle, (torch.ones_like(output),), retain_graph=retain)
        torch.testing.assert_close(weight.grad, expected)
    backwards = 3 if consume else 1
    assert len(calls) == 1 + backwards * (2 if retention == "replay" else 1)
    if finish == "evict":
        cache.evict(handle)
        assert all(ref() is None for ref in physical_outputs)
    cache.release(handle)
    assert all(ref() is None for ref in physical_outputs)
    assert all(
        getattr(owner, field) is None
        for owner in owners
        for field in ("run_function", "rng_states", "outputs", "ctx")
    )
    assert cache.handles() == ()


@pytest.mark.parametrize("no_grad", [False, True])
def test_selective_checkpoint_failure_and_no_grad_clear_module_owner(no_grad):
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    from art.megatron.compile_workarounds import install_reusable_checkpoint_backward

    install_reusable_checkpoint_backward()
    checkpoint = CheckpointWithoutOutput(fp8=None)
    weight = torch.nn.Parameter(torch.ones(8, device="cuda"))
    calls = []

    def block(x):
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("injected selective recompute failure")
        return x.sin()

    def execute(x):
        with torch.set_grad_enabled(not no_grad):
            hidden = checkpoint.checkpoint(block, x * weight)
            output = hidden.square()
            checkpoint.discard_output_and_register_recompute(output)
            if no_grad:
                checkpoint.discard_output_and_register_recompute(output)
        return (output,)

    cache = GraphCache()
    handle, (output,) = cache.run(execute, torch.ones_like(weight))
    owner = getattr(checkpoint, "_art_recompute_owner")
    if no_grad:
        assert owner() is None
        cache.release(handle)
    else:
        with pytest.raises(RuntimeError, match="injected selective recompute failure"):
            cache.backward(handle, (torch.ones_like(output),), retain_graph=True)
        if owner() is not None:
            assert owner().ctx is None
            assert owner().run_function is None
    assert checkpoint.ctx is None
    assert checkpoint.outputs is None
    assert checkpoint.run_function is None
    assert checkpoint.rng_states is None
    assert cache.handles() == ()


@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
def test_gpu_saved_state_offload_eviction_and_replay(retention):
    torch.manual_seed(7)
    cache = GraphCache()
    weight = torch.nn.Parameter(torch.randn(2048, device="cuda"))
    original = torch.randn(2048, 2048, device="cuda")
    reference = torch.nn.Parameter(weight.detach().clone())
    expected = (original.sin() * reference).sum(1)
    expected.sum().backward()
    storage_key = weight.untyped_storage().data_ptr()

    def execute(x):
        return ((x.sin() * weight).sum(1),)

    handle, (output,) = cache.run(
        execute,
        original,
        retention=retention,
        cuda_devices=[torch.cuda.current_device()],
        execution_peak_bytes=3 * original.numel() * original.element_size(),
        keep_on_device=lambda tensor: (
            tensor.untyped_storage().data_ptr() == storage_key
        ),
    )
    torch.testing.assert_close(output, expected)
    if retention == "gpu":
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        state = cache.state(handle)
        assert state.offload_bytes >= original.numel() * original.element_size()
        cache.offload(handle)
        gc.collect()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
        assert before - after >= original.numel() * original.element_size()
        assert cache.state(handle).gpu_bytes <= output.numel() * output.element_size()
    cache.evict(handle)
    assert cache.state(handle).gpu_bytes == 0
    assert cache.state(handle).restore_workspace_bytes == (
        3 * original.numel() * original.element_size()
    )
    original.fill_(100)
    ambient = torch.cuda.get_rng_state()
    cache.backward(handle, (torch.ones_like(output),))
    torch.testing.assert_close(weight.grad, reference.grad)
    assert torch.equal(torch.cuda.get_rng_state(), ambient)


@pytest.mark.parametrize(
    ("retention", "cache_state"),
    [
        ("gpu", "cold"),
        ("gpu", "warm"),
        ("gpu", "artifact"),
        ("cpu", "artifact"),
        ("replay", "artifact"),
    ],
)
def test_compiled_retained_backward_preserves_original_gradient_after_update(
    retention, cache_state, tmp_path, monkeypatch
):
    from torch._dynamo.utils import counters
    from torch._functorch import config as functorch_config

    from art.megatron.training.compile import _configure_dynamo
    from art.trainer_rank import TrainerRank
    from art.trainer_rank._impl import _CheckpointSlot

    # Isolate disk artifacts, including an incompatible donating predecessor.
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    torch.compiler.reset()
    torch.manual_seed(71)
    inputs = torch.randn(48, 32, device="cuda", dtype=torch.float64)
    weight = torch.nn.Parameter(torch.randn(32, 16, device="cuda", dtype=torch.float64))

    def model(x, w):
        return (x @ w).sin().square()

    def warm(compiled):
        compiled(inputs, weight).sum().backward()
        weight.grad = None

    with (
        functorch_config.patch(donated_buffer=True, enable_autograd_cache=True),
        torch._dynamo.config.patch(force_parameter_static_shapes=False),
        cast(Any, torch.compiler.config).patch(
            cache_key_tag=f"graph-oracle:{tmp_path}"
        ),
    ):
        try:
            warm(torch.compile(model, fullgraph=True))
            donated = torch.compiler.save_cache_artifacts()
            assert donated is not None
            torch.compiler.reset()
            assert torch.compiler.load_cache_artifacts(donated[0]) is not None
            _configure_dynamo()
            compiled = torch.compile(model, fullgraph=True)
            # A prior non-retaining backward must not compile a donating kernel
            # that rejects the later retained backward of the same graph shape.
            if cache_state != "cold":
                misses = counters["aot_autograd"]["autograd_cache_miss"]
                warm(compiled)
                assert counters["aot_autograd"]["autograd_cache_miss"] > misses
            if cache_state == "artifact":
                reusable = torch.compiler.save_cache_artifacts()
                assert reusable is not None
                torch.compiler.reset()
                assert torch.compiler.load_cache_artifacts(reusable[0]) is not None
                compiled = torch.compile(model, fullgraph=True)
                hits = counters["aot_autograd"]["autograd_cache_hit"]
                warm(compiled)
                assert counters["aot_autograd"]["autograd_cache_hit"] > hits

            trainer = TrainerRank.__new__(TrainerRank)
            trainer.runtime = SimpleNamespace(model=[], optimizer=None)
            trainer._checkpoint_slots = {"student": _CheckpointSlot(params=(weight,))}
            version = trainer._capture_checkpoint_version("student")
            original = trainer._snapshot_parameter(weight, version)
            z = inputs @ original.detach()
            first = torch.randn_like(z)
            second = torch.randn_like(z)
            expected = [inputs.T @ (2 * z.sin() * z.cos() * g) for g in (first, second)]
            cache = GraphCache()
            handle, (output,) = cache.run(
                lambda x: (compiled(x, original),),
                inputs,
                retention=retention,
                options=SimpleNamespace(allow_replay=retention == "replay"),
                cuda_devices=[torch.cuda.current_device()],
            )
            torch.testing.assert_close(output, z.sin().square())
            with trainer._gradient_transaction():
                cache.backward(handle, (first,), retain_graph=True)
            torch.testing.assert_close(weight.grad, expected[0])
            assert cache.handles() == (handle,)
            weight.grad = None
            with torch.no_grad():
                weight.add_(0.25)
            trainer._checkpoint_slots["student"].revision += 1
            inputs.fill_(999)
            with trainer._gradient_transaction():
                cache.backward(handle, (second,))
            torch.testing.assert_close(weight.grad, expected[1])
            assert original.grad is None
            assert cache.handles() == ()
        finally:
            torch.compiler.reset()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("retention", ["gpu", "cpu"])
@pytest.mark.parametrize("operator", ["rmsnorm", "linear"])
def test_te_three_backwards_match_manual_gradient(compiled, retention, operator):
    from transformer_engine.pytorch import Linear
    from transformer_engine.pytorch.ops import RMSNorm

    from art.megatron.compile_workarounds import install_te_reusable_backward
    from art.megatron.runtime.compile_cache import configure_reusable_backward

    configure_reusable_backward()
    install_te_reusable_backward()
    torch.compiler.reset()
    torch.manual_seed(109)
    inputs = torch.randn(48, 32, device="cuda")
    weight = torch.nn.Parameter(torch.randn(32, device="cuda"))
    operation = (
        RMSNorm(32, device="cuda", dtype=torch.float32)
        if operator == "rmsnorm"
        else Linear(32, 32, bias=False, device="cuda", params_dtype=torch.float32)
    )
    base_weight = cast(torch.Tensor, operation.weight)
    base_weight.requires_grad_(False)
    if operator == "linear":
        # Exact dyadic inputs give the same oracle with TE's TF32 GEMM and
        # PyTorch's FP32 GEMM, without relaxing the retained-gradient check.
        inputs.copy_(torch.randint(-4, 5, inputs.shape, device="cuda") / 4)
        with torch.no_grad():
            weight.copy_(torch.randint(-4, 5, weight.shape, device="cuda") / 4)
            base_weight.copy_(
                torch.randint(-4, 5, base_weight.shape, device="cuda") / 4
            )

    def model(x):
        return operation(x * weight)

    physical = torch.compile(model) if compiled else model
    executions = []

    def execute(x):
        executions.append(True)
        return (physical(x),)

    cache = GraphCache()
    handle, (output,) = cache.run(
        execute,
        inputs,
        retention=retention,
        options=SimpleNamespace(allow_replay=False),
    )
    z = inputs * weight.detach()
    inverse = (z.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    torch.testing.assert_close(
        output, z * inverse if operator == "rmsnorm" else z @ base_weight.T
    )
    for retain in (True, True, False):
        cotangent = torch.randn_like(output)
        if operator == "linear":
            cotangent.copy_(torch.randint(-4, 5, output.shape, device="cuda") / 4)
        if operator == "rmsnorm":
            derivative = cotangent * inverse - z * inverse.pow(3) * (
                cotangent * z
            ).mean(-1, keepdim=True)
        else:
            derivative = cotangent @ base_weight
        weight.grad = None
        cache.backward(handle, (cotangent,), retain_graph=retain)
        torch.testing.assert_close(weight.grad, (inputs * derivative).sum(0))
        assert cache.handles() == ((handle,) if retain else ())
    assert len(executions) == 1
    torch.compiler.reset()


def test_te_retained_backward_failure_clears_unpacked_context(monkeypatch):
    from transformer_engine.pytorch.ops import RMSNorm
    from transformer_engine.pytorch.ops.fuser import _OperationFuserAutogradFunction

    from art.megatron.compile_workarounds import install_te_reusable_backward

    install_te_reusable_backward()
    installed = _OperationFuserAutogradFunction.backward
    install_te_reusable_backward()
    assert _OperationFuserAutogradFunction.backward is installed
    norm = RMSNorm(32, device="cuda", dtype=torch.float32)
    norm.weight.requires_grad_(False)
    weight = torch.nn.Parameter(torch.ones(32, device="cuda"))
    contexts = []

    def fail(ctx, gradient):
        assert ctx.saved_tensors is not None
        contexts.append(ctx)
        raise RuntimeError("injected TE backward failure")

    monkeypatch.setattr(norm, "op_backward", fail)
    cache = GraphCache()
    handle, (output,) = cache.run(
        lambda x: (norm(x * weight),), torch.ones(48, 32, device="cuda")
    )
    with pytest.raises(RuntimeError, match="injected TE backward failure"):
        cache.backward(handle, (torch.ones_like(output),), retain_graph=True)
    assert cache.handles() == ()
    assert len(contexts) == 1
    assert contexts[0].saved_tensors is None
    assert contexts[0]._saved_tensors_range is not None


def test_cpu_saved_views_share_storage_and_exclude_weight_views():
    cache = GraphCache()
    weight = torch.nn.Parameter(torch.randn(8, device="cuda"))
    inputs = torch.randn(4, 8, device="cuda")
    storage_key = weight.untyped_storage().data_ptr()

    def execute(x):
        # Both multiplications save aliased input storage for weight gradients.
        return ((x * weight).sum(), (x[1:] * weight).sum())

    handle, outputs = cache.run(
        execute,
        inputs,
        retention="cpu",
        cuda_devices=[torch.cuda.current_device()],
        keep_on_device=lambda tensor: (
            tensor.untyped_storage().data_ptr() == storage_key
        ),
    )
    cells = [
        cell
        for ref in cache._records[handle].saved or ()
        if (cell := ref()) is not None
    ]
    saved_inputs = [cell.tensor for cell in cells if cell.managed]
    assert len(saved_inputs) == 2
    assert all(value.is_pinned() for value in saved_inputs)
    assert (
        saved_inputs[0].untyped_storage().data_ptr()
        == saved_inputs[1].untyped_storage().data_ptr()
    )
    assert saved_inputs[1].storage_offset() == 8
    size = inputs.numel() * inputs.element_size()
    assert cache.transfer_stats.offload_bytes == size
    assert cache.transfer_stats.offload_count == 1
    assert cache.transfer_stats.offload_max_bytes == size
    assert cache.transfer_stats.offload_seconds > 0
    assert cache.transfer_stats.restore_count == 0
    cache.backward(handle, tuple(torch.ones_like(value) for value in outputs))
    assert cache.handles() == ()
    assert cache.transfer_stats.restore_bytes == size
    assert cache.transfer_stats.restore_count == 1
    assert cache.transfer_stats.restore_max_bytes == size
    assert cache.transfer_stats.restore_seconds > 0
    torch.testing.assert_close(weight.grad, inputs.sum(0) + inputs[1:].sum(0))


def test_saved_alias_restore_uses_one_storage_and_bounded_peak():
    class Aliases(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight, value):
            ctx.save_for_backward(*(value[:, offset:] for offset in range(24)))
            return weight * value.sum()

        @staticmethod
        def backward(ctx, *gradients):
            saved = ctx.saved_tensors
            assert len({value.untyped_storage().data_ptr() for value in saved}) == 1
            return gradients[0] * saved[0].sum(), None

    weight = torch.nn.Parameter(torch.tensor(2.0, device="cuda"))
    inputs = torch.ones(2048, 2048, device="cuda")
    cache = GraphCache()
    handle, (output,) = cache.run(
        lambda x: (Aliases.apply(weight, x),), inputs, retention="cpu"
    )
    size = inputs.numel() * inputs.element_size()
    for retain in (True, False):
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        cache.backward(handle, (torch.ones_like(output),), retain_graph=retain)
        torch.cuda.synchronize()
        increase = torch.cuda.max_memory_allocated() - baseline
        assert increase < size * 2, (increase, size)
        if retain:
            assert cache._records[handle].restored == {}
            assert (
                cache.state(handle).gpu_bytes <= output.numel() * output.element_size()
            )
    torch.testing.assert_close(
        weight.grad, torch.tensor(2 * inputs.numel(), device="cuda", dtype=weight.dtype)
    )
    assert cache.transfer_stats.offload_count == 1
    assert cache.transfer_stats.offload_bytes == size
    assert cache.transfer_stats.restore_count == 2
    assert cache.transfer_stats.restore_bytes == size * 2


def test_saved_physical_output_alias_is_not_offloaded_or_duplicated():
    cache = GraphCache()
    weight = torch.nn.Parameter(torch.ones(2048, device="cuda"))
    handle, (output,) = cache.run(
        lambda x: ((weight * x).exp(),),
        torch.ones(2048, device="cuda"),
        retention="cpu",
    )
    cells = [
        cell
        for ref in cache._records[handle].saved or ()
        if (cell := ref()) is not None
    ]
    physical_outputs = cache._records[handle].outputs
    assert physical_outputs is not None
    physical = physical_outputs[0]
    assert any(
        not cell.managed
        and cell.tensor.untyped_storage().data_ptr()
        == physical.untyped_storage().data_ptr()
        for cell in cells
    )
    cache.backward(handle, (torch.ones_like(output),))
    torch.testing.assert_close(weight.grad, torch.full_like(weight, torch.e))


@pytest.mark.parametrize("retention", ["gpu", "cpu"])
def test_pinned_offload_completes_on_user_stream_and_preserves_strided_views(retention):
    cache = GraphCache()
    source = torch.arange(8192.0, device="cuda").reshape(64, 128)
    weight = torch.nn.Parameter(torch.ones(64, device="cuda"))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        handle, (output,) = cache.run(
            lambda x: ((x.t()[3::7] * weight).sum(),),
            source,
            retention=retention,
        )
        if retention == "gpu":
            assert cache.transfer_stats.offload_count == 0
            cache.offload(handle)
        saved = [
            cell.tensor
            for ref in cache._records[handle].saved or ()
            if (cell := ref()) is not None and cell.managed
        ]
        assert len(saved) == 1 and saved[0].is_pinned()
        # A blocking D2H copy is immediately readable on CPU, without waiting
        # on the producer stream separately. Noncontiguous view metadata stays.
        expected = torch.arange(8192.0).reshape(64, 128).t()[3::7]
        assert saved[0].stride() == expected.stride()
        torch.testing.assert_close(saved[0], expected)
    torch.cuda.current_stream().wait_stream(stream)
    cache.backward(handle, (torch.ones_like(output),))
    assert weight.grad is not None
    torch.testing.assert_close(weight.grad.cpu(), expected.sum(0))
    assert cache.transfer_stats.offload_count == 1
    assert cache.transfer_stats.restore_count == 1


@pytest.mark.parametrize("late", [False, True])
def test_failed_pinned_allocation_releases_partial_saved_state(monkeypatch, late):
    allocate = torch.empty_like
    storages = []

    def fail_second(tensor, **kwargs):
        if kwargs.get("pin_memory"):
            if storages:
                raise RuntimeError("injected pinned allocation failure")
            result = allocate(tensor, **kwargs)
            storages.append(StorageWeakRef(result.untyped_storage()))
            return result
        return allocate(tensor, **kwargs)

    monkeypatch.setattr(torch, "empty_like", fail_second)
    cache = GraphCache()
    weight = torch.nn.Parameter(torch.ones(32, device="cuda"))

    def execute(x):
        partial = x * weight
        return (partial * x.square(),)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        handle = None
        if late:
            handle, _ = cache.run(execute, torch.ones_like(weight))
        with pytest.raises(RuntimeError, match="injected pinned") as failure:
            if handle is None:
                cache.run(execute, torch.ones_like(weight), retention="cpu")
            else:
                cache.offload(handle)
        assert failure.value.__traceback__ is not None
        if handle is not None:
            # A late failure leaves a valid, partially offloaded graph. Its
            # storage remains owned until the caller explicitly releases it.
            assert not storages[0].expired()
            cache.release(handle)
        assert len(storages) == 1 and storages[0].expired()
        assert cache.handles() == ()
        assert cache.transfer_stats.offload_count == 1
        assert cache.transfer_stats.restore_count == 0
    finally:
        if was_enabled:
            gc.enable()
