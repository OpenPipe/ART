from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from art.trainer_rank import AdamParams, TrainerRank, _impl, _optimizer
from art.trainer_rank._impl import _CheckpointSlot
from tests.unit.test_trainer_rank_validation import _runtime


def _scalar_grads(params: Sequence[torch.nn.Parameter], scale: float):
    return tuple(
        torch.zeros_like(param, dtype=torch.float32)
        if param.grad is None
        else param.grad.detach().float().mul(scale)
        for param in params
    )


def _scalar_multiply(tensors, scale, *, inplace=False):
    return tuple(t.mul_(scale) if inplace else t.mul(scale) for t in tensors)


def _scalar_copy(models, masters):
    for model, master in zip(models, masters, strict=True):
        model.copy_(master)
        model.grad = None


@pytest.mark.parametrize("scale", [0.0, -0.3, 1.0, 1e30, float("inf"), float("nan")])
def test_scaled_grads_match_scalar_without_mutating_inputs(scale):
    params = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        param = torch.nn.Parameter(torch.zeros(3, 2, dtype=dtype).T)
        param.grad = torch.tensor(
            [[0.0, -0.0, 0.125], [-0.75, float("inf"), float("nan")]], dtype=dtype
        )
        params.append(param)
    # Repeated references must still produce separately owned reduced gradients.
    params.extend((params[2], torch.nn.Parameter(torch.ones(2))))
    before = [None if p.grad is None else p.grad.clone() for p in params]
    expected = _scalar_grads(params, scale)
    actual = _optimizer._scaled_grads(params, scale)
    for param, old, value, reference in zip(
        params, before, actual, expected, strict=True
    ):
        torch.testing.assert_close(value, reference, atol=0, rtol=0, equal_nan=True)
        assert value.dtype == torch.float32
        if old is not None:
            torch.testing.assert_close(param.grad, old, atol=0, rtol=0, equal_nan=True)
            assert (
                value.untyped_storage().data_ptr()
                != param.grad.untyped_storage().data_ptr()
            )
    assert (
        actual[2].untyped_storage().data_ptr() != actual[4].untyped_storage().data_ptr()
    )
    assert torch.equal(actual[-1], torch.zeros(2))


def test_scaled_grads_batches_without_an_extra_converted_copy(monkeypatch):
    params = [
        torch.nn.Parameter(torch.ones(4, dtype=d))
        for d in (torch.bfloat16, torch.float32)
    ]
    for param in params:
        param.grad = torch.ones_like(param)
    calls = []
    multiply = _optimizer._multiply

    def record(values, scale, *, inplace=False):
        pointers = [value.data_ptr() for value in values]
        result = multiply(values, scale, inplace=inplace)
        calls.append((inplace, pointers, [value.data_ptr() for value in result]))
        return result

    monkeypatch.setattr(_optimizer, "_multiply", record)
    _optimizer._scaled_grads(params, 0.25)
    assert calls[0][0] is True and calls[0][1] == calls[0][2]
    assert calls[1][0] is False and calls[1][1] != calls[1][2]


def test_subclass_conversion_may_return_borrowed_storage():
    cached = torch.tensor([2.0, 4.0])

    class CachedGradient(torch.Tensor):
        def detach(self):
            return self

        def float(self):
            return cached

    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    param.grad = torch.ones(2, dtype=torch.bfloat16).as_subclass(CachedGradient)
    actual = _optimizer._scaled_grads([param], 0.5)[0]
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0]), atol=0, rtol=0)
    torch.testing.assert_close(cached, torch.tensor([2.0, 4.0]), atol=0, rtol=0)


@pytest.mark.parametrize(
    "kind", ["mixed-dtype", "strides", "sparse", "meta", "subclass"]
)
def test_multiply_compatible_and_fallback_values(monkeypatch, kind):
    values = [torch.arange(6, dtype=torch.float32).reshape(2, 3)]
    if kind == "mixed-dtype":
        values.append(torch.ones(2, dtype=torch.float64))
    elif kind == "strides":
        values.append(values[0].T)
    elif kind == "sparse":
        values.append(values[0].to_sparse())
    elif kind == "meta":
        values.append(torch.empty(3, device="meta"))
    else:

        class TensorSubclass(torch.Tensor):
            pass

        values.append(torch.ones(2).as_subclass(TensorSubclass))
    expected = _scalar_multiply(values, -0.7)
    if kind in ("sparse", "meta", "subclass"):
        monkeypatch.setattr(
            torch, "_foreach_mul", lambda *_: pytest.fail("unsupported foreach")
        )
    actual = _optimizer._multiply(values, -0.7)
    for value, reference in zip(actual, expected, strict=True):
        if value.device.type == "meta":
            assert (value.shape, value.dtype, value.device) == (
                reference.shape,
                reference.dtype,
                reference.device,
            )
        else:
            torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "sliced"])
def test_copy_back_matches_scalar_casts_and_preserves_sources(layout):
    def shaped(tensor):
        return (
            tensor
            if layout == "contiguous"
            else tensor.T
            if layout == "transpose"
            else tensor[:, ::2]
        )

    models = [
        shaped(torch.zeros(3, 4, dtype=dtype))
        for dtype in (torch.bfloat16, torch.float32, torch.float64)
    ]
    masters = [shaped(torch.linspace(-1, 1, 12).reshape(3, 4)) for _ in models]
    before = [master.clone() for master in masters]
    expected = [model.clone() for model in models]
    _scalar_copy(expected, masters)
    _optimizer._copy_back(models, masters)
    for model, reference, master, original in zip(
        models, expected, masters, before, strict=True
    ):
        torch.testing.assert_close(model, reference, atol=0, rtol=0)
        torch.testing.assert_close(master, original, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_uniform_copy_back_batches_casts_and_clears_gradients(monkeypatch, dtype):
    models = [torch.nn.Parameter(torch.zeros(size, dtype=dtype)) for size in (3, 7)]
    masters = [torch.linspace(-0.7, 0.9, model.numel()) for model in models]
    for model in models:
        model.grad = torch.ones_like(model)
    versions = [model._version for model in models]
    copied = []
    original = torch._foreach_copy_

    def record(destination, source):
        copied.append(len(destination))
        original(destination, source)

    monkeypatch.setattr(torch, "_foreach_copy_", record)
    with torch.no_grad():
        _optimizer._copy_back(models, masters)
    assert copied == [2]
    for model, master, version in zip(models, masters, versions, strict=True):
        torch.testing.assert_close(model, master.to(dtype), atol=0, rtol=0)
        assert model.grad is None and model._version == version + 1


@pytest.mark.parametrize(
    "alias", ["destination", "cross-pair", "source", "same-pair", "separate-storage"]
)
def test_copy_back_keeps_alias_order(monkeypatch, alias):
    data = torch.arange(6, dtype=torch.float32)
    if alias == "destination":
        models, masters = (
            [data[:3], data[2:5]],
            [torch.full((3,), 8.0), torch.full((3,), 9.0)],
        )
    elif alias == "cross-pair":
        models, masters = [data[:2], data[2:4]], [data[2:4], data[:2]]
    elif alias == "source":
        models, masters = [torch.empty(3), torch.empty(3)], [data[:3], data[2:5]]
    elif alias == "same-pair":
        models, masters = [data], [data]
    else:
        buffer = bytearray(24)
        models = [
            torch.frombuffer(buffer, dtype=torch.float32, count=3, offset=offset)
            for offset in (0, 8)
        ]
        masters = [torch.full((3,), 8.0), torch.full((3,), 9.0)]
        assert (
            models[0].untyped_storage().data_ptr()
            != models[1].untyped_storage().data_ptr()
        )
    expected_models, expected_masters = deepcopy((models, masters))
    if alias == "separate-storage":
        expected_buffer = bytearray(buffer)
        expected_models = [
            torch.frombuffer(
                expected_buffer, dtype=torch.float32, count=3, offset=offset
            )
            for offset in (0, 8)
        ]
    _scalar_copy(expected_models, expected_masters)
    monkeypatch.setattr(
        torch,
        "_foreach_copy_",
        lambda *_: pytest.fail("aliases must retain ordered copies"),
    )
    _optimizer._copy_back(models, masters)
    for actual, expected in zip(models, expected_models, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_copy_back_device_and_subclass_fallback(monkeypatch):
    class TensorSubclass(torch.Tensor):
        pass

    models = [torch.zeros(2).as_subclass(TensorSubclass), torch.empty(3, device="meta")]
    masters = [torch.ones(2), torch.empty(3, device="meta")]
    monkeypatch.setattr(
        torch, "_foreach_copy_", lambda *_: pytest.fail("unsupported foreach")
    )
    _optimizer._copy_back(models, masters)
    torch.testing.assert_close(models[0], torch.ones(2), atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["shape", "length"])
def test_copy_failure_keeps_prior_gradient_cleanup(failure):
    first, second = (torch.nn.Parameter(torch.zeros(size)) for size in (2, 3))
    for param in (first, second):
        param.grad = torch.ones_like(param)
    masters = [torch.ones(2)] if failure == "length" else [torch.ones(2), torch.ones(2)]
    with (
        torch.no_grad(),
        pytest.raises(ValueError if failure == "length" else RuntimeError),
    ):
        _optimizer._copy_back([first, second], masters)
    torch.testing.assert_close(first, torch.ones(2), atol=0, rtol=0)
    assert first.grad is None
    torch.testing.assert_close(second.grad, torch.ones(3), atol=0, rtol=0)


def test_subclass_copy_observes_prior_gradient_cleanup():
    first = torch.nn.Parameter(torch.zeros(2))
    first.grad = torch.ones_like(first)

    class CheckingTensor(torch.Tensor):
        def copy_(self, other, non_blocking=False):
            assert first.grad is None
            return super().copy_(other, non_blocking=non_blocking)

    second = torch.zeros(2).as_subclass(CheckingTensor)
    second.grad = torch.ones(2)
    with torch.no_grad():
        _optimizer._copy_back([first, second], [torch.ones(2), torch.ones(2)])
    assert second.grad is None
    torch.testing.assert_close(second, torch.ones(2), atol=0, rtol=0)


def test_subclass_clipping_failure_keeps_prior_master_gradient(monkeypatch):
    class FailingGradient(torch.Tensor):
        def mul(self, other, *, out=None):
            raise RuntimeError("clip failed")

    trainer = TrainerRank(_runtime())
    params = tuple(torch.nn.Parameter(torch.ones(2)) for _ in range(2))
    for param in params:
        param.grad = torch.ones_like(param)
    trainer._checkpoint_slots["test"] = _CheckpointSlot(params=params)
    masters = tuple(torch.nn.Parameter(torch.ones(2)) for _ in params)
    grads = (torch.ones(2), torch.ones(2).as_subclass(FailingGradient))
    monkeypatch.setattr(
        trainer, "_reduce_dynamic_grads", lambda *_args, **_kwargs: grads
    )
    monkeypatch.setattr(
        trainer,
        "_dynamic_optimizer",
        lambda *_: SimpleNamespace(master_params=masters, optimizer=None),
    )
    with pytest.raises(RuntimeError, match="clip failed"):
        trainer.optim_step(params=AdamParams(learning_rate=0.01, grad_clip_norm=0.0))
    torch.testing.assert_close(masters[0].grad, torch.ones(2), atol=0, rtol=0)
    assert masters[1].grad is None


def test_empty_pointwise_lists():
    assert _optimizer._multiply([], 1.0) == ()
    assert _optimizer._scaled_grads([], 1.0) == ()
    _optimizer._copy_back([], [])


def test_gradient_collective_buckets_keep_order_and_values(monkeypatch):
    from megatron.core import parallel_state as ps

    from art.megatron.training import finalize_grads

    class Group:
        def __init__(self, name):
            self.name = name

        def size(self):
            return 2

    dp, ep, tp = Group("dp"), Group("ep"), Group("tp")
    monkeypatch.setattr(ps, "get_data_parallel_group", lambda **_: dp)
    monkeypatch.setattr(ps, "get_expert_data_parallel_group", lambda: ep)
    monkeypatch.setattr(ps, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        finalize_grads,
        "tensor_parallel_grad_sync",
        lambda param, **_: (
            (tp, torch.distributed.ReduceOp.SUM)
            if getattr(param, "lora_tp_sharded", False)
            else None
        ),
    )

    def run(scalar):
        calls = []

        def reduce(grads, *, group, op):
            calls.append((group.name, str(op), [grad.clone() for grad in grads]))
            for grad in grads:
                grad.mul_(2)

        monkeypatch.setattr(finalize_grads, "coalesced_all_reduce", reduce)
        params = [
            torch.nn.Parameter(torch.ones(3, dtype=dtype))
            for dtype in (torch.bfloat16, torch.float32, torch.bfloat16, torch.float32)
        ]
        for index, param in enumerate(params[:-1]):
            param.grad = torch.full_like(param, index + 1)
        setattr(params[1], "_art_custom_checkpoint_param", True)
        setattr(params[2], "allreduce", False)
        setattr(params[2], "lora_tp_sharded", True)
        with monkeypatch.context() as patch:
            if scalar:
                patch.setattr(_optimizer, "_scaled_grads", _scalar_grads)
            values = TrainerRank._reduce_dynamic_grads(
                TrainerRank.__new__(TrainerRank), params, scale_grads=0.3
            )
        return values, calls

    expected, expected_calls = run(True)
    actual, actual_calls = run(False)
    assert [entry[:2] for entry in actual_calls] == [
        entry[:2] for entry in expected_calls
    ]
    assert [entry[0] for entry in actual_calls] == ["dp", "ep", "tp"]
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
    for (_, _, values), (_, _, references) in zip(
        actual_calls, expected_calls, strict=True
    ):
        for value, reference in zip(values, references, strict=True):
            torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.parametrize("clip", [0.0, 0.1])
def test_optimizer_metrics_weights_and_moments_match_scalar(monkeypatch, clip):
    def run(scalar):
        with monkeypatch.context() as patch:
            if scalar:
                patch.setattr(_optimizer, "_scaled_grads", _scalar_grads)
                patch.setattr(_optimizer, "_multiply", _scalar_multiply)
                patch.setattr(_optimizer, "_copy_back", _scalar_copy)
            trainer = TrainerRank(_runtime())
            params = tuple(
                torch.nn.Parameter(
                    torch.linspace(-0.3, 0.8, 12, dtype=dtype).reshape(3, 4).T
                )
                for dtype in (torch.bfloat16, torch.float32)
            )
            absent = torch.nn.Parameter(torch.ones(4))
            setattr(absent, "_art_custom_checkpoint_param", True)
            trainer._checkpoint_slots["test"] = _CheckpointSlot(
                params=(*params, absent)
            )
            # No initialized process group: preserve real preparation, norm,
            # AdamW and custom-step rules, with only collectives unavailable.
            patch.setattr(
                trainer,
                "_reduce_dynamic_grads",
                lambda params, *, scale_grads: _optimizer._scaled_grads(
                    params, scale_grads
                ),
            )
            metrics = []
            for step in range(3):
                for param in params:
                    param.grad = (
                        torch.linspace(-0.8 + step * 0.1, 1.3, 12, dtype=param.dtype)
                        .reshape(3, 4)
                        .T
                    )
                metrics.append(
                    trainer.optim_step(
                        params=AdamParams(
                            learning_rate=0.01, weight_decay=0.1, grad_clip_norm=clip
                        ),
                        scale_grads=0.37,
                    )
                )
            slot = trainer._checkpoint_slots["test"]
            dynamic = slot.optimizer
            assert dynamic is not None
            assert dynamic.master_params[-1] not in dynamic.optimizer.state
            assert all(p.grad is None for p in slot.params)
            return (
                metrics,
                slot.revision,
                [p.detach().clone() for p in slot.params],
                dynamic.optimizer.state_dict(),
                [p.detach().clone() for p in dynamic.master_params],
            )

    expected, actual = run(True), run(False)
    assert actual[:2] == expected[:2]

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                equal(a, b)
        else:
            assert left == right

    equal(actual[2:], expected[2:])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_gradient_still_refuses_optimizer_state(monkeypatch, value):
    trainer = TrainerRank(_runtime())
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    param.grad = torch.tensor([1.0, value], dtype=param.dtype)
    trainer._checkpoint_slots["test"] = _CheckpointSlot(params=(param,))
    monkeypatch.setattr(
        trainer,
        "_reduce_dynamic_grads",
        lambda params, *, scale_grads: _optimizer._scaled_grads(params, scale_grads),
    )
    result = trainer.optim_step(params=AdamParams(learning_rate=0.01))
    slot = trainer._checkpoint_slots["test"]
    assert result["update_successful"] == 0.0
    assert slot.optimizer is None and slot.revision == 0
    assert param.grad is None
    torch.testing.assert_close(param, torch.ones_like(param), atol=0, rtol=0)
