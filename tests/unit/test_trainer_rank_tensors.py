from __future__ import annotations

from collections import OrderedDict, namedtuple
from dataclasses import dataclass, field
import pickle

import pytest
import torch

from art.trainer_rank import ForwardOutput, TopK
from art.trainer_rank._tensors import (
    CotangentCollector,
    ManagedTensor,
    detach_tree,
    flatten_tensors,
    managed_tensor,
    managed_tree,
    unflatten_tensors,
)


@dataclass(frozen=True, slots=True)
class NestedOutput:
    values: object
    label: str = "root"
    metadata: int = field(default=7, init=False)


def test_tree_preserves_nested_types_metadata_and_tensor_aliases():
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    tokens = torch.tensor([3, 4])
    Pair = namedtuple("Pair", "first second")
    tree = OrderedDict(
        outer=[NestedOutput((value, None)), Pair(value, tokens)],
        size=torch.Size((2, 3)),
        empty=[],
    )
    tensors, spec = flatten_tensors(tree)
    assert len(tensors) == 2
    restored = unflatten_tensors(spec, tuple(t + 1 for t in tensors))
    assert isinstance(restored, OrderedDict)
    assert isinstance(restored["outer"][0], NestedOutput)
    assert isinstance(restored["outer"][1], Pair)
    assert restored["outer"][0].metadata == 7
    assert restored["outer"][0].values[0] is restored["outer"][1].first
    assert restored["outer"][0].values[1] is None
    assert restored["size"] == torch.Size((2, 3))
    assert restored["empty"] == []


def test_forward_packet_pickle_and_detached_storage():
    parameter = torch.tensor([2.0, 3.0], requires_grad=True)
    physical = ForwardOutput(
        parameter.square(), TopK(parameter + 1, torch.tensor([0, 1])), None, None
    )
    packet = detach_tree("forward:1", {"output": physical})
    assert all(
        tensor.grad_fn is None and not tensor.requires_grad for tensor in packet.tensors
    )
    packet = pickle.loads(pickle.dumps(packet))
    collector = CotangentCollector()
    output = collector.attach(packet)["output"]
    assert output.target_logprobs.requires_grad
    assert not output.top_k.tokens.requires_grad
    assert output.logits is None and output.hidden_states is None
    packet.tensors[0].zero_()
    torch.testing.assert_close(physical.target_logprobs, torch.tensor([4.0, 9.0]))


def _model(x, weight):
    hidden = torch.tanh(x @ weight)
    return ForwardOutput(
        hidden.log_softmax(-1),
        TopK(hidden[:, :1], torch.zeros((2, 1), dtype=torch.long)),
        None,
        hidden,
    )


def _loss(outputs, head):
    a, b = outputs
    return ((a.hidden_states @ head) - (b.hidden_states @ head)).square().sum() + (
        a.target_logprobs[:, 0] - b.target_logprobs[:, 1]
    ).square().sum()


@pytest.mark.parametrize("managed", [False, True])
def test_coupled_forwards_and_local_head_match_connected_reference(managed):
    generator = torch.Generator().manual_seed(12)
    weight = torch.randn(
        3, 4, dtype=torch.float64, generator=generator, requires_grad=True
    )
    inputs = [
        torch.randn(2, 3, dtype=torch.float64, generator=generator) for _ in range(2)
    ]
    head = torch.randn(
        4, 1, dtype=torch.float64, generator=generator, requires_grad=True
    )
    physical = [_model(x, weight) for x in inputs]
    collector = CotangentCollector()
    outputs = [
        collector.attach(detach_tree(str(i), value), managed=managed)
        for i, value in enumerate(physical)
    ]
    packets = collector.backward(_loss(outputs, head))
    assert weight.grad is None
    assert [packet.handle for packet in packets] == ["0", "1"]
    actual_outputs, cotangents = [], []
    for packet, value in zip(packets, physical, strict=True):
        leaves, _ = flatten_tensors(value)
        assert packet.gradients[1] is None  # unused top-k logprobs
        assert packet.gradients[2] is None  # integer token IDs
        for leaf, grad in zip(leaves, packet.gradients, strict=True):
            if grad is not None:
                actual_outputs.append(leaf)
                cotangents.append(grad)
    torch.autograd.backward(actual_outputs, cotangents)
    reference_weight = weight.detach().clone().requires_grad_()
    reference_head = head.detach().clone().requires_grad_()
    _loss([_model(x, reference_weight) for x in inputs], reference_head).backward()
    torch.testing.assert_close(weight.grad, reference_weight.grad)
    torch.testing.assert_close(head.grad, reference_head.grad)


def test_aliases_unused_forwards_and_explicit_repeated_backward():
    collector = CotangentCollector()
    value = torch.tensor([2.0, 3.0], requires_grad=True)
    output = collector.attach(detach_tree("used", [value, value]))
    collector.attach(detach_tree("unused", value))
    assert output[0] is output[1]
    loss = (output[0] * output[1]).sum()
    first = collector.backward(loss, retain_graph=True)
    second = collector.backward(loss)
    assert len(first) == len(second) == 1
    torch.testing.assert_close(first[0].gradients[0], 2 * value)
    torch.testing.assert_close(second[0].gradients[0], first[0].gradients[0])
    with pytest.raises(RuntimeError, match="second time"):
        collector.backward(loss)


def test_tuple_backward_and_multiple_snapshots_of_same_handle():
    collector = CotangentCollector()
    packet = detach_tree("head:1", torch.tensor([2.0, 3.0], requires_grad=True))
    a, b = collector.attach(packet), collector.attach(packet)
    gradients = collector.backward((a, b), (torch.ones(2), torch.full((2,), 2.0)))
    assert len(gradients) == 1
    torch.testing.assert_close(gradients[0].gradients[0], torch.full((2,), 3.0))


@pytest.mark.parametrize("managed", [False, True])
def test_local_failure_discards_already_collected_remote_cotangents(managed):
    collector = CotangentCollector()
    bridge = collector.attach(
        detach_tree("first", torch.tensor(3.0, requires_grad=True))
    )
    value = managed_tensor(bridge) if managed else bridge
    seen = []

    def fail_after_collection(*args):
        seen.append(bool(collector._pending))
        raise RuntimeError("local loss failed")

    hook = bridge.grad_fn.register_hook(fail_after_collection)
    with pytest.raises(RuntimeError, match="local loss failed"):
        collector.backward(value, retain_graph=True)
    assert seen == [True]
    assert collector._pending is None
    hook.remove()
    recovered = collector.backward(value)
    assert len(recovered) == 1
    torch.testing.assert_close(recovered[0].gradients[0], torch.tensor(1.0))


def test_unscoped_backward_is_rejected():
    collector = CotangentCollector()
    value = collector.attach(
        detach_tree("forward", torch.tensor(3.0, requires_grad=True))
    )
    with pytest.raises(RuntimeError, match="trainer.backward"):
        value.backward()


def test_empty_and_nondifferentiable_trees():
    collector = CotangentCollector()
    tree = [None, {"tokens": torch.tensor([1, 2]), "labels": []}]
    result = collector.attach(detach_tree("empty", tree))
    assert result[0] is None
    assert not result[1]["tokens"].requires_grad
    assert result[1]["labels"] == []
    assert collector.backward(torch.tensor(2.0, requires_grad=True)) == ()


@pytest.mark.parametrize("reverse", [False, True])
def test_managed_cpu_arithmetic_keeps_original_gradient_paths(reverse):
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([5.0, 7.0], requires_grad=True)
    managed = managed_tree({"a": a})["a"]
    result = b * managed if reverse else managed * b
    assert isinstance(result, ManagedTensor)
    result.sum().backward()
    torch.testing.assert_close(a.grad, b.detach())
    torch.testing.assert_close(b.grad, a.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("managed_device", ["cpu", "cuda"])
@pytest.mark.parametrize("reverse", [False, True])
def test_managed_cross_device_arithmetic_and_original_gradients(
    managed_device, reverse
):
    other_device = "cuda" if managed_device == "cpu" else "cpu"
    a = torch.tensor([2.0, 3.0], device=managed_device, requires_grad=True)
    b = torch.tensor([5.0, 7.0], device=other_device, requires_grad=True)
    managed = managed_tensor(a)
    result = b * managed if reverse else managed * b
    assert result.device.type == managed_device
    result.sum().backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.device == a.device and b.grad.device == b.device
    torch.testing.assert_close(a.grad.cpu(), b.detach().cpu())
    torch.testing.assert_close(b.grad.cpu(), a.detach().cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("input_device", ["cpu", "cuda"])
def test_managed_linear_cpu_cuda_matches_explicit_copies(input_device):
    other_device = "cuda" if input_device == "cpu" else "cpu"
    torch.manual_seed(7)
    inputs = torch.randn(
        2, 3, device=input_device, dtype=torch.float64, requires_grad=True
    )
    head = torch.nn.Linear(3, 4, device=other_device, dtype=torch.float64)
    result = head(managed_tensor(inputs)).square().sum()
    assert result.device.type == input_device
    result.backward()
    expected_input = inputs.detach().cpu().requires_grad_()
    expected_weight = head.weight.detach().cpu().requires_grad_()
    expected_bias = head.bias.detach().cpu().requires_grad_()
    torch.nn.functional.linear(
        expected_input, expected_weight, expected_bias
    ).square().sum().backward()
    for actual, expected in [
        (inputs, expected_input),
        (head.weight, expected_weight),
        (head.bias, expected_bias),
    ]:
        assert actual.grad is not None
        assert actual.grad.device == actual.device
        torch.testing.assert_close(actual.grad.cpu(), expected.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_managed_nested_operands_and_mixed_device_mutation_rejection():
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], device="cuda", requires_grad=True)
    result = torch.cat([managed_tensor(a), b])
    assert result.device.type == "cpu"
    result.sum().backward()
    assert a.grad is not None and b.grad is not None
    assert a.grad.item() == b.grad.item() == 1
    with pytest.raises(RuntimeError, match="mutation"):
        managed_tensor(a.detach()).add_(b.detach())
    with pytest.raises(RuntimeError, match="mutation"):
        torch.add(managed_tensor(a.detach()), b.detach(), out=torch.empty_like(b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_managed_devices_choose_cpu_in_either_direction(reverse):
    cpu = torch.tensor([2.0], requires_grad=True)
    cuda = torch.tensor([3.0], device="cuda", requires_grad=True)
    a, b = managed_tensor(cpu), managed_tensor(cuda)
    result = b * a if reverse else a * b
    assert result.device.type == "cpu"
    result.sum().backward()
    assert cpu.grad is not None and cuda.grad is not None
    assert cpu.grad.device.type == "cpu" and cpu.grad.item() == 3
    assert cuda.grad.device.type == "cuda" and cuda.grad.item() == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_output_bridge_gpu_head_and_remote_model_cotangents():
    weight = torch.tensor(
        [[2.0], [3.0]], device="cuda", dtype=torch.float64, requires_grad=True
    )
    physical = weight.square()
    collector = CotangentCollector()
    output = collector.attach(
        detach_tree("gpu-model", physical, device="cpu"), managed=True
    )
    head = torch.nn.Linear(1, 1, bias=False, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        head.weight.fill_(4)
    loss = head(output).sum()
    assert loss.device.type == "cpu"
    (packet,) = collector.backward(loss)
    assert packet.gradients[0] is not None
    assert packet.gradients[0].device.type == "cpu"
    assert weight.grad is None
    torch.autograd.backward(physical, packet.gradients[0].to(physical.device))
    torch.testing.assert_close(weight.grad, 8 * weight.detach())
    torch.testing.assert_close(
        head.weight.grad, torch.tensor([[13.0]], device="cuda", dtype=torch.float64)
    )


@pytest.mark.parametrize("managed", [False, True])
def test_release_follows_dependent_loss_not_temporary_output(managed):
    import gc
    import weakref

    collector = CotangentCollector()
    released = []
    value = collector.attach(
        detach_tree("released", torch.tensor(2.0, requires_grad=True)),
        managed=managed,
        on_release=lambda: released.append("released"),
    )
    original = weakref.ref(value)
    loss = value + 1  # addition does not save its input Python wrapper
    del value
    gc.collect()
    assert original() is None
    assert not released
    collector.backward(loss)
    assert not released
    del loss
    gc.collect()
    assert released == ["released"]
    gc.collect()
    assert released == ["released"]


def test_release_follows_retained_graph_and_all_output_branches():
    import gc

    collector = CotangentCollector()
    released = []
    outputs = collector.attach(
        detach_tree(
            "retained",
            [
                torch.tensor(2.0, requires_grad=True),
                torch.tensor(3.0, requires_grad=True),
            ],
        ),
        on_release=lambda: released.append(True),
    )
    first, second = [output.square() for output in outputs]
    del outputs
    collector.backward(first, retain_graph=True)
    collector.backward(first, retain_graph=True)
    del first
    gc.collect()
    assert not released
    collector.backward(second)
    del second
    gc.collect()
    assert released == [True]


def test_release_dropped_and_nondifferentiable_packets():
    import gc

    collector = CotangentCollector()
    released = []
    output = collector.attach(
        detach_tree("unused", torch.tensor(2.0, requires_grad=True)),
        on_release=lambda: released.append("unused"),
    )
    del output
    gc.collect()
    assert released == ["unused"]
    output = collector.attach(
        detach_tree("frozen", torch.tensor(3.0)),
        on_release=lambda: released.append("frozen"),
    )
    assert released == ["unused", "frozen"]
    assert output.item() == 3
    with torch.no_grad():
        collector.attach(
            detach_tree("disabled", torch.tensor(4.0, requires_grad=True)),
            on_release=lambda: released.append("disabled"),
        )
    gc.collect()
    assert released == ["unused", "frozen", "disabled"]


@pytest.mark.parametrize("use_grad", [False, True])
def test_managed_backward_preserves_existing_graph_under_no_grad(use_grad):
    source = torch.tensor([2.0, 3.0], requires_grad=True)
    managed = managed_tensor(source)
    loss = managed.square().sum()
    with torch.no_grad():
        assert not (managed * 2).requires_grad
        if use_grad:
            (gradient,) = torch.autograd.grad(loss, source)
        else:
            loss.backward()
            gradient = source.grad
    torch.testing.assert_close(gradient, 2 * source.detach())


def test_managed_autograd_grad_targets_original_tensor_and_higher_derivatives():
    source = torch.tensor([2.0, 3.0], requires_grad=True)
    managed = managed_tensor(source)
    loss = managed.pow(3).sum()
    (gradient,) = torch.autograd.grad(loss, managed, create_graph=True)
    (second,) = torch.autograd.grad(gradient.sum(), managed)
    torch.testing.assert_close(gradient, 3 * source.detach().square())
    torch.testing.assert_close(second, 6 * source.detach())


def test_managed_hooks_and_retained_gradients_use_original_tensor():
    source = torch.tensor([2.0, 3.0], requires_grad=True)
    managed = managed_tensor(source)
    calls = []
    managed.register_hook(lambda gradient: calls.append(gradient.clone()))
    managed.retain_grad()
    managed.square().sum().backward()
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], 2 * source.detach())
    torch.testing.assert_close(managed.grad, source.grad)


def test_local_autograd_grad_of_managed_bridge_output_does_not_commit():
    collector = CotangentCollector()
    value = collector.attach(
        detach_tree("local-grad", torch.tensor(3.0, requires_grad=True)), managed=True
    )
    loss = value.square()
    (gradient,) = torch.autograd.grad(loss, value, retain_graph=True)
    assert gradient.item() == 6 and collector._pending is None
    (packet,) = collector.backward(loss)
    assert packet.gradients[0] is not None
    assert packet.gradients[0].item() == 6


@pytest.mark.parametrize("managed", [False, True])
def test_output_hooks_change_or_reject_collected_cotangents(managed):
    collector = CotangentCollector()
    value = collector.attach(
        detach_tree("hook", torch.tensor(3.0, requires_grad=True)), managed=managed
    )
    hook = value.register_hook(lambda gradient: gradient * 0)
    (packet,) = collector.backward(value.square(), retain_graph=True)
    torch.testing.assert_close(packet.gradients[0], torch.tensor(0.0))
    hook.remove()

    def reject(gradient):
        raise RuntimeError("hook rejects loss")

    value.register_hook(reject)
    with pytest.raises(RuntimeError, match="hook rejects loss"):
        collector.backward(value.square())
    assert collector._pending is None


@pytest.mark.parametrize("managed", [False, True])
def test_unrelated_failed_backward_cannot_enter_an_active_collection(managed):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    collector = CotangentCollector()
    a = collector.attach(
        detach_tree("a", torch.tensor(2.0, requires_grad=True)), managed=managed
    )
    b = collector.attach(detach_tree("b", torch.tensor(3.0, requires_grad=True)))
    paused, resume = Event(), Event()

    def pause(gradient):
        paused.set()
        assert resume.wait(10)
        return gradient

    def fail_foreign(*args):
        raise RuntimeError("foreign backward failed after collection")

    a.register_hook(pause)
    b.grad_fn.register_hook(fail_foreign)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(collector.backward, a)
        try:
            assert paused.wait(10)
            with pytest.raises(RuntimeError, match="owning trainer.backward"):
                b.backward()
            with pytest.raises(RuntimeError, match="already active"):
                collector.backward(b)
        finally:
            resume.set()
        packets = future.result(timeout=10)
    assert [packet.handle for packet in packets] == ["a"]
    torch.testing.assert_close(packets[0].gradients[0], torch.tensor(1.0))


def test_nested_remote_backward_is_rejected_without_polluting_parent_task():
    collector = CotangentCollector()
    a = collector.attach(detach_tree("a", torch.tensor(2.0, requires_grad=True)))
    b = collector.attach(detach_tree("b", torch.tensor(3.0, requires_grad=True)))
    rejected = []

    def nested(gradient):
        with pytest.raises(RuntimeError, match="nested remote backward"):
            b.backward()
        rejected.append(True)
        return gradient

    a.register_hook(nested)
    packets = collector.backward(a)
    assert rejected == [True]
    assert [packet.handle for packet in packets] == ["a"]


@pytest.mark.parametrize("use_reentrant", [False, True])
@pytest.mark.parametrize("managed", [False, True])
def test_local_checkpoint_recomputation_keeps_collection_task(use_reentrant, managed):
    from torch.utils.checkpoint import checkpoint

    collector = CotangentCollector()
    physical = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    value = collector.attach(detach_tree("checkpoint", physical), managed=managed)
    head = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    output = checkpoint(lambda x: (x * head).sin(), value, use_reentrant=use_reentrant)
    (packet,) = collector.backward(output.sum())
    torch.testing.assert_close(
        packet.gradients[0], (physical.detach() * head.detach()).cos() * head.detach()
    )
    torch.testing.assert_close(
        head.grad, (physical.detach() * head.detach()).cos() * physical.detach()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("destination", ["cpu", "cuda"])
def test_cross_device_assignment_rejects_without_discarding_writes(destination):
    source_device = "cuda" if destination == "cpu" else "cpu"
    dst = torch.zeros(2, device=destination)
    src = managed_tensor(torch.tensor([4.0, 5.0], device=source_device))
    with pytest.raises(RuntimeError, match="__setitem__.*cpu.*cuda.*explicit"):
        dst[:] = src
    torch.testing.assert_close(dst, torch.zeros_like(dst))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_device_stateful_modules_require_explicit_placement():
    inputs = managed_tensor(torch.tensor([[1.0, 3.0], [2.0, 6.0]]))
    batch_norm = torch.nn.BatchNorm1d(2, device="cuda")
    assert batch_norm.running_mean is not None and batch_norm.running_var is not None
    mean, variance = batch_norm.running_mean.clone(), batch_norm.running_var.clone()
    with pytest.raises(RuntimeError, match="batch_norm.*explicit"):
        batch_norm(inputs)
    torch.testing.assert_close(batch_norm.running_mean, mean)
    torch.testing.assert_close(batch_norm.running_var, variance)
    weight = torch.full((2, 2), 4.0, device="cuda")
    tokens = managed_tensor(torch.tensor([0, 1]))
    with pytest.raises(RuntimeError, match="embedding.*explicit"):
        torch.nn.functional.embedding(tokens, weight, max_norm=1)
    torch.testing.assert_close(weight, torch.full_like(weight, 4.0))


def test_same_device_managed_batch_norm_preserves_buffer_updates():
    inputs = torch.tensor([[1.0, 3.0], [2.0, 6.0]], requires_grad=True)
    actual, reference = torch.nn.BatchNorm1d(2), torch.nn.BatchNorm1d(2)
    result = actual(managed_tensor(inputs))
    expected = reference(inputs)
    torch.testing.assert_close(result, expected)
    for name, buffer in actual.named_buffers():
        torch.testing.assert_close(buffer, dict(reference.named_buffers())[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_device_common_loss_and_indexing_are_supported():
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = torch.zeros((2, 2), device="cuda", requires_grad=True)
    output = managed_tensor(source)
    loss = torch.nn.functional.mse_loss(output, target)
    assert loss.device.type == "cpu"
    loss.backward()
    assert target.grad is not None and source.grad is not None
    torch.testing.assert_close(source.grad, source.detach() / 2)
    torch.testing.assert_close(target.grad.cpu(), -source.detach() / 2)
    selected = torch.index_select(output, 0, torch.tensor([1], device="cuda"))
    torch.testing.assert_close(selected, source[1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_collection_task_spans_cpu_and_cuda_bridge_nodes():
    collector = CotangentCollector()
    source = [
        torch.tensor([2.0, 3.0], device=device, requires_grad=True)
        for device in ("cpu", "cuda")
    ]
    outputs = [
        collector.attach(detach_tree(str(i), value), managed=True)
        for i, value in enumerate(source)
    ]
    losses = tuple(value.square().sum() for value in outputs)
    for retain_graph in (True, False):
        packets = collector.backward(losses, retain_graph=retain_graph)
        assert len(packets) == 2
        for packet, value in zip(packets, source, strict=True):
            assert packet.gradients[0] is not None
            assert packet.gradients[0].device == value.device
            torch.testing.assert_close(packet.gradients[0], 2 * value.detach())


def test_managed_requires_grad_mutates_original_identity():
    value = managed_tensor(torch.tensor([2.0, 3.0]))
    returned = value.requires_grad_()
    assert returned is value and value.requires_grad
    value.square().sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([4.0, 6.0]))


@pytest.mark.parametrize("different_count", [False, True])
def test_duplicate_handles_reject_incompatible_output_signatures(different_count):
    collector = CotangentCollector()
    a = collector.attach(detach_tree("duplicate", torch.ones(2, requires_grad=True)))
    source = (
        [torch.ones(2, requires_grad=True), torch.ones(2, requires_grad=True)]
        if different_count
        else torch.ones(3, requires_grad=True)
    )
    b = collector.attach(detach_tree("duplicate", source))
    other_loss = sum(value.sum() for value in b) if different_count else b.sum()
    with pytest.raises(ValueError, match="Incompatible output signatures.*duplicate"):
        collector.backward(a.sum() + other_loss)
    assert collector._pending is None and not collector._signatures


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("under_no_grad", [False, True])
def test_bridged_outputs_require_clone_before_inplace_writes(managed, under_no_grad):
    source = torch.tensor([2.0, 3.0], requires_grad=True)
    collector = CotangentCollector()
    output = collector.attach(detach_tree("readonly", source), managed=managed)
    if under_no_grad:
        with pytest.raises(RuntimeError, match="view.*modified|modified.*view"):
            with torch.no_grad():
                output.mul_(2)
            collector.backward(output.sum())
    else:
        with pytest.raises(RuntimeError, match="view.*modified|modified.*view"):
            output.mul_(2)
    torch.testing.assert_close(source, torch.tensor([2.0, 3.0]))
    writable = collector.attach(
        detach_tree("writable", source), managed=managed
    ).clone()
    writable.mul_(2)
    (packet,) = collector.backward(writable.sum())
    torch.testing.assert_close(packet.gradients[0], torch.full((2,), 2.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("source_device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["to", "type_as"])
def test_explicit_tensor_transfer_specs_preserve_destination_and_gradient(
    source_device, method
):
    destination = "cuda" if source_device == "cpu" else "cpu"
    source = torch.tensor([2.0, 3.0], device=source_device, requires_grad=True)
    spec = torch.empty(0, device=destination, dtype=torch.float64)
    result = getattr(managed_tensor(source), method)(spec)
    assert result.device == spec.device and result.dtype == spec.dtype
    result.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "operator", ["__iand__", "__ior__", "__ixor__", "__ilshift__", "__irshift__"]
)
def test_mixed_device_bitwise_inplace_dunders_reject_without_rebinding(operator):
    target = torch.ones(2, dtype=torch.int64, device="cuda")
    source = managed_tensor(torch.ones(2, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="mutation.*explicit"):
        getattr(target, operator)(source)
    torch.testing.assert_close(target, torch.ones_like(target))
    assert target.device.type == "cuda" and type(target) is torch.Tensor


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_managed_ambiguous_cuda_devices_require_explicit_transfer():
    source = managed_tensor(torch.ones(2, device="cuda:0"))
    other = torch.ones(2, device="cuda:1")
    with pytest.raises(RuntimeError, match="explicit move between accelerator devices"):
        source + other
    moved = source.to(other)
    assert moved.device == other.device


@pytest.mark.parametrize(
    "container", ["set", "object", "tensor_key", "default_factory"]
)
def test_output_packets_reject_opaque_tensor_bearing_metadata(container):
    from collections import defaultdict
    from types import SimpleNamespace

    source = torch.tensor(3.0, requires_grad=True)
    tree = (
        {source}
        if container == "set"
        else SimpleNamespace(value=source)
        if container == "object"
        else defaultdict(lambda: source)
        if container == "default_factory"
        else {source: "key"}
    )
    with pytest.raises(TypeError, match="Unsupported output tree metadata"):
        detach_tree("opaque", tree)


def test_graph_release_callback_does_not_run_at_interpreter_shutdown():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from art.trainer_rank._tensors import CotangentCollector, detach_tree
collector = CotangentCollector()
value = collector.attach(
    detach_tree('alive-at-exit', torch.tensor(2., requires_grad=True)),
    on_release=lambda: print('UNEXPECTED_RELEASE_AT_EXIT', flush=True),
)
print('OUTPUT_REMAINS_ALIVE', flush=True)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "OUTPUT_REMAINS_ALIVE" in result.stdout
    assert "UNEXPECTED_RELEASE_AT_EXIT" not in result.stdout


def test_real_microbatch_packet_roundtrip_preserves_unset_and_backward():
    from art.trainer_rank import ForwardInput, MicroBatch, MicroBatchStats, Unset

    source = torch.tensor([2.0, 3.0], requires_grad=True)
    batch = MicroBatch(
        inputs=[ForwardInput(input_tokens=torch.tensor([1, 2]))],
        outputs=[ForwardOutput(source, None, None, None)],
        indices=[2],
        stats=MicroBatchStats(0, 3, 3, 1, 2, 2, 0, 0, 0, False),
    )
    packet = pickle.loads(pickle.dumps(detach_tree("microbatch", batch)))
    collector = CotangentCollector()
    attached = collector.attach(packet)
    assert isinstance(attached, MicroBatch)
    assert attached.inputs[0].checkpoint is Unset
    assert attached.select(["a", "b", "c"]) == ["c"]
    assert attached.stats == batch.stats
    (gradient,) = collector.backward(attached.outputs[0].target_logprobs.square().sum())
    assert gradient.gradients[0] is None  # input token IDs
    torch.testing.assert_close(gradient.gradients[1], 2 * source.detach())


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cloned_output_inplace_preserves_original_hooks(managed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    collector = CotangentCollector()
    value = collector.attach(
        detach_tree("inplace-hook", torch.ones(2, device=device, requires_grad=True)),
        managed=managed,
    ).clone()
    calls = []

    def zero(gradient):
        calls.append(gradient.clone())
        return gradient * 0

    value.register_hook(zero)
    assert value.mul_(2) is value
    (packet,) = collector.backward(value.sum())
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], torch.full_like(value, 2))
    torch.testing.assert_close(packet.gradients[0], torch.zeros_like(value))


@pytest.mark.parametrize("managed", [False, True])
def test_inplace_transpose_updates_shape_and_gradient_on_original(managed):
    collector = CotangentCollector()
    source = torch.arange(6.0).reshape(2, 3).requires_grad_()
    value = collector.attach(detach_tree("transpose", source), managed=managed).clone()
    assert value.transpose_(0, 1) is value
    assert value.shape == (3, 2)
    weights = torch.arange(1.0, 7.0).reshape(3, 2)
    (packet,) = collector.backward((value * weights).sum())
    torch.testing.assert_close(packet.gradients[0], weights.T)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "conversion", ["device", "tensor", "dtype", "type_as", "host_or_cuda", "contiguous"]
)
def test_noop_conversion_preserves_identity_gradient_queries_and_later_hooks(
    managed, device, conversion
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    collector = CotangentCollector()
    source = torch.tensor([2.0, 3.0], device=device, requires_grad=True)
    value = collector.attach(detach_tree("noop", source), managed=managed)
    loss = value.square().sum()
    with torch.no_grad():
        converted = (
            value.to(value.device)
            if conversion == "device"
            else value.to(source)
            if conversion == "tensor"
            else value.to(value.dtype)
            if conversion == "dtype"
            else value.type_as(source)
            if conversion == "type_as"
            else (value.cpu() if device == "cpu" else value.cuda())
            if conversion == "host_or_cuda"
            else value.contiguous()
        )
    assert converted is value
    hook = converted.register_hook(lambda gradient: gradient * 0)
    (gradient,) = torch.autograd.grad(loss, converted, retain_graph=True)
    torch.testing.assert_close(gradient, torch.zeros_like(source))
    hook.remove()
    (gradient,) = torch.autograd.grad(loss, converted, retain_graph=True)
    torch.testing.assert_close(gradient, 2 * source.detach())
    (packet,) = collector.backward(loss)
    torch.testing.assert_close(packet.gradients[0], 2 * source.detach())


def test_same_device_out_preserves_ordinary_output_identity():
    source = managed_tensor(torch.tensor([2.0, 3.0]))
    destination = torch.empty(2)
    result = torch.add(source, 1, out=destination)
    assert result is destination and type(result) is torch.Tensor
    torch.testing.assert_close(destination, torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("reverse", [False, True])
def test_managed_operands_defer_to_other_tensor_snapshot_dispatch(reverse):
    collector = CotangentCollector()
    source = torch.tensor([2.0, 3.0], requires_grad=True)
    managed = collector.attach(detach_tree("model", source), managed=True)
    captures = []

    class SnapshotParameter(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            tensors, spec = flatten_tensors((args, kwargs or {}))
            snapshot = collector.attach(
                detach_tree("head", torch.tensor([5.0, 7.0], requires_grad=True))
            )
            captures.append(True)
            args, kwargs = unflatten_tensors(
                spec,
                tuple(
                    snapshot if isinstance(tensor, cls) else tensor
                    for tensor in tensors
                ),
            )
            return func(*args, **kwargs)

    # A live proxy's storage is deliberately stale; only its dispatch supplies
    # the current captured head value, as the real client parameter handle does.
    proxy = torch.Tensor._make_subclass(SnapshotParameter, torch.zeros(2))
    loss = (proxy * managed if reverse else managed * proxy).sum()
    assert captures == [True]
    packets = {packet.handle: packet for packet in collector.backward(loss)}
    torch.testing.assert_close(packets["model"].gradients[0], torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(packets["head"].gradients[0], source.detach())


@pytest.mark.parametrize("fail", [False, True])
def test_flatten_releases_tensor_references_without_cyclic_gc(fail):
    import gc
    import weakref

    @dataclass
    class Broken:
        value: int = 0

        def __getattribute__(self, name):
            if name == "value":
                raise RuntimeError("broken dataclass field")
            return object.__getattribute__(self, name)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        source = torch.ones(2, requires_grad=True)
        reference = weakref.ref(source)
        tree = [source, Broken()] if fail else [source]
        if fail:
            with pytest.raises(RuntimeError, match="broken dataclass field"):
                flatten_tensors(tree)
        else:
            leaves, spec = flatten_tensors(tree)
            del leaves, spec
        del tree, source
        assert reference() is None
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("attribute", ["T", "mT", "H", "mH", "real", "imag"])
def test_tensor_properties_keep_managed_placement_and_original_gradients(
    device, attribute
):
    source = torch.tensor(
        [[1 + 2j, 3 - 1j], [2 - 1j, 4 + 3j]],
        dtype=torch.complex128,
        device=device,
        requires_grad=True,
    )
    view = getattr(managed_tensor(source), attribute)
    assert isinstance(view, ManagedTensor)
    operand = torch.full(
        (2, 3),
        2.0,
        dtype=view.dtype,
        device="cuda" if device == "cpu" else "cpu",
        requires_grad=True,
    )
    result = view @ operand
    assert result.device == source.device
    result.abs().square().sum().backward()
    expected_source = source.detach().cpu().requires_grad_()
    expected_operand = operand.detach().cpu().requires_grad_()
    (
        getattr(expected_source, attribute) @ expected_operand
    ).abs().square().sum().backward()
    assert source.grad is not None and operand.grad is not None
    assert source.grad.device == source.device and operand.grad.device == operand.device
    torch.testing.assert_close(source.grad.cpu(), expected_source.grad)
    torch.testing.assert_close(operand.grad.cpu(), expected_operand.grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "layouts", [("dense", "sparse"), ("sparse", "dense"), ("sparse", "sparse")]
)
def test_duplicate_handle_sparse_cotangents_accumulate_in_any_order(device, layouts):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    collector = CotangentCollector()
    packet = detach_tree("mixed-grad", torch.ones(4, device=device, requires_grad=True))
    outputs = [collector.attach(packet) for _ in layouts]
    indices = torch.tensor([0, 2], device=device)
    losses = [
        output.sum()
        if layout == "dense"
        else output.gather(0, indices, sparse_grad=True).sum()
        for output, layout in zip(outputs, layouts, strict=True)
    ]
    (collected,) = collector.backward(sum(losses))
    gradient = collected.gradients[0]
    assert gradient is not None
    if layouts == ("sparse", "sparse"):
        assert gradient.is_sparse
        expected = torch.tensor([2.0, 0.0, 2.0, 0.0], device=device)
    else:
        expected = torch.tensor([2.0, 1.0, 2.0, 1.0], device=device)
    torch.testing.assert_close(gradient.to_dense(), expected)


def test_transformers_dataclass_mapping_packet_preserves_fields_entries_and_aliases():
    from typing import cast

    from transformers.modeling_outputs import BaseModelOutput

    source = cast(torch.FloatTensor, torch.tensor([2.0, 3.0], requires_grad=True))
    physical = BaseModelOutput(
        last_hidden_state=source,
        hidden_states=(source, cast(torch.FloatTensor, source * 2)),
    )
    physical["extra"] = source * 3
    packet = pickle.loads(pickle.dumps(detach_tree("mapping", physical)))
    collector = CotangentCollector()
    output = collector.attach(packet)
    assert isinstance(output, BaseModelOutput)
    assert list(output) == list(physical)
    assert output.last_hidden_state is output["last_hidden_state"] is output[0]
    assert output.hidden_states[0] is output.last_hidden_state
    assert output.attentions is None and "attentions" not in output
    assert output.to_tuple()[-1] is output["extra"]
    (gradients,) = collector.backward(
        output.last_hidden_state.sum() + output["extra"].sum()
    )
    assert gradients.gradients[1] is None
    leaves, _ = flatten_tensors(physical)
    selected = [
        (value, gradient)
        for value, gradient in zip(leaves, gradients.gradients, strict=True)
        if gradient is not None
    ]
    torch.autograd.backward(
        [value for value, _ in selected], [gradient for _, gradient in selected]
    )
    torch.testing.assert_close(source.grad, torch.full_like(source, 4))
