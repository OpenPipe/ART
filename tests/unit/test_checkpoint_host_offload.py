from dataclasses import replace

import pytest
from test_trainer_rank_checkpoint_memory import dense_rank
import torch
from torch.utils.checkpoint import checkpoint

from art.megatron.context_parallel.checkpoint import (
    current_host_activation_offload,
    offload_saved_activations,
)
from art.megatron.lora import LoRASlotRef
from art.trainer_rank import ForwardInput


def test_checkpoint_host_roundtrip_preserves_forward_and_all_gradients_exactly():
    x0 = torch.arange(24, dtype=torch.float64).reshape(6, 1, 4) / 24
    w0 = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 16

    def execute(offload):
        x = x0.clone().requires_grad_()
        weights = [w0.clone().requires_grad_(), w0.T.contiguous().requires_grad_()]

        def forward():
            y = checkpoint(
                lambda value: (value @ weights[0]).sin(), x, use_reentrant=True
            )
            return checkpoint(lambda value: value @ weights[1], y, use_reentrant=True)

        if offload:
            with offload_saved_activations(hidden_size=4, verify=True) as stats:
                output = forward()
            # Unpacking remains valid after the forward-only scope ends.
            output.square().sum().backward()
            assert stats.boundaries == stats.unpacked == stats.verified == 2
            assert stats.saved_bytes == 2 * x.numel() * x.element_size()
        else:
            output = forward()
            output.square().sum().backward()
        return [output.detach(), x.grad, *(weight.grad for weight in weights)]

    reference, actual = execute(False), execute(True)
    assert all(torch.equal(a, b) for a, b in zip(reference, actual, strict=True))
    assert current_host_activation_offload() is None


def test_host_budget_is_priced_from_worst_rank_rows_before_forward():
    rank = dense_rank()
    rank._topology_key = lambda: (1, 1, 4, 1)
    rank._parallel_shape = replace(rank._parallel_shape, cp=4)
    groups = ((128, True),)
    retained = rank._checkpoint_memory_floor(groups)[0]
    with offload_saved_activations(
        hidden_size=rank._hidden_size, max_cpu_bytes=retained - 1
    ):
        with pytest.raises(ValueError, match="before forward"):
            rank._checkpoint_memory_floor(groups)


def test_host_budget_refuses_before_unbounded_storage_and_cleans_scope():
    x = torch.ones((8, 1, 4), requires_grad=True)
    size = x.numel() * x.element_size()
    with pytest.raises(ValueError, match="byte budget exceeded"):
        with offload_saved_activations(hidden_size=4, max_cpu_bytes=size) as stats:
            checkpoint(lambda value: value.sin(), x, use_reentrant=True)
            checkpoint(lambda value: value.cos(), x, use_reentrant=True)
    assert stats.boundaries == 1 and stats.saved_bytes == size
    assert current_host_activation_offload() is None
    with offload_saved_activations(hidden_size=4):
        with pytest.raises(RuntimeError, match="cannot overlap"):
            with offload_saved_activations(hidden_size=4):
                pass
    assert current_host_activation_offload() is None


def test_host_offload_keeps_device_workspace_and_every_pending_adapter_gradient():
    rank = dense_rank()
    rank._topology_key = lambda: (1, 1, 4, 1)
    rank._parallel_shape = replace(rank._parallel_shape, cp=4)
    groups = ((128, True),)
    baseline = rank._checkpoint_memory_floor(groups)
    request = ForwardInput(input_tokens=torch.arange(128), hidden_states=True)
    original = rank._memory_signature_from_requests(
        [request], slot_group_count=1, grad_modes=(True,)
    )
    slot = LoRASlotRef("checkpoint", "host")
    pending = (100,) * (rank._num_layers + 1)
    rank._pending_adapter_gradient_bytes = lambda refs: pending
    with offload_saved_activations(hidden_size=rank._hidden_size):
        floor = rank._checkpoint_memory_floor(groups)
        signature = rank._memory_signature_from_requests(
            [request], slot_group_count=1, grad_modes=(True,)
        )
        assert signature != original
        assert signature.checkpoint_host_offload == (rank._hidden_size, 64 * 2**30)
        assert floor[0] == 128 * rank._hidden_size * 2
        assert floor[0] * rank._num_layers == baseline[0]
        assert floor[1] == baseline[1]
        assert rank._checkpoint_input_gradient_bytes(groups, floor[0]) == floor[0]
        gradient_groups = rank._checkpoint_gradient_groups(groups, (slot,))
        assert rank._checkpoint_adapter_gradient_bytes(gradient_groups) == sum(pending)
    assert current_host_activation_offload() is None


def test_host_offload_rejects_unsupported_runtime_before_admission():
    rank = dense_rank()
    request = ForwardInput(input_tokens=torch.arange(128), hidden_states=True)
    with offload_saved_activations(hidden_size=rank._hidden_size):
        with pytest.raises(ValueError, match="one native dense CP uniform root"):
            rank._memory_signature_from_requests(
                [request], slot_group_count=1, grad_modes=(True,)
            )
