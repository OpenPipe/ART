"""Commands cross ranks through CPU storage, independent of CUDA ordinals."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast
import weakref

import pytest
import torch
import torch.multiprocessing as mp
from trainer_rank_test_support import gloo_group

from art.trainer_rank import ForwardInput, ForwardOutput, TrainerRank
from art.trainer_rank._commands import _Command, _encode_command, _Executor
from art.trainer_rank._heads import HeadRegistration
from art.trainer_rank._impl import _CheckpointSlot
from art.trainer_rank._tensors import managed_tensor


def _payload(device: str) -> Any:
    # Local types and closures exercise CloudPickler inside Torch's persistent
    # storage protocol, including tensors that a container walk cannot find.
    @dataclass
    class Payload:
        base: torch.Tensor
        view: torch.Tensor
        module: Any
        parameter: torch.nn.Parameter
        subclass: torch.Tensor
        managed: torch.Tensor
        captured: Any

    class TaggedTensor(torch.Tensor):
        pass

    base = torch.arange(6, device=device, dtype=torch.float64, requires_grad=True)
    captured = base[1::2]

    class Head(torch.nn.Module):
        offset: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, device=device))
            self.tied = self.weight
            self.register_buffer("offset", base.detach()[::2])
            self.register_buffer("alias", self.offset)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * self.weight + self.offset

    module = Head()

    def hook(_module: Any, _args: Any, output: torch.Tensor) -> torch.Tensor:
        # Module.to moves registered state, as in ordinary eager PyTorch. Hooks
        # that capture constants must explicitly follow the output placement.
        return output + captured.to(output)

    module.register_forward_hook(hook)
    return Payload(
        base,
        captured,
        module,
        module.weight,
        base.detach().as_subclass(TaggedTensor),
        managed_tensor(base.detach()),
        lambda: captured,
    )


def _check_payload(value: Any) -> None:
    assert value.base.device.type == value.view.device.type == "cpu"
    assert value.base.requires_grad and value.view.requires_grad
    assert value.base.dtype == torch.float64
    assert value.view.shape == (3,) and value.view.stride() == (2,)
    assert value.view.storage_offset() == 1
    assert value.view.untyped_storage() is value.base.untyped_storage()
    assert value.captured() is value.view
    assert value.module.weight is value.module.tied is value.parameter
    assert value.parameter.device.type == "cpu" and value.parameter.requires_grad
    assert value.module.offset is value.module.alias
    assert not value.module.offset.requires_grad
    assert value.module.offset.untyped_storage() is value.base.untyped_storage()
    assert type(value.subclass).__name__ == "TaggedTensor"
    assert value.subclass.device.type == value.managed.device.type == "cpu"
    assert not value.subclass.requires_grad and not value.managed.requires_grad
    assert value.subclass.untyped_storage() is value.base.untyped_storage()
    assert value.managed.untyped_storage() is value.base.untyped_storage()
    torch.testing.assert_close(
        value.module(torch.ones(3)), torch.tensor([2.0, 6.0, 10.0], dtype=torch.float64)
    )


def test_command_codec_preserves_nested_types_aliases_and_closure_storage() -> None:
    from test_trainer_rank_commands import _Rank

    executor = _Executor(cast(Any, _Rank()), "zero")
    source = _payload("cpu")
    result = executor._decode(_encode_command(_Command(1, "test", (source,), {}, True)))
    assert result.operation == "test" and result.grad_enabled
    _check_payload(result.args[0])
    assert result.args[0].base.untyped_storage() is not source.base.untyped_storage()


_decoded_refs: list[weakref.ReferenceType[torch.Tensor]] = []


def _remember_restore(tensor: torch.Tensor) -> None:
    _decoded_refs.append(weakref.ref(tensor))


class _Remember:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __reduce__(self):
        return _remember_restore, (self.tensor,)


def _transport_worker(physical: int, rendezvous: str, cuda: bool) -> None:
    from test_trainer_rank_commands import _BadRestore
    from test_trainer_rank_custom_tensors import _config, _runtime

    # Deliberately reverse physical rank and CUDA ordinal, as real actors can.
    device = torch.device(f"cuda:{1 - physical}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(device)
    with gloo_group(physical, f"file://{rendezvous}"):
        ps = SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: physical,
            get_context_parallel_rank=lambda: 0,
        )
        core, megatron = ModuleType("megatron.core"), ModuleType("megatron")
        setattr(core, "parallel_state", ps)
        setattr(megatron, "core", core)
        sys.modules.update({"megatron": megatron, "megatron.core": core})
        runtime = _runtime(torch.nn.Linear(1, 1).to(device))
        runtime.rank, runtime.world_size = physical, 2
        rank: Any = TrainerRank(runtime)
        rank._dp_rank_and_size = lambda: (0, 1)
        rank._checkpoint_slots["student"] = _CheckpointSlot(config=cast(Any, _config()))
        executor = _Executor(rank, "zero")
        source = _payload(str(device)) if physical == 0 else None
        foreign_before = torch.cuda.memory_allocated(physical) if cuda else 0

        def broadcast(sequence: int, operation: str, *args: Any) -> _Command:
            command = _Command(sequence, operation, args, {}, True)
            return executor._broadcast(command if physical == 0 else None)

        result = broadcast(1, "test", source)
        assert result.operation == "test", result.args
        _check_payload(result.args[0])
        if cuda:
            assert torch.cuda.memory_allocated(physical) == foreign_before

        # Real registration handlers put CPU-decoded modules and Parameters on
        # each native rank's own model device, preserving tied state and hooks.
        kinds: tuple[Literal["module", "parameter"], ...] = ("module", "parameter")
        for index, kind in enumerate(kinds, start=2):
            value = None if source is None else getattr(source, kind)
            command = broadcast(
                index,
                "head",
                "head_register",
                HeadRegistration("student", kind, kind, cast(Any, value)),
            )
            assert command.operation == "head", command.args
            executor._dispatch(command)
            registered = rank._checkpoint_slots["student"].custom[kind].value
            if kind == "module":
                assert registered.weight.device == registered.offset.device == device
                assert registered.weight is registered.tied
                assert registered.offset is registered.alias
                torch.testing.assert_close(
                    registered(torch.ones(3, device=device)),
                    torch.tensor([2.0, 6.0, 10.0], device=device, dtype=torch.float64),
                )
            else:
                assert registered.device == device and registered.requires_grad

        # Exercise the public ForwardInput command shape and an actual forward
        # dispatch; the lightweight kernel follows native per-rank placement.
        def forward(inputs: ForwardInput) -> ForwardOutput:
            assert inputs.input_tokens.device.type == "cpu"
            values = inputs.input_tokens.to(device=device, dtype=torch.float32)
            return ForwardOutput(None, None, None, values * runtime.model[0].weight)

        rank.forward = forward
        inputs = ForwardInput(input_tokens=torch.tensor([1, 2, 3], device=device))
        command = broadcast(4, "forward", inputs)
        executor._dispatch(command)
        assert executor.state.graphs
        assert all(
            t.device == device for ts in executor.state.graphs.values() for t in ts
        )

        # A peer-local restore exception must be coordinated before dispatch,
        # release partially decoded tensors, and leave the next command usable.
        before = set(executor.state.graphs)
        bad = broadcast(
            5, "forward", _Remember(torch.ones(5, device=device)), _BadRestore()
        )
        assert bad.operation == "error"
        assert "peer deserialization failure" in bad.args[0]
        with pytest.raises(RuntimeError, match="deserialization failed"):
            executor._execute(bad)
        gc.collect()
        assert _decoded_refs and all(ref() is None for ref in _decoded_refs)
        assert set(executor.state.graphs) == before
        assert (
            broadcast(6, "test", torch.ones(2, device=device)).args[0].device.type
            == "cpu"
        )
        if cuda:
            assert torch.cuda.memory_allocated(physical) == foreign_before


@pytest.mark.parametrize("cuda", [False, True], ids=["cpu", "reversed-cuda-indices"])
def test_commands_decode_on_cpu_and_native_handlers_place_locally(
    tmp_path, cuda: bool
) -> None:
    if cuda and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    mp.spawn(_transport_worker, args=(str(tmp_path / "init"), cuda), nprocs=2)
