from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from typing import cast

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from megatron.core import parallel_state as ps  # noqa: E402
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from art.megatron.lora import LoRA, LoRASlotRef, use_lora_slot  # noqa: E402
from art.trainer_rank import (  # noqa: E402
    AdamParams,
    TrainerRank,
    TrainerRankSlotStateError,
)
from art.trainer_rank._checkpoint import (  # noqa: E402
    LocalOptimizerState,
    OptimizerConfig,
    _commit_slot,
)
from art.trainer_rank._impl import (  # noqa: E402
    _CheckpointSlot,
    _distributed_grad_norm,
    _vocab_parallel_log_z,
    _vocab_parallel_target_logprobs,
    _vocab_parallel_topk_from_local,
)


class _CudaValueHead(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(hidden_size, 1)

    def score(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"value": self.projection(hidden)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_dynamic_lora_slots_capture_recompute_context_and_step_independently() -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        lora = LoRA(
            "dense",
            in_features=4,
            out_features=5,
            rank=2,
            alpha=32,
            dtype=torch.float32,
            device=device,
        )
        ref_a = LoRASlotRef("checkpoint", "A")
        ref_b = LoRASlotRef("checkpoint", "B")
        lora.load_lora_slot(
            ref_a, _adapter("dense", rank=1, seed=1), requires_grad=True
        )
        lora.load_lora_slot(
            ref_b, _adapter("dense", rank=4, seed=2), requires_grad=True
        )

        x = torch.randn(7, 4, device=device)
        with use_lora_slot(LoRASlotRef("checkpoint", None)):
            assert torch.equal(lora(x), torch.zeros(7, 5, device=device))
        with use_lora_slot(LoRASlotRef("lora", "missing")):
            assert torch.equal(lora(x), torch.zeros(7, 5, device=device))

        slot_a = lora._slot(ref_a)
        assert slot_a is not None
        with use_lora_slot(ref_a):
            actual = lora(x)
        expected = (x @ slot_a.A_T) @ slot_a.B_T * slot_a.scale
        assert torch.allclose(actual, expected, atol=0, rtol=0)
        assert slot_a.rank == 1
        assert slot_a.scale == 32.0
        slot_b = lora._slot(ref_b)
        assert slot_b is not None
        assert slot_b.scale == 8.0

        trainer = _trainer_for(lora, device)
        cpu_adapter = {
            key: value.cpu().double()
            for key, value in _adapter("dense", rank=3, seed=7).items()
        }
        _install_checkpoint(trainer, "CPU", cpu_adapter)
        cpu_slot = lora._slot(LoRASlotRef("checkpoint", "CPU"))
        assert cpu_slot is not None
        assert cpu_slot.A_T.device == lora.A_T.device
        assert cpu_slot.A_T.dtype == lora.A_T.dtype
        with use_lora_slot(LoRASlotRef("checkpoint", "CPU")):
            assert lora(x).is_cuda

        with trainer.push_checkpoint("A"):
            assert trainer._slot_stack[-1] == ref_a
            with trainer.push_checkpoint(None):
                assert trainer._slot_stack[-1].name is None
            assert trainer._slot_stack[-1] == ref_a
        assert trainer._slot_stack == []

        from megatron.core.tensor_parallel.random import (
            checkpoint as megatron_checkpoint,
        )
        from torch.utils.checkpoint import checkpoint as torch_checkpoint

        _assert_checkpoint_recomputes_with(ref_a, ref_b, lora, torch_checkpoint)
        _assert_checkpoint_recomputes_with(
            ref_a, ref_b, lora, megatron_checkpoint, False
        )
        _assert_step_updates_only(ref_a, ref_b, lora, trainer)
        _assert_reload_replaces_slot_optimizer(ref_a, lora, trainer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_trainer_rank_custom_objects_train_and_become_stale_on_cuda() -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=1))
        head = trainer.module(
            "value_head", lambda: _CudaValueHead(4).to(device), checkpoint="A"
        )
        gain = trainer.parameter(
            "gain", lambda: torch.ones((), device=device), checkpoint="A"
        )
        running = trainer.buffer(
            "running", lambda: torch.tensor(0.25, device=device), checkpoint="A"
        )

        output = head.score(torch.randn(3, 4, device=device))["value"] * gain + running
        with pytest.raises(TrainerRankSlotStateError, match="live backward graph"):
            trainer._guard_slot_can_load(trainer._slot_ref("A"))
        output.sum().backward()
        before = tuple(param.detach().clone() for param in head.parameters()) + (
            gain.detach().clone(),
        )
        trainer.optim_step(
            params=AdamParams(learning_rate=1e-3, weight_decay=0.0),
            checkpoints=["A"],
        )
        after = tuple(head.parameters()) + (gain,)
        assert all(
            not torch.equal(old, new) for old, new in zip(before, after, strict=True)
        )
        torch.testing.assert_close(running, torch.tensor(0.25, device=device))

        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=2))
        for operation in (
            lambda: head.score(torch.ones(1, 4, device=device)),
            lambda: gain + 1,
            lambda: running + 1,
        ):
            with pytest.raises(TrainerRankSlotStateError, match="stale"):
                operation()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")
def test_checkpoint_reload_does_not_alias_the_next_slot() -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        first = LoRA("first", 4, 5, 2, 32, torch.float32, device)
        second = LoRA("second", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(first, device)
        trainer.runtime.model = [torch.nn.Sequential(first, second)]

        def adapter(
            seed: int, *, include_second: bool = True
        ) -> dict[str, torch.Tensor]:
            return _adapter("first", rank=2, seed=seed) | (
                _adapter("second", rank=2, seed=seed + 10) if include_second else {}
            )

        def stage(destination: str, state: dict[str, torch.Tensor]) -> None:
            temporary = f"temporary-{destination}"
            trainer._load_checkpoint_slot(temporary, state, alpha=32.0)
            _commit_slot(trainer, temporary, destination)

        stage("A", adapter(1))
        stage("B", adapter(2))
        slot_b = second._slot(trainer._slot_ref("B"))
        assert slot_b is not None
        expected_b = slot_b.A_T.detach().clone()
        stage("A", adapter(3, include_second=False))
        stage("C", adapter(4))

        slot_b = second._slot(trainer._slot_ref("B"))
        slot_c = second._slot(trainer._slot_ref("C"))
        assert slot_b is not None and slot_c is not None and slot_b is not slot_c
        torch.testing.assert_close(slot_b.A_T, expected_b)


@pytest.mark.parametrize("tp_size", (2, 4))
def test_trainer_rank_tp_head_backward_matches_unsharded_oracle(
    tp_size: int,
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < tp_size:
        pytest.skip(f"requires {tp_size} CUDA devices")
    init_file = tmp_path / f"tp_head_{tp_size}"
    mp.spawn(
        _tp_head_backward_worker,
        args=(tp_size, f"file://{init_file}"),
        nprocs=tp_size,
        join=True,
    )


@pytest.mark.parametrize(
    ("topology", "world"),
    (("dp", 2), ("tp", 2), ("cp", 2), ("tp_cp", 4)),
)
def test_trainer_rank_custom_parameter_reduction_oracle(
    topology: str,
    world: int,
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} CUDA devices")
    init_file = tmp_path / f"custom_parameter_{topology}"
    mp.spawn(
        _custom_parameter_reduction_worker,
        args=(world, f"file://{init_file}", topology),
        nprocs=world,
        join=True,
    )


def _custom_parameter_reduction_worker(
    rank: int,
    world: int,
    init_method: str,
    topology: str,
) -> None:
    torch.cuda.set_device(rank)
    init_process_group("nccl", rank=rank, world_size=world, init_method=init_method)
    try:
        tp_size = 2 if topology == "tp_cp" else world if topology == "tp" else 1
        cp_size = 2 if topology == "tp_cp" else world if topology == "cp" else 1
        ps.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=1,
        )
        device = torch.device("cuda", rank)
        ref = LoRASlotRef("checkpoint", "A")
        lora = LoRA("dense", 1, 1, 1, 1, torch.float32, device)
        lora.load_lora_slot(
            ref,
            {
                "dense.lora_A.weight": torch.ones(1, 1, device=device),
                "dense.lora_B.weight": torch.ones(1, 1, device=device),
            },
            requires_grad=True,
        )
        trainer = _trainer_for(lora, device)
        parameter = trainer.parameter(
            "gain",
            lambda: torch.tensor(float(rank + 1), device=device),
            checkpoint="A",
        )
        torch.testing.assert_close(parameter, torch.tensor(1.0, device=device))
        (parameter * float(rank + 1)).backward()
        (reduced,) = trainer._reduce_dynamic_grads((parameter,), scale_grads=1.0)
        expected = {"dp": 3.0, "tp": 1.5, "cp": 3.0, "tp_cp": 5.0}[topology]
        torch.testing.assert_close(reduced, torch.tensor(expected, device=device))
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _tp_head_backward_worker(rank: int, world: int, init_method: str) -> None:
    torch.cuda.set_device(rank)
    init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        init_method=init_method,
    )
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=world,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        device = torch.device("cuda", rank)
        full = torch.tensor(
            [
                [-1.2, 0.4, 2.1, -0.7, 1.3, 0.2, -2.0, 0.8],
                [0.1, -0.5, 1.7, 0.3, -1.1, 2.4, 0.9, -0.2],
            ],
            device=device,
        )
        local_size = int(full.shape[1]) // world
        local = _local_shard(full, rank, local_size)
        labels = torch.tensor([2, 5], device=device)
        rows = torch.arange(int(full.shape[0]), device=device)
        actual = _vocab_parallel_target_logprobs(
            local,
            labels,
            _vocab_parallel_log_z(local),
            row_offsets=rows,
        )
        (-actual.sum()).backward()

        reference = full.detach().clone().requires_grad_()
        (-torch.log_softmax(reference, dim=-1)[rows, labels].sum()).backward()
        torch.testing.assert_close(
            local.grad,
            reference.grad[:, rank * local_size : (rank + 1) * local_size],
            atol=1e-6,
            rtol=1e-6,
        )

        local = _local_shard(full, rank, local_size)
        local_values, local_tokens = torch.topk(local, k=min(2, local_size), dim=-1)
        actual_topk = _vocab_parallel_topk_from_local(
            local_values,
            local_tokens,
            k=2,
            log_z=_vocab_parallel_log_z(local),
            vocab_start=rank * local_size,
        )
        (-actual_topk.logprobs.sum()).backward()

        reference = full.detach().clone().requires_grad_()
        reference_values, reference_tokens = torch.topk(
            torch.log_softmax(reference, dim=-1), k=2, dim=-1
        )
        (-reference_values.sum()).backward()
        torch.testing.assert_close(actual_topk.tokens, reference_tokens)
        torch.testing.assert_close(
            local.grad,
            reference.grad[:, rank * local_size : (rank + 1) * local_size],
            atol=1e-6,
            rtol=1e-6,
        )

        from megatron.core import tensor_parallel

        local_hidden = torch.randn(2, 1, 3, device=device, requires_grad=True)
        gathered_hidden = tensor_parallel.gather_from_sequence_parallel_region(
            local_hidden,
            tensor_parallel_output_grad=False,
            group=ps.get_tensor_model_parallel_group(check_initialized=False),
        ).squeeze(1)
        gathered_hidden.sum().backward()
        torch.testing.assert_close(local_hidden.grad, torch.ones_like(local_hidden))

        replicated = _grad_param(rank, device, sharded=False, sync_op="sum")
        sharded = _grad_param(rank, device, sharded=True)
        trainer = TrainerRank.__new__(TrainerRank)
        reduced = trainer._reduce_dynamic_grads((replicated, sharded), scale_grads=0.5)
        expected_replicated = 0.5 * sum(range(1, world + 1))
        torch.testing.assert_close(
            reduced[0], torch.tensor([expected_replicated], device=device)
        )
        torch.testing.assert_close(
            reduced[1], torch.tensor([0.5 * (rank + 1)], device=device)
        )
        norm = _distributed_grad_norm(
            (replicated, sharded),
            reduced,
        )
        expected_norm = (
            expected_replicated**2
            + sum(float((0.5 * i) ** 2) for i in range(1, world + 1))
        ) ** 0.5
        assert norm == pytest.approx(expected_norm, rel=1e-6)

        _assert_replica_grad_reduction(rank, world, context_parallel=True)
        _assert_replica_grad_reduction(rank, world, context_parallel=False)
        _assert_distributed_optimizer_restore(device)
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _assert_replica_grad_reduction(
    rank: int,
    world: int,
    *,
    context_parallel: bool,
) -> None:
    ps.destroy_model_parallel()
    torch.distributed.barrier()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=world if context_parallel else 1,
        expert_model_parallel_size=1,
    )
    device = torch.device("cuda", rank)
    param = _grad_param(rank, device, sharded=False)

    trainer = TrainerRank.__new__(TrainerRank)
    (reduced,) = trainer._reduce_dynamic_grads((param,), scale_grads=0.25)
    expected = 0.25 * sum(range(1, world + 1))
    torch.testing.assert_close(reduced, torch.tensor([expected], device=device))
    assert _distributed_grad_norm((param,), (reduced,)) == pytest.approx(expected)


def _assert_distributed_optimizer_restore(device: torch.device) -> None:
    ref = LoRASlotRef("checkpoint", "A")
    adapter = _adapter("dense", rank=2, seed=11)
    lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
    lora.load_lora_slot(ref, adapter, requires_grad=True)
    trainer = _trainer_for(lora, device)
    params = AdamParams(learning_rate=1e-3, weight_decay=0.0, grad_clip_norm=0.0)
    x = torch.randn(3, 4, device=device)

    with use_lora_slot(ref):
        lora(x).sum().backward()
    trainer.optim_step(params=params, checkpoints=["A"])
    state = _optimizer_state(trainer, "A")
    slot = lora._slot(ref)
    assert slot is not None
    adapter = {
        "dense.lora_A.weight": slot.A_T.detach().T.contiguous(),
        "dense.lora_B.weight": slot.B_T.detach().T.contiguous(),
    }
    with use_lora_slot(ref):
        lora(x).sum().backward()
    trainer.optim_step(params=params, checkpoints=["A"])

    restored_lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
    restored = _trainer_for(restored_lora, device)
    _install_checkpoint(restored, "A", adapter)
    restored._checkpoint_slots["A"].optimizer = restored._restore_canonical_optimizer(
        "A", state
    )
    with use_lora_slot(ref):
        restored_lora(x).sum().backward()
    restored.optim_step(params=params, checkpoints=["A"])
    for expected, actual in zip(
        lora.lora_slot_params(ref), restored_lora.lora_slot_params(ref), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _local_shard(full: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    return full[:, rank * size : (rank + 1) * size].clone().requires_grad_()


def _grad_param(
    rank: int, device: torch.device, *, sharded: bool, sync_op: str = "none"
) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.ones(1, device=device))
    param.allreduce = True  # type: ignore[attr-defined]
    param.lora_shard_domain = "tp"  # type: ignore[attr-defined]
    param.lora_tp_sharded = sharded  # type: ignore[attr-defined]
    param.grad_sync_domain = "tp_default"  # type: ignore[attr-defined]
    param.grad_sync_op = sync_op  # type: ignore[attr-defined]
    param.grad = torch.tensor([float(rank + 1)], device=device)
    return param


def _adapter(prefix: str, *, rank: int, seed: int) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        f"{prefix}.lora_A.weight": torch.randn(
            rank, 4, generator=generator, device=device
        ),
        f"{prefix}.lora_B.weight": torch.randn(
            5, rank, generator=generator, device=device
        ),
    }


def _assert_checkpoint_recomputes_with(
    expected_ref: LoRASlotRef,
    ambient_ref: LoRASlotRef,
    lora: LoRA,
    checkpoint,
    *checkpoint_args,
) -> None:
    for param in lora.parameters():
        param.grad = None
    x = torch.randn(3, 4, device="cuda", requires_grad=True)
    with use_lora_slot(expected_ref):
        y = checkpoint(lambda t: lora(t), *checkpoint_args, x)
    with use_lora_slot(ambient_ref):
        y.sum().backward()
    expected_slot = lora._slot(expected_ref)
    ambient_slot = lora._slot(ambient_ref)
    assert expected_slot is not None
    assert ambient_slot is not None
    assert expected_slot.A_T.grad is not None
    assert ambient_slot.A_T.grad is None


def _assert_step_updates_only(
    stepped_ref: LoRASlotRef,
    frozen_ref: LoRASlotRef,
    lora: LoRA,
    trainer: TrainerRank,
) -> None:
    for param in lora.parameters():
        param.grad = None
    with use_lora_slot(stepped_ref):
        lora(torch.randn(5, 4, device="cuda")).sum().backward()
    before_stepped = [p.detach().clone() for p in lora.lora_slot_params(stepped_ref)]
    before_frozen = [p.detach().clone() for p in lora.lora_slot_params(frozen_ref)]
    trainer.optim_step(
        params=AdamParams(learning_rate=1e-3, weight_decay=0.0, grad_clip_norm=1.0),
        checkpoints=[stepped_ref.name or ""],
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            before_stepped, lora.lora_slot_params(stepped_ref), strict=True
        )
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            before_frozen, lora.lora_slot_params(frozen_ref), strict=True
        )
    )


def _assert_reload_replaces_slot_optimizer(
    ref: LoRASlotRef,
    lora: LoRA,
    trainer: TrainerRank,
) -> None:
    assert ref.name is not None
    old_params = trainer._checkpoint_slots[ref.name].params
    assert trainer._checkpoint_slots[ref.name].optimizer is not None

    _install_checkpoint(trainer, ref.name, _adapter("dense", rank=3, seed=9))

    new_params = trainer._checkpoint_slots[ref.name].params
    assert trainer._checkpoint_slots[ref.name].optimizer is None
    assert [tuple(param.shape) for param in new_params] == [(4, 3), (3, 5)]
    assert all(old is not new for old, new in zip(old_params, new_params, strict=True))
    slot = lora._slot(ref)
    assert slot is not None
    assert slot.rank == 3


def _install_checkpoint(
    trainer: TrainerRank, name: str, adapter: dict[str, torch.Tensor]
) -> int:
    loaded = trainer._load_checkpoint_slot(name, adapter, alpha=32.0)
    previous = trainer._checkpoint_slots.get(name)
    trainer._checkpoint_slots[name] = _CheckpointSlot(
        tuple(trainer._iter_slot_parameters(trainer._slot_ref(name))),
        revision=0 if previous is None else previous.revision + 1,
    )
    return loaded


def _optimizer_state(trainer: TrainerRank, name: str) -> LocalOptimizerState:
    dynamic = trainer._checkpoint_slots[name].optimizer
    assert dynamic is not None
    states = [
        cast(dict[str, torch.Tensor], dynamic.optimizer.state[master])
        for master in dynamic.master_params
    ]
    group = dynamic.optimizer.param_groups[0]
    beta1, beta2 = cast(tuple[float, float], group["betas"])
    return LocalOptimizerState(
        masters=tuple(
            master.detach().cpu().clone() for master in dynamic.master_params
        ),
        exp_avgs=tuple(state["exp_avg"].detach().cpu().clone() for state in states),
        exp_avg_sqs=tuple(
            state["exp_avg_sq"].detach().cpu().clone() for state in states
        ),
        steps=tuple(float(state["step"].item()) for state in states),
        config=OptimizerConfig(
            learning_rate=float(group["lr"]),
            beta1=beta1,
            beta2=beta2,
            eps=float(group["eps"]),
            weight_decay=float(group["weight_decay"]),
        ),
    )


def _trainer_for(lora: LoRA, device: torch.device) -> TrainerRank:
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        model=[lora],
        optimizer=None,
        model_support_handler=SimpleNamespace(
            canonicalize_loaded_lora_state=lambda state, _model: state,
            zero_internal_padding_grads=lambda _model: None,
            zero_internal_padding_params=lambda _model: None,
        ),
    )
    trainer.device = device
    trainer._slot_stack = []
    trainer._default_slot_ref = None
    trainer._checkpoint_slots = {
        name: _CheckpointSlot(
            tuple(lora.lora_slot_params(LoRASlotRef("checkpoint", name)))
        )
        for name in ("A", "B")
    }
    trainer._checkpoint_prefetches = {}
    trainer._checkpoint_mutation_tail = None
    return trainer


@contextmanager
def _single_rank_model_parallel():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    init_process_group("nccl", rank=0, world_size=1)
    try:
        ps.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        yield
    finally:
        if getattr(ps, "model_parallel_is_initialized", lambda: False)():
            ps.destroy_model_parallel()
        destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
