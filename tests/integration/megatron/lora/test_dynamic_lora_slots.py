from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import socket
from types import SimpleNamespace

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
    _distributed_grad_norm,
    _vocab_parallel_log_z,
    _vocab_parallel_target_logprobs,
    _vocab_parallel_topk_from_local,
)


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
        assert lora._slot(ref_b).scale == 8.0  # type: ignore[union-attr]

        trainer = _trainer_for(lora, device)
        cpu_adapter = {
            key: value.cpu().double()
            for key, value in _adapter("dense", rank=3, seed=7).items()
        }
        trainer.load_checkpoint_slot("CPU", cpu_adapter)
        cpu_slot = lora._slot(LoRASlotRef("checkpoint", "CPU"))
        assert cpu_slot is not None
        assert cpu_slot.A_T.device == lora.A_T.device
        assert cpu_slot.A_T.dtype == lora.A_T.dtype
        with use_lora_slot(LoRASlotRef("checkpoint", "CPU")):
            assert lora(x).is_cuda

        with trainer.push_checkpoint("A"):
            assert trainer._slot_stack[-1] == ref_a
            with trainer.push_lora(None):
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
    state = trainer.checkpoint_slot_optimizer_state("A")
    assert state is not None
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
    restored.load_checkpoint_slot("A", adapter, optimizer_state=state)
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
    assert lora._slot(expected_ref).A_T.grad is not None  # type: ignore[union-attr]
    assert lora._slot(ambient_ref).A_T.grad is None  # type: ignore[union-attr]


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
    old_params = trainer._checkpoint_slot_params_by_name[ref.name]
    assert ref.name in trainer._dynamic_optimizers

    trainer.load_checkpoint_slot(ref.name, _adapter("dense", rank=3, seed=9))

    new_params = trainer._checkpoint_slot_params_by_name[ref.name]
    assert ref.name not in trainer._dynamic_optimizers
    assert [tuple(param.shape) for param in new_params] == [(4, 3), (3, 5)]
    assert all(old is not new for old, new in zip(old_params, new_params, strict=True))
    assert lora._slot(ref).rank == 3  # type: ignore[union-attr]


def _trainer_for(lora: LoRA, device: torch.device) -> TrainerRank:
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(model=[lora], optimizer=None)
    trainer.device = device
    trainer._slot_stack = []
    trainer._default_slot_ref = None
    trainer._dynamic_optimizers = {}
    trainer._checkpoint_slot_params_by_name = {
        "A": tuple(lora.lora_slot_params(LoRASlotRef("checkpoint", "A"))),
        "B": tuple(lora.lora_slot_params(LoRASlotRef("checkpoint", "B"))),
    }
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
