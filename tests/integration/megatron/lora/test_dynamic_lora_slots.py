from __future__ import annotations

from contextlib import contextmanager
import os
import socket
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from megatron.core import parallel_state as ps  # noqa: E402
from torch.distributed import destroy_process_group, init_process_group  # noqa: E402

from art.megatron.lora import LoRA, LoRASlotRef, use_lora_slot  # noqa: E402
from art.trainer_rank import AdamParams, TrainerRank  # noqa: E402


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
