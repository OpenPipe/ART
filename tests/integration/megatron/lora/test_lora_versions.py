from __future__ import annotations

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from art.megatron.lora import LoRA, LoRASlotRef, use_lora_slot  # noqa: E402
from art.trainer_rank import TrainerRankSlotStateError  # noqa: E402
from art.trainer_rank._checkpoint import (  # noqa: E402
    discard_snapshot_checkpoint,
    snapshot_checkpoint,
)

from .test_dynamic_lora_slots import (  # noqa: E402
    _adapter,
    _install_checkpoint,
    _single_rank_model_parallel,
    _trainer_for,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("train_a", (False, True))
def test_native_capture_preserves_independent_parameter_trainability(
    train_a: bool,
) -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=1))
        ref = LoRASlotRef("checkpoint", "A")
        current = lora._slot(ref)
        assert current is not None
        current.A_T.requires_grad_(train_a)
        current.B_T.requires_grad_(not train_a)
        x = torch.ones(2, 4, device=device)
        with use_lora_slot(ref):
            lora(x).sum().backward()
        expected = [
            None if parameter.grad is None else parameter.grad.clone()
            for parameter in (current.A_T, current.B_T)
        ]
        trainer.zero_grad()
        capture = trainer._capture_lora_version(ref)
        assert capture is not None
        captured = capture.slots[id(lora)]
        assert (captured.A_T.requires_grad, captured.B_T.requires_grad) == (
            train_a,
            not train_a,
        )
        with use_lora_slot(ref, version=capture):
            output = lora(x)
        with trainer._gradient_transaction():
            output.sum().backward()
        for parameter, gradient in zip(
            (current.A_T, current.B_T), expected, strict=True
        ):
            if gradient is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, gradient)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_version_storage_reuse_accounting_and_checkpoint_lifetime() -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=1))
        ref = LoRASlotRef("checkpoint", "A")
        expected_bytes = (4 * 2 + 2 * 5) * 4
        custom = torch.nn.Parameter(torch.ones(100, device=device))
        setattr(custom, "_art_custom_checkpoint_param", True)
        trainer._checkpoint_slots["A"].params += (custom,)
        assert trainer._lora_version_capture_bytes(ref) == expected_bytes
        custom_bytes = custom.numel() * custom.element_size()
        assert trainer._lora_gradient_staging_bytes(ref) == 3 * (
            expected_bytes + custom_bytes
        )
        capture = trainer._capture_lora_version(ref)
        assert capture is not None and capture.nbytes == expected_bytes
        assert trainer._capture_lora_version(ref) is capture
        assert trainer._lora_version_capture_bytes(ref) == 0
        assert all(
            old is not current
            for old in capture.slots[id(lora)].parameters()
            for current in lora.parameters()
        )
        from torch.utils.checkpoint import checkpoint

        with use_lora_slot(ref, version=capture):
            output = checkpoint(
                lora, torch.ones(2, 4, device=device), use_reentrant=False
            )
        reference = weakref.ref(capture)
        del capture
        gc.collect()
        assert reference() is not None
        trainer._checkpoint_slots["A"].revision += 1
        assert trainer._lora_version_capture_bytes(ref) == expected_bytes
        with trainer._gradient_transaction():
            output.sum().backward()
        assert (
            trainer._lora_gradient_staging_bytes(ref)
            == 2 * expected_bytes + 3 * custom_bytes
        )
        del output
        gc.collect()
        assert reference() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_capture_rejects_staleness_and_checkpoint_replacement() -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=1))
        ref = LoRASlotRef("checkpoint", "A")
        capture = trainer._capture_lora_version(ref, max_gradient_staleness=0)
        with use_lora_slot(ref, version=capture):
            output = lora(torch.ones(2, 4, device=device))
        trainer._checkpoint_slots["A"].revision += 1
        with pytest.raises(TrainerRankSlotStateError, match="staleness 1"):
            with trainer._gradient_transaction():
                output.sum().backward(retain_graph=True)
        assert all(p.grad is None for p in trainer._checkpoint_slots["A"].params)
        trainer._checkpoint_slots["A"].generation += 1
        trainer._checkpoint_slots["A"].revision = 0
        with pytest.raises(TrainerRankSlotStateError, match="replaced"):
            with trainer._gradient_transaction():
                output.sum().backward()
        assert all(p.grad is None for p in trainer._checkpoint_slots["A"].params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_newer_weight_replay_keeps_original_gradient_age() -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=1))
        ref = LoRASlotRef("checkpoint", "A")
        origin = trainer._capture_checkpoint_version("A")
        trainer._checkpoint_slots["A"].revision = origin.revision + 2
        with torch.no_grad():
            for parameter in trainer._checkpoint_slots["A"].params:
                parameter.add_(1)
        capture = trainer._capture_lora_version(ref, origin=origin)
        assert capture is not None
        assert capture.version == origin
        assert capture.weight_version.revision == origin.revision + 2
        for old, current in zip(
            capture.slots[id(lora)].parameters(),
            trainer._checkpoint_slots["A"].params,
            strict=True,
        ):
            torch.testing.assert_close(old, current)
        with use_lora_slot(ref, version=capture):
            output = lora(torch.ones(2, 4, device=device))
        trainer._checkpoint_slots["A"].revision = origin.revision + 3
        with pytest.raises(TrainerRankSlotStateError, match="staleness 3"):
            with trainer._gradient_transaction():
                output.sum().backward()
        assert all(p.grad is None for p in trainer._checkpoint_slots["A"].params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_discarded_snapshot_name_cannot_reuse_old_capture() -> None:
    with _single_rank_model_parallel():
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=1))
        trainer._checkpoint_slots["A"].config = {
            "base_model_name_or_path": "test/model",
            "r": 2,
            "lora_alpha": 32.0,
            "target_modules": ["dense"],
        }
        snapshot_checkpoint(trainer, "A", "saved")
        ref = LoRASlotRef("checkpoint", "saved")
        old = trainer._capture_lora_version(ref)
        assert old is not None
        with use_lora_slot(ref, version=old):
            old_output = lora(torch.ones(2, 4, device=device))
        discard_snapshot_checkpoint(trainer, "saved")
        with torch.no_grad():
            for parameter in trainer._checkpoint_slots["A"].params:
                parameter.add_(2)
        snapshot_checkpoint(trainer, "A", "saved")
        assert trainer._lora_version_capture_bytes(ref) == old.nbytes
        new = trainer._capture_lora_version(ref)
        assert new is not None and new is not old
        assert new.version.generation > old.version.generation
        with pytest.raises(TrainerRankSlotStateError, match="replaced"):
            old.validate()
        with use_lora_slot(ref, version=new):
            new_output = lora(torch.ones(2, 4, device=device))
        assert not torch.equal(old_output, new_output)
