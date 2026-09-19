"""Independent CUDA matrix/Adam oracle for historical native LoRA graphs.

This deliberately exercises the production LoRA and TrainerRank optimizer but
uses explicit float64 matrix products and Adam equations for expected values.
Full-model and distributed acceptance are additional gates, not implied here.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from art.megatron.lora import LoRA, LoRASlotRef, use_lora_slot  # noqa: E402
from art.trainer_rank import AdamParams  # noqa: E402

from .test_dynamic_lora_slots import (  # noqa: E402
    _adapter,
    _install_checkpoint,
    _single_rank_model_parallel,
    _trainer_for,
)


def _coupled_loss(first, second):
    return (first * second.tanh()).mean() + 0.13 * first.square().mean()


def _adam(parameters, gradients, moments, step, params):
    """Explicit AdamW, independent of the runtime's optimizer implementation."""
    result = []
    for parameter, gradient, (first, second) in zip(
        parameters, gradients, moments, strict=True
    ):
        first.mul_(params.beta1).add_(gradient, alpha=1 - params.beta1)
        second.mul_(params.beta2).addcmul_(gradient, gradient, value=1 - params.beta2)
        numerator = first / (1 - params.beta1**step)
        denominator = (second / (1 - params.beta2**step)).sqrt() + 1e-8
        result.append(
            parameter * (1 - params.learning_rate * params.weight_decay)
            - params.learning_rate * numerator / denominator
        )
    return result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("recompute", ["none", "torch", "reentrant", "megatron"])
def test_native_old_lora_graph_matches_matrix_and_adam_oracle(recompute, artifact_dir):
    with _single_rank_model_parallel():
        torch.manual_seed(1709)
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=91))
        ref = LoRASlotRef("checkpoint", "A")
        originals = tuple(trainer._checkpoint_slots["A"].params)
        reference = [p.detach().double().requires_grad_() for p in originals]
        scale = 16.0
        params = AdamParams(
            learning_rate=0.05,
            beta1=0.2,
            beta2=0.5,
            weight_decay=0.01,
            grad_clip_norm=0.0,
        )
        moments = [(torch.zeros_like(p), torch.zeros_like(p)) for p in reference]
        inputs = [
            (
                torch.arange(12, device=device).reshape(3, 4) / 13 + offset
            ).requires_grad_()
            for offset in (0.1, -0.3)
        ]
        reference_inputs = [x.detach().double().requires_grad_() for x in inputs]

        def native(x):
            return lora(x * torch.nn.functional.dropout(torch.ones_like(x), 0.25))

        def explicit(x):
            mask = torch.nn.functional.dropout(
                torch.ones_like(x, dtype=torch.float32), 0.25
            )
            return ((x * mask) @ reference[0]) @ reference[1] * scale

        def checkpoint(x):
            if recompute == "none":
                return native(x)
            if recompute == "megatron":
                from megatron.core.tensor_parallel.random import checkpoint

                return checkpoint(native, False, x)
            from torch.utils.checkpoint import checkpoint

            return checkpoint(native, x, use_reentrant=recompute == "reentrant")

        capture = trainer._capture_lora_version(ref, max_gradient_staleness=2)
        rng = torch.cuda.get_rng_state()
        with use_lora_slot(ref, version=capture):
            outputs = [checkpoint(x) for x in inputs]
            unused = checkpoint(inputs[0] + 0.7)
        after_forward = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng)
        reference_outputs = [explicit(x) for x in reference_inputs]
        torch.cuda.set_rng_state(after_forward)
        for actual, expected in zip(outputs, reference_outputs, strict=True):
            torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=2e-5)
        old_loss = _coupled_loss(*outputs)
        old_reference_loss = _coupled_loss(*reference_outputs)

        # A separate, completed step mutates current optimizer weights while
        # both coupled old forwards and an unused output remain alive.
        update_input = torch.linspace(-0.3, 0.8, 20, device=device).reshape(5, 4)
        with use_lora_slot(
            ref, version=trainer._capture_lora_version(ref, max_gradient_staleness=2)
        ):
            update_loss = lora(update_input).square().mean() / scale**2
        update_reference = (
            ((update_input.double() @ reference[0]) @ reference[1]).square().mean()
        )
        update_grads = torch.autograd.grad(update_reference, reference)
        with trainer._gradient_transaction():
            update_loss.backward()
        trainer.optim_step(params=params, checkpoints=["A"])
        expected_current = _adam(reference, update_grads, moments, 1, params)
        current = tuple(trainer._checkpoint_slots["A"].params)
        for actual, expected in zip(current, expected_current, strict=True):
            torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
        assert any(not torch.equal(a, b) for a, b in zip(current, reference))

        expected_gradients = torch.autograd.grad(
            old_reference_loss, [*reference, *reference_inputs]
        )
        torch.rand(17, device=device)
        before_backward = torch.cuda.get_rng_state()
        with trainer._gradient_transaction():
            old_loss.backward()
        assert torch.equal(before_backward, torch.cuda.get_rng_state())
        assert unused.grad_fn is not None
        errors = []
        for actual, expected in zip(
            [p.grad for p in current] + [x.grad for x in inputs],
            expected_gradients,
            strict=True,
        ):
            assert actual is not None
            torch.testing.assert_close(actual.double(), expected, atol=2e-4, rtol=3e-5)
            errors.append(float((actual.double() - expected).abs().max()))
        trainer.optim_step(params=params, checkpoints=["A"])
        expected_final = _adam(
            expected_current, expected_gradients[:2], moments, 2, params
        )
        for actual, expected in zip(
            trainer._checkpoint_slots["A"].params, expected_final, strict=True
        ):
            torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-5)
        (artifact_dir / "oracle.json").write_text(
            json.dumps(
                {
                    "recompute": recompute,
                    "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "max_abs_gradient_errors": errors,
                    "original_version_age": 1,
                    "optimizer_steps": 2,
                },
                indent=2,
            )
            + "\n"
        )
