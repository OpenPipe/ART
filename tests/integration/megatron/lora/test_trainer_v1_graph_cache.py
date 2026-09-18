"""Native LoRA/group-executor cache oracle; distributed/full-model gates are separate."""

import gc
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from art.megatron.lora import LoRA, LoRASlotRef, use_lora_slot  # noqa: E402
from art.megatron.prefix_tree_packing import prefix_tree_pack  # noqa: E402
from art.trainer_rank import (  # noqa: E402
    AdamParams,
    ForwardInput,
    ForwardOptions,
    ForwardOutput,
)
from art.trainer_rank._impl import _ForwardGroupPlan  # noqa: E402

from .test_dynamic_lora_slots import (  # noqa: E402
    _adapter,
    _install_checkpoint,
    _single_rank_model_parallel,
    _trainer_for,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ART_GRAPH_GPU_TEST") != "1", reason="requires a reserved GPU"
)


@pytest.mark.parametrize("retention", ["gpu", "cpu", "replay"])
@pytest.mark.parametrize("output_device", ["model", "cpu"])
def test_group_cache_routes_old_gradients_after_optimizer_update(
    retention, output_device, monkeypatch
):
    with _single_rank_model_parallel():
        torch.manual_seed(42)
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        # Megatron DDP uses `buffers` for a list of gradient-buffer objects.
        wrapper = torch.nn.Module()
        wrapper.add_module("module", lora)
        setattr(wrapper, "buffers", [])
        trainer.runtime.model = [wrapper]
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=91))
        initial_revision = trainer._checkpoint_slots["A"].revision
        ref = LoRASlotRef("checkpoint", "A")
        originals = tuple(trainer._checkpoint_slots["A"].params)
        references = tuple(
            value.detach().clone().requires_grad_() for value in originals
        )
        monkeypatch.setattr(trainer, "_topology", lambda: (1, 1, 1, 1))
        monkeypatch.setattr(
            trainer,
            "_prepare_packed_forward",
            lambda packed: packed.tokens.to(device).float().reshape(-1, 4) / 13,
        )

        def physical(items, inputs):
            from torch.utils.checkpoint import checkpoint

            values = checkpoint(
                lambda x: lora(torch.nn.functional.dropout(x, 0.25)),
                inputs,
                use_reentrant=False,
            )
            return [ForwardOutput(None, None, None, values)]

        monkeypatch.setattr(trainer, "_forward_packed", physical)
        outputs, expected_outputs = [], []
        for offset in (0, 3):
            tokens = torch.arange(12) + offset
            request = ForwardInput(
                input_tokens=tokens,
                hidden_states=True,
                options=ForwardOptions(
                    backward_state=retention, output_device=output_device
                ),
            )
            group = _ForwardGroupPlan(
                ref,
                True,
                (0,),
                (trainer._forward_item(request),),
                prefix_tree_pack([tokens], max_depth=0),
            )
            rng = torch.cuda.get_rng_state()
            output = trainer._execute_graph_group(group)[0].hidden_states
            assert output is not None
            after = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng)
            x = tokens.to(device).float().reshape(-1, 4) / 13
            expected = (
                (torch.nn.functional.dropout(x, 0.25) @ references[0]) @ references[1]
            ) * 16
            torch.cuda.set_rng_state(after)
            torch.testing.assert_close(
                output.to(device), expected, atol=3e-5, rtol=3e-5
            )
            outputs.append(output)
            expected_outputs.append(expected)
            tokens.fill_(99)
        loss = (outputs[0] * outputs[1].tanh()).mean()
        expected_loss = (expected_outputs[0] * expected_outputs[1].tanh()).mean()
        expected_gradients = torch.autograd.grad(expected_loss, references)
        with use_lora_slot(ref, version=trainer._capture_lora_version(ref)):
            update = lora(torch.ones(3, 4, device=device)).square().mean()
        with trainer._gradient_transaction():
            update.backward()
        trainer.optim_step(
            params=AdamParams(learning_rate=0.02, grad_clip_norm=0), checkpoints=["A"]
        )
        assert trainer._checkpoint_slots["A"].revision == initial_revision + 1
        rng = torch.cuda.get_rng_state()
        with trainer._gradient_transaction():
            packets = trainer._forward_cotangent_collector().backward(loss)
            trainer._forward_graph_cache().backward_many(
                [(packet.handle, packet.gradients) for packet in packets]
            )
        for actual, expected in zip(originals, expected_gradients, strict=True):
            torch.testing.assert_close(actual.grad, expected, atol=2e-4, rtol=3e-5)
        assert torch.equal(rng, torch.cuda.get_rng_state())
        assert trainer._forward_graph_cache().handles() == ()

        # Abandoned caller graphs also release cache records, including versions.
        unused = trainer._execute_graph_group(group)[0].hidden_states
        assert trainer._forward_graph_cache().handles()
        del unused
        gc.collect()
        assert trainer._forward_graph_cache().handles() == ()


@pytest.mark.parametrize("mode", ["always", "current_replay"])
def test_native_stale_logprob_correction_keeps_original_gradient_age(mode, monkeypatch):
    from art.trainer_rank import (
        ImportanceSamplingGradientCorrection,
        TrainerRankSlotStateError,
    )

    with _single_rank_model_parallel():
        torch.manual_seed(19)
        device = torch.device("cuda")
        lora = LoRA("dense", 4, 5, 2, 32, torch.float32, device)
        trainer = _trainer_for(lora, device)
        _install_checkpoint(trainer, "A", _adapter("dense", rank=2, seed=91))
        ref = LoRASlotRef("checkpoint", "A")
        origin = trainer._capture_checkpoint_version("A")
        parameters = trainer._checkpoint_slots["A"].params
        historical = [value.detach().clone().requires_grad_() for value in parameters]
        tokens = torch.arange(12)
        x = tokens.to(device).float().reshape(-1, 4) / 13
        monkeypatch.setattr(trainer, "_topology", lambda: (1, 1, 1, 1))
        monkeypatch.setattr(
            trainer,
            "_prepare_packed_forward",
            lambda packed: packed.tokens.to(device).float().reshape(-1, 4) / 13,
        )
        executions = []

        def physical(items, inputs):
            executions.append(torch.is_grad_enabled())
            return [ForwardOutput(lora(inputs).log_softmax(-1)[:, 0], None, None, None)]

        monkeypatch.setattr(trainer, "_forward_packed", physical)
        request = ForwardInput(
            input_tokens=tokens,
            target_tokens=torch.zeros_like(tokens),
            options=ForwardOptions(
                stale_gradient_corrections=(
                    ImportanceSamplingGradientCorrection(
                        policy="always" if mode == "always" else "when_available"
                    ),
                )
            ),
        )
        group = _ForwardGroupPlan(
            ref,
            True,
            (0,),
            (trainer._forward_item(request),),
            prefix_tree_pack([tokens], max_depth=0),
        )
        output = trainer._execute_graph_group(group)[0].target_logprobs
        assert output is not None
        old_logprobs = (((x @ historical[0]) @ historical[1]) * 16).log_softmax(-1)[
            :, 0
        ]
        with use_lora_slot(ref, version=trainer._capture_lora_version(ref)):
            update = lora(torch.ones_like(x)).square().mean()
        with trainer._gradient_transaction():
            update.backward()
        trainer.optim_step(
            params=AdamParams(learning_rate=0.001, grad_clip_norm=0), checkpoints=["A"]
        )
        current = [value.detach().clone().requires_grad_() for value in parameters]
        new_logprobs = (((x @ current[0]) @ current[1]) * 16).log_softmax(-1)[:, 0]
        weights = (new_logprobs.detach() - old_logprobs.detach()).exp().clamp(0, 5)
        expected = torch.autograd.grad(
            ((old_logprobs if mode == "always" else new_logprobs) * weights).sum(),
            historical if mode == "always" else current,
        )
        cache = trainer._forward_graph_cache()
        if mode == "current_replay":
            cache.evict(cache.handles()[0], replay_with_current=True)
        with trainer._gradient_transaction():
            packets = trainer._forward_cotangent_collector().backward(output.sum())
            cache.backward_many(
                [(packet.handle, packet.gradients) for packet in packets]
            )
        for parameter, gradient in zip(parameters, expected, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, atol=2e-4, rtol=5e-5)
        assert executions == [True, mode == "current_replay"]
        assert trainer._version_state()._origins["A"] == {(origin, 2)}
        trainer._checkpoint_slots["A"].revision += 2
        with pytest.raises(TrainerRankSlotStateError, match="staleness 3"):
            trainer._version_state().validate_accumulated(["A"])
