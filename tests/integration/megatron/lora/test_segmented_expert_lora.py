"""Native ART slot/recompute integration for compact shared-outer waves."""

import tempfile
from typing import cast

from megatron.core import parallel_state as ps
import pytest
import torch
from torch.utils import checkpoint as checkpoint_module

from art.megatron.lora import LoRA, LoRASlot, LoRASlotRef
from art.megatron.segmented_expert_lora import (
    ExpertAdapterPlan,
    use_expert_adapter_plan,
)


@pytest.fixture(scope="module", autouse=True)
def parallel_state():
    if not torch.cuda.is_available():
        pytest.skip("one CUDA device is required")
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as init_dir:
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{init_dir}/store", rank=0, world_size=1
        )
        ps.initialize_model_parallel()
        yield
        ps.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("shared", ["A", "B"])
@pytest.mark.parametrize("reentrant", [False, True])
def test_slot_gradients_and_recompute_keep_dispatched_adapter_plan(shared, reentrant):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(41)
    lora = LoRA(
        "experts.{expert}",
        16,
        24,
        8,
        8,
        torch.float32,
        torch.device("cuda"),
        num_local_experts=3,
    )
    refs = tuple(LoRASlotRef("checkpoint", f"run-{i}") for i in range(3))
    for index, ref in enumerate(refs):
        slot = LoRASlot(
            ref=ref,
            a_t=torch.randn(1 if shared == "A" else 3, 16, 3, device="cuda"),
            b_t=torch.randn(1 if shared == "B" else 3, 3, 24, device="cuda"),
            alpha=3.0,
            a_template=lora.A_T,
            b_template=lora.B_T,
            requires_grad=True,
        )
        key = f"slot_{index}"
        lora._slot_keys[ref] = key
        lora._slot_modules[key] = slot
    ids = torch.tensor([2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 2, 1], device="cuda")
    experts = [0] * 7 + [2] * 5
    plan = ExpertAdapterPlan.build(refs, ids, [7, 0, 5])
    x = torch.randn(12, 16, device="cuda", requires_grad=True)
    dy = torch.randn(12, 24, device="cuda")
    rows = []
    for row, (adapter, expert) in enumerate(zip(ids.tolist(), experts, strict=True)):
        slot = lora._slot(refs[adapter])
        assert slot is not None
        rows.append(
            x[row]
            @ slot.A_T[0 if shared == "A" else expert]
            @ slot.B_T[0 if shared == "B" else expert]
        )
    expected = torch.stack(rows)
    expected.backward(dy)
    dx = cast(torch.Tensor, x.grad).clone()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in lora.named_parameters()
        if parameter.grad is not None
    }
    lora.zero_grad(set_to_none=True)
    x.grad = None
    with use_expert_adapter_plan(plan):
        actual = checkpoint_module.checkpoint(
            lambda value: lora(value, [7, 0, 5]), x, use_reentrant=reentrant
        )
    # A subsequent wave changes both geometry and slot ownership before backward.
    other = ExpertAdapterPlan.build(refs, torch.tensor([0], device="cuda"), [0, 1, 0])
    with use_expert_adapter_plan(other):
        actual.backward(dy)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad, dx, atol=2e-5, rtol=1e-5)
    for name, parameter in lora.named_parameters():
        if name in gradients:
            torch.testing.assert_close(
                parameter.grad, gradients[name], atol=2e-5, rtol=1e-5
            )
        else:
            assert parameter.grad is None


@pytest.mark.parametrize("shared", ["A", "B"])
def test_native_ep1_dispatch_outputs_input_probability_and_slot_gradients(shared):
    from types import SimpleNamespace

    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    from art.megatron.segmented_expert_lora import (
        AdapterTokens,
        install_segmented_dispatcher,
        multi_adapter_tokens,
    )

    torch.manual_seed(81)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_moe_experts=3,
        moe_router_topk=2,
        moe_permute_fusion=True,
        moe_token_dispatcher_type="alltoall",
        params_dtype=torch.bfloat16,
    )
    dispatcher = MoEAlltoAllTokenDispatcher(3, [0, 1, 2], config)
    layer = SimpleNamespace(config=config, token_dispatcher=dispatcher)
    install_segmented_dispatcher(layer)
    install_segmented_dispatcher(layer)  # Idempotent, preserves the native object.
    assert layer.token_dispatcher is dispatcher
    prefix = "experts.{expert}." + ("gate_proj" if shared == "A" else "down_proj")
    lora = LoRA(
        prefix,
        16,
        16,
        8,
        8,
        torch.bfloat16,
        torch.device("cuda"),
        num_local_experts=3,
        expert_layout="shared_outer",
    )
    refs = tuple(LoRASlotRef("checkpoint", f"run-{i}") for i in range(3))
    for index, ref in enumerate(refs):
        slot = LoRASlot(
            ref=ref,
            a_t=torch.randn_like(lora.A_T) * 0.02,
            b_t=torch.randn_like(lora.B_T) * 0.02,
            alpha=8 * (index + 1),
            a_template=lora.A_T,
            b_template=lora.B_T,
            requires_grad=True,
        )
        key = f"slot_{index}"
        lora._slot_keys[ref], lora._slot_modules[key] = key, slot
    x = torch.randn(7, 1, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    probabilities = torch.randn(7, 3, device="cuda", requires_grad=True)
    routing = torch.ones(7, 3, device="cuda", dtype=torch.bool)
    routing[torch.arange(7), torch.arange(7) % 3] = False
    ids = torch.tensor([0, 2, 1, 0, 2, 1, 0], device="cuda")
    cotangent = torch.randn_like(x)

    def routed(hidden, routes, probs):
        hidden, probs = dispatcher.dispatch_preprocess(hidden, routes, probs)
        hidden, probs = dispatcher.token_dispatch(hidden, probs)
        hidden, counts, probs = dispatcher.dispatch_postprocess(hidden, probs)
        projected = lora(hidden, counts) * probs[:, None].to(hidden.dtype)
        projected = dispatcher.combine_preprocess(projected)
        projected = dispatcher.token_combine(projected)
        return dispatcher.combine_postprocess(projected)

    expected = torch.zeros_like(x)
    for index, ref in enumerate(refs):
        rows = torch.where(ids == index)[0]
        tokens = AdapterTokens(
            (ref,), torch.zeros(len(rows), dtype=torch.int64, device="cuda")
        )
        with multi_adapter_tokens(tokens):
            result = routed(x[rows], routing[rows], probabilities[rows])
        expected = expected.index_copy(0, rows, result)
    expected.backward(cotangent)
    dx, dp = (
        cast(torch.Tensor, x.grad).clone(),
        cast(torch.Tensor, probabilities.grad).clone(),
    )
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in lora.named_parameters()
        if parameter.grad is not None
    }
    lora.zero_grad(set_to_none=True)
    x.grad = probabilities.grad = None
    with multi_adapter_tokens(AdapterTokens(refs, ids)):
        actual = routed(x, routing, probabilities)
    actual.backward(cotangent)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, dx, atol=0, rtol=0)
    torch.testing.assert_close(probabilities.grad, dp, atol=0, rtol=0)
    for name, parameter in lora.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=0, rtol=0)
        else:
            assert parameter.grad is None
