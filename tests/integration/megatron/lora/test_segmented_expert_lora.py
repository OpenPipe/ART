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
