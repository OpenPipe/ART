"""Slot ownership, row layout, numerical parity, and checkpoint-context oracles."""

from dataclasses import replace
import json
import tempfile
from typing import Any, cast

from megatron.core import parallel_state as ps
import pytest
import torch

from art.megatron.lora import LoRA, LoRASlotRef, use_lora_slot
from art.megatron.segmented_lora import LoRASegments, use_lora_segments


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


def make_case(dtype=torch.float32):
    torch.manual_seed(74)
    lora = LoRA("dense", 64, 96, 8, 8, dtype, torch.device("cuda"))
    refs = tuple(LoRASlotRef("checkpoint", f"adapter-{i}") for i in range(4))
    for i, (ref, rank) in enumerate(zip(refs, (1, 3, 13, 64), strict=True)):
        lora.load_lora_slot(
            ref,
            {
                "dense.lora_A.weight": torch.randn(rank, 64, device="cuda", dtype=dtype)
                * 0.02,
                "dense.lora_B.weight": torch.randn(96, rank, device="cuda", dtype=dtype)
                * 0.02,
            },
            alpha=rank * (0.75 + i * 0.125),
            requires_grad=True,
        )
    # Include an empty adapter, a base segment and a module missing this target.
    refs += (LoRASlotRef("checkpoint", None), LoRASlotRef("lora", "absent"))
    segments = LoRASegments(refs, (11, 0, 17, 23, 3, 5))
    return lora, segments


def sequential(lora, x, segments):
    chunks = x.split(segments.counts)
    outputs = []
    with use_lora_segments(None):
        for ref, chunk in zip(segments.refs, chunks, strict=True):
            with use_lora_slot(ref):
                outputs.append(lora(chunk))
    return torch.cat(outputs)


@pytest.mark.parametrize(
    "checkpoint_kind", ("none", "torch", "torch_reentrant", "megatron")
)
def test_segmented_fp32_outputs_grads_and_recompute(checkpoint_kind):
    torch.backends.cuda.matmul.allow_tf32 = False
    lora, segments = make_case()
    x = torch.randn(segments.rows, 1, 64, device="cuda", requires_grad=True)
    dy = torch.randn(segments.rows, 1, 96, device="cuda")
    expected = sequential(lora, x, segments)
    expected.backward(dy)
    dx = cast(torch.Tensor, x.grad).clone()
    grads = {
        name: p.grad.clone()
        for name, p in lora.named_parameters()
        if p.grad is not None
    }
    lora.zero_grad(set_to_none=True)
    x.grad = None
    with use_lora_segments(segments):
        if checkpoint_kind.startswith("torch"):
            # Access the patched module attribute, as the real runtime does.
            from torch.utils import checkpoint as checkpoint_module

            actual = checkpoint_module.checkpoint(
                lora, x, use_reentrant=checkpoint_kind == "torch_reentrant"
            )
        elif checkpoint_kind == "megatron":
            from megatron.core.tensor_parallel import checkpoint

            actual = checkpoint(lora, False, x)
        else:
            actual = lora(x)
    # Backward must use the captured table even under a conflicting ambient table.
    wrong = LoRASegments((segments.refs[0],), (segments.rows,))
    with use_lora_segments(wrong), use_lora_slot(segments.refs[1]):
        actual.backward(dy)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(x.grad, dx, atol=1e-6, rtol=1e-5)
    for name, p in lora.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name], atol=1e-6, rtol=1e-5)
        else:
            assert p.grad is None, name
    assert lora.A_T.grad is None and lora.B_T.grad is None
    print(
        "ORACLE "
        + json.dumps(
            {
                "dtype": "float32",
                "checkpoint": checkpoint_kind,
                "output_max_abs_error": float(
                    (actual.detach() - expected.detach()).abs().max()
                ),
                "input_grad_max_abs_error": float(
                    (cast(torch.Tensor, x.grad) - dx).abs().max()
                ),
                "param_grad_max_abs_error": max(
                    float((p.grad - grads[name]).abs().max())
                    for name, p in lora.named_parameters()
                    if name in grads
                ),
                "atol": 1e-6,
                "rtol": 1e-5,
                "passed": True,
            }
        )
    )


def test_segment_table_validation_and_zero_rows():
    lora, segments = make_case()
    indices = segments.adapter_indices()
    assert LoRASegments.from_adapter_indices(segments.refs, indices) == segments
    with pytest.raises(ValueError, match="sorted"):
        LoRASegments.from_adapter_indices(segments.refs, indices.flip(0))
    with pytest.raises(ValueError, match="exactly one"):
        LoRASegments((segments.refs[0], segments.refs[0]), (1, 1))
    with use_lora_segments(segments), pytest.raises(ValueError, match="row"):
        lora(torch.zeros(1, 64, device="cuda"))
    lora.zero_grad(set_to_none=True)
    empty = LoRASegments(segments.refs, (0,) * len(segments.refs))
    with use_lora_segments(empty):
        lora(torch.empty(0, 64, device="cuda")).sum().backward()
    for ref in segments.refs[:4]:
        for p in lora.lora_slot_params(ref):
            assert p.grad is not None and not p.grad.any()


def test_segmented_bf16_quack_matches_sequential():
    lora, segments = make_case(torch.bfloat16)
    segments = replace(segments, grouped=True)
    x = torch.randn(
        segments.rows, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dy = torch.randn(segments.rows, 96, device="cuda", dtype=torch.bfloat16)
    expected = sequential(lora, x, segments)
    expected.backward(dy)
    dx = cast(torch.Tensor, x.grad).clone()
    grads = {
        name: p.grad.clone()
        for name, p in lora.named_parameters()
        if p.grad is not None
    }
    lora.zero_grad(set_to_none=True)
    x.grad = None
    with use_lora_segments(segments):
        actual = lora(x)
    actual.backward(dy)
    assert x.grad is not None

    # Keep strict FP32 checks above. BF16 is qualified with a per-adapter
    # relative-L2 envelope; pointwise ratios around cancellation are unstable.
    def rms_error(a, b):
        return float(
            (a.detach().float() - b.detach().float()).norm()
            / b.detach().float().norm().clamp_min(1e-30)
        )

    def assert_bf16_close(a, b):
        assert torch.isfinite(a).all()
        assert rms_error(a, b) < 0.01

    for a, b in zip(
        actual.split(segments.counts), expected.split(segments.counts), strict=True
    ):
        assert_bf16_close(a, b)
    for a, b in zip(
        x.grad.split(segments.counts), dx.split(segments.counts), strict=True
    ):
        assert_bf16_close(a, b)
    for name, p in lora.named_parameters():
        if name in grads:
            assert p.grad is not None
            assert_bf16_close(p.grad, grads[name])
        else:
            assert p.grad is None

    print(
        "ORACLE "
        + json.dumps(
            {
                "dtype": "bfloat16",
                "checkpoint": "none",
                "output_max_abs_error": float(
                    (actual.float() - expected.float()).abs().max()
                ),
                "input_grad_max_abs_error": float(
                    (x.grad.float() - dx.float()).abs().max()
                ),
                "param_grad_max_abs_error": max(
                    float((p.grad.float() - grads[name].float()).abs().max())
                    for name, p in lora.named_parameters()
                    if name in grads
                ),
                "passed_bf16_gate": True,
                "bf16_relative_l2_limit": 0.01,
                "output_relative_l2": rms_error(actual, expected),
                "input_grad_relative_l2": rms_error(x.grad, dx),
                "param_grad_max_relative_l2": max(
                    rms_error(p.grad, grads[name])
                    for name, p in lora.named_parameters()
                    if name in grads
                ),
                "passed_fp32_tolerance": bool(
                    torch.allclose(
                        actual.float(), expected.float(), atol=1e-6, rtol=1e-5
                    )
                ),
            }
        )
    )

    # A scalar reduction supplies stride-zero gradients to this new dense caller.
    lora.zero_grad(set_to_none=True)
    x.grad = None
    with use_lora_segments(segments):
        output = lora(x)
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for ref in segments.refs[:4]:
        for p in lora.lora_slot_params(ref):
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_routed_experts_reject_adapter_segments():
    lora = LoRA("experts.{expert}", 64, 96, 8, 8, torch.float32, torch.device("cuda"))
    segments = LoRASegments((None,), (3,))
    with use_lora_segments(segments):
        with pytest.raises(NotImplementedError, match="expert, adapter"):
            lora(torch.zeros(3, 64, device="cuda"), [3])
        with pytest.raises(NotImplementedError, match="expert, adapter"):
            lora.active_lora_tensors()


def test_empty_shared_fc1_attaches_owning_adapter_gradients():
    from art.megatron.lora import SharedExpertsLinearFC1LoRA

    # The empty path must not invoke the base projection at all.
    base = torch.nn.Linear(64, 192, bias=False, device="cuda")
    wrapped = SharedExpertsLinearFC1LoRA("shared", cast(Any, base), rank=8, alpha=8)
    refs = (LoRASlotRef("checkpoint", "first"), LoRASlotRef("checkpoint", "second"))
    for lora in (wrapped.gate_lora, wrapped.up_lora):
        for ref in refs:
            lora.load_lora_slot(
                ref,
                {
                    f"{lora.adapter_model_prefix}.lora_A.weight": torch.randn(
                        3, 64, device="cuda"
                    ),
                    f"{lora.adapter_model_prefix}.lora_B.weight": torch.randn(
                        96, 3, device="cuda"
                    ),
                },
                requires_grad=True,
            )
    with use_lora_segments(LoRASegments(refs, (0, 0))):
        y, bias = wrapped(torch.empty(0, 1, 64, device="cuda"))
        y.sum().backward()
    assert y.shape == (0, 1, 192) and bias is None
    for lora in (wrapped.gate_lora, wrapped.up_lora):
        assert lora.A_T.grad is None and lora.B_T.grad is None
        for ref in refs:
            assert all(
                p.grad is not None and not p.grad.any()
                for p in lora.lora_slot_params(ref)
            )
