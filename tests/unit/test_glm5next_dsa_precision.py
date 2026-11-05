"""CPU controls for the real trainer reduction and Flash-only CP dKV opt-in."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core import parallel_state as ps

from art.megatron.glm5next.dsa_cp import Glm5NextDsaState
from art.megatron.glm52 import cp_attention
from art.megatron.glm52.state import Glm52PrefixTreeState
from art.trainer_rank import TrainerRank
from tests.integration.megatron.glm5next_dsa_parity import LowRank, lora_gradients


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_harness_matches_trainer_rank_fp32_reduction(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    adapter = LowRank(4, 6, 2, torch.empty((), dtype=dtype))
    params = tuple(adapter.parameters())
    originals = []
    for param in params:
        param.grad = torch.linspace(-1, 1, param.numel()).reshape_as(param).to(dtype)
        originals.append(param.grad.clone())
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(ps, "get_data_parallel_group", lambda **_: group)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_group", lambda **_: None)
    calls = []

    def all_reduce(tensor: torch.Tensor, **kwargs: Any) -> None:
        assert tensor.dtype == torch.float32
        assert kwargs == {"group": group, "op": torch.distributed.ReduceOp.SUM}
        calls.append(tensor.clone())
        # A remote rank's FP32-promoted local gradient. Keep low bits that a
        # BF16 collective or cast-back would discard.
        tensor.add_(0.001)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    expected = TrainerRank._reduce_dynamic_grads(
        cast(Any, SimpleNamespace()), params, scale_grads=1.0
    )
    actual = lora_gradients({"probe": adapter}, cp_group=group)
    assert len(calls) == 2
    for result, reference, param, original in zip(
        actual.values(), expected, params, originals, strict=True
    ):
        assert result.dtype == reference.dtype == torch.float32
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        torch.testing.assert_close(result, original.float() + 0.001, rtol=0, atol=0)
        torch.testing.assert_close(param.grad, original, rtol=0, atol=0)


def test_fp32_control_and_retained_local_bf16_rounding() -> None:
    """FP32 communication cannot recover a partial already rounded locally."""
    parts = [torch.tensor([[1.001]], dtype=torch.float64), torch.tensor([[-1.0]])]
    total = (parts[0] + parts[1]).float()
    results = {}
    for dtype in (torch.float32, torch.bfloat16):
        adapters = [LowRank(1, 1, 1, torch.empty((), dtype=dtype)) for _ in parts]
        for adapter, partial in zip(adapters, parts, strict=True):
            adapter.a.grad = partial.to(dtype)
            adapter.b.grad = partial.to(dtype)
        reduced = [lora_gradients({"probe": adapter}) for adapter in adapters]
        results[dtype] = reduced[0]["probe.a"] + reduced[1]["probe.a"]
    fp32_error = float((results[torch.float32] - total).norm() / total.norm())
    bf16_error = float((results[torch.bfloat16] - total).norm() / total.norm())
    assert fp32_error < 1e-3  # FP32-control passes the original CP gate.
    assert bf16_error > 1e-3  # Real BF16 local rounding remains visible.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("padded", [False, True])
def test_flash_dkv_sums_before_rounding_glm52_keeps_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, padded: bool
) -> None:
    local = SimpleNamespace(is_local_stage=True)
    remote = SimpleNamespace(is_local_stage=False)
    cp = SimpleNamespace(
        rank_plan=SimpleNamespace(
            local_valid_lengths=(2,),
            stage_plans=(local, remote),
            backward_stage_indices=(0, 1),
        )
    )
    length = 3 if padded else 2
    q = torch.zeros(1, length, 1, 576, dtype=dtype)
    kv = torch.zeros(1, length, 576, dtype=dtype)
    # Fixed FP32 sparse-kernel partials isolate the cast/sum order from GEMMs.
    parts = torch.cat(
        (torch.full((1, 2, 576), 1.001), torch.full((1, 2, 576), -1.0)), 1
    )
    monkeypatch.setattr(
        cp_attention,
        "_combined_stage_kv",
        lambda tensor, state: (parts[0].to(dtype), (2, 2)),
    )
    monkeypatch.setattr(
        cp_attention,
        "sparse_mla_backward",
        lambda *args, **kwargs: (torch.zeros_like(q[:, :2]), parts.clone()),
    )
    seen = []

    def add(target: torch.Tensor, grad: torch.Tensor, *args: Any) -> None:
        seen.append((target.dtype, grad.dtype))
        target.add_(grad)

    def launch(grad: torch.Tensor, stage: Any, state: Any, target: torch.Tensor) -> Any:
        return SimpleNamespace(wait_post_process=lambda: add(target, grad))

    monkeypatch.setattr(cp_attention, "reduce_local_stage_rows_", add)
    monkeypatch.setattr(cp_attention, "launch_remote_stage_reduce", launch)
    for state_class in (Glm5NextDsaState, Glm52PrefixTreeState):
        state = state_class.model_construct(context_parallel_state=cp)
        seen.clear()
        _, result = cp_attention._backward(
            q[..., :512],
            q,
            kv,
            torch.zeros(1, 2, 64, dtype=torch.int32),
            q[0, :2, :, :512],
            torch.zeros(2, 1),
            state,
            1.0,
            None,
        )
        accumulation_dtype = torch.float32 if state_class is Glm5NextDsaState else dtype
        assert seen == [(accumulation_dtype, accumulation_dtype)] * 2
        expected = (
            parts[:, :2].to(accumulation_dtype) + parts[:, 2:].to(accumulation_dtype)
        ).to(dtype)
        assert result.dtype == dtype
        torch.testing.assert_close(result[:, :2], expected, rtol=0, atol=0)
        if padded:
            assert not bool(result[:, 2:].any())
        if dtype == torch.bfloat16:
            assert bool(result[:, :2].any()) == (state_class is Glm5NextDsaState)
