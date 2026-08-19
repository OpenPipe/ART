from __future__ import annotations

import pytest
import torch

import art.megatron.lora as lora_module
from art.megatron.lora import (
    EXPERT_TP_GRAD_SYNC_DOMAIN,
    GRAD_SYNC_OP_NONE,
    GRAD_SYNC_OP_SUM,
    LoRA,
    LoraFactor,
    LoRAParallelSpec,
)

EXPERTS = 4
IN_FEATURES = 64
OUT_FEATURES = 96
RANK = 8
COUNTS = (21, 1, 6, 0)
SCALE = 1.75


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_module.ps, "get_expert_model_parallel_rank", lambda: 0)


def _make_lora(shared_factor: LoraFactor, device: torch.device) -> LoRA:
    row = shared_factor == "B"
    return LoRA(
        adapter_model_prefix="base_model.model.layers.0.mlp.experts.{expert}.proj",
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        rank=RANK,
        alpha=RANK * SCALE,
        dtype=torch.bfloat16,
        device=device,
        num_local_experts=EXPERTS,
        moe_parameterization="shared_outer",
        shared_factor=shared_factor,
        a_parallel_spec=LoRAParallelSpec(
            shard_domain="expert_tp",
            sharded=row,
            shard_dim=-2 if row else None,
            grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_NONE if row else GRAD_SYNC_OP_SUM,
        ),
        b_parallel_spec=LoRAParallelSpec(
            shard_domain="expert_tp",
            sharded=not row,
            shard_dim=None if row else -1,
            grad_sync_domain=EXPERT_TP_GRAD_SYNC_DOMAIN,
            grad_sync_op=GRAD_SYNC_OP_SUM if row else GRAD_SYNC_OP_NONE,
        ),
        allreduce=False,
    )


def _positive_rand(
    shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    return (
        torch.rand(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
        + 0.05
    )


def _reference(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for expert, count in enumerate(COUNTS):
        stop = start + count
        if count:
            a_expert = a_t if a_t.ndim == 2 else a_t[expert]
            b_expert = b_t if b_t.ndim == 2 else b_t[expert]
            outputs.append(x[start:stop] @ a_expert @ b_expert)
        start = stop
    return torch.cat(outputs) * SCALE


def _mean_abs_pct(candidate: torch.Tensor, target: torch.Tensor) -> float:
    candidate = candidate.detach().float()
    target = target.detach().float()
    nonzero = target != 0
    assert torch.equal(candidate[~nonzero], target[~nonzero])
    return float(
        ((candidate[nonzero] - target[nonzero]).abs() / target[nonzero].abs()).mean()
        * 100
    )


@pytest.mark.parametrize("shared_factor", ("A", "B"))
def test_compact_shared_outer_module_forward_and_grad(
    shared_factor: LoraFactor,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if "H200" not in torch.cuda.get_device_name(0):
        pytest.skip("validated module test requires H200")

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(
        20260819 + (0 if shared_factor == "A" else 1)
    )
    module = _make_lora(shared_factor, device)
    x_data = _positive_rand(
        (sum(COUNTS), IN_FEATURES), device=device, generator=generator
    )
    a_data = _positive_rand(tuple(module.A_T.shape), device=device, generator=generator)
    b_data = _positive_rand(tuple(module.B_T.shape), device=device, generator=generator)
    grad_out = _positive_rand(
        (sum(COUNTS), OUT_FEATURES), device=device, generator=generator
    )
    with torch.no_grad():
        module.A_T.copy_(a_data)
        module.B_T.copy_(b_data)

    x_ref = x_data.detach().clone().requires_grad_(True)
    a_ref = a_data.detach().clone().requires_grad_(True)
    b_ref = b_data.detach().clone().requires_grad_(True)
    expected = _reference(x_ref, a_ref, b_ref)
    expected.backward(grad_out)

    torch.cuda.reset_peak_memory_stats(device)
    x = x_data.detach().clone().requires_grad_(True)
    actual = module(x, tokens_per_expert=list(COUNTS))
    actual.backward(grad_out)
    torch.cuda.synchronize(device)

    comparisons = {
        "output": (actual, expected, 3.0),
        "input_grad": (x.grad, x_ref.grad, 5.0),
        "A_grad": (module.A_T.grad, a_ref.grad, 5.0),
        "B_grad": (module.B_T.grad, b_ref.grad, 5.0),
    }
    errors: dict[str, float] = {}
    for name, (candidate, target, threshold) in comparisons.items():
        assert candidate is not None and target is not None
        assert torch.count_nonzero(candidate), f"{name} is all zero"
        errors[name] = _mean_abs_pct(candidate, target)
        assert errors[name] <= threshold, f"{name} mean_abs_pct={errors[name]:.6f}%"

    peak_bytes = torch.cuda.max_memory_allocated(device)
    assert peak_bytes <= 4 * 1024**3, (
        f"peak allocation was {peak_bytes / 1024**3:.2f} GiB"
    )
    print(
        f"shared_factor={shared_factor} mean_abs_pct={errors} "
        f"peak_mib={peak_bytes / 1024**2:.2f}"
    )
