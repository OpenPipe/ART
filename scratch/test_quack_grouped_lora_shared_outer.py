from __future__ import annotations

import pytest
import torch

from art.megatron.kernels.cute_grouped_lora_quack import (
    quack_grouped_lora_shared_outer,
)

CASES = (
    ("fc1", 1, (7, 0, 13, 3)),
    ("fc1", 7, (1, 17, 5, 0)),
    ("fc1", 24, (19, 2, 0, 11)),
    ("fc2", 3, (0, 9, 4, 15)),
    ("fc2", 8, (21, 1, 6, 0)),
    ("fc2", 16, (3, 12, 0, 23)),
)


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _positive_rand(
    shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    return (
        torch.rand(shape, device=device, dtype=torch.bfloat16, generator=generator)
        * 0.1
        + 0.05
    )


def _reference(
    x: torch.Tensor,
    a_t: torch.Tensor,
    b_t: torch.Tensor,
    counts: tuple[int, ...],
    *,
    scale: float,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    start = 0
    for expert, count in enumerate(counts):
        stop = start + count
        if count:
            a_expert = a_t if a_t.ndim == 2 else a_t[expert]
            b_expert = b_t if b_t.ndim == 2 else b_t[expert]
            outputs.append(x[start:stop] @ a_expert @ b_expert)
        start = stop
    assert start == x.shape[0]
    return torch.cat(outputs) * scale


def _mean_abs_pct(candidate: torch.Tensor, target: torch.Tensor) -> float:
    candidate = candidate.detach().float()
    target = target.detach().float()
    nonzero = target != 0
    if not torch.equal(candidate[~nonzero], target[~nonzero]):
        return float("inf")
    return float(
        ((candidate[nonzero] - target[nonzero]).abs() / target[nonzero].abs()).mean()
        * 100
    )


def _require_grad(tensor: torch.Tensor, name: str) -> torch.Tensor:
    assert tensor.grad is not None, f"{name}.grad is None"
    assert torch.count_nonzero(tensor.grad), f"{name}.grad is all zero"
    return tensor.grad


@pytest.mark.parametrize(("mode", "rank", "counts"), CASES)
def test_shared_outer_matches_reference(
    mode: str, rank: int, counts: tuple[int, ...]
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260819 + rank)
    experts = len(counts)
    tokens = sum(counts)
    in_features = 64
    out_features = 96
    scale = 1.75

    x_data = _positive_rand((tokens, in_features), device=device, generator=generator)
    if mode == "fc1":
        a_data = _positive_rand((in_features, rank), device=device, generator=generator)
        b_data = _positive_rand(
            (experts, rank, out_features), device=device, generator=generator
        )
    else:
        a_data = _positive_rand(
            (experts, in_features, rank), device=device, generator=generator
        )
        b_data = _positive_rand(
            (rank, out_features), device=device, generator=generator
        )
    grad_out = _positive_rand(
        (tokens, out_features), device=device, generator=generator
    )

    x_ref, a_ref, b_ref = map(_leaf, (x_data, a_data, b_data))
    ref_out = _reference(x_ref, a_ref, b_ref, counts, scale=scale)
    ref_out.backward(grad_out)

    x_got, a_got, b_got = map(_leaf, (x_data, a_data, b_data))
    got_out = quack_grouped_lora_shared_outer(
        x_got, a_got, b_got, list(counts), scale=scale
    )
    saved_a_t, saved_b_t = got_out.grad_fn.saved_tensors[1:3]
    assert saved_a_t.ndim == a_data.ndim
    assert saved_b_t.ndim == b_data.ndim
    got_out.backward(grad_out)

    tensors = {
        "out": (got_out, ref_out),
        "x_grad": (_require_grad(x_got, "x_got"), _require_grad(x_ref, "x_ref")),
        "a_grad": (_require_grad(a_got, "a_got"), _require_grad(a_ref, "a_ref")),
        "b_grad": (_require_grad(b_got, "b_got"), _require_grad(b_ref, "b_ref")),
    }
    errors: dict[str, float] = {}
    for name, (candidate, target) in tensors.items():
        assert torch.count_nonzero(candidate), f"{name} is all zero"
        threshold = 3.0 if name == "out" else 5.0
        errors[name] = _mean_abs_pct(candidate, target)
        assert errors[name] <= threshold, f"{name} mean_abs_pct={errors[name]:.6f}%"
    print(f"{mode=} {rank=} {counts=} mean_abs_pct={errors}")


def test_shared_outer_rejects_two_expert_factors() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    x = torch.empty(2, 8, device="cuda", dtype=torch.bfloat16)
    a_t = torch.empty(2, 8, 1, device="cuda", dtype=torch.bfloat16)
    b_t = torch.empty(2, 1, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="exactly one shared factor"):
        quack_grouped_lora_shared_outer(x, a_t, b_t, [1, 1])
