from __future__ import annotations

import pytest
import torch
from torch import Tensor
import torch.nn.functional as F

from art.megatron.gdn.conv_gelu import packed_varlen_causal_conv
from tests.integration.megatron.gdn_shared_prefix.metrics import assert_mean_abs_pct

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_packed_varlen_causal_conv_gelu_matches_reference_with_final_grads() -> None:
    _run_packed_case(
        lengths=(1, 2, 4, 7),
        channels=9,
        kernel_width=4,
        has_bias=True,
        activation="gelu",
    )


def test_packed_varlen_causal_conv_gelu_matches_reference_without_bias() -> None:
    _run_packed_case(
        lengths=(1, 3, 5),
        channels=7,
        kernel_width=3,
        has_bias=False,
        activation="gelu",
    )


def test_packed_varlen_causal_conv_silu_and_swish_match_reference() -> None:
    for activation in ("silu", "swish"):
        _run_packed_case(
            lengths=(1, 4, 6),
            channels=5,
            kernel_width=5,
            has_bias=True,
            activation=activation,
        )


@pytest.mark.parametrize("activation", ("silu", "swish"))
@pytest.mark.parametrize("final_only", (False, True))
def test_bf16_silu_matches_materialized_activation(
    activation: str, final_only: bool
) -> None:
    inputs = _packed_inputs(
        lengths=(0, 1, 2, 3, 4, 5, 127, 128, 129),
        channels=17,
        kernel_width=4,
        has_bias=True,
        seed=1053,
    )
    conv_in, cu, initial, weight, bias, out_grad, final_grad = (
        value.to(torch.bfloat16) if value.is_floating_point() else value
        for value in inputs
        if value is not None
    )
    if final_only:
        # Distinct raw inputs and cotangents make tail routing observable.
        out_grad.zero_()
    else:
        # Exact BF16 operands accumulate to 1 + 1/256, a rounding tie.
        conv_in.fill_(1)
        initial.fill_(1)
        weight[:] = weight.new_tensor([1, 1 / 256, -1, 1])
        bias.zero_()
        out_grad.fill_(0.3)
        final_grad.fill_(0.5)
    args = conv_in, cu, initial, weight, bias, out_grad, final_grad
    reference = _run_packed_fused(*args, activation=activation, split_silu=True)
    candidate = _run_packed_fused(*args, activation=activation)
    for name, expected in reference.items():
        actual = candidate[name]
        assert expected is not None and actual is not None
        assert expected.dtype == actual.dtype == torch.bfloat16
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        # Numeric equality: BF16 SiLU backward may differ in signed-zero bits.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not final_only:
            assert torch.any(expected != 0), name
    torch.testing.assert_close(
        candidate["final"],
        _packed_reference_final(conv_in, cu, initial),
        rtol=0,
        atol=0,
    )
    if final_only:
        x = conv_in.detach().requires_grad_(True)
        state = initial.detach().requires_grad_(True)
        raw_final = _packed_reference_final(x, cu, state)
        dx, dstate = torch.autograd.grad(raw_final, (x, state), final_grad)
        torch.testing.assert_close(candidate["conv_in_grad"], dx, rtol=0, atol=0)
        torch.testing.assert_close(
            candidate["conv_initial_grad"], dstate, rtol=0, atol=0
        )
        for name in ("weight_grad", "bias_grad"):
            assert torch.count_nonzero(candidate[name]) == 0


def test_packed_varlen_causal_conv_supports_unit_kernel() -> None:
    _run_packed_case(
        lengths=(1, 5),
        channels=4,
        kernel_width=1,
        has_bias=True,
        activation="none",
    )


def test_packed_varlen_causal_conv_rejects_unsupported_activation() -> None:
    conv_in, cu_seqlens, conv_initial, weight, bias, _, _ = _packed_inputs(
        lengths=(2,),
        channels=3,
        kernel_width=2,
        has_bias=True,
        seed=17,
    )
    with pytest.raises(ValueError, match="activation"):
        packed_varlen_causal_conv(
            conv_in,
            cu_seqlens,
            conv_initial,
            weight,
            bias,
            activation="relu",
        )


def _run_packed_case(
    *,
    lengths: tuple[int, ...],
    channels: int,
    kernel_width: int,
    has_bias: bool,
    activation: str,
) -> None:
    inputs = _packed_inputs(
        lengths=lengths,
        channels=channels,
        kernel_width=kernel_width,
        has_bias=has_bias,
        seed=kernel_width * 100 + channels + len(lengths),
    )
    conv_in, cu_seqlens, conv_initial, weight, bias, out_grad, final_grad = inputs
    ref = _run_packed_reference(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        out_grad,
        final_grad,
        activation=activation,
    )
    cand = _run_packed_fused(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        out_grad,
        final_grad,
        activation=activation,
    )
    _assert_packed_results_close(ref, cand)


def _packed_inputs(
    *,
    lengths: tuple[int, ...],
    channels: int,
    kernel_width: int,
    has_bias: bool,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cu_values = [0]
    for length in lengths:
        cu_values.append(cu_values[-1] + length)
    total_tokens = cu_values[-1]
    conv_in = torch.randn(
        total_tokens,
        channels,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    conv_initial = torch.randn(
        len(lengths),
        channels,
        kernel_width - 1,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    weight = torch.randn(
        channels, kernel_width, device="cuda", dtype=torch.float32, generator=generator
    )
    bias = (
        torch.randn(channels, device="cuda", dtype=torch.float32, generator=generator)
        if has_bias
        else None
    )
    out_grad = torch.randn(
        total_tokens,
        channels,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    final_grad = torch.randn(
        len(lengths),
        channels,
        kernel_width - 1,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    cu_seqlens = torch.tensor(cu_values, device="cuda", dtype=torch.int32)
    return conv_in, cu_seqlens, conv_initial, weight, bias, out_grad, final_grad


def _run_packed_reference(
    conv_in: Tensor,
    cu_seqlens: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out_grad: Tensor,
    final_grad: Tensor,
    *,
    activation: str,
) -> dict[str, Tensor | None]:
    conv_in = conv_in.detach().clone().requires_grad_(True)
    conv_initial = conv_initial.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    pieces = []
    for segment in range(int(cu_seqlens.numel()) - 1):
        start = int(cu_seqlens[segment].item())
        end = int(cu_seqlens[segment + 1].item())
        segment_in = conv_in[start:end].transpose(0, 1).unsqueeze(0)
        extended = torch.cat([conv_initial[segment : segment + 1], segment_in], dim=-1)
        out = F.conv1d(extended, weight.unsqueeze(1), bias, groups=conv_in.shape[1])
        pieces.append(_torch_activation(out.squeeze(0).transpose(0, 1), activation))
    out = torch.cat(pieces, dim=0)
    final = _packed_reference_final(conv_in, cu_seqlens, conv_initial)
    ((out * out_grad).sum() + (final * final_grad).sum()).backward()
    return _packed_result(conv_in, conv_initial, weight, bias, out, final)


def _run_packed_fused(
    conv_in: Tensor,
    cu_seqlens: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out_grad: Tensor,
    final_grad: Tensor,
    *,
    activation: str,
    split_silu: bool = False,
) -> dict[str, Tensor | None]:
    conv_in = conv_in.detach().clone().requires_grad_(True)
    conv_initial = conv_initial.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    out, final = packed_varlen_causal_conv(
        conv_in,
        cu_seqlens,
        conv_initial,
        weight,
        bias,
        activation="none" if split_silu else activation,
        output_final_state=True,
    )
    assert final is not None
    if split_silu:
        out = F.silu(out)
    ((out * out_grad).sum() + (final * final_grad).sum()).backward()
    return _packed_result(conv_in, conv_initial, weight, bias, out, final)


def _packed_reference_final(
    conv_in: Tensor, cu_seqlens: Tensor, conv_initial: Tensor
) -> Tensor:
    tail_width = int(conv_initial.shape[-1])
    if tail_width == 0:
        return conv_initial
    pieces = []
    for segment in range(int(cu_seqlens.numel()) - 1):
        start = int(cu_seqlens[segment].item())
        end = int(cu_seqlens[segment + 1].item())
        extended = torch.cat([conv_initial[segment], conv_in[start:end].T], dim=-1)
        length = end - start
        pieces.append(extended[:, length : length + tail_width])
    return torch.stack(pieces, dim=0)


def _torch_activation(tensor: Tensor, activation: str) -> Tensor:
    if activation == "none":
        return tensor
    if activation in ("silu", "swish"):
        return F.silu(tensor)
    if activation == "gelu":
        return F.gelu(tensor)
    raise ValueError(activation)


def _packed_result(
    conv_in: Tensor,
    conv_initial: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out: Tensor,
    final: Tensor,
) -> dict[str, Tensor | None]:
    return {
        "out": out.detach(),
        "final": final.detach(),
        "conv_in_grad": _required_grad(conv_in.grad),
        "conv_initial_grad": _required_grad(conv_initial.grad),
        "weight_grad": _required_grad(weight.grad),
        "bias_grad": None if bias is None else _required_grad(bias.grad),
    }


def _required_grad(grad: Tensor | None) -> Tensor:
    if grad is None:
        raise AssertionError("missing gradient")
    return grad.detach()


def _assert_packed_results_close(
    reference: dict[str, Tensor | None], candidate: dict[str, Tensor | None]
) -> None:
    for name in ("out", "final", "conv_in_grad", "conv_initial_grad", "weight_grad"):
        ref_tensor = reference[name]
        cand_tensor = candidate[name]
        assert ref_tensor is not None and cand_tensor is not None
        assert ref_tensor.dtype == torch.float32
        assert cand_tensor.dtype == torch.float32
        if ref_tensor.numel() > 0:
            assert torch.any(ref_tensor != 0), f"{name} reference is all zero"
        assert_mean_abs_pct(ref_tensor, cand_tensor, name)
    if reference["bias_grad"] is not None:
        assert candidate["bias_grad"] is not None
        assert reference["bias_grad"].dtype == torch.float32
        assert candidate["bias_grad"].dtype == torch.float32
        assert_mean_abs_pct(reference["bias_grad"], candidate["bias_grad"], "bias_grad")
