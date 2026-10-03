from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.bridge")
pytest.importorskip("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TELayerNormColumnParallelLinear,
)

from art.megatron.lora import (  # noqa: E402
    ComponentwiseColumnParallelLinearLoRA,
    LoRASlotRef,
    SelfAttentionLinearQKVLoRA,
    SharedExpertsLinearFC1LoRA,
    use_lora_slot,
)

from ..gdn_shared_prefix.metrics import GDN_CORRECTNESS_DTYPE  # noqa: E402
from ..gdn_shared_prefix.test_real_gdn_cp1_packed_vs_flattened import (  # noqa: E402
    _single_rank_model_parallel,
)
from ..gdn_shared_prefix.test_real_gdn_tp_lora import (  # noqa: E402
    _make_matching_gdn_pair,
    _make_provider,
)


def _projection(family: str):
    if family == "gdn":
        gdn, _ = _make_matching_gdn_pair(tp_size=1, lora=True)
        projection = gdn.in_proj
        return projection, projection.in_proj, (projection.qkv_lora, projection.z_lora)
    provider = _make_provider(tp_size=1)
    output_size = {"qkv": 160, "componentwise": 96, "shared_fc1": 128}[family]
    inner = TELayerNormColumnParallelLinear(
        provider.hidden_size,
        output_size,
        config=provider,
        init_method=provider.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=True,
        is_expert=False,
    )
    if family == "qkv":
        projection = SelfAttentionLinearQKVLoRA(
            "test.attention", inner, 1, 1, provider, {"q_proj", "k_proj", "v_proj"}
        )
        loras = (projection.q_proj_lora, projection.k_proj_lora, projection.v_proj_lora)
    elif family == "componentwise":
        projection = ComponentwiseColumnParallelLinearLoRA(
            "test.components", inner, (32, 64), 1, 1
        )
        loras = (projection.lora,)
    else:
        projection = SharedExpertsLinearFC1LoRA("test.shared", inner, 1, 1)
        loras = (projection.gate_lora, projection.up_lora)
    assert all(lora is not None for lora in loras)
    return projection, inner, loras


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for real fused TE projection backward.",
)
@pytest.mark.parametrize(
    ("family", "adapter"),
    [
        (family, adapter)
        for family in ("gdn", "qkv", "componentwise", "shared_fc1")
        for adapter in ("active", "restored")
    ]
    + [("gdn", "zero_B"), ("gdn", "inactive")],
)
def test_tp1_lora_returned_norm_input_gradient(family: str, adapter: str) -> None:
    with _single_rank_model_parallel():
        projection, inner, loras = _projection(family)
        assert inner.tp_size == 1 and not inner.sequence_parallel
        projection.requires_grad_(False)
        ref = LoRASlotRef("checkpoint", "norm-gradient")
        with torch.no_grad():
            # A base input gradient must not mask a lost returned-norm cotangent.
            inner.weight.zero_()
            inner.layer_norm_weight.fill_(0 if inner.zero_centered_gamma else 1)
            assert inner.layer_norm_bias is None
            for lora in loras:
                lora.A_T.zero_()
                lora.B_T.zero_()
                a = torch.arange(lora.in_features, device=lora.A_T.device)
                lora.A_T[:, 0].copy_((a % 7 - 3).float() / 8)
                if adapter != "zero_B":
                    lora.B_T[0].fill_(1 / 8)
                if adapter == "restored":
                    prefix = lora.adapter_model_prefix
                    assert lora.load_lora_slot(
                        ref,
                        {
                            f"{prefix}.lora_A.weight": lora.A_T.T.clone(),
                            f"{prefix}.lora_B.weight": lora.B_T.T.clone(),
                        },
                        alpha=lora.alpha,
                        requires_grad=False,
                    )
                    # Only the restored slot may supply the adapter input edge.
                    lora.A_T.zero_()
                    lora.B_T.zero_()
        selected = ref if adapter == "restored" else None
        if adapter == "inactive":
            selected = LoRASlotRef("checkpoint", None)
        x = torch.arange(8 * 64, device=inner.weight.device).reshape(8, 1, 64)
        x = ((x % 17 - 8).float() / 8).to(GDN_CORRECTNESS_DTYPE).requires_grad_()
        norms = []

        def retain_norm(module, args, output):
            norm = output[0][1]
            assert norm.shape == x.shape and norm.requires_grad
            norm.retain_grad()
            norms.append(norm)

        with use_lora_slot(selected):
            for lora in loras:
                active = lora.active_lora_tensors()
                if adapter == "inactive":
                    assert active is None
                else:
                    assert active is not None
                    assert bool(torch.count_nonzero(active[1])) == (adapter != "zero_B")
            handle = inner.register_forward_hook(retain_norm)
            try:
                output, bias = projection(x)
            finally:
                handle.remove()
            assert bias is None and len(norms) == 1
            weights = torch.arange(output.shape[-1], device=x.device) % 5 + 1
            (dx,) = torch.autograd.grad((output.float() * weights).sum(), x)
        norm_grad = norms.pop().grad
        assert torch.isfinite(output).all() and torch.isfinite(dx).all()
        if adapter in ("active", "restored"):
            assert norm_grad is not None and torch.isfinite(norm_grad).all()
            assert torch.count_nonzero(norm_grad) > 0
            assert torch.count_nonzero(output) > 0
            assert torch.count_nonzero(dx) > 0
        else:
            assert norm_grad is None or torch.count_nonzero(norm_grad) == 0
            assert torch.count_nonzero(output) == 0
            assert torch.count_nonzero(dx) == 0
