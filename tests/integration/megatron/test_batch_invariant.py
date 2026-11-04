"""Explicit-grant GPU gates for default trainer arithmetic."""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("ART_TEST_BATCH_INVARIANT_GPU") != "1",
        reason="requires a GPU grant and ART_TEST_BATCH_INVARIANT_GPU=1",
    ),
]
LENGTHS = (1, 4, 5, 63, 64, 65, 129, 1000)


def _configure_tiny_provider(provider, *, moe):
    provider.hidden_dropout = provider.attention_dropout = 0.0
    provider.num_layers, provider.hidden_size = 2, 128
    provider.num_attention_heads, provider.num_query_groups = 4, 2
    provider.kv_channels, provider.ffn_hidden_size = 32, 256
    # Qwen's 25% rotary width is now 8; its three sections cover half of it.
    provider.mrope_section = [1, 1, 2]
    provider.linear_attention_freq = [0, 1]
    if moe:
        provider.num_moe_experts, provider.moe_router_topk = 4, 2
        provider.moe_ffn_hidden_size = 128


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefix_batch_and_backward(dtype):
    from art.megatron.batch_invariant.ops import linear, logsumexp, rms_norm

    torch.manual_seed(808)
    x = torch.randn(1000, 131, device="cuda", dtype=dtype)
    w = torch.randn(67, 131, device="cuda", dtype=dtype) * 0.1
    gamma = torch.randn(131, device="cuda", dtype=dtype)

    def forward(xx):
        return linear(rms_norm(xx, gamma, 1e-6), w)

    reference = forward(x)
    for length in LENGTHS:
        xx = x[:length].detach().requires_grad_()
        actual = forward(xx)
        assert torch.equal(actual, reference[:length]), length
        assert torch.equal(logsumexp(actual), logsumexp(reference)[:length])
        other = torch.cat((torch.randn_like(x[:129]) * 100, xx))
        assert torch.equal(forward(other)[129:], actual), length
        (grad,) = torch.autograd.grad(actual.sum(), xx)
        longer = x.detach().requires_grad_()
        dy = torch.zeros_like(reference)
        dy[:length] = 1
        (long_grad,) = torch.autograd.grad(forward(longer), longer, dy)
        assert torch.equal(grad, long_grad[:length]), length
    xx = x[:5].detach().float().requires_grad_()
    ww, gg = (
        w.float().detach().requires_grad_(),
        gamma.float().detach().requires_grad_(),
    )
    actual = linear(rms_norm(xx, gg, 1e-6), ww)
    ga = torch.autograd.grad(logsumexp(actual).sum(), (xx, ww, gg))
    rx, rw, rg = [t.detach().cpu().double().requires_grad_() for t in (xx, ww, gg)]
    expected = torch.nn.functional.linear(
        rx * torch.rsqrt(rx.square().mean(-1, keepdim=True) + 1e-6) * rg, rw
    )
    ge = torch.autograd.grad(expected.logsumexp(-1).sum(), (rx, rw, rg))
    torch.testing.assert_close(actual.cpu().double(), expected, atol=5e-6, rtol=3e-5)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a.cpu().double(), e, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("shared", [None, "A", "B"])
def test_grouped_experts_and_gradients(shared):
    from art.megatron.batch_invariant.ops import grouped_lora

    torch.manual_seed(13)
    x = torch.randn(1000, 35, device="cuda", requires_grad=True)
    a = torch.randn(1 if shared == "A" else 2, 35, 8, device="cuda", requires_grad=True)
    b = torch.randn(1 if shared == "B" else 2, 8, 37, device="cuda", requires_grad=True)
    reference = grouped_lora(x, a, b, [1000, 0])
    for length in LENGTHS:
        assert torch.equal(
            grouped_lora(x[:length], a, b, [length, 0]), reference[:length]
        )
        moved = torch.cat((x[:129], x[:length]))
        actual = grouped_lora(moved, a, b, [129, length])[129:]
        same = grouped_lora(
            x[:length],
            a[:1] if shared == "A" else a[1:],
            b[:1] if shared == "B" else b[1:],
            [length],
        )
        assert torch.equal(actual, same)
    actual = grouped_lora(x[:5], a, b, [2, 3])
    ga = torch.autograd.grad(actual.square().sum(), (x, a, b))
    rx, ra, rb = [t.detach().cpu().double().requires_grad_() for t in (x, a, b)]
    expected = torch.cat(
        [
            (xx @ ra[0 if shared == "A" else e]) @ rb[0 if shared == "B" else e]
            for e, xx in enumerate(rx[:5].split([2, 3]))
        ]
    )
    ge = torch.autograd.grad(expected.square().sum(), (rx, ra, rb))
    for aa, ee in zip(ga, ge, strict=True):
        torch.testing.assert_close(aa.cpu().double(), ee, atol=0.01, rtol=2e-4)


@pytest.mark.parametrize("moe", [False, True])
def test_tiny_trainer_prefix_packing_and_autograd(monkeypatch, tmp_path, moe):
    """Real TE/GDN/Flash/expert dispatch, with the unconditional pre-DDP hook."""
    monkeypatch.setenv("ART_DISABLE_MEGATRON_COMPILE", "1")
    monkeypatch.setenv("ART_MEGATRON_LORA_RANK", "8")
    for name in (
        "TENSOR_MODEL_PARALLEL",
        "EXPERT_MODEL_PARALLEL",
        "CONTEXT_PARALLEL",
        "PIPELINE_MODEL_PARALLEL",
    ):
        monkeypatch.setenv(f"ART_MEGATRON_{name}_SIZE", "1")
    from megatron.core import parallel_state
    import torch.distributed as dist

    from art.megatron.lora import LoRA
    from art.megatron.prefix_tree_packing import prefix_tree_pack
    from art.megatron.prefix_tree_state import create_prefix_tree_state
    from art.megatron.train import _placeholder_attention_mask, build_training_runtime
    from art.megatron.training.microbatches import _gdn_planner_config_for_provider

    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1
    )
    runtime = None
    try:
        runtime = build_training_runtime(
            model_identifier="Qwen/Qwen3.6-35B-A3B" if moe else "Qwen/Qwen3.8-27B",
            model_support_key="qwen3_5_moe" if moe else "qwen3_5_dense",
            model_initialization="random",
            provider_configure=lambda provider: _configure_tiny_provider(
                provider, moe=moe
            ),
            build_optimizer=False,
            print_env=False,
        )
        model = runtime.model[0]
        while hasattr(model, "module"):
            model = model.module
        model = getattr(model, "language_model", model)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRA):
                    module.B_T.normal_(std=1e-3)
        roots = torch.randint(100, 10000, (1000,))

        def forward(parts):
            pack = prefix_tree_pack(parts, max_depth=0)
            state = create_prefix_tree_state(
                pack.group_ids,
                pack.parent_ids,
                input_pos=pack.position_ids,
                target_device=torch.device("cuda"),
                build_gdn_execution_spec=runtime.model_support_handler.build_gdn_execution_spec,
                model_support_handler=runtime.model_support_handler,
                attention_head_dim=runtime.provider.kv_channels,
                attention_value_head_dim=runtime.provider.kv_channels,
                gdn_planner_config=_gdn_planner_config_for_provider(
                    runtime.provider, runtime.model_support_handler
                ),
            )
            pre = model._preprocess(
                input_ids=pack.tokens.cuda(),
                position_ids=pack.position_ids.cuda(),
                packed_seq_params=None,
            )
            hidden, rot, cos, sin, offset, padding = pre[:6]
            return model.decoder(
                hidden_states=hidden,
                attention_mask=_placeholder_attention_mask(torch.device("cuda")),
                rotary_pos_emb=rot,
                rotary_pos_cos=cos,
                rotary_pos_sin=sin,
                rotary_pos_cos_sin=pre[6] if len(pre) == 7 else None,
                packed_seq_params=None,
                sequence_len_offset=offset,
                padding_mask=padding,
                attention_bias=state,
            )

        model.eval()
        with torch.no_grad():
            reference = forward([roots])
            for length in LENGTHS:
                actual = forward([roots[:length]])
                assert torch.equal(actual[:length], reference[:length]), length
                packed = forward([torch.flip(roots[:65], (0,)), roots[:length]])
                assert torch.equal(actual[:length], packed[65 : 65 + length]), length
        model.train()
        forward([roots[:5]]).float().square().sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad = getattr(param, "main_grad", None)
                grad = param.grad if grad is None else grad
                assert grad is not None and torch.isfinite(grad).all(), name
    finally:
        del runtime
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("zero_centered", [False, True])
def test_gdn_gated_norm_backward_fp64(zero_centered):
    from art.megatron.batch_invariant.ops import rms_norm

    torch.manual_seed(820)
    x = torch.randn(5, 131, device="cuda", requires_grad=True)
    w = torch.randn(131, device="cuda", requires_grad=True)
    gate = torch.randn_like(x, requires_grad=True)
    out = rms_norm(x, w, 1e-6, zero_centered=zero_centered) * torch.nn.functional.silu(
        gate
    )
    dy = torch.randn_like(out)
    ga = torch.autograd.grad(out, (x, w, gate), dy)
    rx, rw, rg = [t.detach().cpu().double().requires_grad_() for t in (x, w, gate)]
    expected = (
        rx
        * torch.rsqrt(rx.square().mean(-1, keepdim=True) + 1e-6)
        * (rw + int(zero_centered))
    ) * torch.nn.functional.silu(rg)
    ge = torch.autograd.grad(expected, (rx, rw, rg), dy.cpu().double())
    torch.testing.assert_close(out.cpu().double(), expected, atol=2e-6, rtol=2e-5)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a.cpu().double(), e, atol=5e-6, rtol=1e-4)
