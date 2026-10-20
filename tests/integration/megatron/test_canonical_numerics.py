"""One-GPU dense decoder regression with recomputation and real ART/TE hooks."""

from typing import Any

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_canonical_decoder_gradients_survive_repacking(monkeypatch, tmp_path):
    monkeypatch.setenv("ART_DISABLE_MEGATRON_COMPILE", "1")
    monkeypatch.setenv("ART_MEGATRON_LORA_RANK", "8")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA GPU and megatron_runtime")
    from megatron.core import parallel_state
    import torch.distributed as dist

    from art.megatron.canonical import RootLayout, execution, install
    from art.megatron.lora import LoRA
    from art.megatron.prefix_tree_packing import prefix_tree_pack
    from art.megatron.prefix_tree_state import create_prefix_tree_state
    from art.megatron.train import _placeholder_attention_mask, build_training_runtime
    from art.megatron.training.microbatches import _gdn_planner_config_for_provider

    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    runtime = None
    try:

        def configure(provider):
            provider.hidden_dropout = provider.attention_dropout = 0.0
            provider.num_layers = 4
            if isinstance(getattr(provider, "linear_attention_freq", None), list):
                provider.linear_attention_freq = provider.linear_attention_freq[:4]
            provider.recompute_granularity = "full"
            provider.recompute_method = "uniform"
            provider.recompute_num_layers = 1

        torch.manual_seed(731)
        runtime = build_training_runtime(
            model_identifier="Qwen/Qwen3.5-4B",
            model_initialization="random",
            provider_configure=configure,
            model_support_key="qwen3_5_dense",
            build_optimizer=False,
            print_env=False,
        )
        model: Any = runtime.model[0]
        while hasattr(model, "module"):
            model = model.module
        model = getattr(model, "language_model", model)
        model.train()
        report = install(model)
        assert "attention" in report.values() and "row" in report.values()
        assert install(model) == report
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRA):
                    module.B_T.normal_(std=1e-3)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        generator = torch.Generator().manual_seed(42)
        roots = [
            torch.randint(100, 10000, (n,), generator=generator) for n in (129, 257)
        ]
        reference = None
        # Compare the same datum's loss/gradients packed, reversed, alone, then
        # packed again. Mask companions; preserve the datum's normalization.
        for order in ((0, 1), (1, 0), (0,), (0, 1)):
            layout = RootLayout(tuple(len(roots[i]) for i in order))
            pack = prefix_tree_pack(
                [RootLayout((len(roots[i]),)).pad([roots[i]]) for i in order],
                max_depth=0,
            )
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
            for chunk in runtime.model:
                if hasattr(chunk, "zero_grad_buffer"):
                    chunk.zero_grad_buffer()
                    for group in (
                        chunk.bucket_groups + chunk.expert_parallel_bucket_groups
                    ):
                        group.is_last_microbatch = False
            for param in params.values():
                param.grad = None
            with execution(layout):
                pre = model._preprocess(
                    input_ids=pack.tokens.cuda(),
                    position_ids=pack.position_ids.cuda(),
                    packed_seq_params=None,
                )
                hidden, rot, cos, sin, offset, padding = pre[:6]
                output = model.decoder(
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
                datum = layout.unpad(output)[order.index(0)]
                datum.float().square().mean().backward()
            gradients = {}
            for name, param in params.items():
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                assert grad is not None, name
                gradients[name] = grad.detach().cpu().clone()
            assert gradients and any(torch.count_nonzero(g) for g in gradients.values())
            actual = datum.detach().cpu(), gradients
            if reference is None:
                reference = actual
            else:
                assert torch.equal(reference[0], actual[0])
                assert reference[1].keys() == gradients.keys()
                for name, grad in gradients.items():
                    assert torch.equal(reference[1][name], grad), name
    finally:
        del runtime
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
