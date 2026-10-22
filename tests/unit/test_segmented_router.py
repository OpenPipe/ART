"""Native router semantics must remain local to each adapter's token rows."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist

from art.megatron.segmented_expert_lora import install_segmented_dispatcher
from art.megatron.segmented_lora import LoRASegments, use_lora_segments


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    path = tmp_path_factory.mktemp("router-gloo") / "store"
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


@pytest.mark.parametrize("counts", [(3, 11), (3, 0, 11)])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize(
    "aux,z,capacity", [(0.3, None, None), (0, 0.2, None), (0, None, 0.5)]
)
def test_aux_z_and_capacity_match_independent_adapters(
    monkeypatch, group, per_token, aux, z, capacity, counts
):
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe import router as router_module
    from megatron.core.transformer.moe import token_dispatcher
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Keep all actual routing/loss/autograd math; only GPU gate movement and
    # logging are outside this CPU operator fixture.
    monkeypatch.setattr(
        router_module, "save_to_aux_losses_tracker", lambda *a, **k: None
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss" if aux else "global_aux_loss",
        moe_aux_loss_coeff=aux,
        moe_z_loss_coeff=z,
        moe_expert_capacity_factor=capacity,
        calculate_per_token_loss=per_token,
        add_bias_linear=False,
        moe_router_fusion=False,
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = groups.tp_cp = groups.tp_dp_cp = group
    router = router_module.TopKRouter(config, pg_collection=groups)
    router.set_layer_number(1)
    monkeypatch.setattr(router, "gating", lambda value: value)

    class Dispatcher:
        tp_size = ep_size = 1
        drop_and_pad = False

        def dispatch_postprocess(self, hidden, probabilities):
            return hidden, probabilities

    monkeypatch.setattr(token_dispatcher, "MoEAlltoAllTokenDispatcher", Dispatcher)
    layer = SimpleNamespace(config=config, router=router, token_dispatcher=Dispatcher())
    torch.manual_seed(17)
    logits = torch.randn(sum(counts), 1, 4)
    logits[:3, :, 0] += 4
    logits[3:, :, 1] += 4
    padding = torch.zeros(sum(counts), 1, dtype=torch.bool)
    padding[-2:] = True

    def run(fused):
        value = logits.clone().requires_grad_()
        if fused:
            segments = LoRASegments(
                cast(Any, tuple(f"run-{i}" for i in range(len(counts)))), counts
            )
            with use_lora_segments(segments):
                probabilities, routes = router(value, padding_mask=padding)
        else:
            results = [
                router(part, padding_mask=mask)
                for part, mask in zip(
                    value.split(counts), padding.split(counts), strict=True
                )
                if part.shape[0]
            ]
            probabilities, routes = (
                torch.cat(parts) for parts in zip(*results, strict=True)
            )
        probabilities.sum().backward()
        return probabilities, routes, value.grad

    expected = run(False)
    install_segmented_dispatcher(layer)
    installed = router.forward
    install_segmented_dispatcher(layer)
    assert router.forward is installed
    actual = run(True)
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
    # Installation alone never changes the ordinary path.
    for candidate, reference in zip(run(False), expected, strict=True):
        torch.testing.assert_close(candidate, reference, atol=0, rtol=0)
