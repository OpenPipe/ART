"""CP preparation must deliver the same model-state inputs as dense packing."""

from types import SimpleNamespace
from typing import Any, cast

from megatron.core import parallel_state as ps
import pytest
import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.megatron.glm5next.dsa import Glm5NextSelfAttention
from art.megatron.glm5next.dsa_cp import Glm5NextDsaState
from art.megatron.model_support.handlers.glm5next import Glm5NextHandler
from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import TrainerRank


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_cp_builds_real_glm_state_with_local_layout_and_dimensions(
    monkeypatch, cp_rank
):
    from art.megatron.training import microbatches

    contexts = []

    class Handler(Glm5NextHandler):
        def build_prefix_tree_model_state(self, context: Any) -> dict[str, Any]:
            contexts.append(context)
            return super().build_prefix_tree_model_state(context)

    handler = Handler()
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = cast(
        Any,
        SimpleNamespace(
            provider=SimpleNamespace(kv_channels=8), model_support_handler=handler
        ),
    )
    trainer.device = torch.device("cpu")
    monkeypatch.setattr(ps, "get_context_parallel_group", lambda **_: None)
    monkeypatch.setattr(ps, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(
        microbatches, "_gdn_planner_config_for_provider", lambda *_: None
    )
    batch = prefix_tree_pack([torch.arange(64)], max_depth=0)
    topology = ParallelTopology(tp=1, cp=2, pp=1, dp=1)
    prepared = trainer._prepare_context_parallel_forward(batch, topology=topology)
    assert len(contexts) == 1
    context = contexts[0]
    assert context.attention_head_dim == context.attention_value_head_dim == 8
    assert context.context_parallel_state is prepared.attention_state
    assert (
        context.attention_token_layout_index
        is prepared.attention_state.rank_plan.token_layout_index
    )
    torch.testing.assert_close(context.input_pos, prepared.position_ids)
    torch.testing.assert_close(context.group_ids, batch.group_ids)
    torch.testing.assert_close(context.parent_ids, batch.parent_ids)
    attention = Glm5NextSelfAttention.__new__(Glm5NextSelfAttention)
    state = attention._attention_state(prepared.attention_state)
    assert isinstance(state, Glm5NextDsaState)
    assert state.context_parallel_state is prepared.attention_state
