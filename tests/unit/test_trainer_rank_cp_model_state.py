"""CP preparation must deliver the same model-state inputs as dense packing."""

from types import SimpleNamespace
from typing import Any, cast

from megatron.core import parallel_state as ps
import pytest
import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.megatron.glm52.state import Glm52PrefixTreeState
from art.megatron.model_support.handlers.glm52 import Glm52Handler
from art.megatron.model_support.spec import PrefixTreeModelStateContext
from art.megatron.prefix_tree_packing import prefix_tree_pack
from art.trainer_rank import TrainerRank


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_cp_builds_glm52_state_with_local_layout_and_dimensions(
    monkeypatch: pytest.MonkeyPatch, cp_rank: int
) -> None:
    contexts: list[PrefixTreeModelStateContext] = []

    class Handler(Glm52Handler):
        def context_parallel_workload_profile(self, provider: Any) -> None:
            # This test needs the real state builder, not a full model provider.
            return None

        def build_prefix_tree_model_state(
            self, context: PrefixTreeModelStateContext
        ) -> dict[str, Any]:
            contexts.append(context)
            return super().build_prefix_tree_model_state(context)

    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = cast(
        Any,
        SimpleNamespace(
            provider=SimpleNamespace(kv_channels=8),
            model_support_handler=Handler(),
        ),
    )
    trainer.device = torch.device("cpu")
    monkeypatch.setattr(ps, "get_context_parallel_group", lambda **_: None)
    monkeypatch.setattr(ps, "get_context_parallel_rank", lambda: cp_rank)
    batch = prefix_tree_pack([torch.arange(64)], max_depth=0)
    topology = ParallelTopology(tp=1, cp=2, pp=1, dp=1)

    prepared = trainer._prepare_context_parallel_forward(batch, topology=topology)

    assert len(contexts) == 1
    context = contexts[0]
    assert context.attention_head_dim == context.attention_value_head_dim == 8
    assert context.device == torch.device("cpu")
    cp_state = prepared.attention_state
    assert context.context_parallel_state is cp_state
    assert cp_state.rank_plan.rank == cp_rank
    assert context.attention_token_layout_index is cp_state.rank_plan.token_layout_index
    torch.testing.assert_close(context.input_pos, prepared.position_ids)
    torch.testing.assert_close(context.group_ids, batch.group_ids)
    torch.testing.assert_close(context.parent_ids, batch.parent_ids)
    state = cp_state.model_state["glm52"]
    assert isinstance(state, Glm52PrefixTreeState)
    assert state.context_parallel_state is cp_state
    torch.testing.assert_close(state.position_ids, prepared.position_ids)
    assert state.stages
