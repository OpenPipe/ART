from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("megatron.core")

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttentionSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig

from art.megatron.glm5next.dsa import (
    Glm5NextSelfAttention,
    Glm5NextSparseCore,
    glm5next_core_builder,
)
from art.megatron.glm5next.dsa_cp import (
    build_glm5next_dsa_state,
    dsa_pack_alignment,
    tree_pooled_indexer_topk,
)
from art.megatron.glm5next.kpool_indexer import KPoolIndexer
from art.megatron.glm52 import attention as glm52_attention


class _Linear(nn.Linear):
    def __init__(self, n: int, m: int, **kwargs: Any) -> None:
        super().__init__(n, m, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:  # ty: ignore[invalid-method-override]
        return F.linear(x, self.weight), None


class _Norm(nn.LayerNorm):
    def __init__(self, *, hidden_size: int, eps: float, **kwargs: Any) -> None:
        super().__init__(hidden_size, eps=eps)


def _attention() -> Glm5NextSelfAttention:
    config = MLATransformerConfig(
        num_layers=4,
        hidden_size=32,
        num_attention_heads=2,
        num_query_groups=2,
        kv_channels=8,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_head_dim=8,
        qk_pos_emb_head_dim=0,
        v_head_dim=8,
        layernorm_epsilon=1e-5,
        attention_dropout=0,
        hidden_dropout=0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=8,
        dsa_indexer_loss_coeff=0.0,
    )
    modules = MLASelfAttentionSubmodules(
        linear_q_down_proj=_Linear,
        linear_q_up_proj=_Linear,
        linear_kv_down_proj=_Linear,
        linear_kv_up_proj=_Linear,
        core_attention=glm5next_core_builder(_Linear, _Norm),
        linear_proj=_Linear,
        q_layernorm=_Norm,
        kv_layernorm=_Norm,
    )
    return Glm5NextSelfAttention(
        config,
        modules,
        4,
        AttnMaskType.causal,
        pg_collection=SimpleNamespace(tp=None, cp=None),
    )


def _dense_sparse(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, *, scale: float, **_: Any
) -> torch.Tensor:
    # Independent dense mask; never gather a sentinel as an actual KV token.
    mask = (indices[..., None] == torch.arange(kv.shape[1])[None, None, None, :]).any(
        -2
    )
    assert q.shape[-1] == kv.shape[-1] == 576
    assert not bool(q[..., 512:].any()) and not bool(kv[..., 512:].any())
    scores = torch.einsum("bqhd,bkd->bhqk", q, kv) * scale
    probs = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1).nan_to_num()
    return torch.einsum("bhqk,bkd->bqhd", probs, kv[..., :512])


def test_mcore_nope_forward_backward_matches_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    torch.manual_seed(823)
    attention = _attention()
    config = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=16,
        kv_lora_rank=512,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=8,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        indexer_types=["full"] * 4,
        _attn_implementation="eager",
    )
    reference = hf.Glm5NextTextAttention(config, 3)
    mapping = {
        "q_a_proj": "linear_q_down_proj",
        "q_b_proj": "linear_q_up_proj",
        "kv_a_proj_with_mqa": "linear_kv_down_proj",
        "kv_b_proj": "linear_kv_up_proj",
        "o_proj": "linear_proj",
        "q_a_layernorm": "q_layernorm",
        "kv_a_layernorm": "kv_layernorm",
    }
    for source, target in mapping.items():
        getattr(attention, target).weight.data.copy_(getattr(reference, source).weight)
    assert isinstance(attention.core_attention, Glm5NextSparseCore)
    indexer = attention.core_attention.indexer
    assert isinstance(indexer, KPoolIndexer)
    indexer.load_state_dict(
        {
            key.replace("wq_b.", "linear_wq_b.")
            .replace("wk.", "linear_wk.")
            .replace("weights_proj.", "linear_weights_proj."): value
            for key, value in reference.indexer.state_dict().items()
        }
    )
    monkeypatch.setattr(glm52_attention, "sparse_mla", _dense_sparse)
    length = 24
    state = build_glm5next_dsa_state(
        position_ids=torch.arange(length)[None],
        group_ids=torch.zeros(1, length, dtype=torch.long),
        parent_ids=torch.full((1, length), -1),
        device=torch.device("cpu"),
    )
    x = torch.randn(length, 1, 32, requires_grad=True)
    rx = x.detach().transpose(0, 1).clone().requires_grad_()
    actual, _ = attention(
        x, attention_bias=SimpleNamespace(model_state={"glm5next_dsa": state})
    )
    expected = reference(rx, attention_mask=torch.ones(1, length, dtype=torch.bool))[
        0
    ].transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    assert x.grad is not None and rx.grad is not None
    torch.testing.assert_close(x.grad, rx.grad.transpose(0, 1), rtol=5e-5, atol=3e-6)
    for source, target in mapping.items():
        torch.testing.assert_close(
            getattr(attention, target).weight.grad,
            getattr(reference, source).weight.grad,
            rtol=1e-4,
            atol=1e-5,
        )
    assert all(p.grad is None for p in indexer.parameters())


def test_prefix_tree_excludes_siblings_and_padding() -> None:
    groups = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4 + [-1] * 4])
    parents = torch.tensor([[-1] * 4 + [0] * 8 + [-1] * 4])
    state = build_glm5next_dsa_state(
        position_ids=torch.arange(16)[None],
        group_ids=groups,
        parent_ids=parents,
        device=torch.device("cpu"),
    )
    indexer = KPoolIndexer(
        hidden_size=8, q_lora_rank=8, num_heads=2, head_dim=8, topk=16
    )
    projected = indexer.project(torch.randn(16, 1, 8), torch.randn(16, 1, 8))
    ids = tree_pooled_indexer_topk(indexer, projected, state)
    assert isinstance(ids, torch.Tensor)
    for query in range(16):
        got = set(ids[0, query][ids[0, query] < 16].tolist())
        expected = (
            set(range(query + 1))
            if query < 8
            else set(range(4)) | set(range(8, query + 1))
            if query < 12
            else set()
        )
        assert got == expected


def test_pack_alignment_and_bad_tree_boundary() -> None:
    assert [dsa_pack_alignment(cp) for cp in (1, 2, 8)] == [8, 16, 64]
    with pytest.raises(ValueError, match="align"):
        build_glm5next_dsa_state(
            position_ids=torch.arange(12)[None],
            group_ids=torch.tensor([[0] * 3 + [1] * 9]),
            parent_ids=torch.tensor([[-1] * 3 + [0] * 9]),
            device=torch.device("cpu"),
        )


def test_cp_keeps_future_pool_slots_for_hf_tie_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from art.megatron.glm5next import dsa_cp
    from tests.integration.megatron.glm5next_dsa_parity import metadata

    indexer = KPoolIndexer(
        hidden_size=8, q_lora_rank=8, num_heads=2, head_dim=8, topk=8
    )
    # Every visible pool is tied; truncating invisible slots changes torch.topk.
    projected = (
        torch.zeros(1, 64, 2, 8),
        torch.zeros(1, 64, 8),
        torch.ones(1, 64, 2),
        torch.zeros(1, 64, 8),
    )
    state1, _ = metadata(64, 1, 0, torch.device("cpu"))
    expected = tree_pooled_indexer_topk(indexer, projected, state1)
    assert isinstance(expected, torch.Tensor)
    for rank in range(2):
        state, local_ids = metadata(64, 2, rank, torch.device("cpu"))
        local = tuple(value[:, local_ids].contiguous() for value in projected)
        monkeypatch.setattr(dsa_cp, "launch_remote_stage_fetches", lambda *args: {})

        def stage_rows(
            tensor: torch.Tensor, plan: Any, cp: Any, works: Any
        ) -> torch.Tensor:
            stage = next(
                stage for stage in state.stages if stage.stage_index == plan.stage_index
            )
            return torch.cat(
                (
                    projected[1][0, stage.global_k_ids.long()],
                    projected[3][0, stage.global_k_ids.long()],
                ),
                -1,
            )

        monkeypatch.setattr(dsa_cp, "stage_kv_rows", stage_rows)
        actual = tree_pooled_indexer_topk(indexer, cast(Any, local), state)
        assert not isinstance(actual, torch.Tensor)
        routes = actual.indices
        assert state.combined_global_ids is not None
        global_ids = torch.where(
            routes < state.combined_k_rows,
            state.combined_global_ids[
                routes.clamp_max(state.combined_k_rows - 1).long()
            ],
            64,
        )
        torch.testing.assert_close(
            global_ids.sort(-1).values,
            expected[:, local_ids].long().sort(-1).values,
            rtol=0,
            atol=0,
        )
