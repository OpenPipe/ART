"""CPU estimator metadata; no native buffer, allocator, or distributed replay."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
from test_grouped_planner_replay import emitted
from test_trainer_rank_head_memory import rank as head_rank
from test_trainer_rank_head_memory import request
import torch

from art.trainer_rank import _memory, _planner_replay
from art.trainer_rank import _planner_misses as reports


@pytest.mark.parametrize("held", [None, 512, 9600])
@pytest.mark.parametrize("etp", [1, 2])
def test_emitted_ep4_recomputes_growth_from_owned_dimensions(
    held, etp, monkeypatch, tmp_path
):
    from megatron.core.transformer.moe import fused_a2a

    rank = head_rank()
    rank.runtime.provider.expert_model_parallel_size = 4
    rank.runtime.provider.expert_tensor_parallel_size = etp
    rank.runtime.provider.num_moe_experts = 256
    rank._parallel_shape = replace(rank._parallel_shape, ep=4, etp=etp)
    monkeypatch.setattr(
        rank, "_topology", lambda: SimpleNamespace(tp=1, cp=1, dp=1, pp=1)
    )
    buffer = (
        None
        if held is None
        else SimpleNamespace(
            configurer=SimpleNamespace(
                buffer_config=SimpleNamespace(max_num_of_tokens_per_rank=held)
            )
        )
    )
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", buffer)
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, costs = emitted(
        rank, rank._plan_flat_forward([request(9523, grad=True)]), tmp_path
    )
    assert report["replay_complete"], report["incomplete_reasons"]
    facts = report["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]
    assert facts["hybridep"] == [9523, 4 * etp, 2048, 256 * etp, held or 0]
    result = reports.replay(report)
    assert result["aggregate"]["matches"]
    assert result["estimates"][0]["required_bytes"] == costs[0].required
    # Buffer and provider changes after capture cannot alter the snapshot.
    if buffer is not None:
        buffer.configurer.buffer_config.max_num_of_tokens_per_rank = 1
    rank.runtime.provider.hidden_size = 1
    assert reports.replay(report) == result
    changed = deepcopy(report)
    changed_facts = changed["replay"]["memory_replay"]["estimates"][0]["runtime_facts"]
    changed_facts["hybridep"][4] = 0 if held == 9600 else 9600
    assert not reports.replay(changed)["aggregate"]["matches"]
    assert not torch.cuda.is_initialized()


def test_retained_059_growth_dimensions_and_capacity_boundary():
    # Retained EP4/CP4: packed9523, max local4096, normal capacity5120.
    # Dimensions are a source-derived CPU fixture, not recovered missing facts.
    from art.megatron.train import _hybridep_token_capacity

    capacity = max(_hybridep_token_capacity(9523, 4), 4096)
    assert capacity == 5120
    dimensions = (capacity, 4, 2048, 256, 0)
    assert _memory._hybridep_growth_from_dimensions(dimensions) == 111411200
    assert (
        _memory._hybridep_growth_from_dimensions((*dimensions[:4], 5119)) == 111411200
    )
    assert _memory._hybridep_growth_from_dimensions((*dimensions[:4], 5120)) == 0


@pytest.mark.parametrize(
    "value", [[1] * 6, [1, 1, 1, 1, -1], [1, True, 1, 1, 0], [2**63, 1, 1, 1, 0]]
)
def test_invalid_dimensions_refused(value, tmp_path):
    rank = head_rank()
    facts = _planner_replay.capture(rank, rank._plan_flat_forward([request(1)]))
    facts["hybridep"] = value
    with pytest.raises(ValueError):
        _planner_replay.validate(facts)


def test_version3_zero_growth_replay_preserved(tmp_path):
    rank = head_rank()
    rank._planner_reporter = reports.Reporter(0, spool_dir=tmp_path)
    report, _ = emitted(rank, rank._plan_flat_forward([request(1)]), tmp_path)
    item = report["replay"]["memory_replay"]["estimates"][0]
    item["runtime_facts"]["version"] = 3
    del item["runtime_facts"]["hybridep"]
    del item["runtime_facts"]["hybridep_recompute_rows"]
    assert reports.replay(report)["aggregate"]["matches"]
    item["arguments"]["hybridep_growth_bytes"] = 1
    with pytest.raises(ValueError, match="hybridep_runtime_facts_unsupported"):
        reports.replay(report)


@pytest.mark.parametrize("live", [False, True])
def test_checkpoint_extent_is_frozen_without_retaining_live_graph(live, monkeypatch):
    rank = head_rank()
    facts = _planner_replay.capture(
        rank, rank._plan_flat_forward([request(2, grad=True)])
    )
    rank._parallel_shape = replace(rank._parallel_shape, ep=2, cp=2)
    monkeypatch.setattr(
        rank, "_topology", lambda: SimpleNamespace(tp=1, cp=2, dp=1, pp=1)
    )
    rank._moe_memory_supported = True
    marker = torch.empty(0)
    ref = weakref.ref(marker)
    rank._pending_hybridep_graphs.append(ref)
    rank._hybridep_rows_high_water = 218751
    if not live:
        del marker
    rows = ((2, True),)
    facts["hybridep_recompute_rows"] = _memory._checkpoint_hybridep_rows(rank, rows)
    assert facts["hybridep_recompute_rows"] == (218751 if live else 2)
    expected = rank._checkpoint_memory_floor(rows)
    replay = _planner_replay.ReplayRank.__new__(_planner_replay.ReplayRank)
    # Same immutable constructor scalars; no live runtime is supplied to replay.
    replay._facts = facts
    replay._hidden_size = rank._hidden_size
    replay._gdn_layers = 0
    replay._moe_output_bytes_per_token = rank._moe_output_bytes_per_token
    replay._topology_key = lambda: (1, 1, 2, 1)
    replay._moe_checkpoint_grad_bytes_per_token = facts[
        "checkpoint_moe_bytes_per_token"
    ]
    assert replay._checkpoint_memory_floor(rows) == expected
    rank._hybridep_rows_high_water = 1
    if live:
        del marker
    assert ref() is None
    assert replay._checkpoint_memory_floor(rows) == expected
    assert not torch.cuda.is_initialized()
