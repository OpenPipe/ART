"""Observed EP routed shares: checkpoint epochs, handoff observation, pricing."""

from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
from importlib.util import find_spec
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.trainer_rank import ForwardInput, MaterializedCheckpoint, _impl
from art.trainer_rank._impl import _CheckpointSlot, _MemoryCheck, _SplitForwardPlan
from art.trainer_rank._planner_cost import ParallelShape
from tests.unit.test_trainer_rank_moe_memory import _rank

# Real-data rank 0 of a CP2/EP2 call: its busiest rows and the CP group's
# total, so a balanced EP rank receives 48,397 rows' pairs.
ROWS, TOTAL, BALANCED, TOPK = 52480, 96794, 48397, 8


def _gated(rows=ROWS):
    rank = _rank()
    rank._parallel_shape = ParallelShape(tp=1, cp=2, ep=2)
    rank._ep_group_is_cp_group = True
    rank._moe_memory_supported = True
    plan = rank._plan_flat_forward(
        [ForwardInput(input_tokens=torch.arange(64), target_tokens=torch.arange(64))]
    )
    plan = replace(
        plan,
        signature=replace(plan.signature, topology=(1, 1, 2, 1)),
        groups=(replace(plan.groups[0], slot_ref=rank._slot_ref("policy")),),
    )
    rank._topology = lambda: SimpleNamespace(tp=1, cp=2)
    rank._plan_group_rows = lambda plan: tuple(
        (rows, group.grad_enabled) for group in plan.groups
    )
    rank._cp_group_model_tokens = lambda batch, topology: TOTAL
    rank._checkpoint_slots["policy"] = _CheckpointSlot()
    rank._commit_route_epoch("policy")
    return rank, plan


class _Layer(torch.nn.Module):
    def __init__(self, manager):
        super().__init__()
        self.token_dispatcher = SimpleNamespace(_comm_manager=manager)


def _managers(rank, *layers, topk=TOPK):
    """HybridEP managers holding each layer's last received rows per expert."""
    dispatcher = pytest.importorskip("megatron.core.transformer.moe.token_dispatcher")
    model = torch.nn.Module()
    for index, counts in enumerate(layers):
        manager = object.__new__(dispatcher._HybridEPManager)
        manager.config = SimpleNamespace(moe_router_topk=topk)
        if counts is not None:
            manager.tokens_per_expert = torch.tensor(counts, dtype=torch.int64)
        model.add_module(f"layer{index}", _Layer(manager))
    rank.runtime.model = [model]
    rank._moe_layers = len(layers)


def _exchange(rank, plan, *peers, error=None, sent=None):
    """Run the handoff against simulated peers' payloads; return what was sent."""
    sent = [] if sent is None else sent

    def reduce(values, *, op, sync_across_dp):
        assert op == "MAX" and sync_across_dp
        sent.append(list(values))
        return [max(column) for column in zip(values, *peers, strict=True)]

    rank._recovery_reduce = reduce
    rank._cache_recovery_episode = lambda error=None: nullcontext((None, None))
    rank._try_cache_recovery = lambda *args, **kwargs: None
    rank._release_cached_memory_for_backward(plan, error=error)
    return sent


def test_observed_share_only_ever_raises_routed_rows():
    rank, plan = _gated()
    assert rank._plan_group_routed_rows(plan) == (BALANCED,)
    # Within the cold 1.4 allowance, margin included.
    rank._record_routed_share(0, 1.25)
    assert rank._plan_group_routed_rows(plan) == (BALANCED,)
    rank._record_routed_share(0, 1.45)
    raised = math.ceil(BALANCED * 1.55 / 1.4)
    assert rank._plan_group_routed_rows(plan) == (raised,)
    # A calmer later wave keeps the high-water.
    rank._record_routed_share(0, 1.0)
    assert rank._routed_share_max == {0: 1.45}
    assert rank._plan_group_routed_rows(plan) == (raised,)
    unnamed = replace(plan.groups[0], slot_ref=rank._slot_ref(None))
    assert rank._plan_group_routed_rows(replace(plan, groups=(unnamed,))) == (BALANCED,)
    both = replace(plan, groups=(plan.groups[0], unnamed))
    assert rank._plan_group_routed_rows(both) == (raised, BALANCED)
    # New content under the same name is a new epoch and forgets the old share.
    previous = rank._checkpoint_slots["policy"]
    rank._checkpoint_slots["policy"] = _CheckpointSlot()
    rank._commit_route_epoch("policy", previous)
    assert rank._checkpoint_slots["policy"].route_epoch == 1
    assert rank._routed_share_max == {}
    assert rank._plan_group_routed_rows(plan) == (BALANCED,)


def test_share_is_the_worst_layer_over_priced_balanced_pairs():
    rank, plan = _gated()
    _managers(rank, [200_000, 187_176], [250_000, 253_329])
    share, epoch = rank._local_routed_share(plan)
    assert epoch == 0
    assert share == 503_329 / (TOPK * BALANCED)
    # No-grad waves dispatch the same way, so they are observed too.
    no_grad = replace(plan.groups[0], grad_enabled=False)
    assert rank._local_routed_share(replace(plan, groups=(no_grad,))) == (share, 0)


@pytest.mark.parametrize(
    "change",
    [
        "groups",
        "split",
        "empty",
        "cp1",
        "unnamed",
        "unloaded",
        "ungated",
        "unpriced",
        "layers",
    ],
)
def test_share_needs_one_observed_group(change):
    rank, plan = _gated()
    _managers(rank, [1, 2], [3, 4])
    group = plan.groups[0]
    if change == "groups":
        plan = replace(plan, groups=(group, group))
    elif change == "split":
        plan = _SplitForwardPlan((plan,), ((0,),), plan.request_count)
    elif change == "empty":
        empty = SimpleNamespace(tokens=torch.empty(0, dtype=torch.long))
        plan = replace(plan, groups=(replace(group, packed=empty),))
    elif change == "cp1":
        plan = replace(plan, signature=replace(plan.signature, topology=(1, 1, 1, 1)))
    elif change == "unnamed":
        plan = replace(plan, groups=(replace(group, slot_ref=rank._slot_ref(None)),))
    elif change == "unloaded":
        rank._checkpoint_slots["policy"] = _CheckpointSlot()
    elif change == "ungated":
        rank._ep_group_is_cp_group = False
    elif change == "unpriced":
        rank._moe_memory_supported = False
    else:
        rank._moe_layers = 3
    assert rank._local_routed_share(plan) == (-1.0, -1)


def test_every_rank_must_observe_the_same_epoch():
    rank, plan = _gated()
    _managers(rank, [250_000, 253_329])
    local = 503_329 / (TOPK * BALANCED)
    # A peer on the same checkpoint epoch with a busier layer.
    sent = _exchange(rank, plan, [0.0, 1.0, 1.5, 0.0, -0.0])
    assert sent == [[0.0, 1.0, local, 0.0, -0.0]]
    assert rank._routed_share_max == {0: 1.5}
    # An empty or unsupported peer, or one on another epoch, records nothing.
    for peer in ([0.0, 0.0, -1.0, -1.0, 1.0], [0.0, 1.0, 2.0, 3.0, -3.0]):
        _exchange(rank, plan, peer)
        assert rank._routed_share_max == {0: 1.5}
    # So does a wave that failed anywhere.
    with pytest.raises(RuntimeError, match="another rank"):
        _exchange(rank, plan, [1.0, 1.0, 2.0, 0.0, -0.0])
    error = RuntimeError("local forward")
    with pytest.raises(RuntimeError, match="local forward"):
        _exchange(rank, plan, [0.0, 1.0, 2.0, 0.0, -0.0], error=error)
    assert rank._routed_share_max == {0: 1.5}


def test_a_failed_read_still_joins_the_exchange():
    rank, plan = _gated()
    _managers(rank, None)  # No dispatch yet: nothing to read.
    sent = _exchange(rank, plan, [0.0, 1.0, 1.5, 0.0, -0.0])
    assert sent == [[0.0, 1.0, -1.0, -1.0, 1.0]]
    assert rank._routed_share_max == {}
    # The local forward error skips the read entirely.
    _managers(rank, [250_000, 253_329])
    sent = []
    with pytest.raises(RuntimeError, match="local"):
        _exchange(
            rank,
            plan,
            [0.0, 1.0, 1.5, 0.0, -0.0],
            error=RuntimeError("local"),
            sent=sent,
        )
    assert sent == [[1.0, 1.0, -1.0, -1.0, 1.0]]
    assert rank._routed_share_max == {}


def test_cancellation_during_observation_still_exchanges():
    rank, plan = _gated()

    def cancel(plan):
        raise KeyboardInterrupt

    rank._local_routed_share = cancel
    sent = []
    with pytest.raises(KeyboardInterrupt):
        _exchange(rank, plan, [0.0, 1.0, 1.5, 0.0, -0.0], sent=sent)
    # Peers see a failed forward, not a missing collective.
    assert sent == [[1.0, 1.0, -1.0, -1.0, 1.0]]
    assert rank._routed_share_max == {}


def test_epochs_beyond_exact_float_transport_are_not_observed():
    rank, plan = _gated()
    _managers(rank, [250_000, 253_329])
    rank._checkpoint_slots["policy"].route_epoch = 2**40
    assert rank._local_routed_share(plan) == (-1.0, -1)


def test_raised_rows_reach_admission_beyond_local_rows(monkeypatch):
    rank, plan = _gated()
    coefficient, shared = 64 * 1024, 1024

    def priced(*args, shared_bytes=None, **kwargs):
        # A named slot reprices its MoE layers: one supported layer.
        if shared_bytes is not None:
            shared_bytes.append(shared)
        return coefficient

    monkeypatch.setattr(_impl, "_moe_output_bytes_per_token", priced)
    rank._moe_layers = 1
    rank._memory_check_required = lambda required, **kwargs: required
    before = rank._memory_check(plan)
    cost_before = rank._plan_cost(plan).required
    rank._record_routed_share(0, 1.6)
    (raised,) = rank._plan_group_routed_rows(plan)
    # More rows arrive than this rank holds: the shared part moves onto them.
    assert raised == math.ceil(BALANCED * 1.7 / 1.4) > ROWS
    ref = plan.groups[0].slot_ref
    assert (
        rank._moe_workspace_bytes(ROWS, routed_rows=raised, slot_ref=ref)
        == raised * coefficient
    )
    assert rank._moe_workspace_bytes(ROWS, routed_rows=BALANCED, slot_ref=ref) == (
        (ROWS - BALANCED) * shared + BALANCED * coefficient
    )
    after = rank._memory_check(plan)
    assert after > before and after >= raised * coefficient
    assert rank._plan_cost(plan).required > cost_before


def test_ep1_keeps_the_two_value_exchange():
    rank, plan = _gated()
    rank._parallel_shape = ParallelShape(tp=1, cp=2, ep=1)
    assert _exchange(rank, plan, [0.0, 1.0]) == [[0.0, 1.0]]


def test_recorded_share_reaches_telemetry_and_planner_reports():
    rank, plan = _gated()
    _managers(rank, [250_000, 253_329])
    rank._planner_observation = {"replay": lambda: {"arguments": {}}}
    _exchange(rank, plan, [0.0, 1.0, 1.5, 0.0, -0.0])
    assert rank._planner_observation["replay"]() == {
        "arguments": {},
        "routed_share": 1.5,
    }
    check = _MemoryCheck(1, 2, True)
    rank._snapshot_planning_telemetry(plan, check)
    assert rank.last_forward_telemetry()["routed_share"] == 1.5
    # A later forward without a handoff reports none.
    rank._snapshot_planning_telemetry(plan, check)
    assert rank.last_forward_telemetry()["routed_share"] is None
    # Nor does a new public call after a wave that never reached its snapshot.
    rank._last_routed_share = 1.5
    rank._reset_planning_telemetry()
    rank._snapshot_planning_telemetry(plan, check)
    assert rank.last_forward_telemetry()["routed_share"] is None


def test_planner_snapshot_freezes_the_share_behind_routed_rows():
    rank, plan = _gated()
    rank._record_routed_share(0, 1.45)
    observation = {}
    rank._fill_planner_snapshot(plan, _MemoryCheck(1, 2, True), observation)
    (estimate,) = observation["replay"]()["memory_replay"]["estimates"]
    assert estimate["routed_share_max"] == [1.45]
    # Replay keeps pricing the frozen, raised rows through unchanged arguments.
    assert estimate["arguments"]["group_routed_rows"] == (
        math.ceil(BALANCED * 1.55 / 1.4),
    )
    assert "routed_share_max" not in estimate["arguments"]


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
def test_route_epochs_follow_agreed_commits(tmp_path: Path):
    from art.trainer_rank import _checkpoint
    from tests.unit.test_trainer_rank_custom_tensors import _real_lora_trainer

    trainer, _ = _real_lora_trainer()
    saved = tmp_path / "saved"
    trainer.save_checkpoint(str(saved), "student")
    trainer.load_checkpoint(MaterializedCheckpoint("policy", str(saved)))
    assert trainer._checkpoint_slots["policy"].route_epoch == 0
    trainer._record_routed_share(0, 1.5)
    # A snapshot has its source's weights, so it starts from its share.
    assert trainer.snapshot_checkpoint("policy", "policy:step0")
    assert trainer._checkpoint_slots["policy:step0"].route_epoch == 1
    assert trainer._routed_share_max == {0: 1.5, 1: 1.5}
    trainer._record_routed_share(1, 1.6)

    def fail(*args):
        raise RuntimeError("commit")

    # A failed reload rolls back to the committed content and its epoch; a
    # failed first load leaves no slot. Neither advances the counter.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_checkpoint, "_commit_slot", fail)
        with pytest.raises(RuntimeError, match="commit"):
            trainer.load_checkpoint(MaterializedCheckpoint("policy", str(saved)))
        with pytest.raises(RuntimeError, match="commit"):
            trainer.load_checkpoint(MaterializedCheckpoint("fresh", str(saved)))
    assert trainer._checkpoint_slots["policy"].route_epoch == 0
    assert "fresh" not in trainer._checkpoint_slots
    assert trainer._route_epochs == 2
    assert trainer._routed_share_max == {0: 1.5, 1: 1.6}
    trainer.load_checkpoint(MaterializedCheckpoint("policy", str(saved)))
    assert trainer._checkpoint_slots["policy"].route_epoch == 2
    assert trainer._routed_share_max == {1: 1.6}
    # Prepared snapshots load from disk: a new epoch with nothing inherited.
    prepared = _checkpoint.prepare_checkpoint(str(saved))
    assert _checkpoint.snapshot_prepared_checkpoint(trainer, prepared, "frozen")
    assert trainer._checkpoint_slots["frozen"].route_epoch == 3
    trainer._discard_snapshot_checkpoint("policy:step0")
    assert trainer._routed_share_max == {}


@pytest.mark.skipif(find_spec("megatron") is None, reason="requires Megatron")
@pytest.mark.parametrize("peer", ["name", "epoch"])
def test_loads_and_discards_must_agree_on_their_target(tmp_path: Path, peer):
    from art.trainer_rank import TrainerRankSlotStateError, _checkpoint
    from tests.unit.test_trainer_rank_custom_tensors import _real_lora_trainer

    trainer, _ = _real_lora_trainer()
    saved = tmp_path / "saved"
    trainer.save_checkpoint(str(saved), "student")
    trainer.load_checkpoint(MaterializedCheckpoint("policy", str(saved)))
    assert trainer.snapshot_checkpoint("policy", "policy:step0")
    trainer._record_routed_share(0, 1.5)
    trainer._record_routed_share(1, 1.6)
    gather = _checkpoint._gather

    def disagree(value, group=None):
        # One simulated peer targets another name, or holds another epoch.
        values = gather(value, group)
        if isinstance(value, tuple) and len(value) in (3, 5):
            other = list(value)
            if peer == "name":
                other[1 if len(value) == 3 else 0] = "other"
            else:
                index = 2 if len(value) == 3 else 3
                other[index] = 7
            values = (*values, tuple(other))
        return values

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_checkpoint, "_gather", disagree)
        with pytest.raises(TrainerRankSlotStateError, match="load target differs"):
            _checkpoint.load_checkpoint(
                trainer, _checkpoint.prepare_checkpoint(str(saved)), "policy"
            )
        with pytest.raises(TrainerRankSlotStateError, match="state differs"):
            trainer._discard_snapshot_checkpoint("policy:step0")
    assert trainer._checkpoint_slots["policy"].route_epoch == 0
    assert trainer._checkpoint_slots["policy:step0"].route_epoch == 1
    assert trainer._route_epochs == 2
    assert trainer._routed_share_max == {0: 1.5, 1: 1.6}


def _exchange_worker(rank_index: int, init_method: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank_index,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        rank, plan = _gated()
        rank._cache_recovery_episode = lambda error=None: nullcontext((None, None))
        rank._try_cache_recovery = lambda *args, **kwargs: None
        waves = (
            ((1.2, 0), (1.5, 0), {0: 1.5}),
            ((1.7, 0), (-1.0, -1), {0: 1.5}),
            ((1.7, 0), (1.8, 1), {0: 1.5}),
        )
        for first, second, expected in waves:
            observed = first if rank_index == 0 else second
            rank._local_routed_share = lambda plan, observed=observed: observed
            rank._release_cached_memory_for_backward(plan)
            assert rank._routed_share_max == expected
    finally:
        dist.destroy_process_group()


def test_gloo_exchange_records_only_unanimous_epochs(tmp_path: Path):
    mp.spawn(
        _exchange_worker,
        args=(f"file://{tmp_path / 'store'}",),
        nprocs=2,
        join=True,
    )
