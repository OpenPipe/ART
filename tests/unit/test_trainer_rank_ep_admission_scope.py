"""Public admission with actual reduction routing and explicit CPU peer stand-ins."""

from unittest.mock import patch

import pytest
from test_trainer_rank_planner_options import _oversized
from test_trainer_rank_split import _recording_executor, _request

from art.trainer_rank import _impl as tr


def _exercise(monkeypatch, *, api, ep, allow, peer_veto, count, limit=5):
    rank = _oversized(monkeypatch, limit=limit)
    rank._allow_oversized_batches = allow
    monkeypatch.setattr(rank, "_expert_parallel_active", lambda: ep)
    monkeypatch.setattr(rank, "_try_cache_recovery", lambda *a, **k: False)
    local_group = object()
    monkeypatch.setattr(rank, "_forward_memory_group", lambda: local_group)
    if ep:
        monkeypatch.setattr(
            rank, "_admit_split_rung", lambda *a, **k: pytest.fail("EP split")
        )
    calls = []
    actual_reduce = tr.TrainerRank._recovery_reduce

    def all_reduce(value, *, op, group):
        calls.append((group, value.tolist(), op))
        # For TP=CP=1, an EP peer can be outside the TPxCP group. Without
        # EP, the relevant peers are inside TPxCP. This is a routing model,
        # not an executed distributed collective or native EP workload.
        sees_peer = group is None or not ep
        if peer_veto and sees_peer and value.numel() == 3:
            value[0] = 0

    def reduce(values, *, op, sync_across_dp):
        with (
            patch.object(tr.dist, "is_initialized", return_value=True),
            patch.object(tr.dist, "all_reduce", all_reduce),
        ):
            return actual_reduce(rank, values, op=op, sync_across_dp=sync_across_dp)

    monkeypatch.setattr(rank, "_recovery_reduce", reduce)
    executed = _recording_executor(monkeypatch, rank)
    requests = [_request(i) for i in range(count)]
    refused = False
    try:
        if api == "forward":
            outputs = rank.forward(requests)
        else:
            iterator = rank.forward_batches([requests])
            try:
                batches = list(iterator)
                assert len(batches) == 1
                assert batches[0].stats.global_count == 1
                outputs = batches[0].outputs[0]
            finally:
                iterator.close()
        assert [int(o.target_logprobs.item()) for o in outputs] == list(range(count))
    except tr.TrainerRankMemoryError:
        refused = True
    return refused, executed, calls, local_group


@pytest.mark.parametrize("api", ["forward", "forward_batches"])
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize("peer_veto", [False, True])
def test_oversized_requires_complete_peer_scope(
    monkeypatch, api, count, ep, allow, peer_veto
):
    refused, executed, calls, local_group = _exercise(
        monkeypatch, api=api, ep=ep, allow=allow, peer_veto=peer_veto, count=count
    )
    local_ep = ep and api == "forward"
    accepted = allow and not peer_veto and not local_ep
    assert refused is not accepted
    assert bool(executed) is accepted
    if local_ep:
        # In particular, do not add a late WORLD collective on this path:
        # other DP peers may not have entered memory recovery at all.
        assert all(group is local_group for group, _, _ in calls)
    else:
        assert calls
        # Earlier split-price checks can legitimately be TPxCP-local.
        # The final oversized admission vote must cover the caller scope.
        assert calls[-1][0] is (None if api == "forward_batches" else local_group)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("allow", [False, True])
def test_fitting_local_ep_keeps_existing_behavior(monkeypatch, count, allow):
    refused, executed, calls, _ = _exercise(
        monkeypatch,
        api="forward",
        ep=True,
        allow=allow,
        peer_veto=True,
        count=count,
        limit=100,
    )
    assert not refused and len(executed) == 1
    assert calls == []
