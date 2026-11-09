"""Uncovered short CP splits retain the complete cold memory charge."""

import pytest
from test_trainer_rank_calibration_seed import fake_device
from test_trainer_rank_calibration_trust_region import request

from art.trainer_rank import _memory as memory


@pytest.mark.parametrize("collection_enabled", [True, False])
def test_split_cold_subfloor_preserves_unprofiled_bound(
    monkeypatch, collection_enabled
):
    from unittest.mock import patch

    from test_trainer_rank_calibration_seed import metadata_rank

    from art.trainer_rank import _impl

    with memory.fixed_slope_memory_profiles():
        rank = metadata_rank("dense")
        rank._available_memory_bytes = lambda sample=None: 71_000_000_000
        fake_device(monkeypatch, rank, 2_500_000_000)
        with rank.calibration_rung([request(16384, grad=True)], no_grad=False) as rung:
            list(rung.forward_micro_batches())
        requests = [request(72, grad=True), request(72, grad=True)]
        with memory.fixed_slope_memory_profiles(collection_enabled):
            children = [rank._plan_flat_forward([r]) for r in requests]
            assert all(
                memory._uncovered_subfloor_cp(
                    rank._memory_profiles.get(p.signature),
                    p.signature,
                    p.packed_tokens,
                    p.caller_cp_workload,
                )
                for p in children
            )
            warm_costs = [rank._plan_cost(p) for p in children]
            # A stored v2 profile retains its policy after collection stops.
            with (
                patch.object(rank, "_memory_profiles", {}),
                memory.fixed_slope_memory_profiles(),
            ):
                cold_costs = [rank._plan_cost(p) for p in children]
                cold_single = rank._memory_check(children[0]).estimated_required_bytes
            split_required = rank._split_required_memory(warm_costs)
            # Keep the unsafe witness budget fixed while correcting pricing.
            budget = 95_402_111
            assert budget < cold_single
            rank._available_memory_bytes = lambda sample=None: budget
            assert warm_costs == cold_costs
            assert split_required >= cold_single
            assert not rank._memory_check_required(split_required).fits
            if collection_enabled:
                result = rank._find_admissible_forward(
                    requests, checkpoint=None, refusal_prefix="memory regression"
                )
                assert isinstance(result, _impl._ForwardRefusal)
