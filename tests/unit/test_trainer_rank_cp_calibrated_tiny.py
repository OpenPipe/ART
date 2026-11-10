"""Replay recorded calibration and tiny increments with frozen native facts."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_output_families import native_rank
from test_trainer_rank_cp_request_floor import record_minimal_caller
import torch

from art.megatron.context_parallel.types import ContextParallelConfig
from art.megatron.gdn.gdn_prefix_tree import GdnPlannerConfig
from art.megatron.training import microbatches
from art.trainer_rank import _memory as memory
from art.trainer_rank import _planner_misses as reports
from art.trainer_rank._planner_cost import ModelGeometry

DATA = json.loads(
    (
        Path(__file__).parent
        / "fixtures/memory_fixed_slope/cp8_moe_calibrated_profiles.json"
    ).read_text()
)


def recorded_rank(monkeypatch):
    rank = native_rank("moe")
    qualification = DATA["workload"]["execution_qualification"]
    rank._geometry = ModelGeometry(**qualification["geometry"])
    rank._num_layers = qualification["layers"]
    rank._moe_layers = qualification["moe_layers"]
    rank._gdn_layers = qualification["gdn_layers"]
    rank._coefficient_version, rank._coefficient_table = DATA["profiles"][0][
        "signature"
    ]["planner_coefficients"]
    shapes = reports._memory_signature(DATA["profiles"][0]["signature"]).slot_shapes[0][
        1
    ]
    rank._slot_memory_shapes = lambda ref: shapes
    rank._memory_profiles = {
        reports._memory_signature(x["signature"]): reports._memory_profile(x["profile"])
        for x in DATA["profiles"]
    }
    attention = dict(DATA["workload"]["attention_config"])
    attention["attention_sparse_block_size"] = tuple(
        attention["attention_sparse_block_size"]
    )
    monkeypatch.setattr(
        microbatches,
        "_context_parallel_config_for_provider",
        lambda *args: ContextParallelConfig(**attention),
    )
    monkeypatch.setattr(
        microbatches,
        "_gdn_planner_config_for_provider",
        lambda *args: GdnPlannerConfig(**json.loads(DATA["workload"]["gdn_config"])),
    )
    qualify = memory._qualified_cp_workload

    def frozen_qualification(owner, plan):
        family, loads = qualify(owner, plan)
        fields = json.loads(family)
        fields["execution_qualification"] = dict(
            qualification,
            signature=hashlib.sha256(repr(plan.signature).encode()).hexdigest(),
        )
        return json.dumps(fields, sort_keys=True, separators=(",", ":")), loads

    monkeypatch.setattr(memory, "_qualified_cp_workload", frozen_qualification)
    return rank


@pytest.mark.parametrize(
    "row",
    DATA["requests"],
    ids=lambda row: f"{row['tokens']}-{'backward' if row['grad'] else 'forward'}",
)
def test_recorded_profile_and_real_minimum_cover_tiny_increments(monkeypatch, row):
    with memory.fixed_slope_memory_profiles():
        rank = recorded_rank(monkeypatch)
        original = dict(rank._memory_profiles)
        # The recorded product calibration has no minimum trials. It cannot
        # authorize ordinary fixed-slope work until measured evidence exists.
        with torch.set_grad_enabled(row["grad"]):
            before = rank._memory_check(
                rank._plan_flat_forward([request(row["tokens"], grad=row["grad"])])
            )
            assert (
                not before.fits
                and before.estimated_required_bytes >= memory._MEMORY_DEFERRED_BYTES
            )
        # Use only the actual one-token forward reading. Backward gets this
        # measured sibling floor and its independent physical optimizer guard.
        record_minimal_caller(rank, False, DATA["minimum_forward_measured_bytes"] - 4)
        assert all(
            rank._memory_profiles[key] == profile for key, profile in original.items()
        )
        with torch.set_grad_enabled(row["grad"]):
            plan = rank._plan_flat_forward([request(row["tokens"], grad=row["grad"])])
            estimate = rank._memory_check(plan).estimated_required_bytes
            print(
                dict(
                    tokens=row["tokens"],
                    grad=row["grad"],
                    predicted=estimate,
                    measured=row["measured_bytes"],
                )
            )
            assert row["measured_bytes"] <= estimate < memory._MEMORY_DEFERRED_BYTES
