"""Measured output-class floors, sibling fallback and fail-closed CP8 admission."""

from dataclasses import replace
import json

import pytest
from test_trainer_rank_calibration_seed import fake_device
from test_trainer_rank_calibration_trust_region import request
from test_trainer_rank_cp_output_families import output_modes, output_requests
from test_trainer_rank_cp_request_floor import prepared_rank, record_minimal_caller
import torch

from art.trainer_rank import _memory as memory


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_hidden_floor_witness_uses_measured_sibling_max(kind):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank(kind, include_floor=False)
        record_minimal_caller(rank, True, 310_000_000)
        record_minimal_caller(rank, False, 90_000_000)
        with torch.no_grad():
            plan = rank._plan_flat_forward(
                output_requests(1, False, "none", True, False, None, False)
            )
            assert (
                memory._cp_request_floor(rank, plan.signature, plan.caller_cp_workload)
                == 310_000_000
            )
            assert rank._memory_check(plan).estimated_required_bytes >= 310_000_000
            assert rank._plan_cost(plan).required >= 310_000_000


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_no_measured_sibling_defers_ordinary_flat_and_split(kind):
    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        rank = prepared_rank(kind, include_floor=False)
        for n in (1, 72, 2971, 8192):
            plan = rank._plan_flat_forward([request(n)])
            assert (
                memory._cp_measured_request_floor(
                    rank, plan.signature, plan.caller_cp_workload
                )
                == 0
            )
            assert (
                memory._cp_request_floor(rank, plan.signature, plan.caller_cp_workload)
                >= memory._MEMORY_DEFERRED_BYTES
            )
            assert (
                rank._memory_check(plan).estimated_required_bytes
                >= memory._MEMORY_DEFERRED_BYTES
            )
            assert not rank._memory_check(plan).fits
            assert rank._plan_cost(plan).required >= memory._MEMORY_DEFERRED_BYTES


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_standard_floor_pass_measures_every_planner_output_class(monkeypatch, kind):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank(kind, include_floor=False)
        fake_device(monkeypatch, rank, 100_000_000)
        before = {
            key: profile.rate_observations
            for key, profile in rank._memory_profiles.items()
        }
        intents = []

        def execute(rung):
            assert rung.decision.seed and rung.decision.candidate_packed_tokens == 1
            assert rung.decision.seed_margin == 2
            intents.append(rung.decision.signature_id)
            list(rung.forward_micro_batches())

        rank.calibrate_request_floors(
            request(1), execute=execute, check_pending=lambda: None
        )
        assert len(intents) == len(set(intents))
        assert len(intents) == (124 if kind == "moe" else 62)
        for grad in (False, True):
            with torch.set_grad_enabled(grad):
                for mode in output_modes(kind):
                    for n in (1, 72, 2971, 8191):
                        plan = rank._plan_flat_forward(
                            output_requests(n, grad, *mode, layers=rank._num_layers)
                        )
                        own = memory._cp_measured_request_floor(
                            rank, plan.signature, plan.caller_cp_workload
                        )
                        charged = memory._cp_request_floor(
                            rank, plan.signature, plan.caller_cp_workload
                        )
                        maximum = max(
                            peak
                            for p in rank._memory_profiles.values()
                            for _, peak in getattr(p, "cp_request_floors", ())
                        )
                        assert charged == (own if own else maximum)
                        assert charged > 0
                        assert (
                            rank._memory_check(plan).estimated_required_bytes
                            < memory._MEMORY_DEFERRED_BYTES
                        )
        # Exact numeric widths and mixed signatures are not collapsed into
        # their representatives. They must use an actual sibling maximum or
        # remain deferred; inactive children perform no work.
        maximum = max(
            peak
            for profile in rank._memory_profiles.values()
            for _, peak in getattr(profile, "cp_request_floors", ())
        )
        assert maximum == 99_999_996

        def check_family(inputs):
            plan = rank._plan_flat_forward(inputs)
            check = rank._memory_check(plan)
            if plan.packed_tokens == 0:
                assert rank._plan_cost(plan).required == 0
                return
            own = memory._cp_measured_request_floor(
                rank, plan.signature, plan.caller_cp_workload
            )
            floor = memory._cp_request_floor(
                rank, plan.signature, plan.caller_cp_workload
            )
            assert floor > 0
            if own:
                assert floor == own
            elif floor < memory._MEMORY_DEFERRED_BYTES:
                assert floor == maximum
            else:
                assert not check.fits
            assert check.estimated_required_bytes >= floor

        for grad in (False, True):
            with torch.set_grad_enabled(grad):
                for mode in output_modes(kind):
                    base = output_requests(3, grad, *mode, layers=rank._num_layers)[0]
                    expanded = replace(
                        base,
                        target_tokens=None
                        if base.target_tokens is None
                        else torch.zeros((3, 2, 3), dtype=torch.int64),
                        top_k=None if base.top_k is None else 23,
                    )
                    check_family([expanded])
                    hidden = replace(
                        expanded,
                        input_tokens=expanded.input_tokens + 1024,
                        target_tokens=None,
                        logits=False,
                        top_k=None,
                        hidden_states=True,
                    )
                    check_family([expanded, hidden])
                    check_family([expanded, replace(hidden, no_grad=grad)])
        from art.trainer_rank import ForwardInput

        check_family([ForwardInput(input_tokens=torch.arange(3), no_grad=True)])
        assert all(
            profile.rate_observations == before.get(key, ())
            for key, profile in rank._memory_profiles.items()
        )
        rank.calibrate_request_floors(
            request(1),
            execute=lambda rung: pytest.fail("completed trial repeated"),
            check_pending=lambda: None,
        )


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_large_target_sibling_refuses_hidden_under_two_hundred_mb(monkeypatch, kind):
    # Preserve the reviewer's preparation and execution identity witness.
    from art.trainer_rank import TrainerRankMemoryError, _impl

    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        rank = prepared_rank(kind)
        fake_device(monkeypatch, rank, 1_000_000_000)
        with rank.calibration_rung(
            [request(1)], no_grad=True, seed_uncovered_cp_family=True
        ) as rung:
            assert rung.decision.fits
            list(rung.forward_micro_batches())
        sibling = rank._plan_flat_forward([request(1)])
        hidden = rank._plan_flat_forward(
            output_requests(1, False, "none", True, False, None, False)
        )

        def execution_without_output_identity(plan):
            value = json.loads(plan.caller_cp_workload[0])
            value["execution_qualification"].pop("signature", None)
            return value, plan.caller_cp_workload[1]

        assert execution_without_output_identity(
            sibling
        ) == execution_without_output_identity(hidden)
        sibling_floor = memory._cp_request_floor(
            rank, sibling.signature, sibling.caller_cp_workload
        )
        assert sibling_floor == 999_999_996
        rank._available_memory_bytes = lambda sample=None: 200_000_000
        check = rank._memory_check(hidden)
        assert not check.fits or check.estimated_required_bytes >= sibling_floor
        with pytest.raises(TrainerRankMemoryError):
            rank._select_next_micro_batch(
                tuple(
                    _impl._materialize(r)
                    for r in output_requests(1, False, "none", True, False, None, False)
                ),
                0,
                checkpoint=None,
            )


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_product_ladders_then_floor_pass_need_no_caller_family_list(monkeypatch, kind):
    from test_trainer_rank_cp_output_families import native_rank

    rank = native_rank(kind)
    with memory.fixed_slope_memory_profiles():
        for grad in (True, False):
            with torch.set_grad_enabled(grad):
                for n, peak in ((16384, 2_200_000_000), (8192, 1_300_000_000)):
                    fake_device(monkeypatch, rank, peak)
                    with rank.calibration_rung(
                        [request(n, grad=grad)], no_grad=not grad
                    ) as rung:
                        list(rung.forward_micro_batches())
        fake_device(monkeypatch, rank, 100_000_000)
        pending, completed = set(), set()

        def gate():
            assert not pending

        def execute(rung):
            key = rung.decision.signature_id
            assert key not in completed
            pending.add(key)
            assert len(rung.inputs) == 1 and rung.inputs[0].input_tokens.numel() == 1
            assert rung.inputs[0].no_grad == rung.no_grad
            list(rung.forward_micro_batches())
            pending.remove(key)
            completed.add(key)

        rank.calibrate_request_floors(request(1), execute=execute, check_pending=gate)
        assert len(completed) == (124 if kind == "moe" else 62)
        assert not pending


@pytest.mark.parametrize(
    "phase", ["before_execution", "during_execution", "after_measurement"]
)
def test_pending_floor_intent_blocks_restart_even_with_a_restored_measurement(
    monkeypatch, phase
):
    import pickle

    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank("moe", include_floor=False)
        fake_device(monkeypatch, rank, 100_000_000)
        pending = set()

        def interrupt(rung):
            pending.add(rung.decision.signature_id)
            if phase == "during_execution":
                iterator = rung.forward_micro_batches()
                next(iterator)
                iterator.close()
            elif phase == "after_measurement":
                list(rung.forward_micro_batches())
            raise RuntimeError("completion not durably recorded")

        with pytest.raises(RuntimeError, match="not durably recorded"):
            rank.calibrate_request_floors(
                request(1), execute=interrupt, check_pending=lambda: None
            )
        restored = prepared_rank("moe", include_floor=False)
        restored._memory_profiles = pickle.loads(pickle.dumps(rank._memory_profiles))

        def gate():
            if pending:
                raise RuntimeError("unresolved durable minimum intent")

        with pytest.raises(RuntimeError, match="unresolved durable"):
            restored.calibrate_request_floors(
                request(1),
                execute=lambda rung: pytest.fail("restarted an unresolved trial"),
                check_pending=gate,
            )


def test_floor_pass_requires_a_complete_sample(monkeypatch):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank("dense", include_floor=False)
        with pytest.raises(RuntimeError, match="complete positive"):
            rank.calibrate_request_floors(
                request(1), execute=lambda rung: None, check_pending=lambda: None
            )


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_deleted_hidden_minimum_uses_exact_largest_measured_sibling(kind):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank(kind, include_floor=False)
        record_minimal_caller(rank, False, 200_000_000)
        record_minimal_caller(rank, True, 900_000_000)
        with torch.no_grad():
            scope = memory._CP_FAMILY_CALIBRATION.set(True)
            try:
                plan = rank._plan_flat_forward(
                    output_requests(1, False, "none", True, False, None, False)
                )
            finally:
                memory._CP_FAMILY_CALIBRATION.reset(scope)
            plan.__dict__["_calibration_cp_family_seed"] = True
            rank._update_memory_profile(
                plan,
                100_000_000 + plan.output_bytes,
                retained_bytes=None,
                caller_phase=True,
                rate_eligible=False,
            )
            assert (
                memory._cp_request_floor(rank, plan.signature, plan.caller_cp_workload)
                == 100_000_000
            )
            del rank._memory_profiles[plan.signature]
            repeated = rank._plan_flat_forward(
                output_requests(1, False, "none", True, False, None, False)
            )
            assert (
                memory._cp_measured_request_floor(
                    rank, repeated.signature, repeated.caller_cp_workload
                )
                == 0
            )
            assert (
                memory._cp_request_floor(
                    rank, repeated.signature, repeated.caller_cp_workload
                )
                == 900_000_000
            )
            assert rank._memory_check(repeated).estimated_required_bytes >= 900_000_000


@pytest.mark.parametrize("budget_multiple", [1, 4])
def test_deferred_requirements_refuse_even_with_an_oversized_budget(budget_multiple):
    from test_trainer_rank_calibration_seed import metadata_rank

    rank = metadata_rank("moe")
    rank._available_memory_bytes = lambda sample=None: (
        budget_multiple * memory._MEMORY_DEFERRED_BYTES
    )
    check = rank._memory_check_required(memory._MEMORY_DEFERRED_BYTES)
    assert not check.fits


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_floor_intents_distinguish_native_kernel_families(kind):
    from test_trainer_rank_subfloor_cp import short_kernel

    with memory.fixed_slope_memory_profiles(), torch.no_grad():
        rank = prepared_rank(kind, include_floor=False)
        with rank.calibration_rung(
            [request(1)], no_grad=True, seed_uncovered_cp_family=True
        ) as first:
            original = first.decision.signature_id, first.family_id
        with short_kernel():
            with rank.calibration_rung(
                [request(1)], no_grad=True, seed_uncovered_cp_family=True
            ) as second:
                alternate = second.decision.signature_id, second.family_id
        assert original[0] == alternate[0]
        assert len(original[1]) == len(alternate[1]) == 64
        assert original[1] != alternate[1]


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_sibling_identity_depends_on_workload_not_operation_grouping(kind):
    with memory.fixed_slope_memory_profiles():
        rank = prepared_rank(kind, include_floor=False)
        record_minimal_caller(rank, True, 900_000_000)
        with torch.no_grad():
            plan = rank._plan_flat_forward(
                output_requests(1, False, "none", True, False, None, False)
            )
        shapes = plan.signature.slot_shapes[0][1]
        mixed = replace(
            plan.signature,
            grad_enabled=True,
            grad_modes=(False, True),
            slot_group_count=2,
            slot_shapes=((False, shapes), (True, shapes)),
            planner_coefficients=(999, None),
        )
        assert (
            memory._cp_measured_request_floor(rank, mixed, plan.caller_cp_workload) == 0
        )
        assert (
            memory._cp_request_floor(rank, mixed, plan.caller_cp_workload)
            == 900_000_000
        )
