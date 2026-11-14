"""Calibration seed contracts with native metadata and fake execution on CPU."""

from dataclasses import asdict, replace
from datetime import timedelta
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from test_trainer_rank_active_memory import _rank as active_rank
from test_trainer_rank_calibration_trust_region import observe, request
from test_trainer_rank_checkpoint_memory import dense_rank
from test_trainer_rank_checkpoint_memory import rank as checkpoint_rank
from test_trainer_rank_moe_memory import layer as moe_layer
from test_trainer_rank_unanchored_floor import observe as observe_signature
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from art.megatron.context_parallel.types import ParallelTopology
from art.megatron.lora import SharedExpertsLinearFC1LoRA
from art.megatron.model_support.handlers.default_dense import DefaultDenseHandler
from art.trainer_rank import (
    CalibrationRungDecision,
    ForwardOutput,
    TrainerRank,
    TrainerRankMemoryError,
    _impl,
)
from art.trainer_rank import _memory as memory


def _rank():
    rank = active_rank()
    rank._dp_rank_and_size = lambda: (0, 1)
    return rank


class CP8Rank(TrainerRank):
    def _topology_key(self):
        return (1, 1, 8, 1)

    def _topology(self):
        return ParallelTopology(cp=8)

    def _dp_rank_and_size(self):
        return (0, 1)


def metadata_rank(kind, *, head_dim=256):
    """Real checkpoint/owner types, not a full GPU model or measured static fit."""
    model = cast(
        Any, (checkpoint_rank() if kind == "moe" else dense_rank()).runtime.model[0]
    )
    cfg = model.config
    if kind == "moe":
        cfg.num_moe_experts, cfg.moe_router_topk = 256, 8
        cfg.moe_ffn_hidden_size = cfg.moe_shared_expert_intermediate_size = 512
        model.decoder.layers[0] = cast(Any, moe_layer).__wrapped__().to("meta")
    else:
        for layer in model.decoder.layers:
            layer.mlp = torch.nn.Module()
            owner = SharedExpertsLinearFC1LoRA.__new__(SharedExpertsLinearFC1LoRA)
            torch.nn.Module.__init__(owner)
            owner.non_gated = False
            layer.mlp.linear_fc1 = owner
    cfg.kv_channels = head_dim
    cfg.num_attention_heads = (
        cfg.hidden_size // 128 if head_dim == 128 else 16 if kind == "moe" else 24
    )
    cfg.num_query_groups = 8 if head_dim == 128 else 2 if kind == "moe" else 4
    cfg.context_parallel_size = 8
    cfg.tensor_model_parallel_size = cfg.expert_model_parallel_size = 1
    cfg.expert_tensor_parallel_size = cfg.pipeline_model_parallel_size = 1
    model.register_parameter(
        "cpu_parameter",
        torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False),
    )
    return CP8Rank(
        cast(
            Any,
            SimpleNamespace(
                model=[model],
                optimizer=None,
                provider=cfg,
                model_support_handler=DefaultDenseHandler(),
            ),
        )
    )


def fake_device(monkeypatch, rank, peak):
    executed, state = [], {"peak": peak}
    monkeypatch.setattr(
        _impl.torch.cuda, "max_memory_allocated", lambda *a: state["peak"]
    )

    def run(plan, **kwargs):
        executed.append(plan)
        rank._peak_resets = rank.__dict__.get("_peak_resets", 0) + 1
        state["peak"] = peak // 2
        rank._update_peak_memory_profile(plan, 0)
        state["peak"] = peak
        return [ForwardOutput(torch.zeros(plan.packed_tokens), None, None, None)], 0

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    return executed, state


def execution_oracle(monkeypatch, rank):
    """Assert the cohort admitted this exact plan before model entry."""
    state: dict[str, Any] = {
        "complete": False,
        "fits": False,
        "plan": None,
        "executions": 0,
    }
    original_check = rank._memory_check

    def check(plan, **kwargs):
        calibration = kwargs.get("_calibration") is not None
        if calibration:
            state.update(complete=False, fits=False, plan=None)
        result = original_check(plan, **kwargs)
        if calibration:
            state.update(complete=True, fits=result.fits, plan=plan)
        return result

    def execute(plan):
        assert state["complete"], "model entry preceded admission"
        assert state["fits"], "a refused decision executed"
        assert state["plan"] is plan, "admission belonged to another plan"
        state["executions"] += 1
        return []

    fake_device(monkeypatch, rank, 1_000_000)
    original_run = rank._run_flat_plan_with_memory_tracking

    def run(plan, **kwargs):
        rank._execute_flat_plan(plan)
        return original_run(plan, **kwargs)

    monkeypatch.setattr(rank, "_memory_check", check)
    monkeypatch.setattr(rank, "_execute_flat_plan", execute)
    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    return state


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_real_planner_seed_then_ordinary_8k_then_trust_32k(monkeypatch, kind):
    rank = metadata_rank(kind)
    monkeypatch.setattr(
        rank, "_available_memory_bytes", lambda sample=None: 71_000_000_000
    )
    executed, _ = fake_device(monkeypatch, rank, 2_500_000_000)
    with memory.fixed_slope_memory_profiles():
        with rank.calibration_rung([request(16384, grad=True)], no_grad=False) as rung:
            d = rung.decision
            assert d.seed and d.eligible and d.fits and d.extrapolated
            assert d.observed_rungs == d.largest_observed_packed_tokens == 0
            assert d.measured_base_bytes == d.slope_bytes_per_token == 0
            assert d.delta_tokens == 16384 and d.seed_margin == d.margin == 2.0
            assert d.static_compute_bytes is not None
            assert d.predicted_peak_bytes == 2 * d.static_compute_bytes + d.output_bytes
            assert d.estimated_required_bytes == math.ceil(
                d.predicted_peak_bytes * d.safety_factor
            )
            iterator = rung.forward_micro_batches()
            next(iterator)
            profile = rank._memory_profiles[executed[-1].signature]
            assert profile.rate_observations == ()  # Forward alone cannot anchor.
            assert list(iterator) == []
        assert len(executed) == 1
        assert "_calibration_seed_plan" not in rank.__dict__
        profile = rank._memory_profiles[executed[0].signature]
        assert not memory._unanchored(profile)
        assert profile.rate_observations and profile.rate_observations[-1][0] == 16384
        with rank.calibration_rung(
            [request(16384, grad=True)], no_grad=False
        ) as repeated:
            assert not repeated.decision.seed
        from test_trainer_rank_cp_output_families import fixture_floor

        fixture_floor(rank, True)
        plan = rank._plan_flat_forward([request(8192, grad=True)])
        if isinstance(profile, memory._CallerMemoryProfile):
            assert (
                profile.covering_cp_peak(plan.packed_tokens, plan.caller_cp_workload)
                is not None
            )
        assert rank._memory_check(plan).fits
        # Ordinary execution/complete sampling supplies the second observed rung.
        assert (
            len(
                list(
                    rank.forward_micro_batches(
                        [request(8192, grad=True)], no_grad=False
                    )
                )
            )
            == 1
        )
        with rank.calibration_rung([request(32768, grad=True)], no_grad=False) as grown:
            assert not grown.decision.seed
            assert grown.decision.eligible and grown.decision.fits
            assert grown.decision.observed_rungs == 2


def test_seed_constants_and_backward_compatible_decision_defaults(monkeypatch):
    assert memory._CALIBRATION_SEED_TOKENS == 16384
    assert memory._CALIBRATION_SEED_MARGIN == 2.0
    rank = _rank()
    with memory.fixed_slope_memory_profiles():
        with rank.calibration_rung([request(16384)], no_grad=True) as rung:
            data = asdict(rung.decision)
    for field in ("seed", "seed_margin", "static_compute_bytes"):
        data.pop(field)
    old = CalibrationRungDecision(**data)
    assert old.seed is False and old.seed_margin is old.static_compute_bytes is None


@pytest.mark.parametrize("n", [512, 2048, 8192, 16383, 16385, 32768, 65536])
def test_seed_is_only_exactly_16384(n):
    rank = _rank()
    with memory.fixed_slope_memory_profiles():
        with rank.calibration_rung([request(n)], no_grad=True) as rung:
            assert not rung.decision.seed


@pytest.mark.parametrize("source", ["own", "borrowed", "anchored", "outside_scope"])
def test_seed_excludes_existing_references_and_inactive_scope(source):
    rank = _rank()
    with memory.fixed_slope_memory_profiles():
        if source == "outside_scope":
            observe(rank, 3, 1_000_000)
        elif source == "anchored":
            observe(rank, 8192, 1_000_000)
        else:
            key = rank._plan_flat_forward([request(16384)]).signature
            if source == "borrowed":
                key = replace(key, request_mix=("hidden",))
            observe_signature(rank, key, 512 if source == "own" else 16384, 1_000_000)
        with memory.fixed_slope_memory_profiles(source != "outside_scope"):
            with rank.calibration_rung([request(16384)], no_grad=True) as rung:
                assert not rung.decision.seed


def test_seed_over_budget_never_executes(monkeypatch):
    rank = _rank()
    executed, _ = fake_device(monkeypatch, rank, 1_000_000)
    with memory.fixed_slope_memory_profiles():
        with rank.calibration_rung([request(16384)], no_grad=True) as first:
            required = first.decision.estimated_required_bytes
        monkeypatch.setattr(
            rank, "_available_memory_bytes", lambda sample=None: required - 1
        )
        with rank.calibration_rung([request(16384)], no_grad=True) as rung:
            assert rung.decision.seed and not rung.decision.fits
            with pytest.raises(TrainerRankMemoryError):
                rung.forward_micro_batches()
    assert not executed and "_calibration_seeded_signatures" not in rank.__dict__


def test_attempted_seed_cannot_repeat_without_a_complete_sample(monkeypatch):
    rank = _rank()
    fake_device(monkeypatch, rank, 1_000_000)
    with memory.fixed_slope_memory_profiles():
        with rank.calibration_rung([request(16384)], no_grad=True) as rung:
            iterator = rung.forward_micro_batches()
            next(iterator)
            iterator.close()
        assert "_calibration_seed_plan" not in rank.__dict__
        rank._memory_profiles.clear()
        with rank.calibration_rung([request(16384)], no_grad=True) as retry:
            assert not retry.decision.seed and not retry.decision.fits


def test_invalid_initial_send_keeps_the_generator_owned_until_close(monkeypatch):
    rank = _rank()
    fake_device(monkeypatch, rank, 1_000_000)
    with memory.fixed_slope_memory_profiles():
        with rank.calibration_rung([request(16384)], no_grad=True) as rung:
            iterator = cast(Any, rung.forward_micro_batches())
            with pytest.raises(TypeError):
                iterator.send(1)
            next(iterator)
            iterator.close()
            assert "_calibration_seed_plan" not in rank.__dict__


@pytest.mark.parametrize("grad", [False, True])
def test_cold_user_paths_keep_the_floor_and_cannot_construct_seed(monkeypatch, grad):
    rank = _rank()
    executed, _ = fake_device(monkeypatch, rank, 1_000_000)
    monkeypatch.setattr(
        memory,
        "_calibration_estimate",
        lambda *a: pytest.fail("user path requested a seed"),
    )
    with memory.fixed_slope_memory_profiles():
        plan = rank._plan_flat_forward([request(16384, grad=grad)])
        assert rank._memory_check(plan).estimated_required_bytes >= 1 << 62
        with pytest.raises(TrainerRankMemoryError):
            list(
                rank.forward_micro_batches(
                    [request(16384, grad=grad)], no_grad=not grad
                )
            )
    assert executed == []


@pytest.mark.parametrize("n", [512, 2048])
def test_moe_short_root_family_dominance_coverage_after_seed(monkeypatch, n):
    rank = metadata_rank("moe")
    monkeypatch.setattr(
        rank, "_available_memory_bytes", lambda sample=None: 71_000_000_000
    )
    fake_device(monkeypatch, rank, 2_500_000_000)
    with memory.fixed_slope_memory_profiles():
        with rank.calibration_rung([request(16384, grad=True)], no_grad=False) as rung:
            list(rung.forward_micro_batches())
        from test_trainer_rank_cp_output_families import fixture_floor

        fixture_floor(rank, True)
        plan = rank._plan_flat_forward([request(n, grad=True)])
        profile = rank._memory_profiles[plan.signature]
        assert isinstance(profile, memory._CallerMemoryProfile)
        assert (
            profile.covering_cp_peak(plan.packed_tokens, plan.caller_cp_workload)
            is not None
        )
        assert rank._memory_check(plan).fits


def _seed_worker(index, directory, variant):
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(directory, "gloo")),
        rank=index,
        world_size=2,
        timeout=timedelta(seconds=8),
    )
    rank = _rank()
    rank._dp_rank_and_size = lambda: (0, 1)
    rank._available_memory_bytes = lambda sample=None: (
        1 if variant == "budget" and index == 0 else 71_000_000_000
    )
    rank._plan_head_workspace_bytes = lambda plan: (index + 1) * 1_000_000_000
    ops = []
    original = dist.all_reduce

    def reduce(tensor, op=dist.ReduceOp.SUM, **kwargs):
        ops.append((str(op), list(tensor.shape), str(tensor.dtype)))
        return original(tensor, op=op, **kwargs)

    try:
        with memory.fixed_slope_memory_profiles():
            # Every rank plans the same reference shape; only one learns it.
            p = rank._plan_flat_forward([request(512)])
            if variant == "reference" and index == 1:
                rank._update_memory_profile(
                    p, 1_000_000, retained_bytes=None, caller_phase=True
                )
            with (
                patch.object(dist, "all_reduce", reduce),
                pytest.MonkeyPatch.context() as oracle_patch,
            ):
                oracle = execution_oracle(oracle_patch, rank)
                with rank.calibration_rung([request(16384)], no_grad=True) as rung:
                    decision = asdict(rung.decision)
                    if rung.decision.fits:
                        list(rung.forward_micro_batches())
                    else:
                        with pytest.raises(TrainerRankMemoryError):
                            rung.forward_micro_batches()
                        assert oracle["executions"] == 0
            Path(directory, f"rank{index}.json").write_text(
                json.dumps(
                    dict(
                        decision=decision,
                        operations=ops,
                        executions=oracle["executions"],
                    )
                )
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("variant", ["admit", "budget", "reference"])
def test_seed_gloo_replicates_static_decision_and_mixed_eligibility(tmp_path, variant):
    mp.start_processes(
        _seed_worker,
        args=(str(tmp_path), variant),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    a, b = [json.loads((tmp_path / f"rank{i}.json").read_text()) for i in range(2)]
    assert a == b
    d = a["decision"]
    assert d["fits"] is (variant == "admit")
    assert a["executions"] == int(d["fits"])
    assert d["seed"] is (variant != "reference")
    if variant != "reference":
        assert d["static_compute_bytes"] == 2_000_000_000
        assert d["seed_margin"] == 2.0
    # Callback agreement now follows admission. Preserve the memory-decision
    # operand contract without treating that check as the context's last vote.
    decision_ops = [op for op in a["operations"] if op[1] != [5]]
    decision_index = decision_ops.index(["RedOpType.MAX", [32], "torch.float64"])
    assert decision_ops[decision_index : decision_index + 2] == [
        ["RedOpType.MAX", [32], "torch.float64"],
        ["RedOpType.MIN", [2], "torch.float64"],
    ]
