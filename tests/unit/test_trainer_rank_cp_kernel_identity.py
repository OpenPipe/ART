"""Selected kernel options and authentic pre-qualification caller evidence."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from test_trainer_rank_calibration_seed import metadata_rank
from test_trainer_rank_calibration_trust_region import request
import torch

from art.megatron.flex_attn import compiled
from art.trainer_rank import _memory as memory


def legacy_profile():
    root = Path(__file__).parent / "fixtures/memory_fixed_slope"
    data = (root / "caller_format5.pkl").read_bytes()
    provenance = json.loads((root / "caller_format5.json").read_text())
    assert hashlib.sha256(data).hexdigest() == provenance["sha256"]
    signature, profile = pickle.loads(data)
    assert profile.profile_format_version == 5
    return signature, profile


def test_format5_equal_request_keeps_coverage():
    signature, profile = legacy_profile()
    rank = metadata_rank("moe")
    rank._memory_profiles[signature] = profile
    with memory.fixed_slope_memory_profiles():
        from test_trainer_rank_cp_output_families import fixture_floor

        fixture_floor(rank, True)
        plan = rank._plan_flat_forward([request(16384, grad=True)])
        assert plan.signature == signature
        assert plan.caller_cp_workload[1] == profile.caller_cp_peaks[0][3]
        assert (
            profile.covering_cp_peak(plan.packed_tokens, plan.caller_cp_workload)
            == 2_499_934_464
        )
        # Canonical legacy equality is not a new dominance claim/self-check.
        rank._available_memory_bytes = lambda sample=None: 71_000_000_000
        assert rank._memory_check(plan).fits
        with rank.calibration_rung([request(16384, grad=True)], no_grad=False) as rung:
            assert rung._coverage_guard is None


def test_format5_unqualified_evidence_never_grants_mask_dominance():
    signature, profile = legacy_profile()
    rank = metadata_rank("moe")
    rank._memory_profiles[signature] = profile
    with memory.fixed_slope_memory_profiles():
        plan = rank._plan_flat_forward([request(512, grad=True)])
        assert plan.signature == signature
        assert profile.covering_cp_peak(512, plan.caller_cp_workload) is None
        assert not memory._cp_family_dominates(
            plan.caller_cp_workload[0], profile.caller_cp_peaks[0][2]
        )


def test_blackwell_kernel_families_do_not_share_dominance(monkeypatch):
    rank = metadata_rank("moe")
    with memory.fixed_slope_memory_profiles():
        observed = rank._plan_flat_forward([request(16384, grad=True)])
        requested = rank._plan_flat_forward([request(512, grad=True)])
        rank.device = torch.device("cuda:0")
        rank.runtime.provider.art_flex_compile_crash_config = SimpleNamespace(
            triton_num_stages_2_head_dims=(256,)
        )

        def qualify(plan):
            return replace(
                plan, caller_cp_workload=memory._qualified_cp_workload(rank, plan)
            )

        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
        observed = qualify(observed)
        options: dict[str, Any] = dict(
            backend="TRITON",
            head_dim=256,
            head_dim_v=256,
            triton_num_stages_2_head_dims=(256,),
            device=rank.device,
        )
        blackwell = compiled.get_dense_compiled_flex_attention(**options)
        rank._update_memory_profile(
            observed, 2_500_000_000, retained_bytes=None, caller_phase=True
        )
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
        requested = qualify(requested)
        sm120 = compiled.get_dense_compiled_flex_attention(**options)
        assert blackwell is not sm120
        a = blackwell._torchdynamo_orig_callable
        b = sm120._torchdynamo_orig_callable
        assert a.__code__.co_code == b.__code__.co_code
        assert a.__closure__[0].cell_contents != b.__closure__[0].cell_contents
        profile = rank._memory_profiles[observed.signature]
        assert (
            profile.covering_cp_peak(
                requested.packed_tokens, requested.caller_cp_workload
            )
            is None
        )
        ids = json.loads(requested.caller_cp_workload[0])["execution_qualification"][
            "kernels"
        ]
        for family in map(json.loads, ids):
            assert family["capability"] == [12, 0]
            assert family["options"]["num_stages"] == 1
            assert family["options"]["bwd_BLOCK_N2"] == 32


@pytest.mark.parametrize("kind", ["dense", "sparse"])
@pytest.mark.parametrize("option", ["num_stages", "BLOCK_M", "bwd_BLOCK_N2"])
def test_mask_kind_options_on_one_device_have_distinct_families(
    monkeypatch, kind, option
):
    rank = metadata_rank("moe")
    with memory.fixed_slope_memory_profiles():
        observed = rank._plan_flat_forward([request(16384, grad=True)])
        requested = rank._plan_flat_forward([request(512, grad=True)])
        rank.device = torch.device("cuda:0")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
        factory = (
            compiled._flex_attention_with_options
            if kind == "dense"
            else compiled._sparse_flex_attention_with_options
        )
        # A mask-dependent selector may choose different options on one GPU.
        # Both real compiled entrypoints still share the same factory bytecode.
        entries = {
            True: torch.compile(factory({"BACKEND": "TRITON", option: 32})),
            False: torch.compile(factory({"BACKEND": "TRITON", option: 64})),
        }

        def qualify(plan):
            masks = json.loads(plan.caller_cp_workload[0])["masks"][0]
            with patch.object(
                compiled,
                f"get_{kind}_compiled_flex_attention",
                lambda **kwargs: entries["full" in masks],
            ):
                return replace(
                    plan, caller_cp_workload=memory._qualified_cp_workload(rank, plan)
                )

        observed, requested = qualify(observed), qualify(requested)
        rank._update_memory_profile(
            observed, 2_500_000_000, retained_bytes=None, caller_phase=True
        )
        profile = rank._memory_profiles[observed.signature]
        assert (
            profile.covering_cp_peak(
                requested.packed_tokens, requested.caller_cp_workload
            )
            is None
        )


def test_selected_family_identity_is_canonical_and_includes_sparse_defaults():
    options: dict[str, Any] = {
        "BACKEND": "TRITON",
        "num_stages": 2,
        "bwd_BLOCK_N2": 32,
    }
    family = compiled.compiled_flex_attention_family
    entries = [
        torch.compile(
            compiled._sparse_flex_attention_with_options(
                cast(compiled.FlexKernelOptions, values)
            )
        )
        for values in (options, dict(reversed(list(options.items()))))
    ]
    a, b = [family(entry, device=torch.device("cpu")) for entry in entries]
    assert a is not None
    assert a == b
    values = json.loads(a)
    assert values["kind"] == "sparse" and values["capability"] is None
    assert values["options"] == dict(
        options,
        PRESCALE_QK=False,
        ROWS_GUARANTEED_SAFE=False,
        BLOCKS_ARE_CONTIGUOUS=False,
        WRITE_DQ=True,
        OUTPUT_LOGSUMEXP=True,
        OUTPUT_MAX=False,
    )
    assert family(lambda: None, device=torch.device("cpu")) is None


@pytest.mark.parametrize("kind", ["dense", "sparse", "forced_dense"])
def test_head64_identity_includes_effective_backward_routing(monkeypatch, kind):
    monkeypatch.setattr(
        compiled, "_FORCED_FLEX_EXECUTION_STATE", {"required": False, "fallback": False}
    )
    original = (
        compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"})
        if kind == "sparse"
        else compiled._forced_flex_attention_dense
        if kind == "forced_dense"
        else compiled._flex_attention_with_options({"BACKEND": "FLASH"})
    )
    entry = torch.compile(original)
    identity = compiled.compiled_flex_attention_family(
        entry, device=torch.device("cpu"), head_dim=64
    )
    assert identity is not None
    effective = json.loads(identity)
    assert effective["options"]["bwd_ART_TRITON_BACKWARD_VERSION"] == 1
    # Recreate a pre-routing serialized identity on the very same runtime/device.
    old = dict(effective, version=1, options=dict(effective["options"]))
    del old["options"]["bwd_ART_TRITON_BACKWARD_VERSION"]
    del old["execution_revision"]
    old_identity = json.dumps(old, sort_keys=True, separators=(",", ":"))
    assert old_identity != identity

    def workload(kernel):
        return json.dumps(
            dict(
                version=1,
                masks=[["causal"]],
                execution_qualification=dict(backend="FLASH", kernels=[kernel]),
            ),
            sort_keys=True,
        )

    assert not memory._cp_family_dominates(
        workload(identity), memory._qualified_cp_observation(workload(old_identity))
    )
    assert (
        compiled.compiled_flex_attention_family(entry, device=torch.device("cpu"))
        is None
    )  # FLASH cannot qualify without its effective head dimension.


def test_cache_exhaustion_cannot_reuse_or_record_compiled_caller_evidence(monkeypatch):
    import importlib

    from test_flex_eager_fallback import _FlashContract
    from torch.nn.attention.flex_attention import create_block_mask

    torch._dynamo.reset()
    flex = importlib.import_module("torch.nn.attention.flex_attention")
    monkeypatch.setattr(flex, "_validate_device", lambda *args: None)
    entries = {
        kind: torch.compile(
            factory({"BACKEND": "FLASH"}),
            backend=lambda graph, _: _FlashContract(graph).run,
            dynamic=False,
        )
        for kind, factory in (
            ("dense", compiled._flex_attention_with_options),
            ("sparse", compiled._sparse_flex_attention_with_options),
        )
    }
    for kind, entry in entries.items():
        monkeypatch.setattr(
            compiled,
            f"get_{kind}_compiled_flex_attention",
            lambda _entry=entry, **kw: _entry,
        )
    rank = metadata_rank("moe", head_dim=64)
    rank.device = torch.device("cpu")
    q = torch.ones(1, 1, 2, 64, requires_grad=True)
    k = torch.ones(1, 1, 1, 64, requires_grad=True)
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=128, KV_LEN=128, device="cpu"
    )
    options = dict(block_mask=mask, scale=0.125, enable_gqa=False)
    try:
        with (
            memory.fixed_slope_memory_profiles(),
            torch._dynamo.config.patch(recompile_limit=1),
        ):
            entries["sparse"](q, k, k, **options)
            observed = rank._plan_flat_forward([request(16384, grad=True)])
            assert memory._known_cp_kernels(observed.caller_cp_workload)
            rank._update_memory_profile(
                observed, 2_500_000_000, retained_bytes=None, caller_phase=True
            )
            profile = rank._memory_profiles[observed.signature]
            assert (
                profile.covering_cp_peak(8192, observed.caller_cp_workload) is not None
            )
            assert observed.signature in rank._calibration_memory_rungs
            q = torch.ones(1, 1, 3, 64, requires_grad=True)
            hop = importlib.import_module("torch._higher_order_ops.flex_attention")

            def forbidden_math(*args, **kwargs):
                pytest.fail("compiled caller evidence admitted eager math")

            monkeypatch.setattr(hop, "_math_attention_inner", forbidden_math)
            with memory._compiled_cp_execution(rank, observed):
                with pytest.raises(RuntimeError, match="refusing eager FLASH fallback"):
                    entries["sparse"](q, k, k, **options)
            # Even if an outer caller catches the refusal, the stale plan may
            # neither publish a compiled observation nor extend live rungs.
            rank._update_memory_profile(
                observed, 9_000_000_000, retained_bytes=None, caller_phase=True
            )
            assert rank._memory_profiles[observed.signature] is profile
            assert observed.signature not in rank._calibration_memory_rungs
            requested = memory._qualified_cp_workload(rank, observed)
            assert not memory._known_cp_kernels(requested)
            assert profile.covering_cp_peak(8192, requested) is None
    finally:
        torch._dynamo.reset()


def test_rejected_workload_identity_preserves_ordinary_eager_calls(monkeypatch):
    import importlib

    from torch.nn.attention.flex_attention import create_block_mask

    flex = importlib.import_module("torch.nn.attention.flex_attention")
    monkeypatch.setattr(flex, "_validate_device", lambda *args: None)
    originals = {
        "dense": compiled._flex_attention_with_options({"BACKEND": "FLASH"}),
        "sparse": compiled._sparse_flex_attention_with_options({"BACKEND": "FLASH"}),
    }
    entries = {
        kind: torch.compile(original, backend="eager")
        for kind, original in originals.items()
    }
    for kind, entry in entries.items():
        monkeypatch.setattr(
            compiled,
            f"get_{kind}_compiled_flex_attention",
            lambda _entry=entry, **kw: _entry,
        )
    rank = metadata_rank("moe", head_dim=64)
    rank.device = torch.device("cpu")
    q = torch.ones(1, 1, 2, 64)
    mask = create_block_mask(
        lambda b, h, q, k: q >= 0, B=1, H=1, Q_LEN=2, KV_LEN=2, device="cpu"
    )
    options = dict(block_mask=mask, scale=0.125, enable_gqa=False)
    originals["sparse"](q, q, q, **options)
    with memory.fixed_slope_memory_profiles():
        plan = rank._plan_flat_forward([request(16384, grad=True)])
        assert not memory._known_cp_kernels(plan.caller_cp_workload)
        # Dense qualified successfully before sparse rejected. Neither query
        # may leave an active execution requirement behind.
        assert all(
            not compiled._flex_execution_state(entry)["required"]
            for entry in entries.values()
        )
        with memory._compiled_cp_execution(rank, plan):
            originals["sparse"](q, q, q, **options)
    assert not rank._memory_profiles
    originals["sparse"](q, q, q, **options)


@pytest.mark.parametrize("split", [False, True])
def test_compiled_requirement_is_scoped_to_qualified_plan(monkeypatch, split):
    from types import SimpleNamespace

    entries = {
        kind: torch.compile(factory({"BACKEND": "FLASH"}))
        for kind, factory in (
            ("dense", compiled._flex_attention_with_options),
            ("sparse", compiled._sparse_flex_attention_with_options),
        )
    }
    for kind, entry in entries.items():
        monkeypatch.setattr(
            compiled,
            f"get_{kind}_compiled_flex_attention",
            lambda _entry=entry, **kw: _entry,
        )
    rank = metadata_rank("moe", head_dim=64)
    rank.device = torch.device("cpu")
    with memory.fixed_slope_memory_profiles():
        plan = rank._plan_flat_forward([request(16384, grad=True)])
    assert memory._known_cp_kernels(plan.caller_cp_workload)
    states = [compiled._flex_execution_state(entry) for entry in entries.values()]
    assert not any(state["required"] for state in states)
    candidate = SimpleNamespace(subforwards=(plan, plan)) if split else plan
    # A retained v2 workload needs the guard even outside collection scope.
    with pytest.raises(ValueError, match="caller failed"):
        with memory._compiled_cp_execution(rank, candidate):
            assert all(state["required"] for state in states)
            with memory._compiled_cp_execution(rank, plan):
                assert all(state["required"] for state in states)
            assert all(state["required"] for state in states)
            raise ValueError("caller failed")
    assert not any(state["required"] for state in states)
    with memory._compiled_cp_execution(rank, SimpleNamespace(caller_cp_workload=None)):
        assert not any(state["required"] for state in states)
