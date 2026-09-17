"""Selected-slot pricing through the real CPU loader; no model/CUDA execution."""

from dataclasses import replace
from functools import partial

import pytest
from test_trainer_rank_converted_memory import expected, weights
from test_trainer_rank_moe_memory import layer as layer
from test_trainer_rank_pending_memory import rank_with_moe
import torch

from art.megatron.lora import LoRA, LoRASlotRef, use_lora_slot
from art.trainer_rank import ForwardInput, _gdn_memory
from art.trainer_rank._impl import (
    Unset,
    _CheckpointSlot,
    _expert_lora_weight_storage,
    _ForwardRefusal,
    _moe_dispatch_preprocess,
    _SplitForwardPlan,
)


def load_slot(rank, name, selected_rank):
    """Supply omitted inert DP1 fixture metadata, then use the real slot loader."""
    ref = LoRASlotRef("checkpoint", name)
    for chunk in rank.runtime.model:
        for index, lora in enumerate(chunk.modules()):
            if type(lora) is not LoRA:
                continue
            expert = lora.A_T.ndim == 3
            lora.adapter_model_prefix = f"layer{index}" + (
                ".{expert}" if expert else ""
            )
            if not hasattr(lora, "_slot_keys"):
                lora._slot_keys = {}
                lora._slot_modules = torch.nn.ModuleDict()
            lora._expert_offset = 0
            count = lora.A_T.shape[0] if expert else 1
            lora._expert_ids = tuple(range(count))
            for param in (lora.A_T, lora.B_T):
                param.lora_shard_domain = "expert_tensor" if expert else "tp"
                param.lora_tp_sharded = False
            inputs, outputs = lora.A_T.shape[-2], lora.B_T.shape[-1]
            # expand creates a small CPU source view; the real loader stacks,
            # makes contiguous tensors and clones its actual slot Parameters.
            adapter = {}
            for i in range(count):
                prefix = lora.adapter_model_prefix.format(expert=i)
                adapter[prefix + ".lora_A.weight"] = torch.zeros(
                    (), dtype=torch.bfloat16
                ).expand(selected_rank, inputs)
                adapter[prefix + ".lora_B.weight"] = torch.zeros(
                    (), dtype=torch.bfloat16
                ).expand(outputs, selected_rank)
            assert lora.load_lora_slot(ref, adapter, requires_grad=False)
            assert lora._slot(ref).rank == selected_rank
    rank._checkpoint_slots[name] = _CheckpointSlot()
    return ref


def request(name, *, rows=1, grad=False):
    return ForwardInput(
        input_tokens=torch.arange(rows),
        hidden_states=True,
        checkpoint=name,
        no_grad=not grad,
    )


@pytest.mark.parametrize("base,selected", [(8, 64), (64, 8), (8, 1)])
@pytest.mark.parametrize("grad", [False, True])
def test_selected_rank_prices_real_loaded_parameters(layer, base, selected, grad):
    rank, _ = rank_with_moe(weights(layer, base))
    original = rank._moe_workspace_bytes(1, checkpoint_grad=grad)
    ref = load_slot(rank, "selected", selected)
    adapter = layer.experts.linear_fc2.lora
    active = adapter._slot(ref)
    expected_transposes = 256 * max(8, selected) * (512 + 2048) * 2
    storage = _expert_lora_weight_storage(adapter, ref)
    assert storage is not None
    assert storage[1] == expected_transposes
    assert active.A_T.shape[-1] == selected and adapter.A_T.shape[-1] == base
    for rows in (1, 64, 50640):
        assert rank._moe_workspace_bytes(
            rows, checkpoint_grad=grad, slot_ref=ref
        ) == expected(rows, selected, grad)
    assert rank._moe_workspace_bytes(0, checkpoint_grad=grad, slot_ref=ref) == 0
    assert rank._moe_workspace_bytes(1, checkpoint_grad=grad) == original
    assert rank._moe_workspace_bytes(1, checkpoint_grad=grad, slot_ref=ref) != original
    plan = rank._plan_flat_forward([request("selected", grad=grad)], ensure_slots=False)
    required = rank._memory_check(plan).estimated_required_bytes
    assert required == rank._plan_cost(plan).required
    assert required >= int(expected(1, selected, grad) * 1.1)
    rank._available_memory_bytes = lambda: required - 1
    assert not rank._memory_check(plan).fits
    assert not torch.cuda.is_initialized()


def test_mixed_slot_context_and_exact_search_fallback(layer):
    rank, _ = rank_with_moe(weights(layer, 8))
    small = load_slot(rank, "small", 1)
    large = load_slot(rank, "large", 64)
    requests = [request("small", rows=2), request("large", rows=1, grad=True)]
    with use_lora_slot(small):
        lora = layer.experts.linear_fc2.lora
        original = lora.active_lora_tensors()[0]
        assert original is lora._slot(small).A_T
        plan = rank._plan_flat_forward(requests, ensure_slots=False)
        assert tuple(g.slot_ref for g in plan.groups) == (small, large)
        assert rank._moe_workspace_bytes(
            1, checkpoint_grad=True, slot_ref=large
        ) == expected(1, 64, True)
        assert rank._estimate_flat_forward(requests) is None
        required = rank._memory_check(plan).estimated_required_bytes
        assert required == rank._plan_cost(plan).required
        rank._available_memory_bytes = lambda: required
        admitted = rank._search_next_micro_batch([requests], 0)
        assert not isinstance(admitted, _ForwardRefusal) and admitted.check.fits
        assert admitted.check.estimated_required_bytes == required
        rank._available_memory_bytes = lambda: 1
        assert isinstance(rank._search_next_micro_batch([requests], 0), _ForwardRefusal)
        assert lora.active_lora_tensors()[0] is original
    cost = rank._split_chunk_lower_cost(
        requests, tuple(r.input_tokens for r in requests), checkpoint=Unset
    )
    assert cost.required <= rank._plan_cost(plan).required


def test_gdn_pending_uses_selected_output_rank(layer):
    rank, gd = rank_with_moe(weights(layer, 8))
    small, large = load_slot(rank, "small", 1), load_slot(rank, "large", 64)
    small_shapes = _gdn_memory.model_shapes(rank, small)
    large_shapes = _gdn_memory.model_shapes(rank, large)
    assert small_shapes is not None and large_shapes is not None
    small_shape = small_shapes[1][0]
    large_shape = large_shapes[1][0]
    assert (small_shape.output_lora_rank, large_shape.output_lora_rank) == (1, 64)
    p = rank._plan_flat_forward(
        [request("large", rows=65, grad=True)], ensure_slots=False
    )
    group = p.groups[0]
    buckets = _gdn_memory.cp1_buckets(group.packed.segments)
    assert (
        large_shape.pending(65, buckets) - small_shape.pending(65, buckets)
        == 65 * (64 - 1) * 2
    )
    retained, workspace = _gdn_memory.plan_floor(rank, p)
    assert retained == 65 * 40 * 2048 * 2
    assert workspace == expected(65, 64, True) + large_shape.pending(65, buckets)
    assert gd.out_proj.lora.A_T.shape[1] == 1


def test_profile_and_split_key_separate_slot_layout_and_grad_mode(layer):
    rank, _ = rank_with_moe(weights(layer, 8))
    load_slot(rank, "small", 1)
    load_slot(rank, "large", 64)
    small = rank._plan_flat_forward([request("small", grad=True)], ensure_slots=False)
    large = rank._plan_flat_forward([request("large", grad=True)], ensure_slots=False)
    rank._update_memory_profile(small, 2**30, retained_bytes=1)
    assert small.signature != large.signature
    assert (
        small.signature in rank._memory_profiles
        and large.signature not in rank._memory_profiles
    )
    assert not rank._all_ranks_have_memory_profile(
        packed_tokens=large.packed_tokens, signature=large.signature
    )
    # Give the selected layout its own genuine empirical floor. It dominates
    # static demand without adding that static demand a second time.
    rank._update_memory_profile(large, 2**30, retained_bytes=1)
    assert rank._memory_check(large).estimated_required_bytes == int(2**30 * 1.1)
    left = rank._plan_flat_forward(
        [request("small", grad=True), request("large")], ensure_slots=False
    )
    right = rank._plan_flat_forward(
        [request("small"), request("large", grad=True)], ensure_slots=False
    )
    assert left.signature != right.signature
    split = _SplitForwardPlan((small,), ((0,),), 1)
    changed = replace(split, subforwards=(replace(small, signature=large.signature),))
    assert rank._split_memory_key(split) != rank._split_memory_key(changed)


def test_slot_reload_reprices_without_mutating_constructor_or_profile(layer):
    rank, _ = rank_with_moe(weights(layer, 8))
    ref = load_slot(rank, "reload", 1)
    small = rank._plan_flat_forward([request("reload")], ensure_slots=False)
    rank._update_memory_profile(small, 2**30, retained_bytes=1)
    load_slot(rank, "reload", 64)
    large = rank._plan_flat_forward([request("reload")], ensure_slots=False)
    assert large.signature != small.signature
    assert large.signature not in rank._memory_profiles
    assert rank._moe_workspace_bytes(1, slot_ref=ref) == expected(1, 64, False)
    assert rank._moe_workspace_bytes(1) == expected(1, 8, False)


@pytest.mark.parametrize(
    "kind", ["foreign", "wrong owner", "keywords", "slot override"]
)
def test_slot_pricing_retains_original_owner_guards(layer, kind):
    rank, _ = rank_with_moe(weights(layer, 8))
    ref = load_slot(rank, "loaded", 64)
    dispatcher = layer.token_dispatcher
    if kind == "slot override":
        lora = layer.experts.linear_fc2.lora
        lora._slot = lambda selected: lora._slot_modules["slot_0"]
        assert _expert_lora_weight_storage(lora, ref) is None
    else:
        dispatcher.dispatch_preprocess = (
            (lambda *args: None)
            if kind == "foreign"
            else partial(_moe_dispatch_preprocess, object())
            if kind == "wrong owner"
            else partial(
                _moe_dispatch_preprocess, dispatcher, hidden_states=torch.empty(0)
            )
        )
        assert rank._moe_workspace_bytes(1, slot_ref=ref) == 0


@pytest.mark.parametrize("kind", ["generic", "inactive", "without megatron"])
def test_generic_signature_needs_neither_megatron_nor_module_walk(monkeypatch, kind):
    import builtins
    from types import SimpleNamespace

    from art.trainer_rank import TrainerRank
    from art.trainer_rank._impl import _LocalLoRASlotRef

    rank = TrainerRank.__new__(TrainerRank)
    rank._moe_layers = 0 if kind == "generic" else 1
    rank._gdn_layers = 0
    rank.runtime = object()  # There is deliberately no model/modules facade.
    ref = (
        _LocalLoRASlotRef(name="selected")
        if kind == "without megatron"
        else SimpleNamespace(name=None if kind == "inactive" else "selected")
    )
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "art.megatron.lora":
            raise AssertionError("generic pricing must not import Megatron")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert rank._slot_memory_shapes(ref) == ()
