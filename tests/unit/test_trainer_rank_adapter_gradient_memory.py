"""Adapter gradients at the recompute backward's peak: CPU admission math.

Qwen3.6-35B-A3B CP2 allocator traces: a short first wave peaks at layer 0 with
830-900 MB of expert LoRA gradients live, a long one at the last layer.
"""

from collections.abc import Sequence
import random

import pytest
from test_trainer_rank_checkpoint_memory import rank, requests
import torch

from art.megatron.lora import LoRA, LoRASlotRef
from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import (
    _COLD_RECOMPUTE_TRANSIENT_BYTES as COLD,
)
from art.trainer_rank._impl import _MemoryProfile, _SubforwardCost

POLICY = LoRASlotRef("checkpoint", "policy")
OTHER = LoRASlotRef("checkpoint", "other")


def oracle(pending, boundaries):
    """Live bytes beyond the floor while backward recomputes each layer."""
    layers = len(boundaries)
    return max(
        0,
        *(
            sum(pending[index:layers]) + pending[layers] - sum(boundaries[index + 1 :])
            for index in range(layers)
        ),
    )


def with_pending(monkeypatch, r, pending):
    # Like the real method: no slots, nothing pending.
    monkeypatch.setattr(
        r,
        "_pending_adapter_gradient_bytes",
        lambda refs: tuple(pending) if tuple(refs) else (),
    )


@pytest.mark.parametrize(
    "pending, boundaries",
    [
        # Uniform gradients above the boundaries: layer 0 is the peak.
        ([23] * 4 + [0], [4] * 4),
        # Uniform gradients below the boundaries: the last layer is the peak.
        ([3] * 4 + [0], [10] * 4),
        # Attention and GDN layers differ; the peak is interior to neither end.
        ([0, 100, 0, 100, 5], [60, 60, 60, 60]),
        # Uneven boundaries, gradients outside the decoder live throughout.
        ([7, 0, 50, 1, 9], [1, 80, 2, 3]),
    ],
)
def test_extra_is_the_worst_backward_layer_over_real_sizes(
    monkeypatch, pending, boundaries
):
    r = rank()
    with_pending(monkeypatch, r, pending)
    assert r._checkpoint_adapter_gradient_bytes((POLICY,), boundaries) == oracle(
        pending, boundaries
    )


def test_extra_matches_every_layer_for_random_sizes(monkeypatch):
    r = rank()
    generator = random.Random(0)
    for _ in range(200):
        layers = generator.randint(1, 12)
        pending = [
            generator.choice((0, generator.randint(0, 50))) for _ in range(layers + 1)
        ]
        boundaries = [generator.randint(0, 40) for _ in range(layers)]
        with_pending(monkeypatch, r, pending)
        extra = r._checkpoint_adapter_gradient_bytes((POLICY,), boundaries)
        assert extra == oracle(pending, boundaries)


def test_no_pending_gradients_add_nothing(monkeypatch):
    r = rank()
    with_pending(monkeypatch, r, ())
    assert r._checkpoint_adapter_gradient_bytes((POLICY,), [4] * 40) == 0


def lora(**slots: list[torch.nn.Parameter]) -> LoRA:
    module = LoRA.__new__(LoRA)
    torch.nn.Module.__init__(module)
    module._slot_keys = {}
    by_name = {
        LoRASlotRef("checkpoint", name): params for name, params in slots.items()
    }
    module.lora_slot_params = lambda ref: by_name.get(ref, [])  # type: ignore[method-assign]
    return module


def parameter(elements: int, *, dtype=torch.bfloat16) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(elements, dtype=dtype))


def adapter_rank(
    layers: Sequence[torch.nn.Module], outside: torch.nn.Module
) -> TrainerRank:
    r = rank()
    block = r.runtime.model[0].decoder
    block.layers = torch.nn.ModuleList(layers)
    r.runtime.model[0].head = outside
    return r


def test_pending_gradients_count_local_unallocated_slot_parameters():
    shared = parameter(10)
    allocated = parameter(1000)
    allocated.grad = torch.zeros_like(allocated)
    master = parameter(1000)
    setattr(master, "main_grad", torch.zeros(1000))
    frozen = parameter(1000)
    frozen.requires_grad_(False)
    layers = [
        lora(policy=[parameter(3), shared]),
        lora(policy=[parameter(5)], other=[parameter(1000)]),
        lora(policy=[allocated, master, frozen]),
        lora(policy=[shared, parameter(4, dtype=torch.float32)]),
    ]
    r = adapter_rank(layers, lora(policy=[parameter(6)]))
    pending = r._pending_adapter_gradient_bytes([POLICY])
    # BF16 bytes per layer; the shared parameter counts once, at its highest
    # layer; allocated, main-grad and frozen parameters are not pending; the
    # parameter outside the decoder is live throughout.
    assert pending == (3 * 2, 5 * 2, 0, (10 + 0) * 2 + 4 * 4, 6 * 2)
    assert r._pending_adapter_gradient_bytes([POLICY, OTHER]) == (
        3 * 2,
        5 * 2 + 1000 * 2,
        0,
        10 * 2 + 4 * 4,
        6 * 2,
    )
    assert r._pending_adapter_gradient_bytes([LoRASlotRef("checkpoint", "x")]) == ()
    assert r._pending_adapter_gradient_bytes([]) == ()


def test_a_parameter_the_head_also_uses_is_live_throughout():
    tied = parameter(7)
    layers = [lora(policy=[tied, parameter(3)]), lora(policy=[parameter(5)])]
    r = adapter_rank(layers, lora(policy=[tied]))
    # The head's backward runs before the decoder's and allocates it first.
    assert r._pending_adapter_gradient_bytes([POLICY]) == (3 * 2, 5 * 2, 7 * 2)


def test_a_module_the_head_also_uses_is_live_throughout():
    shared = lora(policy=[parameter(7)])
    r = adapter_rank([shared, torch.nn.Module()], shared)
    assert r._pending_adapter_gradient_bytes([POLICY]) == (0, 0, 7 * 2)


def test_base_model_groups_own_no_adapter_gradients(monkeypatch):
    r = rank()
    values = r._estimate_flat_forward(requests(67, 4096))
    with_pending(monkeypatch, r, [23 * 2**20] * 40 + [0])
    n, out, signature, groups, head = values
    both = r._subforward_cost(
        packed_tokens=n,
        output_bytes=out,
        signature=signature,
        logical_tokens=n,
        group_rows=((33, True), (34, True), (4096, False)),
        slot_refs=(POLICY, LoRASlotRef("checkpoint", None), None),
        head_workspace_bytes=head,
    )
    # The base group's slot has no adapter, so a split with or without it
    # still shares one slot's gradients.
    assert both.checkpoint_adapter_gradient_slots == '[["checkpoint", "policy"]]'


def test_other_checkpoint_parameters_are_live_throughout():
    from art.trainer_rank._impl import _CheckpointSlot

    adapter = parameter(3)
    custom = parameter(11)
    r = adapter_rank([lora(policy=[adapter])], torch.nn.Module())
    r._checkpoint_slots["policy"] = _CheckpointSlot(params=(adapter, custom))
    # A custom object's parameter has no decoder position; LoRA ones keep theirs.
    assert r._pending_adapter_gradient_bytes([POLICY]) == (3 * 2, 11 * 2)


def test_a_step_with_allocated_gradients_prices_no_extra():
    params = [parameter(100) for _ in range(4)]
    r = adapter_rank([lora(policy=[p]) for p in params], torch.nn.Module())
    assert r._pending_adapter_gradient_bytes([POLICY]) == (200, 200, 200, 200, 0)
    for p in params:
        p.grad = torch.zeros_like(p)
    # Later waves of the step find them in the availability baseline.
    assert r._pending_adapter_gradient_bytes([POLICY]) == ()


def test_slotless_lora_modules_hold_no_slot_parameters():
    module = LoRA.__new__(LoRA)
    torch.nn.Module.__init__(module)
    r = adapter_rank([module], torch.nn.Module())
    assert r._pending_adapter_gradient_bytes([POLICY]) == ()


def priced(r, values, slot_refs):
    n, out, signature, groups, head = values
    return r._subforward_cost(
        packed_tokens=n,
        output_bytes=out,
        signature=signature,
        logical_tokens=n,
        group_rows=groups,
        slot_refs=slot_refs,
        head_workspace_bytes=head,
    )


def test_cost_and_estimate_charge_the_extra_while_gradients_are_pending(monkeypatch):
    r = rank()
    values = r._estimate_flat_forward(requests(67, 4096))
    n, out, signature, groups, head = values
    retained, workspace = r._checkpoint_memory_floor(groups)
    boundary = retained // 40
    pending = [23 * 2**20] * 40 + [0]
    with_pending(monkeypatch, r, pending)
    extra = oracle(pending, [boundary] * 40)
    assert extra > 0
    cost = priced(r, values, (POLICY, None))
    assert cost.checkpoint_adapter_gradient == extra
    assert cost.checkpoint_adapter_gradient_slots == '[["checkpoint", "policy"]]'
    assert cost.required == int(
        (out + retained + workspace + COLD + retained + extra) * 1.1
    )
    estimate = r._estimate_required_memory_bytes_from_values(
        packed_tokens=n,
        output_bytes=out,
        signature=signature,
        logical_tokens=n,
        group_rows=groups,
        slot_refs=(POLICY, None),
        head_workspace_bytes=head,
    )
    assert estimate == cost.required
    # Only gradient groups' slots count; no slot, no extra.
    assert priced(r, values, (None, POLICY)).checkpoint_adapter_gradient == 0
    with_pending(monkeypatch, r, ())
    plain = priced(r, values, (POLICY, None))
    assert plain.checkpoint_adapter_gradient == 0
    assert plain.required == int((out + 2 * retained + workspace + COLD) * 1.1)
    # A learned profile drops the first-execution transients, not the extra.
    with_pending(monkeypatch, r, pending)
    r._memory_profiles[signature] = _MemoryProfile(bytes_per_token=1, packed_tokens=n)
    assert priced(r, values, (POLICY, None)).required == int(
        (out + 2 * retained + workspace + extra) * 1.1
    )


def child(extra: int, slots: str, workspace: int = 10) -> _SubforwardCost:
    return _SubforwardCost(
        required=int((1 + workspace + 1 + extra) * 1.1),
        retained=0,
        checkpoint_retained=1,
        checkpoint_workspace=workspace,
        checkpoint_input_gradient=1,
        checkpoint_adapter_gradient=extra,
        checkpoint_adapter_gradient_slots=slots,
    )


def test_split_charges_shared_gradients_once_and_distinct_slots_each():
    shared = [child(500, "a"), child(300, "a")]
    assert TrainerRank._split_required_memory(shared) == int((2 + 2 + 10 + 500) * 1.1)
    distinct = [child(500, "a"), child(300, "b")]
    assert TrainerRank._split_required_memory(distinct) == int((2 + 2 + 10 + 800) * 1.1)
    # A child with no pending gradients does not change the shared charge.
    assert TrainerRank._split_required_memory([child(500, "a"), child(0, "")]) == int(
        (2 + 2 + 10 + 500) * 1.1
    )


def test_cheap_estimate_defers_while_a_gradient_slot_has_pending_gradients(
    monkeypatch,
):
    r = rank()
    monkeypatch.setattr(r, "_ensure_checkpoint_slots_for", lambda *a, **k: None)
    monkeypatch.setattr(
        r,
        "_resolve_slot_ref",
        lambda request, checkpoint: POLICY if not request.no_grad else None,
    )
    seen: list[tuple[LoRASlotRef, ...]] = []

    def pending(refs):
        seen.append(tuple(refs))
        return (1,) * 41

    monkeypatch.setattr(r, "_pending_adapter_gradient_bytes", pending)
    assert r._estimate_flat_forward(requests(67, 4096)) is None
    assert seen == [(POLICY,)]
    monkeypatch.setattr(r, "_pending_adapter_gradient_bytes", lambda refs: ())
    assert r._estimate_flat_forward(requests(67, 4096)) is not None
