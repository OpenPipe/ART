"""The head's backward and the decoder's recompute peak apart: CPU admission math.

Qwen3.6-35B-A3B CP2 allocator traces (EP1 and EP2): the head's buffers are
freed before the recompute backward allocates its workspace and adapter
gradients, and TE's first-GEMM workspaces are live at the head's peak.
"""

import itertools
import random

import pytest
from test_trainer_rank_adapter_gradient_memory import (
    OTHER,
    POLICY,
    oracle,
    priced,
    with_pending,
    with_slot_pending,
)
from test_trainer_rank_checkpoint_memory import rank, requests

from art.megatron.lora import LoRASlotRef
from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import (
    _COLD_RECOMPUTE_TRANSIENT_BYTES as COLD,
)
from art.trainer_rank._impl import _MemoryProfile

MiB = 2**20


def covered_rank(monkeypatch):
    r = rank()
    # The MoE stage covers the policy slot's recompute, as the constructor's.
    monkeypatch.setattr(r, "_moe_recompute_covered_for", lambda ref: True)
    return r


def head_oracle(chains):
    """Worst adapter bytes live beyond the floor while one group's head runs.

    Any set of the other groups may have run their backward first: each holds
    all its gradients and none of its boundaries. The running group holds only
    its gradients outside the decoder.
    """
    worst = 0
    for index, (pending, _) in enumerate(chains):
        others = chains[:index] + chains[index + 1 :]
        for count in range(len(others) + 1):
            for done in itertools.combinations(others, count):
                live = pending[-1] + sum(sum(p) - sum(b) for p, b in done)
                worst = max(worst, live)
    return worst


def test_head_adapter_gradients_match_every_order_for_random_sizes(monkeypatch):
    r = rank()
    generator = random.Random(2)
    slots = [LoRASlotRef("checkpoint", f"slot{index}") for index in range(4)]
    for _ in range(200):
        layers = generator.randint(1, 5)
        chains, groups, by_slot = [], [], {}
        for slot in slots[: generator.randint(1, 4)]:
            pending = [generator.randint(0, 30) for _ in range(layers + 1)]
            boundaries = [generator.randint(0, 30) for _ in range(layers)]
            by_slot[(slot,)] = pending
            chains.append((pending, boundaries))
            groups.append((slot, boundaries))
        with_slot_pending(monkeypatch, r, by_slot)
        assert r._checkpoint_adapter_gradient_bytes(groups, head=True) == head_oracle(
            chains
        )


def test_one_group_head_meets_only_its_gradients_outside_the_decoder(monkeypatch):
    r = rank()
    with_pending(monkeypatch, r, [30] * 4 + [7])
    assert r._checkpoint_adapter_gradient_bytes(((POLICY, [2] * 4),), head=True) == 7
    # The decoder stage still meets its own gradients.
    assert r._checkpoint_adapter_gradient_bytes(((POLICY, [2] * 4),)) == oracle(
        [30] * 4 + [7], [2] * 4
    )
    with_slot_pending(
        monkeypatch, r, {(POLICY,): [30] * 4 + [7], (OTHER,): [1] * 4 + [0]}
    )
    # The policy group may run first and raise the other's head; a group whose
    # boundaries outweigh its gradients never raises the policy's.
    groups = ((POLICY, [2] * 4), (OTHER, [40] * 4))
    assert r._checkpoint_adapter_gradient_bytes(groups, head=True) == 127 - 8
    groups = ((POLICY, [2] * 4), (OTHER, [2] * 4))
    with_slot_pending(
        monkeypatch, r, {(POLICY,): [30] * 4 + [7], (OTHER,): [0] * 4 + [0]}
    )
    assert r._checkpoint_adapter_gradient_bytes(groups, head=True) == 127 - 8


@pytest.mark.parametrize("head", [0, 64 * MiB, 700 * MiB, 3000 * MiB])
def test_cost_and_estimate_price_the_larger_stage(monkeypatch, head):
    r = covered_rank(monkeypatch)
    n, out, signature, groups, _ = r._estimate_flat_forward(requests(67, 4096))
    values = (n, out, signature, groups, head)
    retained, workspace = r._checkpoint_memory_floor(groups)
    gradient = 67 * 2048 * 2
    boundary = retained // 40
    pending = [23 * MiB] * 40 + [0]
    with_pending(monkeypatch, r, pending)
    extra = oracle(pending, [boundary] * 40)
    te = r._te_workspace_growth_bytes()
    state = 67 * r._backward_row_state_bytes()
    decoder = workspace + extra
    stage = head + 2 * gradient + state + te if head else 0
    cost = priced(r, values, (POLICY, None))
    assert cost.checkpoint_input_gradient == gradient
    assert cost.required == int(
        (out + retained + gradient + max(decoder, stage) + COLD) * 1.1
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
    # Warm: no first-execution transients and no TE growth in either stage.
    r._te_workspace_growth_bytes = lambda: 0
    r._memory_profiles[signature] = _MemoryProfile(bytes_per_token=1, packed_tokens=n)
    decoder = r._checkpoint_memory_floor(groups)[1] + extra
    assert decoder == workspace - te + extra
    stage = head + 2 * gradient + state if head else 0
    assert priced(r, values, (POLICY, None)).required == int(
        (out + retained + gradient + max(decoder, stage)) * 1.1
    )


def test_head_bound_short_wave_no_longer_adds_the_adapter_extra(monkeypatch):
    r = covered_rank(monkeypatch)
    n, out, signature, groups, _ = r._estimate_flat_forward(requests(67, 4096))
    retained, workspace = r._checkpoint_memory_floor(groups)
    head = 2 * workspace
    pending = [23 * MiB] * 40 + [0]
    with_pending(monkeypatch, r, pending)
    extra = oracle(pending, [retained // 40] * 40)
    assert head > workspace and extra > 800 * MiB
    cost = priced(r, (n, out, signature, groups, head), (POLICY, None))
    unstaged = int((out + retained + head + 67 * 2048 * 2 + extra + COLD) * 1.1)
    assert cost.required < unstaged


@pytest.mark.parametrize("uncovered", ["cp4", "dense"])
def test_untraced_recompute_keeps_the_head_in_the_decoder_stage(monkeypatch, uncovered):
    r = covered_rank(monkeypatch)
    n, out, signature, groups, _ = r._estimate_flat_forward(requests(67, 4096))
    if uncovered == "cp4":
        monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 4, 1))
    else:
        r._moe_output_bytes_per_token = r._moe_checkpoint_grad_bytes_per_token = 0
    head = 700 * MiB
    retained, workspace = r._checkpoint_memory_floor(groups)
    gradient = r._checkpoint_input_gradient_bytes(groups)
    assert gradient == retained  # One gradient per boundary.
    with_pending(monkeypatch, r, ())
    cost = priced(r, (n, out, signature, groups, head), (POLICY, None))
    assert cost.checkpoint_workspace == max(workspace, head) + COLD
    assert cost.required == int(
        (out + retained + max(workspace, head) + gradient + COLD) * 1.1
    )


def test_split_keeps_each_head_stage_beside_every_adapter_gradient(monkeypatch):
    r = covered_rank(monkeypatch)
    head = 700 * MiB
    values = r._estimate_flat_forward(requests(67, 4096))
    n, out, signature, groups, _ = values
    pending = [23 * MiB] * 40 + [0]
    with_pending(monkeypatch, r, pending)
    left = priced(r, (n, out, signature, groups, head), (POLICY, None))
    right = priced(r, (n, out, signature, groups, head), (POLICY, None))
    gradient = left.checkpoint_input_gradient
    stage = (
        head
        + 2 * gradient
        + 67 * r._backward_row_state_bytes()
        + r._te_workspace_growth_bytes()
    )
    assert (
        left.checkpoint_workspace
        == max(r._checkpoint_memory_floor(groups)[1], stage) + COLD
    )
    # A later child's decoder backward can precede an earlier child's head:
    # the split charges that head stage with the shared adapter gradients.
    split = TrainerRank._split_required_memory([left, right])
    assert split >= int(
        (
            2 * (left.checkpoint_retained + gradient)
            + stage
            + COLD
            + left.checkpoint_adapter_gradient
        )
        * 1.1
    )


def test_each_row_keeps_its_rope_embedding_and_index_state():
    import torch

    r = rank()
    model = r.runtime.model[0]
    assert r._backward_row_state_bytes() == 256
    # Qwen3.6-35B-A3B: a 64-wide rotary embedding (32 frequencies), FP32.
    model.rotary_pos_emb = torch.nn.Module()
    model.rotary_pos_emb.inv_freq = torch.ones(32)
    assert r._backward_row_state_bytes() == 64 * 4 + 256
