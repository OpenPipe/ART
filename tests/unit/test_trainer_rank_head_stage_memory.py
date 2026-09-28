"""The head's backward and the decoder's recompute peak apart: CPU admission math.

Qwen3.6-35B-A3B CP2 allocator traces (EP1 and EP2): the head's buffers are
freed before the recompute backward allocates its workspace and adapter
gradients, and TE's first-GEMM workspaces are live at the head's peak.
"""

from dataclasses import replace
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


def staged(r, values, slot_refs):
    """``priced`` for a head whose backward is the traced one."""
    n, out, signature, groups, head = values
    return r._subforward_cost(
        packed_tokens=n,
        output_bytes=out,
        signature=signature,
        logical_tokens=n,
        group_rows=groups,
        slot_refs=slot_refs,
        head_workspace_bytes=head,
        head_backward_traced=True,
    )


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
    cost = staged(r, values, (POLICY, None))
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
        head_backward_traced=True,
    )
    assert estimate == cost.required
    # An untraced head keeps the unstaged price as a floor.
    unstaged = max(workspace, head) + extra
    untraced = priced(r, values, (POLICY, None))
    assert untraced.required == int(
        (out + retained + gradient + max(unstaged, decoder, stage) + COLD) * 1.1
    )
    assert untraced.checkpoint_workspace == cost.checkpoint_workspace
    # Warm: no first-execution transients and no TE growth in either stage.
    r._te_workspace_growth_bytes = lambda: 0
    r._memory_profiles[signature] = _MemoryProfile(bytes_per_token=1, packed_tokens=n)
    decoder = r._checkpoint_memory_floor(groups)[1] + extra
    assert decoder == workspace - te + extra
    stage = head + 2 * gradient + state if head else 0
    assert staged(r, values, (POLICY, None)).required == int(
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
    values = (n, out, signature, groups, head)
    unstaged = int((out + retained + head + 67 * 2048 * 2 + extra + COLD) * 1.1)
    assert staged(r, values, (POLICY, None)).required < unstaged
    # Untraced (e.g. a top-k-only head, whose backward keeps its recomputed
    # logits beside their gradient): never below the unstaged price.
    assert priced(r, values, (POLICY, None)).required == unstaged


@pytest.mark.parametrize("uncovered", ["cp4", "uncovered_moe"])
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
    # Even for a traced head.
    cost = staged(r, (n, out, signature, groups, head), (POLICY, None))
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
    left = staged(r, (n, out, signature, groups, head), (POLICY, None))
    right = staged(r, (n, out, signature, groups, head), (OTHER, None))
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
    # The left child's head can run after the right child's decoder backward:
    # both children's boundaries and incoming gradients, the left head stage
    # and every adapter gradient the right one allocated (distinct slots).
    after_right = (
        left.checkpoint_retained
        + right.checkpoint_retained
        + 2 * gradient
        + stage
        + COLD
        + right.checkpoint_adapter_gradient
    )
    split = TrainerRank._split_required_memory([left, right])
    assert split >= int(after_right * 1.1)
    assert split >= int(
        (after_right + left.checkpoint_adapter_gradient) * 1.1
    )  # Either may run first, and distinct slots each allocate their own.


def test_traced_head_backward_is_target_only_on_the_traced_path(monkeypatch):
    from test_trainer_rank_head_memory import rank as head_rank
    from test_trainer_rank_head_memory import request

    from art.trainer_rank import _impl

    r = head_rank()
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    target = request(512, grad=True)
    # Until the fused statistics have run in this process, nothing is traced.
    state = {"succeeded": set(), "failed": False}
    monkeypatch.setattr(_impl, "_TRITON_STATS_STATE", state)
    assert r._head_backward_traced([target], 512) is False
    # The top-k kernel's success does not prove the target-only one.
    state["succeeded"].add("local_topk_stats")
    assert r._head_backward_traced([target], 512) is False
    state["succeeded"].add("local_logsumexp_stats")
    assert r._head_backward_traced([target], 512) is True
    # Top-k, logits and hidden-state outputs keep further dense gradients.
    for extra in ({"top_k": 2}, {"logits": True}, {"hidden_states": True}):
        other = replace(target, **extra)
        assert r._head_backward_traced([target, other], 512) is False
    topk_only = replace(target, target_tokens=None, top_k=2)
    assert r._head_backward_traced([topk_only], 512) is False
    # The fused statistics need 64 rows; bounds that straddle them are open.
    assert r._head_backward_traced([target], 63) is False
    assert r._head_backward_traced([target], 63, 64) is None
    assert r._head_backward_traced([target], 32, 63) is False
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_TOPK", "0")
    assert r._head_backward_traced([target], 512) is False
    monkeypatch.delenv("ART_TRAINER_RANK_TRITON_TOPK")
    # Only CP2 was traced.
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 1, 1))
    assert r._head_backward_traced([target], 512) is False
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    # One error sent a kernel to the FP32 fallback silently: never again.
    state["failed"] = True
    assert r._head_backward_traced([target], 512) is False
    state["failed"] = False
    r.runtime.model[0].config.use_mup = True
    assert r._head_backward_traced([target], 512) is False


def test_undecided_heads_bound_both_ways():
    from art.trainer_rank._impl import _traced_states

    assert _traced_states([]) == (False,)
    assert _traced_states([True, True]) == (True,)
    assert _traced_states([True, False, None]) == (False,)
    # Lower bounds take the cheaper, acceptance the dearer of both.
    assert _traced_states([True, None]) == (True, False)


def test_each_row_keeps_its_rope_embedding_and_index_state():
    import torch

    r = rank()
    model = r.runtime.model[0]
    assert r._backward_row_state_bytes() == 256
    # Qwen3.6-35B-A3B: a 64-wide rotary embedding (32 frequencies), FP32.
    model.rotary_pos_emb = torch.nn.Module()
    model.rotary_pos_emb.inv_freq = torch.ones(32)
    assert r._backward_row_state_bytes() == 64 * 4 + 256


def test_plan_stages_only_traced_gradient_heads(monkeypatch):
    from test_trainer_rank_head_memory import rank as head_rank
    from test_trainer_rank_head_memory import request

    from art.trainer_rank import _impl

    monkeypatch.setattr(
        _impl,
        "_TRITON_STATS_STATE",
        {"succeeded": {"local_logsumexp_stats"}, "failed": False},
    )
    r = head_rank()
    target = request(512, grad=True)
    topk_only = replace(target, target_tokens=None, top_k=2)
    plans = [r._plan_flat_forward([request]) for request in (target, topk_only)]
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    # A top-k-only head's backward keeps its recomputed logits beside their
    # gradient, twice the one buffer its workspace prices: never staged.
    assert [r._plan_head_backward_traced(plan) for plan in plans] == [True, False]
    no_grad = r._plan_flat_forward([request(512)])
    assert r._plan_head_backward_traced(no_grad) is False


def test_head_stage_covers_a_small_chunks_fp32_fallback(monkeypatch):
    from test_trainer_rank_head_memory import rank as head_rank

    r = head_rank()
    monkeypatch.setattr(r, "_moe_recompute_covered_for", lambda ref: True)
    monkeypatch.setattr(r, "_checkpoint_gradient_covered", lambda *a, **k: True)
    r._te_workspace_growth_bytes = lambda: 0
    groups = ((8, True),)
    dense = 248320 * 2
    state = 8 * r._backward_row_state_bytes()
    # However the CP split leaves a rank's projected rows, a chunk below the
    # fused minimum takes the FP32 fallback: nine buffers of up to 63 rows.
    stage = r._checkpoint_head_stage_bytes(3 * 8 * dense, 0, groups, None)
    assert stage == 9 * 63 * dense + state
    stage = r._checkpoint_head_stage_bytes(3 * 512 * dense, 0, groups, None)
    assert stage == 3 * 512 * dense + state
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", "512")
    stage = r._checkpoint_head_stage_bytes(3 * 512 * dense, 0, groups, None)
    assert stage == 9 * 511 * dense + state


def test_a_staged_plan_runs_its_fused_statistics_strictly(monkeypatch):
    from types import SimpleNamespace

    from art.trainer_rank import _impl, topk

    state = {"succeeded": {"local_logsumexp_stats"}, "failed": False}
    monkeypatch.setattr(_impl, "_TRITON_STATS_STATE", state)

    def fail(*args, **kwargs):
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(topk, "local_logsumexp_stats", fail)
    chunk = SimpleNamespace(is_cuda=True, shape=(512, 248320))
    # Unstaged: the FP32 fallback, and no later plan stages again.
    assert _impl._try_triton_stats("local_logsumexp_stats", chunk) is None
    assert state["failed"] is True
    # Staged: the plan's price assumed the fused path, so it raises instead.
    with pytest.raises(RuntimeError, match="admitted on their memory"):
        _impl._try_triton_stats("local_logsumexp_stats", chunk, strict=True)
    # Too few rows is a predictable fallback the head stage prices.
    small = SimpleNamespace(is_cuda=True, shape=(63, 248320))
    assert _impl._try_triton_stats("local_logsumexp_stats", small, strict=True) is None


def test_execution_binds_the_staging_its_latest_price_used(monkeypatch):
    from test_trainer_rank_head_memory import rank as head_rank
    from test_trainer_rank_head_memory import request

    from art.trainer_rank import _impl

    monkeypatch.setattr(
        _impl,
        "_TRITON_STATS_STATE",
        {"succeeded": {"local_logsumexp_stats"}, "failed": False},
    )
    r = head_rank()
    plan = r._plan_flat_forward([request(512, grad=True)])
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    seen = []
    monkeypatch.setattr(
        r,
        "_execute_flat_plan_groups",
        lambda plan: seen.append(r._head_statistics_strict) or [],
    )
    assert r._plan_head_backward_traced(plan) is True
    r._execute_flat_plan(plan)
    # A later price that no longer stages (a kernel failed since) unbinds it.
    _impl._TRITON_STATS_STATE["failed"] = True
    assert r._plan_head_backward_traced(plan) is False
    r._execute_flat_plan(plan)
    assert seen == [True, False]
    assert r._head_statistics_strict is False


def test_several_labels_per_row_are_not_traced(monkeypatch):
    from test_trainer_rank_head_memory import rank as head_rank
    from test_trainer_rank_head_memory import request

    from art.trainer_rank import _impl

    monkeypatch.setattr(
        _impl,
        "_TRITON_STATS_STATE",
        {"succeeded": {"local_logsumexp_stats"}, "failed": False},
    )
    r = head_rank()
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    single = request(512, grad=True)
    several = replace(
        single, target_tokens=single.target_tokens.unsqueeze(1).repeat(1, 4)
    )
    assert r._head_backward_traced([single], 512) is True
    assert r._head_backward_traced([several], 512) is False
