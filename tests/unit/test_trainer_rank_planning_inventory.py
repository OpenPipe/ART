"""A width search may reuse model facts, but never a later wave's gradients."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from inspect import unwrap
import sys
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import ForwardInput, TrainerRank, _impl
from art.trainer_rank._memory import _PLANNING_INVENTORY, _with_planning_inventory
from art.trainer_rank._micro_batch_planner import _search_next_micro_batch


@dataclass(frozen=True)
class Ref:
    kind: str
    name: str


class Owner(torch.nn.Module):
    def __init__(self, refs):
        super().__init__()
        self.lookups = 0
        self.A_T = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16))
        self.B_T = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.bfloat16))
        self._slot_keys: dict[Ref, str] = {}
        self.slots = torch.nn.ModuleDict()
        for i, ref in enumerate(refs):
            slot = torch.nn.Module()
            slot.A_T = torch.nn.Parameter(torch.zeros(4, i + 1, dtype=torch.bfloat16))
            slot.B_T = torch.nn.Parameter(torch.zeros(i + 1, 4, dtype=torch.bfloat16))
            self.slots[str(i)] = slot
            self._slot_keys[ref] = str(i)

    def forward(self, x):
        return x

    def _slot(self, ref):
        self.lookups += 1
        return self.slots[self._slot_keys[ref]]

    def lora_slot_params(self, ref):
        return list(self._slot(ref).parameters())

    def active_lora_tensors(self):
        return self.A_T, self.B_T, 1

    def _active_lora_storage(self):
        return self.active_lora_tensors()

    def _expert_views(self, a, b, scale):
        return a, b, scale


class Model(torch.nn.Module):
    def __init__(self, refs):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, num_layers=4, padded_vocab_size=32)
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([Owner(refs) for _ in range(4)])

    def _preprocess(self):
        pass


@pytest.fixture
def make_rank(monkeypatch):
    # Metadata-only owners keep this test independent of Bridge/TE/CUDA.
    monkeypatch.setitem(
        sys.modules,
        "art.megatron.lora",
        SimpleNamespace(LoRA=Owner, LoRASlotRef=Ref),
    )
    monkeypatch.setattr(_impl, "_gdn_layer_count", lambda _: 0)
    monkeypatch.setattr(_impl, "_mixer_top_gaps", lambda _: None)
    monkeypatch.setattr(_impl, "_lora_modules_per_layer", lambda _: 1)
    monkeypatch.setattr(_impl, "_moe_layer_count", lambda _: 0)
    monkeypatch.setattr(_impl, "_dense_fc1_adapted", lambda _: True)

    def make(*, cached=True, cp=8):
        refs = Ref("checkpoint", "a"), Ref("checkpoint", "b")
        model = cast(Any, Model(refs))
        rank = cast(
            Any,
            TrainerRank(
                cast(
                    Any,
                    SimpleNamespace(
                        model=[model],
                        optimizer=None,
                        provider=SimpleNamespace(hidden_size=4, num_layers=4),
                        model_support_handler=SimpleNamespace(
                            build_gdn_execution_spec=False
                        ),
                    ),
                )
            ),
        )
        rank._default_slot_ref = refs[0]
        rank._topology_key = lambda: (1, 1, cp, 1)
        rank._topology = lambda: SimpleNamespace(tp=1, cp=cp)
        rank._parallel_shape = _impl.ParallelShape(tp=1, cp=cp, ep=1, etp=1)
        rank._dp_rank_and_size = lambda: (0, 1)
        rank._max_rank_model_tokens = lambda batch, **_: max(
            1, -(-batch.tokens.numel() // cp)
        )
        rank._recorded_one_layer_recompute = True
        rank._checkpoint_slots = {
            ref.name: _impl._CheckpointSlot(
                params=tuple(
                    p
                    for layer in model.decoder.layers
                    for p in layer._slot(ref).parameters()
                )
            )
            for ref in refs
        }
        rank._available_memory_bytes = lambda: 1 << 40
        rank._all_ranks_true = lambda value: value
        if not cached:
            # The decorator is the only width-search change; its unchanged body
            # is the uncached admission oracle from before this optimization.
            rank._search_next_micro_batch = MethodType(
                unwrap(_search_next_micro_batch), rank
            )
        return rank, model, refs

    return make


def facts(rank, ref):
    return rank._slot_memory_shapes(ref), rank._pending_adapter_gradient_bytes((ref,))


def test_each_inventory_walk_runs_once_and_empty_gradients_are_cached(make_rank):
    rank, model, refs = make_rank()
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    expected = facts(rank, refs[0])
    assert expected[1] == ()

    @_with_planning_inventory
    def search(rank):
        first = facts(rank, refs[0])
        lookups = sum(owner.lookups for owner in model.decoder.layers)
        assert facts(rank, refs[0]) == first == expected
        assert sum(owner.lookups for owner in model.decoder.layers) == lookups
        assert facts(rank, refs[1])[0] != first[0]

    search(rank)
    assert _PLANNING_INVENTORY.get() is None


def test_next_search_observes_gradient_allocation_and_slot_replacement(make_rank):
    rank, model, refs = make_rank()

    @_with_planning_inventory
    def search(rank):
        first = facts(rank, refs[0])
        assert facts(rank, refs[0]) == first
        return first

    before = search(rank)
    for p in rank._checkpoint_slots[refs[0].name].params:
        p.main_grad = torch.zeros_like(p, dtype=torch.float32)
    slot = model.decoder.layers[0]._slot(refs[0])
    slot.A_T = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    after = search(rank)
    assert before[0] != after[0]
    assert before[1] != after[1]
    assert after == facts(rank, refs[0])


def test_errors_nested_ranks_and_threads_restore_inventory(make_rank):
    first, _, refs = make_rank()
    second, _, _ = make_rank()

    @_with_planning_inventory
    def inner(rank, fail=False):
        inventory = _PLANNING_INVENTORY.get()
        assert inventory is not None and inventory.rank is rank
        value = facts(rank, refs[0])
        if fail:
            raise ValueError("planning failure")
        return value

    @_with_planning_inventory
    def outer(rank):
        inventory = _PLANNING_INVENTORY.get()
        assert inner(second) == facts(second, refs[0])
        assert _PLANNING_INVENTORY.get() is inventory
        with pytest.raises(ValueError, match="planning failure"):
            inner(rank, True)
        assert _PLANNING_INVENTORY.get() is inventory
        return facts(rank, refs[0])

    assert outer(first) == inner(first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(inner, (first, second))) == [
            facts(first, refs[0]),
            facts(second, refs[0]),
        ]
    assert _PLANNING_INVENTORY.get() is None


def corpus(kind):
    generator = torch.Generator().manual_seed(804)
    if kind == "long":
        return [
            ForwardInput(input_tokens=torch.arange(8192) + i, hidden_states=True)
            for i in range(3)
        ]
    rows = []
    production = kind == "production"
    for i in range(568 if production else 96):
        prompt = torch.full((240 if production else 32 + i // 16 % 3,), 1000 + i // 16)
        tail = torch.randint(
            10000, 20000, (213 if production else 64 + i % 19,), generator=generator
        )
        tokens = torch.cat((prompt, tail))
        rows.append(
            ForwardInput(
                input_tokens=tokens,
                target_tokens=tokens,
                no_grad=(i % 3 == 0) if kind == "mixed" else False,
                checkpoint=("a" if i % 2 else "b") if kind == "slots" else _impl.Unset,
            )
        )
    return rows


@pytest.mark.parametrize("kind", ("rl", "production", "mixed", "slots", "long"))
@pytest.mark.parametrize("cp", (1, 8))
def test_wave_indices_layouts_checks_and_packing_hashes_match_uncached(
    make_rank, kind, cp
):
    inputs = corpus(kind)
    results = []
    for cached in (False, True):
        rank, _, _ = make_rank(cached=cached, cp=cp)
        # Seed every signature encountered while ramping, retaining real
        # packing/shape/signature and estimator arithmetic on both paths.
        for width in (1, 2, 4, 8, 16, 32, 64):
            plan = rank._plan_flat_forward(inputs[:width])
            rank._memory_profiles[plan.signature] = _impl._MemoryProfile(
                bytes_per_token=10000,
                packed_tokens=1_000_000,
                retained_compute_bytes_per_token=1,
            )
        limit = rank._memory_check(
            rank._plan_flat_forward(inputs[:32])
        ).estimated_required_bytes
        rank._available_memory_bytes = lambda: limit
        selected = []
        start = 0
        while start < len(inputs):
            candidate = rank._select_next_micro_batch(inputs, start)
            selected.append(
                (
                    candidate.indices,
                    rank._packing_fingerprints(candidate.plan),
                    candidate.check,
                    tuple(g.packed.segments for g in candidate.plan.groups),
                    candidate.cold_start,
                )
            )
            rank._last_global_micro_batch_size = max(
                rank._last_global_micro_batch_size or 0, candidate.stats_global_count
            )
            start += candidate.stats_global_count
        results.append(selected)
    assert results[0] == results[1]


def test_memory_tight_split_matches_uncached(make_rank, monkeypatch):
    inputs = corpus("rl")[:4]
    results = []
    for cached in (False, True):
        rank, _, _ = make_rank(cached=cached, cp=1)
        monkeypatch.setattr(rank, "_retained_memory_bytes", lambda *_, **__: 0)
        monkeypatch.setattr(
            rank,
            "_estimate_required_memory_bytes_from_values",
            lambda *, packed_tokens, **_: 100 * packed_tokens,
        )
        plan = rank._plan_flat_forward(inputs)
        rank._memory_profiles[plan.signature] = _impl._MemoryProfile(
            bytes_per_token=100000,
            packed_tokens=100000,
            retained_compute_bytes_per_token=1,
        )
        limit = rank._memory_check(plan).estimated_required_bytes * 3 // 4
        rank._available_memory_bytes = lambda: limit
        candidate = rank._select_next_micro_batch([inputs], 0)
        assert isinstance(candidate.plan, _impl._SplitForwardPlan)
        results.append(
            (
                candidate.indices,
                rank._packing_fingerprints(candidate.plan),
                candidate.check,
            )
        )
    assert results[0] == results[1]
