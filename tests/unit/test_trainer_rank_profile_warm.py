"""A signature's first plan seeds its memory profile; later plans set a warm rate.

CPU admission math, not a bound. On real Qwen3.6-35B-A3B (40 layers, CP2/EP2)
the run's first, small, cold wave peaked at 275 KB per packed token while every
later wave ran 206-239 KB; the max-merged rate priced every later wave with the
first wave's one-time costs.
"""

from dataclasses import replace

from test_trainer_rank_checkpoint_memory import rank, requests

from art.trainer_rank._impl import _MemoryProfile

FIRST, WARM = 3_000_000, 2_000_000  # Per packed token; far above the static floor.


def _plans(r):
    first, larger = (
        r._plan_flat_forward(requests(1024, 64)),
        r._plan_flat_forward(requests(8192, 64)),
    )
    assert first.signature == larger.signature
    assert larger.packed_tokens > first.packed_tokens
    return first, larger


def _observe(r, plan, rate):
    r._update_memory_profile(
        plan, plan.output_bytes + rate * plan.packed_tokens, retained_bytes=None
    )


def _required(r, plan, logical_tokens=None):
    return r._estimate_required_memory_bytes_from_values(
        packed_tokens=plan.packed_tokens,
        output_bytes=0,
        signature=plan.signature,
        logical_tokens=logical_tokens or plan.packed_tokens,
    )


def test_the_first_plans_one_time_costs_price_only_smaller_waves():
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    # The first plan's caller-phase update is still that plan: provisional.
    _observe(r, first, FIRST)
    profile = r._memory_profiles[first.signature]
    assert profile.bytes_per_token == FIRST
    assert profile.warm_bytes_per_token is None
    assert _required(r, larger) == int(FIRST * larger.packed_tokens * 1.1)
    _observe(r, larger, WARM)
    profile = r._memory_profiles[first.signature]
    # Smaller waves keep today's max-merged rate; larger ones the warm rate.
    assert profile.bytes_per_token == FIRST
    assert profile.warm_bytes_per_token == WARM
    assert profile.warm_packed_tokens == larger.packed_tokens
    assert profile.warm_logical_per_packed == 1
    assert _required(r, larger) == int(WARM * larger.packed_tokens * 1.1)
    assert _required(r, first) == int(FIRST * first.packed_tokens * 1.1)


def test_the_warm_rate_ratchets_and_its_extent_grows_down():
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    _observe(r, larger, WARM)
    # A later, smaller plan: its higher rate is kept, and it extends the warm
    # extent down to its own size.
    middle = r._plan_flat_forward(requests(4096, 64))
    _observe(r, middle, WARM + 100_000)
    profile = r._memory_profiles[first.signature]
    assert profile.warm_bytes_per_token == WARM + 100_000
    assert profile.warm_packed_tokens == middle.packed_tokens
    assert profile.bytes_per_token == FIRST
    # A lower warm rate never lowers it.
    _observe(r, larger, WARM - 100_000)
    assert r._memory_profiles[first.signature].warm_bytes_per_token == WARM + 100_000


def test_profiles_without_a_warm_rate_price_as_before():
    """Replayed reports and callers that set profiles directly carry no warm rate."""
    r = rank()
    first, larger = _plans(r)
    r._memory_profiles[first.signature] = _MemoryProfile(
        bytes_per_token=FIRST, packed_tokens=first.packed_tokens
    )
    assert _required(r, first) == int(FIRST * first.packed_tokens * 1.1)
    # A later plan's observation is warm: replay has no seed to exclude.
    _observe(r, larger, WARM)
    assert r._memory_profiles[first.signature].warm_bytes_per_token == WARM


def test_each_signature_and_a_cleared_profile_seeds_again():
    """Profiles are keyed by memory signature; a new signature (a different
    group count, slot shapes or topology) or a cleared profile starts cold."""
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    _observe(r, larger, WARM)
    other = r._plan_flat_forward(requests(8192, 0)[:1])
    assert other.signature != first.signature
    _observe(r, other, FIRST)
    assert r._memory_profiles[other.signature].warm_bytes_per_token is None
    # Clearing a profile makes the next plan its first again.
    del r._memory_profiles[first.signature]
    _observe(r, larger, FIRST)
    assert r._memory_profiles[first.signature].warm_bytes_per_token is None
    assert _required(r, larger) == int(FIRST * larger.packed_tokens * 1.1)


def test_a_later_plan_at_the_seeds_freed_address_is_warm():
    """CPython reuses a freed plan's address for the next plan, so the seed is
    held by weak reference rather than ``id``."""
    r = rank()
    _, larger = _plans(r)
    seed = replace(larger)
    _observe(r, seed, FIRST)
    address = id(seed)
    del seed
    later = replace(larger)
    assert id(later) == address
    _observe(r, later, WARM)
    assert r._memory_profiles[larger.signature].warm_bytes_per_token == WARM


def test_a_warm_rate_learned_under_lighter_sharing_scales_for_deeper_sharing():
    """A deeply shared first plan and unshared later plans: the later plans'
    lower rate prices unshared waves, but deeper-shared waves scale it up by
    the sharing gap, so they never price below today's fit over every plan.
    A later plan that shares as deeply extends the warm rate to them."""
    r = rank()
    _, larger = _plans(r)
    shared = replace(larger, logical_tokens=larger.packed_tokens * 64)
    deep = shared.logical_tokens

    def today():
        profile = r._memory_profiles[larger.signature]
        r._memory_profiles[larger.signature] = replace(
            profile, warm_bytes_per_token=None, warm_packed_tokens=None
        )
        prices = _required(r, larger), _required(r, larger, deep)
        r._memory_profiles[larger.signature] = profile
        return prices

    _observe(r, shared, FIRST)
    _observe(r, larger, WARM)
    profile = r._memory_profiles[larger.signature]
    assert profile.logical_per_packed == 64
    assert profile.warm_logical_per_packed == 1
    unshared, deeper = today()
    assert _required(r, larger) == int(WARM * larger.packed_tokens * 1.1) < unshared
    assert _required(r, larger, deep) == deeper
    _observe(r, replace(shared), WARM)
    assert r._memory_profiles[larger.signature].warm_logical_per_packed == 64
    assert _required(r, larger, deep) < today()[1]
