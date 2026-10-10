"""A signature's first plan seeds its memory profile; later plans set a warm fit.

CPU admission math, not a bound. On real Qwen3.6-35B-A3B (40 layers, CP2/EP2)
the run's first, small, cold wave peaked at 275 KB per packed token while every
later wave ran 206-247 KB; the max-merged rate priced every later wave with the
first wave's one-time costs.
"""

from dataclasses import replace

from test_trainer_rank_active_memory import _rank as packed_rank
from test_trainer_rank_active_memory import _requests as packed_requests
from test_trainer_rank_checkpoint_memory import rank, requests
import torch

from art.trainer_rank import _impl
from art.trainer_rank._impl import _MemoryProfile, _packed_priced

FIRST, WARM = 3_000_000, 2_000_000  # Per packed token; far above the static floor.


def _plans(r):
    first, larger = (
        r._plan_flat_forward(requests(1024, 64)),
        r._plan_flat_forward(requests(8192, 64)),
    )
    assert first.signature == larger.signature
    assert larger.packed_tokens > first.packed_tokens
    return first, larger


def _update(r, plan, rate, *, caller_phase):
    r._update_memory_profile(
        plan,
        plan.output_bytes + rate * plan.packed_tokens,
        retained_bytes=None,
        caller_phase=caller_phase,
    )


def _observe(r, plan, rate):
    """One flat wave: its forward return, then its caller phase (backward)."""
    _update(r, plan, rate, caller_phase=False)
    _update(r, plan, rate, caller_phase=True)


def _required(r, plan, logical_tokens=None):
    return r._estimate_required_memory_bytes_from_values(
        packed_tokens=plan.packed_tokens,
        output_bytes=0,
        signature=plan.signature,
        logical_tokens=logical_tokens or plan.packed_tokens,
    )


def test_the_first_plans_one_time_costs_do_not_price_later_waves():
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    profile = r._memory_profiles[first.signature]
    assert profile.bytes_per_token == FIRST
    assert profile.warm_bytes_per_token is None
    assert profile.caller_plans == 1
    assert _required(r, larger) == int(FIRST * larger.packed_tokens * 1.1)
    _observe(r, larger, WARM)
    profile = r._memory_profiles[first.signature]
    assert profile.bytes_per_token == FIRST
    assert profile.warm_bytes_per_token == WARM
    assert profile.warm_packed_tokens == larger.packed_tokens
    assert profile.warm_logical_per_packed == 1
    assert _required(r, larger) == int(WARM * larger.packed_tokens * 1.1)
    # Smaller waves: the lower of today's rate and the warm fit at the
    # smallest later plan's size.
    assert _required(r, first) == int(FIRST * first.packed_tokens * 1.1)
    near = r._plan_flat_forward(requests(6144, 64))
    assert WARM * larger.packed_tokens < FIRST * near.packed_tokens
    assert _required(r, near) == int(WARM * larger.packed_tokens * 1.1)


def test_forward_only_observations_never_set_the_warm_fit():
    """Split children and dp_rank_forward observe forward only; a flat wave's
    caller phase also includes its backward."""
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    _update(r, larger, WARM, caller_phase=False)
    assert r._memory_profiles[first.signature].warm_packed_tokens is None
    assert _required(r, larger) == int(FIRST * larger.packed_tokens * 1.1)
    # A later caller phase is warm, and its peak includes the forward's.
    _update(r, larger, WARM, caller_phase=True)
    assert r._memory_profiles[first.signature].warm_bytes_per_token == WARM


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


def test_profiles_without_a_warm_fit_price_as_before():
    """Replayed reports and callers that set profiles directly carry no warm
    fit; the first caller phase after that seeds it."""
    r = rank()
    first, larger = _plans(r)
    r._memory_profiles[first.signature] = _MemoryProfile(
        bytes_per_token=FIRST, packed_tokens=first.packed_tokens
    )
    assert _required(r, first) == int(FIRST * first.packed_tokens * 1.1)
    _observe(r, larger, WARM)
    assert r._memory_profiles[first.signature].warm_bytes_per_token is None
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
    _observe(r, shared, WARM)
    assert r._memory_profiles[larger.signature].warm_logical_per_packed == 64
    assert _required(r, larger, deep) < today()[1]


def test_cost_stays_monotone_across_the_warm_extent():
    """The width search accepts on a larger layout's price and rejects on a
    smaller one's, so a smaller wave must never price above a larger one."""
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    _observe(r, larger, WARM)
    n = larger.packed_tokens

    def required(packed, logical=None):
        return r._estimate_required_memory_bytes_from_values(
            packed_tokens=packed,
            output_bytes=0,
            signature=larger.signature,
            logical_tokens=logical or packed,
        )

    sweep = [required(t) for t in (1, n // 2, (2 * n) // 3, n - 1, n, n + 1, 2 * n)]
    assert sweep == sorted(sweep)
    shared = [required(t, 4 * n) for t in (1, n // 4, n - 1, n, 2 * n, 4 * n)]
    assert shared == sorted(shared)


def test_packed_priced_cost_stays_monotone_across_the_warm_extent(monkeypatch):
    # The fixture's requests are short; the short-request gate has its own tests.
    monkeypatch.setattr(_impl, "_PACKED_PRICED_MIN_REQUEST_TOKENS", 1)
    r = packed_rank()
    signature = r._plan_flat_forward(packed_requests("target_tokens")).signature
    assert _packed_priced(signature, r._one_layer_recompute())
    today = _MemoryProfile(bytes_per_token=50_000, packed_tokens=8)
    warm = replace(
        today,
        warm_bytes_per_token=20_000,
        warm_packed_tokens=1000,
        warm_logical_per_packed=4,
    )

    def cost(profile, packed, logical=8_000):
        r._memory_profiles[signature] = profile
        return r._subforward_cost(
            packed_tokens=packed,
            logical_tokens=logical,
            output_bytes=0,
            signature=signature,
        ).required

    sweep = [cost(warm, t) for t in (1, 500, 999, 1000, 1001, 2000, 8000)]
    assert sweep == sorted(sweep)
    # Just below the warm extent: priced as the smallest later plan, not today.
    assert sweep[2] == sweep[3] < cost(today, 999)
    rows = [cost(warm, 1000, logical) for logical in (1000, 8_000, 32_000, 64_000)]
    assert rows == sorted(rows)


def test_split_children_never_set_the_warm_fit(monkeypatch):
    """The real split ladder: children observe forward only, and a split
    wave's caller peak goes to split floors, never to the warm fit."""
    from test_trainer_rank_split import _packed_budget, _request
    from test_trainer_rank_split import _rank as split_rank

    from art.trainer_rank import ForwardOutput

    r = split_rank(monkeypatch)
    monkeypatch.setattr(r, "_retained_memory_bytes", lambda *_args, **_kwargs: 0)
    forward_rate, backward_rate = 100, 300
    phases = []

    def update(plan, baseline, retained_after=None, *, caller_phase=False, **_):
        # Stands in for the CUDA peak read: backward peaks above forward.
        phases.append((plan.request_count, caller_phase))
        rate = backward_rate if caller_phase else forward_rate
        r._update_memory_profile(
            plan,
            plan.output_bytes + rate * plan.packed_tokens,
            retained_bytes=None,
            caller_phase=caller_phase,
        )

    def run(plan, **_kwargs):
        # Production's forward-return update, then no CUDA baseline.
        r._update_peak_memory_profile(plan, 0, 0)
        return [ForwardOutput(None, None, None, None)] * plan.request_count, None

    monkeypatch.setattr(r, "_update_peak_memory_profile", update)
    monkeypatch.setattr(r, "_run_flat_plan_with_memory_tracking", run)
    _packed_budget(monkeypatch, r, 20)
    items = [[_request(0)], [_request(m) for m in range(1, 5)], [_request(5)]]
    batches = r.forward_micro_batches(items)
    next(batches)
    assert next(batches).stats.subforward_count > 1
    (signature,) = r._memory_profiles
    # The seed's caller phase, then forward-only split children: no warm fit.
    assert r._memory_profiles[signature].warm_packed_tokens is None
    assert [p for p in phases if p[1]] == [(1, True)]
    list(batches)
    # Only the later flat wave's caller phase fits the warm rate.
    assert [p for p in phases if p[1]] == [(1, True), (1, True)]
    assert r._memory_profiles[signature].warm_bytes_per_token == backward_rate


def test_a_width_accepted_on_the_no_sharing_bound_never_executes_above_it(
    monkeypatch,
):
    """The width search accepts a width on the cheap no-sharing count. With a
    warm fit, that count can reach the warm size while the shared layout
    executed falls below it; the executed plan must not price higher."""
    from art.trainer_rank import ForwardInput, ForwardOutput

    monkeypatch.setattr(_impl, "_PACKED_PRICED_MIN_REQUEST_TOKENS", 1)
    r = packed_rank()
    monkeypatch.setattr(r, "_dp_rank_and_size", lambda: (0, 1))
    prefix = torch.arange(100)
    items = [
        [
            ForwardInput(
                input_tokens=torch.cat(
                    [prefix, torch.arange(1000 + 10 * i, 1010 + 10 * i)]
                ),
                target_tokens=torch.cat(
                    [prefix, torch.arange(1000 + 10 * i, 1010 + 10 * i)]
                ),
            )
            for i in range(4)
        ]
    ]
    plan = r._plan_flat_forward(items[0])
    assert plan.packed_tokens < 200 <= plan.logical_tokens
    r._memory_profiles[plan.signature] = _MemoryProfile(
        bytes_per_token=100_000,
        packed_tokens=plan.logical_tokens,
        logical_per_packed=4,
        warm_bytes_per_token=20_000,
        warm_packed_tokens=200,
        warm_logical_per_packed=4,
    )
    own = r._memory_check(plan).estimated_required_bytes
    monkeypatch.setattr(r, "_available_memory_bytes", lambda: 10 * own)
    monkeypatch.setattr(
        r,
        "_run_flat_plan_with_memory_tracking",
        lambda plan, **_: (
            [ForwardOutput(None, None, None, None)] * plan.request_count,
            None,
        ),
    )
    (batch,) = list(r.forward_micro_batches(items))
    # Accepted on the no-sharing bound, which priced more tokens than ran.
    assert batch.stats.packed_tokens == plan.packed_tokens
    assert batch.stats.estimated_required_bytes > own


def _peak_reader(monkeypatch, r):
    """Stand in for the CUDA peak counter, reset by each tracked forward."""
    state = {"peak": 0}
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: state["peak"])

    def forward(plan, peak):
        r._peak_resets = r.__dict__.get("_peak_resets", 0) + 1
        state["peak"] = peak
        r._update_peak_memory_profile(plan, 0, 0)
        return r._peak_reading

    return state, forward


def _caller_phase(r, plan, interval):
    r._update_peak_memory_profile(plan, 0, caller_phase=True, interval=interval)


def test_only_a_whole_caller_phase_fits_the_warm_profile(monkeypatch):
    r = rank()
    first, larger = _plans(r)
    state, forward = _peak_reader(monkeypatch, r)
    n = larger.packed_tokens
    _caller_phase(
        r, first, forward(first, first.output_bytes + FIRST * first.packed_tokens)
    )
    signature = first.signature

    def wave(interrupt):
        interval = forward(larger, larger.output_bytes + WARM * n)
        state["peak"] = larger.output_bytes + (WARM + 500_000) * n  # backward
        interrupt()
        _caller_phase(r, larger, interval)
        return r._memory_profiles[signature]

    # A nested forward during the yield resets the counter: not whole, even
    # when its own peak is higher than this wave's forward.
    def nested():
        forward(larger, larger.output_bytes + (WARM + 100_000) * n)

    assert wave(nested).warm_packed_tokens is None
    # An untracked reset: the counter falls below this wave's forward peak.
    assert wave(lambda: state.update(peak=0)).warm_packed_tokens is None
    # Their readings raised the pending rate; an uninterrupted wave fits it.
    assert wave(lambda: None).warm_bytes_per_token == WARM + 500_000


def test_other_readings_raise_but_never_fit_the_warm_profile():
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    _update(r, larger, WARM, caller_phase=False)
    assert r._memory_profiles[first.signature].warm_packed_tokens is None
    _observe(r, larger, WARM)
    profile = r._memory_profiles[first.signature]
    assert profile.warm_bytes_per_token == WARM
    # A higher forward-only or interrupted reading raises the warm rate, but
    # neither extends the warm extent nor its sharing.
    middle = r._plan_flat_forward(requests(4096, 64))
    shared = replace(middle, logical_tokens=middle.packed_tokens * 8)
    _update(r, shared, WARM + 300_000, caller_phase=False)
    raised = r._memory_profiles[first.signature]
    assert raised.warm_bytes_per_token == WARM + 300_000
    assert raised.warm_packed_tokens == profile.warm_packed_tokens
    assert raised.warm_logical_per_packed == profile.warm_logical_per_packed
    _update(r, larger, WARM - 100_000, caller_phase=False)
    assert r._memory_profiles[first.signature].warm_bytes_per_token == WARM + 300_000


def test_a_nested_forward_during_the_yield_cannot_fit_the_warm_profile(monkeypatch):
    """The real micro-batch loop: a caller that runs another tracked forward
    after its backward leaves a peak counter reset since the wave's forward."""
    from test_trainer_rank_split import _packed_budget, _request
    from test_trainer_rank_split import _rank as split_rank

    from art.trainer_rank import ForwardOutput

    r = split_rank(monkeypatch)
    monkeypatch.setattr(r, "_retained_memory_bytes", lambda *_args, **_kwargs: 0)
    state, forward = _peak_reader(monkeypatch, r)

    def run(plan, **_kwargs):
        forward(plan, plan.output_bytes + 100 * plan.packed_tokens)
        return [ForwardOutput(None, None, None, None)] * plan.request_count, 0

    monkeypatch.setattr(r, "_run_flat_plan_with_memory_tracking", run)
    _packed_budget(monkeypatch, r, 10)
    items = [[_request(m)] for m in range(3)]
    batches = r.forward_micro_batches(items)
    for index, batch in enumerate(batches):
        (signature,) = r._memory_profiles
        state["peak"] += 200 * batch.stats.packed_tokens  # the caller's backward
        if index == 1:
            run(r._plan_flat_forward(items[0]))  # a nested tracked forward
    profile = r._memory_profiles[signature]
    assert profile.caller_plans == 2
    # Only the uninterrupted third wave fit the warm rate, backward included.
    assert profile.warm_bytes_per_token == 300


def test_each_tracked_forward_starts_a_new_peak_interval(monkeypatch):
    """The real tracked forward: its counter reset starts a new interval, so a
    caller phase that continues an earlier one is not whole."""
    r = rank()
    first, larger = _plans(r)
    state = {"peak": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *_: state.update(peak=0)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: state["peak"])
    monkeypatch.setattr(r, "device", torch.device("cuda", 0))

    def execute(plan):
        state["peak"] = plan.output_bytes + FIRST * plan.packed_tokens
        return [None] * plan.request_count

    monkeypatch.setattr(r, "_execute_flat_plan", execute)
    check = _impl._MemoryCheck(0, 0, True)
    _, baseline = r._run_flat_plan_with_memory_tracking(first, check=check, context="t")
    seed = r._peak_reading
    _caller_phase(r, first, seed)
    _, baseline = r._run_flat_plan_with_memory_tracking(
        larger, check=check, context="t"
    )
    interval = r._peak_reading
    assert baseline == 0 and interval[0] == seed[0] + 1
    r._run_flat_plan_with_memory_tracking(first, check=check, context="t")  # nested
    _caller_phase(r, larger, interval)
    assert r._memory_profiles[larger.signature].warm_packed_tokens is None


def test_a_higher_reading_before_the_first_warm_plan_is_kept():
    """A seed, then a higher forward-only reading, then a lower whole warm
    plan: the warm rate keeps the higher post-seed reading."""
    r = rank()
    first, larger = _plans(r)
    _observe(r, first, FIRST)
    _update(r, larger, WARM + 300_000, caller_phase=False)
    assert _required(r, larger) == int(FIRST * larger.packed_tokens * 1.1)
    _observe(r, larger, WARM)
    profile = r._memory_profiles[first.signature]
    assert profile.warm_bytes_per_token == WARM + 300_000
    assert _required(r, larger) == int((WARM + 300_000) * larger.packed_tokens * 1.1)
