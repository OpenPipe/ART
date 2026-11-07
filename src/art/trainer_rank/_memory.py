"""TrainerRank memory estimation, profiling and admission accounting.

These are ``TrainerRank`` method bodies moved out of ``_impl`` verbatim: each
function takes the owning rank as ``self`` and ``TrainerRank`` binds them as
methods, so ``self._x(...)`` dispatch and per-instance overrides keep working.

Module globals the bodies used to read from ``_impl`` (``torch``, ``dist``,
``os``, sibling helpers) are still resolved through ``_impl`` at call time, so
tests that patch ``_impl.torch`` and friends keep intercepting them; only pure
stdlib helpers are imported here directly. Referencing ``_impl`` as a module
also lets the circular import resolve lazily.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import math
from types import MethodType
from typing import TYPE_CHECKING, Any

from art.trainer_rank import _impl, _memory_model

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from art.megatron.lora import LoRASlotRef
    from art.trainer_rank._impl import AdapterSelection, AnyForwardInput, TrainerRank


@dataclass(frozen=True)
class _ScopedMemorySignature(_impl._MemorySignature):
    checkpoint_partition: tuple[int, tuple[int, int, int]] | None = None
    bounded_forward_peak: bool = False
    bounded_caller_peak: bool = False


_CPWorkload = tuple[str, tuple[tuple[int, ...], ...]]
_CPCallerPeak = tuple[int, int, str, tuple[tuple[int, ...], ...]]

_FIXED_SLOPE_PROFILES: ContextVar[bool] = ContextVar(
    "art_fixed_slope_profiles", default=False
)


@contextmanager
def fixed_slope_memory_profiles(enabled: bool = True) -> Iterator[None]:
    """Collect fresh v2 profiles; existing v2 profiles retain their model outside this scope."""
    token = _FIXED_SLOPE_PROFILES.set(enabled)
    try:
        yield
    finally:
        _FIXED_SLOPE_PROFILES.reset(token)


def _fixed_slope_enabled(profile: _impl._MemoryProfile | None) -> bool:
    return _FIXED_SLOPE_PROFILES.get() or (
        profile is not None
        and profile.peak_model_version == _memory_model._MEMORY_PROFILE_VERSION
    )


def _unanchored(profile: _impl._MemoryProfile | None) -> bool:
    points = (
        profile.rate_observations
        if profile is not None and profile.profile_format_version >= 3
        else profile.compute_observations
        if profile is not None
        else ()
    )
    return (
        profile is None
        or not points
        or points[-1][0] < _memory_model._MEMORY_SLOPE_MIN_TOKENS
    )


_MEMORY_DEFERRED_BYTES = 1 << 62


def _rate_floor_tokens(signature: _impl._MemorySignature) -> int:
    # Global pre-shard rows: 64 rows per CP rank, capped so every 8K anchor
    # qualifies. Neither a local shard nor its ownership vector sets this floor.
    return min(
        64 * max(1, signature.topology[2]), _memory_model._MEMORY_SLOPE_MIN_TOKENS
    )


def _rate_points(profile: _impl._MemoryProfile, floor: int):
    if profile.peak_model_version != 2:
        return ()
    if profile.profile_format_version >= 3:
        return tuple(
            point for point in profile.rate_observations or () if point[0] >= floor
        )
    # Earlier v2 hulls may have discarded a qualifying intermediate rate in
    # favor of a tiny peak. Only their definitely anchored observations seed
    # the new rate channel; never reuse their max-merged scalar rate.
    return tuple(point for point in profile.compute_observations if point[0] >= 8192)


def _absolute_floor(profile: _impl._MemoryProfile, floor: int) -> int:
    return max(
        profile.absolute_compute_floor,
        max((peak for n, peak in profile.compute_observations if n < floor), default=0),
    )


def _trusted_extent(profile: _impl._MemoryProfile) -> int:
    if profile.profile_format_version >= 4:
        return max(profile.eligible_packed_tokens, profile.subfloor_packed_tokens)
    if profile.profile_format_version == 3:
        # Old peak hulls have already lost some sub-floor token extents. Do not
        # infer a replicated learning decision from rank-local peak pruning.
        return max((n for n, _ in profile.rate_observations or ()), default=0)
    return profile.packed_tokens


_RETAINED_POINT_BUDGET = 64


def _merge_retained_peaks(*observations: Iterable[tuple[int, int]]):
    """Bounded monotone concave majorant of all retained observations.

    Round token positions DOWN onto a fixed power-of-two grid. A monotone
    majorant at that earlier position also bounds the original observation.
    Prefix MAX plus the upper hull bounds every intervening prefix floor;
    queries round linear interpolation UP. There are at most 64 grid points,
    even for arbitrary Python integers (the final bucket is saturated).
    Hull vertices are sufficient constraints, so union/refit is associative.
    """
    peaks: dict[int, int] = {}
    for rows in observations:
        for n, peak in rows:
            if type(n) is not int or n <= 0 or type(peak) is not int or peak < 0:
                raise ValueError("invalid retained memory observation")
            bucket = 1 << min(n.bit_length() - 1, _RETAINED_POINT_BUDGET - 1)
            peaks[bucket] = max(peaks.get(bucket, 0), peak)
    result = []
    maximum = 0
    for n, peak in sorted(peaks.items()):
        maximum = max(maximum, peak)
        result.append((n, maximum))
    return _memory_model.merge_points(result)


def _retained_floor_peak(points: tuple[tuple[int, int], ...], tokens: int) -> int:
    if not points or tokens < points[0][0]:
        return 0
    if tokens >= points[-1][0]:
        return points[-1][1]
    low, high = 0, len(points) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if points[middle][0] <= tokens:
            low = middle
        else:
            high = middle
    left, right = points[low], points[high]
    span = right[0] - left[0]
    return left[1] + ((right[1] - left[1]) * (tokens - left[0]) + span - 1) // span


def _retained_rate_extent(profile: _impl._MemoryProfile) -> int:
    if profile.profile_format_version >= 3:
        # Older v2 evidence has no independent retained-rate extent. Disable
        # that discount until a new qualified retained observation supplies it.
        return profile.retained_rate_tokens
    return profile.packed_tokens


def _retained_floor_points(profile: _impl._MemoryProfile):
    if profile.profile_format_version == 3 and profile.absolute_compute_floor:
        # This encoding did not persist retained extents. Its entire untrusted
        # compute peak may have remained live; preserve that bound on upgrade.
        return ((1, profile.absolute_compute_floor),)
    return profile.retained_compute_peaks


def _profile_scope(signature: _impl._MemorySignature):
    return (
        getattr(signature, "checkpoint_partition", None),
        getattr(signature, "bounded_forward_peak", False),
        getattr(signature, "bounded_caller_peak", False),
    )


@dataclass(frozen=True)
class _MemoryRateReference:
    source: str
    rate: tuple[int, int]
    fixed: int
    minimum: int
    extent: int


def _borrow_covers(
    profile: _impl._MemoryProfile,
    tokens: int | float | None,
    cp: int,
    workload: _CPWorkload | None,
) -> bool:
    if not isinstance(profile, _CallerMemoryProfile):
        return True
    if tokens is None:
        return False
    if cp > 1:
        return profile.covering_cp_peak(tokens, workload) is not None
    return any(tokens <= n for n, _ in profile.caller_compute_peaks)


def _rate_reference(
    self: TrainerRank,
    signature: _impl._MemorySignature,
    profile: _impl._MemoryProfile | None,
    *,
    packed_tokens: int | float | None = None,
    caller_cp_workload: _CPWorkload | None = None,
) -> _MemoryRateReference | None:
    from fractions import Fraction

    floor = _rate_floor_tokens(signature)

    def reference(source, profiles):
        points = [(p, _rate_points(p, floor)) for p in profiles]
        points = [(p, rows) for p, rows in points if rows]
        if not points:
            return None
        if source != "signature":
            minimum = max(rows[0][0] for _, rows in points)
            points = [
                (p, rows)
                for p, rows in points
                if _borrow_covers(
                    p,
                    None if packed_tokens is None else max(packed_tokens, minimum),
                    signature.topology[2],
                    caller_cp_workload,
                )
            ]
            if not points:
                return None
        rate = max(
            (point for _, rows in points for point in rows),
            key=lambda point: (Fraction(point[1], point[0]), point[0]),
        )
        return _MemoryRateReference(
            source,
            (rate[1], rate[0]),
            max(
                max(p.fixed_compute_bytes, _absolute_floor(p, floor)) for p, _ in points
            ),
            max(rows[0][0] for _, rows in points),
            max(rows[-1][0] for _, rows in points),
        )

    if profile is not None and (own := reference("signature", (profile,))):
        return own
    # A TrainerRank owns one model. Never cross its replica geometry, operation
    # or planning regime. Root counterparts additionally match the base key.
    compatible = [
        (key, value)
        for key, value in self._memory_profiles.items()
        if key.topology == signature.topology
        and key.planner_coefficients == signature.planner_coefficients
        and key.grad_enabled == signature.grad_enabled
        and key.grad_modes == signature.grad_modes
        and key.slot_shapes == signature.slot_shapes
        and key.slot_group_count == signature.slot_group_count
        and value.peak_model_version == 2
        and (not signature.grad_enabled or value.caller_plans > 0)
        and _borrow_covers(
            value, packed_tokens, signature.topology[2], caller_cp_workload
        )
    ]
    roots = [
        value
        for key, value in compatible
        if isinstance(key, _ScopedMemorySignature)
        and (
            (key.bounded_caller_peak and signature.grad_enabled)
            or (key.bounded_forward_peak and not signature.grad_enabled)
        )
        and key.checkpoint_partition is None
        and all(
            getattr(key, name) == getattr(signature, name)
            for name, field in _impl._MemorySignature.__dataclass_fields__.items()
            if field.compare
        )
    ]
    if root := reference("calibration_root", roots):
        return root
    return reference(
        "model_anchor",
        [
            p
            for key, p in compatible
            if not _unanchored(p) and _profile_scope(key) == _profile_scope(signature)
        ],
    )


def _cold_extrapolation_factor(
    profile: _impl._MemoryProfile | None, packed_tokens: int
) -> float:
    if _fixed_slope_enabled(profile) and packed_tokens > (
        0 if profile is None else profile.packed_tokens
    ):
        # Anchoring selects NEW's slope; it does not qualify extrapolation.
        # The 8K-only boot option still needs the cold margin beyond its last
        # observation (the supplied sweep otherwise under-predicts at 131K).
        return _memory_model._MEMORY_COLD_EXTRAPOLATION_FACTOR
    return 1.0


def _profile_compute_peak(profile: _impl._MemoryProfile, tokens: int | float) -> float:
    if profile.peak_model_version == _memory_model._MEMORY_PROFILE_VERSION:
        # A lone long anchor does not qualify downward extrapolation of fixed
        # workspace. Price below the observed range at its smallest observation.
        extent = (
            profile.rate_observations
            if profile.profile_format_version >= 3 and profile.rate_observations
            else profile.compute_observations
        )
        if extent:
            tokens = max(tokens, extent[0][0])
        return max(
            _memory_model.predict(
                profile.fixed_compute_bytes, profile.compute_slope, tokens
            ),
            profile.legacy_bytes_per_token * tokens,
        )
    return profile.bytes_per_token * tokens


def _merge_memory_profile_model_fields(
    *profiles: _impl._MemoryProfile,
) -> dict[str, Any]:
    """Profile restore hook: union evidence, never max fitted coefficients."""
    if all(p.peak_model_version == 1 for p in profiles):
        return {}
    observations = _memory_model.merge_points(
        *(p.compute_observations for p in profiles)
    )
    extra: dict[str, Any] = {}
    formats = [p for p in profiles if p.profile_format_version >= 3]
    if formats:
        floor = formats[0].rate_floor_tokens
        if any(p.rate_floor_tokens != floor for p in formats):
            raise ValueError("incompatible global memory rate floors")
        rates = _memory_model.merge_points(*(_rate_points(p, floor) for p in profiles))
        absolute = max(_absolute_floor(p, floor) for p in profiles)
        extra = dict(
            profile_format_version=5,
            rate_floor_tokens=floor,
            rate_observations=rates,
            absolute_compute_floor=absolute,
            eligible_packed_tokens=max(
                max((n for n, _ in _rate_points(p, floor)), default=0)
                if p.profile_format_version < 4
                else p.eligible_packed_tokens
                for p in profiles
            ),
            subfloor_packed_tokens=max(p.subfloor_packed_tokens for p in profiles),
            retained_rate_tokens=max(p.retained_rate_tokens for p in profiles),
            retained_compute_peaks=_merge_retained_peaks(
                *(_retained_floor_points(p) for p in profiles)
            ),
        )
        fixed, slope = _memory_model.fit_points(
            observations, rate_points=rates, absolute_floor=absolute
        )
    else:
        fixed, slope = _memory_model.fit_points(observations)
    return dict(
        peak_model_version=_memory_model._MEMORY_PROFILE_VERSION,
        compute_observations=observations,
        fixed_compute_bytes=fixed,
        compute_slope=slope,
        legacy_bytes_per_token=max(
            max(
                p.legacy_bytes_per_token,
                p.bytes_per_token if p.peak_model_version == 1 else 0,
            )
            for p in profiles
        ),
        **extra,
    )


def _validate_memory_profile_model(profile: _impl._MemoryProfile) -> None:
    """Validate the extra v2 fields after a persisted dataclass is decoded."""
    if type(
        profile.peak_model_version
    ) is not int or profile.peak_model_version not in (1, 2):
        raise ValueError("unsupported memory profile version")
    if profile.peak_model_version == 1:
        if (
            profile.compute_observations
            or profile.fixed_compute_bytes != 0
            or profile.compute_slope != (0, 1)
            or profile.legacy_bytes_per_token != 0
            or profile.profile_format_version != 1
            or profile.rate_floor_tokens != 0
            or profile.rate_observations is not None
            or profile.absolute_compute_floor != 0
            or profile.eligible_packed_tokens != 0
            or profile.subfloor_packed_tokens != 0
            or profile.retained_rate_tokens != 0
            or profile.retained_compute_peaks
        ):
            raise ValueError("unexpected fixed/slope fields in legacy profile")
        return
    points = profile.compute_observations
    if not points or _memory_model.merge_points(points) != points:
        raise ValueError("invalid memory observation hull")
    if profile.profile_format_version in (3, 4, 5):
        floor, rates, absolute = (
            profile.rate_floor_tokens,
            profile.rate_observations,
            profile.absolute_compute_floor,
        )
        if (
            type(floor) is not int
            or not 64 <= floor <= 8192
            or floor % 64
            or type(rates) is not tuple
            or _memory_model.merge_points(rates) != rates
            or any(n < floor for n, _ in rates)
            or type(absolute) is not int
            or absolute < 0
            or absolute < max((peak for n, peak in points if n < floor), default=0)
            or profile.packed_tokens < max((n for n, _ in rates), default=0)
        ):
            raise ValueError("invalid rate-qualified memory evidence")
        fixed, slope = _memory_model.fit_points(
            points, rate_points=rates, absolute_floor=absolute
        )
    elif (
        profile.profile_format_version == 2
        and profile.rate_floor_tokens == 0
        and profile.rate_observations is None
        and profile.absolute_compute_floor == 0
    ):
        fixed, slope = _memory_model.fit_points(points)
    else:
        raise ValueError("unsupported memory evidence format")
    extents = (
        profile.eligible_packed_tokens,
        profile.subfloor_packed_tokens,
        profile.retained_rate_tokens,
    )
    if profile.profile_format_version >= 4:
        if (
            any(
                type(n) is not int or not 0 <= n <= profile.packed_tokens
                for n in extents
            )
            or profile.eligible_packed_tokens
            != max((n for n, _ in profile.rate_observations or ()), default=0)
            or profile.subfloor_packed_tokens >= profile.rate_floor_tokens
            or profile.retained_rate_tokens > profile.eligible_packed_tokens
            or type(profile.retained_compute_peaks) is not tuple
            or any(
                type(n) is not int or n <= 0 or type(peak) is not int or peak < 0
                for n, peak in profile.retained_compute_peaks
            )
            or any(
                a[0] >= b[0] or a[1] > b[1]
                for a, b in zip(
                    profile.retained_compute_peaks, profile.retained_compute_peaks[1:]
                )
            )
            or (
                profile.profile_format_version == 5
                and (
                    len(profile.retained_compute_peaks) > _RETAINED_POINT_BUDGET
                    or _merge_retained_peaks(profile.retained_compute_peaks)
                    != profile.retained_compute_peaks
                )
            )
            or any(n > profile.packed_tokens for n, _ in profile.retained_compute_peaks)
        ):
            raise ValueError("invalid independent memory extents or retained floors")
    elif any(extents) or profile.retained_compute_peaks:
        raise ValueError("unexpected independent memory evidence in old format")
    if (
        type(profile.fixed_compute_bytes) is not int
        or profile.fixed_compute_bytes != fixed
        or len(profile.compute_slope) != 2
        or any(type(v) is not int for v in profile.compute_slope)
        or profile.compute_slope != slope
        or not math.isfinite(profile.legacy_bytes_per_token)
        or profile.legacy_bytes_per_token < 0
        or profile.packed_tokens < points[-1][0]
    ):
        raise ValueError("invalid fixed/slope memory fit")


@dataclass(frozen=True)
class _CallerMemoryProfile(_impl._MemoryProfile):
    caller_compute_peaks: tuple[tuple[int, int], ...] = ()
    compute_peak_floor: int = 0
    caller_cp_peaks: tuple[_CPCallerPeak, ...] = ()

    def covering_cp_peak(
        self, tokens: int | float, cp_workload: _CPWorkload | None
    ) -> int | None:
        if cp_workload is not None:
            family, loads = cp_workload
            for length, peak, measured_family, measured in self.caller_cp_peaks:
                if (
                    tokens <= length
                    and family == measured_family
                    and len(loads) == len(measured)
                    and all(
                        len(want) == len(have)
                        and all(w <= h for w, h in zip(want, have, strict=True))
                        for want, have in zip(loads, measured, strict=True)
                    )
                ):
                    return peak
        return None

    def compute_peak(
        self,
        tokens: int | float,
        *,
        context_parallel: bool = False,
        cp_workload: _CPWorkload | None = None,
    ) -> float:
        # CP peaks are rank-local. Require componentwise coverage on every
        # rank, including workspace geometry; a max-rank comparison could reuse
        # another rank's larger reading without ever measuring this rank's load.
        if context_parallel:
            if (peak := self.covering_cp_peak(tokens, cp_workload)) is not None:
                return max(
                    _profile_compute_peak(self, tokens)
                    if self.peak_model_version == 2
                    else peak,
                    self.compute_peak_floor,
                )
        else:
            if self.peak_model_version == 2:
                return max(_profile_compute_peak(self, tokens), self.compute_peak_floor)
            # Preserve #1165's CP1 curve and arithmetic exactly.
            for length, peak in self.caller_compute_peaks:
                if tokens <= length:
                    return max(peak, self.compute_peak_floor)
        # No discounted extrapolation: even a higher short reload cannot erase
        # earlier growth constraints beyond the measured caller extent.
        if context_parallel and self.profile_format_version >= 3:
            # Removing tiny-derived rates must not reopen an uncovered CP
            # ownership/workspace class. Keep its existing collective refusal.
            return max(_MEMORY_DEFERRED_BYTES, _profile_compute_peak(self, tokens))
        return max(
            self.bytes_per_token * tokens,
            self.legacy_bytes_per_token * tokens,
            self.compute_peak_floor,
        )


@dataclass(frozen=True)
class _BoundedMemoryProfile(_impl._MemoryProfile):
    forward_compute_peaks: tuple[tuple[int, int], ...] = ()

    def compute_peak(self, tokens: int | float) -> float:
        if self.peak_model_version == _memory_model._MEMORY_PROFILE_VERSION:
            return _profile_compute_peak(self, tokens)
        for length, peak in self.forward_compute_peaks:
            if tokens <= length:
                return peak
        longest, peak = self.forward_compute_peaks[-1]
        slope = max(
            (
                (high_peak - low_peak) / (high - low)
                for i, (low, low_peak) in enumerate(self.forward_compute_peaks)
                for high, high_peak in self.forward_compute_peaks[i + 1 :]
                if high >= 2 * low
            ),
            default=None,
        )
        horizon = 4 * longest
        # One observation cannot separate fixed from growing memory; grow
        # proportionally until a second root fits a marginal rate.
        rate = peak / longest if slope is None else max(slope, peak / horizon)
        if tokens <= horizon:
            return peak + rate * (tokens - longest)
        return (peak + rate * (horizon - longest)) * tokens / horizon


@dataclass(frozen=True)
class _CallerMemoryProfileV2(_impl._MemoryProfileV2, _CallerMemoryProfile):
    pass


@dataclass(frozen=True)
class _BoundedMemoryProfileV2(_impl._MemoryProfileV2, _BoundedMemoryProfile):
    pass


def _memory_profile_from_values(values: dict[str, Any]) -> _impl._MemoryProfile:
    """Decode the versioned state while preserving the legacy record classes."""
    values = dict(values)
    version = values.get("peak_model_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ValueError("unsupported memory profile version")
    if version == 1:
        # Accept earlier v1 encoders that explicitly emitted inert v2 defaults.
        for name in (
            "peak_model_version",
            "compute_observations",
            "fixed_compute_bytes",
            "compute_slope",
            "legacy_bytes_per_token",
            "profile_format_version",
            "rate_floor_tokens",
            "rate_observations",
            "absolute_compute_floor",
            "eligible_packed_tokens",
            "subfloor_packed_tokens",
            "retained_rate_tokens",
            "retained_compute_peaks",
        ):
            if name in values and values.pop(name) != getattr(
                _impl._MemoryProfile, name
            ):
                raise ValueError("unexpected fixed/slope fields in legacy profile")
    cls = _impl._MemoryProfileV2 if version == 2 else _impl._MemoryProfile
    if "caller_compute_peaks" in values:
        cls = _CallerMemoryProfileV2 if version == 2 else _CallerMemoryProfile
    elif "forward_compute_peaks" in values:
        cls = _BoundedMemoryProfileV2 if version == 2 else _BoundedMemoryProfile
    profile = cls(**values)
    _validate_memory_profile_model(profile)
    if profile.profile_format_version == 4:
        profile = replace(
            profile,
            profile_format_version=5,
            retained_compute_peaks=_merge_retained_peaks(
                profile.retained_compute_peaks
            ),
        )
        _validate_memory_profile_model(profile)
    return profile


_BOUNDED_FORWARD_PEAKS: ContextVar[bool] = ContextVar(
    "art_bounded_forward_peaks", default=False
)


@contextmanager
def bounded_forward_peaks() -> Iterator[None]:
    """Opt into single-root forward peak calibration.

    Keep at most eight (packed tokens, compute peak) observations per signature,
    max-merging repeated lengths and taking a nondecreasing upper envelope.
    Retain the shortest and seven longest lengths when the set fills. Within
    that range, use the smallest covering root's peak.
    Beyond it, grow through 4x the longest root at the greater of the steepest
    nonnegative marginal rate between retained roots at least 2x apart and a
    quarter of the longest root's peak/token rate. Without such a pair, grow at
    the full peak/token rate. Beyond that trust horizon, grow proportionally
    from the horizon estimate.
    Analytic floors still apply. Grow calibration roots before admitting long
    requests; profiles remain separate from ordinary TrainerRank requests.
    """
    token = _BOUNDED_FORWARD_PEAKS.set(True)
    try:
        yield
    finally:
        _BOUNDED_FORWARD_PEAKS.reset(token)


_DIRECT_FORWARD: ContextVar[bool] = ContextVar(
    "art_direct_forward_profile", default=False
)


@contextmanager
def direct_forward_profile() -> Iterator[None]:
    """Keep direct-forward calibration separate from yielded caller phases."""
    token = _DIRECT_FORWARD.set(True)
    try:
        yield
    finally:
        _DIRECT_FORWARD.reset(token)


def _split_required_memory(costs: Sequence[_impl._SubforwardCost]) -> int:
    # A grown HybridEP buffer persists into later children: charge the
    # largest growth once, beside whichever child peaks highest.
    growth = max(cost.hybridep_growth for cost in costs)
    required = (
        sum(cost.retained for cost in costs)
        + max(
            cost.ephemeral - int(cost.hybridep_growth * _impl._MEMORY_SAFETY_FACTOR)
            for cost in costs
        )
        + int(growth * _impl._MEMORY_SAFETY_FACTOR)
    )
    if any(cost.checkpoint_input_gradient for cost in costs):
        # The caller owns all returned graphs. A calibrated forward-retained
        # discount cannot replace the sum of their input-gradient extents.
        # Children training the same slots share their adapter gradients,
        # allocated once by whichever child's backward reaches a layer
        # first; the largest child's extra covers any order. Different
        # slots have disjoint gradients, charged per child.
        adapter = [
            cost.checkpoint_adapter_gradient
            for cost in costs
            if cost.checkpoint_adapter_gradient
        ]
        shared = (
            len(
                {
                    cost.checkpoint_adapter_gradient_slots
                    for cost in costs
                    if cost.checkpoint_adapter_gradient
                }
            )
            <= 1
        )
        checkpoint = (
            sum(
                cost.checkpoint_retained + cost.checkpoint_input_gradient
                for cost in costs
            )
            + max(cost.checkpoint_workspace for cost in costs)
            + ((max(adapter) if shared else sum(adapter)) if adapter else 0)
            + growth
        )
        required = max(required, int(checkpoint * _impl._MEMORY_SAFETY_FACTOR))
    return required


def _split_memory_key(plan: _impl._SplitForwardPlan) -> bytes | None:
    # Bound traversal and hashing; retain only a digest, never tensor values,
    # token tables or graphs. Oversized structural identities stay unlearned.
    if len(plan.subforwards) > 1024:
        return None
    digest, remaining, nodes = (
        hashlib.sha256(),
        262144 - 32 * len(plan.subforwards),
        65536,
    )

    def feed(value: Any) -> None:
        nonlocal remaining, nodes
        nodes -= 1
        if nodes < 0:
            raise ValueError("split key node cap")
        if isinstance(value, tuple):
            remaining -= 2
            if remaining < 0:
                raise ValueError("split key byte cap")
            feed(len(value))
            digest.update(b"(")
            for child in value:
                feed(child)
            digest.update(b")")
            return
        if value is not None and type(value) not in (str, int, bool):
            raise ValueError("unsupported split key field")
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError("split key string cap")
        if isinstance(value, int) and value.bit_length() > 64:
            raise ValueError("split key integer cap")
        encoded = repr(value).encode()
        remaining -= len(encoded) + 1
        if remaining < 0:
            raise ValueError("split key byte cap")
        digest.update(encoded + b";")

    try:
        feed((plan.request_count, len(plan.subforwards)))
        outer, children = digest, []
        for p, indices in zip(plan.subforwards, plan.request_indices, strict=True):
            digest = hashlib.sha256()
            signature = p.signature
            feed(
                (
                    indices,
                    signature.topology,
                    signature.planner_coefficients,
                    signature.slot_group_count,
                    signature.request_mix,
                    signature.grad_enabled,
                    signature.grad_modes,
                    signature.slot_shapes,
                    signature.short_requests,
                    p.packed_tokens,
                    p.logical_tokens,
                    p.inactive_logical_tokens,
                    p.output_bytes,
                    p.output_metadata,
                    p.selected_max_depth,
                    len(p.groups),
                )
            )
            if isinstance(signature, _ScopedMemorySignature):
                feed(
                    (
                        signature.checkpoint_partition,
                        signature.bounded_forward_peak,
                        signature.bounded_caller_peak,
                    )
                )
            for g in p.groups:
                slot = (
                    None
                    if g.slot_ref is None
                    else ("checkpoint", g.slot_ref.name)
                    if isinstance(g.slot_ref, _impl._LocalLoRASlotRef)
                    else (g.slot_ref.kind, g.slot_ref.name)
                )
                feed(
                    (
                        slot,
                        g.grad_enabled,
                        g.request_indices,
                        len(g.packed.segments),
                    )
                )
                for segment in g.packed.segments:
                    feed(
                        (
                            segment.sequence_indices,
                            segment.start,
                            segment.end,
                            segment.packed_start,
                            segment.group_id,
                            segment.parent_id,
                        )
                    )
                feed(len(g.items))
                for item in g.items:
                    feed(
                        (
                            tuple(item.input_ids.shape),
                            str(item.input_ids.dtype),
                            None
                            if item.labels is None
                            else (tuple(item.labels.shape), str(item.labels.dtype)),
                            item.request.top_k,
                            item.request.logits,
                            item.request.hidden_states,
                        )
                    )
            children.append(digest.digest())
    except ValueError:
        return None
    # Cost/profile updates may reorder the same children. Each digest still
    # binds its original request mapping/partition; retain the observed max
    # across execution orders, not an unmeasured bound for every order.
    outer.update(b"".join(sorted(children)))
    return outer.digest()


def _record_split_memory_floor(
    self: TrainerRank, plan: _impl._SplitForwardPlan, baseline: int, forward_peak: int
) -> None:
    # Only normal caller completion learns a floor; child retained profiles
    # stay forward-only. Keep earlier child peaks despite their later resets.
    key = self._split_memory_key(plan)
    if key is None:
        self._split_memory_floor_status = "unsupported_key"
        return
    if key not in self._split_memory_floors and len(self._split_memory_floors) >= 1024:
        self._split_memory_floor_status = "cache_full_not_learned"
        return
    observed = max(
        0,
        max(forward_peak, int(_impl.torch.cuda.max_memory_allocated(self.device)))
        - baseline,
    )
    self._split_memory_floors[key] = max(
        self._split_memory_floors.get(key, 0), observed
    )
    self._split_memory_floor_status = "recorded"


def _split_plan_memory_check(
    self: TrainerRank,
    plan: _impl._SplitForwardPlan,
    costs: Sequence[_impl._SubforwardCost],
) -> _impl._MemoryCheck:
    required = self._split_required_memory(costs)
    key = self._split_memory_key(plan)
    empirical = (
        0
        if key is None
        else int(self._split_memory_floors.get(key, 0) * _impl._MEMORY_SAFETY_FACTOR)
    )
    # Same existing reduction order and count; never reduce while returning
    # from caller work, where a failed/empty peer may not participate.
    return self._memory_check_required(max(required, empirical))


def _head_vocabulary(self: TrainerRank) -> int:
    """Runtime eligibility and this rank's vocabulary, without reading values.

    TP > 1 runs the same 512-row chunked head over each rank's vocabulary
    shard; it is priced only where the explicit TP x SP checkpoint floor
    applies, whose shallow shapes it can dominate.
    """
    _, tp, _, pp = self._topology_key()
    if (
        self._padded_vocab_size is None
        or len(self.runtime.model) != 1
        or pp != 1
        or self._padded_vocab_size % tp
        or (tp > 1 and not _checkpoint_layers(self, ((1, True),)))
    ):
        return 0
    vocabulary = self._padded_vocab_size // tp
    try:
        model = _impl._language_model(self.runtime.model[0])
    except (AttributeError, RuntimeError):
        return 0
    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        return 0

    head = getattr(model, "output_layer", None)
    if head is None or type(head) is not ColumnParallelLinear:
        return 0
    weight = head.weight
    if (
        weight is None
        and getattr(model, "share_embeddings_and_output_weights", False) is True
    ):
        weight = getattr(
            getattr(getattr(model, "embedding", None), "word_embeddings", None),
            "weight",
            None,
        )
    if weight is None:
        return 0
    config = getattr(model, "config", None)
    if (
        type(weight) not in (_impl.torch.Tensor, _impl.torch.nn.Parameter)
        or weight.dtype is not _impl.torch.bfloat16
        or tuple(weight.shape) != (vocabulary, self._hidden_size)
        or head.output_size_per_partition != vocabulary
        or head.output_size != self._padded_vocab_size
        or head.input_size != self._hidden_size
        or getattr(config, "params_dtype", None) is not _impl.torch.bfloat16
        or getattr(config, "fp32_residual_connection", None) is not False
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or "forward" in vars(head)
        or "_forward_impl" in vars(head)
        or getattr(head, "_forward_hooks", None)
        or getattr(head, "_forward_pre_hooks", None)
        or any(
            name in vars(self)
            for name in (
                "_project_head",
                "_project_vocab_parallel",
                "_checkpointed_head_stats",
                "_local_head_stats",
                "_local_head_stats_parts",
                "_local_logits_from_hidden_rows",
            )
        )
    ):
        return 0
    return int(vocabulary)


def _dense_head_bytes(vocabulary: int, rows: int) -> int:
    return min(rows, _impl._HEAD_CHUNK_TOKENS) * vocabulary * 2


def _head_workspace_bytes(self: TrainerRank, rows: int) -> int:
    """One dense BF16 head tensor, not complete statistics/backward memory."""
    return _dense_head_bytes(_head_vocabulary(self), rows) if rows > 0 else 0


def _head_target_backward(self: TrainerRank) -> bool:
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    model = _impl._language_model(self.runtime.model[0])
    scale = getattr(model, "_scale_logits", None)
    return (
        type(scale) is MethodType
        and scale.__self__ is model
        and scale.__func__ is LanguageModule._scale_logits
        and getattr(model.config, "use_mup", None) is False
    )


def _group_head_workspace_bytes(
    self: TrainerRank,
    rows: int,
    requests: Sequence[AnyForwardInput],
    *,
    grad_enabled: bool,
    positions: Sequence[torch.Tensor] | None = None,
    lower_bound: bool = False,
) -> int:
    """Partial dense head component: eager statistics or logits copies.

    Capacity reserves the eager path even when optional Triton may succeed.
    Its BF16 logits, FP32 conversion, subtraction and exp overlap. This is
    not a bound for row vectors, inter-chunk liveness or library workspaces.
    Rejection lower bounds retain only the unconditional dense components.
    TP > 1 heads (priced with the explicit TP x SP floor) instead follow each
    chunk's statistics path: ``_tp_head_workspace_bytes``. Statistics capacity
    adds their index vectors (``_head_target_bytes``) on both.
    """
    dense = self._head_workspace_bytes(rows)
    if dense and self._topology_key()[1] > 1:
        return _tp_head_workspace_bytes(
            self,
            rows,
            requests,
            grad_enabled=grad_enabled,
            positions=positions,
            lower_bound=lower_bound,
        )
    needs_statistics = any(
        request.target_tokens is not None or request.top_k is not None
        for request in requests
    )
    if not dense or (
        not needs_statistics and (lower_bound or not any(r.logits for r in requests))
    ):
        return dense
    if _head_target_backward(self):
        if not lower_bound:
            # need_log_z is group-wide, including logits-only chunks and
            # chunks overlapping ignored labels. A short final chunk can
            # also take the eager path; optional success is not guaranteed.
            # Without statistics, local logits and both indexed copies
            # overlap. Requested output storage is charged separately.
            if not needs_statistics:
                return 3 * dense
            return 7 * dense + _head_target_bytes(
                self._head_projection_rows(
                    requests, positions=positions, uncapped=True
                ),
                requests,
                grad_enabled=grad_enabled,
            )
        if not grad_enabled or not any(
            request.target_tokens is not None for request in requests
        ):
            return dense
        # Every statistics path's backward holds the labelled chunk's logits
        # and their gradient.
        target_dense = (
            self._head_workspace_bytes(
                self._head_target_chunk_rows(
                    requests, positions=positions, lower_bound=lower_bound
                )
            )
            if any(request.logits or request.top_k is not None for request in requests)
            else dense
        )
        return max(dense, 2 * target_dense)
    return dense


def _tp_head_workspace_bytes(
    self: TrainerRank,
    rows: int,
    requests: Sequence[AnyForwardInput],
    *,
    grad_enabled: bool,
    positions: Sequence[torch.Tensor] | None = None,
    lower_bound: bool = False,
) -> int:
    """The TP > 1 head component, per chunk statistics path (#1068's model at TP1).

    The eager FP32 statistics keep #1068's seven BF16 buffers. Chunks whose
    kernel is attempted run it, or after a failure the bounded eager
    statistics (``_impl._EagerLocalStats``): the kernel path's buffers plus
    ``_eager_stats_extra_bytes``. A gradient wave's kernel path holds the
    saved logits and the statistics gradient: target and top-k gradients
    join it inside the statistics backward. Requested logits hold the local
    logits, their indexed copy, the gather buffer and its concatenation
    (Megatron's gather along the vocabulary): 2 + 2 TP. Statistics capacity
    adds their index vectors: ``_head_target_bytes``.
    """
    dense = self._head_workspace_bytes(rows)
    if not _head_target_backward(self):
        return dense
    tp = self._topology_key()[1]
    needs_statistics = any(
        request.target_tokens is not None or request.top_k is not None
        for request in requests
    )
    if not needs_statistics:
        logits = any(request.logits for request in requests)
        return dense if lower_bound or not logits else (2 + 2 * tp) * dense
    kernel = 2 * dense if grad_enabled else dense
    low = self._head_projection_rows(
        requests, positions=positions, lower_bound=True, uncapped=True
    )
    if lower_bound:
        # A rejection bound monotone in rows: only unconditional components,
        # and the eager statistics only when no kernel can run at all.
        return max(
            kernel,
            0
            if _impl._triton_head_stats_available(self)
            else 7 * self._head_workspace_bytes(low),
        )
    if positions is not None:
        # The selected layout's packed union: every chunk exactly.
        fallback = _head_fallback_bytes(self, low, low)
    else:
        # An acceptance bound: any projected count the union can take.
        high = self._head_projection_rows(requests, positions=positions, uncapped=True)
        fallback = _head_fallback_bytes(self, low, high)
    logits = (2 + 2 * tp) * dense if any(request.logits for request in requests) else 0
    return max(
        kernel + _eager_stats_extra_bytes(_head_vocabulary(self), rows),
        fallback,
        logits,
    ) + _head_target_bytes(
        self._head_projection_rows(requests, positions=positions, uncapped=True),
        requests,
        grad_enabled=grad_enabled,
    )


# The statistics' index vectors beside the head buffers, measured on CPU and
# H200 at up to 1,000 targets per row. Per target (a label or top-k entry) in
# the running chunk: its row and column, their grouping by row and FP32
# gradient (32 B in the kernel paths' backward; the TP1 fallback's forward
# holds 24 B beside the top-k selection).
_HEAD_CHUNK_TARGET_BYTES = 36
# Per label of the wave: the checkpointed (row, label) pairs, each chunk's
# log-probs and their gradient (28 B).
_HEAD_LABEL_BYTES = 32
# Per top-k entry of a gradient wave: the requested log-probs' gradient.
_HEAD_TOP_K_GRADIENT_BYTES = 4
# Per row of the running chunk: maxima, sums, log-normalizers, positions.
# Not the Triton backward's (row, vocab block) target offsets, 16 B per
# 4,096-entry block (about 500 B per row of a 124,160-entry shard): TP > 1
# covers them with the bounded statistics' increment, charged even when the
# kernel runs, and TP1 with the fallback's seven buffers. Trimming either
# needs this term to price them.
_HEAD_CHUNK_ROW_BYTES = 128


def _head_target_bytes(
    rows: int, requests: Sequence[AnyForwardInput], *, grad_enabled: bool
) -> int:
    """Statistics storage for all ``rows`` projected head rows: indices for
    the largest chunk, and labels, top-k gradients and FP32 normalization
    vectors retained across chunks. Requested outputs are charged separately."""
    chunk = min(rows, _impl._HEAD_CHUNK_TOKENS)
    labels = width = top_k = 0
    for request in requests:
        tokens = int(request.input_tokens.numel())
        if request.target_tokens is not None and tokens:
            labels += int(request.target_tokens.numel())
            width += -(-int(request.target_tokens.numel()) // tokens)
        if request.top_k is not None:
            top_k += tokens * int(request.top_k)
            width += int(request.top_k)
    return (
        _HEAD_CHUNK_ROW_BYTES * chunk
        + _HEAD_CHUNK_TARGET_BYTES * min(chunk * width, labels + top_k)
        + _HEAD_LABEL_BYTES * labels
        + (_HEAD_TOP_K_GRADIENT_BYTES * top_k if grad_enabled else 0)
        # Row normalization retains the FP32 global sum and, on the optional
        # kernel/bounded paths, its FP32 scale outside the dense checkpoint.
        + (8 * rows if grad_enabled else 0)
    )


def _eager_stats_extra_bytes(vocabulary: int, rows: int) -> int:
    """The bounded eager statistics' live increment over the kernel path for a
    ``rows`` x ``vocabulary`` head chunk (``_impl._EagerLocalStats``): one FP32
    row sub-chunk, plus per-row FP32 maxima, sums and their gradient."""
    rows = min(rows, _impl._HEAD_CHUNK_TOKENS)
    step = -(-rows // _impl._EAGER_STATS_SUBCHUNKS)
    return 4 * vocabulary * step + 16 * rows


def _frozen_head_bytes(
    vocabulary: int,
    rows: int,
    *,
    target_backward: bool,
    statistics: bool,
    grad: bool,
    tp: int,
    targets: int,
) -> int:
    """Replay's head charge from frozen facts, as live admission prices it.

    TP1 is #1068's capacity charge. TP > 1 is ``_tp_head_workspace_bytes``'s
    kernel path plus the bounded statistics' increment, or the gathered logits
    copies. Capture declines TP > 1 waves with an eager chunk or with logits
    beside statistics: the facts record neither. ``targets`` is
    ``_head_target_bytes`` of the verified requests.
    """
    dense = _dense_head_bytes(vocabulary, rows)
    if not dense or not target_backward:
        return dense
    if not statistics:
        return (3 if tp == 1 else 2 + 2 * tp) * dense
    if tp == 1:
        return 7 * dense + targets
    kernel = 2 * dense if grad else dense
    return kernel + _eager_stats_extra_bytes(vocabulary, rows) + targets


def _head_fallback_bytes(self: TrainerRank, low: int, high: int) -> int:
    """Seven BF16 buffers of the largest head chunk that runs the eager fallback
    for any projected-row count from ``low`` to ``high``."""
    chunk = _impl._HEAD_CHUNK_TOKENS
    vocabulary = _head_vocabulary(self)
    sizes: set[int] = set()
    if high - low >= chunk:
        sizes.update(range(1, min(high, chunk) + 1))  # every tail occurs
    else:
        for rows in range(low, high + 1):
            sizes.update(_head_chunk_sizes(rows))
    eager = [
        size for size in sizes if not _impl._triton_head_stats(self, size, vocabulary)
    ]
    return 7 * self._head_workspace_bytes(max(eager)) if eager else 0


def _head_chunk_sizes(rows: int) -> tuple[int, ...]:
    """The distinct chunk sizes the head runs over ``rows`` projected rows."""
    chunk = _impl._HEAD_CHUNK_TOKENS
    return tuple(
        size for size in (min(rows, chunk), rows % chunk * (rows > chunk)) if size
    )


def _plan_head_workspace_bytes(self: TrainerRank, plan: _impl._FlatForwardPlan) -> int:
    peak = 0
    for group in plan.groups:
        requests = tuple(item.request for item in group.items)
        peak = max(
            peak,
            self._group_head_workspace_bytes(
                self._head_projection_rows(
                    requests, positions=group.packed.positions_by_sequence
                ),
                requests,
                grad_enabled=group.grad_enabled,
                positions=group.packed.positions_by_sequence,
            ),
        )
    return peak


def _plan_hybridep_growth_bytes(self: TrainerRank, plan: _impl._FlatForwardPlan) -> int:
    """HybridEP buffer growth this plan triggers before its forward.

    The buffer is allocated outside the PyTorch allocator after admission,
    so neither the free-memory sample nor a learned peak includes it.
    """
    return _hybridep_growth_from_dimensions(_plan_hybridep_dimensions(self, plan))


def _plan_hybridep_dimensions(
    self: TrainerRank, plan: _impl._FlatForwardPlan
) -> tuple[int, int, int, int, int] | None:
    """Capacity, communication ranks, hidden width, expert columns, held rows."""
    provider: Any = getattr(getattr(self, "runtime", None), "provider", None)
    ep = int(getattr(provider, "expert_model_parallel_size", 1) or 1)
    if ep <= 1 or not plan.groups:
        return None
    from megatron.core.transformer.moe import fused_a2a

    from art.megatron.train import _hybridep_token_capacity

    topology = self._topology()
    sequence = max(
        int(
            _impl._pad_packed_batch(
                group.packed, multiple=int(topology.tp)
            ).tokens.shape[1]
        )
        for group in plan.groups
    )
    rows = max(rows for rows, _ in self._plan_group_rows(plan))
    capacity = max(_hybridep_token_capacity(sequence, int(topology.cp)), rows)
    current = fused_a2a._hybrid_ep_buffer
    held = (
        0
        if current is None
        else int(current.configurer.buffer_config.max_num_of_tokens_per_rank)
    )
    etp = int(getattr(provider, "expert_tensor_parallel_size", 1) or 1)
    ranks = ep * etp
    return (
        capacity,
        ranks,
        int(provider.hidden_size),
        int(provider.num_moe_experts) * etp,
        held,
    )


def _hybridep_growth_from_dimensions(
    dimensions: tuple[int, int, int, int, int] | None,
) -> int:
    if dimensions is None:
        return 0
    capacity, ranks, hidden, experts, held = dimensions
    if _impl._hybridep_rows_per_rank(capacity, ranks) <= held:
        return 0
    # The old buffer stays referenced while its replacement is allocated.
    return _impl._hybridep_buffer_bytes(capacity, ranks, hidden, experts)


def _checkpoint_moe_bytes_per_token(self: TrainerRank) -> int:
    forward = self._moe_output_bytes_per_token
    gradient = self._moe_checkpoint_grad_bytes_per_token
    if (
        type(forward) is not int
        or forward < 0
        or type(gradient) is not int
        or gradient < forward
    ):
        raise ValueError("Invalid constructor checkpoint MoE coefficient")
    return gradient


def _moe_workspace_terms(
    self: TrainerRank,
    *,
    checkpoint_grad: bool = False,
    slot_ref: "LoRASlotRef | None" = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Maximum of same-layer affine stages, not a retained multi-layer bank.

    The constructor cache covers original tensors. Explicit slots are
    repriced from their tensor metadata and original owners, including
    this rank's exact dispatcher wrapper. Ordinary non-checkpoint gradients
    retain only forward-stage coverage.
    """
    coefficient = (
        self._checkpoint_moe_bytes_per_token()
        if checkpoint_grad
        else self._moe_output_bytes_per_token
    )
    stages = getattr(
        self,
        "_moe_gradient_stages" if checkpoint_grad else "_moe_forward_stages",
        (),
    )
    if slot_ref is not None and slot_ref.name is not None:
        selected: list[tuple[int, int]] = []
        coefficient = (
            _impl._moe_output_bytes_per_token(
                self.runtime.model,
                self._parallel_shape,
                checkpoint_grad=checkpoint_grad,
                converted_stages=selected,
                slot_ref=slot_ref,
            )
            if self._moe_layers
            else 0
        )
        stages = tuple(selected) if coefficient else ()
    if type(stages) is not tuple or any(
        type(stage) is not tuple
        or len(stage) != 2
        or any(type(value) is not int or value < 0 for value in stage)
        for stage in stages
    ):
        raise ValueError("Invalid constructor converted-weight stages")
    return coefficient, stages


def _moe_workspace_from_terms(
    rows: int, terms: tuple[int, tuple[tuple[int, int], ...]]
) -> int:
    coefficient, stages = terms
    return (
        max(
            rows * coefficient,
            *(rows * per_row + fixed for per_row, fixed in stages),
        )
        if stages and rows > 0
        else rows * coefficient
    )


def _moe_workspace_bytes(
    self: TrainerRank,
    rows: int,
    *,
    checkpoint_grad: bool = False,
    slot_ref: "LoRASlotRef | None" = None,
) -> int:
    return _moe_workspace_from_terms(
        rows,
        _moe_workspace_terms(self, checkpoint_grad=checkpoint_grad, slot_ref=slot_ref),
    )


def _checkpoint_layers(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
) -> int:
    """Conservative saved-boundary charge and one disjoint MoE workspace.

    Count actual local full/uniform/1 boundaries, including aliases, rather
    than claiming measured distinct storage. Only this call's new groups
    enter the term; already-live graphs remain in the availability baseline.
    No-grad groups also keep decoder input, current layer input, its MLP
    residual and norm output across the MoE stage. Count these four row
    tensors separately from returned outputs, allowing storage aliases.
    This is not a bound for custom preprocessing, attention, or all backward.
    With sequence parallelism a rank saves only its shard of each boundary;
    that is priced only where ``_sequence_parallel_floor_covered`` holds,
    and there, for gradient waves, with the recomputed layer's explicit
    workspace (``_sequence_parallel_workspace_bytes``) and its GDN layer's
    recurrent states for ``gdn_segments`` (gradient groups' segments) plus
    padding, beside one input-gradient shard instead of the TP1 repeat.
    """
    if not group_rows or len(self.runtime.model) != 1:
        return 0
    try:
        decoder = _impl._language_model(self.runtime.model[0]).decoder
    except (AttributeError, RuntimeError):
        return 0
    try:
        from megatron.core.transformer.transformer_block import TransformerBlock
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        return 0

    if type(decoder) is not TransformerBlock:
        return 0
    config = decoder.config
    layers = len(decoder.layers)
    _, tp, cp, pp = self._topology_key()
    expected = {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "distribute_saved_activations": False,
        "sequence_parallel": tp > 1,
        "fp32_residual_connection": False,
        "cpu_offloading": False,
        "cuda_graph_impl": "none",
    }
    if (
        decoder.training is not True
        or layers <= 0
        or layers != decoder.num_layers_per_pipeline_rank
        or layers != config.num_layers
        or config.hidden_size != self._hidden_size
        or config.params_dtype is not _impl.torch.bfloat16
        or self._param_dtype_size != 2
        or next(self.runtime.model[0].parameters()).dtype is not _impl.torch.bfloat16
        or pp != 1
        or (tp > 1 and not self._sequence_parallel_floor_covered(tp, cp))
        or any(
            type(getattr(config, name, None)) is not type(value)
            or getattr(config, name) != value
            for name, value in expected.items()
        )
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or any(
            name in vars(decoder)
            for name in ("forward", "_checkpointed_forward", "_get_layer")
        )
        or getattr(decoder, "_forward_hooks", None)
        or getattr(decoder, "_forward_pre_hooks", None)
    ):
        return 0
    return layers


def _checkpoint_memory_floor(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
    slot_refs: tuple["LoRASlotRef | None", ...] | None = None,
    gdn_segments: int = 0,
) -> tuple[int, int]:
    layers = _checkpoint_layers(self, group_rows)
    if not layers:
        return 0, 0
    return _checkpoint_floor_from_facts(
        self,
        group_rows,
        slot_refs,
        gdn_segments,
        layers,
        hybridep_rows=_checkpoint_hybridep_rows(self, group_rows),
    )


def _checkpoint_hybridep_rows(
    self: TrainerRank, group_rows: tuple[tuple[int, bool], ...]
) -> int | None:
    """Live graph extent needed by the supported HybridEP recompute path."""
    hybrid_rows = None
    if (
        any(grad for _, grad in group_rows)
        and self._topology_key() == (1, 1, 2, 1)
        and self._parallel_shape == _impl.ParallelShape(tp=1, cp=2, ep=2, etp=1)
        and self._moe_memory_supported
    ):
        hybrid_rows = max(rows for rows, _ in group_rows)
        if any(ref() is not None for ref in self._pending_hybridep_graphs):
            hybrid_rows = max(hybrid_rows, self._hybridep_rows_high_water)
    return hybrid_rows


def _native_dense_cp(self: TrainerRank) -> bool:
    cp = self._topology_key()[2]
    return (
        cp in (2, 4, 8)
        and self._parallel_shape.cp == cp
        and not self._moe_layers
        and self._dense_fc1_adapted
    )


def _checkpoint_floor_from_facts(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
    slot_refs: tuple["LoRASlotRef | None", ...] | None,
    gdn_segments: int,
    layers: int,
    *,
    hybridep_rows: int | None = None,
) -> tuple[int, int]:
    if not layers:
        return 0, 0
    gradient_rows = sum(rows for rows, grad in group_rows if grad)
    tp = self._topology_key()[1]
    dense_cp = _native_dense_cp(self)
    # Physical rows are padded to a multiple of TP; each rank saves its shard.
    retained = (
        sum(-(-rows // tp) for rows, grad in group_rows if grad)
        * layers
        * self._hidden_size
        * 2
    )
    if gradient_rows:
        self._checkpoint_moe_bytes_per_token()
    refs = (None,) * len(group_rows) if slot_refs is None else slot_refs
    workspace = max(
        self._moe_workspace_bytes(rows, checkpoint_grad=grad, slot_ref=ref)
        + (0 if grad else 4 * rows * self._hidden_size * 2)
        for (rows, grad), ref in zip(group_rows, refs, strict=True)
    )
    if (tp > 1 or dense_cp) and gradient_rows:
        # The recomputed layer's workspace, for the largest gradient group:
        # groups run their backward one after another.
        workspace = max(
            workspace,
            *(
                self._sequence_parallel_workspace_bytes(rows)
                for rows, grad in group_rows
                if grad
            ),
        )
    if (tp > 1 or dense_cp) and self._gdn_layers and gradient_rows:
        # Recurrent states grow with segments, not rows; backward recomputes
        # one layer at a time. Padding to TP adds up to TP - 1 one-token
        # roots per group. Kernel-internal chunk states are not bounded here.
        roots = gdn_segments + (tp - 1) * sum(grad for _, grad in group_rows)
        workspace += math.ceil(roots * self._gdn_segment_layer_bytes())
    if hybridep_rows is not None:
        workspace = max(workspace, -(-hybridep_rows // 4) * 4 * self._hidden_size * 2)
    from art.megatron.context_parallel.checkpoint import current_checkpoint_partition

    partition = current_checkpoint_partition()
    if partition is not None and gradient_rows:
        degree, shape = partition
        if group_rows != ((shape[0], True),):
            raise ValueError("checkpoint shards require one complete gradient root")
        width = math.ceil(math.prod(shape) / degree)
        retained = width * self._param_dtype_size * layers
        # Recompute still runs the complete CP1 layer, beside the all-gather
        # buffer. Only the saved boundaries shrink; workspace does not.
        workspace = max(workspace, self._sequence_parallel_workspace_bytes(shape[0]))
        workspace += width * degree * self._param_dtype_size
    return retained, workspace


def _sequence_parallel_workspace_bytes(self: TrainerRank, rows: int) -> int:
    """The recomputed layer's live workspace at a TP x SP backward peak.

    Traced on dense Qwen3.8-27B at TP4 (25,728 rows) and TP2 (20,816 to
    66,284 rows, one to four GDN segments), cold and warm: every peak was a
    recomputed layer's MLP FC1 stage. Its workspace covers, over the gathered
    rows, the SP-gathered norm output (2H), the FC1 stage (6F/TP: GEMM
    output, gate and up LoRA outputs and their sum; the SwiGLU live set if
    wider) and the recomputed mixer at its projection widths / TP (GDN held
    0.885-0.887 of that bound); over the sharded rows, norm outputs (3.22H
    measured, priced at 3.25H); and the wave length's fp32 rotary cache
    (``kv_channels`` per row bounds the traced 64-dim partial rotary).

    Recomputing layer i holds the L-shard ``retained`` term's i + 1
    checkpoint inputs plus its own residual: L + 1 - gap shards, where gap
    counts the layers above it. Each mixer kind peaks at its highest layer;
    the boundary shards beyond L are charged here, per kind, from the
    decoder's layer order (one extra shard for a kind whose order is
    unknown). In the traced model the top layer is attention and the top GDN
    layer (one below) peaks: exactly L shards. The input-gradient shard, LoRA
    intermediates, GDN segment states and cold transients are separate terms.
    """
    if rows <= 0:
        return 0
    geometry = self._geometry
    tp = self._topology_key()[1]
    hidden = self._hidden_size
    ffn = geometry.ffn_hidden_size or 4 * hidden
    attention_layers = self._num_layers > self._gdn_layers
    attention = (
        (7 if self._attention_output_gate else 5)
        * geometry.num_attention_heads
        * geometry.kv_channels
        + 3 * geometry.num_query_groups * geometry.kv_channels
    )
    gdn = (
        4 * geometry.gdn_key_heads * geometry.gdn_key_head_dim
        + 8 * geometry.gdn_value_heads * geometry.gdn_value_head_dim
    )
    shard = -(-rows // tp) * hidden
    gaps = getattr(self, "_mixer_top_gaps", None) or (None, None)
    peak = (
        max(
            rows * -(-width // tp) + (1 - (0 if gap is None else gap)) * shard
            for present, width, gap in (
                (attention_layers, attention, gaps[0]),
                (bool(self._gdn_layers), gdn, gaps[1]),
            )
            if present
        )
        if self._num_layers
        else 0
    )
    stage = max(6, self._mlp_activation_factor) * ffn
    gathered = rows * (2 * hidden + -(-stage // tp))
    sharded = -(-rows // tp) * -(-13 * hidden // 4)
    rotary = 2 * rows * geometry.kv_channels if attention_layers else 0
    return (gathered + sharded + peak + rotary) * self._param_dtype_size


def _sequence_parallel_lora_floor(
    self: TrainerRank,
    floor: tuple[int, int],
    group_rows: tuple[tuple[int, bool], ...],
    signature: _impl._MemorySignature,
) -> tuple[int, int]:
    """Add the recomputed layer's LoRA row-by-rank intermediates.

    Each adapted module of the recomputed layer saves rows x rank over the
    gathered rows (FC1 gate/up measured 16 B/row at rank 4). Charge the
    decoder's most-adapted layer at the gradient slots' largest rank, from
    the signature's slot shapes ((ndim, *A.shape, *B.shape) per module).
    Native dense CP also reserves that many bytes for backward factor-product
    gradients beside the saved forward products. Counting every target at the
    largest selected rank bounds layers with different target/rank inventories.
    """
    retained, workspace = floor
    dense_cp = _native_dense_cp(self)
    if not retained or (self._topology_key()[1] == 1 and not dense_cp):
        return floor
    rank = max(
        (
            shape[2]
            for grad, shapes in signature.slot_shapes
            if grad
            for shape in shapes
            if len(shape) == 5 and shape[0] == 2 and shape[2] == shape[3]
        ),
        default=0,
    )
    modules = getattr(self, "_lora_modules_per_layer", None)
    if not modules:
        # Unread (e.g. reports without the recorded count): every adapted
        # module of the slot, which bounds any one layer's.
        modules = max(
            (len(shapes) for grad, shapes in signature.slot_shapes if grad), default=0
        )
    rows = max((rows for rows, grad in group_rows if grad), default=0)
    copies = 2 if dense_cp else 1
    lora_bytes = copies * rows * modules * rank * self._param_dtype_size
    if dense_cp and (
        not rank
        or self._geometry.moe_experts
        or self._geometry.moe_ffn_hidden_size
        or not all(
            len(shapes) >= modules
            and all(
                not shape  # A recognized owner with an inactive selected slot.
                or (
                    len(shape) == 5
                    and shape[0] == 2
                    and min(shape) > 0
                    and shape[2] == shape[3]
                )
                for shape in shapes
            )
            for grad, shapes in signature.slot_shapes
            if grad
        )
    ):
        # Dense dX refinement requires a complete native 2-D inventory and
        # supported geometry. (0,) records an unknown owner/implementation;
        # missing/partial metadata is unknown even with zero recognized owners.
        # Restore main's per-layer allowance beside the explicit workspace and
        # any known rank charge, rather than letting unknown mean rank zero.
        workspace += retained
    return retained, workspace + lora_bytes


def _cold_recompute_transient_bytes(self: TrainerRank) -> int:
    """Fixed transients of an unprofiled gradient wave's first execution."""
    return (
        _impl._SEQUENCE_PARALLEL_COLD_TRANSIENT_BYTES
        if self._topology_key()[1] > 1
        else _impl._COLD_RECOMPUTE_TRANSIENT_BYTES
    )


def _checkpoint_input_gradient_bytes(
    self: TrainerRank, group_rows: tuple[tuple[int, bool], ...], retained: int
) -> int:
    """The input-gradient term beside the checkpoint floor's boundaries.

    TP1's legacy boundary-gradient charge also stands in for workspace.
    Native dense CP and TP x SP price that workspace explicitly, beside one
    input-gradient shard of the group being recomputed.
    """
    tp = self._topology_key()[1]
    dense_cp = _native_dense_cp(self)
    from art.megatron.context_parallel.checkpoint import current_checkpoint_partition

    partition = current_checkpoint_partition()
    if partition is not None and retained:
        # One full recomputed root's dX; its layer workspace is explicit.
        return math.prod(partition[1]) * self._param_dtype_size
    if not retained or (tp == 1 and not dense_cp):
        return retained
    rows = max((rows for rows, grad in group_rows if grad), default=0)
    return -(-rows // tp) * self._hidden_size * self._param_dtype_size


def _gradient_slots(
    group_rows: Sequence[tuple[int, bool]],
    slot_refs: Sequence[LoRASlotRef | None] | None,
) -> frozenset[LoRASlotRef]:
    """Gradient groups' adapter slots; the base model (no name) has none."""
    return frozenset(
        ref
        for (_, grad), ref in zip(
            group_rows, slot_refs or (None,) * len(group_rows), strict=True
        )
        if grad and ref is not None and ref.name is not None
    )


def _pending_adapter_gradient_bytes(
    self: TrainerRank, refs: Iterable[LoRASlotRef]
) -> tuple[int, ...]:
    """Local adapter gradient bytes the next backward allocates, per decoder layer.

    One entry per decoder layer, then one for parameters outside the
    decoder. Backward reaches a parameter's highest decoder layer first, so
    a parameter shared across layers counts there once; those outside the
    decoder count as live throughout. Only unallocated gradients count:
    within a step, later waves find the rest in the availability baseline.
    Empty when none is pending.
    """
    refs = tuple(dict.fromkeys(refs))
    if not refs or len(self.runtime.model) != 1:
        return ()
    try:
        from art.megatron.lora import LoRA
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        return ()
    chunk = self.runtime.model[0]
    try:
        layers = _impl._language_model(chunk).decoder.layers
    except (AttributeError, RuntimeError):
        return ()
    layer_of: dict[int, int] = {}
    params: dict[int, torch.nn.Parameter] = {}

    def slot_params(
        modules: Iterable[torch.nn.Module],
    ) -> Iterator[torch.nn.Parameter]:
        for module in modules:
            # A LoRA without slot tables holds no slot parameters.
            if isinstance(module, LoRA) and "_slot_keys" in vars(module):
                for ref in refs:
                    yield from module.lora_slot_params(ref)

    for index, layer in enumerate(layers):
        for param in slot_params(layer.modules()):
            params[id(param)] = param
            layer_of[id(param)] = max(layer_of.get(id(param), -1), index)
    # Outside the decoder (a head runs its backward first) is live
    # throughout, even for a parameter or module a decoder layer also uses:
    # walk every path except through the layers themselves.
    outside: list[torch.nn.Module] = []
    visited: set[int] = set()
    pending_modules: list[torch.nn.Module] = [chunk]
    while pending_modules:
        module = pending_modules.pop()
        if module is layers or id(module) in visited:
            continue
        visited.add(id(module))
        outside.append(module)
        pending_modules.extend(module.children())
    for param in slot_params(outside):
        params[id(param)] = param
        layer_of[id(param)] = len(layers)
    # A checkpoint's other trainable parameters (custom objects) have no
    # decoder position; count them as live throughout.
    for ref in refs:
        slot = (
            None
            if ref.name is None
            else getattr(self, "_checkpoint_slots", {}).get(ref.name)
        )
        for param in () if slot is None else slot.params:
            if id(param) not in params:
                params[id(param)] = param
                layer_of[id(param)] = len(layers)
    sizes = [0] * (len(layers) + 1)
    for param_id, param in params.items():
        if (
            param.requires_grad
            and param.grad is None
            and getattr(param, "main_grad", None) is None
        ):
            sizes[layer_of[param_id]] += param.numel() * param.element_size()
    return tuple(sizes) if any(sizes) else ()


def _checkpoint_gradient_groups(
    self: TrainerRank,
    group_rows: Sequence[tuple[int, bool]],
    slot_refs: Sequence[LoRASlotRef | None] | None,
) -> tuple[tuple[LoRASlotRef | None, tuple[int, ...]], ...]:
    """Each gradient group's adapter slot and per-layer saved boundaries.

    In execution order, as ``_checkpoint_memory_floor`` prices them: every
    decoder layer saves the group's rows (this rank's TP shard). The base
    model (no name) has no adapter slot.
    """
    tp = self._topology_key()[1]
    return tuple(
        (
            ref if ref is not None and ref.name is not None else None,
            (-(-rows // tp) * self._hidden_size * 2,) * self._num_layers,
        )
        for (rows, grad), ref in zip(
            group_rows, slot_refs or (None,) * len(group_rows), strict=True
        )
        if grad
    )


def _checkpoint_adapter_gradient_bytes(
    self: TrainerRank, groups: Sequence[tuple[LoRASlotRef | None, Sequence[int]]]
) -> int:
    """The recompute backward's adapter-gradient peak beyond released boundaries.

    ``groups`` gives each gradient group's adapter slot (None for the base
    model) and each decoder layer's saved-boundary bytes
    (``_adapter_gradient_walk``).
    """
    chains = []
    for slot, boundaries in groups:
        pending = () if slot is None else self._pending_adapter_gradient_bytes((slot,))
        if pending and len(pending) != len(boundaries) + 1:
            return 0
        chains.append((pending or (0,) * (len(boundaries) + 1), boundaries))
    return self._adapter_gradient_walk(chains)


def _adapter_gradient_walk(
    chains: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> int:
    """The adapter-gradient peak beyond the floor over gradient groups' backward.

    Each chain is a gradient group's pending gradient bytes (per decoder
    layer, then outside the decoder) and saved-boundary bytes per layer.
    Backward recomputes the last layer first. While it recomputes layer i it
    still holds the saved boundaries of layers 0..i and every adapter
    gradient allocated so far: those of layers i..L-1 (a layer allocates its
    own during its backward) and any outside the decoder. The floor already
    prices all L boundaries at once, so one group's extra peak is
    max(0, max over i of gradients(i..) - boundaries(i+1..)), taken at every
    layer over the real per-layer gradient sizes and the caller's per-layer
    boundaries, not along a uniform-layer line. A short
    first wave peaks at layer 0 (Qwen3.6-35B-A3B CP2: 830-900 MB of expert
    LoRA gradients live at its peak), a long one at the last layer.
    Groups run their backward one after another, not layer by layer
    together: autograd drains the last-forwarded group's chain first, and
    separate backward calls can come in either order. While one group runs,
    each group already run holds all its gradients and none of its
    boundaries, and each group yet to run all its boundaries. Any set of the
    other groups can have run first, so the worst adds every other group
    whose gradients outweigh its boundaries.
    """
    nets = [sum(pending) - sum(boundaries) for pending, boundaries in chains]
    others = sum(max(0, net) for net in nets)
    worst = 0
    for (pending, boundaries), net in zip(chains, nets, strict=True):
        extra = gradients = pending[-1]
        released = 0
        for index in range(len(boundaries) - 1, -1, -1):
            gradients += pending[index]
            extra = max(extra, gradients - released)
            released += boundaries[index]
        worst = max(worst, extra + others - max(0, net))
    return worst


def _retained_memory_bytes(
    self: TrainerRank,
    signature: _impl._MemorySignature,
    *,
    packed_tokens: int,
    logical_tokens: int,
    output_bytes: int,
    required: int,
    checkpoint_retained_bytes: int = 0,
) -> int:
    """Forward-retained bytes, independent of a later backward peak.

    Retain the full estimate until observed near the current scale and
    sharing ratio, so a small forward cannot authorize a much larger split.
    """

    profile = self._memory_profiles.get(signature)
    if profile is None or profile.retained_compute_bytes_per_token is None:
        return required
    if (
        packed_tokens
        > _retained_rate_extent(profile) * _impl._MEMORY_PROFILE_TRUST_GROWTH
    ):
        return required
    ratio = logical_tokens / max(1, packed_tokens)
    if ratio > profile.logical_per_packed * _impl._MEMORY_PROFILE_TRUST_GROWTH:
        return required
    rate = profile.retained_compute_bytes_per_token
    if _impl._packed_priced(signature, self._one_layer_recompute()):
        # Saved head indices, masks and caller saves stay live until
        # backward. The ratio window above bounds the sharing extrapolation.
        retained_compute = (
            rate * packed_tokens
            + _impl._PACKED_PRICED_LOGICAL_ROW_BYTES * logical_tokens
        )
    else:
        retained_compute = rate * max(
            packed_tokens, logical_tokens / profile.logical_per_packed
        )
    retained_compute = max(
        retained_compute,
        _retained_floor_peak(profile.retained_compute_peaks, packed_tokens),
    )
    retained = output_bytes + max(checkpoint_retained_bytes, retained_compute)
    return min(required, int(retained * _impl._MEMORY_SAFETY_FACTOR))


def _estimate_flat_forward(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    checkpoint: AdapterSelection = _impl.Unset,
    exact: bool = False,
    memory_minimal: bool = False,
    sync_planning_errors: bool = False,
    gdn_segments: list[int] | None = None,
) -> tuple[int, int, _impl._MemorySignature, tuple[tuple[int, bool], ...], int] | None:
    """Estimate packed tokens for width probing.

    Cheap mode (``exact=False``) is one O(tokens) CPU walk of the packing
    primitive and preserves its CUDA None-contract: with
    ``memory_minimal=False`` it is the no-sharing count, an upper bound on
    any planner-selected layout (safe for accepting a width); with
    ``memory_minimal=True`` it is the full-sharing count, the lower bound
    whose feasibility is monotone in width (valid for rejecting one).
    ``exact=True`` prices the planner's actual layouts (memoized by
    content) and is used only inside the band where those bounds disagree.
    Under CP it returns None: per-rank floors need materialized layouts.
    ``gdn_segments`` receives each gradient group's segment count: exact
    layouts' actual counts; in cheap mode, the same kind of bound as the
    token count (twice the requests, as a radix tree has fewer, or one).
    """

    if sync_planning_errors:
        self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
    with self._planning_status(sync_planning_errors):
        groups = self._group_active_request_indices(
            requests,
            checkpoint=checkpoint,
            ensure_slots=not sync_planning_errors,
        )
        if self._moe_layers and any(
            ref is not None and ref.name is not None for (ref, _), _ in groups
        ):
            # This cheap return type has no slot metadata. Materialize the
            # exact plan instead of admitting with the constructor rank.
            return None
        gradient_slots = [
            ref
            for (ref, grad), _ in groups
            if grad and ref is not None and ref.name is not None
        ]
        if (
            gradient_slots
            and getattr(self, "_recompute_granularity", None) == "full"
            and self._pending_adapter_gradient_bytes(gradient_slots)
        ):
            # The step's first backward allocates adapter gradients that
            # only slot metadata can price; the exact plan carries it.
            return None
        if (
            any(mode for (_, mode), _ in groups)
            and _impl._gdn_memory.model_shapes(self) is not None
        ):
            # Pending saves require the actual bucket/replayed-tail geometry.
            # Existing unavailable handling materializes before admission.
            return None
        if self._topology_key()[2] > 1:
            # CP token ownership can be uneven and memory floors are priced
            # per rank. Use the existing exact-plan fallback; a global token
            # count alone cannot price its peak.
            return None
        packed_tokens = 0
        head_workspace_bytes = 0
        group_rows: list[tuple[int, bool]] = []
        for (_slot, grad_enabled), group_indices in groups:
            head_requests = tuple(requests[index] for index in group_indices)
            lower = self._head_projection_rows(head_requests, lower_bound=True)
            upper = self._head_projection_rows(head_requests)
            if exact:
                tree, layout = self._select_group_layout(
                    tuple(
                        requests[index]
                        .input_tokens.reshape(-1)
                        .to(dtype=_impl.torch.long)
                        for index in group_indices
                    ),
                    memory_minimal=memory_minimal,
                    grad_enabled=grad_enabled,
                )
                physical_rows = self._physical_tokens(layout.packed_tokens)
                packed_tokens += physical_rows
                group_rows.append((physical_rows, grad_enabled))
                if grad_enabled and gdn_segments is not None:
                    gdn_segments.append(len(layout.segments))
                projected = upper
                positions = None
                # TP > 1 heads price each chunk's statistics path, so the
                # exact count needs the packed union beyond one chunk too.
                spread = (
                    self._topology_key()[1] > 1
                    and upper == _impl._HEAD_CHUNK_TOKENS
                    and self._head_projection_rows(
                        head_requests, lower_bound=True, uncapped=True
                    )
                    != self._head_projection_rows(head_requests, uncapped=True)
                )
                if lower != upper or spread:
                    packed = _impl.materialize_prefix_tree_layout(
                        tuple(
                            request.input_tokens.reshape(-1).to(dtype=_impl.torch.long)
                            for request in head_requests
                        ),
                        tree,
                        layout,
                        verify_shared_tokens=False,
                    )
                    projected = self._head_projection_rows(
                        head_requests, positions=packed.positions_by_sequence
                    )
                    positions = packed.positions_by_sequence
                head_workspace_bytes = max(
                    head_workspace_bytes,
                    self._group_head_workspace_bytes(
                        projected,
                        head_requests,
                        grad_enabled=grad_enabled,
                        positions=positions,
                    ),
                )
                continue
            # Radix depth is bounded by the number of rows, so ``len(group)``
            # is an unlimited-sharing depth for this group; it is a bound for
            # estimation, not a sharing policy.
            group_packed_tokens = _impl.estimate_prefix_tree_packed_tokens(
                (requests[index].input_tokens.reshape(-1) for index in group_indices),
                max_depth=len(group_indices) if memory_minimal else 0,
            )
            if group_packed_tokens is None:
                return None
            physical_rows = self._physical_tokens(group_packed_tokens)
            packed_tokens += physical_rows
            group_rows.append((physical_rows, grad_enabled))
            if grad_enabled and gdn_segments is not None:
                # Bounds like the token counts: at most twice the requests
                # without sharing (acceptance), at least one with full
                # sharing (rejection); exact pricing counts the rest.
                gdn_segments.append(1 if memory_minimal else 2 * len(group_indices))
            head_workspace_bytes = max(
                head_workspace_bytes,
                self._group_head_workspace_bytes(
                    lower if memory_minimal else upper,
                    head_requests,
                    grad_enabled=grad_enabled,
                    lower_bound=memory_minimal,
                ),
            )

        return (
            packed_tokens,
            self._estimate_group_request_output_bytes(requests),
            self._memory_signature_from_requests(
                requests,
                slot_group_count=len(groups),
                grad_modes=tuple(mode for (_, mode), _ in groups),
                slot_groups=tuple(key for key, _ in groups),
            ),
            tuple(group_rows),
            head_workspace_bytes,
        )


def _rate_learning_eligible(
    self: TrainerRank, plan: _impl._FlatForwardPlan, *, borrow: bool
) -> bool:
    profile = self._memory_profiles.get(plan.signature)
    extent = 0 if profile is None else _trusted_extent(profile)
    if borrow and (
        reference := _rate_reference(
            self,
            plan.signature,
            profile,
            packed_tokens=plan.packed_tokens,
            caller_cp_workload=getattr(plan, "caller_cp_workload", None),
        )
    ):
        extent = max(extent, reference.extent)
    return plan.packed_tokens >= _rate_floor_tokens(plan.signature) and (
        (profile is not None and not _unanchored(profile))
        or plan.packed_tokens <= extent * _impl._MEMORY_PROFILE_TRUST_GROWTH
    )


def _complete_caller_memory_profile(
    self: TrainerRank,
    plan: _impl._AnyForwardPlan,
    baseline: int | None,
    interval: tuple[int, int] | None,
    forward_peak: int,
) -> None:
    """One control-only MIN at the replicated iterator's common resume point.

    No plan/sample/error branch bypasses this vote. Handoff's existing status
    exchange is too early: the caller interval is still open then. Waiting for
    the next admission is too late and would omit the final wave. Local byte
    readings are frozen before the vote and are never reduced.
    """
    error: BaseException | None = None
    reading = None
    eligible = complete = False
    try:
        if isinstance(plan, _impl._FlatForwardPlan) and baseline is not None:
            peak = int(_impl.torch.cuda.max_memory_allocated(self.device))
            reading = (
                self.__dict__.get("_peak_resets", 0),
                peak,
            )
            eligible = _rate_learning_eligible(self, plan, borrow=True)
            complete = (
                interval is not None
                and interval[0] == reading[0]
                and reading[1] >= interval[1]
            )
    except BaseException as exc:
        error = exc
    exchange_error: BaseException | None = None
    try:
        agreed = self._recovery_reduce(
            [
                float(error is None),
                float(reading is not None),
                float(eligible),
                float(complete),
            ],
            op="MIN",
            sync_across_dp=True,
        )
    except BaseException as exc:
        if error is None:
            raise
        exchange_error = exc
    if error is not None:
        raise self._memory_error_with_reduction_note(error, exchange_error)
    if not agreed[0]:
        raise RuntimeError("Memory profile sampling failed on another rank")
    if isinstance(plan, _impl._FlatForwardPlan):
        if reading is not None:
            self._update_peak_memory_profile(
                plan,
                baseline,
                caller_phase=True,
                interval=interval,
                _reading=reading,
                _agreed=(bool(agreed[1] and agreed[2]), bool(agreed[1] and agreed[3])),
            )
    elif baseline is not None:
        self._record_split_memory_floor(plan, baseline, forward_peak)


def _update_peak_memory_profile(
    self: TrainerRank,
    plan: _impl._FlatForwardPlan,
    baseline: int | None,
    retained_after: int | None = None,
    *,
    caller_phase: bool = False,
    interval: tuple[int, int] | None = None,
    _reading: tuple[int, int] | None = None,
    _agreed: tuple[bool, bool] | None = None,
) -> None:
    if baseline is None:
        return
    if _reading is not None:
        resets, peak = _reading
    else:
        peak = int(_impl.torch.cuda.max_memory_allocated(self.device))
        resets = self.__dict__.get("_peak_resets", 0)
    observation = getattr(self, "_planner_observation", None)
    if observation is not None:
        observation["peak"] = max(observation["peak"], peak)
    self._peak_reading = (resets, peak)
    profile = self._memory_profiles.get(plan.signature)
    rate_eligible = True
    if _fixed_slope_enabled(profile):
        # Forward children / direct forwards may execute asymmetrically or
        # fail before peers reach the end. They never vote or borrow a learning
        # window here; the common caller boundary supplies that permission.
        rate_eligible = (
            _rate_learning_eligible(self, plan, borrow=False)
            if _agreed is None
            else _agreed[0]
        )
    self._update_memory_profile(
        plan,
        max(0, peak - baseline),
        retained_bytes=(
            None if retained_after is None else max(0, retained_after - baseline)
        ),
        # Only a whole wave: a nested forward resetting the counter during
        # the yield, or an untracked reset (the counter fell below this
        # wave's forward peak), leaves the peak since then short of it.
        caller_phase=caller_phase
        and interval is not None
        and interval[0] == resets
        and peak >= interval[1]
        and (_agreed is None or _agreed[1]),
        **({"rate_eligible": rate_eligible} if _fixed_slope_enabled(profile) else {}),
    )


def _estimate_group_request_output_bytes(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
) -> int:
    total = 0
    for request in requests:
        if request.routed_experts is not None:
            # Retained int64 replay targets for attention and GDN layouts.
            total += 2 * request.routed_experts.numel() * 8
        seq_len = int(request.input_tokens.numel())
        if request.target_tokens is not None:
            total += int(request.target_tokens.numel()) * _impl._dtype_size(
                _impl.torch.float32
            )
        if request.top_k is not None:
            total += (
                seq_len
                * int(request.top_k)
                * (
                    _impl._dtype_size(_impl.torch.float32)
                    + _impl._dtype_size(_impl.torch.long)
                )
            )
        if request.logits:
            if self._padded_vocab_size is None:
                raise RuntimeError("logits output memory requires a GPT model")
            total += seq_len * self._padded_vocab_size * self._param_dtype_size
        if request.hidden_states:
            total += seq_len * self._hidden_size * self._param_dtype_size
    return total


def _memory_signature_from_requests(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    slot_group_count: int,
    grad_modes: Iterable[bool],
    slot_groups: Iterable[tuple["LoRASlotRef | None", bool]] = (),
) -> _impl._MemorySignature:
    from art.megatron.context_parallel.checkpoint import current_checkpoint_partition

    partition = current_checkpoint_partition()
    if partition is not None:
        # Native MoE storage sharding failed full-model gradient parity on H200.
        if self._moe_layers:
            raise ValueError("checkpoint shards are qualified for dense models only")
        degree, shape = partition
        if (
            self._topology_key() != (degree, 1, 1, 1)
            or len(requests) != 1
            or int(requests[0].input_tokens.numel()) != shape[0]
            or shape[2] != self._hidden_size
            or not _checkpoint_layers(self, ((shape[0], True),))
        ):
            raise ValueError("checkpoint shards require one CP1 uniform-recompute root")
    modes = tuple(sorted(grad_modes))
    topology = self._topology_key()
    slot_groups = tuple(slot_groups)
    if not slot_groups and _native_dense_cp(self):
        # Direct estimator callers use the template adapter when no slot
        # selection was supplied, just as _slot_lora_tensors(None) does.
        slot_groups = tuple((None, grad) for grad in modes)
    shapes = tuple(
        sorted((grad, self._slot_memory_shapes(ref)) for ref, grad in slot_groups)
    )
    signature = _impl._MemorySignature(
        topology=topology,
        planner_coefficients=(self._coefficient_version, self._coefficient_table),
        slot_group_count=slot_group_count,
        request_mix=tuple(
            sorted({_impl._request_mix_key(request) for request in requests})
        ),
        grad_enabled=any(modes),
        grad_modes=modes,
        slot_shapes=shapes if any(any(shape) for _, shape in shapes) else (),
        short_requests=any(_impl._short_request(request) for request in requests),
    )
    bounded = (
        _BOUNDED_FORWARD_PEAKS.get()
        and len(requests) == 1
        and requests[0].input_tokens.ndim == 1
        and (modes == (False,) or (modes == (True,) and _DIRECT_FORWARD.get()))
        and _impl._request_mix_key(requests[0]) in ("hidden", "hidden+routes")
        and topology[1] == topology[3] == 1
        and (
            topology[2] == 1
            or (topology[2] in (2, 4, 8) and self._parallel_shape.cp == topology[2])
        )
        and self._parallel_shape.ep == self._parallel_shape.etp == 1
        and bool(
            _checkpoint_layers(self, ((int(requests[0].input_tokens.numel()), False),))
        )
        and (
            self._moe_memory_supported
            if self._moe_layers
            else self._dense_fc1_adapted and bool(self._geometry.ffn_hidden_size)
        )
    )
    # Single-root checkpointed MoE training has fixed head/adapter workspace.
    # Keep its complete caller peaks separate by topology and from forward or
    # multi-root profiles. CP observations use global TP-padded packed tokens,
    # as admission does. CP ownership need not be monotone in global length:
    # a shorter root can give a rank more model rows and different workspace.
    # CP discounts require a covering global length AND per-rank ownership
    # and workspace in the same attention/GDN family. Each rank uses its own
    # complete peak; collective admission still checks every local fit.
    caller_bounded = (
        len(requests) == slot_group_count == 1
        and requests[0].input_tokens.ndim == 1
        and modes == (True,)
        and _impl._request_mix_key(requests[0]) == "target:single"
        and topology[0] == topology[1] == topology[3] == 1
        and topology[2] >= 1
        and self._parallel_shape.cp == topology[2]
        and self._parallel_shape.ep == self._parallel_shape.etp == 1
        and self._moe_layers
        and self._moe_memory_supported
        and bool(
            _checkpoint_layers(self, ((int(requests[0].input_tokens.numel()), True),))
        )
    )
    if partition is not None or bounded or caller_bounded:
        return _ScopedMemorySignature(
            **vars(signature),
            checkpoint_partition=partition,
            bounded_forward_peak=bounded,
            bounded_caller_peak=bool(caller_bounded),
        )
    return signature


def _slot_memory_shapes(
    self: TrainerRank, ref: "LoRASlotRef | None"
) -> tuple[tuple[int, ...], ...]:
    """Separate empirical trust across actual selected adapter layouts.

    TP x SP and native dense CP checkpoint floors price LoRA intermediates
    from these ranks, including attention-only dense models and the template
    adapter selected by a None reference. For dense CP, (0,) marks incomplete
    classification; () is a recognized owner with an inactive selected slot.
    """
    dense_cp = _native_dense_cp(self)
    if (
        (ref is None and not dense_cp)
        or (ref is not None and ref.name is None)
        or isinstance(ref, _impl._LocalLoRASlotRef)
        or not (
            getattr(self, "_moe_layers", 0)
            or getattr(self, "_gdn_layers", 0)
            or getattr(self, "_sequence_parallel", False)
            or dense_cp
        )
    ):
        # Generic/no-component planning must not import Megatron or walk
        # model owners just to construct its existing memory signature.
        return ()
    from art.megatron.lora import LoRA

    shapes = []
    owned: set[int] = set()
    for chunk in self.runtime.model:
        for module in chunk.modules():
            if id(module) in owned:
                continue
            if isinstance(module, LoRA):
                if dense_cp:
                    # Slot Parameters belong to this recognized owner; they
                    # are not separate, unclassified adapter implementations.
                    owned.update(
                        id(child) for child in module.modules() if child is not module
                    )
                try:
                    tensors = _impl._slot_lora_tensors(module, ref)
                    shape = (
                        ()
                        if tensors is None
                        else (tensors[0].ndim, *tensors[0].shape, *tensors[1].shape)
                    )
                    if (
                        dense_cp
                        and tensors is not None
                        and any(
                            tensor.element_size() != self._param_dtype_size
                            for tensor in tensors
                        )
                    ):
                        shapes.append((0,))
                except (AttributeError, RuntimeError, TypeError):
                    shape = (0,)
                shapes.append(shape)
                if dense_cp and (
                    any(
                        name in vars(module)
                        or getattr(type(module), name) is not getattr(LoRA, name)
                        for name in (
                            "forward",
                            "active_lora_tensors",
                            "_slot",
                            "_active_lora_storage",
                            "_expert_views",
                        )
                    )
                    or module._forward_hooks
                    or module._forward_pre_hooks
                ):
                    # Inherited native behavior is priced; an overridden
                    # kernel/selection path needs its own workspace proof.
                    shapes.append((0,))
            elif dense_cp and (
                any(param.requires_grad for param in module.parameters(recurse=False))
                or any(
                    hasattr(module, name)
                    for name in (
                        "A_T",
                        "B_T",
                        "lora_A",
                        "lora_B",
                        "_lora_params",
                        "lora_slot_params",
                    )
                )
            ):
                # All trainable tensors must be owned by a recognized adapter.
                # Also catch frozen/inactive foreign factor owners. Never infer
                # completeness from the number of recognized owners alone.
                shapes.append((0,))
    return tuple(shapes)


def _memory_check(
    self: TrainerRank,
    forward: _impl._FlatForwardPlan,
    *,
    sync_across_dp: bool = False,
    sync_planning_errors: bool = False,
) -> _impl._MemoryCheck:
    with self._planning_status(sync_planning_errors):
        required = self._estimate_required_memory_bytes_from_values(
            packed_tokens=forward.packed_tokens,
            output_bytes=forward.output_bytes,
            signature=forward.signature,
            logical_tokens=forward.active_logical_tokens,
            gdn_segments=forward.grad_segment_count,
            group_rows=self._plan_group_rows(forward),
            slot_refs=tuple(g.slot_ref for g in forward.groups),
            head_workspace_bytes=self._plan_head_workspace_bytes(forward),
            checkpoint_floor=_impl._gdn_memory.plan_floor(self, forward),
            retained_tokens=self._plan_retained_tokens(forward),
            caller_cp_workload=getattr(forward, "caller_cp_workload", None),
        )
        growth = int(
            self._plan_hybridep_growth_bytes(forward) * _impl._MEMORY_SAFETY_FACTOR
        )
        if growth:
            growth = math.ceil(
                growth
                * _cold_extrapolation_factor(
                    self._memory_profiles.get(forward.signature), forward.packed_tokens
                )
            )
        required += growth
    return self._memory_check_required(required, sync_across_dp=sync_across_dp)


def _refresh_memory_check(
    self: TrainerRank, check: _impl._MemoryCheck, *, sync_across_dp: bool
) -> _impl._MemoryCheck:
    decision = _impl._planner_evidence.current(self)
    # Re-sample each rank against its own demand, not another DP rank's maximum.
    # Older synthetic checks may lack the local producer; keep their safe bound.
    with decision.refresh_of(check.sample) if decision is not None else nullcontext():
        return self._memory_check_required(
            check.estimated_required_bytes
            if check.local_required_bytes is None
            else check.local_required_bytes,
            sync_across_dp=sync_across_dp,
        )


def _memory_check_required(
    self: TrainerRank,
    required: int,
    *,
    sync_across_dp: bool = False,
) -> _impl._MemoryCheck:
    decision = _impl._planner_evidence.current(self)
    details: dict[str, Any] | None = {} if decision is not None else None
    local_required, scope = required, "local"
    if _impl.dist.is_available() and _impl.dist.is_initialized():
        group = (
            getattr(self, "_forward_replica_group", None)
            if sync_across_dp
            else self._forward_memory_group()
        )
        scope = "world" if group is None else "tp_cp"
        values = _impl.torch.tensor(
            [float(required), 0.0, 0.0],
            device=self._forward_collective_device() if sync_across_dp else self.device,
            dtype=_impl.torch.float64,
        )
        _impl.dist.all_reduce(values[0], op=_impl.dist.ReduceOp.MAX, group=group)
        required = int(values[0].item())
        error: BaseException | None = None
        try:
            available = (
                self._available_memory_bytes()
                if details is None
                else self._available_memory_bytes(details)
            )
            local_available = available
        except BaseException as exc:
            error, available = exc, -1
        exchange_error: BaseException | None = None
        try:
            # A healthy communicator carries local failure to every peer
            # in the existing MIN. This cannot repair a poisoned backend.
            values[1] = available
            # Extrema can come from different DP ranks. Agree that every rank
            # fits its own demand while retaining extrema for diagnostics.
            values[2] = available >= local_required
            _impl.dist.all_reduce(values[1:], op=_impl.dist.ReduceOp.MIN, group=group)
            available = int(values[1].item())
            fits = bool(values[2].item())
        except BaseException as exc:
            if error is None:
                raise
            exchange_error = exc
        if error is not None:
            raise self._memory_error_with_reduction_note(error, exchange_error)
        if available < 0:
            raise RuntimeError("Memory admission failed on another rank")
    else:
        available = (
            self._available_memory_bytes()
            if details is None
            else self._available_memory_bytes(details)
        )
        local_available = available
        fits = required <= available
    sample = None
    if decision is not None:
        try:
            sample = decision.observe(
                scope=scope,
                local_required_bytes=local_required,
                local_available_bytes=local_available,
                reduced_required_bytes=required,
                reduced_available_bytes=available,
                safety_factor=_impl._MEMORY_SAFETY_FACTOR,
                reserve_fraction=_impl._MEMORY_RESERVE_FRACTION,
                **(details or {}),
            )
        except Exception:
            _impl._planner_misses._warn("could not retain admission sample")
    return _impl._MemoryCheck(
        estimated_required_bytes=required,
        available_bytes=available,
        fits=fits,
        local_required_bytes=local_required,
        local_available_bytes=local_available,
        sample=sample,
    )


def _forward_memory_group() -> dist.ProcessGroup | None:
    try:
        from megatron.core import parallel_state as ps

        return ps.get_tensor_and_context_parallel_group(check_initialized=False)
    except (AssertionError, ImportError, RuntimeError, ValueError):
        return None


def _estimate_required_memory_bytes_from_values(
    self: TrainerRank,
    *,
    packed_tokens: int,
    output_bytes: int,
    signature: _impl._MemorySignature,
    logical_tokens: int | None = None,
    gdn_segments: int = 0,
    group_rows: tuple[tuple[int, bool], ...] = (),
    slot_refs: tuple["LoRASlotRef | None", ...] | None = None,
    head_workspace_bytes: int = 0,
    checkpoint_floor: tuple[int, int] = (0, 0),
    retained_tokens: int | None = None,
    include_checkpoint_input_gradient: bool = True,
    checkpoint_memory: tuple[int, int] | None = None,
    caller_cp_workload: _CPWorkload | None = None,
    include_cold_extrapolation: bool = True,
) -> int:
    if packed_tokens <= 0:
        return output_bytes
    profiled = self._memory_profiles.get(signature)
    activation_factor = max(4, min(16, self._num_layers // 4 + 4))
    static_compute = (
        packed_tokens * self._hidden_size * self._param_dtype_size * activation_factor
    )
    no_grad_stage = 0
    if (
        not self._geometry.moe_experts
        and self._geometry.ffn_hidden_size
        and self._dense_fc1_adapted
        and signature.topology[1] == 1
        and (
            signature.topology[2] == 1
            or getattr(signature, "bounded_forward_peak", False)
        )
    ):
        # A dense no-grad layer peaks at its FC1 stage (the base GEMM output,
        # the adapter output and their sum: 6F, or the SwiGLU live set if
        # wider) beside the residual pair, embedding and norm output (4H), per
        # row: Qwen3.8-27B TP1/CP1 traces at 7k-174k rows. Under CP use actual
        # physical rank rows, never a global count divided by degree. Groups
        # run one after another, so the largest no-grad group bounds it.
        rows = max(
            (n for n, grad in group_rows if not grad),
            default=0 if signature.grad_enabled else packed_tokens,
        )
        no_grad_stage = (
            rows
            * self._param_dtype_size
            * (
                max(6, self._mlp_activation_factor) * self._geometry.ffn_hidden_size
                + 4 * self._hidden_size
            )
        )
    if signature.grad_enabled and self._recompute_granularity != "full":
        geometry = self._geometry
        hidden = self._hidden_size
        tp = max(1, self._topology_key()[1])
        sp = tp if self._sequence_parallel else 1
        # Gathered LoRA inputs alias norm output without sequence sharding.
        gathered = hidden if sp > 1 else 0
        common = 2 * hidden / sp + gathered
        attention_width = geometry.num_attention_heads * geometry.kv_channels or hidden
        kv_width = geometry.num_query_groups * geometry.kv_channels or hidden
        gated = self._attention_output_gate
        attention = common + ((7 if gated else 5) * attention_width + 3 * kv_width) / tp
        if 0 < geometry.num_query_groups < tp:
            # SelfAttentionLinearQKVLoRA constructs global QKV before
            # slicing it when KV groups cannot be partitioned across TP.
            attention += ((2 if gated else 1) * attention_width + 2 * kv_width) * (
                1 - 1 / tp
            )
        gdn = (
            common
            + (
                4 * geometry.gdn_key_heads * geometry.gdn_key_head_dim
                + 8 * geometry.gdn_value_heads * geometry.gdn_value_head_dim
            )
            / tp
        )
        ffn_width = geometry.ffn_hidden_size or 4 * hidden
        mlp = common + self._mlp_activation_factor * ffn_width / tp
        if geometry.moe_experts:
            # Keep the worst-case dispatch envelope: random-weight runs
            # cannot establish an EP discount for pretrained routing.
            ffn_width = (
                geometry.moe_topk * geometry.moe_ffn_hidden_size
                + geometry.moe_shared_expert_ffn
            ) or ffn_width
            mlp = common + 6 * ffn_width + 2 * hidden * max(0, geometry.moe_topk - 1)
        gdn_layers = min(self._num_layers, self._gdn_layers)
        retained_features = (
            (self._num_layers - gdn_layers) * attention
            + gdn_layers * gdn
            + self._num_layers * mlp
        )
        if self._recompute_granularity == "selective":
            checkpointed = (
                self._checkpointed_moe_layers
                if geometry.moe_experts
                else self._num_layers
                if "mlp" in self._recompute_modules
                else 0
            )
            # Checkpoints keep their input (and MoE's external norm); one
            # live MLP still needs workspace, including worst-case dispatch.
            checkpoint_input = (2 if geometry.moe_experts else 1) * hidden / sp
            retained_features -= max(0, checkpointed - 1) * (mlp - checkpoint_input)
        # Each GDN segment can retain an initial and a final recurrent
        # state (fp32), plus convolution history. Unlike token activations,
        # these do not shrink with segment length.
        gdn_state_bytes = gdn_segments * gdn_layers * self._gdn_segment_layer_bytes()
        static_compute = max(
            static_compute,
            # Cold eager runs allocate ~58 MiB beyond warm retention for
            # native kernel initialization; a slope alone misses short inputs.
            64 * 2**20
            + gdn_state_bytes
            + (
                packed_tokens
                if retained_tokens is None or geometry.moe_experts
                else retained_tokens
            )
            * self._param_dtype_size
            * retained_features,
        )
    # Groups execute sequentially: summed packed rows conservatively bound
    # this FC2 component, not all workspace or retained graphs.
    static_compute = max(
        static_compute,
        *(
            self._moe_workspace_bytes(
                sum(rows for rows, _ in group_rows)
                if signature.topology[2] > 1 and group_rows
                else packed_tokens,
                slot_ref=ref,
            )
            for ref in (slot_refs or (None,))
        ),
    )
    retained, workspace = (
        self._sequence_parallel_lora_floor(
            self._checkpoint_memory_floor(group_rows, slot_refs, gdn_segments),
            group_rows,
            signature,
        )
        if checkpoint_memory is None
        else checkpoint_memory
    )
    backward = 0
    if include_checkpoint_input_gradient and retained:
        # The backward's other end and cold transients, as _subforward_cost.
        backward = self._checkpoint_input_gradient_bytes(
            group_rows, retained
        ) + self._checkpoint_adapter_gradient_bytes(
            self._checkpoint_gradient_groups(group_rows, slot_refs)
        )
        if profiled is None and any(grad for _, grad in group_rows):
            backward += self._cold_recompute_transient_bytes()
    static_compute = max(
        static_compute,
        max(retained, checkpoint_floor[0])
        + max(workspace, head_workspace_bytes, checkpoint_floor[1])
        + backward,
        # A no-grad group's forward stage, beside any gradient groups'
        # boundaries but not the backward's input gradient.
        max(retained, checkpoint_floor[0]) + no_grad_stage,
    )
    if signature.topology[2] > 1:
        # Local head results coexist with full CP outputs during gathering.
        # Uneven rank plans can assign all of an item's rows to one rank.
        static_compute += output_bytes
    # Memory that grows with logical rows makes a profile learned under
    # lighter sharing underestimate a deeper-shared plan; scale the trusted
    # estimate up by the ratio gap. Packed pricing instead charges head and
    # caller memory per logical row and extrapolates the rest at most the
    # usual trust growth. Normalize before multiplying: cancelling packed
    # tokens through two float operations can otherwise make a larger warm
    # layout cheaper.
    packed_priced = profiled is not None and _impl._packed_priced(
        signature, self._one_layer_recompute()
    )

    def profiled_tokens(logical_per_packed: float) -> int | float:
        if logical_tokens is None:
            return packed_tokens
        return max(
            packed_tokens,
            logical_tokens
            / logical_per_packed
            / (_impl._MEMORY_PROFILE_TRUST_GROWTH if packed_priced else 1),
        )

    # The trust window limits calibration growth, not the empirical floor.
    # Dropping that floor beyond the window can admit a larger request that
    # was refused just inside it, even below a previously observed peak.
    def empirical(profiled: _impl._MemoryProfile) -> int:
        profiled_bytes = _profile_compute_peak(
            profiled, profiled_tokens(profiled.logical_per_packed)
        )
        if isinstance(profiled, _CallerMemoryProfile):
            profiled_bytes = profiled.compute_peak(
                profiled_tokens(profiled.logical_per_packed),
                context_parallel=signature.topology[2] > 1,
                cp_workload=caller_cp_workload,
            )
        elif (
            isinstance(profiled, _BoundedMemoryProfile)
            and profiled.forward_compute_peaks
        ):
            profiled_bytes = profiled.compute_peak(
                profiled_tokens(profiled.logical_per_packed)
            )
        elif (
            profiled.peak_model_version == 1
            and profiled.warm_bytes_per_token is not None
            and profiled.warm_packed_tokens is not None
            and profiled.warm_logical_per_packed is not None
        ):
            # Later plans' own sharing: a rate learned under lighter
            # sharing scales up for deeper-shared plans, as above. Pricing
            # smaller waves as the smallest later plan keeps cost monotone
            # in tokens, which the width search's bounds rely on; it is
            # sound where a wave's memory is a fixed cost plus a per-token
            # rate, as that plan's rate covers its share of the fixed cost.
            profiled_bytes = min(
                profiled_bytes,
                profiled.warm_bytes_per_token
                * max(
                    profiled.warm_packed_tokens,
                    profiled_tokens(profiled.warm_logical_per_packed),
                ),
            )
        return int(
            profiled_bytes
            + (
                _impl._PACKED_PRICED_LOGICAL_ROW_BYTES * logical_tokens
                # Branch states grow with segments, not packed rows;
                # backward recomputes one layer at a time.
                + gdn_segments * self._gdn_segment_layer_bytes()
                if packed_priced and logical_tokens is not None
                else 0
            )
        )

    compute = (
        static_compute if profiled is None else max(static_compute, empirical(profiled))
    )
    if _fixed_slope_enabled(profiled) and _unanchored(profiled):
        floor = _rate_floor_tokens(signature)
        tokens = profiled_tokens(
            1.0 if profiled is None else profiled.logical_per_packed
        )
        reference = _rate_reference(
            self,
            signature,
            profiled,
            packed_tokens=tokens,
            caller_cp_workload=caller_cp_workload,
        )
        if reference is None:
            if packed_tokens >= floor:
                # Finite and exactly representable through the existing MAX/
                # MIN collectives. Never price an unknown large wave fixed-only.
                compute = max(compute, _MEMORY_DEFERRED_BYTES)
        else:
            fixed = 0 if profiled is None else _absolute_floor(profiled, floor)
            if reference.source != "signature":
                fixed += reference.fixed
                tokens = max(tokens, reference.minimum)
            compute = max(
                compute, fixed + _memory_model.predict(0, reference.rate, tokens)
            )
    factor = (
        _cold_extrapolation_factor(profiled, packed_tokens)
        if include_cold_extrapolation
        else 1.0
    )
    return (
        math.ceil((output_bytes + compute) * _impl._MEMORY_SAFETY_FACTOR * factor)
        if factor != 1
        else int((output_bytes + compute) * _impl._MEMORY_SAFETY_FACTOR)
    )


def _gdn_segment_layer_bytes(self: TrainerRank) -> float:
    """Initial and final fp32 recurrent states plus convolution history."""
    geometry = self._geometry
    return (
        2
        / max(1, self._topology_key()[1])
        * (
            4
            * geometry.gdn_value_heads
            * geometry.gdn_key_head_dim
            * geometry.gdn_value_head_dim
            + self._param_dtype_size
            * (
                2 * geometry.gdn_key_heads * geometry.gdn_key_head_dim
                + geometry.gdn_value_heads * geometry.gdn_value_head_dim
            )
            * max(0, geometry.gdn_conv_kernel - 1)
        )
    )


def _available_memory_bytes(
    self: TrainerRank, sample: dict[str, Any] | None = None
) -> int:
    if not (_impl.torch.cuda.is_available() and self.device.type == "cuda"):
        return 1 << 60
    free, total = _impl.torch.cuda.mem_get_info(self.device)
    allocator = _impl.torch.cuda.get_allocator_backend()
    stats = None
    if allocator == "native":
        # Cached bytes are not physical free memory. This sample does not
        # reserve memory for execution or the caller's later backward.
        if sample is None:
            allocated = int(_impl.torch.cuda.memory_allocated(self.device))
        else:
            # memory_allocated uses this same stats read. Retain its other
            # already-returned fields without another allocator query.
            stats = _impl.torch.cuda.memory_stats(self.device)
            allocated = int(stats.get("allocated_bytes.all.current", 0))
        reusable_reserved = 0
    else:
        # Preserve the previous, unqualified policy for other backends.
        allocated = int(_impl.torch.cuda.memory_allocated(self.device))
        reserved = int(_impl.torch.cuda.memory_reserved(self.device))
        reusable_reserved = max(0, reserved - allocated)
    reserve = int(total * _impl._MEMORY_RESERVE_FRACTION)
    available = max(0, int(free) + reusable_reserved - reserve)
    test_cap = None
    if _impl.os.environ.get(_impl._TEST_HOOKS_ENV) == "1":
        limit = _impl.os.environ.get(_impl._TEST_MEMORY_LIMIT_ENV)
        if limit:
            test_cap = int(limit)
            # Test-only: cap the usable budget relative to the current
            # allocation so acceptance cells can induce split/decline
            # behavior deterministically without ballast tensors.
            available = min(available, max(0, int(limit) - allocated))
    if sample is not None:
        try:
            sample.update(
                allocator=allocator,
                physical_free_bytes=int(free),
                physical_total_bytes=int(total),
                allocated_bytes=allocated,
                reserve_bytes=reserve,
                test_cap_bytes=test_cap,
                reserved_bytes=(
                    stats.get("reserved_bytes.all.current")
                    if stats is not None
                    else reserved
                    if allocator != "native"
                    else None
                ),
                inactive_split_bytes=(
                    stats.get("inactive_split_bytes.all.current")
                    if stats is not None
                    else None
                ),
            )
        except Exception:
            sample.clear()
    return available


def _all_ranks_have_memory_profile(
    self: TrainerRank,
    *,
    packed_tokens: int,
    signature: _impl._MemorySignature,
) -> bool:
    profile = self._memory_profiles.get(signature)
    local = packed_tokens <= 0 or (
        profile is not None
        and _trusted_extent(profile) * _impl._MEMORY_PROFILE_TRUST_GROWTH
        >= packed_tokens
    )
    return self._all_ranks_true(local)


def _update_memory_profile(
    self: TrainerRank,
    plan: _impl._FlatForwardPlan,
    peak_delta_bytes: int,
    *,
    retained_bytes: int | None,
    caller_phase: bool = False,
    rate_eligible: bool = True,
) -> None:
    if plan.packed_tokens <= 0:
        return
    compute_delta = max(0, peak_delta_bytes - plan.output_bytes)
    bytes_per_token = compute_delta / max(1, plan.packed_tokens)
    previous = self._memory_profiles.get(plan.signature)
    fixed_slope = _fixed_slope_enabled(previous)
    if fixed_slope and previous is not None and previous.peak_model_version == 1:
        # Explicit legacy restoration uses the merge helper. Fresh collection
        # must not import a scalar v1 poison rate into v2 diagnostic guards.
        previous = None
    model_fields: dict[str, Any] = {}
    if fixed_slope:
        floor = _rate_floor_tokens(plan.signature)
        qualifies = rate_eligible and plan.packed_tokens >= floor
        observations = _memory_model.merge_points(
            () if previous is None else previous.compute_observations,
            ((plan.packed_tokens, compute_delta),),
        )
        rates = _memory_model.merge_points(
            () if previous is None else _rate_points(previous, floor),
            ((plan.packed_tokens, compute_delta),) if qualifies else (),
        )
        absolute = max(
            0 if previous is None else _absolute_floor(previous, floor),
            0 if qualifies else compute_delta,
        )
        fixed, slope = _memory_model.fit_points(
            observations, rate_points=rates, absolute_floor=absolute
        )
        legacy_rate = 0.0 if previous is None else previous.legacy_bytes_per_token
        model_fields = dict(
            peak_model_version=_memory_model._MEMORY_PROFILE_VERSION,
            compute_observations=observations,
            fixed_compute_bytes=fixed,
            compute_slope=slope,
            legacy_bytes_per_token=legacy_rate,
            profile_format_version=5,
            rate_floor_tokens=floor,
            rate_observations=rates,
            absolute_compute_floor=absolute,
            eligible_packed_tokens=max(
                0
                if previous is None
                else previous.eligible_packed_tokens
                if previous.profile_format_version >= 4
                else max((n for n, _ in _rate_points(previous, floor)), default=0),
                plan.packed_tokens if qualifies else 0,
            ),
            subfloor_packed_tokens=max(
                0 if previous is None else previous.subfloor_packed_tokens,
                plan.packed_tokens if plan.packed_tokens < floor else 0,
            ),
        )
    logical_per_packed = plan.active_logical_tokens / max(1, plan.packed_tokens)
    warm_rate = None if previous is None else previous.warm_bytes_per_token
    warm_tokens = None if previous is None else previous.warm_packed_tokens
    warm_sharing = None if previous is None else previous.warm_logical_per_packed
    caller_plans = 0 if previous is None else previous.caller_plans
    # Only a flat wave's whole caller phase covers the caller's backward;
    # split children and dp_rank_forward observe forward only. A caller that
    # defers backward past the yield is not covered. The profile's first
    # such plan also pays one-time costs, so it is not warm.
    learns_rates = not fixed_slope or (rate_eligible and plan.packed_tokens >= floor)
    if caller_phase and learns_rates:
        if caller_plans:
            warm_rate = max(bytes_per_token, warm_rate or 0.0)
            warm_tokens = min(plan.packed_tokens, warm_tokens or plan.packed_tokens)
            warm_sharing = max(logical_per_packed, warm_sharing or 1.0)
        caller_plans += 1
    elif caller_plans and learns_rates:
        # After the seed, other readings cannot fit the warm profile, but a
        # higher one still raises its rate (inert until a whole warm plan
        # fits the rest), as it raises the fit over every observation.
        warm_rate = max(warm_rate or 0.0, bytes_per_token)
    retained_fraction = None if previous is None else previous.retained_fraction
    retained_compute = (
        None if previous is None else previous.retained_compute_bytes_per_token
    )
    if retained_bytes is not None:
        observed = min(1.0, retained_bytes / max(1, peak_delta_bytes))
        # Max-merge once observed. ``None`` (never observed) is distinct
        # from an observed 1.0, so a later, lower observation cannot
        # replace it.
        retained_fraction = (
            observed if retained_fraction is None else max(retained_fraction, observed)
        )
        if learns_rates:
            observed_compute = max(
                0, min(retained_bytes, peak_delta_bytes) - plan.output_bytes
            ) / max(1, plan.packed_tokens)
            retained_compute = max(retained_compute or 0.0, observed_compute)
    if fixed_slope:
        model_fields.update(
            retained_rate_tokens=max(
                0 if previous is None else previous.retained_rate_tokens,
                plan.packed_tokens
                if retained_bytes is not None and learns_rates
                else 0,
            ),
            retained_compute_peaks=_merge_retained_peaks(
                () if previous is None else _retained_floor_points(previous),
                (
                    (
                        plan.packed_tokens,
                        math.ceil(
                            max(
                                0,
                                min(retained_bytes, peak_delta_bytes)
                                - plan.output_bytes,
                            )
                        ),
                    ),
                )
                if retained_bytes is not None
                else (),
            ),
        )
    profile_cls = _impl._MemoryProfileV2 if fixed_slope else _impl._MemoryProfile
    self._memory_profiles[plan.signature] = profile_cls(
        # Preserve this legacy diagnostic/transport field. Rate-qualified admission
        # sources rates exclusively from rate_observations, never this maximum.
        bytes_per_token=max(
            bytes_per_token,
            0.0 if previous is None else previous.bytes_per_token,
        ),
        packed_tokens=max(
            plan.packed_tokens,
            0 if previous is None else previous.packed_tokens,
        ),
        logical_per_packed=max(
            logical_per_packed if learns_rates else 1.0,
            1.0 if previous is None else previous.logical_per_packed,
        ),
        retained_fraction=retained_fraction,
        retained_compute_bytes_per_token=retained_compute,
        warm_bytes_per_token=warm_rate,
        warm_packed_tokens=warm_tokens,
        warm_logical_per_packed=warm_sharing,
        caller_plans=caller_plans,
        **model_fields,
    )
    if (
        isinstance(plan.signature, _ScopedMemorySignature)
        and plan.signature.bounded_caller_peak
    ):
        peaks = dict(
            previous.caller_compute_peaks
            if isinstance(previous, _CallerMemoryProfile)
            else ()
        )
        if caller_phase:
            peaks[plan.packed_tokens] = max(
                compute_delta, peaks.get(plan.packed_tokens, 0)
            )
        observations = []
        peak = 0
        for length, compute in sorted(peaks.items()):
            peak = max(peak, compute)
            observations.append((length, peak))
        if len(observations) > 8:
            observations = [observations[0], *observations[-7:]]
        cp_peaks = {
            (length, family, loads): compute
            for length, compute, family, loads in (
                previous.caller_cp_peaks
                if isinstance(previous, _CallerMemoryProfile)
                else ()
            )
        }
        cp_workload = getattr(plan, "caller_cp_workload", None)
        if caller_phase and plan.signature.topology[2] > 1 and cp_workload is not None:
            family, loads = cp_workload
            key = (plan.packed_tokens, family, loads)
            cp_peaks[key] = max(compute_delta, cp_peaks.get(key, 0))
        cp_observations = sorted(
            (length, compute, family, loads)
            for (length, family, loads), compute in cp_peaks.items()
        )
        # Never merge different ownership vectors into a synthetic measurement.
        # Evicting evidence only removes permission to discount, not the shared
        # max rate or floor. Bound storage like the CP1 curve.
        if len(cp_observations) > 8:
            cp_observations = [cp_observations[0], *cp_observations[-7:]]
        caller_cls = _CallerMemoryProfileV2 if fixed_slope else _CallerMemoryProfile
        self._memory_profiles[plan.signature] = caller_cls(
            **vars(self._memory_profiles[plan.signature]),
            caller_compute_peaks=tuple(observations),
            caller_cp_peaks=tuple(cp_observations),
            compute_peak_floor=max(
                compute_delta,
                previous.compute_peak_floor
                if isinstance(previous, _CallerMemoryProfile)
                else 0,
            ),
        )
    if (
        isinstance(plan.signature, _ScopedMemorySignature)
        and plan.signature.bounded_forward_peak
    ):
        peaks = dict(
            previous.forward_compute_peaks
            if isinstance(previous, _BoundedMemoryProfile)
            else ()
        )
        peaks[plan.packed_tokens] = max(compute_delta, peaks.get(plan.packed_tokens, 0))
        observations = []
        peak = 0
        for length, compute in sorted(peaks.items()):
            peak = max(peak, compute)
            observations.append((length, peak))
        if len(observations) > 8:
            observations = [observations[0], *observations[-7:]]
        bounded_cls = _BoundedMemoryProfileV2 if fixed_slope else _BoundedMemoryProfile
        self._memory_profiles[plan.signature] = bounded_cls(
            **vars(self._memory_profiles[plan.signature]),
            forward_compute_peaks=tuple(observations),
        )
