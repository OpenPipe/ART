from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

import numpy as np
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, model_validator

ART_MOE_ROUTING_METADATA_KEY = "art_moe_routing"

PROMPT_TOKEN_IDS_KEY = "prompt_token_ids"
COMPLETION_TOKEN_IDS_KEY = "completion_token_ids"
PROMPT_ROUTED_EXPERTS_KEY = "prompt_routed_experts"
COMPLETION_ROUTED_EXPERTS_KEY = "completion_routed_experts"
ROUTED_EXPERTS_KEY = "routed_experts"

_ROUTING_RESPONSE_KEYS = {
    PROMPT_TOKEN_IDS_KEY,
    COMPLETION_TOKEN_IDS_KEY,
    "output_token_ids",
    "token_ids",
    PROMPT_ROUTED_EXPERTS_KEY,
    COMPLETION_ROUTED_EXPERTS_KEY,
    ROUTED_EXPERTS_KEY,
}
_ROUTING_EXPERT_KEYS = {
    PROMPT_ROUTED_EXPERTS_KEY,
    COMPLETION_ROUTED_EXPERTS_KEY,
    ROUTED_EXPERTS_KEY,
}

TokenRoute = list[list[int]]
MoeRouteArray = np.ndarray
MoeRouteCacheKey = tuple[int, str, str]
MoeRouteDecodeCache = dict[MoeRouteCacheKey, list[tuple[str, MoeRouteArray]]]
MoePromptRouteCacheKey = tuple[
    tuple[int, ...],
    tuple[int, str, str, str] | tuple[str, tuple[Any, ...] | None],
]
MoePromptRouteCache = dict[MoePromptRouteCacheKey, list[MoeRouteArray]]
MISSING_EXPERT_ID = -1


def _has_routing_experts(metadata: dict[str, Any]) -> bool:
    return any(metadata.get(key) is not None for key in _ROUTING_EXPERT_KEYS)


class MoeRoutingAlignmentStats(BaseModel):
    choices_with_routing: int = 0
    routed_tokens: int = 0
    prompt_route_cache_hits: int = 0
    prompt_route_cache_misses: int = 0
    prompt_route_bytes: int = 0
    completion_route_bytes: int = 0
    decode_prompt_routes_s: float = 0.0
    decode_completion_routes_s: float = 0.0
    token_id_validation_s: float = 0.0
    prompt_route_validation_s: float = 0.0
    append_overlay_s: float = 0.0


class MoeRoutingPackStats(BaseModel):
    packed_tokens: int = 0


class MoeRouteSegments(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    segments: tuple[MoeRouteArray, ...]

    @property
    def shape(self) -> tuple[int, int, int]:
        first = self.segments[0]
        return (
            sum(segment.shape[0] for segment in self.segments),
            first.shape[1],
            first.shape[2],
        )

    def iter_slices(
        self, start: int, end: int
    ) -> tuple[tuple[int, MoeRouteArray], ...]:
        slices: list[tuple[int, MoeRouteArray]] = []
        offset = 0
        for segment in self.segments:
            segment_end = offset + segment.shape[0]
            overlap_start = max(start, offset)
            overlap_end = min(end, segment_end)
            if overlap_start < overlap_end:
                slices.append(
                    (
                        overlap_start,
                        segment[overlap_start - offset : overlap_end - offset],
                    )
                )
            offset = segment_end
            if offset >= end:
                break
        return tuple(slices)


class PackedMoeRoutingReplay(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    expert_indices: Any
    token_mask: Any
    num_layers: int
    topk: int
    num_experts: int
    pack_stats: MoeRoutingPackStats

    @model_validator(mode="after")
    def _validate(self) -> "PackedMoeRoutingReplay":
        if self.expert_indices.ndim != 4:
            raise RuntimeError(
                "expert_indices must have shape "
                "[num_sequences, sequence_length, num_layers, topk], got "
                f"{tuple(self.expert_indices.shape)}"
            )
        if self.token_mask.shape != self.expert_indices.shape[:2]:
            raise RuntimeError(
                "token_mask shape must match packed route tokens, got "
                f"{tuple(self.token_mask.shape)} vs "
                f"{tuple(self.expert_indices.shape[:2])}"
            )
        if self.num_layers != int(self.expert_indices.shape[2]):
            raise RuntimeError(
                f"num_layers={self.num_layers} does not match "
                f"expert_indices.shape[2]={self.expert_indices.shape[2]}"
            )
        if self.topk != int(self.expert_indices.shape[3]):
            raise RuntimeError(
                f"topk={self.topk} does not match "
                f"expert_indices.shape[3]={self.expert_indices.shape[3]}"
            )
        if self.num_experts <= 0:
            raise RuntimeError(f"num_experts must be >0, got {self.num_experts}")
        if self.topk > self.num_experts:
            raise RuntimeError(
                f"MoE routing topk cannot exceed num_experts: topk={self.topk}, "
                f"num_experts={self.num_experts}"
            )
        return self


def attach_moe_routing_metadata_to_choice(
    *,
    choice: Choice,
    response_payload: dict[str, Any],
    choice_index: int = 0,
    routed_experts: MoeRouteArray | None = None,
) -> None:
    metadata: dict[str, Any] = {
        key: response_payload[key]
        for key in _ROUTING_RESPONSE_KEYS
        if key in response_payload
    }
    raw_choices = response_payload.get("choices")
    if isinstance(raw_choices, list) and choice_index < len(raw_choices):
        raw_choice = raw_choices[choice_index]
        if isinstance(raw_choice, dict):
            metadata.update(
                {
                    key: raw_choice[key]
                    for key in _ROUTING_RESPONSE_KEYS
                    if key in raw_choice
                }
            )
    if routed_experts is not None:
        metadata[ROUTED_EXPERTS_KEY] = routed_experts
    if not metadata or not _has_routing_experts(metadata):
        return
    extra = choice.model_extra
    if extra is None:
        raise RuntimeError("OpenAI Choice.model_extra is unavailable for route capture")
    extra[ART_MOE_ROUTING_METADATA_KEY] = metadata


def choice_moe_routing_metadata(choice: Choice) -> dict[str, Any] | None:
    extra = choice.model_extra or {}
    nested = extra.get(ART_MOE_ROUTING_METADATA_KEY)
    if isinstance(nested, dict):
        if not _has_routing_experts(nested):
            return None
        return nested
    top_level = {key: extra[key] for key in _ROUTING_RESPONSE_KEYS if key in extra}
    if not _has_routing_experts(top_level):
        return None
    return top_level or None


def align_choice_routes_to_tokenized_result(
    *,
    token_ids: list[int],
    choices: list[Choice],
    choice_offsets: list[int],
    choice_token_lengths: list[int],
    route_decode_cache: MoeRouteDecodeCache | None = None,
    prompt_route_cache: MoePromptRouteCache | None = None,
) -> tuple[MoeRouteArray | MoeRouteSegments | None, MoeRoutingAlignmentStats]:
    if not (len(choices) == len(choice_offsets) == len(choice_token_lengths)):
        raise RuntimeError(
            "Choice routing alignment inputs differ in length: "
            f"choices={len(choices)}, offsets={len(choice_offsets)}, "
            f"lengths={len(choice_token_lengths)}"
        )
    aligned: MoeRouteArray | None = None
    route_mask: np.ndarray | None = None
    route_segments: list[MoeRouteArray] = []
    route_shape: tuple[int, int] | None = None
    covered_until = 0
    stats = MoeRoutingAlignmentStats()
    saw_routing = False
    saw_missing = False
    for choice, offset, token_length in zip(
        choices, choice_offsets, choice_token_lengths
    ):
        metadata = choice_moe_routing_metadata(choice)
        if metadata is None:
            saw_missing = True
            continue
        saw_routing = True
        stats.choices_with_routing += 1
        prompt_token_ids = _normalize_token_ids(metadata.get(PROMPT_TOKEN_IDS_KEY))
        completion_token_ids = _completion_token_ids(metadata)
        prompt_routes, completion_routes = _choice_routes(
            metadata,
            prompt_token_ids=prompt_token_ids,
            completion_token_count=len(completion_token_ids),
            route_decode_cache=route_decode_cache,
            prompt_route_cache=prompt_route_cache,
            prompt_policy_cache_key=_prompt_policy_cache_key(choice),
            stats=stats,
        )
        timing_start = _route_alignment_time_ns()
        if prompt_token_ids != token_ids[:offset]:
            raise RuntimeError(
                "vLLM routed prompt token ids do not match ART-tokenized prefix: "
                f"offset={offset}, vllm_len={len(prompt_token_ids)}, "
                f"art_len={offset}"
            )
        if completion_token_ids != token_ids[offset : offset + token_length]:
            raise RuntimeError(
                "vLLM routed completion token ids do not match ART-tokenized choice: "
                f"offset={offset}, vllm_len={len(completion_token_ids)}, "
                f"art_len={token_length}"
            )
        _add_route_alignment_elapsed(stats, "token_id_validation_s", timing_start)
        if prompt_routes.shape[0] != len(prompt_token_ids):
            raise RuntimeError(
                "prompt_routed_experts length does not match prompt_token_ids: "
                f"{prompt_routes.shape[0]} != {len(prompt_token_ids)}"
            )
        if completion_routes.shape[0] not in {
            len(completion_token_ids),
            max(len(completion_token_ids) - 1, 0),
        }:
            raise RuntimeError(
                "completion_routed_experts length does not match completion_token_ids: "
                f"{completion_routes.shape[0]} != {len(completion_token_ids)}"
            )
        current_shape = _common_route_shape(prompt_routes, completion_routes)
        if route_shape is None:
            route_shape = current_shape
        elif route_shape != current_shape:
            raise RuntimeError("MoE route arrays must have one rectangular shape")
        (
            aligned,
            route_mask,
            covered_until,
        ) = _timed_append_or_overlay_routes(
            stats=stats,
            aligned=aligned,
            route_mask=route_mask,
            route_segments=route_segments,
            covered_until=covered_until,
            token_count=len(token_ids),
            route_shape=route_shape,
            start=0,
            routes=prompt_routes,
        )
        (
            aligned,
            route_mask,
            covered_until,
        ) = _timed_append_or_overlay_routes(
            stats=stats,
            aligned=aligned,
            route_mask=route_mask,
            route_segments=route_segments,
            covered_until=covered_until,
            token_count=len(token_ids),
            route_shape=route_shape,
            start=offset,
            routes=completion_routes,
        )
        stats.routed_tokens = (
            int(route_mask.sum()) if route_mask is not None else covered_until
        )
    if saw_routing and saw_missing:
        raise RuntimeError("Some trainable choices had MoE routes while others did not")
    if not saw_routing:
        return None, stats
    if aligned is not None:
        return aligned, stats
    if covered_until == len(token_ids):
        if len(route_segments) == 1:
            return route_segments[0], stats
        return MoeRouteSegments(segments=tuple(route_segments)), stats
    if route_shape is None:
        raise RuntimeError("MoE routing metadata did not contain any routed tokens")
    aligned, route_mask = _materialize_route_segments(
        token_count=len(token_ids),
        route_shape=route_shape,
        route_segments=route_segments,
    )
    stats.routed_tokens = int(route_mask.sum())
    return aligned, stats


def _timed_append_or_overlay_routes(
    *,
    stats: MoeRoutingAlignmentStats,
    aligned: MoeRouteArray | None,
    route_mask: np.ndarray | None,
    route_segments: list[MoeRouteArray],
    covered_until: int,
    token_count: int,
    route_shape: tuple[int, int],
    start: int,
    routes: MoeRouteArray,
) -> tuple[MoeRouteArray | None, np.ndarray | None, int]:
    timing_start = _route_alignment_time_ns()
    try:
        return _append_or_overlay_routes(
            aligned=aligned,
            route_mask=route_mask,
            route_segments=route_segments,
            covered_until=covered_until,
            token_count=token_count,
            route_shape=route_shape,
            start=start,
            routes=routes,
        )
    finally:
        _add_route_alignment_elapsed(stats, "append_overlay_s", timing_start)


def _append_or_overlay_routes(
    *,
    aligned: MoeRouteArray | None,
    route_mask: np.ndarray | None,
    route_segments: list[MoeRouteArray],
    covered_until: int,
    token_count: int,
    route_shape: tuple[int, int],
    start: int,
    routes: MoeRouteArray,
) -> tuple[MoeRouteArray | None, np.ndarray | None, int]:
    if routes.shape[0] == 0:
        return aligned, route_mask, covered_until
    if aligned is None and start == covered_until:
        route_segments.append(routes)
        return aligned, route_mask, covered_until + routes.shape[0]
    if aligned is None:
        aligned, route_mask = _materialize_route_segments(
            token_count=token_count,
            route_shape=route_shape,
            route_segments=route_segments,
        )
    assert route_mask is not None
    _overlay_routes(aligned, route_mask, start, routes)
    return aligned, route_mask, covered_until


def _materialize_route_segments(
    *,
    token_count: int,
    route_shape: tuple[int, int],
    route_segments: list[MoeRouteArray],
) -> tuple[MoeRouteArray, np.ndarray]:
    num_layers, topk = route_shape
    aligned = np.full(
        (token_count, num_layers, topk),
        MISSING_EXPERT_ID,
        dtype=np.int32,
    )
    route_mask = np.zeros(token_count, dtype=np.bool_)
    offset = 0
    for routes in route_segments:
        _overlay_routes(aligned, route_mask, offset, routes)
        offset += routes.shape[0]
    return aligned, route_mask


def _overlay_routes(
    aligned: MoeRouteArray,
    route_mask: np.ndarray,
    start: int,
    routes: MoeRouteArray,
) -> None:
    if routes.shape[0] == 0:
        return
    end = start + routes.shape[0]
    existing = route_mask[start:end]
    fill = ~existing
    if bool(fill.any()):
        aligned[start:end][fill] = routes[fill]
        existing[fill] = True


def _normalize_token_ids(raw: Any) -> list[int]:
    if raw is None:
        raise RuntimeError("Missing routed token ids")
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected routed token ids list, got {type(raw)}")
    return [int(token_id) for token_id in raw]


def _normalize_routes(
    raw: Any,
    *,
    field_name: str,
    route_decode_cache: MoeRouteDecodeCache | None = None,
) -> MoeRouteArray:
    if isinstance(raw, np.ndarray):
        _validate_route_array(raw, field_name=field_name)
        raw.flags.writeable = False
        return raw
    if isinstance(raw, str):
        if route_decode_cache is not None:
            key = _route_cache_key(raw)
            for cached_raw, cached_array in route_decode_cache.get(key, []):
                if cached_raw == raw:
                    return cached_array
        array = _decode_vllm_routed_experts(raw, field_name=field_name)
        if route_decode_cache is not None:
            route_decode_cache.setdefault(key, []).append((raw, array))
        return array
    if raw is None:
        raise RuntimeError(f"Missing {field_name}")
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected {field_name} list, got {type(raw)}")
    if len(raw) == 0:
        return np.empty((0, 0, 0), dtype=np.int32)
    array = np.asarray(raw, dtype=np.int32)
    _validate_route_array(array, field_name=field_name)
    return array


def _decode_vllm_routed_experts(raw: str, *, field_name: str) -> MoeRouteArray:
    try:
        payload = base64.b64decode(raw)
        stream = io.BytesIO(payload)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        if dtype.hasobject or not np.issubdtype(dtype, np.integer):
            raise RuntimeError(f"{field_name} must contain integer routes")
        array = np.frombuffer(
            payload,
            dtype=dtype,
            count=int(np.prod(shape)),
            offset=stream.tell(),
        ).reshape(shape, order="F" if fortran_order else "C")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {field_name} as base64 .npy") from exc
    _validate_route_array(array, field_name=field_name)
    array.flags.writeable = False
    return array


def _route_cache_key(raw: str) -> MoeRouteCacheKey:
    return len(raw), raw[:96], raw[-96:]


def _prompt_policy_cache_key(choice: Choice) -> tuple[Any, ...] | None:
    extra = choice.model_extra or {}
    spans = extra.get("policy_token_spans")
    if not isinstance(spans, list) or not spans:
        return None
    first = spans[0]
    if not isinstance(first, dict):
        return None
    return (
        int(first.get("policy_version", 0)),
        str(first.get("lora_slot", "")),
        int(first.get("update_seq", 0)),
    )


def _prompt_route_cache_key(
    *,
    prompt_token_ids: list[int],
    raw: Any,
) -> MoePromptRouteCacheKey | None:
    if not isinstance(raw, str):
        return None
    middle = len(raw) // 2
    route_signature = (len(raw), raw[:96], raw[middle : middle + 96], raw[-96:])
    return tuple(prompt_token_ids), route_signature


def _validate_route_array(array: MoeRouteArray, *, field_name: str) -> None:
    if array.ndim != 3:
        raise RuntimeError(
            f"Expected {field_name} array with rank 3, got shape {array.shape}"
        )
    if array.shape[0] > 0 and (array.shape[1] <= 0 or array.shape[2] <= 0):
        raise RuntimeError(f"{field_name} must have non-empty layer and topk axes")


def _common_route_shape(*arrays: MoeRouteArray) -> tuple[int, int]:
    shape: tuple[int, int] | None = None
    for array in arrays:
        if array.shape[0] == 0:
            continue
        candidate = (int(array.shape[1]), int(array.shape[2]))
        if shape is None:
            shape = candidate
        elif shape != candidate:
            raise RuntimeError("MoE route arrays must have one rectangular shape")
    if shape is None:
        raise RuntimeError("MoE routing metadata did not contain any routed tokens")
    return shape


def _completion_token_ids(metadata: dict[str, Any]) -> list[int]:
    for key in (COMPLETION_TOKEN_IDS_KEY, "output_token_ids", "token_ids"):
        if key in metadata:
            return _normalize_token_ids(metadata[key])
    raise RuntimeError("Missing routed completion token ids")


def _choice_routes(
    metadata: dict[str, Any],
    *,
    prompt_token_ids: list[int],
    completion_token_count: int,
    route_decode_cache: MoeRouteDecodeCache | None = None,
    prompt_route_cache: MoePromptRouteCache | None = None,
    prompt_policy_cache_key: tuple[Any, ...] | None = None,
    stats: MoeRoutingAlignmentStats | None = None,
) -> tuple[MoeRouteArray, MoeRouteArray]:
    if PROMPT_ROUTED_EXPERTS_KEY in metadata:
        completion_timing_start = _route_alignment_time_ns()
        completion_routes = _completion_routes(
            metadata, route_decode_cache=route_decode_cache
        )
        if stats is not None:
            stats.completion_route_bytes += _route_payload_size(
                _completion_routes_payload(metadata)
            )
            _add_route_alignment_elapsed(
                stats, "decode_completion_routes_s", completion_timing_start
            )
        return (
            _prompt_routes(
                metadata,
                prompt_token_ids=prompt_token_ids,
                route_decode_cache=route_decode_cache,
                prompt_route_cache=prompt_route_cache,
                stats=stats,
            ),
            completion_routes,
        )

    timing_start = _route_alignment_time_ns()
    routes = _normalize_routes(
        metadata.get(ROUTED_EXPERTS_KEY),
        field_name=ROUTED_EXPERTS_KEY,
        route_decode_cache=route_decode_cache,
    )
    if stats is not None:
        route_bytes = _route_payload_size(metadata.get(ROUTED_EXPERTS_KEY))
        stats.prompt_route_bytes += route_bytes
        stats.completion_route_bytes += route_bytes
        _add_route_alignment_elapsed(stats, "decode_prompt_routes_s", timing_start)
    expected_lengths = {
        len(prompt_token_ids) + completion_token_count,
        len(prompt_token_ids) + max(completion_token_count - 1, 0),
    }
    if len(routes) not in expected_lengths:
        raise RuntimeError(
            "routed_experts length does not match prompt/completion token ids: "
            f"{len(routes)} not in {sorted(expected_lengths)}"
        )
    prompt_routes = _combined_prompt_routes(
        routes=routes,
        prompt_token_ids=prompt_token_ids,
        prompt_route_cache=prompt_route_cache,
        prompt_policy_cache_key=prompt_policy_cache_key,
        stats=stats,
    )
    completion_routes = _readonly_route_copy(routes[len(prompt_token_ids) :])
    return prompt_routes, completion_routes


def _prompt_routes(
    metadata: dict[str, Any],
    *,
    prompt_token_ids: list[int],
    route_decode_cache: MoeRouteDecodeCache | None = None,
    prompt_route_cache: MoePromptRouteCache | None = None,
    stats: MoeRoutingAlignmentStats | None = None,
) -> MoeRouteArray:
    timing_start = _route_alignment_time_ns()
    raw = metadata.get(PROMPT_ROUTED_EXPERTS_KEY)
    try:
        if stats is not None:
            stats.prompt_route_bytes += _route_payload_size(raw)
        cache_key = (
            _prompt_route_cache_key(prompt_token_ids=prompt_token_ids, raw=raw)
            if prompt_route_cache is not None
            else None
        )
        if cache_key is not None and prompt_route_cache is not None:
            cached = _single_cached_prompt_route(prompt_route_cache, cache_key)
            if cached is not None:
                if stats is not None:
                    stats.prompt_route_cache_hits += 1
                return cached
            if stats is not None:
                stats.prompt_route_cache_misses += 1
        routes = _normalize_routes(
            raw,
            field_name=PROMPT_ROUTED_EXPERTS_KEY,
            route_decode_cache=route_decode_cache,
        )
        if cache_key is not None and prompt_route_cache is not None:
            prompt_route_cache[cache_key] = [routes]
        return routes
    finally:
        if stats is not None:
            _add_route_alignment_elapsed(stats, "decode_prompt_routes_s", timing_start)


def _single_cached_prompt_route(
    prompt_route_cache: MoePromptRouteCache,
    cache_key: MoePromptRouteCacheKey,
) -> MoeRouteArray | None:
    cached = prompt_route_cache.get(cache_key)
    if not cached:
        return None
    return cached[0]


def _combined_prompt_routes(
    *,
    routes: MoeRouteArray,
    prompt_token_ids: list[int],
    prompt_route_cache: MoePromptRouteCache | None,
    prompt_policy_cache_key: tuple[Any, ...] | None,
    stats: MoeRoutingAlignmentStats | None,
) -> MoeRouteArray:
    prompt_routes = routes[: len(prompt_token_ids)]
    if prompt_route_cache is None:
        return _readonly_route_copy(prompt_routes)
    cache_key: MoePromptRouteCacheKey = (
        tuple(prompt_token_ids),
        ("combined", prompt_policy_cache_key),
    )
    timing_start = _route_alignment_time_ns()
    try:
        for cached in prompt_route_cache.get(cache_key, []):
            if np.array_equal(cached, prompt_routes):
                if stats is not None:
                    stats.prompt_route_cache_hits += 1
                return cached
        if stats is not None:
            stats.prompt_route_cache_misses += 1
        cached_prompt = _readonly_route_copy(prompt_routes)
        prompt_route_cache.setdefault(cache_key, []).append(cached_prompt)
        return cached_prompt
    finally:
        if stats is not None:
            _add_route_alignment_elapsed(
                stats, "prompt_route_validation_s", timing_start
            )


def _readonly_route_copy(routes: MoeRouteArray) -> MoeRouteArray:
    copied = np.array(routes, copy=True)
    copied.flags.writeable = False
    return copied


def _completion_routes(
    metadata: dict[str, Any],
    *,
    route_decode_cache: MoeRouteDecodeCache | None = None,
) -> MoeRouteArray:
    if COMPLETION_ROUTED_EXPERTS_KEY in metadata:
        return _normalize_routes(
            metadata[COMPLETION_ROUTED_EXPERTS_KEY],
            field_name=COMPLETION_ROUTED_EXPERTS_KEY,
            route_decode_cache=route_decode_cache,
        )
    if ROUTED_EXPERTS_KEY in metadata:
        return _normalize_routes(
            metadata[ROUTED_EXPERTS_KEY],
            field_name=ROUTED_EXPERTS_KEY,
            route_decode_cache=route_decode_cache,
        )
    raise RuntimeError("Missing routed completion experts")


def _completion_routes_payload(metadata: dict[str, Any]) -> Any:
    if COMPLETION_ROUTED_EXPERTS_KEY in metadata:
        return metadata[COMPLETION_ROUTED_EXPERTS_KEY]
    return metadata.get(ROUTED_EXPERTS_KEY)


def _route_payload_size(raw: Any) -> int:
    if isinstance(raw, np.ndarray):
        return int(raw.nbytes)
    if isinstance(raw, str):
        return len(raw)
    if isinstance(raw, list):
        return len(raw)
    return 0


def _route_alignment_time_ns() -> int:
    return (
        time.perf_counter_ns()
        if os.environ.get("ART_PROFILE_MOE_ROUTE_ALIGNMENT") == "1"
        else 0
    )


def _add_route_alignment_elapsed(
    stats: MoeRoutingAlignmentStats, field_name: str, start_ns: int
) -> None:
    if start_ns == 0:
        return
    setattr(
        stats,
        field_name,
        float(getattr(stats, field_name)) + (time.perf_counter_ns() - start_ns) / 1e9,
    )
