from __future__ import annotations

import base64
import io
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
MISSING_EXPERT_ID = -1


def _has_routing_experts(metadata: dict[str, Any]) -> bool:
    return any(metadata.get(key) is not None for key in _ROUTING_EXPERT_KEYS)


class MoeRoutingAlignmentStats(BaseModel):
    choices_with_routing: int = 0
    routed_tokens: int = 0


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
            prompt_token_count=len(prompt_token_ids),
            completion_token_count=len(completion_token_ids),
            route_decode_cache=route_decode_cache,
        )
        expected_prompt_ids = token_ids[:offset]
        expected_completion_ids = token_ids[offset : offset + token_length]
        if prompt_token_ids != expected_prompt_ids:
            raise RuntimeError(
                "vLLM routed prompt token ids do not match ART-tokenized prefix: "
                f"offset={offset}, vllm_len={len(prompt_token_ids)}, "
                f"art_len={len(expected_prompt_ids)}"
            )
        if completion_token_ids != expected_completion_ids:
            raise RuntimeError(
                "vLLM routed completion token ids do not match ART-tokenized choice: "
                f"offset={offset}, vllm_len={len(completion_token_ids)}, "
                f"art_len={len(expected_completion_ids)}"
            )
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
        ) = _append_or_overlay_routes(
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
        ) = _append_or_overlay_routes(
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
        if dtype.hasobject:
            raise RuntimeError(f"{field_name} cannot contain object dtype routes")
        array = np.frombuffer(
            payload,
            dtype=dtype,
            count=int(np.prod(shape)),
            offset=stream.tell(),
        ).reshape(shape, order="F" if fortran_order else "C")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {field_name} as base64 .npy") from exc
    array = np.asarray(array, dtype=np.int32)
    _validate_route_array(array, field_name=field_name)
    array.flags.writeable = False
    return array


def _route_cache_key(raw: str) -> MoeRouteCacheKey:
    return len(raw), raw[:96], raw[-96:]


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
    prompt_token_count: int,
    completion_token_count: int,
    route_decode_cache: MoeRouteDecodeCache | None = None,
) -> tuple[MoeRouteArray, MoeRouteArray]:
    if PROMPT_ROUTED_EXPERTS_KEY in metadata:
        return (
            _normalize_routes(
                metadata.get(PROMPT_ROUTED_EXPERTS_KEY),
                field_name=PROMPT_ROUTED_EXPERTS_KEY,
                route_decode_cache=route_decode_cache,
            ),
            _completion_routes(metadata, route_decode_cache=route_decode_cache),
        )

    routes = _normalize_routes(
        metadata.get(ROUTED_EXPERTS_KEY),
        field_name=ROUTED_EXPERTS_KEY,
        route_decode_cache=route_decode_cache,
    )
    expected_lengths = {
        prompt_token_count + completion_token_count,
        prompt_token_count + max(completion_token_count - 1, 0),
    }
    if len(routes) not in expected_lengths:
        raise RuntimeError(
            "routed_experts length does not match prompt/completion token ids: "
            f"{len(routes)} not in {sorted(expected_lengths)}"
        )
    return routes[:prompt_token_count], routes[prompt_token_count:]


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
