from __future__ import annotations

import os
import time
from typing import Any, cast

import numpy as np
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, model_validator

from ..openai import ART_MOE_ROUTING_METADATA_KEY

PROMPT_TOKEN_IDS_KEY = "prompt_token_ids"
COMPLETION_TOKEN_IDS_KEY = "completion_token_ids"
ROUTED_EXPERTS_KEY = "routed_experts"
NUM_EXPERTS_KEY = "num_experts"


class MoeRouteArray(np.ndarray):
    num_experts: int

    def __new__(
        cls,
        array: np.ndarray,
        *,
        num_experts: int,
        validate: bool = True,
    ) -> "MoeRouteArray":
        result = np.asarray(array).view(cls)
        result.num_experts = int(num_experts)
        if validate:
            _validate_route_array(result, field_name=ROUTED_EXPERTS_KEY)
        result.flags.writeable = False
        return result

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        self.num_experts = int(getattr(source, "num_experts", 0))


def moe_route_dtype(num_experts: int) -> np.dtype[Any]:
    if not 1 <= num_experts <= 65_536:
        raise RuntimeError(
            f"MoE routing requires num_experts in [1, 65536], got {num_experts}"
        )
    return np.dtype(np.uint8 if num_experts <= 256 else np.uint16)


class MoeRoutingAlignmentStats(BaseModel):
    choices_with_routing: int = 0
    routed_tokens: int = 0
    prompt_route_bytes: int = 0
    completion_route_bytes: int = 0
    token_id_validation_s: float = 0.0
    append_overlay_s: float = 0.0


class MoeRoutingPackStats(BaseModel):
    packed_tokens: int = 0


class MoeRouteSegments(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    segments: tuple[MoeRouteArray, ...]

    @model_validator(mode="after")
    def _validate(self) -> "MoeRouteSegments":
        if not self.segments:
            raise RuntimeError("MoE route segments cannot be empty")
        contract = {
            (segment.num_experts, segment.dtype, *segment.shape[1:])
            for segment in self.segments
        }
        if len(contract) != 1:
            raise RuntimeError("MoE route segments must share one exact contract")
        return self

    @property
    def num_experts(self) -> int:
        return self.segments[0].num_experts

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
                        cast(
                            MoeRouteArray,
                            segment[overlap_start - offset : overlap_end - offset],
                        ),
                    )
                )
            offset = segment_end
            if offset >= end:
                break
        return tuple(slices)


class PackedMoeRoutingReplay(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    expert_indices: Any
    num_experts: int
    pack_stats: MoeRoutingPackStats

    @model_validator(mode="after")
    def _validate(self) -> "PackedMoeRoutingReplay":
        if self.expert_indices.ndim != 4:
            raise RuntimeError(
                "expert_indices must have shape "
                "[num_layers, num_sequences, sequence_length, topk], got "
                f"{tuple(self.expert_indices.shape)}"
            )
        if min(map(int, self.expert_indices.shape)) <= 0:
            raise RuntimeError("expert_indices axes must be non-empty")
        expected_dtype = str(moe_route_dtype(self.num_experts))
        actual_dtype = str(self.expert_indices.dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise RuntimeError(
                f"{self.num_experts} experts require {expected_dtype} replay ids, "
                f"got {actual_dtype}"
            )
        if self.topk > self.num_experts:
            raise RuntimeError(
                f"MoE routing topk cannot exceed num_experts: topk={self.topk}, "
                f"num_experts={self.num_experts}"
            )
        return self

    @property
    def num_layers(self) -> int:
        return int(self.expert_indices.shape[0])

    @property
    def topk(self) -> int:
        return int(self.expert_indices.shape[3])


def attach_moe_routing_metadata_to_choice(
    *,
    choice: Choice,
    response_payload: dict[str, Any],
    choice_index: int = 0,
    routed_experts: np.ndarray | None = None,
    num_experts: int | None = None,
) -> None:
    if routed_experts is None:
        return
    num_experts = int(num_experts or getattr(routed_experts, "num_experts", 0))
    routes = MoeRouteArray(routed_experts, num_experts=num_experts)
    metadata: dict[str, Any] = {
        PROMPT_TOKEN_IDS_KEY: response_payload.get(PROMPT_TOKEN_IDS_KEY),
        ROUTED_EXPERTS_KEY: routes,
        NUM_EXPERTS_KEY: num_experts,
    }
    raw_choices = response_payload.get("choices")
    if isinstance(raw_choices, list) and choice_index < len(raw_choices):
        raw_choice = raw_choices[choice_index]
        if isinstance(raw_choice, dict):
            metadata[COMPLETION_TOKEN_IDS_KEY] = next(
                (
                    raw_choice[key]
                    for key in (
                        COMPLETION_TOKEN_IDS_KEY,
                        "output_token_ids",
                        "token_ids",
                    )
                    if key in raw_choice
                ),
                None,
            )
    _normalize_token_ids(metadata[PROMPT_TOKEN_IDS_KEY])
    _normalize_token_ids(metadata.get(COMPLETION_TOKEN_IDS_KEY))
    extra = choice.model_extra
    if extra is None:
        raise RuntimeError("OpenAI Choice.model_extra is unavailable for route capture")
    extra[ART_MOE_ROUTING_METADATA_KEY] = metadata


def choice_moe_routing_metadata(choice: Choice) -> dict[str, Any] | None:
    extra = choice.model_extra or {}
    nested = extra.get(ART_MOE_ROUTING_METADATA_KEY)
    if not isinstance(nested, dict):
        return None
    return nested if isinstance(nested.get(ROUTED_EXPERTS_KEY), np.ndarray) else None


def align_choice_routes_to_tokenized_result(
    *,
    token_ids: list[int],
    choices: list[Choice],
    choice_offsets: list[int],
    choice_token_lengths: list[int],
) -> tuple[MoeRouteArray | MoeRouteSegments | None, MoeRoutingAlignmentStats]:
    if not (len(choices) == len(choice_offsets) == len(choice_token_lengths)):
        raise RuntimeError(
            "Choice routing alignment inputs differ in length: "
            f"choices={len(choices)}, offsets={len(choice_offsets)}, "
            f"lengths={len(choice_token_lengths)}"
        )
    aligned: np.ndarray | None = None
    route_mask: np.ndarray | None = None
    route_segments: list[MoeRouteArray] = []
    route_shape: tuple[int, int] | None = None
    num_experts: int | None = None
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
            stats=stats,
        )
        if num_experts is None:
            num_experts = prompt_routes.num_experts
        elif num_experts != prompt_routes.num_experts:
            raise RuntimeError("MoE route captures disagree on exact expert count")
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
                "Binary prompt route length does not match prompt_token_ids: "
                f"{prompt_routes.shape[0]} != {len(prompt_token_ids)}"
            )
        if completion_routes.shape[0] not in {
            len(completion_token_ids),
            max(len(completion_token_ids) - 1, 0),
        }:
            raise RuntimeError(
                "Binary completion route length does not match completion_token_ids: "
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
    if num_experts is None:
        raise RuntimeError("MoE routing metadata omitted exact expert count")
    if aligned is not None:
        assert route_mask is not None
        _fill_missing_routes(aligned, route_mask, num_experts=num_experts)
        return MoeRouteArray(aligned, num_experts=num_experts), stats
    if covered_until == len(token_ids):
        if len(route_segments) == 1:
            return route_segments[0], stats
        return MoeRouteSegments(segments=tuple(route_segments)), stats
    if route_shape is None:
        raise RuntimeError("MoE routing metadata did not contain any routed tokens")
    missing = deterministic_moe_routes(
        np.arange(covered_until, len(token_ids), dtype=np.int64),
        route_shape=route_shape,
        num_experts=num_experts,
    )
    route_segments.append(missing)
    stats.routed_tokens = covered_until
    return MoeRouteSegments(segments=tuple(route_segments)), stats


def _timed_append_or_overlay_routes(
    *,
    stats: MoeRoutingAlignmentStats,
    aligned: np.ndarray | None,
    route_mask: np.ndarray | None,
    route_segments: list[MoeRouteArray],
    covered_until: int,
    token_count: int,
    route_shape: tuple[int, int],
    start: int,
    routes: MoeRouteArray,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
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
    aligned: np.ndarray | None,
    route_mask: np.ndarray | None,
    route_segments: list[MoeRouteArray],
    covered_until: int,
    token_count: int,
    route_shape: tuple[int, int],
    start: int,
    routes: MoeRouteArray,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
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
) -> tuple[np.ndarray, np.ndarray]:
    num_layers, topk = route_shape
    dtype = route_segments[0].dtype if route_segments else np.dtype(np.uint8)
    aligned = np.zeros((token_count, num_layers, topk), dtype=dtype)
    route_mask = np.zeros(token_count, dtype=np.bool_)
    offset = 0
    for routes in route_segments:
        _overlay_routes(aligned, route_mask, offset, routes)
        offset += routes.shape[0]
    return aligned, route_mask


def _overlay_routes(
    aligned: np.ndarray,
    route_mask: np.ndarray,
    start: int,
    routes: MoeRouteArray,
) -> None:
    if routes.shape[0] == 0:
        return
    end = start + routes.shape[0]
    existing = route_mask[start:end]
    if bool(existing.any()) and not np.array_equal(
        aligned[start:end][existing], routes[existing]
    ):
        raise RuntimeError("Overlapping routed experts disagree for the same token")
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


def _validate_route_array(array: MoeRouteArray, *, field_name: str) -> None:
    if array.ndim != 3:
        raise RuntimeError(
            f"Expected {field_name} array with rank 3, got shape {array.shape}"
        )
    if array.shape[0] > 0 and (array.shape[1] <= 0 or array.shape[2] <= 0):
        raise RuntimeError(f"{field_name} must have non-empty layer and topk axes")
    expected_dtype = moe_route_dtype(array.num_experts)
    if array.dtype != expected_dtype:
        raise RuntimeError(
            f"{array.num_experts} experts require {expected_dtype} routes, "
            f"got {array.dtype}"
        )
    if array.shape[-1] > array.num_experts:
        raise RuntimeError("MoE routing top-k exceeds exact expert count")
    flat = array.reshape(-1, array.shape[-1])
    for start in range(0, len(flat), 1 << 20):
        rows = np.sort(flat[start : start + (1 << 20)], axis=1)
        if rows.size and int(rows.max()) >= array.num_experts:
            raise RuntimeError("MoE route expert id is outside the exact model range")
        if rows.shape[1] > 1 and bool(np.any(rows[:, 1:] == rows[:, :-1])):
            raise RuntimeError(
                "MoE route expert ids must be distinct per token and layer"
            )


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
    stats: MoeRoutingAlignmentStats | None = None,
) -> tuple[MoeRouteArray, MoeRouteArray]:
    routes = metadata.get(ROUTED_EXPERTS_KEY)
    if not isinstance(routes, np.ndarray):
        raise RuntimeError("Missing binary routed experts")
    num_experts = int(metadata.get(NUM_EXPERTS_KEY, 0))
    if isinstance(routes, MoeRouteArray):
        if routes.num_experts != num_experts:
            raise RuntimeError("MoE route array disagrees with its exact expert count")
    else:
        routes = MoeRouteArray(routes, num_experts=num_experts)
    expected_lengths = {
        len(prompt_token_ids) + completion_token_count,
        len(prompt_token_ids) + max(completion_token_count - 1, 0),
    }
    if len(routes) not in expected_lengths:
        raise RuntimeError(
            "routed_experts length does not match prompt/completion token ids: "
            f"{len(routes)} not in {sorted(expected_lengths)}"
        )
    prompt_routes = _readonly_route_view(routes[: len(prompt_token_ids)])
    completion_routes = _readonly_route_view(routes[len(prompt_token_ids) :])
    if stats is not None:
        stats.prompt_route_bytes += int(prompt_routes.nbytes)
        stats.completion_route_bytes += int(completion_routes.nbytes)
    return prompt_routes, completion_routes


def _readonly_route_view(routes: np.ndarray) -> MoeRouteArray:
    route_view = cast(MoeRouteArray, routes)
    route_view.flags.writeable = False
    return route_view


def _fill_missing_routes(
    routes: np.ndarray, mask: np.ndarray, *, num_experts: int
) -> None:
    missing = np.flatnonzero(~mask)
    if missing.size:
        routes[missing] = deterministic_moe_routes(
            missing,
            route_shape=(int(routes.shape[1]), int(routes.shape[2])),
            num_experts=num_experts,
        )
        mask[missing] = True


def deterministic_moe_routes(
    positions: np.ndarray,
    *,
    route_shape: tuple[int, int],
    num_experts: int,
) -> MoeRouteArray:
    num_layers, topk = route_shape
    if num_layers <= 0 or not 1 <= topk <= num_experts:
        raise RuntimeError(
            "MoE route shape requires positive layers and top-k in expert range"
        )
    routes = np.empty(
        (len(positions), num_layers, topk), dtype=moe_route_dtype(num_experts)
    )
    base = (
        (positions.astype(np.uint64, copy=False)[:, None] + 1) * 1_299_709
        + np.arange(1, num_layers + 1, dtype=np.uint64)[None, :] * 97_003
    ) % num_experts
    for slot in range(topk):
        routes[:, :, slot] = (base + slot) % num_experts
    return MoeRouteArray(routes, num_experts=num_experts, validate=False)


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
