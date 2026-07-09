from __future__ import annotations

import base64
import io
from typing import Any

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


def _has_routing_experts(metadata: dict[str, Any]) -> bool:
    return any(metadata.get(key) is not None for key in _ROUTING_EXPERT_KEYS)


class MoeRoutingAlignmentStats(BaseModel):
    choices_with_routing: int = 0
    routed_tokens: int = 0
    overlap_conflict_rows: int = 0
    overlap_conflict_slots: int = 0
    overlap_compared_slots: int = 0


class MoeRoutingPackStats(BaseModel):
    packed_tokens: int = 0
    prefix_tree_rows: int = 0
    prefix_tree_conflict_rows: int = 0
    prefix_tree_conflict_slots: int = 0
    prefix_tree_compared_slots: int = 0

    def add_alignment(self, stats: MoeRoutingAlignmentStats) -> None:
        self.prefix_tree_conflict_rows += stats.overlap_conflict_rows
        self.prefix_tree_conflict_slots += stats.overlap_conflict_slots
        self.prefix_tree_compared_slots += stats.overlap_compared_slots


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
) -> tuple[list[TokenRoute | None] | None, MoeRoutingAlignmentStats]:
    if not (len(choices) == len(choice_offsets) == len(choice_token_lengths)):
        raise RuntimeError(
            "Choice routing alignment inputs differ in length: "
            f"choices={len(choices)}, offsets={len(choice_offsets)}, "
            f"lengths={len(choice_token_lengths)}"
        )
    aligned: list[TokenRoute | None] = [None] * len(token_ids)
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
        if len(prompt_routes) != len(prompt_token_ids):
            raise RuntimeError(
                "prompt_routed_experts length does not match prompt_token_ids: "
                f"{len(prompt_routes)} != {len(prompt_token_ids)}"
            )
        if len(completion_routes) not in {
            len(completion_token_ids),
            max(len(completion_token_ids) - 1, 0),
        }:
            raise RuntimeError(
                "completion_routed_experts length does not match completion_token_ids: "
                f"{len(completion_routes)} != {len(completion_token_ids)}"
            )
        for position, route in enumerate(prompt_routes):
            _overlay_route(aligned, position, route, stats)
        for offset_delta, route in enumerate(completion_routes):
            _overlay_route(aligned, offset + offset_delta, route, stats)
        stats.routed_tokens = sum(route is not None for route in aligned)
    if saw_routing and saw_missing:
        raise RuntimeError("Some trainable choices had MoE routes while others did not")
    return (aligned if saw_routing else None), stats


def _overlay_route(
    aligned: list[TokenRoute | None],
    position: int,
    route: TokenRoute,
    stats: MoeRoutingAlignmentStats,
) -> None:
    existing = aligned[position]
    if existing is None:
        aligned[position] = route
        return
    compared, conflicts = _count_route_slot_conflicts(existing, route)
    stats.overlap_compared_slots += compared
    stats.overlap_conflict_slots += conflicts
    if conflicts:
        stats.overlap_conflict_rows += 1


def _count_route_slot_conflicts(left: TokenRoute, right: TokenRoute) -> tuple[int, int]:
    _validate_route_shape(left)
    _validate_route_shape(right)
    if len(left) != len(right) or any(
        len(left_layer) != len(right_layer)
        for left_layer, right_layer in zip(left, right)
    ):
        raise RuntimeError("Cannot compare MoE routes with different shapes")
    compared = 0
    conflicts = 0
    for left_layer, right_layer in zip(left, right):
        for left_expert, right_expert in zip(left_layer, right_layer):
            compared += 1
            conflicts += int(int(left_expert) != int(right_expert))
    return compared, conflicts


def _normalize_token_ids(raw: Any) -> list[int]:
    if raw is None:
        raise RuntimeError("Missing routed token ids")
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected routed token ids list, got {type(raw)}")
    return [int(token_id) for token_id in raw]


def _normalize_routes(raw: Any, *, field_name: str) -> list[TokenRoute]:
    if isinstance(raw, str):
        raw = _decode_vllm_routed_experts(raw, field_name=field_name)
    if raw is None:
        raise RuntimeError(f"Missing {field_name}")
    if not isinstance(raw, list):
        raise RuntimeError(f"Expected {field_name} list, got {type(raw)}")
    routes: list[TokenRoute] = []
    for token_route in raw:
        if not isinstance(token_route, list):
            raise RuntimeError(f"Expected token route list in {field_name}")
        route: TokenRoute = []
        for layer_route in token_route:
            if not isinstance(layer_route, list):
                raise RuntimeError(f"Expected layer route list in {field_name}")
            route.append([int(expert_id) for expert_id in layer_route])
        _validate_route_shape(route)
        routes.append(route)
    return routes


def _decode_vllm_routed_experts(raw: str, *, field_name: str) -> list[Any]:
    import numpy as np

    try:
        array = np.load(io.BytesIO(base64.b64decode(raw)), allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {field_name} as base64 .npy") from exc
    if array.ndim != 3:
        raise RuntimeError(
            f"Expected {field_name} array with rank 3, got shape {array.shape}"
        )
    return array.tolist()


def _validate_route_shape(route: TokenRoute) -> None:
    if not route:
        raise RuntimeError("MoE token route cannot have zero layers")
    topk = len(route[0])
    if topk <= 0:
        raise RuntimeError("MoE token route cannot have zero topk")
    if any(len(layer_route) != topk for layer_route in route):
        raise RuntimeError("MoE token route topk must be constant across layers")


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
) -> tuple[list[TokenRoute], list[TokenRoute]]:
    if PROMPT_ROUTED_EXPERTS_KEY in metadata:
        return (
            _normalize_routes(
                metadata.get(PROMPT_ROUTED_EXPERTS_KEY),
                field_name=PROMPT_ROUTED_EXPERTS_KEY,
            ),
            _completion_routes(metadata),
        )

    routes = _normalize_routes(
        metadata.get(ROUTED_EXPERTS_KEY),
        field_name=ROUTED_EXPERTS_KEY,
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


def _completion_routes(metadata: dict[str, Any]) -> list[TokenRoute]:
    if COMPLETION_ROUTED_EXPERTS_KEY in metadata:
        return _normalize_routes(
            metadata[COMPLETION_ROUTED_EXPERTS_KEY],
            field_name=COMPLETION_ROUTED_EXPERTS_KEY,
        )
    if ROUTED_EXPERTS_KEY in metadata:
        return _normalize_routes(
            metadata[ROUTED_EXPERTS_KEY],
            field_name=ROUTED_EXPERTS_KEY,
        )
    raise RuntimeError("Missing routed completion experts")


def count_route_slot_conflicts(left: TokenRoute, right: TokenRoute) -> tuple[int, int]:
    return _count_route_slot_conflicts(left, right)
