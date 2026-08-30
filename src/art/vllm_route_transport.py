from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import struct
from typing import TYPE_CHECKING, Any, Literal, Protocol

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.distributed.data_plane import (
    AsyncByteStreamPublisher,
    ByteStreamTransfer,
    receive_byte_stream,
)
from art.preprocessing.moe_routing import (
    ART_MOE_ROUTING_METADATA_KEY,
    MoeRouteArray,
    attach_moe_routing_metadata_to_choice,
)

if TYPE_CHECKING:
    from art.trajectories import TrajectoryGroup

MAGIC = b"ARTRTE2\0"
HEADER = struct.Struct("<8sQII")
ROUTE_HEADER = struct.Struct("<IB3xQQQ")
DTYPES = {1: "u1", 2: "<u2"}
ROUTE_BUNDLE_FORMAT = "art_inference_route_bundle_v1"
RETAINED_ROUTE_BUNDLE_KEY = "retained_route_bundle"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RouteBundleChoiceLayout(_Contract):
    choice_index: int = Field(ge=0)
    dtype: Literal["uint8", "uint16"]
    shape: tuple[int, int, int]
    offset: int = Field(ge=0)
    byte_count: int = Field(ge=1)
    token_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_storage(self) -> "RouteBundleChoiceLayout":
        if any(value < 1 for value in self.shape):
            raise ValueError("route bundle choice dimensions must be positive")
        item_size = 1 if self.dtype == "uint8" else 2
        if item_size * _numel(self.shape) != self.byte_count:
            raise ValueError("route bundle choice byte count does not match its layout")
        return self


class RouteBundleLayout(_Contract):
    protocol_version: Literal[1] = 1
    format: Literal["art_inference_route_bundle_v1"] = ROUTE_BUNDLE_FORMAT
    bundle_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, max_length=256)
    owner_id: str = Field(min_length=1, max_length=512)
    model_identity: str = Field(min_length=1, max_length=4096)
    response_id: str = Field(min_length=1, max_length=512)
    num_experts: int = Field(ge=1, le=65_536)
    choices: tuple[RouteBundleChoiceLayout, ...] = Field(min_length=1)
    byte_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_identity(self) -> "RouteBundleLayout":
        if tuple(choice.choice_index for choice in self.choices) != tuple(
            sorted(choice.choice_index for choice in self.choices)
        ):
            raise ValueError("route bundle choices must use stable index order")
        if len({choice.choice_index for choice in self.choices}) != len(self.choices):
            raise ValueError("route bundle choice indices must be unique")
        cursor = 0
        route_shape = self.choices[0].shape[1:]
        expected_dtype = "uint8" if self.num_experts <= 256 else "uint16"
        for choice in self.choices:
            if choice.offset != cursor:
                raise ValueError("route bundle choices must exactly partition storage")
            if choice.shape[1:] != route_shape:
                raise ValueError("route bundle choices disagree on model route layout")
            if choice.shape[2] > self.num_experts:
                raise ValueError("route bundle top-k exceeds exact expert count")
            if choice.dtype != expected_dtype:
                raise ValueError("route bundle dtype disagrees with expert count")
            cursor += choice.byte_count
        if cursor != self.byte_count:
            raise ValueError("route bundle layout does not fill its exact byte count")
        if self.bundle_id != route_bundle_id(self):
            raise ValueError("route bundle ID does not match its immutable identity")
        return self


class RouteBundleObjectRef(_Contract):
    """Provider-neutral authenticated identity for one retained route object.

    ``holder_local`` locators are opaque to ART and name a lease-fenced object
    exposed by the paired holder through its injected reader. The reader may
    use NIXL or POSIX shared memory without changing the route contract.
    """

    store: Literal["caios", "holder_local"] = "caios"
    locator: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RetainedRouteBundleRef(_Contract):
    object: RouteBundleObjectRef
    layout: RouteBundleLayout
    lease_id: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _validate_object(self) -> "RetainedRouteBundleRef":
        if (
            self.object.size_bytes != self.layout.byte_count
            or self.object.sha256 != self.layout.sha256
        ):
            raise ValueError("retained route object differs from its exact layout")
        return self


class RouteBundleBatchTransfer(_Contract):
    stream: ByteStreamTransfer
    layouts: tuple[RouteBundleLayout, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_stream(self) -> "RouteBundleBatchTransfer":
        if len({layout.bundle_id for layout in self.layouts}) != len(self.layouts):
            raise ValueError("retained route transfer repeats a bundle")
        if len({layout.response_id for layout in self.layouts}) != len(self.layouts):
            raise ValueError("retained route transfer repeats a response")
        if sum(layout.byte_count for layout in self.layouts) != self.stream.byte_count:
            raise ValueError("retained route transfer size differs from its layouts")
        return self

    async def receive_into(
        self,
        groups: Iterable["TrajectoryGroup"],
        *,
        timeout_s: float,
    ) -> int:
        payload = await receive_byte_stream(self.stream, timeout_s=timeout_s)
        return hydrate_retained_route_bundles(groups, self.layouts, payload)


class RouteBundleReader(Protocol):
    """Service-injected lease-aware reader; provider access stays outside ART."""

    def read_stream(
        self, ref: RouteBundleObjectRef, *, lease_id: str
    ) -> AsyncIterator[bytes | bytearray | memoryview]: ...


@dataclass(frozen=True)
class RouteBundlePayload:
    layout: RouteBundleLayout
    chunks: tuple[memoryview, ...]


async def publish_retained_route_bundle_transfer(
    refs: tuple[RetainedRouteBundleRef, ...],
    *,
    reader: RouteBundleReader,
    stream_id: str,
    advertise_host: str,
) -> tuple[RouteBundleBatchTransfer, AsyncByteStreamPublisher]:
    """Expose one bounded pull stream over the object reader selected by service."""

    if not refs:
        raise ValueError("retained route transfer requires at least one bundle")
    if len({ref.layout.bundle_id for ref in refs}) != len(refs):
        raise ValueError("retained route transfer repeats a bundle")

    async def source() -> AsyncIterator[bytes | bytearray | memoryview]:
        for ref in refs:
            async for chunk in reader.read_stream(ref.object, lease_id=ref.lease_id):
                yield chunk

    publisher = await AsyncByteStreamPublisher.create(
        stream_id,
        advertise_host=advertise_host,
        byte_count=sum(ref.object.size_bytes for ref in refs),
        source=source,
    )
    try:
        transfer = RouteBundleBatchTransfer(
            stream=publisher.transfer,
            layouts=tuple(ref.layout for ref in refs),
        )
    except BaseException:
        await publisher.close()
        raise
    return transfer, publisher


def attach_retained_route_bundle(
    response: ChatCompletion, ref: RetainedRouteBundleRef
) -> None:
    """Attach one private retained handle without exposing route bytes."""

    _validate_response_binding(response, ref.layout)
    marker = {RETAINED_ROUTE_BUNDLE_KEY: ref.model_dump(mode="json")}
    for choice in response.choices:
        extra = choice.model_extra
        if extra is None:
            raise RuntimeError("OpenAI Choice.model_extra is unavailable")
        if ART_MOE_ROUTING_METADATA_KEY in extra:
            raise RuntimeError("response already carries MoE routing metadata")
        extra[ART_MOE_ROUTING_METADATA_KEY] = marker


def retained_route_bundles_from_groups(
    groups: Iterable["TrajectoryGroup"],
) -> tuple[RetainedRouteBundleRef, ...]:
    refs = []
    for group in groups:
        for trajectory in group.trajectories:
            for exchange in trajectory.exchanges.chat_completions:
                response = exchange.response
                marked = tuple(
                    ref
                    for choice in response.choices
                    if (ref := _retained_ref_from_choice(choice)) is not None
                )
                if not marked:
                    continue
                if len(marked) != len(response.choices) or any(
                    ref != marked[0] for ref in marked[1:]
                ):
                    raise RuntimeError(
                        "retained route handle does not cover the whole response"
                    )
                ref = marked[0]
                _validate_response_binding(response, ref.layout)
                refs.append(ref)
    return unique_retained_route_bundles(refs)


def unique_retained_route_bundles(
    refs: Iterable[RetainedRouteBundleRef],
) -> tuple[RetainedRouteBundleRef, ...]:
    unique: dict[str, RetainedRouteBundleRef] = {}
    responses: dict[str, str] = {}
    for ref in refs:
        previous = unique.setdefault(ref.layout.bundle_id, ref)
        if previous != ref:
            raise RuntimeError("retained route bundle identity changed")
        previous_bundle = responses.setdefault(
            ref.layout.response_id, ref.layout.bundle_id
        )
        if previous_bundle != ref.layout.bundle_id:
            raise RuntimeError("retained response maps to multiple route bundles")
    return tuple(unique.values())


def hydrate_retained_route_bundles(
    groups: Iterable["TrajectoryGroup"],
    layouts: tuple[RouteBundleLayout, ...],
    payload: bytes | bytearray | memoryview,
) -> int:
    """Hydrate private references directly into ART's existing route arrays."""

    responses: dict[str, list[ChatCompletion]] = defaultdict(list)
    for group in groups:
        for trajectory in group.trajectories:
            for exchange in trajectory.exchanges.chat_completions:
                responses[exchange.response.id].append(exchange.response)
    view = memoryview(payload).cast("B").toreadonly()
    try:
        offset = 0
        hydrated = 0
        for layout in layouts:
            bound = responses.get(layout.response_id)
            if not bound:
                raise RuntimeError("retained route bundle has no selected response")
            end = offset + layout.byte_count
            if end > len(view):
                raise RuntimeError("retained route transfer ended before its layout")
            first_payload = bound[0].model_dump(mode="python")
            token_ids = {
                choice.choice_index: _response_token_ids(
                    first_payload, choice.choice_index
                )
                for choice in layout.choices
            }
            routes = decode_retained_route_bundle(
                layout, view[offset:end], token_ids=token_ids
            )
            for response in bound:
                _validate_response_binding(response, layout)
                response_payload = response.model_dump(mode="python")
                for position, choice in enumerate(response.choices):
                    extra = choice.model_extra
                    if extra is None:
                        raise RuntimeError("OpenAI Choice.model_extra is unavailable")
                    if ART_MOE_ROUTING_METADATA_KEY in extra:
                        raise RuntimeError(
                            "selected response already carries MoE routing metadata"
                        )
                    attach_moe_routing_metadata_to_choice(
                        choice=choice,
                        response_payload=response_payload,
                        choice_index=position,
                        routed_experts=routes[int(choice.index)],
                        num_experts=layout.num_experts,
                    )
                hydrated += 1
            offset = end
        if offset != len(view):
            raise RuntimeError("retained route transfer has trailing bytes")
        return hydrated
    finally:
        view.release()


def is_routed_experts_response(body: bytes) -> bool:
    return body.startswith(MAGIC)


def decode_routed_experts_response(
    body: bytes,
) -> tuple[ChatCompletion, dict[int, MoeRouteArray]]:
    import numpy as np

    response, _, num_experts, encoded = _parse_routed_experts_response(body)
    return response, {
        choice_index: MoeRouteArray(
            np.frombuffer(chunk, dtype=dtype).reshape(shape),
            num_experts=num_experts,
        )
        for choice_index, dtype, shape, chunk in encoded
    }


def retained_route_bundle_from_response(
    body: bytes,
    *,
    request_id: str,
    owner_id: str,
    model_identity: str,
) -> tuple[ChatCompletion, RouteBundlePayload]:
    """Split one binary response into JSON plus immutable route-only views."""

    response, response_payload, num_experts, encoded = _parse_routed_experts_response(
        body
    )
    layouts = []
    chunks = []
    digest = hashlib.sha256()
    offset = 0
    for choice_index, dtype, shape, chunk in encoded:
        token_ids = _response_token_ids(response_payload, choice_index)
        dtype_name: Literal["uint8", "uint16"] = "uint8" if dtype == "u1" else "uint16"
        layouts.append(
            RouteBundleChoiceLayout(
                choice_index=choice_index,
                dtype=dtype_name,
                shape=shape,
                offset=offset,
                byte_count=len(chunk),
                token_ids_sha256=_token_ids_sha256(token_ids),
            )
        )
        chunks.append(chunk)
        digest.update(chunk)
        offset += len(chunk)
    identity: dict[str, Any] = {
        "protocol_version": 1,
        "format": ROUTE_BUNDLE_FORMAT,
        "request_id": request_id,
        "owner_id": owner_id,
        "model_identity": model_identity,
        "response_id": response.id,
        "num_experts": num_experts,
        "choices": [layout.model_dump(mode="json") for layout in layouts],
        "byte_count": offset,
        "sha256": digest.hexdigest(),
    }
    layout = RouteBundleLayout(bundle_id=route_bundle_id(identity), **identity)
    return response, RouteBundlePayload(layout=layout, chunks=tuple(chunks))


def decode_retained_route_bundle(
    layout: RouteBundleLayout,
    payload: bytes | bytearray | memoryview,
    *,
    token_ids: Mapping[int, tuple[int, ...]],
) -> dict[int, MoeRouteArray]:
    """Validate one fetched retained object before exposing route array views."""

    import numpy as np

    view = memoryview(payload).cast("B").toreadonly()
    if len(view) != layout.byte_count:
        raise RuntimeError("retained route bundle changed its exact byte count")
    if hashlib.sha256(view).hexdigest() != layout.sha256:
        raise RuntimeError("retained route bundle changed its exact digest")
    routes = {}
    for choice in layout.choices:
        expected_tokens = token_ids.get(choice.choice_index)
        if expected_tokens is None or _token_ids_sha256(expected_tokens) != (
            choice.token_ids_sha256
        ):
            raise RuntimeError("retained route bundle changed its token identity")
        dtype = np.dtype("u1" if choice.dtype == "uint8" else "<u2")
        chunk = view[choice.offset : choice.offset + choice.byte_count]
        routes[choice.choice_index] = MoeRouteArray(
            np.frombuffer(chunk, dtype=dtype).reshape(choice.shape),
            num_experts=layout.num_experts,
        )
    return routes


def route_bundle_id(layout: RouteBundleLayout | Mapping[str, Any]) -> str:
    payload = (
        layout.model_dump(mode="json", exclude={"bundle_id"})
        if isinstance(layout, RouteBundleLayout)
        else {key: value for key, value in layout.items() if key != "bundle_id"}
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parse_routed_experts_response(
    body: bytes,
) -> tuple[
    ChatCompletion,
    Mapping[str, Any],
    int,
    tuple[tuple[int, str, tuple[int, int, int], memoryview], ...],
]:
    if len(body) < HEADER.size:
        raise RuntimeError("Truncated ART routed-experts response header")
    magic, json_size, route_count, num_experts = HEADER.unpack_from(body)
    if magic != MAGIC:
        raise RuntimeError("Invalid ART routed-experts response magic")
    offset = HEADER.size
    json_end = offset + json_size
    if json_end > len(body):
        raise RuntimeError("Truncated ART routed-experts JSON response")
    json_body = body[offset:json_end]
    response_payload = json.loads(json_body)
    if not isinstance(response_payload, dict):
        raise RuntimeError("ART routed-experts JSON response must be an object")
    response = ChatCompletion.model_validate(response_payload)
    offset = json_end
    routes = []
    choice_indices = set()
    for _ in range(route_count):
        if offset + ROUTE_HEADER.size > len(body):
            raise RuntimeError("Truncated ART routed-experts array header")
        choice_index, dtype_code, tokens, layers, topk = ROUTE_HEADER.unpack_from(
            body, offset
        )
        offset += ROUTE_HEADER.size
        dtype_name = DTYPES.get(dtype_code)
        if dtype_name is None:
            raise RuntimeError(f"Unknown ART route dtype code {dtype_code}")
        item_size = 1 if dtype_name == "u1" else 2
        size = int(tokens * layers * topk * item_size)
        end = offset + size
        if end > len(body):
            raise RuntimeError("Truncated ART routed-experts array")
        if choice_index in choice_indices:
            raise RuntimeError(f"Duplicate routed experts for choice {choice_index}")
        choice_indices.add(choice_index)
        routes.append(
            (
                choice_index,
                dtype_name,
                (int(tokens), int(layers), int(topk)),
                memoryview(body)[offset:end].toreadonly(),
            )
        )
        offset = end
    if offset != len(body):
        raise RuntimeError("Unexpected trailing bytes in ART routed-experts response")
    return response, response_payload, int(num_experts), tuple(routes)


def _response_token_ids(
    payload: Mapping[str, Any], choice_index: int
) -> tuple[int, ...]:
    prompt = payload.get("prompt_token_ids")
    choices = payload.get("choices")
    if not isinstance(prompt, list) or not isinstance(choices, list):
        raise RuntimeError("retained routes require exact response token IDs")
    choice = next(
        (
            value
            for value in choices
            if isinstance(value, dict) and value.get("index") == choice_index
        ),
        None,
    )
    if choice is None:
        raise RuntimeError("retained routes do not match a response choice")
    completion = next(
        (
            choice[key]
            for key in ("completion_token_ids", "output_token_ids", "token_ids")
            if isinstance(choice.get(key), list)
        ),
        None,
    )
    if completion is None or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (*prompt, *completion)
    ):
        raise RuntimeError("retained routes require nonnegative exact token IDs")
    return tuple((*prompt, *completion))


def _retained_ref_from_choice(choice: Any) -> RetainedRouteBundleRef | None:
    extra = choice.model_extra
    nested = (extra or {}).get(ART_MOE_ROUTING_METADATA_KEY)
    if nested is None:
        return None
    if not isinstance(nested, dict):
        raise RuntimeError("MoE routing metadata must be an object")
    payload = nested.get(RETAINED_ROUTE_BUNDLE_KEY)
    if payload is None:
        return None
    try:
        return RetainedRouteBundleRef.model_validate(payload)
    except ValueError as error:
        raise RuntimeError("retained route handle is malformed") from error


def _validate_response_binding(
    response: ChatCompletion, layout: RouteBundleLayout
) -> None:
    if response.id != layout.response_id:
        raise RuntimeError("retained routes belong to another response")
    choice_indices = tuple(sorted(int(choice.index) for choice in response.choices))
    if choice_indices != tuple(choice.choice_index for choice in layout.choices):
        raise RuntimeError("retained routes do not match response choices")
    payload = response.model_dump(mode="python")
    for choice in layout.choices:
        if (
            _token_ids_sha256(_response_token_ids(payload, choice.choice_index))
            != choice.token_ids_sha256
        ):
            raise RuntimeError("retained routes do not match response tokens")


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for extent in shape:
        value *= extent
    return value
