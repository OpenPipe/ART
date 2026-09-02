from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import TYPE_CHECKING, Any, Literal, Protocol

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, model_validator

from art.distributed.adapter_transport import (
    NixlAdapterSender,
    NixlMemorySource,
    nixl_read_bytes,
)
from art.distributed.data_plane import (
    AsyncByteStreamPublisher,
    ByteStreamTransfer,
    receive_byte_stream,
)
from art.distributed.specs import LocalTransferEndpoint
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
ART_PRIVATE_ROUTE_OBJECT_HEADER = "x-art-route-object"
ART_PRIVATE_ROUTE_SOURCE_PREPARE_PATH = "/art/internal/v1/route-sources:prepare"
ART_PRIVATE_ROUTE_SOURCE_RELEASE_PREFIX = "/art/internal/v1/route-sources"


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


class NixlRouteTransfer(_Contract):
    """Bounded registered-memory source for one cross-domain route batch."""

    protocol_version: Literal[1] = 1
    stream_id: str = Field(min_length=1, max_length=512)
    target_host_id: str = Field(min_length=1, max_length=255)
    source: NixlMemorySource


class LocalRouteObjectView(_Contract):
    """Lease-bound local path for an otherwise opaque retained object."""

    source: RouteBundleObjectRef
    path: str = Field(min_length=1, max_length=4096)


class HolderRouteSourceObject(_Contract):
    """One service-leased holder object selected for a packing operation."""

    request_id: str = Field(min_length=1, max_length=256)
    object: RouteBundleObjectRef
    lease_id: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def _validate_holder_object(self) -> "HolderRouteSourceObject":
        if self.object.store != "holder_local":
            raise ValueError("holder route source referenced another object store")
        return self


def holder_route_source_operation_id(
    *,
    stream_id: str,
    source_endpoint: LocalTransferEndpoint,
    target_endpoint: LocalTransferEndpoint,
    objects: tuple[HolderRouteSourceObject, ...],
) -> str:
    payload = {
        "stream_id": stream_id,
        "source_endpoint": source_endpoint.model_dump(mode="json"),
        "target_endpoint": target_endpoint.model_dump(mode="json"),
        "objects": [item.model_dump(mode="json") for item in objects],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class HolderRouteSourceRequest(_Contract):
    """Exact private request to expose retained holder bytes to one packer."""

    protocol_version: Literal[1] = 1
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stream_id: str = Field(min_length=1, max_length=512)
    source_endpoint: LocalTransferEndpoint
    target_endpoint: LocalTransferEndpoint
    objects: tuple[HolderRouteSourceObject, ...] = Field(min_length=1, max_length=256)

    @classmethod
    def create(
        cls,
        refs: tuple[RetainedRouteBundleRef, ...],
        *,
        stream_id: str,
        source_endpoint: LocalTransferEndpoint,
        target_endpoint: LocalTransferEndpoint,
    ) -> "HolderRouteSourceRequest":
        objects = tuple(
            HolderRouteSourceObject(
                request_id=ref.layout.request_id,
                object=ref.object,
                lease_id=ref.lease_id,
            )
            for ref in refs
        )
        return cls(
            operation_id=holder_route_source_operation_id(
                stream_id=stream_id,
                source_endpoint=source_endpoint,
                target_endpoint=target_endpoint,
                objects=objects,
            ),
            stream_id=stream_id,
            source_endpoint=source_endpoint,
            target_endpoint=target_endpoint,
            objects=objects,
        )

    @model_validator(mode="after")
    def _validate_operation(self) -> "HolderRouteSourceRequest":
        if len({item.request_id for item in self.objects}) != len(self.objects):
            raise ValueError("holder route source repeats a request")
        if len({item.object.locator for item in self.objects}) != len(self.objects):
            raise ValueError("holder route source repeats an object")
        expected = holder_route_source_operation_id(
            stream_id=self.stream_id,
            source_endpoint=self.source_endpoint,
            target_endpoint=self.target_endpoint,
            objects=self.objects,
        )
        if self.operation_id != expected:
            raise ValueError("holder route source operation identity changed")
        return self

    @property
    def backend(self) -> Literal["local", "nixl"]:
        source = self.source_endpoint
        target = self.target_endpoint
        return (
            "local"
            if (source.domain, source.root) == (target.domain, target.root)
            else "nixl"
        )


class HolderRouteSourceReceipt(_Contract):
    """Holder proof for one active local view or NIXL registration."""

    protocol_version: Literal[1] = 1
    request: HolderRouteSourceRequest
    backend: Literal["local", "nixl"]
    local_objects: tuple[LocalRouteObjectView, ...] = ()
    nixl: NixlRouteTransfer | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> "HolderRouteSourceReceipt":
        if self.backend != self.request.backend:
            raise ValueError("holder route source backend changed")
        objects = tuple(item.object for item in self.request.objects)
        if self.backend == "local":
            if (
                self.nixl is not None
                or tuple(view.source for view in self.local_objects) != objects
            ):
                raise ValueError("local holder route source changed its objects")
            root = Path(self.request.source_endpoint.root)
            if any(
                str(Path(view.path).resolve()) != view.path
                or root not in Path(view.path).resolve().parents
                for view in self.local_objects
            ):
                raise ValueError("local holder route source escaped its transfer root")
        elif (
            self.local_objects
            or self.nixl is None
            or self.nixl.stream_id != self.request.stream_id
            or self.nixl.target_host_id != self.request.target_endpoint.host_id
            or self.nixl.source.byte_count
            != sum(item.object.size_bytes for item in self.request.objects)
        ):
            raise ValueError("NIXL holder route source changed its operation")
        return self


class RouteBundleBatchTransfer(_Contract):
    stream: ByteStreamTransfer | None = None
    nixl: NixlRouteTransfer | None = None
    local_objects: tuple[LocalRouteObjectView, ...] = ()
    local_transfer_root: str | None = Field(default=None, max_length=4096)
    layouts: tuple[RouteBundleLayout, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_stream(self) -> "RouteBundleBatchTransfer":
        if len({layout.bundle_id for layout in self.layouts}) != len(self.layouts):
            raise ValueError("retained route transfer repeats a bundle")
        if len({layout.response_id for layout in self.layouts}) != len(self.layouts):
            raise ValueError("retained route transfer repeats a response")
        transports = sum(
            (self.stream is not None, self.nixl is not None, bool(self.local_objects))
        )
        if transports != 1:
            raise ValueError("retained route transfer must use one transport")
        if bool(self.local_objects) != (self.local_transfer_root is not None):
            raise ValueError("local route transfer requires its explicit root")
        expected_bytes = sum(layout.byte_count for layout in self.layouts)
        transferred_bytes = (
            self.stream.byte_count
            if self.stream is not None
            else self.nixl.source.byte_count
            if self.nixl is not None
            else expected_bytes
        )
        if transferred_bytes != expected_bytes:
            raise ValueError("retained route transfer size differs from its layouts")
        root = (
            None
            if self.local_transfer_root is None
            else Path(self.local_transfer_root).resolve()
        )
        if self.local_objects and (
            str(root) != self.local_transfer_root
            or len(self.local_objects) != len(self.layouts)
            or any(
                view.source.size_bytes != layout.byte_count
                or view.source.sha256 != layout.sha256
                or view.source.store != "holder_local"
                or str(Path(view.path).resolve()) != view.path
                or Path(view.path).resolve().suffix != ".routes"
                or root not in Path(view.path).resolve().parents
                for view, layout in zip(self.local_objects, self.layouts, strict=True)
            )
        ):
            raise ValueError("local route objects differ from their layouts")
        return self

    @property
    def backend(self) -> Literal["stream", "local", "nixl"]:
        if self.stream is not None:
            return "stream"
        return "nixl" if self.nixl is not None else "local"

    async def receive_payload(
        self, *, timeout_s: float, target_host_id: str | None = None
    ) -> bytearray:
        if self.stream is not None:
            return await receive_byte_stream(self.stream, timeout_s=timeout_s)
        nixl = self.nixl
        if nixl is not None:
            if target_host_id != nixl.target_host_id:
                raise RuntimeError(
                    "NIXL route transfer reached the wrong runtime domain"
                )
            from art.utils.lifecycle import complete_to_thread

            payload, cancelled = await complete_to_thread(
                lambda: nixl_read_bytes(
                    nixl.source,
                    transfer_id=nixl.stream_id,
                    timeout_s=timeout_s,
                )
            )
            if cancelled is not None:
                raise cancelled
            return payload
        return await asyncio.to_thread(
            _read_local_route_objects,
            self.local_objects,
            self.local_transfer_root,
        )

    async def receive_into(
        self,
        groups: Iterable["TrajectoryGroup"],
        *,
        timeout_s: float,
        target_host_id: str | None = None,
    ) -> int:
        payload = await self.receive_payload(
            timeout_s=timeout_s, target_host_id=target_host_id
        )
        return hydrate_retained_route_bundles(groups, self.layouts, payload)


class RouteBundleReader(Protocol):
    """Service-injected lease-aware reader; provider access stays outside ART."""

    @property
    def retained_route_transport(self) -> Literal["holder_local", "caios_lota"]: ...

    @property
    def local_transfer_endpoint(self) -> LocalTransferEndpoint | None: ...

    async def resolve_local_view(
        self, ref: RouteBundleObjectRef, *, lease_id: str
    ) -> LocalRouteObjectView: ...

    def read_stream(
        self, ref: RouteBundleObjectRef, *, lease_id: str
    ) -> AsyncIterator[bytes | bytearray | memoryview]: ...

    async def prepare_transfer_source(
        self, request: HolderRouteSourceRequest
    ) -> HolderRouteSourceReceipt: ...

    async def release_transfer_source(
        self, receipt: HolderRouteSourceReceipt
    ) -> None: ...


@dataclass(frozen=True)
class RouteBundlePayload:
    layout: RouteBundleLayout
    chunks: tuple[memoryview, ...]


@dataclass
class NixlRoutePublisher:
    """Own the registered source bytes until the remote packing RPC settles."""

    transfer: NixlRouteTransfer
    _sender: NixlAdapterSender

    async def close(self) -> None:
        from art.utils.lifecycle import complete_to_thread

        _, cancelled = await complete_to_thread(self._sender.close)
        if cancelled is not None:
            raise cancelled


@dataclass
class HolderRouteSourceHandle:
    """Release an ephemeral holder source without releasing its service lease."""

    receipt: HolderRouteSourceReceipt
    _reader: RouteBundleReader
    _closed: bool = False

    async def close(self) -> None:
        if self._closed:
            return
        await self._reader.release_transfer_source(self.receipt)
        self._closed = True


async def prepare_holder_route_bundle_transfer(
    refs: tuple[RetainedRouteBundleRef, ...],
    *,
    reader: RouteBundleReader,
    stream_id: str,
    source_endpoint: LocalTransferEndpoint,
    target_endpoint: LocalTransferEndpoint,
) -> tuple[RouteBundleBatchTransfer, HolderRouteSourceHandle]:
    """Prepare the exact holder object where it physically resides."""

    request = HolderRouteSourceRequest.create(
        refs,
        stream_id=stream_id,
        source_endpoint=source_endpoint,
        target_endpoint=target_endpoint,
    )
    receipt = await reader.prepare_transfer_source(request)
    handle = HolderRouteSourceHandle(receipt=receipt, _reader=reader)
    try:
        if receipt.request != request:
            raise RuntimeError("holder route source receipt changed its request")
        transfer = RouteBundleBatchTransfer(
            nixl=receipt.nixl,
            local_objects=receipt.local_objects,
            local_transfer_root=(
                source_endpoint.root if receipt.backend == "local" else None
            ),
            layouts=tuple(ref.layout for ref in refs),
        )
    except BaseException:
        await handle.close()
        raise
    return transfer, handle


async def publish_retained_route_bundle_nixl_transfer(
    refs: tuple[RetainedRouteBundleRef, ...],
    *,
    reader: RouteBundleReader,
    stream_id: str,
    target_host_id: str,
) -> tuple[RouteBundleBatchTransfer, NixlRoutePublisher]:
    if not refs:
        raise ValueError("retained route transfer requires at least one bundle")
    if len({ref.layout.bundle_id for ref in refs}) != len(refs):
        raise ValueError("retained route transfer repeats a bundle")
    payload = await _read_retained_route_payload(refs, reader)
    from art.utils.lifecycle import complete_to_thread

    publisher, cancelled = await complete_to_thread(
        lambda: register_retained_route_nixl_source(
            payload, stream_id=stream_id, target_host_id=target_host_id
        )
    )
    if cancelled is not None:
        await publisher.close()
        raise cancelled
    try:
        transfer = RouteBundleBatchTransfer(
            nixl=publisher.transfer,
            layouts=tuple(ref.layout for ref in refs),
        )
    except BaseException:
        await publisher.close()
        raise
    return transfer, publisher


def register_retained_route_nixl_source(
    payload: bytearray, *, stream_id: str, target_host_id: str
) -> NixlRoutePublisher:
    sender = NixlAdapterSender()
    try:
        transfer = NixlRouteTransfer(
            stream_id=stream_id,
            target_host_id=target_host_id,
            source=sender.register_bytes(payload),
        )
    except BaseException:
        sender.close()
        raise
    return NixlRoutePublisher(transfer, sender)


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


def local_retained_route_bundle_transfer(
    refs: tuple[RetainedRouteBundleRef, ...],
    views: tuple[LocalRouteObjectView, ...],
    *,
    local_transfer_root: str,
) -> RouteBundleBatchTransfer:
    if not refs:
        raise ValueError("local retained route transfer requires at least one bundle")
    if tuple(view.source for view in views) != tuple(ref.object for ref in refs):
        raise ValueError("local route views differ from their retained objects")
    return RouteBundleBatchTransfer(
        local_objects=views,
        local_transfer_root=local_transfer_root,
        layouts=tuple(ref.layout for ref in refs),
    )


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


def retained_local_route_bundle_from_response(
    body: bytes,
    *,
    object_header: str,
    request_id: str,
    owner_id: str,
    model_identity: str,
    lease_id: str,
) -> tuple[ChatCompletion, RetainedRouteBundleRef]:
    """Bind an authenticated local object header to the exact binary response."""

    response, payload = retained_route_bundle_from_response(
        body,
        request_id=request_id,
        owner_id=owner_id,
        model_identity=model_identity,
    )
    if not object_header or len(object_header) > 16_384:
        raise ValueError("local route object header is missing or oversized")
    padding = "=" * (-len(object_header) % 4)
    try:
        decoded = base64.b64decode(
            object_header + padding,
            altchars=b"-_",
            validate=True,
        )
        object_ref = RouteBundleObjectRef.model_validate_json(decoded)
    except (ValueError, TypeError) as error:
        raise ValueError("local route object header is malformed") from error
    if object_ref.store != "holder_local":
        raise ValueError("local route response referenced another object store")
    if (
        object_ref.size_bytes != payload.layout.byte_count
        or object_ref.sha256 != payload.layout.sha256
    ):
        raise ValueError("local route object differs from the binary response")
    return response, RetainedRouteBundleRef(
        object=object_ref,
        layout=payload.layout,
        lease_id=lease_id,
    )


async def _read_retained_route_payload(
    refs: tuple[RetainedRouteBundleRef, ...], reader: RouteBundleReader
) -> bytearray:
    payload = bytearray()
    for ref in refs:
        start = len(payload)
        async for chunk in reader.read_stream(ref.object, lease_id=ref.lease_id):
            if len(payload) + len(chunk) - start > ref.object.size_bytes:
                raise RuntimeError("retained route reader exceeded its declared size")
            payload.extend(chunk)
        if len(payload) - start != ref.object.size_bytes:
            raise RuntimeError("retained route reader ended before its declared size")
    return payload


def _read_local_route_objects(
    views: tuple[LocalRouteObjectView, ...],
    local_transfer_root: str | None,
) -> bytearray:
    assert local_transfer_root is not None
    root = Path(local_transfer_root)
    payload = bytearray()
    for view in views:
        ref = view.source
        target = Path(view.path).resolve()
        if (
            ref.store != "holder_local"
            or target.suffix != ".routes"
            or root not in target.parents
        ):
            raise RuntimeError("local route object escaped its runtime namespace")
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != ref.size_bytes:
                raise RuntimeError("local route object changed size or type")
            digest = hashlib.sha256()
            start = len(payload)
            while chunk := os.read(descriptor, 1 << 20):
                payload.extend(chunk)
                digest.update(chunk)
            if (
                len(payload) - start != ref.size_bytes
                or digest.hexdigest() != ref.sha256
            ):
                raise RuntimeError("local route object changed digest")
        finally:
            os.close(descriptor)
    return payload


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
