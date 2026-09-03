"""Private retained-route wire contracts for the isolated vLLM runtime."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeVar

from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

ART_PRIVATE_ROUTE_SOURCE_PREPARE_PATH = "/art/internal/v1/route-sources:prepare"
ART_PRIVATE_ROUTE_SOURCE_RELEASE_PREFIX = "/art/internal/v1/route-sources"
HOLDER_ROUTE_RESPONSE_MEDIA_TYPE = "application/vnd.art.holder-route-response-v1+json"
ROUTE_BUNDLE_FORMAT = "art_inference_route_bundle_v1"
_T = TypeVar("_T")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LocalTransferEndpoint(_Contract):
    host_id: str = Field(min_length=1, max_length=255)
    domain: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    root: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def _validate_root(self) -> "LocalTransferEndpoint":
        root = Path(self.root)
        if not root.is_absolute() or str(root.resolve()) != self.root:
            raise ValueError("local transfer root must be a canonical absolute path")
        return self


class PairedTransferIdentity(_Contract):
    lora_backend: Literal["local", "nixl"]
    trainer_endpoints: tuple[LocalTransferEndpoint, ...] = Field(
        min_length=1, max_length=256
    )
    inference_endpoints: tuple[LocalTransferEndpoint, ...] = Field(
        min_length=1, max_length=256
    )
    lora_source_host_id: str = Field(min_length=1, max_length=255)
    route_source: LocalTransferEndpoint | None = None
    route_delivery: Literal["none", "local", "nixl", "mixed"] = "none"

    @model_validator(mode="after")
    def _validate_topology(self) -> "PairedTransferIdentity":
        for endpoints in (self.trainer_endpoints, self.inference_endpoints):
            if len({endpoint.host_id for endpoint in endpoints}) != len(endpoints):
                raise ValueError("paired transfer repeats a host endpoint")
        if self.lora_source_host_id not in {
            endpoint.host_id for endpoint in self.trainer_endpoints
        }:
            raise ValueError("LoRA source must be a trainer endpoint")
        if self.route_source is not None and self.route_source not in (
            self.inference_endpoints
        ):
            raise ValueError("route source must be an inference endpoint")
        identities = {
            (endpoint.domain, endpoint.root)
            for endpoint in self.trainer_endpoints + self.inference_endpoints
        }
        expected_lora = "local" if len(identities) == 1 else "nixl"
        if self.lora_backend != expected_lora:
            raise ValueError("LoRA transfer backend differs from its endpoints")
        expected_route = _route_delivery(self.route_source, self.trainer_endpoints)
        if self.route_delivery != expected_route:
            raise ValueError("route delivery differs from its physical endpoints")
        return self

    def route_target(self, target_host_id: str) -> LocalTransferEndpoint:
        try:
            return next(
                endpoint
                for endpoint in self.trainer_endpoints
                if endpoint.host_id == target_host_id
            )
        except StopIteration:
            raise RuntimeError("route target is not a trainer endpoint") from None

    def route_backend(self, target_host_id: str) -> Literal["local", "nixl"]:
        if self.route_source is None:
            raise RuntimeError("paired transfer has no holder route source")
        return _local_transfer_backend(
            self.route_source, self.route_target(target_host_id)
        )


def _local_transfer_backend(
    source: LocalTransferEndpoint, target: LocalTransferEndpoint
) -> Literal["local", "nixl"]:
    return (
        "local"
        if (source.domain, source.root) == (target.domain, target.root)
        else "nixl"
    )


def _route_delivery(
    source: LocalTransferEndpoint | None,
    targets: tuple[LocalTransferEndpoint, ...],
) -> Literal["none", "local", "nixl", "mixed"]:
    if source is None:
        return "none"
    backends = {_local_transfer_backend(source, target) for target in targets}
    if len(backends) == 1:
        return backends.pop()
    return "mixed"


class NixlMemorySource(_Contract):
    agent: str = Field(min_length=1, max_length=255)
    metadata_b64: str = Field(min_length=1, max_length=1 << 20)
    address: int = Field(gt=0)
    byte_count: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_metadata(self) -> "NixlMemorySource":
        try:
            base64.b64decode(self.metadata_b64, validate=True)
        except ValueError as error:
            raise ValueError("NIXL source metadata is not valid base64") from error
        return self


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
        indices = tuple(choice.choice_index for choice in self.choices)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise ValueError("route bundle choices require unique stable indexes")
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
    store: Literal["caios", "holder_local"] = "caios"
    locator: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HolderRouteResponseEnvelope(_Contract):
    protocol_version: Literal[1] = 1
    response: ChatCompletion
    object: RouteBundleObjectRef
    layout: RouteBundleLayout

    @model_validator(mode="after")
    def _validate_bundle(self) -> "HolderRouteResponseEnvelope":
        if (
            self.object.store != "holder_local"
            or self.object.size_bytes != self.layout.byte_count
            or self.object.sha256 != self.layout.sha256
        ):
            raise ValueError("holder route object differs from its exact layout")
        _validate_response_binding(self.response, self.layout)
        return self


class NixlRouteTransfer(_Contract):
    protocol_version: Literal[1] = 1
    stream_id: str = Field(min_length=1, max_length=512)
    target_host_id: str = Field(min_length=1, max_length=255)
    source: NixlMemorySource


class LocalRouteObjectView(_Contract):
    source: RouteBundleObjectRef
    path: str = Field(min_length=1, max_length=4096)


class HolderRouteSourceObject(_Contract):
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
    protocol_version: Literal[1] = 1
    operation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    stream_id: str = Field(min_length=1, max_length=512)
    source_endpoint: LocalTransferEndpoint
    target_endpoint: LocalTransferEndpoint
    objects: tuple[HolderRouteSourceObject, ...] = Field(min_length=1, max_length=256)

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
        return _local_transfer_backend(self.source_endpoint, self.target_endpoint)


class HolderRouteSourceReceipt(_Contract):
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


def holder_route_layout_from_parts(
    response_body: bytes,
    route_payload: bytes,
    choices: tuple[
        tuple[int, Literal["uint8", "uint16"], tuple[int, int, int], int, int],
        ...,
    ],
    *,
    num_experts: int,
    request_id: str,
    owner_id: str,
    model_identity: str,
) -> tuple[ChatCompletion, RouteBundleLayout]:
    response_payload = json.loads(response_body)
    if not isinstance(response_payload, dict):
        raise RuntimeError("ART JSON response must be an object")
    response = ChatCompletion.model_validate(response_payload)
    view = memoryview(route_payload)
    layouts = []
    digest = hashlib.sha256()
    cursor = 0
    for choice_index, dtype, shape, offset, byte_count in choices:
        end = offset + byte_count
        if offset != cursor or byte_count < 1 or end > len(view):
            raise RuntimeError("route-only storage layout is not an exact partition")
        chunk = view[offset:end]
        layouts.append(
            RouteBundleChoiceLayout(
                choice_index=choice_index,
                dtype=dtype,
                shape=shape,
                offset=offset,
                byte_count=byte_count,
                token_ids_sha256=_token_ids_sha256(
                    _response_token_ids(response_payload, choice_index)
                ),
            )
        )
        digest.update(chunk)
        cursor = end
    if cursor != len(view):
        raise RuntimeError("route-only storage layout has trailing bytes")
    identity: dict[str, Any] = {
        "protocol_version": 1,
        "format": ROUTE_BUNDLE_FORMAT,
        "request_id": request_id,
        "owner_id": owner_id,
        "model_identity": model_identity,
        "response_id": response.id,
        "num_experts": num_experts,
        "choices": [layout.model_dump(mode="json") for layout in layouts],
        "byte_count": cursor,
        "sha256": digest.hexdigest(),
    }
    return response, RouteBundleLayout(bundle_id=route_bundle_id(identity), **identity)


def route_bundle_id(layout: RouteBundleLayout | Mapping[str, Any]) -> str:
    payload = (
        layout.model_dump(mode="json", exclude={"bundle_id"})
        if isinstance(layout, RouteBundleLayout)
        else {key: value for key, value in layout.items() if key != "bundle_id"}
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def _validate_response_binding(
    response: ChatCompletion, layout: RouteBundleLayout
) -> None:
    if response.id != layout.response_id:
        raise RuntimeError("retained routes belong to another response")
    indices = tuple(sorted(int(choice.index) for choice in response.choices))
    if indices != tuple(choice.choice_index for choice in layout.choices):
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


def _load_nixl() -> tuple[Any, Any, Any]:
    for name in ("nixl_cu13", "nixl_cu12", "nixl"):
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            continue
        site_packages = Path(next(iter(spec.submodule_search_locations))).parent
        core = site_packages / f".{name}.mesonpy.libs"
        dependencies = site_packages / f"{name}.libs"
        plugin = dependencies / "nixl"
        ucx = dependencies / "ucx"
        if not (core / "libnixl.so").is_file():
            raise RuntimeError(f"{name} is missing its bundled libnixl.so")
        if not (plugin / "libplugin_UCX.so").is_file():
            raise RuntimeError(f"{name} is missing its bundled UCX plugin")
        os.environ["NIXL_LIBRARY_DIR"] = str(core)
        os.environ["NIXL_DEPENDENCY_LIBRARY_DIR"] = str(dependencies)
        os.environ["NIXL_PLUGIN_DIR"] = str(plugin)
        os.environ["UCX_MODULE_DIR"] = str(ucx)
        os.environ.setdefault("UCX_NET_DEVICES", "all")
        os.environ.setdefault("UCX_TLS", "rc,rc_gda,cuda_copy")
        os.environ.setdefault("UCX_IB_GDA_RETAIN_INACTIVE_CTX", "yes")
        libraries = (str(core), str(dependencies))
        inherited = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            dict.fromkeys((*libraries, *filter(None, inherited)))
        )
        module = importlib.import_module(name)
        return (
            module.nixl_agent,
            module.nixl_agent_config,
            module.nixl_thread_sync_t,
        )
    raise RuntimeError("NIXL Python bindings are unavailable in the vLLM runtime")


class _NixlByteSource:
    def __init__(self, payload: bytearray) -> None:
        if not payload:
            raise ValueError("NIXL route source must not be empty")
        agent_type, config_type, sync_type = _load_nixl()
        self._payload = payload
        self._block = torch.frombuffer(payload, dtype=torch.uint8)
        self._agent = agent_type(
            f"art-route-source-{os.getpid()}-{id(self):x}",
            config_type(
                enable_prog_thread=True,
                enable_listen_thread=False,
                backends=["UCX"],
                sync_mode=sync_type.NIXL_THREAD_SYNC_STRICT,
            ),
        )
        self._registration = self._agent.register_memory(
            (self._block,), backends=["UCX"]
        )
        self.source = NixlMemorySource(
            agent=self._agent.name,
            metadata_b64=base64.b64encode(self._agent.get_agent_metadata()).decode(),
            address=self._block.data_ptr(),
            byte_count=len(payload),
        )

    def close(self) -> None:
        if self._registration is None:
            return
        self._agent.deregister_memory(self._registration, backends=["UCX"])
        self._registration = None
        self._block = None
        self._payload = None


@dataclass
class NixlRoutePublisher:
    transfer: NixlRouteTransfer
    _source: _NixlByteSource

    async def close(self) -> None:
        _, cancelled = await complete_to_thread(self._source.close)
        if cancelled is not None:
            raise cancelled


def register_retained_route_nixl_source(
    payload: bytearray, *, stream_id: str, target_host_id: str
) -> NixlRoutePublisher:
    source = _NixlByteSource(payload)
    try:
        transfer = NixlRouteTransfer(
            stream_id=stream_id,
            target_host_id=target_host_id,
            source=source.source,
        )
    except BaseException:
        source.close()
        raise
    return NixlRoutePublisher(transfer=transfer, _source=source)


async def complete_to_thread(
    operation: Callable[[], _T],
) -> tuple[_T, asyncio.CancelledError | None]:
    task = asyncio.create_task(asyncio.to_thread(operation))
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.cancelled():
                break
            cancelled = cancelled or error
        except BaseException:
            break
    try:
        result = task.result()
    except BaseException as error:
        if cancelled is not None:
            cancelled.add_note(f"operation also failed: {error}")
            raise cancelled
        raise
    return result, cancelled
