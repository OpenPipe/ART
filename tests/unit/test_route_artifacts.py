import hashlib
import struct
from types import SimpleNamespace

import pytest
import torch

from art.distributed import ClusterSpec, HostSpec
import art.distributed.art_runtime as art_runtime_module
from art.distributed.art_runtime import ArtRuntime
from art.distributed.specs import LocalTransferEndpoint
from art.route_artifacts import (
    MaterializedRouteArtifactProvider,
    RouteArtifactExportReceipt,
    materialize_route_artifact,
)
from art.training import OperationRef
from art.vllm_route_transport import (
    LocalRouteObjectView,
    RetainedRouteBundleRef,
    RouteBundleChoiceLayout,
    RouteBundleLayout,
    RouteBundleObjectRef,
    local_retained_route_bundle_transfer,
    publish_retained_route_bundle_nixl_transfer,
    route_bundle_id,
)


def test_opaque_holder_route_never_selects_path_loader() -> None:
    from art.distributed.art_runtime import _route_refs_are_local_files

    ref = SimpleNamespace(object=SimpleNamespace(locator="holder-local:owner:bundle"))
    assert not _route_refs_are_local_files((ref,))  # type: ignore[arg-type]

    ref.object.locator = "/dev/shm/art_vllm_routes/bundle.routes"
    assert _route_refs_are_local_files((ref,))  # type: ignore[arg-type]


def _route_ref(payload: bytes, locator: str) -> RetainedRouteBundleRef:
    choice = RouteBundleChoiceLayout(
        choice_index=0,
        dtype="uint8",
        shape=(len(payload), 1, 1),
        offset=0,
        byte_count=len(payload),
        token_ids_sha256="a" * 64,
    )
    identity = {
        "protocol_version": 1,
        "format": "art_inference_route_bundle_v1",
        "request_id": "b" * 64,
        "owner_id": "paired-slot",
        "model_identity": "model@generation",
        "response_id": "response",
        "num_experts": 1,
        "choices": [choice.model_dump(mode="json")],
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    layout = RouteBundleLayout(bundle_id=route_bundle_id(identity), **identity)
    return RetainedRouteBundleRef(
        object=RouteBundleObjectRef(
            store="holder_local",
            locator=locator,
            size_bytes=len(payload),
            sha256=layout.sha256,
        ),
        layout=layout,
        lease_id="route-capture-" + "b" * 64,
    )


def _runtime(
    root: str, *, inference_domain: str, remote_trainer: bool = False
) -> ArtRuntime:
    hosts = [
        HostSpec(
            host_id="trainer",
            node_rank=0,
            worker_address="tcp://trainer:1",
            cpu_slots=1,
            local_transfer_domain="trainer-domain",
            local_transfer_root=root,
        ),
        HostSpec(
            host_id="inference",
            node_rank=1,
            worker_address="tcp://inference:1",
            cpu_slots=1,
            local_transfer_domain=inference_domain,
            local_transfer_root=root,
        ),
    ]
    if remote_trainer:
        hosts.append(
            HostSpec(
                host_id="trainer-remote",
                node_rank=2,
                worker_address="tcp://trainer-remote:1",
                cpu_slots=1,
                local_transfer_domain="trainer-remote-domain",
                local_transfer_root=root,
            )
        )
    cluster = ClusterSpec(
        hosts=tuple(hosts),
        controller_host_id="trainer",
    )
    runtime = object.__new__(ArtRuntime)
    runtime.topology = SimpleNamespace(
        cluster=cluster,
        trainer=SimpleNamespace(
            ranks=tuple(
                SimpleNamespace(host_id=host_id)
                for host_id in (
                    ("trainer", "trainer-remote") if remote_trainer else ("trainer",)
                )
            ),
            coordinator_rank=0,
        ),
        model_services=(
            SimpleNamespace(members=(SimpleNamespace(host_id="inference"),)),
        ),
    )
    runtime._nixl_transport = object()
    return runtime


@pytest.mark.asyncio
async def test_split_inference_uses_controller_local_route_view(tmp_path) -> None:
    payload = b"shared-route"
    path = tmp_path / "bundle.routes"
    path.write_bytes(payload)
    ref = _route_ref(payload, "holder-local:opaque:bundle")

    class Reader:
        retained_route_transport = "holder_local"
        local_transfer_endpoint = LocalTransferEndpoint(
            host_id="trainer", domain="trainer-domain", root=str(tmp_path)
        )

        async def resolve_local_view(self, source, *, lease_id):
            assert (source, lease_id) == (ref.object, ref.lease_id)
            return LocalRouteObjectView(source=source, path=str(path))

    runtime = _runtime(str(tmp_path), inference_domain="inference-domain")
    runtime._route_bundle_reader = Reader()
    paired = runtime._paired_transfer_identity()
    assert paired.lora_backend == "nixl"
    assert paired.route_delivery == "local"
    transfer, publisher = await runtime._holder_route_transfer(
        (ref,), batch_id="batch", target_host_id="trainer"
    )

    assert publisher is None
    assert await transfer.receive_payload(timeout_s=1) == payload


@pytest.mark.asyncio
async def test_remote_trainer_route_target_selects_nixl(tmp_path, monkeypatch) -> None:
    ref = _route_ref(b"remote-route", "holder-local:opaque:bundle")
    reader = SimpleNamespace(
        retained_route_transport="holder_local",
        local_transfer_endpoint=LocalTransferEndpoint(
            host_id="trainer", domain="trainer-domain", root=str(tmp_path)
        ),
    )
    called = False

    async def publish(refs, **kwargs):
        nonlocal called
        called = True
        assert refs == (ref,)
        assert kwargs["reader"] is reader
        assert kwargs["target_host_id"] == "trainer-remote"
        return "nixl-transfer", "source-registration"

    monkeypatch.setattr(
        art_runtime_module, "publish_retained_route_bundle_nixl_transfer", publish
    )
    runtime = _runtime(
        str(tmp_path), inference_domain="inference-domain", remote_trainer=True
    )
    runtime._route_bundle_reader = reader
    paired = runtime._paired_transfer_identity()
    assert paired.route_delivery == "mixed"
    assert paired.route_backend("trainer-remote") == "nixl"

    assert await runtime._holder_route_transfer(
        (ref,), batch_id="batch", target_host_id="trainer-remote"
    ) == ("nixl-transfer", "source-registration")
    assert called


class _NixlHandle:
    def __init__(self, local, remote) -> None:
        self.local = local
        self.remote = remote

    def release(self) -> None:
        return None


class _NixlAgent:
    memory: dict[int, torch.Tensor] = {}

    def __init__(self, name: str) -> None:
        self.name = name

    def register_memory(self, blocks, **_kwargs):
        addresses = tuple(block.data_ptr() for block in blocks)
        self.memory.update(zip(addresses, blocks, strict=True))
        return addresses

    def deregister_memory(self, addresses, **_kwargs) -> None:
        for address in addresses:
            self.memory.pop(address)

    def get_agent_metadata(self) -> bytes:
        return self.name.encode()

    def add_remote_agent(self, metadata: bytes) -> str:
        return metadata.decode()

    def remove_remote_agent(self, _name: str) -> None:
        return None

    def get_xfer_descs(self, blocks, **_kwargs):
        return blocks

    def initialize_xfer(self, operation, local, remote, _agent, **_kwargs):
        assert operation == "READ"
        return _NixlHandle(local, remote)

    def transfer(self, handle: _NixlHandle) -> str:
        address, count, _device = handle.remote[0]
        handle.local[0].copy_(self.memory[address].narrow(0, 0, count))
        return "DONE"


@pytest.mark.asyncio
async def test_nixl_route_transfer_moves_exact_reader_bytes(monkeypatch) -> None:
    payload = b"nixl-route"
    ref = _route_ref(payload, "holder-local:opaque:nixl")

    class Reader:
        async def read_stream(self, source, *, lease_id):
            assert (source, lease_id) == (ref.object, ref.lease_id)
            yield payload[:3]
            yield payload[3:]

    monkeypatch.setattr("art.distributed.adapter_transport._new_agent", _NixlAgent)
    transfer, publisher = await publish_retained_route_bundle_nixl_transfer(
        (ref,), reader=Reader(), stream_id="routes", target_host_id="packer"
    )
    try:
        assert (
            await transfer.receive_payload(timeout_s=1, target_host_id="packer")
            == payload
        )
    finally:
        await publisher.close()
    assert not _NixlAgent.memory


@pytest.mark.asyncio
async def test_route_artifact_stream_materializes_for_exact_operation(
    tmp_path,
) -> None:
    payload = b"exact-routes"
    source = _route_ref(
        payload,
        "holder-local:opaque:" + hashlib.sha256(payload).hexdigest(),
    )
    export = RouteArtifactExportReceipt.create(
        attempt_id="c" * 64,
        tenant_id="tenant",
        run_id="run",
        operation_id="d" * 64,
        source=source,
    )
    encoded = export.model_dump_json().encode()

    async def chunks():
        framed = struct.pack("!I", len(encoded)) + encoded + payload
        yield framed[:7]
        yield framed[7:19]
        yield framed[19:]

    artifact = await materialize_route_artifact(chunks(), root=tmp_path)
    assert artifact.export == export
    assert artifact.local.object.locator.endswith(".routes")
    assert artifact.local.layout == source.layout

    provider = MaterializedRouteArtifactProvider((artifact,))
    operation = OperationRef(
        run_id="run",
        operation_id="d" * 64,
        sequence_id=1,
        learner_parent_version=0,
        kind="forward_backward",
    )
    handle = await provider.acquire(operation=operation, bundles=(artifact.local,))
    transfer = local_retained_route_bundle_transfer(
        (source,),
        (
            LocalRouteObjectView(
                source=source.object,
                path=artifact.local.object.locator,
            ),
        ),
        local_transfer_root=str(tmp_path),
    )
    assert await transfer.receive_payload(timeout_s=1) == payload
    assert (
        b"".join(
            [
                chunk
                async for chunk in provider.read_stream(
                    artifact.local.object, lease_id=artifact.local.lease_id
                )
            ]
        )
        == payload
    )
    await provider.release(handle)
    with pytest.raises(RuntimeError, match="no active ownership"):
        await anext(
            provider.read_stream(
                artifact.local.object, lease_id=artifact.local.lease_id
            )
        )


@pytest.mark.asyncio
async def test_route_artifact_rejects_truncated_payload(tmp_path) -> None:
    payload = b"x"
    source = _route_ref(b"xx", "holder-local:opaque:truncated")
    export = RouteArtifactExportReceipt.create(
        attempt_id="c" * 64,
        tenant_id="tenant",
        run_id="run",
        operation_id="d" * 64,
        source=source,
    )
    encoded = export.model_dump_json().encode()

    async def chunks():
        yield struct.pack("!I", len(encoded)) + encoded + payload

    with pytest.raises(RuntimeError, match="changed identity"):
        await materialize_route_artifact(chunks(), root=tmp_path)
