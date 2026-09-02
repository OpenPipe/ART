import hashlib
import struct
from types import SimpleNamespace

import pytest

from art.route_artifacts import (
    MaterializedRouteArtifactProvider,
    RouteArtifactExportReceipt,
    materialize_route_artifact,
)
from art.training import OperationRef
from art.vllm_route_transport import (
    RetainedRouteBundleRef,
    RouteBundleChoiceLayout,
    RouteBundleLayout,
    RouteBundleObjectRef,
    local_retained_route_bundle_transfer,
    route_bundle_id,
)


def test_opaque_holder_route_never_selects_path_loader() -> None:
    from art.distributed.art_runtime import _route_refs_are_local_files

    ref = SimpleNamespace(object=SimpleNamespace(locator="holder-local:owner:bundle"))
    assert not _route_refs_are_local_files((ref,))  # type: ignore[arg-type]

    ref.object.locator = "/dev/shm/art_vllm_routes/bundle.routes"
    assert _route_refs_are_local_files((ref,))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_route_artifact_stream_materializes_for_exact_operation(
    tmp_path, monkeypatch
) -> None:
    payload = b"exact-routes"
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
    source = RetainedRouteBundleRef(
        object=RouteBundleObjectRef(
            store="holder_local",
            locator=f"holder-local:opaque:{layout.bundle_id}",
            size_bytes=len(payload),
            sha256=layout.sha256,
        ),
        layout=layout,
        lease_id="route-capture-" + "b" * 64,
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
    monkeypatch.setenv("ART_VLLM_ROUTE_SHM_ROOT", str(tmp_path))
    transfer = local_retained_route_bundle_transfer((artifact.local,))
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
    choice = RouteBundleChoiceLayout(
        choice_index=0,
        dtype="uint8",
        shape=(2, 1, 1),
        offset=0,
        byte_count=2,
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
        "byte_count": 2,
        "sha256": hashlib.sha256(b"xx").hexdigest(),
    }
    layout = RouteBundleLayout(bundle_id=route_bundle_id(identity), **identity)
    source = RetainedRouteBundleRef(
        object=RouteBundleObjectRef(
            store="holder_local",
            locator="holder-local:opaque:" + layout.bundle_id,
            size_bytes=2,
            sha256=layout.sha256,
        ),
        layout=layout,
        lease_id="route-capture-" + "b" * 64,
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
        yield struct.pack("!I", len(encoded)) + encoded + payload

    with pytest.raises(RuntimeError, match="changed identity"):
        await materialize_route_artifact(chunks(), root=tmp_path)
