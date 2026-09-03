import asyncio
import hashlib
import json

from msgspec import msgpack
import numpy as np
import pytest

from art.distributed import (
    PackingRequestArtifact,
    PackingRequestManifest,
    PackingRequestSource,
)
from art.distributed.data_plane import ByteStreamTransfer, PackedBatchInbox
from art.distributed.packing import (
    PackingRequest,
    PackingResult,
    PackingTransferRequest,
    TokenMatrixBatchTransfer,
    encode_token_matrix_batch,
    resolve_retained_token_matrix_routes,
)
from art.training.token_matrix import (
    NamedLossRequest,
    RetainedTokenRoutes,
    TextDatum,
    TokenMatrix,
    TokenMatrixBatch,
    dense_row,
)
from art.vllm_route_transport import (
    RetainedRouteBundleRef,
    RouteBundleChoiceLayout,
    RouteBundleLayout,
    RouteBundleObjectRef,
    route_bundle_id,
)


def _batch() -> TokenMatrixBatch:
    return TokenMatrixBatch(
        matrices=(
            TokenMatrix(
                matrix_id="matrix-0",
                rows=(
                    dense_row("token_ids", "int64", (3,), (10, 11, 12)),
                    dense_row("target_token_ids", "int64", (3, 1), (11, 12, 0)),
                    dense_row("loss_weights", "float32", (3, 1), (1.0, 1.0, 0.0)),
                ),
            ),
        )
    )


def test_packing_request_manifest_retains_exact_sft_bootstrap() -> None:
    request = PackingRequest(
        batch=_batch(),
        loss=NamedLossRequest(name="cross_entropy", normalize_advantages=False),
        generation_id="length-objective-generation",
        packed_sequence_length=128,
    )
    source = PackingRequestSource(
        stage="length_trainability",
        command_index=1,
        learner_parent_version=0,
        text_datums=(
            TextDatum(
                datum_id="matrix-0",
                messages=(
                    {"role": "user", "content": "Return the learned marker."},
                    {"role": "assistant", "content": "MARK MARK"},
                ),
                assistant_turns="last",
            ),
        ),
    )
    manifest = PackingRequestManifest.create(
        context={"model_identity": {"base_model": "model", "handler": "dense"}},
        inputs=(PackingRequestArtifact.capture(request, source=source),),
    )

    restored = PackingRequestManifest.model_validate_json(manifest.canonical_bytes())

    assert restored == manifest
    assert restored.inputs[0].source.learner_parent_version == 0
    assert restored.inputs[0].source.text_datums[0].datum_id == "matrix-0"
    assert restored.inputs[0].request == request


def test_retained_route_sidecar_resolves_by_matrix_id() -> None:
    batch = _batch()
    token_ids = batch.matrices[0].row("token_ids").dense_values()
    route_payload = bytes((1, 2, 3))
    choice = RouteBundleChoiceLayout(
        choice_index=0,
        dtype="uint8",
        shape=(3, 1, 1),
        offset=0,
        byte_count=len(route_payload),
        token_ids_sha256=hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    sha256 = hashlib.sha256(route_payload).hexdigest()
    layout_identity = {
        "protocol_version": 1,
        "format": "art_inference_route_bundle_v1",
        "request_id": "request",
        "owner_id": "owner",
        "model_identity": "model",
        "response_id": "response",
        "num_experts": 4,
        "choices": (choice.model_dump(mode="json"),),
        "byte_count": len(route_payload),
        "sha256": sha256,
    }
    layout = RouteBundleLayout(
        bundle_id=route_bundle_id(layout_identity),
        request_id="request",
        owner_id="owner",
        model_identity="model",
        response_id="response",
        num_experts=4,
        choices=(choice,),
        byte_count=len(route_payload),
        sha256=sha256,
    )
    ref = RetainedRouteBundleRef(
        object=RouteBundleObjectRef(
            locator="object",
            size_bytes=len(route_payload),
            sha256=layout.sha256,
        ),
        layout=layout,
        lease_id="lease",
    )
    routed_batch = TokenMatrixBatch(
        matrices=batch.matrices,
        routes=(
            RetainedTokenRoutes(
                matrix_id="matrix-0",
                bundle=ref.model_dump(mode="json"),
                choice_index=0,
            ),
        ),
    )

    request = PackingRequest(
        batch=routed_batch,
        loss=NamedLossRequest(name="cross_entropy"),
        generation_id="generation",
        retained_route_bundles=(ref,),
        packed_sequence_length=8,
    )
    resolved = resolve_retained_token_matrix_routes(
        request.batch, (layout,), route_payload
    )

    assert resolved["matrix-0"].num_experts == 4
    assert np.asarray(resolved["matrix-0"]).reshape(-1).tolist() == [1, 2, 3]


@pytest.mark.asyncio
async def test_actor_packs_one_streamed_canonical_batch_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art.distributed.monarch_actor import ArtHostService
    import art.preprocessing.pack as pack_module

    batch = _batch()
    assert (
        TokenMatrixBatch.model_validate(
            msgpack.decode(encode_token_matrix_batch(batch))
        )
        == batch
    )
    request = PackingRequest(
        batch=batch,
        loss=NamedLossRequest(name="cross_entropy"),
        return_token_logprobs=False,
        generation_id="generation",
        packed_sequence_length=8,
        collect_packing_shapes=True,
    )
    legacy_fields = {
        "trajectory_groups",
        "tokenized_batch",
        "group_ids",
        "record_ids",
        "min_source_version",
        "max_source_version",
    }
    assert legacy_fields.isdisjoint(PackingRequest.model_fields)

    transfer = TokenMatrixBatchTransfer(
        stream=ByteStreamTransfer(
            stream_id="batch",
            host="127.0.0.1",
            port=1,
            token="0" * 64,
            byte_count=1,
        )
    )

    async def receive_batch(
        _transfer: TokenMatrixBatchTransfer, *, timeout_s: float
    ) -> TokenMatrixBatch:
        assert timeout_s == 1.0
        return batch

    monkeypatch.setattr(TokenMatrixBatchTransfer, "receive", receive_batch)
    calls = 0
    pack = pack_module.packed_tensors_from_token_matrices

    def counted_pack(*args, **kwargs):
        nonlocal calls
        calls += 1
        return pack(*args, **kwargs)

    monkeypatch.setattr(pack_module, "packed_tensors_from_token_matrices", counted_pack)
    service = object.__new__(ArtHostService)
    service.host_id = "host-a"
    service._packing_lock = asyncio.Lock()
    service._packed_batches = PackedBatchInbox(
        host_id=service.host_id, capacity_bytes=1 << 20
    )
    wire_request = PackingTransferRequest.from_request(
        request,
        batch_transfer=transfer,
        route_bundle_transfer=None,
    )
    operation = ArtHostService.__dict__["pack_batch"]
    try:
        result = await operation._method.__wrapped__(
            service, wire_request, "batch", 1.0
        )
    finally:
        service._packed_batches.store.close()

    assert calls == 1
    assert result.ref.training_outcome.accepted_trainable_tokens == 2
    assert result.ref.logical_loss_terms == 2
    assert result.ref.token_matrix_output_map.matrix_ids == ("matrix-0",)
    assert result.ref.prefix_tree_packing_stats is not None
    assert result.ref.prefix_tree_packing_stats.logical_tokens == 3
    assert result.ref.prefix_tree_packing_stats.physical_tokens == 3
    assert result.packed_group_shapes[0].leaves[0].matrix_id == "matrix-0"
    assert {
        "trainable_assistant_tokens",
        "loss_bearing_tokens",
        "non_padding_tokens",
    }.isdisjoint(PackingResult.model_fields)
