from __future__ import annotations

import asyncio
from pathlib import Path

from art_vllm_runtime.binary_routes import encode_routed_experts_response_parts
from art_vllm_runtime.local_route_store import (
    LocalRouteStore,
    encode_route_object_header,
)
import numpy as np

from art.vllm_route_transport import (
    LocalRouteObjectView,
    local_retained_route_bundle_transfer,
    retained_local_route_bundle_from_response,
)


def test_local_route_object_binds_exact_binary_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ART_VLLM_ROUTE_SHM_ROOT", str(tmp_path))
    response = (
        b'{"id":"response","choices":[{"finish_reason":"stop","index":0,'
        b'"logprobs":null,"message":{"content":"ok","role":"assistant"},'
        b'"output_token_ids":[13]}],"created":0,"model":"model",'
        b'"object":"chat.completion","prompt_token_ids":[11,12]}'
    )
    body, route_payload = encode_routed_experts_response_parts(
        response,
        {0: np.asarray([[[1]], [[2]], [[3]]], dtype=np.uint8)},
        num_experts=8,
    )
    store = LocalRouteStore("runtime")
    ref = store.retain("a" * 64, route_payload)

    completion, bundle = retained_local_route_bundle_from_response(
        body,
        object_header=encode_route_object_header(ref),
        request_id="request",
        owner_id="runtime",
        model_identity="model@generation",
        lease_id="route-capture-request",
    )

    assert completion.id == "response"
    assert bundle.object.model_dump(mode="json") == ref
    assert Path(bundle.object.locator).read_bytes() == route_payload
    assert bundle.layout.byte_count == len(route_payload)
    assert store.accepts_transfer_root(str(tmp_path))
    assert store.read_many((ref,)) == route_payload
    transfer = local_retained_route_bundle_transfer(
        (bundle,),
        (LocalRouteObjectView(source=bundle.object, path=store.local_view(ref)),),
        local_transfer_root=str(tmp_path),
    )
    assert asyncio.run(transfer.receive_payload(timeout_s=1)) == route_payload
    store.release(ref)
    assert not Path(bundle.object.locator).exists()
