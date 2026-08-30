from types import SimpleNamespace

import httpx
import pytest

from art.megatron.paired_inference import (
    MegatronPairedInferencePublisher,
    _paired_lora_transport,
)


class _ReceiptClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.requests: list[tuple[str, dict[str, object]]] = []

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str] | None,
        timeout: float,
    ) -> httpx.Response:
        del headers, timeout
        self.requests.append((url, json))
        return self.response


def _publisher() -> MegatronPairedInferencePublisher:
    publisher = object.__new__(MegatronPairedInferencePublisher)
    publisher.service = SimpleNamespace(
        leader_endpoint=SimpleNamespace(url="http://holder.test:8000")
    )
    publisher.api_key = "secret"
    return publisher


def test_paired_lora_transport_follows_resolved_placement() -> None:
    spec = SimpleNamespace(
        trainer_mesh=SimpleNamespace(ranks=(SimpleNamespace(host_id="trainer"),))
    )

    assert (
        _paired_lora_transport(
            spec,
            SimpleNamespace(members=(SimpleNamespace(host_id="trainer"),)),
        )
        == "local"
    )
    assert (
        _paired_lora_transport(
            spec,
            SimpleNamespace(members=(SimpleNamespace(host_id="inference"),)),
        )
        == "nixl"
    )


@pytest.mark.asyncio
async def test_lost_update_response_recovers_exact_holder_receipt() -> None:
    holder_result = {
        "status": "updated",
        "generation_id": "generation-2",
        "update_seq": 2,
        "update_identity": "update-2",
        "apply_s": 0.125,
    }
    request = httpx.Request(
        "POST", "http://holder.test:8000/art/in_flight_lora_update/receipt"
    )
    client = _ReceiptClient(
        httpx.Response(
            200,
            request=request,
            json={
                "operation_id": "operation-2",
                "state": "settled",
                "response_status": 200,
                "response": holder_result,
            },
        )
    )
    payload: dict[str, object] = {
        "operation_id": "operation-2",
        "generation_id": "generation-2",
    }

    recovered = await _publisher()._recover_update_receipt(
        client,
        payload,
        httpx.ReadError("response lost", request=request),
    )

    assert recovered == holder_result
    assert client.requests == [
        (
            "http://holder.test:8000/art/in_flight_lora_update/receipt",
            payload,
        )
    ]


@pytest.mark.asyncio
async def test_ambiguous_holder_receipt_does_not_reexecute_update() -> None:
    request = httpx.Request(
        "POST", "http://holder.test:8000/art/in_flight_lora_update/receipt"
    )
    client = _ReceiptClient(
        httpx.Response(
            200,
            request=request,
            json={"operation_id": "operation-2", "state": "ambiguous"},
        )
    )

    with pytest.raises(RuntimeError, match="outcome is ambiguous"):
        await _publisher()._recover_update_receipt(
            client,
            {"operation_id": "operation-2"},
            httpx.ReadError("response lost", request=request),
        )

    assert len(client.requests) == 1
