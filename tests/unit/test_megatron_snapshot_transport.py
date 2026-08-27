from types import SimpleNamespace
from typing import Any

import pytest

from art.megatron.runtime import executor


@pytest.mark.parametrize("has_prepared_identity", [False, True])
def test_live_snapshot_transport_defers_missing_identity_to_receiver(
    monkeypatch: pytest.MonkeyPatch,
    has_prepared_identity: bool,
) -> None:
    prepared_identity = object() if has_prepared_identity else None
    transfers: list[dict[str, Any]] = []
    tensors = object()

    def transfer(lora: object, targets: tuple[object, ...], **kwargs: Any) -> None:
        transfers.append({"lora": lora, "targets": targets, **kwargs})

    publisher = object.__new__(executor._GenerationPublisher)
    publisher.runtime = SimpleNamespace(rank=0)
    publisher._transfer_lora_snapshot = transfer
    target = object()
    prepared = SimpleNamespace(
        distributed_adapter=None,
        publication_targets=(target,),
        adapter_object_target=None,
        adapter=SimpleNamespace(
            lora="lora",
            tensors=tensors,
            model_identity=prepared_identity,
        ),
    )

    result = publisher._transfer_prepared_snapshot(prepared, 0.0)

    assert transfers == [
        {
            "lora": "lora",
            "targets": (target,),
            "prepared_tensors": tensors,
            "model_identity": prepared_identity,
        }
    ]
    assert "time/snapshot_transport_identity_s" not in result.metrics
