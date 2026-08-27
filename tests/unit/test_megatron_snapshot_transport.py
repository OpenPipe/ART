from types import SimpleNamespace
from typing import Any

import pytest

from art.megatron.runtime import executor


@pytest.mark.parametrize("has_prepared_identity", [False, True])
def test_live_snapshot_transport_resolves_exact_identity_off_prepare_path(
    monkeypatch: pytest.MonkeyPatch,
    has_prepared_identity: bool,
) -> None:
    computed_identity = object()
    prepared_identity = object() if has_prepared_identity else None
    identity_calls: list[object] = []
    transfers: list[dict[str, Any]] = []
    tensors = object()

    def identity(payload: object) -> object:
        identity_calls.append(payload)
        return computed_identity

    def transfer(lora: object, targets: tuple[object, ...], **kwargs: Any) -> None:
        transfers.append({"lora": lora, "targets": targets, **kwargs})

    monkeypatch.setattr(executor, "prepared_safetensors_identity", identity)
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

    expected_identity = prepared_identity or computed_identity
    assert identity_calls == ([] if has_prepared_identity else [tensors])
    assert transfers == [
        {
            "lora": "lora",
            "targets": (target,),
            "prepared_tensors": tensors,
            "model_identity": expected_identity,
        }
    ]
    assert result.metrics["time/snapshot_transport_identity_s"] >= 0.0
