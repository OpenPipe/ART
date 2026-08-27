import base64
import hashlib
from types import SimpleNamespace

import torch

from art.distributed import adapter_transport
from art.utils.safetensors import (
    FileIdentity,
    PreparedSafetensors,
    prepare_safetensors,
)


def test_nixl_receiver_computes_identity_from_received_bytes(tmp_path) -> None:
    generation_id = "generation-1"
    payload = torch.arange(64, dtype=torch.uint8)
    slot = adapter_transport._RegisteredSlot(0, payload.clone(), None)
    slot.generation_id = generation_id
    target = SimpleNamespace(
        path=str(tmp_path / generation_id),
        slot_id=0,
        capacity_bytes=payload.numel(),
        prepare_s=0.0,
        pool_wait_s=0.0,
        registration_s=0.0,
    )
    config = b"{}"
    notification = adapter_transport.AdapterTransferNotification(
        generation_id=generation_id,
        used_bytes=payload.numel(),
        config_identity=FileIdentity(
            size_bytes=len(config),
            sha256=hashlib.sha256(config).hexdigest(),
        ),
        adapter_config_b64=base64.b64encode(config).decode(),
        sender_staging_s=0.0,
        sender_registration_s=0.0,
    )
    receiver = object.__new__(adapter_transport.AdapterSnapshotReceiver)
    receiver.host_id = "host-1"
    receiver._pending = {
        generation_id: adapter_transport._PendingReceive(target=target, slot=slot)
    }
    receiver._local_pending = {}
    receiver._materialized = set()
    receiver._take_notification = lambda _generation_id: notification
    receiver._finish = lambda _generation_id: None

    result = receiver.poll(generation_id)

    assert result is not None
    expected = hashlib.sha256(memoryview(payload.numpy())).hexdigest()
    assert result.model_identity == FileIdentity(
        size_bytes=payload.numel(), sha256=expected
    )
    assert (tmp_path / generation_id / "adapter_model.safetensors").read_bytes() == bytes(
        payload.tolist()
    )


def test_local_receiver_computes_identity_from_materialized_file(
    tmp_path, monkeypatch
) -> None:
    generation_id = "generation-local"
    tensors = {"weight": torch.arange(32, dtype=torch.bfloat16)}
    prepared = prepare_safetensors(tensors)
    template = tmp_path / "template"
    template.mkdir()
    (template / "adapter_model.safetensors").write_bytes(b"0" * prepared.nbytes)
    (template / "adapter_config.json").write_text(
        '{"art_lora_format":"vllm"}', encoding="utf-8"
    )
    monkeypatch.setenv("ART_LOCAL_ADAPTER_TRANSFER_ROOT", str(tmp_path / "local"))
    receiver = adapter_transport.AdapterSnapshotReceiver(
        "host-1", str(tmp_path / "receiver")
    )
    target = receiver.prepare(
        generation_id, str(template), transport="local", timeout_s=1.0
    )
    snapshot = SimpleNamespace(
        tensors=tensors,
        adapter_config={"base_model_name_or_path": "fixture"},
    )

    adapter_transport.AdapterSnapshotSender().send(
        snapshot,
        (target,),
        prepared_tensors=prepared,
        model_identity=None,
    )
    result = receiver.poll(generation_id)

    assert result is not None
    payload = (tmp_path / "local" / "host-1" / generation_id / "adapter_model.safetensors")
    assert result.model_identity == FileIdentity(
        size_bytes=payload.stat().st_size,
        sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
    )
    receiver.release(generation_id)
