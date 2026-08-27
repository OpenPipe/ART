import base64
import hashlib
from types import SimpleNamespace

import torch

from art.distributed import adapter_transport
from art.utils.safetensors import FileIdentity, PreparedSafetensors


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
