from types import SimpleNamespace

from art.distributed.adapter_transport import (
    AdapterSnapshotReceiver,
    AdapterSnapshotSender,
    AdapterTransferTarget,
)


def test_adapter_transport_state_is_bounded_and_survives_close(
    tmp_path, monkeypatch
) -> None:
    receiver = AdapterSnapshotReceiver("inference-0", str(tmp_path), pool_capacity=2)
    assert receiver.state().model_dump() == {
        "schema_version": 1,
        "host_id": "inference-0",
        "pool_capacity": 2,
        "registered_slots": 0,
        "registered_capacity_bytes": 0,
        "active_registered_slots": 0,
        "pending_nixl_receives": 0,
        "pending_local_receives": 0,
        "materialized_generations": 0,
        "pending_notifications": 0,
        "closed": False,
    }
    receiver.close()
    assert receiver.state().closed is True

    sender = AdapterSnapshotSender()
    monkeypatch.setattr(sender, "_send_local", lambda *_args, **_kwargs: None)
    target = AdapterTransferTarget(
        transport="local",
        host_id="inference-0",
        generation_id="generation-1",
        path="/dev/shm/generation-1",
        remote_agent="/tmp/adapter.sock",
        remote_metadata_b64="-",
        remote_address=0,
        remote_device_id=0,
        slot_id=0,
        capacity_bytes=1,
        prepare_s=0,
        pool_wait_s=0,
        registration_s=0,
    )
    sender.send(
        SimpleNamespace(adapter_config={}),
        (target,),
        prepared_tensors=SimpleNamespace(),
    )
    assert sender.state().model_dump() == {
        "schema_version": 1,
        "transport": "local",
        "active_transfers": 0,
        "completed_transfers": 1,
        "registered_buffers": 0,
        "registered_capacity_bytes": 0,
        "remote_agents": 0,
        "closed": False,
    }
    sender.close()
    assert sender.state().closed is True
