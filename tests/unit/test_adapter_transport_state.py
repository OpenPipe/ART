import base64
from types import SimpleNamespace

import pytest

from art.distributed import adapter_transport
from art.distributed.adapter_transport import (
    AdapterSnapshotReceiver,
    AdapterSnapshotSender,
    AdapterTransferTarget,
    NixlAdapterSender,
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
        "poisoned": False,
        "unreleased_handles": 0,
        "closed": False,
    }
    sender.close()
    assert sender.state().closed is True


class _Handle:
    def __init__(self, *, release_error: bool = False) -> None:
        self.release_error = release_error

    def release(self) -> None:
        if self.release_error:
            raise RuntimeError("transfer remains active")


class _Agent:
    name = "sender"

    def __init__(self, *, state: str, release_error: bool = False) -> None:
        self.state = state
        self.release_error = release_error
        self.removed: list[str] = []

    def get_xfer_descs(self, value, **_kwargs):
        return value

    def add_remote_agent(self, metadata: bytes) -> str:
        return metadata.decode()

    def remove_remote_agent(self, remote_agent: str) -> None:
        self.removed.append(remote_agent)

    def deregister_memory(self, _registration, **_kwargs) -> None:
        return None

    def initialize_xfer(self, *_args, **_kwargs):
        return _Handle(release_error=self.release_error)

    def transfer(self, _handle) -> str:
        return self.state

    def check_xfer_state(self, _handle) -> str:
        return self.state


def _nixl_target(*, epoch: str, timeout_s: float = 1) -> AdapterTransferTarget:
    return AdapterTransferTarget(
        host_id="inference-0",
        generation_id="generation-1",
        path="/adapter/generation-1",
        remote_agent=epoch,
        remote_metadata_b64=base64.b64encode(epoch.encode()).decode(),
        remote_address=1,
        remote_device_id=0,
        slot_id=0,
        capacity_bytes=1,
        prepare_s=0,
        pool_wait_s=0,
        registration_s=0,
        transfer_timeout_s=timeout_s,
    )


def _sender(agent: _Agent, monkeypatch) -> NixlAdapterSender:
    sender = NixlAdapterSender()
    sender._agent = agent
    sender._block = SimpleNamespace(
        numel=lambda: 1,
        narrow=lambda *_args: object(),
    )
    sender._registration = object()
    monkeypatch.setattr(sender, "_ensure_capacity", lambda _used_bytes: 0)
    monkeypatch.setattr(adapter_transport, "_copy_payload", lambda *_args: None)
    return sender


def test_nixl_sender_invalidates_restarted_receiver_metadata(monkeypatch) -> None:
    agent = _Agent(state="DONE")
    sender = _sender(agent, monkeypatch)
    payload = SimpleNamespace(nbytes=1)

    sender.send(payload, {}, (_nixl_target(epoch="receiver-1"),))
    sender.send(payload, {}, (_nixl_target(epoch="receiver-2"),))

    assert agent.removed == ["receiver-1"]
    assert sender.state().remote_agents == 1
    assert sender.state().completed_transfers == 2


def test_nixl_sender_never_reuses_buffer_after_failed_cancel(monkeypatch) -> None:
    sender = _sender(_Agent(state="PROC", release_error=True), monkeypatch)
    target = _nixl_target(epoch="receiver-1", timeout_s=0.000001)

    with pytest.raises(TimeoutError) as caught:
        sender.send(SimpleNamespace(nbytes=1), {}, (target,))

    assert any("sender is poisoned" in note for note in caught.value.__notes__)
    assert sender.state().poisoned is True
    assert sender.state().unreleased_handles == 1
    with pytest.raises(RuntimeError, match="requires a runtime restart"):
        sender.send(SimpleNamespace(nbytes=1), {}, (target,))

    sender._unreleased_handles[0].release_error = False
    sender.close()
    assert sender.state().closed is True
    assert sender.state().unreleased_handles == 0
