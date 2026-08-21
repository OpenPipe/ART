from multiprocessing import get_context
import os
from pathlib import Path

import pytest
import torch

from art.megatron.runtime.nvme_residency import (
    NvmeResidencyStore,
    NvmeResidencyStoreConfig,
)
from art.megatron.runtime.residency import ResidencyKey
from art.megatron.runtime.tensor_residency import TensorResidencyMover
from art.utils import disk_admission as disk_admission_module
from art.utils.disk_admission import (
    DiskAdmission,
    DiskAdmissionConfig,
    DiskAdmissionManifest,
    DiskCatalogClaim,
)

_MIB = 1 << 20


def _admission(tmp_path: Path, *, runtime_floor: int = 0) -> DiskAdmission:
    return DiskAdmission(
        DiskAdmissionConfig(
            shared_storage_mount=tmp_path,
            storage_identity=f"test-{tmp_path.name}",
            node_identity="test-node",
            runtime_free_floor_bytes=runtime_floor,
            progress_update_bytes=_MIB,
        )
    )


def _manifest(admission: DiskAdmission) -> DiskAdmissionManifest:
    return DiskAdmissionManifest.model_validate_json(
        admission.manifest_path.read_bytes()
    )


def _write_claimed_reservation_and_crash(tmp_path: Path, planned_bytes: int) -> None:
    admission = _admission(tmp_path)
    lease = admission.reserve(
        incoming_peak_bytes=planned_bytes,
        purpose="test",
        owned_paths=(tmp_path / "payload.bin",),
        catalog_claim=DiskCatalogClaim(
            kind="test",
            claim_id="claim",
            claim_owner="owner",
            claim_epoch=1,
        ),
    )
    (tmp_path / "payload.bin").write_bytes(bytes(planned_bytes))
    lease.record_written(planned_bytes)
    os._exit(0)


def test_progress_is_bounded_and_does_not_fsync_global_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admission = _admission(tmp_path)
    original_fsync = disk_admission_module.os.fsync
    fsyncs: list[int] = []

    def counted_fsync(descriptor: int) -> None:
        fsyncs.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(disk_admission_module.os, "fsync", counted_fsync)
    target = tmp_path / "payload.bin"
    lease = admission.reserve(
        incoming_peak_bytes=4 * _MIB,
        purpose="test",
        owned_paths=(target,),
    )
    assert len(fsyncs) == 2

    target.write_bytes(bytes(4 * _MIB))
    for written_bytes in range(_MIB, 5 * _MIB, _MIB):
        lease.record_written(written_bytes)

    active = admission.active_reservations()[0]
    durable = _manifest(admission).reservations[lease.reservation_id]
    assert active.remaining_bytes == 0
    assert durable.remaining_bytes == durable.planned_bytes
    assert lease._lease_path.stat().st_size <= (
        disk_admission_module._PROGRESS_SLOT_BYTES
        * disk_admission_module._PROGRESS_SLOTS
    )
    assert len(fsyncs) == 2

    lease.complete()

    closed = _manifest(admission).reservations[lease.reservation_id]
    assert (closed.state, closed.remaining_bytes) == ("completed", 0)
    assert len(fsyncs) == 4


def test_nvme_chunks_add_no_progress_fsyncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = NvmeResidencyStore(
        NvmeResidencyStoreConfig(
            root=str(tmp_path / "l3"),
            io_chunk_bytes=_MIB,
            disk_admission=DiskAdmissionConfig(
                shared_storage_mount=tmp_path,
                storage_identity=f"test-{tmp_path.name}",
                node_identity="test-node",
                runtime_free_floor_bytes=0,
                progress_update_bytes=_MIB,
            ),
        )
    )
    image = TensorResidencyMover().host_image(
        (torch.zeros(3 * _MIB, dtype=torch.uint8),)
    )
    key = ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id="generation",
        topology_fingerprint="topology",
        adapter_layout_fingerprint="layout",
    )
    original_fsync = disk_admission_module.os.fsync
    fsyncs: list[int] = []

    def counted_fsync(descriptor: int) -> None:
        fsyncs.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(disk_admission_module.os, "fsync", counted_fsync)

    manifest = store.write(key, image)

    assert manifest.payload_bytes == 3 * _MIB
    assert len(fsyncs) == 8
    assert store._disk_admission.active_reservations() == ()


def test_live_progress_is_sampled_before_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def free_space(_path: Path) -> int:
        events.append("free")
        return 10 * _MIB

    monkeypatch.setattr(disk_admission_module, "_statvfs_free", free_space)
    admission = _admission(tmp_path, runtime_floor=3 * _MIB)
    first_path = tmp_path / "first.bin"
    first = admission.reserve(
        incoming_peak_bytes=4 * _MIB,
        purpose="test",
        owned_paths=(first_path,),
    )
    first_path.write_bytes(bytes(2 * _MIB))
    first.record_written(2 * _MIB)

    original_read = disk_admission_module._read_lease_progress

    def read_progress(path: Path, mount: Path, planned_bytes: int) -> int:
        events.append("progress")
        return original_read(path, mount, planned_bytes)

    monkeypatch.setattr(disk_admission_module, "_read_lease_progress", read_progress)
    events.clear()
    second = admission.reserve(
        incoming_peak_bytes=4 * _MIB,
        purpose="test",
        owned_paths=(tmp_path / "second.bin",),
    )

    assert events == ["progress", "free"]
    second.cancel()
    first.cancel()


def test_torn_latest_progress_slot_retains_prior_progress(tmp_path: Path) -> None:
    admission = _admission(tmp_path)
    target = tmp_path / "payload.bin"
    lease = admission.reserve(
        incoming_peak_bytes=4 * _MIB,
        purpose="test",
        owned_paths=(target,),
    )
    target.write_bytes(bytes(2 * _MIB))
    lease.record_written(_MIB)
    lease.record_written(2 * _MIB)
    os.pwrite(
        lease._lease_descriptor,
        b"\0",
        disk_admission_module._PROGRESS_SLOT_BYTES,
    )

    assert admission.active_reservations()[0].remaining_bytes == 3 * _MIB
    lease.cancel()


def test_crash_discards_claimed_reservation_volatile_progress(
    tmp_path: Path,
) -> None:
    planned_bytes = 2 * _MIB
    child = get_context("spawn").Process(
        target=_write_claimed_reservation_and_crash,
        args=(tmp_path, planned_bytes),
    )
    child.start()
    child.join()
    assert child.exitcode == 0
    admission = _admission(tmp_path)
    durable = next(iter(_manifest(admission).reservations.values()))
    assert (
        disk_admission_module._read_lease_progress(
            Path(durable.lease_path), admission.mount, planned_bytes
        )
        == planned_bytes
    )

    active = admission.active_reservations()
    assert len(active) == 1
    assert active[0].remaining_bytes == planned_bytes
    assert admission.reap_dead_reservations(lambda _claim: False) == (
        durable.reservation_id,
    )
