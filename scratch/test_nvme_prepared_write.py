from pathlib import Path

import torch

from art.megatron.runtime.nvme_residency import (
    NvmeResidencyStore,
    NvmeResidencyStoreConfig,
)
from art.megatron.runtime.residency import ResidencyKey
from art.megatron.runtime.tensor_residency import TensorResidencyMover
from art.utils.disk_admission import DiskAdmissionConfig


def test_prepared_write_reuses_one_layout_and_does_not_reparse(
    tmp_path: Path, monkeypatch
) -> None:
    store = NvmeResidencyStore(
        NvmeResidencyStoreConfig(
            root=str(tmp_path / "l3"),
            disk_admission=DiskAdmissionConfig(
                shared_storage_mount=tmp_path,
                storage_identity=f"test-{tmp_path.name}",
                node_identity="test-node",
                runtime_free_floor_bytes=0,
            ),
        )
    )
    key = ResidencyKey(
        tenant_id="tenant",
        run_id="run",
        generation_id="generation",
        representation="weights",
        topology_fingerprint="topology",
        adapter_layout_fingerprint="adapter",
    )
    source = torch.arange(64, dtype=torch.float32)
    image = TensorResidencyMover().host_image((source,))
    builds = reads = 0
    build = store._manifest
    read = store._read_manifest

    def counted_build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return build(*args, **kwargs)

    def counted_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return read(*args, **kwargs)

    monkeypatch.setattr(store, "_manifest", counted_build)
    monkeypatch.setattr(store, "_read_manifest", counted_read)

    plan = store.prepare_write(key, image)
    manifest = store.write_prepared(plan, image)
    mapped = store.map_newly_committed(plan, (source,))

    assert manifest == plan.manifest
    assert builds == 1
    assert reads == 0
    assert torch.equal(mapped.tensors()[0], source)
