import torch

from art.megatron.tensor_snapshot import (
    PendingCpuSnapshot,
    PinnedCpuSnapshotStager,
    SnapshotReadBarrier,
    _CudaFence,
)
from art.megatron.weights.lora_publish import _stage_published_tensors


def _cuda() -> torch.device:
    assert torch.cuda.is_available()
    return torch.device("cuda", 0)


def test_grouped_staging_preserves_values_layout_order_and_aliases() -> None:
    device = _cuda()
    shared = torch.arange(24, dtype=torch.float32, device=device).view(6, 4)
    tensors = {
        "z_regular": torch.full((2, 5), 91.0, device=device),
        "b_alias": shared[2:],
        "half": torch.arange(7, dtype=torch.float16, device=device),
        "a_alias": shared[:4],
        "m_noncontiguous": torch.arange(15, dtype=torch.float32, device=device)
        .view(3, 5)
        .t(),
    }
    expected = {key: tensor.cpu() for key, tensor in tensors.items()}
    source_strides = {key: tensor.stride() for key, tensor in tensors.items()}

    builder = PinnedCpuSnapshotStager(reusable=True).begin()
    staged = _stage_published_tensors(tensors, builder)
    pending = builder.finish(staged)
    assert len(pending._sources) == 4
    result = pending.resolve()

    assert tuple(result) == (
        "b_alias",
        "a_alias",
        "half",
        "m_noncontiguous",
        "z_regular",
    )
    for key, tensor in result.items():
        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
        assert tensor.shape == tensors[key].shape
        assert tensor.stride() == (
            source_strides[key]
            if key in {"a_alias", "b_alias"}
            else tensors[key].contiguous().stride()
        )
    assert pending._sources == ()

    float_storage = result["a_alias"].untyped_storage().data_ptr()
    assert result["b_alias"].untyped_storage().data_ptr() == float_storage
    assert result["m_noncontiguous"].untyped_storage().data_ptr() == float_storage
    assert result["z_regular"].untyped_storage().data_ptr() == float_storage
    assert result["half"].untyped_storage().data_ptr() != float_storage
    assert (
        result["b_alias"].storage_offset() - result["a_alias"].storage_offset()
        == tensors["b_alias"].storage_offset() - tensors["a_alias"].storage_offset()
    )
    result["a_alias"][2, 0] = -123.0
    assert result["b_alias"][0, 0].item() == -123.0


def test_grouped_staging_reuses_sorted_target_layout() -> None:
    device = _cuda()
    stager = PinnedCpuSnapshotStager(reusable=True)

    def stage(items: list[tuple[str, float]]) -> tuple[dict[str, int], int]:
        tensors = {
            key: torch.full((1024,), value, dtype=torch.float32, device=device)
            for key, value in items
        }
        builder = stager.begin()
        result = builder.finish(_stage_published_tensors(tensors, builder)).resolve()
        offsets = {key: int(tensor.storage_offset()) for key, tensor in result.items()}
        return offsets, next(iter(result.values())).untyped_storage().data_ptr()

    first_offsets, first_storage = stage([("z", 3.0), ("a", 1.0), ("m", 2.0)])
    stager.reset()
    second_offsets, second_storage = stage([("m", 5.0), ("z", 6.0), ("a", 4.0)])

    assert first_offsets == second_offsets
    assert first_offsets["a"] < first_offsets["m"] < first_offsets["z"]
    assert first_storage == second_storage
    assert stager._buffers is not None and len(stager._buffers) == 1


def test_grouped_staging_has_no_aggregate_cuda_allocation() -> None:
    device = _cuda()
    tensors = {
        f"tensor_{index:02d}": torch.full(
            (2 * 1024 * 1024,), index, dtype=torch.bfloat16, device=device
        )
        for index in range(8)
    }
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)

    builder = PinnedCpuSnapshotStager().begin()
    pending = builder.finish(_stage_published_tensors(tensors, builder))
    peak_extra = torch.cuda.max_memory_allocated(device) - allocated
    result = pending.resolve()

    assert peak_extra == 0
    for index, tensor in enumerate(result.values()):
        assert tensor.flatten()[0].item() == index


def test_snapshot_barrier_key_does_not_fence_another_run() -> None:
    device = torch.cuda.current_device()
    delayed_stream = torch.cuda.Stream(device=device)
    mutation_stream = torch.cuda.Stream(device=device)
    delayed = torch.cuda.Event(blocking=True)
    ready = torch.cuda.Event(blocking=True)
    with torch.cuda.stream(delayed_stream):
        torch.cuda._sleep(2_000_000_000)
        delayed.record()
    ready.record(torch.cuda.current_stream(device))
    ready.synchronize()

    barrier = SnapshotReadBarrier()
    barrier.register(
        PendingCpuSnapshot(None, (_CudaFence(device, delayed),), ()), key="run_a"
    )
    barrier.register(
        PendingCpuSnapshot(None, (_CudaFence(device, ready),), ()), key="run_b"
    )
    mutation_done = torch.cuda.Event(blocking=True)
    with torch.cuda.stream(mutation_stream):
        barrier.wait_before_mutation(key="run_b")
        mutation_done.record()
    mutation_done.synchronize()

    assert not delayed.query()
    global_wait_done = torch.cuda.Event(blocking=True)
    with torch.cuda.stream(mutation_stream):
        barrier.wait_before_mutation()
        global_wait_done.record()
    global_wait_done.synchronize()
    assert delayed.query()

    barrier.register(PendingCpuSnapshot(None, (_CudaFence(device, delayed),), ()))
    barrier.synchronize()
