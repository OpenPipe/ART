from types import SimpleNamespace

import pytest
import torch

import art.megatron.runtime.executor as executor_module
from art.megatron.runtime.executor import _GenerationPublisher
from art.utils.safetensors import PreparedSafetensors, SafetensorsLayout


def _profile(rank: int, value: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        tensors={
            "adapter_A.weight": torch.full((2, rank), value),
            "adapter_B.weight": torch.full((rank, 3), value + 1),
        }
    )


def _aliased_profile(value: float) -> SimpleNamespace:
    storage = torch.arange(12, dtype=torch.float32) + value
    return SimpleNamespace(
        tensors={
            "adapter_A.weight": storage[:4],
            "adapter_B.weight": storage[4:],
        }
    )


def _prepared_bytes(prepared: PreparedSafetensors) -> bytes:
    return b"".join(chunk.numpy().tobytes() for chunk in prepared.chunks)


def _count_layout_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[str, tuple[int, ...]], ...]]:
    builds: list[tuple[tuple[str, tuple[int, ...]], ...]] = []

    def build(tensors: dict[str, torch.Tensor]) -> SafetensorsLayout:
        builds.append(
            tuple((name, tuple(tensor.shape)) for name, tensor in tensors.items())
        )
        return SafetensorsLayout(tensors)

    monkeypatch.setattr(executor_module, "SafetensorsLayout", build)
    return builds


@pytest.mark.parametrize("capacity", (2, 4))
def test_lora_layout_cache_plateaus_and_keeps_hot_shapes(
    monkeypatch: pytest.MonkeyPatch, capacity: int
) -> None:
    builds = _count_layout_builds(monkeypatch)
    publisher = _GenerationPublisher(SimpleNamespace(), capacity=capacity)
    expected_capacity = 2 * capacity
    try:
        for rank in range(1, 65):
            publisher._prepare_lora_tensors(_profile(rank))
            assert len(publisher._lora_layouts) <= expected_capacity

        assert publisher._lora_layout_capacity == expected_capacity
        assert len(publisher._lora_layouts) == expected_capacity
        assert len(builds) == 64

        for value in range(128):
            hot = _profile(64, float(value))
            prepared = publisher._prepare_lora_tensors(hot)
            expected = SafetensorsLayout(hot.tensors).bind(hot.tensors)
            assert _prepared_bytes(prepared) == _prepared_bytes(expected)

        assert len(publisher._lora_layouts) == expected_capacity
        assert len(builds) == 64
    finally:
        publisher.close()


def test_lora_layout_cache_is_lru_and_keys_storage_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds = _count_layout_builds(monkeypatch)
    publisher = _GenerationPublisher(SimpleNamespace(), capacity=2)
    try:
        for rank in range(1, 5):
            publisher._prepare_lora_tensors(_profile(rank))

        publisher._prepare_lora_tensors(_profile(1))
        publisher._prepare_lora_tensors(_profile(5))
        publisher._prepare_lora_tensors(_profile(1))
        assert len(builds) == 5

        publisher._prepare_lora_tensors(_profile(2))
        assert len(builds) == 6

        aliased = _aliased_profile(0)
        publisher._prepare_lora_tensors(aliased)
        rebuilt = _aliased_profile(100)
        prepared = publisher._prepare_lora_tensors(rebuilt)
        expected = SafetensorsLayout(rebuilt.tensors).bind(rebuilt.tensors)
        assert _prepared_bytes(prepared) == _prepared_bytes(expected)
        aliased_build_count = len(builds)

        split = SimpleNamespace(
            tensors={name: tensor.clone() for name, tensor in rebuilt.tensors.items()}
        )
        publisher._prepare_lora_tensors(split)
        assert len(builds) == aliased_build_count + 1
        assert len(publisher._lora_layouts) == 4
    finally:
        publisher.close()
