from collections import deque
from itertools import islice
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import NamedTuple

import torch

_DTYPES = {
    dtype: name
    for name, dtype in {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "C64": torch.complex64,
        "U16": getattr(torch, "uint16", None),
        "U32": getattr(torch, "uint32", None),
        "U64": getattr(torch, "uint64", None),
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
    }.items()
    if dtype is not None
}


class PreparedSafetensors(NamedTuple):
    chunks: tuple[torch.Tensor, ...]

    @property
    def nbytes(self) -> int:
        return sum(chunk.numel() for chunk in self.chunks)


class _TensorLayout(NamedTuple):
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    storage: int
    offset: int
    nbytes: int


class _StorageLayout(NamedTuple):
    nbytes: int
    chunks: tuple[tuple[int, int], ...]


class SafetensorsLayout:
    """Reusable file layout for immutable CPU snapshots with stable shapes."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        storage_indices: dict[tuple[int, int], int] = {}
        storages: list[list[tuple[int, int]]] = []
        storage_bytes: list[int] = []
        entries: list[_TensorLayout] = []
        for name, tensor in sorted(tensors.items()):
            _validate_tensor(name, tensor)
            storage = tensor.untyped_storage()
            key = storage.data_ptr(), storage.nbytes()
            storage_index = storage_indices.get(key)
            if storage_index is None:
                storage_index = len(storages)
                storage_indices[key] = storage_index
                storages.append([])
                storage_bytes.append(storage.nbytes())
            offset = tensor.data_ptr() - storage.data_ptr()
            entries.append(
                _TensorLayout(
                    name,
                    tensor.dtype,
                    tuple(tensor.shape),
                    storage_index,
                    offset,
                    tensor.nbytes,
                )
            )
            storages[storage_index].append((offset, tensor.nbytes))

        layouts: list[_StorageLayout] = []
        for size, intervals in zip(storage_bytes, storages, strict=True):
            ordered = sorted(intervals)
            cursor = 0
            coalesced = True
            for offset, length in ordered:
                if offset != cursor:
                    coalesced = False
                    break
                cursor += length
            layouts.append(
                _StorageLayout(
                    size,
                    ((0, size),) if coalesced and cursor == size else tuple(intervals),
                )
            )

        data_offsets: dict[str, tuple[int, int]] = {}
        output_offset = 0
        for storage_index, layout in enumerate(layouts):
            storage_entries = [
                entry for entry in entries if entry.storage == storage_index
            ]
            if len(layout.chunks) == 1 and layout.chunks[0] == (0, layout.nbytes):
                for entry in storage_entries:
                    data_offsets[entry.name] = (
                        output_offset + entry.offset,
                        output_offset + entry.offset + entry.nbytes,
                    )
                output_offset += layout.nbytes
                continue
            for entry in storage_entries:
                data_offsets[entry.name] = (
                    output_offset,
                    output_offset + entry.nbytes,
                )
                output_offset += entry.nbytes

        header = {
            entry.name: {
                "dtype": _DTYPES[entry.dtype],
                "shape": list(entry.shape),
                "data_offsets": list(data_offsets[entry.name]),
            }
            for entry in entries
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        self._entries = tuple(entries)
        self._storages = tuple(layouts)
        self._prefix = torch.frombuffer(
            bytearray(struct.pack("<Q", len(encoded)) + encoded), dtype=torch.uint8
        )

    def bind(self, tensors: dict[str, torch.Tensor]) -> PreparedSafetensors:
        bound: list[torch.Tensor | None] = [None] * len(self._storages)
        for entry in self._entries:
            tensor = tensors.get(entry.name)
            if tensor is None:
                raise RuntimeError(f"Safetensors tensor disappeared: {entry.name}")
            _validate_tensor(entry.name, tensor)
            storage = tensor.untyped_storage()
            if (
                tensor.dtype != entry.dtype
                or tuple(tensor.shape) != entry.shape
                or storage.nbytes() != self._storages[entry.storage].nbytes
                or tensor.data_ptr() - storage.data_ptr() != entry.offset
            ):
                raise RuntimeError(f"Safetensors tensor layout changed: {entry.name}")
            owner = bound[entry.storage]
            if owner is None:
                bound[entry.storage] = torch.empty(0, dtype=torch.uint8).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
            elif owner.untyped_storage().data_ptr() != storage.data_ptr():
                raise RuntimeError("Safetensors storage aliasing changed")
        if len(tensors) != len(self._entries):
            raise RuntimeError("Safetensors tensor set changed")
        owners = tuple(owner for owner in bound if owner is not None)
        if len(owners) != len(bound):
            raise RuntimeError("Safetensors storage disappeared")
        return PreparedSafetensors(
            (
                self._prefix,
                *(
                    owner.narrow(0, offset, length)
                    for owner, layout in zip(owners, self._storages, strict=True)
                    for offset, length in layout.chunks
                ),
            )
        )


def _writev_all(fd: int, buffers: list[memoryview]) -> None:
    pending = deque(buffer for buffer in buffers if buffer.nbytes)
    iov_max = os.sysconf("SC_IOV_MAX")
    while pending:
        written = os.writev(fd, tuple(islice(pending, iov_max)))
        if written <= 0:
            raise OSError("Short vectored write")
        while pending and written >= pending[0].nbytes:
            written -= pending.popleft().nbytes
        if written:
            pending[0] = pending[0][written:]


def _validate_tensor(name: str, tensor: torch.Tensor) -> None:
    if sys.byteorder != "little":
        raise RuntimeError("ART's zero-copy safetensors writer requires little endian")
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise RuntimeError(f"Tensor {name!r} must be contiguous CPU storage")
    if tensor.dtype not in _DTYPES:
        raise RuntimeError(f"Unsupported safetensors dtype: {tensor.dtype}")


def prepare_safetensors(tensors: dict[str, torch.Tensor]) -> PreparedSafetensors:
    entries: list[tuple[str, torch.Tensor]] = []
    data_offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, tensor in sorted(tensors.items()):
        _validate_tensor(name, tensor)
        entries.append((name, tensor))
        data_offsets[name] = offset, offset + tensor.nbytes
        offset += tensor.nbytes
    header = {
        name: {
            "dtype": _DTYPES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": list(data_offsets[name]),
        }
        for name, tensor in entries
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    prefix = torch.frombuffer(
        bytearray(struct.pack("<Q", len(encoded)) + encoded), dtype=torch.uint8
    )
    return PreparedSafetensors(
        (prefix, *(tensor.reshape(-1).view(torch.uint8) for _name, tensor in entries))
    )


def save_prepared_safetensors(prepared: PreparedSafetensors, path: Path) -> None:
    """Stream a prepared safetensors payload without rebuilding tensor metadata."""
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / path.name
        with temporary_path.open("wb", buffering=0) as output:
            _writev_all(
                output.fileno(),
                [memoryview(chunk.numpy()) for chunk in prepared.chunks],
            )
        temporary_path.replace(path)


def save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Stream CPU tensor buffers without copying them into GIL-held bytes."""
    save_prepared_safetensors(prepare_safetensors(tensors), path)
