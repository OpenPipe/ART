from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from msgspec import msgpack
from pydantic import BaseModel, ConfigDict, Field
from safetensors import safe_open
import torch

from art.utils.safetensors import (
    FileIdentity,
    PreparedSafetensors,
    prepare_safetensors,
    save_prepared_safetensors,
)

_METADATA = "__art_optimizer_state__"
_TENSOR_PREFIX = "tensor_"
_TENSOR = 0
_TUPLE = 1
_LIST = 2
_DICT = 3
_SCALAR = 4


class PreparedOptimizerArchive(BaseModel):
    """Exact, write-ready optimizer archive backed by immutable CPU tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    payload: PreparedSafetensors
    tensor_count: int = Field(ge=0)

    @property
    def nbytes(self) -> int:
        return self.payload.nbytes

    def write(self, path: Path) -> FileIdentity:
        return save_prepared_safetensors(self.payload, path)


def prepare_optimizer_archive(state: Any) -> PreparedOptimizerArchive:
    tensors: list[torch.Tensor] = []
    indices: dict[int, int] = {}

    def encode(value: Any) -> list[Any]:
        if isinstance(value, torch.Tensor):
            index = indices.get(id(value))
            if index is None:
                index = len(tensors)
                indices[id(value)] = index
                tensors.append(value)
            return [_TENSOR, index]
        if isinstance(value, tuple):
            if hasattr(value, "_fields"):
                raise TypeError("optimizer archives do not support named tuples")
            return [_TUPLE, [encode(item) for item in value]]
        if isinstance(value, list):
            return [_LIST, [encode(item) for item in value]]
        if isinstance(value, dict):
            return [_DICT, [[encode(key), encode(item)] for key, item in value.items()]]
        if value is None or type(value) in {bool, int, float, str, bytes}:
            return [_SCALAR, value]
        raise TypeError(f"unsupported optimizer archive value: {type(value)!r}")

    metadata = torch.frombuffer(
        bytearray(msgpack.encode(encode(state))), dtype=torch.uint8
    )
    named = {
        **{
            f"{_TENSOR_PREFIX}{index:08d}": tensor
            for index, tensor in enumerate(tensors)
        },
        _METADATA: metadata,
    }
    return PreparedOptimizerArchive(
        payload=prepare_safetensors(named), tensor_count=len(tensors)
    )


def load_optimizer_archive(path: Path) -> Any:
    with safe_open(path, framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        if _METADATA not in keys:
            raise RuntimeError("optimizer archive has no state metadata")
        metadata = archive.get_tensor(_METADATA)
        if metadata.dtype != torch.uint8 or metadata.ndim != 1:
            raise RuntimeError("optimizer archive metadata tensor is invalid")
        node = msgpack.decode(memoryview(metadata.numpy()))
        tensor_names = tuple(sorted(keys - {_METADATA}))
        expected = tuple(
            f"{_TENSOR_PREFIX}{index:08d}" for index in range(len(tensor_names))
        )
        if tensor_names != expected:
            raise RuntimeError("optimizer archive tensor coverage is invalid")
        tensors = tuple(archive.get_tensor(name) for name in tensor_names)

    def decode(value: Any) -> Any:
        if not isinstance(value, list) or len(value) != 2 or type(value[0]) is not int:
            raise RuntimeError("optimizer archive state node is invalid")
        tag, payload = value
        if tag == _TENSOR:
            if type(payload) is not int or payload < 0 or payload >= len(tensors):
                raise RuntimeError("optimizer archive tensor reference is invalid")
            return tensors[payload]
        if tag in {_TUPLE, _LIST}:
            if not isinstance(payload, list):
                raise RuntimeError("optimizer archive sequence node is invalid")
            items = [decode(item) for item in payload]
            return tuple(items) if tag == _TUPLE else items
        if tag == _DICT:
            if not isinstance(payload, list):
                raise RuntimeError("optimizer archive mapping node is invalid")
            result = {}
            for pair in payload:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise RuntimeError("optimizer archive mapping entry is invalid")
                result[decode(pair[0])] = decode(pair[1])
            return result
        if tag == _SCALAR and (
            payload is None or type(payload) in {bool, int, float, str, bytes}
        ):
            return payload
        raise RuntimeError("optimizer archive state tag is invalid")

    return decode(node)
