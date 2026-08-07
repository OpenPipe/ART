import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

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


def save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    """Stream CPU tensor buffers without copying them into GIL-held bytes."""
    if sys.byteorder != "little":
        raise RuntimeError("ART's zero-copy safetensors writer requires little endian")
    header: dict[str, Any] = {}
    buffers: list[memoryview] = []
    offset = 0
    for name, tensor in sorted(tensors.items()):
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise RuntimeError(f"Tensor {name!r} must be contiguous CPU storage")
        dtype = _DTYPES.get(tensor.dtype)
        if dtype is None:
            raise RuntimeError(f"Unsupported safetensors dtype: {tensor.dtype}")
        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + data.nbytes],
        }
        offset += data.nbytes
        buffers.append(data)

    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / path.name
        with temporary_path.open("wb", buffering=0) as output:
            for data in (
                memoryview(struct.pack("<Q", len(encoded))),
                encoded,
                *buffers,
            ):
                while data:
                    written = output.write(data)
                    if not written:
                        raise OSError(f"Short write while saving {path}")
                    data = data[written:]
        temporary_path.replace(path)
