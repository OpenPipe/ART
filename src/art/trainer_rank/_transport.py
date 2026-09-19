"""Storage-aware snapshots shared by client operations and physical commands."""

from io import BytesIO
from typing import Any

import cloudpickle
import torch


def encode(value: Any) -> bytes:
    # Torch preserves shared storages; cloudpickle supports callback-local types.
    stream = BytesIO()
    torch.save(value, stream, pickle_module=cloudpickle)
    return stream.getvalue()


def decode(payload: bytes) -> Any:
    # Sender CUDA ordinals are not receiver devices. Native handlers place tensors.
    return torch.load(BytesIO(payload), map_location="cpu", weights_only=False)
