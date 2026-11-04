"""Process-lifetime forward certificate; checkpoint replay sees the same policy.

Missing entries deliberately retain the existing persistent implementation.
An admitted pinned entry never falls back on an unsupported M/layout/alignment.
"""

from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch


def signature(a, b):
    return str((a.shape[1], b.shape[1], str(a.dtype), a.stride(), b.stride()))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def certificate():
    path = os.environ.get("ART_INVARIANT_CERTIFICATE")
    if path is None:
        return None
    if digest(path) != os.environ.get("ART_INVARIANT_CERTIFICATE_SHA256"):
        raise RuntimeError("forward certificate hash mismatch")
    data = json.loads(Path(path).read_text())
    library = data["library"]
    if digest(library) != data["library_sha256"]:
        raise RuntimeError("pinned binding changed after admission")
    if (
        torch.__version__ != "2.11.0+cu128"
        or torch.version.cuda != "12.8"
        or importlib.metadata.version("transformer-engine") != "2.11.0"
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        raise RuntimeError(
            "forward certificate requires Torch 2.11/CUDA 12.8/TE 2.11/SM90"
        )
    return data


@lru_cache(maxsize=1)
def binding():
    from .lt import Binding

    return Binding(certificate()["library"])


def forward(a, b):
    from .fast_ops import persistent

    if len(a) == 0:
        return a.new_empty((0, b.shape[1]))
    data = certificate()
    entry = None if data is None else data["gemms"].get(signature(a, b))
    if entry is None:
        if data is not None and any(
            key.startswith(str((a.shape[1], b.shape[1], str(a.dtype)))[:-1] + ",")
            for key in data["gemms"]
        ):
            raise RuntimeError("layout outside pinned certificate; no fallback")
        return persistent(a, b)
    if entry.get("backend") == "triton":
        return persistent(a, b)
    if not entry["equality"] or not entry["accuracy"] or not entry["faster"]:
        raise RuntimeError("uncertified pinned GEMM entry")
    if len(a) not in entry["m_values"]:
        raise RuntimeError(f"M={len(a)} outside pinned certificate; no fallback")
    # Separate storage for every invocation on the caller's stream. The Torch
    # allocator records stream ownership; no shared mutable workspace races.
    workspace = torch.empty(8 << 20, dtype=torch.uint8, device=a.device)
    return binding().run(a, b, bytes.fromhex(entry["opaque_hex"]), workspace)[0]


def norm(x, weight, eps, zero_centered):
    data = certificate()
    key = str((x.shape[-1], float(eps), bool(zero_centered), str(x.dtype)))
    if data is None or key not in data["norms"]:
        return None
    from .te_norm import pad4

    return pad4(x, weight, eps, zero_centered)
