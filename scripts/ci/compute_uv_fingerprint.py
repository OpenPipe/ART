#!/usr/bin/env python3
"""Compute a stable fingerprint for the prek CI uv-cache contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute full prek CI dependency fingerprint used by uv-cache gating."
        )
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--uv-lock",
        type=Path,
        default=Path("uv.lock"),
        help="Path to uv.lock",
    )
    parser.add_argument(
        "--base-image",
        default="pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel",
        help="Base image reference used for CI runtime/build cache compatibility",
    )
    parser.add_argument(
        "--python-mm",
        default="3.11",
        help="Python major.minor string used in CI (for example: 3.11)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=16,
        help="Fingerprint length (hex chars)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.pyproject.exists():
        raise SystemExit(f"pyproject file not found: {args.pyproject}")
    if not args.uv_lock.exists():
        raise SystemExit(f"uv lock file not found: {args.uv_lock}")

    payload: dict[str, Any] = {
        "inputs": {
            "pyproject_sha256": _sha256_file(args.pyproject),
            "uv_lock_sha256": _sha256_file(args.uv_lock),
        },
        "ci_context": {
            "fingerprint_schema_version": 3,
            "cache_kind": "full_uv_cache",
            "cache_scope": "prek_all_extras_group_dev",
            "cache_target": "uv_cache",
            "cache_python_platform": "linux_x86_64",
            "cache_package_manager": "uv",
            "cache_link_mode": "copy",
            "cache_asset_layout": "release_chunked_parts_v1",
        },
    }
    payload["ci_context"].update(
        {
            "base_image": args.base_image,
            "python_mm": args.python_mm,
        }
    )
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    print(digest[: args.length])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
