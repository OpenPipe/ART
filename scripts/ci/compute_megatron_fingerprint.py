#!/usr/bin/env python3
"""Compute a stable fingerprint for the Megatron CI prebuilt image contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        normalized = [_normalize(item) for item in value]
        if all(isinstance(item, str) for item in normalized):
            return sorted(normalized)
        return normalized
    return value


def _select_megatron_payload(pyproject_data: dict[str, Any]) -> dict[str, Any]:
    project = pyproject_data.get("project", {})
    tool = pyproject_data.get("tool", {})
    uv = tool.get("uv", {})

    optional_dependencies = project.get("optional-dependencies", {})
    megatron_dependencies = optional_dependencies.get("megatron", [])

    uv_sources = uv.get("sources", {})
    uv_overrides = uv.get("override-dependencies", [])
    uv_no_build_isolation = uv.get("no-build-isolation-package", [])
    uv_extra_build_dependencies = uv.get("extra-build-dependencies", {})
    uv_extra_build_variables = uv.get("extra-build-variables", {})

    relevant_override_prefixes = (
        "apex",
        "megatron",
        "ml-dtypes",
        "nv-grouped-gemm",
        "numpy",
        "transformer-engine",
    )
    relevant_overrides = [
        dep
        for dep in uv_overrides
        if dep.split()[0].startswith(relevant_override_prefixes)
    ]

    relevant_extra_build_dependency_keys = [
        key
        for key in ("apex", "transformer-engine-torch", "nv-grouped-gemm")
        if key in uv_extra_build_dependencies
    ]
    relevant_extra_build_variable_keys = [
        key
        for key in ("apex", "transformer-engine-torch", "nv-grouped-gemm")
        if key in uv_extra_build_variables
    ]

    payload = {
        "requires_python": project.get("requires-python"),
        "megatron_dependencies": megatron_dependencies,
        "uv": {
            "sources": {"apex": uv_sources.get("apex")},
            "override_dependencies": relevant_overrides,
            "no_build_isolation_package": uv_no_build_isolation,
            "extra_build_dependencies": {
                key: uv_extra_build_dependencies[key]
                for key in relevant_extra_build_dependency_keys
            },
            "extra_build_variables": {
                key: uv_extra_build_variables[key]
                for key in relevant_extra_build_variable_keys
            },
        },
    }

    return _normalize(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Megatron dependency fingerprint used by prebuilt CI image gating."
        )
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--base-image",
        default="pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel",
        help="Base image reference used for the prebuilt CI image",
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

    pyproject_data = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    payload = _select_megatron_payload(pyproject_data)
    payload["ci_context"] = {
        "base_image": args.base_image,
        "python_mm": args.python_mm,
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    print(digest[: args.length])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
