from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


def _load_parser_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "src/art/megatron/gdn/gdn_prefix_tree.py"
    spec = importlib.util.spec_from_file_location(
        "_art_gdn_prefix_tree_for_tests", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load parser module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MODULE = _load_parser_module()

GdnPackedExecutionSpec: Any = _MODULE.GdnPackedExecutionSpec
build_gdn_rank_execution_plan: Any = _MODULE.build_gdn_rank_execution_plan
parse_gdn_prefix_tree_segments: Any = _MODULE.parse_gdn_prefix_tree_segments
