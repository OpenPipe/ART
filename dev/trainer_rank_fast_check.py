from __future__ import annotations

from importlib.util import find_spec
import subprocess
import sys

FAST_TESTS = (
    "tests/unit/test_trainer_rank_validation.py",
    "tests/unit/test_trainer_rank_weird_shapes.py",
    "tests/unit/test_prefix_tree_packing.py",
)

MEGATRON_FAST_TESTS = (
    "tests/unit/test_prefix_tree.py",
    "tests/unit/test_prefix_tree_attention_builder.py",
    "tests/unit/test_prefix_tree_grad_parity.py",
)


def main() -> None:
    tests = (*FAST_TESTS, *(MEGATRON_FAST_TESTS if find_spec("megatron") else ()))
    command = [sys.executable, "-m", "pytest", "--tb=short", *tests, *sys.argv[1:]]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
