from __future__ import annotations

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


def _has_megatron() -> bool:
    try:
        import megatron.core.packed_seq_params  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def main() -> None:
    tests = (*FAST_TESTS, *(MEGATRON_FAST_TESTS if _has_megatron() else ()))
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "pytest", "--tb=short", *tests, *sys.argv[1:]]
        )
    )


if __name__ == "__main__":
    main()
