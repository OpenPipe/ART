from __future__ import annotations

import subprocess
import sys

FAST_TESTS = (
    "tests/unit/test_trainer_rank_validation.py",
    "tests/unit/test_trainer_rank_weird_shapes.py",
    "tests/unit/test_shared_prefix_packing.py",
    "tests/unit/test_shared_prefix_tree.py",
    "tests/unit/test_shared_prefix_attention_builder.py",
    "tests/unit/test_shared_prefix_grad_parity.py",
)


def main() -> None:
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "pytest", "--tb=short", *FAST_TESTS, *sys.argv[1:]]
        )
    )


if __name__ == "__main__":
    main()
