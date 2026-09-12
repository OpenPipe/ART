"""CPU microbenchmark; no private corpus or timing assertions.

Run: uv run --no-project --python 3.12 python scripts/benchmark_assistant_prefix.py
The exact production helper AST is loaded without importing the ML runtime.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


def linear(left, right):
    matched = 0
    while (
        matched < len(left) and matched < len(right) and left[matched] == right[matched]
    ):
        matched += 1
    return matched


def inputs(kind, size, position):
    if kind == "tokens":
        left = [257 + (i * 17) % 100003 for i in range(size)]
        right = [257 + (i * 17) % 100003 for i in range(size)]
    else:
        alphabet = "abcdefghijk" if kind == "ascii" else "α中文🌀́"
        left = (alphabet * (size // len(alphabet) + 1))[:size]
        right = (left + "!")[:-1]
    if position == "proper_prefix":
        right = right[:-1]
    elif position != "equal" and size:
        index = {
            "early": 0,
            "near_start": min(3, size - 1),
            "middle": size // 2,
            "late": size - 1,
        }[position]
        if isinstance(right, list):
            right[index] = -1009
        else:
            right = right[:index] + "!" + right[index + 1 :]
    return left, right


def measure(function, left, right, repeats):
    iterations = 1
    while True:
        start = time.perf_counter()
        for _ in range(iterations):
            function(left, right)
        if time.perf_counter() - start >= 0.003 or iterations >= 16384:
            break
        iterations *= 2
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            function(left, right)
        samples.append((time.perf_counter() - start) / iterations)
    return {"median_seconds": statistics.median(samples), "iterations": iterations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[0, 16, 1024, 65536])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 0 for size in args.sizes):
        parser.error("sizes must be nonnegative; repeats must be positive")
    source = Path(__file__).resolve().parents[1] / "src/art/trajectories/_tokenize.py"
    raw = source.read_bytes()
    node = next(
        node
        for node in ast.parse(raw).body
        if isinstance(node, ast.FunctionDef) and node.name == "_common_prefix_length"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    candidate = namespace["_common_prefix_length"]
    rows = []
    for kind in ("ascii", "unicode", "tokens"):
        for size in args.sizes:
            positions = (
                ("equal",)
                if size == 0
                else ("early", "near_start", "middle", "late", "equal", "proper_prefix")
            )
            for position in positions:
                left, right = inputs(kind, size, position)
                expected = linear(left, right)
                assert candidate(left, right) == expected
                order = [("linear", linear), ("candidate", candidate)]
                if len(rows) % 2:
                    order.reverse()
                measurements = {
                    name: measure(function, left, right, args.repeats)
                    for name, function in order
                }
                baseline, improved = measurements["linear"], measurements["candidate"]
                rows.append(
                    dict(
                        kind=kind,
                        size=size,
                        position=position,
                        prefix=expected,
                        linear=baseline,
                        candidate=improved,
                        speedup=baseline["median_seconds"] / improved["median_seconds"],
                    )
                )
    assert source.read_bytes() == raw, "production source changed during measurement"
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                implementation=platform.python_implementation(),
                source_sha256=hashlib.sha256(raw).hexdigest(),
                helper_ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest(),
                repeats=args.repeats,
                results=rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
