"""Landing acceptance: no hardcoded planner policy anywhere in production.

The holistic-planner landing contract removes user-selected depth, width,
head-chunk, and memory-safety decisions. Deleting the knobs from the public
signature is not enough: the same policies must not survive as hardcoded
constants or internal parameter defaults. This module statically scans
production sources and fails on any residue.

Exemptions, per the behavior spec:
- The low-level Megatron packing primitive may retain ``max_depth`` for tests
  and specialized preprocessing (``src/art/megatron/prefix_tree_packing.py``
  and its ``prefix_tree*`` siblings). TrainerRank must never call it with a
  literal policy value.

These tests are EXPECTED TO FAIL until the holistic planner lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_RANK_ROOT = REPO_ROOT / "src" / "art" / "trainer_rank"
SRC_ROOT = REPO_ROOT / "src" / "art"

BANNED_KNOBS = (
    "shared_prefix_max_depth",
    "head_chunk_tokens",
    "memory_safety_factor",
    "memory_reserve_fraction",
)

PACKING_PRIMITIVE_EXEMPT = (
    SRC_ROOT / "megatron" / "prefix_tree.py",
    SRC_ROOT / "megatron" / "prefix_tree_packing.py",
    SRC_ROOT / "megatron" / "prefix_tree_state.py",
)


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _violations_in_file(path: Path, banned: tuple[str, ...]) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            names = [
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
            ]
            for name in names:
                if name in banned:
                    found.append(
                        f"{path}:{node.lineno}: parameter {name!r} in {node.name}()"
                    )
        elif isinstance(node, ast.keyword) and node.arg in banned:
            found.append(f"{path}:{node.lineno}: keyword argument {node.arg!r}")
        elif isinstance(node, (ast.Name, ast.Attribute)):
            identifier = node.id if isinstance(node, ast.Name) else node.attr
            if identifier in banned:
                found.append(f"{path}:{node.lineno}: identifier {identifier!r}")
    return found


def test_trainer_rank_package_has_no_policy_knob_identifiers() -> None:
    violations: list[str] = []
    for path in _python_files(TRAINER_RANK_ROOT):
        violations.extend(_violations_in_file(path, BANNED_KNOBS))
    assert not violations, (
        "planner policy knobs survive inside src/art/trainer_rank:\n"
        + "\n".join(violations)
    )


def test_no_caller_passes_policy_knobs() -> None:
    violations: list[str] = []
    for path in _python_files(SRC_ROOT):
        if path in PACKING_PRIMITIVE_EXEMPT or TRAINER_RANK_ROOT in path.parents:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg in BANNED_KNOBS:
                violations.append(
                    f"{path}:{node.lineno}: keyword argument {node.arg!r}"
                )
    assert not violations, "callers still pass planner policy knobs:\n" + "\n".join(
        violations
    )


def test_trainer_rank_never_hardcodes_a_depth_policy() -> None:
    """`max_depth=<literal>` inside TrainerRank is a hardcoded sharing policy.

    The planner must derive depth decisions from data, model, topology, and
    memory facts. Passing a literal depth to the packing primitive (or any
    helper) from production TrainerRank code reintroduces the old fixed-depth
    behavior by the back door. The single exception is ``max_depth=0``: a
    no-sharing token count is not a sharing policy — it shares nothing and is
    only valid as a conservative upper-bound estimate.
    """

    violations: list[str] = []
    for path in _python_files(TRAINER_RANK_ROOT):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.keyword)
                and node.arg == "max_depth"
                and isinstance(node.value, ast.Constant)
                and node.value.value != 0
            ):
                violations.append(
                    f"{path}:{node.lineno}: max_depth={node.value.value!r}"
                )
    assert not violations, (
        "TrainerRank hardcodes a sharing depth policy:\n" + "\n".join(violations)
    )


@pytest.mark.parametrize("knob", BANNED_KNOBS)
def test_docs_and_examples_do_not_teach_the_removed_knob(knob: str) -> None:
    """User-facing docs must not keep teaching removed knobs.

    Scans documentation and examples (not source) for the knob name so stale
    guidance is caught at landing time. Historical changelogs are exempt.
    """

    offenders: list[str] = []
    for root in (REPO_ROOT / "docs", REPO_ROOT / "examples"):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".md", ".mdx", ".py", ".ipynb"):
                continue
            if "changelog" in path.name.lower():
                continue
            if knob in path.read_text(errors="ignore"):
                offenders.append(str(path))
    assert not offenders, (
        f"documentation/examples still reference removed knob {knob!r}:\n"
        + "\n".join(offenders)
    )
