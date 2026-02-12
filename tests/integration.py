import argparse
import ast
from pathlib import Path
import re
import sys
from typing import Any, Dict

from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import nbformat
import pytest

NOTEBOOKS = [
    {
        "path": "../examples/temporal_clue/temporal-clue.ipynb",
        "variables": {
            "TRAINING_STEPS": 1,
        },
    },
    {
        "path": "../examples/prisoners-dilemma.ipynb",
        "variables": {
            "TRAINING_STEPS": 1,
            "PRISONERS_DILEMMA_ROUNDS": 10,
        },
    },
    {
        "path": "../examples/art-e.ipynb",
        "variables": {
            "training_config": {
                "groups_per_step": 2,
                "num_epochs": 1,
                "rollouts_per_group": 4,
                "learning_rate": 1e-5,
                "max_steps": 1,
            },
        },
    },
    {
        "path": "../examples/rock-paper-tool-use.ipynb",
        "variables": {"TRAINING_STEPS": 1},
    },
]


def make_patch_source() -> str:
    """
    This is a patch to the art.TrainableModel class to change project name logging these runs as test runs.
    """
    project = "Tester"
    return (
        "try:\n"
        "    import art as _art\n"
        "    import art.model as _art_model\n"
        "except Exception:\n"
        "    pass\n"
        "else:\n"
        "    _orig_tm_init = _art_model.TrainableModel.__init__\n"
        "    def _patched_tm_init(self, *args, **kwargs):\n"
        f"        kwargs['project'] = {project!r}\n"
        "        result = _orig_tm_init(self, *args, **kwargs)\n"
        "        return result\n"
        "    _art_model.TrainableModel.__init__ = _patched_tm_init\n"
        "    _art.TrainableModel = _art_model.TrainableModel\n"
    )


def make_variable_override_source(variables: Dict[str, Any]) -> str:
    """
    Create source code to override variables in the notebook.
    """
    if not variables:
        return ""

    lines = ["# Variable overrides for testing"]
    for var_name, var_value in variables.items():
        lines.append(f"{var_name} = {var_value!r}")

    return "\n".join(lines)


def _override_variables_in_notebook(nb, variables: Dict[str, Any]) -> list[str]:
    """Replace top-level assignments inside code cells.

    Replaces entire assignment statements (including multi-line values) using AST
    ranges to avoid leaving trailing lines that can break indentation.

    Returns the list of variable names that were NOT found and replaced.
    """
    if not variables:
        return []

    replaced: set[str] = set()

    for cell in nb.cells:
        if getattr(cell, "cell_type", "") != "code":
            continue
        source: str = getattr(cell, "source", "") or ""
        if not source:
            continue

        # Try AST-based replacement first
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None

        if tree is not None:
            lines = source.splitlines(keepends=True)
            pending_edits: list[tuple[int, int, str]] = []

            for node in getattr(tree, "body", []) or []:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    for t in targets:
                        if isinstance(t, ast.Name):
                            var_name = t.id
                            if var_name in variables and var_name not in replaced:
                                start_ln = getattr(node, "lineno", 1) - 1
                                end_ln = (
                                    getattr(
                                        node, "end_lineno", getattr(node, "lineno", 1)
                                    )
                                    - 1
                                )
                                # Preserve exact indentation used at the start line
                                indent_match = re.match(r"[ \t]*", lines[start_ln])
                                indent = indent_match.group(0) if indent_match else ""
                                replacement = (
                                    f"{indent}{var_name} = {variables[var_name]!r}\n"
                                )
                                pending_edits.append((start_ln, end_ln, replacement))
                                replaced.add(var_name)
                                break

            if pending_edits:
                # Apply edits from bottom to top to keep indices valid
                pending_edits.sort(key=lambda e: e[0], reverse=True)
                for start_ln, end_ln, replacement in pending_edits:
                    lines[start_ln : end_ln + 1] = [replacement]
                cell.source = "".join(lines)
                continue

        # Fallback to single-line regex replacement if AST failed or found nothing
        for var_name, var_value in variables.items():
            if var_name in replaced:
                continue
            pattern = re.compile(
                rf"^([ \t]*){re.escape(var_name)}\s*=\s*.*$", re.MULTILINE
            )

            def _sub(m):
                replaced.add(var_name)
                return f"{m.group(1)}{var_name} = {var_value!r}"

            source_new, n = pattern.subn(_sub, source, count=1)
            if n:
                source = source_new
        cell.source = source

    missing = [name for name in variables.keys() if name not in replaced]
    return missing


class _NotebookPlugin:
    def __init__(self, notebook_configs: list[dict]) -> None:
        self.notebook_configs = notebook_configs

    def pytest_generate_tests(self, metafunc) -> None:
        if "notebook_config" in metafunc.fixturenames:
            metafunc.parametrize("notebook_config", self.notebook_configs)


def test_notebook_execution(notebook_config: dict) -> None:
    notebook_path = notebook_config["path"]
    variables = notebook_config.get("variables", {})

    p = Path(notebook_path).resolve()
    if not (p.is_file() and p.suffix == ".ipynb"):
        pytest.skip(f"Notebook not found or invalid: {p}")

    nb = nbformat.read(p, as_version=4)

    # Replace variables directly inside existing cells first
    missing_variables: list[str] = _override_variables_in_notebook(nb, variables)

    # Insert the patch source at the beginning
    nb.cells.insert(0, nbformat.v4.new_code_cell(source=make_patch_source()))

    # For variables not found in the notebook, insert a small override cell
    if missing_variables:
        override_source = make_variable_override_source(
            {k: variables[k] for k in missing_variables}
        )
        nb.cells.insert(1, nbformat.v4.new_code_cell(source=override_source))

    try:
        NotebookClient(nb).execute(cwd=str(p.parent))
    except CellExecutionError as e:
        pytest.fail(str(e))
    # Echo cell outputs to stdout/stderr so prints are visible in console
    for cell in nb.cells:
        if getattr(cell, "cell_type", None) != "code":
            continue
        for output in getattr(cell, "outputs", []) or []:
            output_type = output.get("output_type")
            if output_type == "stream":
                text = output.get("text", "")
                name = output.get("name", "stdout")
                if name == "stderr":
                    print(text, end="", file=sys.stderr)
                else:
                    print(text, end="")
            elif output_type in ("execute_result", "display_data"):
                data = output.get("data", {}) or {}
                if "text/plain" in data:
                    print(data["text/plain"])
            elif output_type == "error":
                traceback_lines = output.get("traceback") or []
                if traceback_lines:
                    print("\n".join(traceback_lines), file=sys.stderr)

    # Print success message after test completes successfully
    notebook_name = Path(notebook_path).name
    print(f"\n✅ Test completed successfully: {notebook_name}")
    print("-" * 50)


def parse_indexes(indexes_str: str, max_index: int) -> list[int]:
    """
    Parse index specification string into list of valid indexes.

    Supports:
    - Single indexes: "0", "2"
    - Comma-separated: "0,2,3"
    - Ranges: "1-3" (inclusive)
    - Mixed: "0,2-4,6"

    Args:
        indexes_str: String specification of indexes
        max_index: Maximum valid index (exclusive)

    Returns:
        List of valid, unique indexes in ascending order
    """
    indexes = set()

    try:
        for part in indexes_str.split(","):
            part = part.strip()
            if not part:
                continue

            if "-" in part:
                # Handle range like "1-3"
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip())
                end = int(end_str.strip())

                if start > end:
                    print(f"Warning: Invalid range '{part}' - start > end")
                    continue

                for i in range(start, end + 1):
                    if 0 <= i < max_index:
                        indexes.add(i)
                    else:
                        print(f"Warning: Index {i} is out of range (0-{max_index - 1})")
            else:
                # Handle single index
                index = int(part)
                if 0 <= index < max_index:
                    indexes.add(index)
                else:
                    print(f"Warning: Index {index} is out of range (0-{max_index - 1})")

    except ValueError as e:
        print(f"Error parsing indexes '{indexes_str}': {e}")
        return []

    return sorted(list(indexes))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run integration tests for notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available notebooks (by index):
{chr(10).join(f"  {i}: {config['path']}" for i, config in enumerate(NOTEBOOKS))}

Examples:
  uv run integration.py              # Run all notebooks
  uv run integration.py -n 0         # Run only notebook 0
  uv run integration.py -n 0,2       # Run notebooks 0 and 2
  uv run integration.py -n 1-3       # Run notebooks 1, 2, and 3
""",
    )
    parser.add_argument(
        "--notebooks",
        "-n",
        type=str,
        help="Comma-separated list of notebook indexes to run (e.g., '0,2' or '1-3'). If not specified, all notebooks will be run.",
    )

    args = parser.parse_args()

    here = Path(__file__).parent.resolve()

    # Determine which notebooks to run
    if args.notebooks:
        selected_indexes = parse_indexes(args.notebooks, len(NOTEBOOKS))
        if not selected_indexes:
            print("Error: No valid notebook indexes specified")
            return 1
        selected_configs = [NOTEBOOKS[i] for i in selected_indexes]
        print(f"Running notebooks at indexes: {sorted(selected_indexes)}")
    else:
        selected_configs = NOTEBOOKS
        print("Running all notebooks")

    # Process notebook configurations
    processed_configs = []
    for config in selected_configs:
        if isinstance(config, str):
            # Handle legacy string format
            processed_config = {"path": config, "variables": {}}
        else:
            processed_config = config.copy()

        # Resolve path relative to this file
        p = (here / processed_config["path"]).resolve()  # ty:ignore[unsupported-operator]
        if not p.exists():
            print(f"Warning: notebook not found: {p}")
        processed_config["path"] = str(p)
        processed_configs.append(processed_config)

    # Invoke pytest programmatically, injecting notebook params via a plugin
    # -s: do not capture stdout so notebook output is visible
    # --maxfail=1: stop on first failure to terminate the whole training run
    pytest_args = [
        "-s",
        "--maxfail=1",
        str(here / "integration.py"),
    ]
    result_code = pytest.main(
        args=pytest_args, plugins=[_NotebookPlugin(processed_configs)]
    )
    return int(result_code)


if __name__ == "__main__":
    sys.exit(main())
