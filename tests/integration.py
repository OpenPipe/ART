import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


NOTEBOOKS = [
    {
        "path": "../examples/temporal_clue/temporal-clue.ipynb",
        "variables": {
            "TRAINING_STEPS": 1,
        },
    },
    {
        "path": "../examples/tic_tac_toe/tic-tac-toe.ipynb",
        "variables": {
            "TRAINING_STEPS": 1,
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
        "path": "../examples/prisoners-dilemma.ipynb",
        "variables": {
            "TRAINING_STEPS": 1,
            "PRISONERS_DILEMMA_ROUNDS": 10,
        },
    },
    {
        "path": "../examples/rock-paper-tool-use.ipynb",
        "variables": {"TRAINING_STEPS": 1},
    },
]


def make_patch_source() -> str:
    """
    This is a patch to the art.TrainableModel class to force _internal_config to None to avoid CUDA illegal memory access error.
    It also changes project name and model name logging these runs as test runs.
    """
    project = "Tester"
    return (
        "import datetime as _dt\n"
        "try:\n"
        "    import art as _art\n"
        "    import art.model as _art_model\n"
        "except Exception:\n"
        "    pass\n"
        "else:\n"
        "    _orig_tm_init = _art_model.TrainableModel.__init__\n"
        "    def _patched_tm_init(self, *args, **kwargs):\n"
        "        name = kwargs.get('name')\n"
        "        if name:\n"
        "            _suffix = _dt.datetime.utcnow().strftime('%Y%m%d-%H%M%SZ')\n"
        "            kwargs['name'] = f'{name}-{_suffix}'\n"
        f"        kwargs['project'] = {project!r}\n"
        "        result = _orig_tm_init(self, *args, **kwargs)\n"
        "        # Force _internal_config to None\n"
        "        self._internal_config = None\n"
        "        return result\n"
        "    def _patched_setattr(self, name, value):\n"
        "        if name == '_internal_config':\n"
        "            # Always set _internal_config to None\n"
        "            object.__setattr__(self, name, None)\n"
        "        else:\n"
        "            object.__setattr__(self, name, value)\n"
        "    _art_model.TrainableModel.__init__ = _patched_tm_init\n"
        "    _art_model.TrainableModel.__setattr__ = _patched_setattr\n"
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


def main() -> int:
    here = Path(__file__).parent.resolve()

    # Process notebook configurations
    processed_configs = []
    for config in NOTEBOOKS:
        if isinstance(config, str):
            # Handle legacy string format
            processed_config = {"path": config, "variables": {}}
        else:
            processed_config = config.copy()

        # Resolve path relative to this file
        p = (here / processed_config["path"]).resolve()
        if not p.exists():
            print(f"Warning: notebook not found: {p}")
        processed_config["path"] = str(p)
        processed_configs.append(processed_config)

    # Invoke pytest programmatically, injecting notebook params via a plugin
    # -s: do not capture stdout so notebook output is visible
    # --maxfail=1: stop on first failure to terminate the whole training run
    args = [
        "-s",
        "--maxfail=1",
        str(here / "integration.py"),
    ]
    result_code = pytest.main(args=args, plugins=[_NotebookPlugin(processed_configs)])
    return int(result_code)


if __name__ == "__main__":
    sys.exit(main())
