"""Use installed MCore/TE signatures and constructors, never stub linears."""

import ast
import importlib
import inspect
from pathlib import Path

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
import pytest

from tests.integration.megatron.glm5next_constructors import check_native_construction


def test_direct_mcore_constructor_call_sites_match_installed_classes() -> None:
    """Also bind spec/factory call sites, including the reference-spec branch."""
    root = Path(__file__).parents[2] / "src/art/megatron/glm5next"
    checked = set()
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "megatron.core"
            ):
                assert node.module is not None
                module = importlib.import_module(node.module)
                for alias in node.names:
                    cls = getattr(module, alias.name)
                    if inspect.isclass(cls):
                        classes[alias.asname or alias.name] = cls
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in classes
            ):
                # Dynamic build_module/super calls are bound by the native audit.
                assert not any(isinstance(arg, ast.Starred) for arg in node.args)
                assert all(keyword.arg is not None for keyword in node.keywords)
                signature = inspect.signature(classes[node.func.id])
                signature.bind(
                    *[object() for _ in node.args],
                    **{
                        keyword.arg: object()
                        for keyword in node.keywords
                        if keyword.arg is not None
                    },
                )
                checked.add(node.func.id)
    assert {"TESpecProvider", "ModuleSpec", "MLASelfAttentionSubmodules"} <= checked


@pytest.mark.parametrize("name", ["F4", "F8", "full"])
def test_native_model_and_lora_constructors_on_meta(name: str) -> None:
    path = Path(__file__).parent / "fixtures/glm5next_configs" / name / "config.json"
    result = check_native_construction(path)
    assert result["status"] == "pass" and not result["cuda_initialized"]
    audit = result["constructors"]
    used = {row["constructor"] for row in audit["calls"]}
    assert any("TEColumnParallelLinear.__init__" in key for key in used)
    assert any("TERowParallelLinear.__init__" in key for key in used)
    assert any("TEGroupedMLP.__init__" in key for key in used)
    assert any("TENorm.__new__" in key for key in used)
    assert any("GPTModel.__init__" in key for key in used)
    for row in audit["calls"]:
        if row["constructor"].endswith(
            ("TEColumnParallelLinear.__init__", "TERowParallelLinear.__init__")
        ):
            assert "is_expert" in row["supplied"]
    if name == "full":
        assert result["layers"] == 45
        assert result["trainable_elements"] == 77_923_584 * 32


@pytest.mark.parametrize("cls", [TEColumnParallelLinear, TERowParallelLinear])
def test_real_signature_rejects_original_missing_expert_argument(cls: type) -> None:
    signature = inspect.signature(cls)
    required = {
        key: object()
        for key, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind == inspect.Parameter.KEYWORD_ONLY
    }
    assert "is_expert" in required
    signature.bind(128, 128, **required)
    del required["is_expert"]
    with pytest.raises(TypeError, match="is_expert"):
        signature.bind(128, 128, **required)
