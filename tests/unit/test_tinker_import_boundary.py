"""Exercise the package initializer without importing optional training extras."""

import builtins
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class TinkerImportBoundaryTests(unittest.TestCase):
    def load_package(self):
        path = Path(__file__).parents[2] / "src/art/tinker/__init__.py"
        spec = spec_from_file_location("_tested_tinker", path)
        assert spec is not None and spec.loader is not None
        module = module_from_spec(spec)
        original_import = builtins.__import__

        def import_without_training(name, *args, **kwargs):
            level = args[3] if len(args) > 3 else kwargs.get("level", 0)
            if level or name.split(".")[0] not in sys.stdlib_module_names:
                raise AssertionError(f"Eager optional dependency: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_training):
            spec.loader.exec_module(module)
        return module

    def test_import_does_not_load_training_exports(self):
        package = self.load_package()
        self.assertEqual(
            package.__all__,
            ["TinkerBackend", "get_renderer_name", "OpenAICompatibleTinkerServer"],
        )
        self.assertFalse(set(package.__all__) & package.__dict__.keys())

    def test_exports_resolve_original_objects_once_on_demand(self):
        package = self.load_package()
        objects = {name: object() for name in package.__all__}
        package.import_module = Mock(return_value=SimpleNamespace(**objects))
        for name, target in zip(package.__all__, (".backend", ".renderers", ".server")):
            self.assertIs(getattr(package, name), objects[name])
            self.assertIs(getattr(package, name), objects[name])
            package.import_module.assert_called_once_with(target, "_tested_tinker")
            package.import_module.reset_mock()

    def test_unknown_attribute_does_not_load_dependencies(self):
        package = self.load_package()
        package.import_module = Mock()
        with self.assertRaises(AttributeError):
            getattr(package, "missing")
        package.import_module.assert_not_called()

    def test_requested_export_preserves_missing_dependency_error(self):
        package = self.load_package()
        error = ModuleNotFoundError("missing training dependency", name="mp_actors")
        package.import_module = Mock(side_effect=error)
        with self.assertRaises(ModuleNotFoundError) as raised:
            getattr(package, "TinkerBackend")
        self.assertIs(raised.exception, error)
        self.assertNotIn("TinkerBackend", package.__dict__)


if __name__ == "__main__":
    unittest.main()
