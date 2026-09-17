"""Bundle ART's shared sources as real files, including in standalone sdists."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        for source in (root / "src/art_vllm_runtime/_shared").glob("*.py"):
            if source.is_symlink():
                build_data["force_include"][str(source.resolve())] = str(
                    source.relative_to(root)
                )
