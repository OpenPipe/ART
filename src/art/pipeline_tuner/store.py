from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import PipelineAutotunerProfile

if TYPE_CHECKING:
    from art.model import TrainableModel


class PipelineTunerProfileStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def for_model(cls, model: TrainableModel) -> "PipelineTunerProfileStore":
        return cls(Path(model._get_output_dir()) / "pipeline_tuner")

    def resolve(self, profile: str | None) -> Path:
        name = profile or "latest"
        path = Path(name)
        if path.is_absolute() or path.suffix == ".json":
            return path
        return self.root / f"{name}.json"

    def load(self, profile: str | None) -> PipelineAutotunerProfile:
        return PipelineAutotunerProfile.model_validate_json(
            self.resolve(profile).read_text(encoding="utf-8")
        )

    def save(self, profile: str, data: PipelineAutotunerProfile) -> Path:
        path = self.resolve(profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        return path
