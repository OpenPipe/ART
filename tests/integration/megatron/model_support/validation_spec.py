from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MinimalLayerCoverageReport(BaseModel):
    base_model: str
    model_key: str
    requested_num_layers: int
    recommended_min_layers: int
    covered: bool
    missing_layer_families: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)


class ValidationStageResult(BaseModel):
    name: str
    passed: bool = False
    skipped: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifact_dir: str | None = None


class ValidationReport(BaseModel):
    git: dict[str, Any]
    base_model: str
    model_key: str
    passed: bool = False
    complete: bool = False
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    stages: list[ValidationStageResult] = Field(default_factory=list)
