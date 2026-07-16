"""Deprecated compatibility names for automatic trajectory capture."""

from .trajectories import (
    auto_trajectory,  # ty: ignore[deprecated]
    capture_auto_trajectory,  # ty: ignore[deprecated]
)

__all__ = ["auto_trajectory", "capture_auto_trajectory"]
