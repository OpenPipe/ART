from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class PipelineState:
    """Shared state across pipeline stages."""

    # Policy versioning
    policy_version: int = 0
    next_training_step: int = 0

    # Scenario tracking
    scenario_offset: int = 0
    total_scenarios_consumed: int = 0
    last_eval_step: int = 0

    # Metrics
    discarded_stale_samples: int = 0

    # Synchronization
    policy_updated: asyncio.Condition = field(default_factory=asyncio.Condition)
    done: bool = False
