from __future__ import annotations

from typing import Any, Protocol

import torch

from art.megatron.context_parallel.layout_index import TokenLayoutIndex

from .contract import LinearRecurrentContract, PartitionKind
from .prefix_tree import RecurrentPackedExecutionSpec


class LinearRecurrentFamilyAdapter(Protocol):
    """Family-owned recurrent planning and device-materialization boundary.

    ``build_rank_plan`` returns an executable plan materialized on CPU;
    ``materialize_rank_plan`` moves that plan to the requested accelerator.
    """

    @property
    def family_key(self) -> str: ...

    @property
    def partition_kind(self) -> PartitionKind: ...

    @property
    def global_decision_type(self) -> type[Any]: ...

    @property
    def rank_plan_type(self) -> type[Any]: ...

    def validate_planning_inputs(
        self,
        contract: LinearRecurrentContract,
        planner_config: object | None,
    ) -> None: ...

    def build_global_decision(
        self,
        spec: RecurrentPackedExecutionSpec,
        *,
        contract: LinearRecurrentContract,
        token_layout_index: TokenLayoutIndex,
        cp_size: int,
        planner_config: object | None,
    ) -> object: ...

    def validate_global_decision(
        self,
        spec: RecurrentPackedExecutionSpec,
        decision: object,
        *,
        contract: LinearRecurrentContract,
        token_layout_index: TokenLayoutIndex,
        cp_size: int,
        planner_config: object | None,
    ) -> None: ...

    def build_rank_plan(
        self,
        spec: RecurrentPackedExecutionSpec,
        decision: object,
        *,
        contract: LinearRecurrentContract,
        cp_rank: int,
        planner_config: object | None,
    ) -> object: ...

    def validate_rank_plan(
        self,
        spec: RecurrentPackedExecutionSpec,
        plan: object,
        *,
        contract: LinearRecurrentContract,
        cp_rank: int,
        planner_config: object | None,
        device: torch.device | str,
    ) -> None: ...

    def materialize_rank_plan(
        self,
        plan: object,
        *,
        device: torch.device | str,
    ) -> object: ...

    def model_token_counts(
        self,
        decision: object,
        *,
        attention_token_counts: tuple[int, ...],
    ) -> tuple[int, ...]: ...
