from __future__ import annotations

import copy
import functools
import math
import os
from typing import Any, cast

from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.core.transformer.moe.router import TopKRouter
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch


class ExpertParallelLayout(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_experts: int = Field(gt=0)
    ep_size: int = Field(gt=0)
    physical_to_logical: tuple[int | None, ...]

    @model_validator(mode="after")
    def _validate_layout(self) -> ExpertParallelLayout:
        if len(self.physical_to_logical) % self.ep_size:
            raise ValueError("physical expert slots must divide evenly across EP ranks")
        logical = tuple(
            expert for expert in self.physical_to_logical if expert is not None
        )
        if logical != tuple(range(self.logical_experts)):
            raise ValueError(
                "physical expert slots must contain every logical expert once"
            )
        for ep_rank in range(self.ep_size):
            local = self.local_logical_experts(ep_rank)
            real_count = sum(expert is not None for expert in local)
            if any(expert is None for expert in local[:real_count]):
                raise ValueError("masked expert slots must be local-rank suffixes")
        return self

    @classmethod
    def build(
        cls,
        logical_experts: int,
        ep_size: int,
        *,
        slots_per_rank_multiple: int = 1,
    ) -> ExpertParallelLayout:
        if slots_per_rank_multiple <= 0:
            raise ValueError("slots_per_rank_multiple must be positive")
        logical_slots_per_rank = math.ceil(logical_experts / ep_size)
        slots_per_rank = (
            math.ceil(logical_slots_per_rank / slots_per_rank_multiple)
            * slots_per_rank_multiple
        )
        short_rank_count = logical_slots_per_rank * ep_size - logical_experts
        short_ranks = (
            {
                math.floor((index + 0.5) * ep_size / short_rank_count)
                for index in range(short_rank_count)
            }
            if short_rank_count
            else set()
        )
        next_expert = 0
        physical_to_logical: list[int | None] = []
        for ep_rank in range(ep_size):
            local_count = logical_slots_per_rank - (ep_rank in short_ranks)
            physical_to_logical.extend(range(next_expert, next_expert + local_count))
            physical_to_logical.extend([None] * (slots_per_rank - local_count))
            next_expert += local_count
        return cls(
            logical_experts=logical_experts,
            ep_size=ep_size,
            physical_to_logical=tuple(physical_to_logical),
        )

    @property
    def physical_experts(self) -> int:
        return len(self.physical_to_logical)

    @property
    def slots_per_rank(self) -> int:
        return self.physical_experts // self.ep_size

    @property
    def logical_to_physical(self) -> tuple[int, ...]:
        result = [0] * self.logical_experts
        for physical, logical in enumerate(self.physical_to_logical):
            if logical is not None:
                result[logical] = physical
        return tuple(result)

    def local_logical_experts(self, ep_rank: int) -> tuple[int | None, ...]:
        if not 0 <= ep_rank < self.ep_size:
            raise ValueError(f"invalid EP rank {ep_rank} for EP={self.ep_size}")
        start = ep_rank * self.slots_per_rank
        return self.physical_to_logical[start : start + self.slots_per_rank]

    def logical_expert(self, physical_expert: int) -> int | None:
        if not 0 <= physical_expert < self.physical_experts:
            raise ValueError(
                f"invalid physical expert {physical_expert}; "
                f"expected [0, {self.physical_experts})"
            )
        return self.physical_to_logical[physical_expert]


def configure_expert_parallel_layout(config: Any) -> ExpertParallelLayout | None:
    logical_experts = int(getattr(config, "num_moe_experts", 0) or 0)
    ep_size = int(getattr(config, "expert_model_parallel_size", 1) or 1)
    if logical_experts == 0:
        return None
    raw_ranks_per_domain = os.environ.get("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN")
    ranks_per_domain = int(raw_ranks_per_domain) if raw_ranks_per_domain else None
    if ranks_per_domain is not None and ranks_per_domain <= 0:
        raise ValueError("HybridEP ranks per NVLink domain must be positive")
    layout = ExpertParallelLayout.build(
        logical_experts,
        ep_size,
        slots_per_rank_multiple=(
            1 if ranks_per_domain is None else 4 // math.gcd(4, ranks_per_domain)
        ),
    )
    if layout.physical_experts == logical_experts:
        return None
    config.art_expert_parallel_layout = layout
    return layout


def activate_expert_parallel_layout(config: Any) -> ExpertParallelLayout | None:
    layout = get_expert_parallel_layout(config)
    if layout is not None:
        config.num_moe_experts = layout.physical_experts
    return layout


def get_expert_parallel_layout(config: Any) -> ExpertParallelLayout | None:
    layout = getattr(config, "art_expert_parallel_layout", None)
    if layout is None:
        return None
    if not isinstance(layout, ExpertParallelLayout):
        raise TypeError(f"invalid ART expert parallel layout: {type(layout).__name__}")
    return layout


class _LogicalRouterMixin:
    def __init__(
        self,
        config: Any,
        pg_collection: Any = None,
        is_mtp_layer: bool = False,
    ) -> None:
        layout = get_expert_parallel_layout(config)
        if layout is None:
            raise RuntimeError("logical router requires a non-uniform expert layout")
        logical_config = copy.copy(config)
        logical_config.num_moe_experts = layout.logical_experts
        parent = cast(Any, super())
        parent.__init__(
            config=logical_config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
        )
        physical_to_logical = [
            layout.logical_experts if expert is None else expert
            for expert in layout.physical_to_logical
        ]
        cast(Any, self).register_buffer(
            "_physical_to_logical",
            torch.tensor(physical_to_logical, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities, routing_map = cast(Any, super()).forward(input, padding_mask)
        physical_to_logical = cast(torch.Tensor, getattr(self, "_physical_to_logical"))
        return (
            _expand_logical_experts(probabilities, physical_to_logical),
            _expand_logical_experts(routing_map, physical_to_logical),
        )


@functools.cache
def logical_router_type(router_type: type) -> type:
    if issubclass(router_type, _LogicalRouterMixin):
        return router_type
    logical_type = type(
        f"ArtLogical{router_type.__name__}",
        (_LogicalRouterMixin, router_type),
        {"__module__": __name__},
    )
    AutoMapping.register_module_type(logical_type.__name__, "replicated")
    return logical_type


LogicalTopKRouter = logical_router_type(TopKRouter)


def _expand_logical_experts(
    tensor: torch.Tensor, physical_to_logical: torch.Tensor
) -> torch.Tensor:
    tensor = torch.cat(
        (tensor, tensor.new_zeros(*tensor.shape[:-1], 1)),
        dim=-1,
    )
    return tensor.index_select(-1, physical_to_logical)


def patch_moe_routers(block_spec: Any) -> int:
    patched = 0
    for layer_spec in getattr(block_spec, "layer_specs", ()) or ():
        layer_submodules = getattr(layer_spec, "submodules", None)
        mlp_spec = getattr(layer_submodules, "mlp", None)
        moe_submodules = getattr(mlp_spec, "submodules", None)
        if moe_submodules is not None and hasattr(moe_submodules, "router"):
            moe_submodules.router = logical_router_type(moe_submodules.router)
            patched += 1
    return patched
