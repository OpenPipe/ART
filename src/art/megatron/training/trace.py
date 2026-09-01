from __future__ import annotations

from typing import Any

import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.preprocessing.pack import PackedTensors


def context_parallel_trace_token_uids_enabled(
    topology: ParallelTopology,
    moe_routing_replay_controller: Any | None,
) -> bool:
    return int(topology.cp) > 1 and moe_routing_replay_controller is not None


def packed_sequence_token_uids(
    micro: PackedTensors,
    *,
    device: torch.device,
) -> torch.Tensor:
    del device
    return torch.arange(
        int(micro["tokens"].shape[1]),
        dtype=torch.int64,
    ).unsqueeze(0)


def sft_sequence_token_uids(
    inputs: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    del device
    attention_mask = inputs["attention_mask"].reshape(-1)
    actual_len = max(int(attention_mask.sum().item()), 1)
    total_tokens = int(inputs["input_ids"].numel())
    token_uids = torch.full(
        (1, total_tokens),
        -1,
        dtype=torch.int64,
    )
    token_uids[:, :actual_len] = torch.arange(
        actual_len,
        dtype=torch.int64,
    ).unsqueeze(0)
    return token_uids


def flatten_local_token_uids(
    token_uids: torch.Tensor | None,
) -> torch.Tensor | None:
    if token_uids is None:
        return None
    return (
        token_uids.transpose(0, 1)
        .contiguous()
        .reshape(-1)
        .to(dtype=torch.int64)
        .contiguous()
    )


def prepare_replay_local_input_token_uids(
    moe_routing_replay_controller: Any | None,
    token_uids: torch.Tensor | None,
    attention_state: Any | None = None,
) -> None:
    if moe_routing_replay_controller is None or not hasattr(
        moe_routing_replay_controller,
        "prepare_micro_targets",
    ):
        return
    token_uid_sets = _routing_replay_token_uid_sets(
        token_uids,
        attention_state=attention_state,
    )
    moe_routing_replay_controller.prepare_micro_targets(token_uid_sets)


def _routing_replay_token_uid_sets(
    token_uids: torch.Tensor | None,
    *,
    attention_state: Any | None,
) -> dict[str, torch.Tensor | None]:
    attention_token_uids = flatten_local_token_uids(token_uids)
    plan = getattr(attention_state, "gdn_execution_plan", None)
    if plan is not None:
        return {
            "attention": attention_token_uids,
            "gdn": torch.tensor(
                tuple(getattr(plan, "gdn_token_indices")),
                dtype=torch.int64,
            ),
        }
    return {"attention": attention_token_uids}
