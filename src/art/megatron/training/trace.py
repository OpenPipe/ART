from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import os
from typing import Any

import torch

from art.megatron.context_parallel.types import ParallelTopology
from art.preprocessing.pack import PackedTensors

ROOT_OUTPUT_TOKEN_UIDS_ATTR = "_art_root_output_token_uids"
TRACE_ROW_TOKEN_UIDS_ATTR = "_art_trace_row_token_uids"
TRACE_UID_SPAN_ATTR = "_art_trace_uid_span"


def trace_token_uids_enabled() -> bool:
    raw = os.environ.get("ART_MEGATRON_ATTACH_TOKEN_UIDS", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def context_parallel_trace_token_uids_enabled(
    topology: ParallelTopology,
    moe_routing_replay_controller: Any | None,
) -> bool:
    return int(topology.cp) > 1 and (
        moe_routing_replay_controller is not None or trace_token_uids_enabled()
    )


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


def _set_root_output_trace_token_uids(
    root_module: torch.nn.Module,
    token_uids: torch.Tensor | None,
) -> None:
    if token_uids is None:
        if hasattr(root_module, ROOT_OUTPUT_TOKEN_UIDS_ATTR):
            delattr(root_module, ROOT_OUTPUT_TOKEN_UIDS_ATTR)
        return
    setattr(
        root_module,
        ROOT_OUTPUT_TOKEN_UIDS_ATTR,
        token_uids.detach().to(device="cpu", dtype=torch.int64).contiguous(),
    )


def _set_module_trace_token_uids(
    model_chunks: Sequence[torch.nn.Module],
    token_uids: torch.Tensor | None,
) -> None:
    row_token_uids = flatten_local_token_uids(token_uids)
    for chunk in model_chunks:
        for module in chunk.modules():
            if row_token_uids is None:
                if hasattr(module, TRACE_ROW_TOKEN_UIDS_ATTR):
                    delattr(module, TRACE_ROW_TOKEN_UIDS_ATTR)
                if hasattr(module, TRACE_UID_SPAN_ATTR):
                    delattr(module, TRACE_UID_SPAN_ATTR)
                continue
            setattr(
                module,
                TRACE_ROW_TOKEN_UIDS_ATTR,
                row_token_uids.detach()
                .to(device="cpu", dtype=torch.int64)
                .contiguous(),
            )
            if hasattr(module, TRACE_UID_SPAN_ATTR):
                delattr(module, TRACE_UID_SPAN_ATTR)


@contextmanager
def attach_trace_token_uids(
    model_chunks: Sequence[torch.nn.Module],
    token_uids: torch.Tensor | None,
) -> Iterator[None]:
    attach_module_token_uids = trace_token_uids_enabled()
    _set_root_output_trace_token_uids(model_chunks[0], token_uids)
    if attach_module_token_uids:
        _set_module_trace_token_uids(model_chunks, token_uids)
    try:
        yield
    finally:
        _set_root_output_trace_token_uids(model_chunks[0], None)
        if attach_module_token_uids:
            _set_module_trace_token_uids(model_chunks, None)
