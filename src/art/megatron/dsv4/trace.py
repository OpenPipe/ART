from typing import Any

import torch
from torch import Tensor

TRACE_ROW_TOKEN_UIDS_ATTR = "_art_trace_row_token_uids"
TRACE_UID_SPAN_ATTR = "_art_trace_uid_span"


def attach_trace_row_token_uids(
    module: Any,
    output: Tensor,
    *,
    tp_group: Any | None = None,
) -> Tensor:
    row_uids = getattr(module, TRACE_ROW_TOKEN_UIDS_ATTR, None)
    if not isinstance(row_uids, torch.Tensor) or output.ndim == 0:
        return output
    row_uids = row_uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    row_count = int(output.shape[0])
    if int(row_uids.numel()) == row_count:
        selected = row_uids
    else:
        tp_size = int(tp_group.size()) if tp_group is not None else _tp_world_size()
        if tp_size <= 1 or int(row_uids.numel()) != row_count * tp_size:
            return output
        tp_rank = int(tp_group.rank()) if tp_group is not None else _tp_rank()
        selected = row_uids.narrow(0, tp_rank * row_count, row_count)
    setattr(output, TRACE_ROW_TOKEN_UIDS_ATTR, selected.contiguous())
    uid_span = getattr(module, TRACE_UID_SPAN_ATTR, None)
    if isinstance(uid_span, int) and uid_span > 0:
        setattr(output, TRACE_UID_SPAN_ATTR, int(uid_span))
    return output


def _tp_world_size() -> int:
    try:
        from megatron.core import parallel_state as ps

        return int(ps.get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def _tp_rank() -> int:
    try:
        from megatron.core import parallel_state as ps

        return int(ps.get_tensor_model_parallel_rank())
    except Exception:
        return 0
