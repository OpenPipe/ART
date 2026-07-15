from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor

from .trace_uids import TRACE_ROW_TOKEN_UIDS_ATTR, TRACE_UID_SPAN_ATTR


class GdnTraceTokenUidHooks:
    def attach_token_uids(self, tensor: Tensor, token_uids: Tensor) -> Tensor:
        setattr(
            tensor,
            TRACE_ROW_TOKEN_UIDS_ATTR,
            token_uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1),
        )
        setattr(tensor, TRACE_UID_SPAN_ATTR, None)
        return tensor

    def token_uids_from_tensor(self, tensor: Tensor) -> Tensor | None:
        token_uids = getattr(tensor, TRACE_ROW_TOKEN_UIDS_ATTR, None)
        if not isinstance(token_uids, Tensor):
            return None
        return token_uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)

    def set_module_token_uids(self, module: Any, token_uids: Tensor | None) -> None:
        if module is None or token_uids is None:
            return
        setattr(
            module,
            TRACE_ROW_TOKEN_UIDS_ATTR,
            token_uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1),
        )
        if hasattr(module, TRACE_UID_SPAN_ATTR):
            delattr(module, TRACE_UID_SPAN_ATTR)

    @torch.compiler.disable
    def prepare_in_proj_token_uids(self, gdn: Any, hidden_states: Tensor) -> None:
        from art.megatron.gdn import operator as gdn_operator

        token_uids = self.token_uids_from_tensor(hidden_states)
        if token_uids is None:
            return
        projection = gdn_operator._gdn_input_projection(gdn)
        if projection is None:
            return
        output_uids = self.column_parallel_input_token_uids(
            token_uids,
            hidden_states,
            projection,
        )
        in_proj = getattr(gdn, "in_proj", None)
        self.set_module_token_uids(in_proj, output_uids)
        self.set_module_token_uids(projection, output_uids)
        self.set_module_token_uids(getattr(in_proj, "qkv_lora", None), output_uids)
        self.set_module_token_uids(getattr(in_proj, "z_lora", None), output_uids)

    @torch.compiler.disable
    def column_parallel_input_token_uids(
        self, token_uids: Tensor, hidden_states: Tensor, projection: Any
    ) -> Tensor:
        from art.megatron.gdn import operator as gdn_operator

        if not gdn_operator._uses_sequence_parallel(projection):
            return token_uids.to(dtype=torch.int64).reshape(-1)
        seq_len, batch_size, _hidden_size = hidden_states.shape
        expected = int(seq_len) * int(batch_size)
        if int(token_uids.numel()) != expected:
            return token_uids.to(dtype=torch.int64).reshape(-1)
        uid_tensor = (
            token_uids.to(device=hidden_states.device, dtype=torch.int64)
            .reshape(batch_size, seq_len)
            .transpose(0, 1)
            .contiguous()
            .unsqueeze(-1)
        )
        gathered = gdn_operator._column_parallel_input(uid_tensor, projection)
        return (
            gathered.squeeze(-1)
            .transpose(0, 1)
            .contiguous()
            .reshape(-1)
            .detach()
            .to(device="cpu", dtype=torch.int64)
        )

    def set_out_proj_lora_token_uids(self, gdn: Any, hidden_states: Tensor) -> None:
        token_uids = self.token_uids_from_tensor(hidden_states)
        if token_uids is None:
            return
        self.set_module_token_uids(
            getattr(getattr(gdn, "out_proj", None), "lora", None),
            token_uids,
        )

    def set_out_norm_token_uids(self, gdn: Any, token_uids: Tensor) -> None:
        from art.megatron.gdn import operator as gdn_operator

        self.set_module_token_uids(
            getattr(gdn, "out_norm", None),
            token_uids.repeat_interleave(gdn_operator._local_value_heads(gdn)),
        )

    def set_out_proj_token_uids(
        self,
        gdn: Any,
        hidden_states: Tensor,
        *,
        sequence_parallel_output: bool,
    ) -> None:
        from art.megatron.gdn import operator as gdn_operator

        token_uids = self.token_uids_from_tensor(hidden_states)
        if token_uids is None:
            return
        projection = gdn_operator._gdn_output_projection(gdn)
        output_uids = self.row_parallel_output_token_uids(
            token_uids,
            hidden_states,
            projection,
            sequence_parallel_output=sequence_parallel_output,
        )
        self.set_module_token_uids(gdn, output_uids)
        self.set_module_token_uids(getattr(gdn, "out_proj", None), output_uids)
        self.set_module_token_uids(projection, output_uids)

    def row_parallel_output_token_uids(
        self,
        token_uids: Tensor,
        hidden_states: Tensor,
        projection: Any | None,
        *,
        sequence_parallel_output: bool,
    ) -> Tensor:
        from art.megatron.gdn import operator as gdn_operator

        token_uids = token_uids.to(dtype=torch.int64).reshape(-1)
        if (
            projection is None
            or not gdn_operator._uses_sequence_parallel(projection)
            or not sequence_parallel_output
        ):
            return token_uids
        token_count = gdn_operator._hidden_token_count(hidden_states)
        if token_count <= 0:
            return token_uids.new_empty((0,))
        if int(token_uids.numel()) != token_count:
            return token_uids
        tp_size = gdn_operator._tp_world_size(projection)
        tp_rank = gdn_operator._tp_rank(projection)
        rows_per_rank, remainder = divmod(token_count, tp_size)
        if remainder != 0:
            return token_uids
        start = tp_rank * rows_per_rank
        return token_uids[start : start + rows_per_rank].contiguous()

    def pad_token_uids_for_stream(self, token_uids: Tensor, stream: Tensor) -> Tensor:
        from art.megatron.gdn import operator as gdn_operator

        token_count = gdn_operator._hidden_token_count(stream)
        if token_count == int(token_uids.numel()):
            return token_uids
        padded = token_uids.new_full((token_count,), -1)
        copied = min(token_count, int(token_uids.numel()))
        if copied:
            padded[:copied] = token_uids[:copied]
        return padded


@contextmanager
def install_gdn_trace_token_uid_hooks() -> Iterator[None]:
    from art.megatron.gdn import operator as gdn_operator

    previous = gdn_operator.set_gdn_trace_token_uid_hooks(GdnTraceTokenUidHooks())
    try:
        yield
    finally:
        gdn_operator.set_gdn_trace_token_uid_hooks(previous)
