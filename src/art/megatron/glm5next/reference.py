from __future__ import annotations

import inspect
from types import FunctionType, MethodType
from typing import Any

import torch
from torch import Tensor, nn


class HfReferenceAttention(nn.Module):
    """Temporary CP1/TP1 HF-equivalent attention for early model integration.

    These modules require the oracle's newer Transformers environment, rather
    than changing ART's shared runtime pins. M2/M3 replace them through the layer
    spec. Packed/tree inputs expand independent causal paths for correctness;
    these deliberately slow fallbacks do not claim native kernel throughput.
    """

    def __init__(
        self,
        config: Any,
        layer_number: int,
        submodules: Any = None,
        pg_collection: Any = None,
        **_: Any,
    ) -> None:
        super().__init__()
        del submodules, pg_collection
        if config.tensor_model_parallel_size != 1 or config.context_parallel_size != 1:
            raise ValueError("GLM HF reference attention requires TP1/CP1.")
        try:
            from transformers.models.glm5_next.modeling_glm5_next import (
                Glm5NextTextAttention,
                Glm5NextTextLinearAttention,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GLM reference attention needs Transformers with glm5_next support; "
                "use the oracle environment until M2/M3 native attention lands."
            ) from exc
        self.layer_number = layer_number
        self.config = config
        self.is_linear = bool(config.linear_attention_freq[layer_number - 1])
        attention = (
            Glm5NextTextLinearAttention if self.is_linear else Glm5NextTextAttention
        )
        self.hf_reference = attention(config.glm5next_hf_config, layer_number - 1)
        if self.is_linear:
            # HF auto-selects installed FLA/causal-conv CUDA functions even for
            # CPU tensors. Bind an instance-local copy of its forward globals
            # to the undecorated PyTorch references, without changing HF globally.
            forward = inspect.unwrap(self.hf_reference.forward)
            namespace = dict(forward.__globals__)
            for name in (
                "causal_conv1d_fn",
                "chunk_kimi_delta_attention",
                "recurrent_kimi_delta_attention",
            ):
                namespace[name] = inspect.unwrap(namespace[name])
            reference_forward = FunctionType(
                forward.__code__,
                namespace,
                forward.__name__,
                forward.__defaults__,
                forward.__closure__,
            )
            setattr(
                self.hf_reference,
                "forward",
                MethodType(reference_forward, self.hf_reference),
            )
        if not self.is_linear:
            indexer = self.hf_reference.indexer
            if indexer is not None:
                indexer.requires_grad_(False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        packed_seq_params: Any = None,
        inference_context: Any = None,
        attention_bias: Any = None,
        **_: Any,
    ) -> tuple[Tensor, None]:
        if inference_context is not None:
            raise NotImplementedError("GLM reference attention supports training only.")
        if attention_mask is not None and attention_bias is None:
            raise NotImplementedError(
                "GLM reference attention requires ART tree metadata for masked batches."
            )
        value = hidden_states.transpose(0, 1).contiguous()
        if attention_bias is not None:
            result = self._prefix_tree(value, attention_bias)
        elif packed_seq_params is not None:
            bounds = packed_seq_params.cu_seqlens_q.detach().cpu().tolist()
            if value.shape[0] != 1 or bounds[0] != 0 or bounds[-1] != value.shape[1]:
                raise ValueError(
                    "GLM reference THD boundaries must cover one compact row."
                )
            if any(end <= start for start, end in zip(bounds, bounds[1:])):
                raise ValueError("GLM reference THD sequences must be nonempty.")
            result = torch.cat(
                [
                    self._sequence(value[:, start:end])
                    for start, end in zip(bounds, bounds[1:])
                ],
                dim=1,
            )
        else:
            result = self._sequence(value)
        return result.transpose(0, 1).contiguous(), None

    def _sequence(self, value: Tensor) -> Tensor:
        if self.is_linear:
            return self.hf_reference(value)
        # Native HF's k-pool indexer consumes a 2D validity mask and builds
        # its own causal sparse mask from the selected token indices.
        mask = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
        return self.hf_reference(value, attention_mask=mask)[0]

    def _prefix_tree(self, value: Tensor, state: Any) -> Tensor:
        from art.megatron.prefix_tree import parse_prefix_tree

        groups, parents = state.group_ids, state.parent_ids
        if tuple(groups.shape) != tuple(value.shape[:2]):
            raise ValueError(
                "GLM reference tree metadata must match attention token layout."
            )
        rows = parse_prefix_tree(group_ids=groups.cpu(), parent_ids=parents.cpu())
        result = torch.zeros_like(value)
        for row in rows:
            by_id = {segment.group_id: segment for segment in row.segments}
            for segment in row.segments:
                path = [by_id[group] for group in segment.ancestors] + [segment]
                inputs = torch.cat(
                    [
                        value[row.row_index : row.row_index + 1, node.start : node.end]
                        for node in path
                    ],
                    dim=1,
                )
                output = self._sequence(inputs)
                result[row.row_index, segment.start : segment.end] = output[
                    0, -(segment.end - segment.start) :
                ]
        return result
