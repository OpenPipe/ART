"""TrainerRank memory estimation, profiling and admission accounting.

These are ``TrainerRank`` method bodies moved out of ``_impl`` verbatim: each
function takes the owning rank as ``self`` and ``TrainerRank`` binds them as
methods, so ``self._x(...)`` dispatch and per-instance overrides keep working.

Module globals the bodies used to read from ``_impl`` (``torch``, ``dist``,
``os``, sibling helpers) are still resolved through ``_impl`` at call time, so
tests that patch ``_impl.torch`` and friends keep intercepting them; only pure
stdlib helpers are imported here directly. Referencing ``_impl`` as a module
also lets the circular import resolve lazily.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import nullcontext
import gc
import hashlib
import math
from types import MethodType
from typing import TYPE_CHECKING, Any

from art.trainer_rank import _impl

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from art.megatron.lora import LoRASlotRef
    from art.trainer_rank._impl import (
        AdapterSelection,
        AnyForwardInput,
        TrainerRank,
        _GroupLayout,
    )


def _split_required_memory(costs: Sequence[_impl._SubforwardCost]) -> int:
    # A grown HybridEP buffer persists into later children: charge the
    # largest growth once, beside whichever child peaks highest.
    growth = max(cost.hybridep_growth for cost in costs)
    required = (
        sum(cost.retained for cost in costs)
        + max(
            cost.ephemeral - int(cost.hybridep_growth * _impl._MEMORY_SAFETY_FACTOR)
            for cost in costs
        )
        + int(growth * _impl._MEMORY_SAFETY_FACTOR)
    )
    if any(cost.checkpoint_input_gradient for cost in costs):
        # The caller owns all returned graphs. A calibrated forward-retained
        # discount cannot replace the sum of their input-gradient extents.
        # Children training the same slots share their adapter gradients,
        # allocated once by whichever child's backward reaches a layer
        # first; the largest child's extra covers any order. Different
        # slots have disjoint gradients, charged per child.
        adapter = [
            cost.checkpoint_adapter_gradient
            for cost in costs
            if cost.checkpoint_adapter_gradient
        ]
        shared = (
            len(
                {
                    cost.checkpoint_adapter_gradient_slots
                    for cost in costs
                    if cost.checkpoint_adapter_gradient
                }
            )
            <= 1
        )
        checkpoint = (
            sum(
                cost.checkpoint_retained + cost.checkpoint_input_gradient
                for cost in costs
            )
            + max(cost.checkpoint_workspace for cost in costs)
            + ((max(adapter) if shared else sum(adapter)) if adapter else 0)
            + growth
        )
        required = max(required, int(checkpoint * _impl._MEMORY_SAFETY_FACTOR))
    return required


def _split_memory_key(plan: _impl._SplitForwardPlan) -> bytes | None:
    # Bound traversal and hashing; retain only a digest, never tensor values,
    # token tables or graphs. Oversized structural identities stay unlearned.
    if len(plan.subforwards) > 1024:
        return None
    digest, remaining, nodes = (
        hashlib.sha256(),
        262144 - 32 * len(plan.subforwards),
        65536,
    )

    def feed(value: Any) -> None:
        nonlocal remaining, nodes
        nodes -= 1
        if nodes < 0:
            raise ValueError("split key node cap")
        if isinstance(value, tuple):
            remaining -= 2
            if remaining < 0:
                raise ValueError("split key byte cap")
            feed(len(value))
            digest.update(b"(")
            for child in value:
                feed(child)
            digest.update(b")")
            return
        if value is not None and type(value) not in (str, int, bool):
            raise ValueError("unsupported split key field")
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError("split key string cap")
        if isinstance(value, int) and value.bit_length() > 64:
            raise ValueError("split key integer cap")
        encoded = repr(value).encode()
        remaining -= len(encoded) + 1
        if remaining < 0:
            raise ValueError("split key byte cap")
        digest.update(encoded + b";")

    try:
        feed((plan.request_count, len(plan.subforwards)))
        outer, children = digest, []
        for p, indices in zip(plan.subforwards, plan.request_indices, strict=True):
            digest = hashlib.sha256()
            signature = p.signature
            feed(
                (
                    indices,
                    signature.topology,
                    signature.planner_coefficients,
                    signature.slot_group_count,
                    signature.request_mix,
                    signature.grad_enabled,
                    signature.grad_modes,
                    signature.memory_placement,
                    signature.slot_shapes,
                    signature.short_requests,
                    p.packed_tokens,
                    p.logical_tokens,
                    p.inactive_logical_tokens,
                    p.output_bytes,
                    p.output_metadata,
                    p.selected_max_depth,
                    len(p.groups),
                )
            )
            for g in p.groups:
                slot = (
                    None
                    if g.slot_ref is None
                    else ("checkpoint", g.slot_ref.name)
                    if isinstance(g.slot_ref, _impl._LocalLoRASlotRef)
                    else (g.slot_ref.kind, g.slot_ref.name)
                )
                feed(
                    (
                        slot,
                        g.grad_enabled,
                        g.request_indices,
                        len(g.packed.segments),
                    )
                )
                for segment in g.packed.segments:
                    feed(
                        (
                            segment.sequence_indices,
                            segment.start,
                            segment.end,
                            segment.packed_start,
                            segment.group_id,
                            segment.parent_id,
                        )
                    )
                feed(len(g.items))
                for item in g.items:
                    feed(
                        (
                            tuple(item.input_ids.shape),
                            str(item.input_ids.dtype),
                            None
                            if item.labels is None
                            else (tuple(item.labels.shape), str(item.labels.dtype)),
                            item.request.top_k,
                            item.request.logits,
                            item.request.hidden_states,
                        )
                    )
            children.append(digest.digest())
    except ValueError:
        return None
    # Cost/profile updates may reorder the same children. Each digest still
    # binds its original request mapping/partition; retain the observed max
    # across execution orders, not an unmeasured bound for every order.
    outer.update(b"".join(sorted(children)))
    return outer.digest()


def _record_split_memory_floor(
    self: TrainerRank, plan: _impl._SplitForwardPlan, baseline: int, forward_peak: int
) -> None:
    # Only normal caller completion learns a floor; child retained profiles
    # stay forward-only. Keep earlier child peaks despite their later resets.
    key = self._split_memory_key(plan)
    if key is None:
        self._split_memory_floor_status = "unsupported_key"
        return
    if key not in self._split_memory_floors and len(self._split_memory_floors) >= 1024:
        self._split_memory_floor_status = "cache_full_not_learned"
        return
    observed = max(
        0,
        max(forward_peak, int(_impl.torch.cuda.max_memory_allocated(self.device)))
        - baseline,
    )
    self._split_memory_floors[key] = max(
        self._split_memory_floors.get(key, 0), observed
    )
    self._split_memory_floor_status = "recorded"


def _split_plan_memory_check(
    self: TrainerRank,
    plan: _impl._SplitForwardPlan,
    costs: Sequence[_impl._SubforwardCost],
) -> _impl._MemoryCheck:
    required = self._split_required_memory(costs)
    key = self._split_memory_key(plan)
    empirical = (
        0
        if key is None
        else int(self._split_memory_floors.get(key, 0) * _impl._MEMORY_SAFETY_FACTOR)
    )
    # Same existing reduction order and count; never reduce while returning
    # from caller work, where a failed/empty peer may not participate.
    return self._memory_check_required(max(required, empirical))


def _head_vocabulary(self: TrainerRank) -> int:
    """Runtime eligibility and this rank's vocabulary, without reading values.

    TP > 1 runs the same 512-row chunked head over each rank's vocabulary
    shard; it is priced only where the explicit TP x SP checkpoint floor
    applies, whose shallow shapes it can dominate.
    """
    _, tp, _, pp = self._topology_key()
    if (
        self._padded_vocab_size is None
        or len(self.runtime.model) != 1
        or pp != 1
        or self._padded_vocab_size % tp
        or (tp > 1 and not _checkpoint_layers(self, ((1, True),)))
    ):
        return 0
    vocabulary = self._padded_vocab_size // tp
    try:
        model = _impl._language_model(self.runtime.model[0])
    except (AttributeError, RuntimeError):
        return 0
    try:
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        return 0

    head = getattr(model, "output_layer", None)
    if head is None or type(head) is not ColumnParallelLinear:
        return 0
    weight = head.weight
    if (
        weight is None
        and getattr(model, "share_embeddings_and_output_weights", False) is True
    ):
        weight = getattr(
            getattr(getattr(model, "embedding", None), "word_embeddings", None),
            "weight",
            None,
        )
    if weight is None:
        return 0
    config = getattr(model, "config", None)
    if (
        type(weight) not in (_impl.torch.Tensor, _impl.torch.nn.Parameter)
        or weight.dtype is not _impl.torch.bfloat16
        or tuple(weight.shape) != (vocabulary, self._hidden_size)
        or head.output_size_per_partition != vocabulary
        or head.output_size != self._padded_vocab_size
        or head.input_size != self._hidden_size
        or getattr(config, "params_dtype", None) is not _impl.torch.bfloat16
        or getattr(config, "fp32_residual_connection", None) is not False
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or "forward" in vars(head)
        or "_forward_impl" in vars(head)
        or getattr(head, "_forward_hooks", None)
        or getattr(head, "_forward_pre_hooks", None)
        or any(
            name in vars(self)
            for name in (
                "_project_head",
                "_project_vocab_parallel",
                "_checkpointed_head_stats",
                "_local_head_stats",
                "_local_logits_from_hidden_rows",
            )
        )
    ):
        return 0
    return int(vocabulary)


def _dense_head_bytes(vocabulary: int, rows: int) -> int:
    return min(rows, _impl._HEAD_CHUNK_TOKENS) * vocabulary * 2


def _head_workspace_bytes(self: TrainerRank, rows: int) -> int:
    """One dense BF16 head tensor, not complete statistics/backward memory."""
    return _dense_head_bytes(_head_vocabulary(self), rows) if rows > 0 else 0


def _head_target_backward(self: TrainerRank) -> bool:
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    model = _impl._language_model(self.runtime.model[0])
    scale = getattr(model, "_scale_logits", None)
    return (
        type(scale) is MethodType
        and scale.__self__ is model
        and scale.__func__ is LanguageModule._scale_logits
        and getattr(model.config, "use_mup", None) is False
    )


def _group_head_workspace_bytes(
    self: TrainerRank,
    rows: int,
    requests: Sequence[AnyForwardInput],
    *,
    grad_enabled: bool,
    positions: Sequence[torch.Tensor] | None = None,
    lower_bound: bool = False,
) -> int:
    """Partial dense head component: eager statistics or logits copies.

    Capacity reserves the eager path even when optional Triton may succeed.
    Its BF16 logits, FP32 conversion, subtraction and exp overlap. This is
    not a bound for row vectors, inter-chunk liveness or library workspaces.
    Rejection lower bounds retain only the unconditional dense components.
    TP > 1 heads (priced with the explicit TP x SP floor) instead follow each
    chunk's statistics path: ``_tp_head_workspace_bytes``. Statistics capacity
    adds their index vectors (``_head_target_bytes``) on both.
    """
    dense = self._head_workspace_bytes(rows)
    if dense and self._topology_key()[1] > 1:
        return _tp_head_workspace_bytes(
            self,
            rows,
            requests,
            grad_enabled=grad_enabled,
            positions=positions,
            lower_bound=lower_bound,
        )
    needs_statistics = any(
        request.target_tokens is not None or request.top_k is not None
        for request in requests
    )
    if not dense or (
        not needs_statistics and (lower_bound or not any(r.logits for r in requests))
    ):
        return dense
    if _head_target_backward(self):
        if not lower_bound:
            # need_log_z is group-wide, including logits-only chunks and
            # chunks overlapping ignored labels. A short final chunk can
            # also take the eager path; optional success is not guaranteed.
            # Without statistics, local logits and both indexed copies
            # overlap. Requested output storage is charged separately.
            if not needs_statistics:
                return 3 * dense
            return 7 * dense + _head_target_bytes(
                rows, requests, grad_enabled=grad_enabled
            )
        if not grad_enabled or not any(
            request.target_tokens is not None for request in requests
        ):
            return dense
        # Every statistics path's backward holds the labelled chunk's logits
        # and their gradient.
        target_dense = (
            self._head_workspace_bytes(
                self._head_target_chunk_rows(
                    requests, positions=positions, lower_bound=lower_bound
                )
            )
            if any(request.logits or request.top_k is not None for request in requests)
            else dense
        )
        return max(dense, 2 * target_dense)
    return dense


def _tp_head_workspace_bytes(
    self: TrainerRank,
    rows: int,
    requests: Sequence[AnyForwardInput],
    *,
    grad_enabled: bool,
    positions: Sequence[torch.Tensor] | None = None,
    lower_bound: bool = False,
) -> int:
    """The TP > 1 head component, per chunk statistics path (#1068's model at TP1).

    The eager FP32 statistics keep #1068's seven BF16 buffers. Chunks whose
    kernel is attempted run it, or after a failure the bounded eager
    statistics (``_impl._EagerLocalStats``): the kernel path's buffers plus
    ``_eager_stats_extra_bytes``. A gradient wave's kernel path holds the
    saved logits and the statistics gradient: target and top-k gradients
    join it inside the statistics backward. Requested logits hold the local
    logits, their indexed copy, the gather buffer and its concatenation
    (Megatron's gather along the vocabulary): 2 + 2 TP. Statistics capacity
    adds their index vectors: ``_head_target_bytes``.
    """
    dense = self._head_workspace_bytes(rows)
    if not _head_target_backward(self):
        return dense
    tp = self._topology_key()[1]
    needs_statistics = any(
        request.target_tokens is not None or request.top_k is not None
        for request in requests
    )
    if not needs_statistics:
        logits = any(request.logits for request in requests)
        return dense if lower_bound or not logits else (2 + 2 * tp) * dense
    kernel = 2 * dense if grad_enabled else dense
    low = self._head_projection_rows(
        requests, positions=positions, lower_bound=True, uncapped=True
    )
    if lower_bound:
        # A rejection bound monotone in rows: only unconditional components,
        # and the eager statistics only when no kernel can run at all.
        return max(
            kernel,
            0
            if _impl._triton_head_stats_available(self)
            else 7 * self._head_workspace_bytes(low),
        )
    if positions is not None:
        # The selected layout's packed union: every chunk exactly.
        fallback = _head_fallback_bytes(self, low, low)
    else:
        # An acceptance bound: any projected count the union can take.
        high = self._head_projection_rows(requests, positions=positions, uncapped=True)
        fallback = _head_fallback_bytes(self, low, high)
    logits = (2 + 2 * tp) * dense if any(request.logits for request in requests) else 0
    return max(
        kernel + _eager_stats_extra_bytes(_head_vocabulary(self), rows),
        fallback,
        logits,
    ) + _head_target_bytes(rows, requests, grad_enabled=grad_enabled)


# The statistics' index vectors beside the head buffers, measured on CPU and
# H200 at up to 1,000 targets per row. Per target (a label or top-k entry) in
# the running chunk: its row and column, their grouping by row and FP32
# gradient (32 B in the kernel paths' backward; the TP1 fallback's forward
# holds 24 B beside the top-k selection).
_HEAD_CHUNK_TARGET_BYTES = 36
# Per label of the wave: the checkpointed (row, label) pairs, each chunk's
# log-probs and their gradient (28 B).
_HEAD_LABEL_BYTES = 32
# Per top-k entry of a gradient wave: the requested log-probs' gradient.
_HEAD_TOP_K_GRADIENT_BYTES = 4
# Per row of the running chunk: maxima, sums, log-normalizers, positions.
# Not the Triton backward's (row, vocab block) target offsets, 16 B per
# 4,096-entry block (about 500 B per row of a 124,160-entry shard): TP > 1
# covers them with the bounded statistics' increment, charged even when the
# kernel runs, and TP1 with the fallback's seven buffers. Trimming either
# needs this term to price them.
_HEAD_CHUNK_ROW_BYTES = 128


def _head_target_bytes(
    rows: int, requests: Sequence[AnyForwardInput], *, grad_enabled: bool
) -> int:
    """The statistics' index vectors for ``rows`` projected head rows: per row
    and target of the largest chunk, and per label and top-k entry retained
    across chunks. Requested outputs are charged separately."""
    chunk = min(rows, _impl._HEAD_CHUNK_TOKENS)
    labels = width = top_k = 0
    for request in requests:
        tokens = int(request.input_tokens.numel())
        if request.target_tokens is not None and tokens:
            labels += int(request.target_tokens.numel())
            width += -(-int(request.target_tokens.numel()) // tokens)
        if request.top_k is not None:
            top_k += tokens * int(request.top_k)
            width += int(request.top_k)
    return (
        _HEAD_CHUNK_ROW_BYTES * chunk
        + _HEAD_CHUNK_TARGET_BYTES * min(chunk * width, labels + top_k)
        + _HEAD_LABEL_BYTES * labels
        + (_HEAD_TOP_K_GRADIENT_BYTES * top_k if grad_enabled else 0)
    )


def _eager_stats_extra_bytes(vocabulary: int, rows: int) -> int:
    """The bounded eager statistics' live increment over the kernel path for a
    ``rows`` x ``vocabulary`` head chunk (``_impl._EagerLocalStats``): one FP32
    row sub-chunk, plus per-row FP32 maxima, sums and their gradient."""
    rows = min(rows, _impl._HEAD_CHUNK_TOKENS)
    step = -(-rows // _impl._EAGER_STATS_SUBCHUNKS)
    return 4 * vocabulary * step + 16 * rows


def _frozen_head_bytes(
    vocabulary: int,
    rows: int,
    *,
    target_backward: bool,
    statistics: bool,
    grad: bool,
    tp: int,
    targets: int,
) -> int:
    """Replay's head charge from frozen facts, as live admission prices it.

    TP1 is #1068's capacity charge. TP > 1 is ``_tp_head_workspace_bytes``'s
    kernel path plus the bounded statistics' increment, or the gathered logits
    copies. Capture declines TP > 1 waves with an eager chunk or with logits
    beside statistics: the facts record neither. ``targets`` is
    ``_head_target_bytes`` of the verified requests.
    """
    dense = _dense_head_bytes(vocabulary, rows)
    if not dense or not target_backward:
        return dense
    if not statistics:
        return (3 if tp == 1 else 2 + 2 * tp) * dense
    if tp == 1:
        return 7 * dense + targets
    kernel = 2 * dense if grad else dense
    return kernel + _eager_stats_extra_bytes(vocabulary, rows) + targets


def _head_fallback_bytes(self: TrainerRank, low: int, high: int) -> int:
    """Seven BF16 buffers of the largest head chunk that runs the eager fallback
    for any projected-row count from ``low`` to ``high``."""
    chunk = _impl._HEAD_CHUNK_TOKENS
    vocabulary = _head_vocabulary(self)
    sizes: set[int] = set()
    if high - low >= chunk:
        sizes.update(range(1, min(high, chunk) + 1))  # every tail occurs
    else:
        for rows in range(low, high + 1):
            sizes.update(_head_chunk_sizes(rows))
    eager = [
        size for size in sizes if not _impl._triton_head_stats(self, size, vocabulary)
    ]
    return 7 * self._head_workspace_bytes(max(eager)) if eager else 0


def _head_chunk_sizes(rows: int) -> tuple[int, ...]:
    """The distinct chunk sizes the head runs over ``rows`` projected rows."""
    chunk = _impl._HEAD_CHUNK_TOKENS
    return tuple(
        size for size in (min(rows, chunk), rows % chunk * (rows > chunk)) if size
    )


def _plan_head_workspace_bytes(self: TrainerRank, plan: _impl._FlatForwardPlan) -> int:
    peak = 0
    for group in plan.groups:
        requests = tuple(item.request for item in group.items)
        peak = max(
            peak,
            self._group_head_workspace_bytes(
                self._head_projection_rows(
                    requests, positions=group.packed.positions_by_sequence
                ),
                requests,
                grad_enabled=group.grad_enabled,
                positions=group.packed.positions_by_sequence,
            ),
        )
    return peak


def _plan_hybridep_growth_bytes(self: TrainerRank, plan: _impl._FlatForwardPlan) -> int:
    """HybridEP buffer growth this plan triggers before its forward.

    The buffer is allocated outside the PyTorch allocator after admission,
    so neither the free-memory sample nor a learned peak includes it.
    """
    return _hybridep_growth_from_dimensions(_plan_hybridep_dimensions(self, plan))


def _plan_hybridep_dimensions(
    self: TrainerRank, plan: _impl._FlatForwardPlan
) -> tuple[int, int, int, int, int] | None:
    """Capacity, communication ranks, hidden width, expert columns, held rows."""
    provider: Any = getattr(getattr(self, "runtime", None), "provider", None)
    ep = int(getattr(provider, "expert_model_parallel_size", 1) or 1)
    if ep <= 1 or not plan.groups:
        return None
    from megatron.core.transformer.moe import fused_a2a

    from art.megatron.train import _hybridep_token_capacity

    topology = self._topology()
    sequence = max(
        int(
            _impl._pad_packed_batch(
                group.packed, multiple=int(topology.tp)
            ).tokens.shape[1]
        )
        for group in plan.groups
    )
    rows = max(rows for rows, _ in self._plan_group_rows(plan))
    capacity = max(_hybridep_token_capacity(sequence, int(topology.cp)), rows)
    current = fused_a2a._hybrid_ep_buffer
    held = (
        0
        if current is None
        else int(current.configurer.buffer_config.max_num_of_tokens_per_rank)
    )
    etp = int(getattr(provider, "expert_tensor_parallel_size", 1) or 1)
    ranks = ep * etp
    return (
        capacity,
        ranks,
        int(provider.hidden_size),
        int(provider.num_moe_experts) * etp,
        held,
    )


def _hybridep_growth_from_dimensions(
    dimensions: tuple[int, int, int, int, int] | None,
) -> int:
    if dimensions is None:
        return 0
    capacity, ranks, hidden, experts, held = dimensions
    if _impl._hybridep_rows_per_rank(capacity, ranks) <= held:
        return 0
    # The old buffer stays referenced while its replacement is allocated.
    return _impl._hybridep_buffer_bytes(capacity, ranks, hidden, experts)


def _checkpoint_moe_bytes_per_token(self: TrainerRank) -> int:
    forward = self._moe_output_bytes_per_token
    gradient = self._moe_checkpoint_grad_bytes_per_token
    if (
        type(forward) is not int
        or forward < 0
        or type(gradient) is not int
        or gradient < forward
    ):
        raise ValueError("Invalid constructor checkpoint MoE coefficient")
    return gradient


def _moe_workspace_terms(
    self: TrainerRank,
    *,
    checkpoint_grad: bool = False,
    slot_ref: "LoRASlotRef | None" = None,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Maximum of same-layer affine stages, not a retained multi-layer bank.

    The constructor cache covers original tensors. Explicit slots are
    repriced from their tensor metadata and original owners, including
    this rank's exact dispatcher wrapper. Ordinary non-checkpoint gradients
    retain only forward-stage coverage.
    """
    coefficient = (
        self._checkpoint_moe_bytes_per_token()
        if checkpoint_grad
        else self._moe_output_bytes_per_token
    )
    stages = getattr(
        self,
        "_moe_gradient_stages" if checkpoint_grad else "_moe_forward_stages",
        (),
    )
    if slot_ref is not None and slot_ref.name is not None:
        selected: list[tuple[int, int]] = []
        coefficient = (
            _impl._moe_output_bytes_per_token(
                self.runtime.model,
                self._parallel_shape,
                checkpoint_grad=checkpoint_grad,
                converted_stages=selected,
                slot_ref=slot_ref,
            )
            if self._moe_layers
            else 0
        )
        stages = tuple(selected) if coefficient else ()
    if type(stages) is not tuple or any(
        type(stage) is not tuple
        or len(stage) != 2
        or any(type(value) is not int or value < 0 for value in stage)
        for stage in stages
    ):
        raise ValueError("Invalid constructor converted-weight stages")
    return coefficient, stages


def _moe_workspace_from_terms(
    rows: int, terms: tuple[int, tuple[tuple[int, int], ...]]
) -> int:
    coefficient, stages = terms
    return (
        max(
            rows * coefficient,
            *(rows * per_row + fixed for per_row, fixed in stages),
        )
        if stages and rows > 0
        else rows * coefficient
    )


def _moe_workspace_bytes(
    self: TrainerRank,
    rows: int,
    *,
    checkpoint_grad: bool = False,
    slot_ref: "LoRASlotRef | None" = None,
) -> int:
    return _moe_workspace_from_terms(
        rows,
        _moe_workspace_terms(self, checkpoint_grad=checkpoint_grad, slot_ref=slot_ref),
    )


def _checkpoint_layers(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
) -> int:
    """Conservative saved-boundary charge and one disjoint MoE workspace.

    Count actual local full/uniform/1 boundaries, including aliases, rather
    than claiming measured distinct storage. Only this call's new groups
    enter the term; already-live graphs remain in the availability baseline.
    No-grad groups also keep decoder input, current layer input, its MLP
    residual and norm output across the MoE stage. Count these four row
    tensors separately from returned outputs, allowing storage aliases.
    This is not a bound for custom preprocessing, attention, or all backward.
    With sequence parallelism a rank saves only its shard of each boundary;
    that is priced only where ``_sequence_parallel_floor_covered`` holds,
    and there, for gradient waves, with the recomputed layer's explicit
    workspace (``_sequence_parallel_workspace_bytes``) and its GDN layer's
    recurrent states for ``gdn_segments`` (gradient groups' segments) plus
    padding, beside one input-gradient shard instead of the TP1 repeat.
    """
    if not group_rows or len(self.runtime.model) != 1:
        return 0
    try:
        decoder = _impl._language_model(self.runtime.model[0]).decoder
    except (AttributeError, RuntimeError):
        return 0
    try:
        from megatron.core.transformer.transformer_block import TransformerBlock
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        return 0

    if type(decoder) is not TransformerBlock:
        return 0
    config = decoder.config
    layers = len(decoder.layers)
    _, tp, cp, pp = self._topology_key()
    expected = {
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "distribute_saved_activations": False,
        "sequence_parallel": tp > 1,
        "fp32_residual_connection": False,
        "cpu_offloading": False,
        "cuda_graph_impl": "none",
    }
    if (
        decoder.training is not True
        or layers <= 0
        or layers != decoder.num_layers_per_pipeline_rank
        or layers != config.num_layers
        or config.hidden_size != self._hidden_size
        or config.params_dtype is not _impl.torch.bfloat16
        or self._param_dtype_size != 2
        or next(self.runtime.model[0].parameters()).dtype is not _impl.torch.bfloat16
        or pp != 1
        or (tp > 1 and not self._sequence_parallel_floor_covered(tp, cp))
        or any(
            type(getattr(config, name, None)) is not type(value)
            or getattr(config, name) != value
            for name, value in expected.items()
        )
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or any(
            name in vars(decoder)
            for name in ("forward", "_checkpointed_forward", "_get_layer")
        )
        or getattr(decoder, "_forward_hooks", None)
        or getattr(decoder, "_forward_pre_hooks", None)
    ):
        return 0
    return layers


def _checkpoint_memory_floor(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
    slot_refs: tuple["LoRASlotRef | None", ...] | None = None,
    gdn_segments: int = 0,
) -> tuple[int, int]:
    layers = _checkpoint_layers(self, group_rows)
    if not layers:
        return 0, 0
    return _checkpoint_floor_from_facts(
        self,
        group_rows,
        slot_refs,
        gdn_segments,
        layers,
        hybridep_rows=_checkpoint_hybridep_rows(self, group_rows),
    )


def _checkpoint_hybridep_rows(
    self: TrainerRank, group_rows: tuple[tuple[int, bool], ...]
) -> int | None:
    """Live graph extent needed by the supported HybridEP recompute path."""
    hybrid_rows = None
    if (
        any(grad for _, grad in group_rows)
        and self._topology_key() == (1, 1, 2, 1)
        and self._parallel_shape == _impl.ParallelShape(tp=1, cp=2, ep=2, etp=1)
        and self._moe_memory_supported
    ):
        hybrid_rows = max(rows for rows, _ in group_rows)
        if any(
            _impl._graph_marker_is_live(ref) for ref in self._pending_hybridep_graphs
        ):
            hybrid_rows = max(hybrid_rows, self._hybridep_rows_high_water)
    return hybrid_rows


def _checkpoint_floor_from_facts(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
    slot_refs: tuple["LoRASlotRef | None", ...] | None,
    gdn_segments: int,
    layers: int,
    *,
    hybridep_rows: int | None = None,
) -> tuple[int, int]:
    if not layers:
        return 0, 0
    gradient_rows = sum(rows for rows, grad in group_rows if grad)
    _, tp, _, _ = self._topology_key()
    # Physical rows are padded to a multiple of TP; each rank saves its shard.
    retained = (
        sum(-(-rows // tp) for rows, grad in group_rows if grad)
        * layers
        * self._hidden_size
        * 2
    )
    if gradient_rows:
        self._checkpoint_moe_bytes_per_token()
    refs = (None,) * len(group_rows) if slot_refs is None else slot_refs
    workspace = max(
        self._moe_workspace_bytes(rows, checkpoint_grad=grad, slot_ref=ref)
        + (0 if grad else 4 * rows * self._hidden_size * 2)
        for (rows, grad), ref in zip(group_rows, refs, strict=True)
    )
    if tp > 1 and gradient_rows:
        # The recomputed layer's workspace, for the largest gradient group:
        # groups run their backward one after another.
        workspace = max(
            workspace,
            *(
                self._sequence_parallel_workspace_bytes(rows)
                for rows, grad in group_rows
                if grad
            ),
        )
    if tp > 1 and self._gdn_layers and gradient_rows:
        # Recurrent states grow with segments, not rows; backward recomputes
        # one layer at a time. Padding to TP adds up to TP - 1 one-token
        # roots per group. Kernel-internal chunk states are not bounded here.
        roots = gdn_segments + (tp - 1) * sum(grad for _, grad in group_rows)
        workspace += math.ceil(roots * self._gdn_segment_layer_bytes())
    if hybridep_rows is not None:
        workspace = max(workspace, -(-hybridep_rows // 4) * 4 * self._hidden_size * 2)
    return retained, workspace


def _sequence_parallel_workspace_bytes(self: TrainerRank, rows: int) -> int:
    """The recomputed layer's live workspace at a TP x SP backward peak.

    Traced on dense Qwen3.8-27B at TP4 (25,728 rows) and TP2 (20,816 to
    66,284 rows, one to four GDN segments), cold and warm: every peak was a
    recomputed layer's MLP FC1 stage. Its workspace covers, over the gathered
    rows, the SP-gathered norm output (2H), the FC1 stage (6F/TP: GEMM
    output, gate and up LoRA outputs and their sum; the SwiGLU live set if
    wider) and the recomputed mixer at its projection widths / TP (GDN held
    0.885-0.887 of that bound); over the sharded rows, norm outputs (3.22H
    measured, priced at 3.25H); and the wave length's fp32 rotary cache
    (``kv_channels`` per row bounds the traced 64-dim partial rotary).

    Recomputing layer i holds the L-shard ``retained`` term's i + 1
    checkpoint inputs plus its own residual: L + 1 - gap shards, where gap
    counts the layers above it. Each mixer kind peaks at its highest layer;
    the boundary shards beyond L are charged here, per kind, from the
    decoder's layer order (one extra shard for a kind whose order is
    unknown). In the traced model the top layer is attention and the top GDN
    layer (one below) peaks: exactly L shards. The input-gradient shard, LoRA
    intermediates, GDN segment states and cold transients are separate terms.
    """
    if rows <= 0:
        return 0
    geometry = self._geometry
    tp = self._topology_key()[1]
    hidden = self._hidden_size
    ffn = geometry.ffn_hidden_size or 4 * hidden
    attention_layers = self._num_layers > self._gdn_layers
    attention = (
        (7 if self._attention_output_gate else 5)
        * geometry.num_attention_heads
        * geometry.kv_channels
        + 3 * geometry.num_query_groups * geometry.kv_channels
    )
    gdn = (
        4 * geometry.gdn_key_heads * geometry.gdn_key_head_dim
        + 8 * geometry.gdn_value_heads * geometry.gdn_value_head_dim
    )
    shard = -(-rows // tp) * hidden
    gaps = getattr(self, "_mixer_top_gaps", None) or (None, None)
    peak = (
        max(
            rows * -(-width // tp) + (1 - (0 if gap is None else gap)) * shard
            for present, width, gap in (
                (attention_layers, attention, gaps[0]),
                (bool(self._gdn_layers), gdn, gaps[1]),
            )
            if present
        )
        if self._num_layers
        else 0
    )
    stage = max(6, self._mlp_activation_factor) * ffn
    gathered = rows * (2 * hidden + -(-stage // tp))
    sharded = -(-rows // tp) * -(-13 * hidden // 4)
    rotary = 2 * rows * geometry.kv_channels if attention_layers else 0
    return (gathered + sharded + peak + rotary) * self._param_dtype_size


def _sequence_parallel_lora_floor(
    self: TrainerRank,
    floor: tuple[int, int],
    group_rows: tuple[tuple[int, bool], ...],
    signature: _impl._MemorySignature,
) -> tuple[int, int]:
    """Add the recomputed layer's LoRA ``x @ A`` intermediates to a TP x SP floor.

    Each adapted module of the recomputed layer saves rows x rank over the
    gathered rows (FC1 gate/up measured 16 B/row at rank 4). Charge the
    decoder's most-adapted layer at the gradient slots' largest rank, from
    the signature's slot shapes ((ndim, *A.shape, *B.shape) per module).
    """
    retained, workspace = floor
    if not retained or self._topology_key()[1] == 1:
        return floor
    rank = max(
        (
            shape[2]
            for grad, shapes in signature.slot_shapes
            if grad
            for shape in shapes
            if len(shape) == 5 and shape[0] == 2
        ),
        default=0,
    )
    modules = getattr(self, "_lora_modules_per_layer", None)
    if not modules:
        # Unread (e.g. reports without the recorded count): every adapted
        # module of the slot, which bounds any one layer's.
        modules = max(
            (len(shapes) for grad, shapes in signature.slot_shapes if grad), default=0
        )
    rows = max((rows for rows, grad in group_rows if grad), default=0)
    return retained, workspace + rows * modules * rank * self._param_dtype_size


def _cold_recompute_transient_bytes(self: TrainerRank) -> int:
    """Fixed transients of an unprofiled gradient wave's first execution."""
    return (
        _impl._SEQUENCE_PARALLEL_COLD_TRANSIENT_BYTES
        if self._topology_key()[1] > 1
        else _impl._COLD_RECOMPUTE_TRANSIENT_BYTES
    )


def _checkpoint_input_gradient_bytes(
    self: TrainerRank, group_rows: tuple[tuple[int, bool], ...], retained: int
) -> int:
    """The input-gradient term beside the checkpoint floor's boundaries.

    TP1 charges one logical gradient per boundary, which also stands in for
    the recompute workspace. Under TP x SP the workspace is priced
    explicitly, and the recompute peak held one input-gradient shard of the
    group being recomputed.
    """
    tp = self._topology_key()[1]
    if not retained or tp == 1:
        return retained
    rows = max((rows for rows, grad in group_rows if grad), default=0)
    return -(-rows // tp) * self._hidden_size * self._param_dtype_size


def _dense_mlp_widths(
    self: TrainerRank, slot_refs: Sequence["LoRASlotRef | None"] | None = None
) -> tuple[int, int]:
    """The covered dense (gradient stage, no-grad transient) per row, or 0s.

    Only at TP1/CP2/PP1, where it was traced, and only where the checkpoint
    floor prices the decoder (``_checkpoint_layers``). Named slots are
    rechecked: their adapters must stay within the priced rank.
    """
    stage = getattr(self, "_dense_recompute_bytes_per_token", 0)
    no_grad = getattr(self, "_dense_no_grad_bytes_per_token", 0)
    if (
        type(stage) is not int
        or type(no_grad) is not int
        or stage <= 0
        or no_grad <= 0
        or self._topology_key()[1:] != (1, 2, 1)
        or not _checkpoint_layers(self, ((1, True),))
    ):
        return 0, 0
    for ref in slot_refs or ():
        if ref is None or ref.name is None:
            continue
        slot = _impl._dense_mlp_recompute_bytes_per_token(
            self.runtime.model, ref, hidden_size=self._hidden_size
        )
        if not all(slot):
            return 0, 0
        stage, no_grad = max(stage, slot[0]), max(no_grad, slot[1])
    return stage, no_grad


def _dense_mixer_widths(self: TrainerRank) -> dict[str, int]:
    """A covered dense CP2 model's recomputed mixer bytes per row, by layer type.

    Beside the CP executor's retained records, an attention layer keeps its
    input norm output and five query- and five KV-width tensors (Qwen3.8-27B
    CP2 traces: 78.1 and 79.9 KB per row on the two ranks, 81.9 KB priced).
    A GDN layer keeps its input and norm outputs, the projected q/k/v, their
    l2norm outputs expanded to the value heads, five more value-width
    tensors, the chunk decay matrix and the CP exchange's value-width output
    (Qwen3.6-35B-A3B CP2 traces: 88 KB measured, 94 KB priced).
    """
    geometry = self._geometry
    hidden = self._hidden_size
    widths: dict[str, int] = {}
    if self._gdn_layers < self._num_layers:
        q = geometry.num_attention_heads * geometry.kv_channels or hidden
        kv = geometry.num_query_groups * geometry.kv_channels or hidden
        widths["attention"] = hidden + 5 * q + 5 * kv
    if self._gdn_layers:
        value = geometry.gdn_value_heads * geometry.gdn_value_head_dim
        widths["gdn"] = (
            2 * hidden
            + 2 * geometry.gdn_key_heads * geometry.gdn_key_head_dim
            + 2 * geometry.gdn_value_heads * geometry.gdn_key_head_dim
            + 7 * value
            + 64 * geometry.gdn_value_heads
        )
    return {kind: width * self._param_dtype_size for kind, width in widths.items()}


def _dense_layout_floors(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
    slot_refs: tuple["LoRASlotRef | None", ...] | None,
    layouts: tuple[_GroupLayout, ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    """Each CP rank's (boundaries, workspace) for a covered dense model, or None.

    All groups gradient or all no-grad, on every rank's own layouts. A saved
    layer input arrives in the GDN layout when the layer follows a GDN layer
    in its island, else in the attention layout. A recomputed attention
    layer keeps its mixer width on its attention rows plus the executor's
    retained records, a GDN layer its width on its GDN rows; either keeps
    its residual, pre-MLP norm output and MLP stage on those rows. A no-grad
    layer holds its stage on its rows. A GDN layer also holds the rank's
    recurrent states (``_GroupLayout.gdn_states``, half a
    ``_gdn_segment_layer_bytes`` each).
    """
    refs = (None,) * len(group_rows) if slot_refs is None else slot_refs
    if layouts is None or len({grad for _, grad in group_rows}) != 1:
        return None
    stage, no_grad = self._dense_mlp_widths(refs)
    if not stage:
        return None
    gradient = group_rows[0][1]
    hidden = self._hidden_size * 2
    inputs = self._layer_gdn_inputs()
    gdn_inputs = sum(inputs)
    attention_inputs = len(inputs) - gdn_inputs
    widths = self._dense_mixer_widths()
    floors: list[tuple[int, int]] = []
    for rank in range(len(layouts[0].attention_rows)):
        retained = workspace = 0
        for layout in layouts:
            attention = max(1, layout.attention_rows[rank])
            gdn = (
                attention if layout.gdn_rows is None else max(1, layout.gdn_rows[rank])
            )
            states = layout.gdn_states[rank] if layout.gdn_states else 0
            if gradient:
                retained += hidden * (attention_inputs * attention + gdn_inputs * gdn)
            for kind, rows, extra in (
                ("attention", attention, layout.attention_retained[rank] * gradient),
                ("gdn", gdn, math.ceil(states * self._gdn_segment_layer_bytes() / 2)),
            ):
                if kind in widths:
                    per_row = widths[kind] + 2 * hidden + stage if gradient else no_grad
                    workspace = max(workspace, rows * per_row + extra)
        floors.append((retained, workspace))
    return tuple(floors)


def _dense_adapter_gradient_extra(
    self: TrainerRank,
    floor: tuple[int, int],
    floors: tuple[tuple[int, int], ...],
    group_rows: tuple[tuple[int, bool], ...],
    slot_refs: tuple["LoRASlotRef | None", ...] | None,
    layouts: tuple[_GroupLayout, ...] | None,
) -> int:
    """Adapter gradients at the recompute peak beyond ``floor`` on CP layouts.

    ``floor`` is ``_checkpoint_memory_floor``'s (boundaries, workspace) and
    ``floors`` each rank's (``_dense_layout_floors``). Each rank releases
    its own boundaries as backward proceeds (``_layout_layer_boundaries``):
    a rank with fewer rows releases less, so its extra is larger, but its own
    floor is smaller by what it never saved. Pair each rank's extra with its
    own boundaries plus the larger of its workspace and the floor's.
    """
    assert layouts is not None
    slots = [
        slot for slot, _ in self._checkpoint_gradient_groups(group_rows, slot_refs)
    ]
    extras = [
        self._checkpoint_adapter_gradient_bytes(tuple(zip(slots, rank, strict=True)))
        for rank in self._layout_layer_boundaries(layouts)
    ]
    if not any(extras):
        return 0
    retained, workspace = floor
    return max(
        0,
        max(
            rank_retained + max(rank_workspace, workspace) + extra
            for (rank_retained, rank_workspace), extra in zip(
                floors, extras, strict=True
            )
        )
        - retained
        - workspace,
    )


def _layer_gdn_inputs(self: TrainerRank) -> tuple[bool, ...]:
    """Per decoder layer, whether its saved input arrives in the GDN layout.

    It does when the layer follows a GDN layer in its island
    (``_art_gdn_island_boundary``); otherwise it is in the attention layout.
    """
    decoder = _impl._language_model(self.runtime.model[0]).decoder
    return tuple(
        getattr(getattr(layer, "_art_gdn_island_boundary", None), "input_layout", "")
        == "gdn"
        for layer in decoder.layers
    )


def _layout_layer_boundaries(
    self: TrainerRank, layouts: tuple[_GroupLayout, ...]
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Each CP rank's saved-boundary bytes per group and decoder layer.

    As ``_dense_layout_floors`` prices them: a layer saves its
    input in the GDN layout when it follows a GDN layer in its island and in
    the attention layout otherwise, on that rank's rows of the group.
    """
    gdn_inputs = self._layer_gdn_inputs()
    hidden = self._hidden_size * 2
    ranks = []
    for rank in range(len(layouts[0].attention_rows)):
        groups = []
        for layout in layouts:
            attention = max(1, layout.attention_rows[rank])
            gdn = (
                attention if layout.gdn_rows is None else max(1, layout.gdn_rows[rank])
            )
            groups.append(
                tuple(hidden * (gdn if is_gdn else attention) for is_gdn in gdn_inputs)
            )
        ranks.append(tuple(groups))
    return tuple(ranks)


def _layout_pricing_supported(
    self: TrainerRank, topology: tuple[int, int, int, int]
) -> bool:
    """Whether layout-aware pricing models this runtime and plan shape.

    Only for a covered dense model (``_dense_mlp_widths``); other models keep
    the busiest-rank pricing.
    """
    _dp, tp, cp, pp = topology
    if (tp, cp, pp) != (1, 2, 1):
        return False
    if not self._dense_mlp_widths()[0]:
        return False
    geometry = self._geometry
    if not geometry.num_attention_heads or not geometry.kv_channels:
        return False
    if len(self.runtime.model) != 1:
        return False
    try:
        decoder = _impl._language_model(self.runtime.model[0]).decoder
        from art.megatron.context_parallel.core_attention import (
            ArtContextParallelCoreAttention,
        )
    except (AttributeError, RuntimeError, ModuleNotFoundError):
        return False
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return False
    for layer in layers:
        boundary = getattr(layer, "_art_gdn_island_boundary", None)
        if boundary is not None and boundary.is_gdn:
            continue
        if self._gdn_layers and boundary is None:
            return False
        core = getattr(getattr(layer, "self_attention", None), "core_attention", None)
        if (
            type(core) is not ArtContextParallelCoreAttention
            or getattr(core, "softmax_offset", None) is not None
        ):
            return False
    return True


def _minimum_layouts(
    self: TrainerRank, physical_rows: Sequence[int], cp: int
) -> tuple[_GroupLayout, ...]:
    """Even-share layouts keeping the least attention state: a lower bound.

    Every rank's total grows with its own rows, and every rank keeps at
    least an aligned local stage's state per row (each row attends to
    itself), so the largest rank's total is at least the total at the mean
    rows. The mean is at least the floor of an even share, which is why
    this rounds down; rounding up can exceed a split's exact cost.
    """
    from art.megatron.context_parallel.executor import (
        minimum_retained_bytes_per_row,
    )

    geometry = self._geometry
    per_row = minimum_retained_bytes_per_row(
        q_heads=int(geometry.num_attention_heads),
        kv_heads=int(geometry.num_query_groups),
        head_dim=int(geometry.kv_channels),
        value_head_dim=int(geometry.kv_channels),
        element_size=self._param_dtype_size,
    )
    return tuple(
        _impl._GroupLayout(
            attention_rows=(rows // cp,) * cp,
            gdn_rows=(rows // cp,) * cp if self._gdn_layers else None,
            attention_retained=(rows // cp * per_row,) * cp,
        )
        for rows in physical_rows
    )


def _te_workspace_growth_bytes(self: TrainerRank) -> int:
    """Transformer Engine's cuBLAS workspaces, until this device has its own.

    TE caches a workspace per (device, overlap, grouped GEMM) for the
    process. A covered dense model runs ordinary GEMMs only, so this
    device's ordinary entry marks it warm; an unreadable cache counts as
    cold. A cold call allocates only that one, but about 130 MB of other
    first-use buffers beside it (Qwen3.8-27B CP2: 164 MB in all), which the
    full allowance covers.
    """
    try:
        from transformer_engine.pytorch.cpp_extensions import gemm
    except ImportError:
        return _impl._TE_CUBLAS_WORKSPACE_BYTES
    # functools.lru_cache keeps its entries in a dict its wrapper references.
    key = (self.device.index, False, False)
    if any(
        type(entries) is dict and key in entries
        for entries in gc.get_referents(gemm.get_cublas_workspace)
    ):
        return 0
    return _impl._TE_CUBLAS_WORKSPACE_BYTES


def _gradient_slots(
    group_rows: Sequence[tuple[int, bool]],
    slot_refs: Sequence[LoRASlotRef | None] | None,
) -> frozenset[LoRASlotRef]:
    """Gradient groups' adapter slots; the base model (no name) has none."""
    return frozenset(
        ref
        for (_, grad), ref in zip(
            group_rows, slot_refs or (None,) * len(group_rows), strict=True
        )
        if grad and ref is not None and ref.name is not None
    )


def _pending_adapter_gradient_bytes(
    self: TrainerRank, refs: Iterable[LoRASlotRef]
) -> tuple[int, ...]:
    """Local adapter gradient bytes the next backward allocates, per decoder layer.

    One entry per decoder layer, then one for parameters outside the
    decoder. Backward reaches a parameter's highest decoder layer first, so
    a parameter shared across layers counts there once; those outside the
    decoder count as live throughout. Only unallocated gradients count:
    within a step, later waves find the rest in the availability baseline.
    Empty when none is pending.
    """
    refs = tuple(dict.fromkeys(refs))
    if not refs or len(self.runtime.model) != 1:
        return ()
    try:
        from art.megatron.lora import LoRA
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        return ()
    chunk = self.runtime.model[0]
    try:
        layers = _impl._language_model(chunk).decoder.layers
    except (AttributeError, RuntimeError):
        return ()
    layer_of: dict[int, int] = {}
    params: dict[int, torch.nn.Parameter] = {}

    def slot_params(
        modules: Iterable[torch.nn.Module],
    ) -> Iterator[torch.nn.Parameter]:
        for module in modules:
            # A LoRA without slot tables holds no slot parameters.
            if isinstance(module, LoRA) and "_slot_keys" in vars(module):
                for ref in refs:
                    yield from module.lora_slot_params(ref)

    for index, layer in enumerate(layers):
        for param in slot_params(layer.modules()):
            params[id(param)] = param
            layer_of[id(param)] = max(layer_of.get(id(param), -1), index)
    # Outside the decoder (a head runs its backward first) is live
    # throughout, even for a parameter or module a decoder layer also uses:
    # walk every path except through the layers themselves.
    outside: list[torch.nn.Module] = []
    visited: set[int] = set()
    pending_modules: list[torch.nn.Module] = [chunk]
    while pending_modules:
        module = pending_modules.pop()
        if module is layers or id(module) in visited:
            continue
        visited.add(id(module))
        outside.append(module)
        pending_modules.extend(module.children())
    for param in slot_params(outside):
        params[id(param)] = param
        layer_of[id(param)] = len(layers)
    # A checkpoint's other trainable parameters (custom objects) have no
    # decoder position; count them as live throughout.
    for ref in refs:
        slot = (
            None
            if ref.name is None
            else getattr(self, "_checkpoint_slots", {}).get(ref.name)
        )
        for param in () if slot is None else slot.params:
            if id(param) not in params:
                params[id(param)] = param
                layer_of[id(param)] = len(layers)
    sizes = [0] * (len(layers) + 1)
    for param_id, param in params.items():
        if (
            param.requires_grad
            and param.grad is None
            and getattr(param, "main_grad", None) is None
        ):
            sizes[layer_of[param_id]] += param.numel() * param.element_size()
    return tuple(sizes) if any(sizes) else ()


def _checkpoint_gradient_groups(
    self: TrainerRank,
    group_rows: Sequence[tuple[int, bool]],
    slot_refs: Sequence[LoRASlotRef | None] | None,
) -> tuple[tuple[LoRASlotRef | None, tuple[int, ...]], ...]:
    """Each gradient group's adapter slot and per-layer saved boundaries.

    In execution order, as ``_checkpoint_memory_floor`` prices them: every
    decoder layer saves the group's rows (this rank's TP shard). The base
    model (no name) has no adapter slot.
    """
    tp = self._topology_key()[1]
    return tuple(
        (
            ref if ref is not None and ref.name is not None else None,
            (-(-rows // tp) * self._hidden_size * 2,) * self._num_layers,
        )
        for (rows, grad), ref in zip(
            group_rows, slot_refs or (None,) * len(group_rows), strict=True
        )
        if grad
    )


def _checkpoint_adapter_gradient_bytes(
    self: TrainerRank, groups: Sequence[tuple[LoRASlotRef | None, Sequence[int]]]
) -> int:
    """The recompute backward's adapter-gradient peak beyond released boundaries.

    ``groups`` gives each gradient group's adapter slot (None for the base
    model) and each decoder layer's saved-boundary bytes
    (``_adapter_gradient_walk``).
    """
    chains = []
    for slot, boundaries in groups:
        pending = () if slot is None else self._pending_adapter_gradient_bytes((slot,))
        if pending and len(pending) != len(boundaries) + 1:
            return 0
        chains.append((pending or (0,) * (len(boundaries) + 1), boundaries))
    return self._adapter_gradient_walk(chains)


def _adapter_gradient_walk(
    chains: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> int:
    """The adapter-gradient peak beyond the floor over gradient groups' backward.

    Each chain is a gradient group's pending gradient bytes (per decoder
    layer, then outside the decoder) and saved-boundary bytes per layer.
    Backward recomputes the last layer first. While it recomputes layer i it
    still holds the saved boundaries of layers 0..i and every adapter
    gradient allocated so far: those of layers i..L-1 (a layer allocates its
    own during its backward) and any outside the decoder. The floor already
    prices all L boundaries at once, so one group's extra peak is
    max(0, max over i of gradients(i..) - boundaries(i+1..)), taken at every
    layer over the real per-layer gradient sizes and the caller's per-layer
    boundaries, not along a uniform-layer line. A short
    first wave peaks at layer 0 (Qwen3.6-35B-A3B CP2: 830-900 MB of expert
    LoRA gradients live at its peak), a long one at the last layer.
    Groups run their backward one after another, not layer by layer
    together: autograd drains the last-forwarded group's chain first, and
    separate backward calls can come in either order. While one group runs,
    each group already run holds all its gradients and none of its
    boundaries, and each group yet to run all its boundaries. Any set of the
    other groups can have run first, so the worst adds every other group
    whose gradients outweigh its boundaries.
    """
    nets = [sum(pending) - sum(boundaries) for pending, boundaries in chains]
    others = sum(max(0, net) for net in nets)
    worst = 0
    for (pending, boundaries), net in zip(chains, nets, strict=True):
        extra = gradients = pending[-1]
        released = 0
        for index in range(len(boundaries) - 1, -1, -1):
            gradients += pending[index]
            extra = max(extra, gradients - released)
            released += boundaries[index]
        worst = max(worst, extra + others - max(0, net))
    return worst


def _retained_memory_bytes(
    self: TrainerRank,
    signature: _impl._MemorySignature,
    *,
    packed_tokens: int,
    logical_tokens: int,
    output_bytes: int,
    required: int,
    checkpoint_retained_bytes: int = 0,
) -> int:
    """Forward-retained bytes, independent of a later backward peak.

    Retain the full estimate until observed near the current scale and
    sharing ratio, so a small forward cannot authorize a much larger split.
    """

    profile = self._memory_profiles.get(signature)
    if profile is None or profile.retained_compute_bytes_per_token is None:
        return required
    if packed_tokens > profile.packed_tokens * _impl._MEMORY_PROFILE_TRUST_GROWTH:
        return required
    ratio = logical_tokens / max(1, packed_tokens)
    if ratio > profile.logical_per_packed * _impl._MEMORY_PROFILE_TRUST_GROWTH:
        return required
    rate = profile.retained_compute_bytes_per_token
    if _impl._packed_priced(signature, self._one_layer_recompute()):
        # Saved head indices, masks and caller saves stay live until
        # backward. The ratio window above bounds the sharing extrapolation.
        retained_compute = (
            rate * packed_tokens
            + _impl._PACKED_PRICED_LOGICAL_ROW_BYTES * logical_tokens
        )
    else:
        retained_compute = rate * max(
            packed_tokens, logical_tokens / profile.logical_per_packed
        )
    retained = output_bytes + max(checkpoint_retained_bytes, retained_compute)
    return min(required, int(retained * _impl._MEMORY_SAFETY_FACTOR))


def _estimate_flat_forward(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    checkpoint: AdapterSelection = _impl.Unset,
    exact: bool = False,
    memory_minimal: bool = False,
    sync_planning_errors: bool = False,
    gdn_segments: list[int] | None = None,
) -> tuple[int, int, _impl._MemorySignature, tuple[tuple[int, bool], ...], int] | None:
    """Estimate packed tokens for width probing.

    Cheap mode (``exact=False``) is one O(tokens) CPU walk of the packing
    primitive and preserves its CUDA None-contract: with
    ``memory_minimal=False`` it is the no-sharing count, an upper bound on
    any planner-selected layout (safe for accepting a width); with
    ``memory_minimal=True`` it is the full-sharing count, the lower bound
    whose feasibility is monotone in width (valid for rejecting one).
    ``exact=True`` prices the planner's actual layouts (memoized by
    content) and is used only inside the band where those bounds disagree.
    Under CP it returns None: per-rank floors need materialized layouts.
    ``gdn_segments`` receives each gradient group's segment count: exact
    layouts' actual counts; in cheap mode, the same kind of bound as the
    token count (twice the requests, as a radix tree has fewer, or one).
    """

    if sync_planning_errors:
        self._ensure_checkpoint_slots_for(requests, checkpoint=checkpoint)
    with self._planning_status(sync_planning_errors):
        groups = self._group_active_request_indices(
            requests,
            checkpoint=checkpoint,
            ensure_slots=not sync_planning_errors,
        )
        if self._moe_layers and any(
            ref is not None and ref.name is not None for (ref, _), _ in groups
        ):
            # This cheap return type has no slot metadata. Materialize the
            # exact plan instead of admitting with the constructor rank.
            return None
        gradient_slots = [
            ref
            for (ref, grad), _ in groups
            if grad and ref is not None and ref.name is not None
        ]
        if (
            gradient_slots
            and getattr(self, "_recompute_granularity", None) == "full"
            and self._pending_adapter_gradient_bytes(gradient_slots)
        ):
            # The step's first backward allocates adapter gradients that
            # only slot metadata can price; the exact plan carries it.
            return None
        if (
            any(mode for (_, mode), _ in groups)
            and _impl._gdn_memory.model_shapes(self) is not None
        ):
            # Pending saves require the actual bucket/replayed-tail geometry.
            # Existing unavailable handling materializes before admission.
            return None
        if self._topology_key()[2] > 1:
            # CP token ownership can be uneven and memory floors are priced
            # per rank. Use the existing exact-plan fallback; a global token
            # count alone cannot price its peak.
            return None
        packed_tokens = 0
        head_workspace_bytes = 0
        group_rows: list[tuple[int, bool]] = []
        for (_slot, grad_enabled), group_indices in groups:
            head_requests = tuple(requests[index] for index in group_indices)
            lower = self._head_projection_rows(head_requests, lower_bound=True)
            upper = self._head_projection_rows(head_requests)
            if exact:
                tree, layout = self._select_group_layout(
                    tuple(
                        requests[index]
                        .input_tokens.reshape(-1)
                        .to(dtype=_impl.torch.long)
                        for index in group_indices
                    ),
                    memory_minimal=memory_minimal,
                    grad_enabled=grad_enabled,
                )
                physical_rows = self._physical_tokens(layout.packed_tokens)
                packed_tokens += physical_rows
                group_rows.append((physical_rows, grad_enabled))
                if grad_enabled and gdn_segments is not None:
                    gdn_segments.append(len(layout.segments))
                projected = upper
                positions = None
                # TP > 1 heads price each chunk's statistics path, so the
                # exact count needs the packed union beyond one chunk too.
                spread = (
                    self._topology_key()[1] > 1
                    and upper == _impl._HEAD_CHUNK_TOKENS
                    and self._head_projection_rows(
                        head_requests, lower_bound=True, uncapped=True
                    )
                    != self._head_projection_rows(head_requests, uncapped=True)
                )
                if lower != upper or spread:
                    packed = _impl.materialize_prefix_tree_layout(
                        tuple(
                            request.input_tokens.reshape(-1).to(dtype=_impl.torch.long)
                            for request in head_requests
                        ),
                        tree,
                        layout,
                        verify_shared_tokens=False,
                    )
                    projected = self._head_projection_rows(
                        head_requests, positions=packed.positions_by_sequence
                    )
                    positions = packed.positions_by_sequence
                head_workspace_bytes = max(
                    head_workspace_bytes,
                    self._group_head_workspace_bytes(
                        projected,
                        head_requests,
                        grad_enabled=grad_enabled,
                        positions=positions,
                    ),
                )
                continue
            # Radix depth is bounded by the number of rows, so ``len(group)``
            # is an unlimited-sharing depth for this group; it is a bound for
            # estimation, not a sharing policy.
            group_packed_tokens = _impl.estimate_prefix_tree_packed_tokens(
                (requests[index].input_tokens.reshape(-1) for index in group_indices),
                max_depth=len(group_indices) if memory_minimal else 0,
            )
            if group_packed_tokens is None:
                return None
            physical_rows = self._physical_tokens(group_packed_tokens)
            packed_tokens += physical_rows
            group_rows.append((physical_rows, grad_enabled))
            if grad_enabled and gdn_segments is not None:
                # Bounds like the token counts: at most twice the requests
                # without sharing (acceptance), at least one with full
                # sharing (rejection); exact pricing counts the rest.
                gdn_segments.append(1 if memory_minimal else 2 * len(group_indices))
            head_workspace_bytes = max(
                head_workspace_bytes,
                self._group_head_workspace_bytes(
                    lower if memory_minimal else upper,
                    head_requests,
                    grad_enabled=grad_enabled,
                    lower_bound=memory_minimal,
                ),
            )

        return (
            packed_tokens,
            self._estimate_group_request_output_bytes(requests),
            self._memory_signature_from_requests(
                requests,
                slot_group_count=len(groups),
                grad_modes=tuple(mode for (_, mode), _ in groups),
                slot_groups=tuple(key for key, _ in groups),
            ),
            tuple(group_rows),
            head_workspace_bytes,
        )


def _update_peak_memory_profile(
    self: TrainerRank,
    plan: _impl._FlatForwardPlan,
    baseline: int | None,
    retained_after: int | None = None,
    *,
    caller_phase: bool = False,
    interval: tuple[int, int] | None = None,
) -> None:
    if baseline is None:
        return
    peak = int(_impl.torch.cuda.max_memory_allocated(self.device))
    observation = getattr(self, "_planner_observation", None)
    if observation is not None:
        observation["peak"] = max(observation["peak"], peak)
    resets = self.__dict__.get("_peak_resets", 0)
    self._peak_reading = (resets, peak)
    self._update_memory_profile(
        plan,
        max(0, peak - baseline),
        retained_bytes=(
            None if retained_after is None else max(0, retained_after - baseline)
        ),
        # Only a whole wave: a nested forward resetting the counter during
        # the yield, or an untracked reset (the counter fell below this
        # wave's forward peak), leaves the peak since then short of it.
        caller_phase=caller_phase
        and interval is not None
        and interval[0] == resets
        and peak >= interval[1],
    )


def _estimate_group_request_output_bytes(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
) -> int:
    total = 0
    for request in requests:
        if request.routed_experts is not None:
            # Retained int64 replay targets for attention and GDN layouts.
            total += 2 * request.routed_experts.numel() * 8
        seq_len = int(request.input_tokens.numel())
        if request.target_tokens is not None:
            total += int(request.target_tokens.numel()) * _impl._dtype_size(
                _impl.torch.float32
            )
        if request.top_k is not None:
            total += (
                seq_len
                * int(request.top_k)
                * (
                    _impl._dtype_size(_impl.torch.float32)
                    + _impl._dtype_size(_impl.torch.long)
                )
            )
        if request.logits:
            if self._padded_vocab_size is None:
                raise RuntimeError("logits output memory requires a GPT model")
            total += seq_len * self._padded_vocab_size * self._param_dtype_size
        if request.hidden_states:
            total += seq_len * self._hidden_size * self._param_dtype_size
    return total


def _memory_signature_from_requests(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
    *,
    slot_group_count: int,
    grad_modes: Iterable[bool],
    slot_groups: Iterable[tuple["LoRASlotRef | None", bool]] = (),
) -> _impl._MemorySignature:
    modes = tuple(sorted(grad_modes))
    shapes = tuple(
        sorted((grad, self._slot_memory_shapes(ref)) for ref, grad in slot_groups)
    )
    return _impl._MemorySignature(
        topology=self._topology_key(),
        planner_coefficients=(self._coefficient_version, self._coefficient_table),
        slot_group_count=slot_group_count,
        request_mix=tuple(
            sorted({_impl._request_mix_key(request) for request in requests})
        ),
        grad_enabled=any(modes),
        grad_modes=modes,
        slot_shapes=shapes if any(any(shape) for _, shape in shapes) else (),
        short_requests=any(_impl._short_request(request) for request in requests),
    )


def _slot_memory_shapes(
    self: TrainerRank, ref: "LoRASlotRef | None"
) -> tuple[tuple[int, ...], ...]:
    """Separate empirical trust across actual selected adapter layouts.

    Sequence-parallel (TP > 1) models record them too: the explicit TP x SP
    checkpoint floor prices LoRA intermediates from these ranks.
    """
    if (
        ref is None
        or ref.name is None
        or isinstance(ref, _impl._LocalLoRASlotRef)
        or not (
            getattr(self, "_moe_layers", 0)
            or getattr(self, "_gdn_layers", 0)
            or getattr(self, "_sequence_parallel", False)
        )
    ):
        # Generic/no-component planning must not import Megatron or walk
        # model owners just to construct its existing memory signature.
        return ()
    from art.megatron.lora import LoRA

    shapes = []
    for chunk in self.runtime.model:
        for module in chunk.modules():
            if type(module) is LoRA:
                tensors = _impl._slot_lora_tensors(module, ref)
                shapes.append(
                    ()
                    if tensors is None
                    else (tensors[0].ndim, *tensors[0].shape, *tensors[1].shape)
                )
    return tuple(shapes)


def _memory_check(
    self: TrainerRank,
    forward: _impl._FlatForwardPlan,
    *,
    sync_across_dp: bool = False,
    sync_planning_errors: bool = False,
) -> _impl._MemoryCheck:
    with self._planning_status(sync_planning_errors):
        required = self._estimate_required_memory_bytes_from_values(
            packed_tokens=forward.packed_tokens,
            output_bytes=forward.output_bytes,
            signature=forward.signature,
            logical_tokens=forward.active_logical_tokens,
            gdn_segments=forward.grad_segment_count,
            group_rows=self._plan_group_rows(forward),
            slot_refs=tuple(g.slot_ref for g in forward.groups),
            head_workspace_bytes=self._plan_head_workspace_bytes(forward),
            checkpoint_floor=_impl._gdn_memory.plan_floor(self, forward),
            retained_tokens=self._plan_retained_tokens(forward),
        ) + int(self._plan_hybridep_growth_bytes(forward) * _impl._MEMORY_SAFETY_FACTOR)
    return self._memory_check_required(required, sync_across_dp=sync_across_dp)


def _refresh_memory_check(
    self: TrainerRank, check: _impl._MemoryCheck, *, sync_across_dp: bool
) -> _impl._MemoryCheck:
    decision = _impl._planner_evidence.current(self)
    # Re-sample each rank against its own demand, not another DP rank's maximum.
    # Older synthetic checks may lack the local producer; keep their safe bound.
    with decision.refresh_of(check.sample) if decision is not None else nullcontext():
        refreshed = self._memory_check_required(
            check.estimated_required_bytes
            if check.local_required_bytes is None
            else check.local_required_bytes,
            sync_across_dp=sync_across_dp,
        )
    return _impl.replace(
        check,
        estimated_required_bytes=refreshed.estimated_required_bytes,
        available_bytes=refreshed.available_bytes,
        fits=refreshed.fits and check.cpu_fits,
        sample=refreshed.sample,
        local_required_bytes=refreshed.local_required_bytes,
        local_available_bytes=refreshed.local_available_bytes,
    )


def _memory_check_required(
    self: TrainerRank,
    required: int,
    *,
    sync_across_dp: bool = False,
) -> _impl._MemoryCheck:
    decision = _impl._planner_evidence.current(self)
    details: dict[str, Any] | None = {} if decision is not None else None
    local_required, scope = required, "local"
    if _impl.dist.is_available() and _impl.dist.is_initialized():
        group = None if sync_across_dp else self._forward_memory_group()
        scope = "world" if group is None else "tp_cp"
        values = _impl.torch.tensor(
            [float(required), 0.0, 0.0],
            device=self.device if self.device.type == "cuda" else "cpu",
            dtype=_impl.torch.float64,
        )
        _impl.dist.all_reduce(values[0], op=_impl.dist.ReduceOp.MAX, group=group)
        required = int(values[0].item())
        error: BaseException | None = None
        try:
            available = (
                self._available_memory_bytes()
                if details is None
                else self._available_memory_bytes(details)
            )
            local_available = available
        except BaseException as exc:
            error, available = exc, -1
        exchange_error: BaseException | None = None
        try:
            # A healthy communicator carries local failure to every peer
            # in the existing MIN. This cannot repair a poisoned backend.
            values[1] = available
            # Extrema can come from different DP ranks. Agree that every rank
            # fits its own demand while retaining extrema for diagnostics.
            values[2] = available >= local_required
            _impl.dist.all_reduce(values[1:], op=_impl.dist.ReduceOp.MIN, group=group)
            available = int(values[1].item())
            fits = bool(values[2].item())
        except BaseException as exc:
            if error is None:
                raise
            exchange_error = exc
        if error is not None:
            raise self._memory_error_with_reduction_note(error, exchange_error)
        if available < 0:
            raise RuntimeError("Memory admission failed on another rank")
    else:
        available = (
            self._available_memory_bytes()
            if details is None
            else self._available_memory_bytes(details)
        )
        local_available = available
        fits = required <= available
    sample = None
    if decision is not None:
        try:
            sample = decision.observe(
                scope=scope,
                local_required_bytes=local_required,
                local_available_bytes=local_available,
                reduced_required_bytes=required,
                reduced_available_bytes=available,
                safety_factor=_impl._MEMORY_SAFETY_FACTOR,
                reserve_fraction=_impl._MEMORY_RESERVE_FRACTION,
                **(details or {}),
            )
        except Exception:
            _impl._planner_misses._warn("could not retain admission sample")
    return _impl._MemoryCheck(
        estimated_required_bytes=required,
        available_bytes=available,
        fits=fits,
        local_required_bytes=local_required,
        local_available_bytes=local_available,
        sample=sample,
    )


def _forward_memory_group() -> dist.ProcessGroup | None:
    try:
        from megatron.core import parallel_state as ps

        return ps.get_tensor_and_context_parallel_group(check_initialized=False)
    except (AssertionError, ImportError, RuntimeError, ValueError):
        return None


def _estimate_required_memory_bytes_from_values(
    self: TrainerRank,
    *,
    packed_tokens: int,
    output_bytes: int,
    signature: _impl._MemorySignature,
    logical_tokens: int | None = None,
    gdn_segments: int = 0,
    group_rows: tuple[tuple[int, bool], ...] = (),
    slot_refs: tuple["LoRASlotRef | None", ...] | None = None,
    head_workspace_bytes: int = 0,
    checkpoint_floor: tuple[int, int] = (0, 0),
    retained_tokens: int | None = None,
    include_checkpoint_input_gradient: bool = True,
    checkpoint_memory: tuple[int, int] | None = None,
) -> int:
    if packed_tokens <= 0:
        return output_bytes
    profiled = self._memory_profiles.get(signature)
    activation_factor = max(4, min(16, self._num_layers // 4 + 4))
    static_compute = (
        packed_tokens * self._hidden_size * self._param_dtype_size * activation_factor
    )
    no_grad_stage = 0
    # A covered dense CP2 rank's no-grad width (_dense_mlp_widths).
    _dense_stage, covered = (
        self._dense_mlp_widths(slot_refs)
        if not signature.grad_enabled
        and group_rows
        and self._layout_pricing_supported(signature.topology)
        else (0, 0)
    )
    if covered or (
        not self._geometry.moe_experts
        and self._geometry.ffn_hidden_size
        and self._dense_fc1_adapted
        and signature.topology[1:3] == (1, 1)
    ):
        # A dense no-grad layer peaks at its FC1 stage beside the residual
        # pair, embedding and norm output (_dense_no_grad_row_elements). At
        # CP1 every packed row is local, which the per-packed-token floor
        # above prices at only 16H. A covered CP2 rank holds only its share,
        # which that floor over-prices (Qwen3.8-27B: +5.7% for one group), so
        # there the stage, at its CP2 width, replaces it. Groups run one
        # after another, so the largest no-grad group bounds it.
        rows = max(
            (n for n, grad in group_rows if not grad),
            default=0 if signature.grad_enabled else packed_tokens,
        )
        no_grad_stage = rows * (
            covered
            or self._param_dtype_size
            * _impl._dense_no_grad_row_elements(
                self._geometry.ffn_hidden_size,
                self._hidden_size,
                self._mlp_activation_factor,
            )
        )
        if covered:
            static_compute = no_grad_stage
    if signature.grad_enabled and self._recompute_granularity != "full":
        geometry = self._geometry
        hidden = self._hidden_size
        tp = max(1, self._topology_key()[1])
        sp = tp if self._sequence_parallel else 1
        # Gathered LoRA inputs alias norm output without sequence sharding.
        gathered = hidden if sp > 1 else 0
        common = 2 * hidden / sp + gathered
        attention_width = geometry.num_attention_heads * geometry.kv_channels or hidden
        kv_width = geometry.num_query_groups * geometry.kv_channels or hidden
        gated = self._attention_output_gate
        attention = common + ((7 if gated else 5) * attention_width + 3 * kv_width) / tp
        if 0 < geometry.num_query_groups < tp:
            # SelfAttentionLinearQKVLoRA constructs global QKV before
            # slicing it when KV groups cannot be partitioned across TP.
            attention += ((2 if gated else 1) * attention_width + 2 * kv_width) * (
                1 - 1 / tp
            )
        gdn = (
            common
            + (
                4 * geometry.gdn_key_heads * geometry.gdn_key_head_dim
                + 8 * geometry.gdn_value_heads * geometry.gdn_value_head_dim
            )
            / tp
        )
        ffn_width = geometry.ffn_hidden_size or 4 * hidden
        mlp = common + self._mlp_activation_factor * ffn_width / tp
        if geometry.moe_experts:
            # Keep the worst-case dispatch envelope: random-weight runs
            # cannot establish an EP discount for pretrained routing.
            ffn_width = (
                geometry.moe_topk * geometry.moe_ffn_hidden_size
                + geometry.moe_shared_expert_ffn
            ) or ffn_width
            mlp = common + 6 * ffn_width + 2 * hidden * max(0, geometry.moe_topk - 1)
        gdn_layers = min(self._num_layers, self._gdn_layers)
        retained_features = (
            (self._num_layers - gdn_layers) * attention
            + gdn_layers * gdn
            + self._num_layers * mlp
        )
        if self._recompute_granularity == "selective":
            checkpointed = (
                self._checkpointed_moe_layers
                if geometry.moe_experts
                else self._num_layers
                if "mlp" in self._recompute_modules
                else 0
            )
            # Checkpoints keep their input (and MoE's external norm); one
            # live MLP still needs workspace, including worst-case dispatch.
            checkpoint_input = (2 if geometry.moe_experts else 1) * hidden / sp
            retained_features -= max(0, checkpointed - 1) * (mlp - checkpoint_input)
        # Each GDN segment can retain an initial and a final recurrent
        # state (fp32), plus convolution history. Unlike token activations,
        # these do not shrink with segment length.
        gdn_state_bytes = gdn_segments * gdn_layers * self._gdn_segment_layer_bytes()
        static_compute = max(
            static_compute,
            # Cold eager runs allocate ~58 MiB beyond warm retention for
            # native kernel initialization; a slope alone misses short inputs.
            64 * 2**20
            + gdn_state_bytes
            + (
                packed_tokens
                if retained_tokens is None or geometry.moe_experts
                else retained_tokens
            )
            * self._param_dtype_size
            * retained_features,
        )
    # Groups execute sequentially: summed packed rows conservatively bound
    # this FC2 component, not all workspace or retained graphs.
    static_compute = max(
        static_compute,
        *(
            self._moe_workspace_bytes(
                sum(rows for rows, _ in group_rows)
                if signature.topology[2] > 1 and group_rows
                else packed_tokens,
                slot_ref=ref,
            )
            for ref in (slot_refs or (None,))
        ),
    )
    retained, workspace = (
        self._sequence_parallel_lora_floor(
            self._checkpoint_memory_floor(group_rows, slot_refs, gdn_segments),
            group_rows,
            signature,
        )
        if checkpoint_memory is None
        else checkpoint_memory
    )
    backward = 0
    if include_checkpoint_input_gradient and retained:
        # The backward's other end and cold transients, as _subforward_cost.
        backward = self._checkpoint_input_gradient_bytes(
            group_rows, retained
        ) + self._checkpoint_adapter_gradient_bytes(
            self._checkpoint_gradient_groups(group_rows, slot_refs)
        )
        if profiled is None and any(grad for _, grad in group_rows):
            backward += self._cold_recompute_transient_bytes()
    static_compute = max(
        static_compute,
        max(retained, checkpoint_floor[0])
        + max(workspace, head_workspace_bytes, checkpoint_floor[1])
        + backward,
        # A no-grad group's forward stage, beside any gradient groups'
        # boundaries but not the backward's input gradient.
        max(retained, checkpoint_floor[0]) + no_grad_stage,
    )
    if covered:
        # TE's cuBLAS workspace persists beside whichever stage peaks.
        static_compute += self._te_workspace_growth_bytes()
    if signature.topology[2] > 1:
        # Local head results coexist with full CP outputs during gathering.
        # Uneven rank plans can assign all of an item's rows to one rank.
        static_compute += output_bytes
    # Memory that grows with logical rows makes a profile learned under
    # lighter sharing underestimate a deeper-shared plan; scale the trusted
    # estimate up by the ratio gap. Packed pricing instead charges head and
    # caller memory per logical row and extrapolates the rest at most the
    # usual trust growth. Normalize before multiplying: cancelling packed
    # tokens through two float operations can otherwise make a larger warm
    # layout cheaper.
    packed_priced = profiled is not None and _impl._packed_priced(
        signature, self._one_layer_recompute()
    )

    def profiled_tokens(logical_per_packed: float) -> int | float:
        if logical_tokens is None:
            return packed_tokens
        return max(
            packed_tokens,
            logical_tokens
            / logical_per_packed
            / (_impl._MEMORY_PROFILE_TRUST_GROWTH if packed_priced else 1),
        )

    # The trust window limits calibration growth, not the empirical floor.
    # Dropping that floor beyond the window can admit a larger request that
    # was refused just inside it, even below a previously observed peak.
    if profiled is None:
        compute = static_compute
    else:
        profiled_bytes = profiled.bytes_per_token * profiled_tokens(
            profiled.logical_per_packed
        )
        if (
            profiled.warm_bytes_per_token is not None
            and profiled.warm_packed_tokens is not None
            and profiled.warm_logical_per_packed is not None
        ):
            # Later plans' own sharing: a rate learned under lighter
            # sharing scales up for deeper-shared plans, as above. Pricing
            # smaller waves as the smallest later plan keeps cost monotone
            # in tokens, which the width search's bounds rely on; it is
            # sound where a wave's memory is a fixed cost plus a per-token
            # rate, as that plan's rate covers its share of the fixed cost.
            profiled_bytes = min(
                profiled_bytes,
                profiled.warm_bytes_per_token
                * max(
                    profiled.warm_packed_tokens,
                    profiled_tokens(profiled.warm_logical_per_packed),
                ),
            )
        compute = max(
            static_compute,
            int(
                profiled_bytes
                + (
                    _impl._PACKED_PRICED_LOGICAL_ROW_BYTES * logical_tokens
                    # Branch states grow with segments, not packed rows;
                    # backward recomputes one layer at a time.
                    + gdn_segments * self._gdn_segment_layer_bytes()
                    if packed_priced and logical_tokens is not None
                    else 0
                )
            ),
        )
    return int((output_bytes + compute) * _impl._MEMORY_SAFETY_FACTOR)


def _gdn_segment_layer_bytes(self: TrainerRank) -> float:
    """Initial and final fp32 recurrent states plus convolution history."""
    geometry = self._geometry
    return (
        2
        / max(1, self._topology_key()[1])
        * (
            4
            * geometry.gdn_value_heads
            * geometry.gdn_key_head_dim
            * geometry.gdn_value_head_dim
            + self._param_dtype_size
            * (
                2 * geometry.gdn_key_heads * geometry.gdn_key_head_dim
                + geometry.gdn_value_heads * geometry.gdn_value_head_dim
            )
            * max(0, geometry.gdn_conv_kernel - 1)
        )
    )


def _available_memory_bytes(
    self: TrainerRank, sample: dict[str, Any] | None = None
) -> int:
    if not (_impl.torch.cuda.is_available() and self.device.type == "cuda"):
        return 1 << 60
    free, total = _impl.torch.cuda.mem_get_info(self.device)
    allocator = _impl.torch.cuda.get_allocator_backend()
    stats = None
    if allocator == "native":
        # Cached bytes are not physical free memory. This sample does not
        # reserve memory for execution or the caller's later backward.
        if sample is None:
            allocated = int(_impl.torch.cuda.memory_allocated(self.device))
        else:
            # memory_allocated uses this same stats read. Retain its other
            # already-returned fields without another allocator query.
            stats = _impl.torch.cuda.memory_stats(self.device)
            allocated = int(stats.get("allocated_bytes.all.current", 0))
        reusable_reserved = 0
    else:
        # Preserve the previous, unqualified policy for other backends.
        allocated = int(_impl.torch.cuda.memory_allocated(self.device))
        reserved = int(_impl.torch.cuda.memory_reserved(self.device))
        reusable_reserved = max(0, reserved - allocated)
    reserve = int(total * _impl._MEMORY_RESERVE_FRACTION)
    available = max(0, int(free) + reusable_reserved - reserve)
    test_cap = None
    if _impl.os.environ.get(_impl._TEST_HOOKS_ENV) == "1":
        limit = _impl.os.environ.get(_impl._TEST_MEMORY_LIMIT_ENV)
        if limit:
            test_cap = int(limit)
            # Test-only: cap the usable budget relative to the current
            # allocation so acceptance cells can induce split/decline
            # behavior deterministically without ballast tensors.
            available = min(available, max(0, int(limit) - allocated))
    if sample is not None:
        try:
            sample.update(
                allocator=allocator,
                physical_free_bytes=int(free),
                physical_total_bytes=int(total),
                allocated_bytes=allocated,
                reserve_bytes=reserve,
                test_cap_bytes=test_cap,
                reserved_bytes=(
                    stats.get("reserved_bytes.all.current")
                    if stats is not None
                    else reserved
                    if allocator != "native"
                    else None
                ),
                inactive_split_bytes=(
                    stats.get("inactive_split_bytes.all.current")
                    if stats is not None
                    else None
                ),
            )
        except Exception:
            sample.clear()
    return available


def _all_ranks_have_memory_profile(
    self: TrainerRank,
    *,
    packed_tokens: int,
    signature: _impl._MemorySignature,
) -> bool:
    profile = self._memory_profiles.get(signature)
    local = packed_tokens <= 0 or (
        profile is not None
        and profile.packed_tokens * _impl._MEMORY_PROFILE_TRUST_GROWTH >= packed_tokens
    )
    return self._all_ranks_true(local)


def _update_memory_profile(
    self: TrainerRank,
    plan: _impl._FlatForwardPlan,
    peak_delta_bytes: int,
    *,
    retained_bytes: int | None,
    caller_phase: bool = False,
) -> None:
    if plan.packed_tokens <= 0:
        return
    compute_delta = max(0, peak_delta_bytes - plan.output_bytes)
    bytes_per_token = compute_delta / max(1, plan.packed_tokens)
    previous = self._memory_profiles.get(plan.signature)
    logical_per_packed = plan.active_logical_tokens / max(1, plan.packed_tokens)
    warm_rate = None if previous is None else previous.warm_bytes_per_token
    warm_tokens = None if previous is None else previous.warm_packed_tokens
    warm_sharing = None if previous is None else previous.warm_logical_per_packed
    caller_plans = 0 if previous is None else previous.caller_plans
    # Only a flat wave's whole caller phase covers the caller's backward;
    # split children and dp_rank_forward observe forward only. A caller that
    # defers backward past the yield is not covered. The profile's first
    # such plan also pays one-time costs, so it is not warm.
    if caller_phase:
        if caller_plans:
            warm_rate = max(bytes_per_token, warm_rate or 0.0)
            warm_tokens = min(plan.packed_tokens, warm_tokens or plan.packed_tokens)
            warm_sharing = max(logical_per_packed, warm_sharing or 1.0)
        caller_plans += 1
    elif caller_plans:
        # After the seed, other readings cannot fit the warm profile, but a
        # higher one still raises its rate (inert until a whole warm plan
        # fits the rest), as it raises the fit over every observation.
        warm_rate = max(warm_rate or 0.0, bytes_per_token)
    retained_fraction = None if previous is None else previous.retained_fraction
    retained_compute = (
        None if previous is None else previous.retained_compute_bytes_per_token
    )
    if retained_bytes is not None:
        observed = min(1.0, retained_bytes / max(1, peak_delta_bytes))
        # Max-merge once observed. ``None`` (never observed) is distinct
        # from an observed 1.0, so a later, lower observation cannot
        # replace it.
        retained_fraction = (
            observed if retained_fraction is None else max(retained_fraction, observed)
        )
        observed_compute = max(
            0, min(retained_bytes, peak_delta_bytes) - plan.output_bytes
        ) / max(1, plan.packed_tokens)
        retained_compute = max(retained_compute or 0.0, observed_compute)
    self._memory_profiles[plan.signature] = _impl._MemoryProfile(
        bytes_per_token=max(
            bytes_per_token,
            0.0 if previous is None else previous.bytes_per_token,
        ),
        packed_tokens=max(
            plan.packed_tokens,
            0 if previous is None else previous.packed_tokens,
        ),
        logical_per_packed=max(
            logical_per_packed,
            1.0 if previous is None else previous.logical_per_packed,
        ),
        retained_fraction=retained_fraction,
        retained_compute_bytes_per_token=retained_compute,
        warm_bytes_per_token=warm_rate,
        warm_packed_tokens=warm_tokens,
        warm_logical_per_packed=warm_sharing,
        caller_plans=caller_plans,
    )
