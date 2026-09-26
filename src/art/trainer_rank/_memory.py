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

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
import hashlib
import math
from types import MethodType
from typing import TYPE_CHECKING, Any

from art.trainer_rank import _impl

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

    from art.megatron.lora import LoRASlotRef
    from art.trainer_rank._impl import AdapterSelection, AnyForwardInput, TrainerRank


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
        checkpoint = (
            sum(
                cost.checkpoint_retained + cost.checkpoint_input_gradient
                for cost in costs
            )
            + max(cost.checkpoint_workspace for cost in costs)
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


def _head_workspace_bytes(self: TrainerRank, rows: int) -> int:
    """One dense BF16 head tensor, not complete statistics/backward memory."""
    if (
        rows <= 0
        or self._padded_vocab_size is None
        or len(self.runtime.model) != 1
        or self._topology_key()[1::2] != (1, 1)
    ):
        return 0
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
        or tuple(weight.shape) != (self._padded_vocab_size, self._hidden_size)
        or head.output_size_per_partition != self._padded_vocab_size
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
                "_local_head_stats",
                "_local_logits_from_hidden_rows",
            )
        )
    ):
        return 0
    return min(rows, _impl._HEAD_CHUNK_TOKENS) * int(self._padded_vocab_size) * 2


def _group_head_workspace_bytes(
    self: TrainerRank,
    rows: int,
    requests: Sequence[AnyForwardInput],
    *,
    grad_enabled: bool,
    positions: Sequence[torch.Tensor] | None = None,
    lower_bound: bool = False,
) -> int:
    """One logits buffer, or logits + both dense target-backward gradients.

    The supported head path overlaps indexing and statistics gradients
    with recomputed logits; cold library workspaces remain outside this
    component. Pair each group's mode with its own projected rows.
    """
    dense = self._head_workspace_bytes(rows)
    if (
        not dense
        or not grad_enabled
        or not any(request.target_tokens is not None for request in requests)
    ):
        return dense
    from megatron.core.models.common.language_module.language_module import (
        LanguageModule,
    )

    model = _impl._language_model(self.runtime.model[0])
    scale = getattr(model, "_scale_logits", None)
    if (
        type(scale) is MethodType
        and scale.__self__ is model
        and scale.__func__ is LanguageModule._scale_logits
        and getattr(model.config, "use_mup", None) is False
    ):
        # IndexBackward's dense result overlaps saved logits and grad_logits.
        # The FP32 fallback already exceeds this three-buffer component.
        target_dense = (
            self._head_workspace_bytes(
                self._head_target_chunk_rows(
                    requests, positions=positions, lower_bound=lower_bound
                )
            )
            if any(request.logits or request.top_k is not None for request in requests)
            else dense
        )
        return max(dense, 3 * target_dense)
    return dense


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
    provider: Any = getattr(getattr(self, "runtime", None), "provider", None)
    ep = int(getattr(provider, "expert_model_parallel_size", 1) or 1)
    if ep <= 1 or not plan.groups:
        return 0
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
    if _impl._hybridep_rows_per_rank(capacity, ranks) <= held:
        return 0
    # The old buffer stays referenced while its replacement is allocated.
    return _impl._hybridep_buffer_bytes(
        capacity,
        ranks,
        int(provider.hidden_size),
        int(provider.num_moe_experts) * etp,
    )


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


def _moe_workspace_bytes(
    self: TrainerRank,
    rows: int,
    *,
    checkpoint_grad: bool = False,
    slot_ref: "LoRASlotRef | None" = None,
) -> int:
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
    return (
        max(
            rows * coefficient,
            *(rows * per_row + fixed for per_row, fixed in stages),
        )
        if stages and rows > 0
        else rows * coefficient
    )


def _checkpoint_memory_floor(
    self: TrainerRank,
    group_rows: tuple[tuple[int, bool], ...],
    slot_refs: tuple["LoRASlotRef | None", ...] | None = None,
    gdn_segments: int = 0,
) -> tuple[int, int]:
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
    and there, for gradient waves, the recomputed GDN layer's recurrent
    states for ``gdn_segments`` (gradient groups' segments) plus padding.
    """
    gradient_rows = sum(rows for rows, grad in group_rows if grad)
    if not group_rows or len(self.runtime.model) != 1:
        return 0, 0
    try:
        decoder = _impl._language_model(self.runtime.model[0]).decoder
    except (AttributeError, RuntimeError):
        return 0, 0
    try:
        from megatron.core.transformer.transformer_block import TransformerBlock
    except ModuleNotFoundError as error:
        if error.name != "megatron":
            raise
        return 0, 0

    if type(decoder) is not TransformerBlock:
        return 0, 0
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
        or (tp > 1 and not self._sequence_parallel_floor_covered(layers, tp, cp))
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
        return 0, 0
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
    if tp > 1 and self._gdn_layers and gradient_rows:
        # Recurrent states grow with segments, not rows; backward recomputes
        # one layer at a time. Padding to TP adds up to TP - 1 one-token
        # roots per group. Kernel-internal chunk states are not bounded here.
        roots = gdn_segments + (tp - 1) * sum(grad for _, grad in group_rows)
        workspace += math.ceil(roots * self._gdn_segment_layer_bytes())
    if (
        gradient_rows
        and self._topology_key() == (1, 1, 2, 1)
        and self._parallel_shape == _impl.ParallelShape(tp=1, cp=2, ep=2, etp=1)
        and self._moe_memory_supported
    ):
        # Recompute runs after _execute_flat_plan restores the communication
        # high-water. Combine allocates a fresh BF16 [P, H] before cropping;
        # this is separate from already-held native buffer capacity. Do not
        # prune graph references or reset execution state while estimating.
        rows = max(rows for rows, _ in group_rows)
        if any(ref() is not None for ref in self._pending_hybridep_graphs):
            rows = max(rows, self._hybridep_rows_high_water)
        workspace = max(workspace, -(-rows // 4) * 4 * self._hidden_size * 2)
    return retained, workspace


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
                mixed_targets = (
                    grad_enabled
                    and any(
                        request.target_tokens is not None for request in head_requests
                    )
                    and any(
                        request.logits or request.top_k is not None
                        for request in head_requests
                    )
                )
                if lower != upper or (
                    mixed_targets
                    and self._head_target_chunk_rows(head_requests, lower_bound=True)
                    != self._head_target_chunk_rows(head_requests)
                ):
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
) -> None:
    if baseline is None:
        return
    peak = int(_impl.torch.cuda.max_memory_allocated(self.device))
    observation = getattr(self, "_planner_observation", None)
    if observation is not None:
        observation["peak"] = max(observation["peak"], peak)
    self._update_memory_profile(
        plan,
        max(0, peak - baseline),
        retained_bytes=(
            None if retained_after is None else max(0, retained_after - baseline)
        ),
    )


def _estimate_group_request_output_bytes(
    self: TrainerRank,
    requests: Sequence[AnyForwardInput],
) -> int:
    total = 0
    for request in requests:
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
    """Separate empirical trust across actual selected adapter layouts."""
    if (
        ref is None
        or ref.name is None
        or isinstance(ref, _impl._LocalLoRASlotRef)
        or not (getattr(self, "_moe_layers", 0) or getattr(self, "_gdn_layers", 0))
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
    # The existing admission operand can already be a cross-rank maximum.
    # Preserve its local producer separately; never price the plan again.
    with decision.refresh_of(check.sample) if decision is not None else nullcontext():
        return self._memory_check_required(
            check.estimated_required_bytes, sync_across_dp=sync_across_dp
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
            [float(required), 0.0],
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
            _impl.dist.all_reduce(values[1], op=_impl.dist.ReduceOp.MIN, group=group)
            available = int(values[1].item())
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
        fits=required <= available,
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
) -> int:
    if packed_tokens <= 0:
        return output_bytes
    profiled = self._memory_profiles.get(signature)
    activation_factor = max(4, min(16, self._num_layers // 4 + 4))
    static_compute = (
        packed_tokens * self._hidden_size * self._param_dtype_size * activation_factor
    )
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
    retained, workspace = self._checkpoint_memory_floor(
        group_rows, slot_refs, gdn_segments
    )
    static_compute = max(
        static_compute,
        max(retained, checkpoint_floor[0])
        + max(workspace, head_workspace_bytes, checkpoint_floor[1])
        + (retained if include_checkpoint_input_gradient else 0),
    )
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
    profiled_tokens: int | float = packed_tokens
    packed_priced = profiled is not None and _impl._packed_priced(
        signature, self._one_layer_recompute()
    )
    if profiled is not None and logical_tokens is not None:
        profiled_tokens = max(
            packed_tokens,
            logical_tokens
            / profiled.logical_per_packed
            / (_impl._MEMORY_PROFILE_TRUST_GROWTH if packed_priced else 1),
        )
    # The trust window limits calibration growth, not the empirical floor.
    # Dropping that floor beyond the window can admit a larger request that
    # was refused just inside it, even below a previously observed peak.
    if profiled is None:
        compute = static_compute
    else:
        compute = max(
            static_compute,
            int(
                profiled.bytes_per_token * profiled_tokens
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
) -> None:
    if plan.packed_tokens <= 0:
        return
    compute_delta = max(0, peak_delta_bytes - plan.output_bytes)
    bytes_per_token = compute_delta / max(1, plan.packed_tokens)
    previous = self._memory_profiles.get(plan.signature)
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
            plan.active_logical_tokens / max(1, plan.packed_tokens),
            1.0 if previous is None else previous.logical_per_packed,
        ),
        retained_fraction=retained_fraction,
        retained_compute_bytes_per_token=retained_compute,
    )
