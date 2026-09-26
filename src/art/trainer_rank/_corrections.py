"""Numerical helpers for explicitly requested selected-token correction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Iterator

import torch

from ._options import ImportanceSamplingGradientCorrection, ResolvedForwardOptions


def importance_weights(
    original_logprobs: torch.Tensor,
    current_logprobs: torch.Tensor,
    correction: ImportanceSamplingGradientCorrection,
) -> torch.Tensor:
    """Detached clipped p_current/p_original, without low-precision overflow.

    The caller must supply probabilities for identical events (token IDs and
    contexts). Original probabilities must be positive; current zero probability
    is allowed. Missing support and undefined 0/0 ratios raise rather than being
    silently replaced. Computation uses at least float32 and promotes to float64
    for float64 inputs or clipping bounds outside the float32 normal range.
    """
    if original_logprobs.shape != current_logprobs.shape:
        raise ValueError("correction logprob shapes must match exactly")
    if original_logprobs.device != current_logprobs.device:
        raise ValueError("correction logprobs must be on the same device")
    if (
        not original_logprobs.is_floating_point()
        or not current_logprobs.is_floating_point()
    ):
        raise TypeError("correction logprobs must be floating-point tensors")
    if not bool(torch.isfinite(original_logprobs).all()):
        raise ValueError("original logprobs must be finite (positive sampling support)")
    if bool((torch.isnan(current_logprobs) | torch.isposinf(current_logprobs)).any()):
        raise ValueError("current logprobs must be finite or negative infinity")
    dtype = (
        torch.float64
        if torch.float64 in (original_logprobs.dtype, current_logprobs.dtype)
        or correction.clip_high > torch.finfo(torch.float32).max
        or any(
            0 < bound < torch.finfo(torch.float32).tiny
            for bound in (correction.clip_low, correction.clip_high)
        )
        else torch.float32
    )
    log_ratio = current_logprobs.detach().to(dtype) - original_logprobs.detach().to(
        dtype
    )
    if correction.clip_high == 0:
        return torch.zeros_like(log_ratio)
    log_low = math.log(correction.clip_low) if correction.clip_low else -math.inf
    return (
        log_ratio.clamp(log_low, math.log(correction.clip_high))
        .exp()
        .clamp(correction.clip_low, correction.clip_high)
    )


def correct_logprob_cotangent(
    cotangent: torch.Tensor,
    *,
    original_logprobs: torch.Tensor,
    current_logprobs: torch.Tensor | None,
    correction: ImportanceSamplingGradientCorrection,
    original_tokens: torch.Tensor | None = None,
    current_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply explicitly requested importance weights to aligned logprob outputs.

    For top-k, both token tensors are required by the caller's output contract.
    The IDs must match elementwise: independently recomputing top-k can change
    their identity/order. Values are full-vocabulary logprobs, not probabilities
    renormalized over top-k. No forward is performed here.
    """
    if cotangent.shape != original_logprobs.shape:
        raise ValueError("cotangent and correction logprob shapes must match exactly")
    active = cotangent != 0
    if not bool(active.any()):
        return cotangent
    if current_logprobs is None:
        if correction.policy == "always":
            raise RuntimeError(
                "importance sampling correction requires current logprobs"
            )
        return cotangent
    if (original_tokens is None) != (current_tokens is None):
        raise ValueError("correction requires both original and current token IDs")
    if original_tokens is not None and current_tokens is not None:
        if (
            original_tokens.shape != original_logprobs.shape
            or current_tokens.shape != current_logprobs.shape
        ):
            raise ValueError("correction token IDs must match logprob shapes")
        if not torch.equal(original_tokens, current_tokens):
            raise ValueError(
                "correction must compare the same token IDs in the same order"
            )
    if current_logprobs.shape != original_logprobs.shape:
        raise ValueError("correction logprob shapes must match exactly")
    if current_logprobs.device != original_logprobs.device:
        raise ValueError("correction logprobs must be on the same device")
    selected = active.to(original_logprobs.device)
    weights = importance_weights(
        original_logprobs[selected], current_logprobs[selected], correction
    )
    corrected = cotangent.clone()
    corrected[active] = (cotangent[active] * weights.to(cotangent.device)).to(
        cotangent.dtype
    )
    return corrected


@dataclass(frozen=True)
class _OutputCorrection:
    index: int
    kind: str
    original_logprobs: torch.Tensor
    token_index: int | None = None
    original_tokens: torch.Tensor | None = None
    logits_index: int | None = None


@dataclass(frozen=True)
class ForwardCorrectionContext:
    """Correction metadata and owned CPU copies, independent of physical graphs.

    Current tensors must come from the original inputs/contexts, with the same
    flattened output layout. Correct only stale forwards; exact original-weight
    replay itself does not produce current probabilities. Validate physical
    current-weight replay even when corrections are disabled.
    """

    output_count: int
    correction: ImportanceSamplingGradientCorrection | None
    outputs: tuple[_OutputCorrection, ...]

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor
            for output in self.outputs
            for tensor in (output.original_logprobs, output.original_tokens)
            if tensor is not None
        )

    def requires_current(self, gradients: Sequence[torch.Tensor | None]) -> bool:
        """Whether active eligible outputs need current data before mutation."""
        if len(gradients) != self.output_count:
            raise ValueError(
                "correction cotangents must match the captured output count"
            )
        for output in self.outputs:
            gradient = gradients[output.index]
            if (
                gradient is not None
                and gradient.shape != output.original_logprobs.shape
            ):
                raise ValueError(
                    "correction cotangent shape must match the original output"
                )
        return (
            self.correction is not None
            and self.correction.policy == "always"
            and any(
                (gradient := gradients[output.index]) is not None
                and bool((gradient != 0).any())
                for output in self.outputs
            )
        )

    def correct(
        self,
        gradients: Sequence[torch.Tensor | None],
        current_tensors: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Stage corrected cotangents without mutating gradients or model state."""
        self.requires_current(gradients)
        if current_tensors is not None and len(current_tensors) != self.output_count:
            raise ValueError("current tensors must match the captured output count")
        corrected = list(gradients)
        if self.correction is None:
            return tuple(corrected)
        for output in self.outputs:
            gradient = gradients[output.index]
            original = output.original_logprobs
            if gradient is None or not bool((gradient != 0).any()):
                continue
            current = (
                None
                if current_tensors is None
                else current_tensors[output.index].detach()
            )
            if current is not None and output.original_tokens is not None:
                assert current_tensors is not None and output.token_index is not None
                tokens = current_tensors[output.token_index]
                original_tokens = output.original_tokens.to(tokens.device)
                if tokens.shape != original_tokens.shape:
                    raise ValueError(
                        "current top-k token shape must match original top-k"
                    )
                if not torch.equal(tokens, original_tokens):
                    # A changed top-k ordering can still contain every old ID.
                    sorted_tokens, order = tokens.sort(dim=-1)
                    positions = torch.searchsorted(
                        sorted_tokens.contiguous(), original_tokens.contiguous()
                    ).clamp_max(tokens.shape[-1] - 1)
                    matched = sorted_tokens.gather(-1, positions) == original_tokens
                    if bool((matched | (gradient == 0).to(matched.device)).all()):
                        current = current.gather(-1, order.gather(-1, positions))
                    elif output.logits_index is not None:
                        logits = current_tensors[output.logits_index].detach()
                        dtype = (
                            torch.float64
                            if logits.dtype == torch.float64
                            else torch.float32
                        )
                        logits = logits.to(dtype)
                        current = logits.gather(
                            -1, output.original_tokens.to(logits.device)
                        ) - logits.logsumexp(-1, keepdim=True)
                    else:
                        # The new top-k lacks original events: no ratio is available.
                        current = None
            corrected[output.index] = correct_logprob_cotangent(
                gradient,
                original_logprobs=original
                if current is None
                else original.to(current.device),
                current_logprobs=current,
                correction=self.correction,
            )
        return tuple(corrected)

    def validate_replay(
        self,
        gradients: Sequence[torch.Tensor | None],
        current_tensors: Sequence[torch.Tensor],
    ) -> None:
        """Require active top-k cotangents to address the same replayed events.

        Ratio evaluation on a separate current forward can realign top-k IDs.
        Physical current-weight replay cannot feed original-position cotangents
        to a Jacobian whose selected token at that position has changed.
        """
        self.requires_current(gradients)
        if len(current_tensors) != self.output_count:
            raise ValueError("current tensors must match the captured output count")
        for output in self.outputs:
            gradient = gradients[output.index]
            if output.original_tokens is None or gradient is None:
                continue
            active = gradient != 0
            if not bool(active.any()):
                continue
            assert output.token_index is not None
            tokens = current_tensors[output.token_index]
            original = output.original_tokens.to(tokens.device)
            if tokens.shape != original.shape or bool(
                ((tokens != original) & active.to(tokens.device)).any()
            ):
                raise RuntimeError(
                    "current replay changed active top-k token identities; "
                    "replay the original weights instead"
                )


def capture_forward_corrections(
    outputs: Any,
    tensors: Sequence[torch.Tensor],
    options: ResolvedForwardOptions,
) -> ForwardCorrectionContext:
    """Map a ForwardOutput tree to the caller's deduplicated flat tensor layout."""
    from ._impl import ForwardOutput

    def leaves(value: Any) -> Iterator[Any]:
        if isinstance(value, ForwardOutput):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from leaves(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from leaves(item)
        else:
            raise TypeError("correction capture requires a ForwardOutput tree")

    corrections = options.stale_gradient_corrections
    correction = corrections[0] if corrections else None
    indices = {id(tensor): index for index, tensor in enumerate(tensors)}
    if len(indices) != len(tensors):
        raise ValueError("correction capture requires deduplicated flat tensors")
    entries: dict[int, _OutputCorrection] = {}
    for output in leaves(outputs):
        for kind, tensor in (
            ("target_logprobs", output.target_logprobs),
            ("top_k", None if output.top_k is None else output.top_k.logprobs),
        ):
            if (
                tensor is None
                or not tensor.requires_grad
                or (correction is None and kind != "top_k")
            ):
                continue
            index = indices[id(tensor)]
            entry = _OutputCorrection(
                index=index,
                kind=kind,
                original_logprobs=tensor.detach().to("cpu", copy=True),
                token_index=indices[id(output.top_k.tokens)]
                if kind == "top_k"
                else None,
                original_tokens=output.top_k.tokens.detach().to("cpu", copy=True)
                if kind == "top_k"
                else None,
                logits_index=indices[id(output.logits)]
                if kind == "top_k" and output.logits is not None
                else None,
            )
            if index in entries and (
                entries[index].kind != kind
                or entries[index].token_index != entry.token_index
            ):
                raise ValueError(
                    "an aliased output tensor has ambiguous correction semantics"
                )
            entries.setdefault(index, entry)
    return ForwardCorrectionContext(len(tensors), correction, tuple(entries.values()))
