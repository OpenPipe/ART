from __future__ import annotations

from typing import Any, cast

import torch

# Bound transient FP32 dot-product workspace (~16 MiB at hidden_size=2048) even
# at 131k-262k context lengths while remaining strictly row-independent.
_SHARED_GATE_CHUNK_ROWS = 2048


class _RowInvariantSharedExpertGateFn(torch.autograd.Function):
    """Row-count-invariant shared-expert sigmoid gate.

    Megatron's ``SharedExpertMLP`` evaluates ``F.linear(hidden_states,
    self.gate_weight)`` with ``gate_weight`` shaped ``[1, hidden_size]``.
    On Hopper/Blackwell GPUs, cuBLAS BF16 ``N=1`` GEMV selects different
    reduction kernels depending on row count ``M`` (for example aligned
    ``M % 16 == 0`` vs unaligned ``M``, or ``M <= 4096`` vs ``M > 4096``),
    injecting up to ~7.8e-3 row-count-dependent drift between single-sequence
    and radix-packed CP batches. Computing each row's dot product via
    chunked FP32 elementwise multiplication and ``sum(dim=-1)`` makes both
    forward outputs and input gradients bit-exact invariant to ``M`` while
    saving zero extra FP32 activations.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_size = int(gate_weight.shape[-1])
        x_2d = hidden_states.reshape(-1, hidden_size)
        rows = int(x_2d.shape[0])
        w_fp32 = gate_weight.detach().float().reshape(1, hidden_size)
        if rows <= _SHARED_GATE_CHUNK_ROWS:
            logits = (x_2d.float() * w_fp32).sum(dim=-1, keepdim=True)
            gate_score_2d = torch.sigmoid(logits.to(dtype=hidden_states.dtype))
        else:
            gate_score_2d = torch.empty(
                (rows, 1),
                device=x_2d.device,
                dtype=hidden_states.dtype,
            )
            for start in range(0, rows, _SHARED_GATE_CHUNK_ROWS):
                end = min(start + _SHARED_GATE_CHUNK_ROWS, rows)
                logits_chunk = (x_2d[start:end].float() * w_fp32).sum(
                    dim=-1, keepdim=True
                )
                gate_score_2d[start:end] = torch.sigmoid(
                    logits_chunk.to(dtype=hidden_states.dtype)
                )
        gate_score = gate_score_2d.reshape(*orig_shape[:-1], 1)
        if any(ctx.needs_input_grad):
            ctx.save_for_backward(hidden_states, gate_weight, gate_score)
        return gate_score

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        (grad_gate_score,) = grad_outputs
        hidden_states, gate_weight, gate_score = ctx.saved_tensors
        orig_shape = hidden_states.shape
        hidden_size = int(gate_weight.shape[-1])
        x_2d = hidden_states.reshape(-1, hidden_size)
        rows = int(x_2d.shape[0])
        g_2d = gate_score.reshape(rows, 1).float()
        d_logits = grad_gate_score.reshape(rows, 1).float() * g_2d * (1.0 - g_2d)

        dx: torch.Tensor | None = None
        if ctx.needs_input_grad[0]:
            w_fp32 = gate_weight.detach().float().reshape(1, hidden_size)
            if rows <= _SHARED_GATE_CHUNK_ROWS:
                dx = (
                    (d_logits * w_fp32)
                    .to(dtype=hidden_states.dtype)
                    .reshape(orig_shape)
                )
            else:
                dx_2d = torch.empty_like(x_2d)
                for start in range(0, rows, _SHARED_GATE_CHUNK_ROWS):
                    end = min(start + _SHARED_GATE_CHUNK_ROWS, rows)
                    dx_2d[start:end] = (d_logits[start:end] * w_fp32).to(
                        dtype=hidden_states.dtype
                    )
                dx = dx_2d.reshape(orig_shape)

        dw: torch.Tensor | None = None
        if ctx.needs_input_grad[1]:
            dw_fp32 = torch.zeros(
                (1, hidden_size),
                device=x_2d.device,
                dtype=torch.float32,
            )
            for start in range(0, rows, _SHARED_GATE_CHUNK_ROWS):
                end = min(start + _SHARED_GATE_CHUNK_ROWS, rows)
                dw_fp32 += torch.matmul(
                    d_logits[start:end].transpose(0, 1),
                    x_2d[start:end].float(),
                )
            dw = dw_fp32.to(dtype=gate_weight.dtype).reshape(gate_weight.shape)

        return dx, dw


@torch.compiler.disable
def row_invariant_shared_expert_gate(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute ``sigmoid(hidden_states @ gate_weight.T)`` invariantly to row count."""
    if int(hidden_states.numel()) == 0:
        zero = (
            hidden_states.sum() * 0.0
            + gate_weight.to(dtype=hidden_states.dtype).sum() * 0.0
        )
        return zero.expand((*hidden_states.shape[:-1], 1)).clone()
    return cast(
        torch.Tensor,
        _RowInvariantSharedExpertGateFn.apply(hidden_states, gate_weight),
    )


def _is_patched(fn: Any) -> bool:
    if getattr(fn, "__art_shared_expert_gate__", False):
        return True
    orig = getattr(fn, "_torchdynamo_orig_callable", None)
    return bool(orig is not None and getattr(orig, "__art_shared_expert_gate__", False))


def install_shared_expert_gate_patch() -> None:
    """Patch Megatron's ``SharedExpertMLP`` gate to be row-count invariant."""
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    original_forward = SharedExpertMLP.forward
    if not _is_patched(original_forward):
        was_compile_disabled = bool(
            getattr(original_forward, "__art_compile_disabled__", False)
        )
        inner_forward = getattr(
            original_forward, "_torchdynamo_orig_callable", original_forward
        )

        def _row_invariant_shared_expert_forward(
            self: Any,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            gate_weight = getattr(self, "gate_weight", None)
            if not getattr(self, "use_shared_expert_gate", False) or not isinstance(
                gate_weight, torch.Tensor
            ):
                return inner_forward(self, hidden_states)
            output, _ = super(SharedExpertMLP, self).forward(hidden_states)
            gate_score = row_invariant_shared_expert_gate(hidden_states, gate_weight)
            return output * gate_score

        setattr(
            _row_invariant_shared_expert_forward, "__art_shared_expert_gate__", True
        )
        patched_forward: Any = _row_invariant_shared_expert_forward
        if was_compile_disabled:
            patched_forward = torch.compiler.disable(
                _row_invariant_shared_expert_forward
            )
            setattr(patched_forward, "__art_compile_disabled__", True)
            setattr(patched_forward, "__art_shared_expert_gate__", True)
        setattr(SharedExpertMLP, "forward", patched_forward)

    original_pre_forward_comm = getattr(SharedExpertMLP, "pre_forward_comm", None)
    if original_pre_forward_comm is not None and not _is_patched(
        original_pre_forward_comm
    ):
        was_comm_disabled = bool(
            getattr(original_pre_forward_comm, "__art_compile_disabled__", False)
        )
        inner_pre_forward_comm = getattr(
            original_pre_forward_comm,
            "_torchdynamo_orig_callable",
            original_pre_forward_comm,
        )

        def _row_invariant_pre_forward_comm(
            self: Any,
            input: torch.Tensor,
        ) -> Any:
            result = inner_pre_forward_comm(self, input)
            gate_weight = getattr(self, "gate_weight", None)
            if getattr(self, "use_shared_expert_gate", False) and isinstance(
                gate_weight, torch.Tensor
            ):
                stream = getattr(self, "stream", None)
                if stream is not None and input.is_cuda:
                    with torch.cuda.stream(stream):
                        self.gate_score = row_invariant_shared_expert_gate(
                            input, gate_weight
                        )
                else:
                    self.gate_score = row_invariant_shared_expert_gate(
                        input, gate_weight
                    )
            return result

        setattr(_row_invariant_pre_forward_comm, "__art_shared_expert_gate__", True)
        patched_comm: Any = _row_invariant_pre_forward_comm
        if was_comm_disabled:
            patched_comm = torch.compiler.disable(_row_invariant_pre_forward_comm)
            setattr(patched_comm, "__art_compile_disabled__", True)
            setattr(patched_comm, "__art_shared_expert_gate__", True)
        setattr(SharedExpertMLP, "pre_forward_comm", patched_comm)
