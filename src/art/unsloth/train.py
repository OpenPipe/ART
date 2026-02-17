"""
Clean Unsloth training loop — no TRL, no monkey patches, no nest_asyncio.

Replaces:
  - art/unsloth/train.py (monkey-patched GRPOTrainer)
  - art/unsloth/training_utils.py (async queue bridging)
  - The _training_state cached_property in service.py (GRPOTrainer init + fake dataset)

Same compute, same loss, same logprob calculation. Just owns the loop.
"""

import gc
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, cast

import torch
from peft.peft_model import PeftModelForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from art import dev, types
from art.loss import loss_fn, shift_tensor
from art.preprocessing.inputs import TrainInputs, create_train_inputs
from art.preprocessing.pack import PackedTensors


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

CausalLM = Any


@dataclass
class TrainingState:
    """Everything needed for training — no GRPOTrainer."""

    model: CausalLM
    tokenizer: PreTrainedTokenizerBase
    peft_model: PeftModelForCausalLM
    optimizer: torch.optim.Optimizer
    device: torch.device
    _pinned_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    _is_offloaded: bool = False
    _warmed_up: bool = False

    def offload_to_cpu(self) -> None:
        if self._is_offloaded:
            return
        for name, param in self.peft_model.named_parameters():
            if param.device.type == "cuda":
                buf = self._get_pinned_buffer(name, param)
                buf.copy_(param.data, non_blocking=True)
                param.data = buf
        for param_id, state in self.optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cuda":
                    key = f"opt_{id(param_id)}_{k}"
                    buf = self._get_pinned_buffer(key, v)
                    buf.copy_(v, non_blocking=True)
                    state[k] = buf
        torch.cuda.synchronize()
        self._is_offloaded = True
        gc_and_empty_cuda_cache()

    def reload_to_gpu(self) -> None:
        if not self._is_offloaded:
            return
        for _name, param in self.peft_model.named_parameters():
            if param.device.type == "cpu":
                gpu = torch.empty(param.shape, dtype=param.dtype, device=self.device)
                gpu.copy_(param.data, non_blocking=True)
                param.data = gpu
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                    gpu = torch.empty(v.shape, dtype=v.dtype, device=self.device)
                    gpu.copy_(v, non_blocking=True)
                    state[k] = gpu
        torch.cuda.synchronize()
        self._is_offloaded = False

    def _get_pinned_buffer(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        if (
            key not in self._pinned_buffers
            or self._pinned_buffers[key].shape != tensor.shape
        ):
            self._pinned_buffers[key] = torch.empty(
                tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True
            )
        return self._pinned_buffers[key]


# ---------------------------------------------------------------------------
# Initialization — replaces _training_state cached_property
# ---------------------------------------------------------------------------


def init_training_state(
    base_model: str,
    config: dev.InternalModelConfig,
    device: torch.device,
    checkpoint_dir: str | None = None,
) -> TrainingState:
    """Load model + create optimizer. No GRPOTrainer, no fake dataset."""
    import unsloth

    init_args = config.get("init_args", {})
    init_args["model_name"] = checkpoint_dir or base_model

    model, tokenizer = cast(
        tuple[CausalLM, PreTrainedTokenizerBase],
        unsloth.FastLanguageModel.from_pretrained(**init_args),
    )

    if hasattr(model, "peft_config") and model.peft_config is not None:
        peft_model = cast(PeftModelForCausalLM, model)
    else:
        peft_model = cast(
            PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                model, **config.get("peft_args", {})
            ),
        )

    # Direct optimizer — no TRL wrapper
    trainer_args = config.get("trainer_args", {})
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=trainer_args.get("learning_rate", 5e-6),
        betas=(
            trainer_args.get("adam_beta1", 0.9),
            trainer_args.get("adam_beta2", 0.99),
        ),
        weight_decay=trainer_args.get("weight_decay", 0.01),
    )

    return TrainingState(
        model=model,
        tokenizer=tokenizer,
        peft_model=peft_model,
        optimizer=optimizer,
        device=device,
    )


# ---------------------------------------------------------------------------
# Training step — replaces monkey-patched trainer.train() + process_train_batch
# ---------------------------------------------------------------------------


async def train_step(
    state: TrainingState,
    packed_tensors: PackedTensors,
    config: types.TrainConfig,
    _config: dev.TrainConfig,
    verbose: bool = False,
) -> AsyncIterator[dict[str, float]]:
    """One training step over packed tensors. Yields metrics per batch."""

    num_sequences = packed_tensors["tokens"].shape[0]

    # Warmup pass (first time only) — small slice, throwaway lr
    if not state._warmed_up:
        warmup_inputs = create_train_inputs(packed_tensors, 0, config, _config, warmup=True)
        _forward_backward_step(state, warmup_inputs, is_warmup=True)
        state._warmed_up = True
        gc_and_empty_cuda_cache()

    # Actual training
    for offset in range(num_sequences):
        inputs = create_train_inputs(packed_tensors, offset, config, _config, warmup=False)
        metrics = _forward_backward_step(state, inputs, is_warmup=False)
        if verbose:
            print(f"  batch {offset+1}/{num_sequences} — loss={metrics.get('loss', 0):.4f}")
        yield metrics


def _forward_backward_step(
    state: TrainingState,
    inputs: TrainInputs,
    is_warmup: bool = False,
) -> dict[str, float]:
    """Forward → loss → backward → step. The whole thing. 20 lines."""

    config: types.TrainConfig = inputs.pop("config")  # type: ignore
    _config: dev.TrainConfig = inputs.pop("_config")  # type: ignore
    inputs.pop("return_new_logprobs", None)

    # Set learning rate
    for pg in state.optimizer.param_groups:
        pg["lr"] = 1e-9 if is_warmup else config.learning_rate

    # Handle pixel values
    if inputs.get("pixel_values") and inputs["pixel_values"][0] is not None:
        inputs["pixel_values"] = inputs["pixel_values"][0]  # type: ignore
    else:
        del inputs["pixel_values"]  # type: ignore
    if inputs.get("image_grid_thw") and inputs["image_grid_thw"][0] is not None:
        inputs["image_grid_thw"] = inputs["image_grid_thw"][0]  # type: ignore
    else:
        del inputs["image_grid_thw"]  # type: ignore

    # Move to device
    inputs = {
        k: v.to(state.device) for k, v in inputs.items()
    }  # type: ignore

    # Dtype for autocast
    accelerate_mp = os.environ.get("ACCELERATE_MIXED_PRECISION")
    force_f32 = os.environ.get("UNSLOTH_FORCE_FLOAT32")
    if accelerate_mp is None or accelerate_mp == "fp16" or force_f32 == "1":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.bfloat16

    batch_size, seq_len = inputs["tokens"].size()

    # Attention mask — same tree-structured mask
    attn_bias = _build_attn_bias(
        batch_size, seq_len, state.device,
        inputs["group_ids"], inputs["parent_ids"], cast_dtype,
    )

    # LM head for logprob calculation
    lm_head_t = cast(
        torch.Tensor,
        state.peft_model.get_output_embeddings().weight.t(),
    )
    next_input_ids = shift_tensor(inputs["tokens"], 0)
    chunk_size = _config.get("logprob_calculation_chunk_size", 1024)
    assert seq_len % chunk_size == 0

    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    forward_kwargs = {}
    if "pixel_values" in inputs:
        forward_kwargs["pixel_values"] = inputs["pixel_values"]
    if "image_grid_thw" in inputs:
        forward_kwargs["image_grid_thw"] = inputs["image_grid_thw"]

    # Pass 1: new logprobs (with gradients)
    new_logprobs, entropies = _calculate_logprobs(
        state.peft_model, inputs["tokens"], attn_bias, forward_kwargs,
        next_input_ids, lm_head_t, chunk_size, cast_dtype,
        inference_mode=False, disable_adapter=False,
    )

    # Pass 2: reference logprobs (only if beta > 0)
    ref_logprobs = None
    if config.beta > 0.0:
        ref_logprobs, _ = _calculate_logprobs(
            state.peft_model, inputs["tokens"], attn_bias, forward_kwargs,
            next_input_ids, lm_head_t, chunk_size, cast_dtype,
            inference_mode=True, disable_adapter=True,
        )

    del attn_bias

    # Loss
    loss = loss_fn(inputs, new_logprobs, ref_logprobs, entropies, _config)
    total_loss = loss.mean_policy_loss + config.beta * loss.mean_kl

    # Backward + step
    state.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in state.peft_model.parameters() if p.requires_grad],
        max_norm=1.0,
    )
    state.optimizer.step()

    # Metrics
    metrics: dict[str, float] = {
        "loss": total_loss.item(),
        "policy_loss": loss.mean_policy_loss.item(),
        "learning_rate": config.learning_rate,
    }
    if loss.mean_entropy is not None:
        metrics["entropy"] = loss.mean_entropy.item()
    if config.beta > 0.0:
        metrics["kl_div"] = loss.mean_kl.item()

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint — replaces trainer.save_model()
# ---------------------------------------------------------------------------


def save_checkpoint(state: TrainingState, output_dir: str) -> str:
    from art.utils.get_model_step import get_step_from_dir
    from art.utils.output_dirs import get_step_checkpoint_dir

    next_step = get_step_from_dir(output_dir) + 1
    checkpoint_dir = get_step_checkpoint_dir(output_dir, next_step)
    os.makedirs(checkpoint_dir, exist_ok=True)
    state.peft_model.save_pretrained(checkpoint_dir)
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Helpers — same compute, just not buried inside monkey patches
# ---------------------------------------------------------------------------


def _build_attn_bias(
    batch_size: int, seq_len: int, device: torch.device,
    group_ids: torch.Tensor, parent_ids: torch.Tensor, dtype: torch.dtype,
) -> torch.Tensor:
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    causal = causal.unsqueeze(0).expand(batch_size, seq_len, seq_len)
    group_mask = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    parent_mask = parent_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    mask = causal & (group_mask | parent_mask)
    return torch.where(
        mask,
        torch.tensor(0.0, dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )


def _calculate_logprobs(
    model: PeftModelForCausalLM,
    input_ids: torch.Tensor,
    causal_mask: torch.Tensor,
    forward_kwargs: dict[str, torch.Tensor],
    next_input_ids: torch.Tensor,
    lm_head_t: torch.Tensor,
    chunk_size: int,
    cast_dtype: torch.dtype,
    inference_mode: bool,
    disable_adapter: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass → chunked logprob calculation."""
    with (
        torch.inference_mode() if inference_mode else nullcontext(),
        model.disable_adapter() if disable_adapter else nullcontext(),
        torch.amp.autocast_mode.autocast(device_type="cuda", dtype=cast_dtype),
    ):
        hidden_states = model(
            input_ids=input_ids, causal_mask=causal_mask, **forward_kwargs
        ).logits  # [B, S, H]

    lm_head_t = lm_head_t.to(hidden_states.dtype)
    batch_size, seq_len, _ = hidden_states.shape

    log_probs = torch.empty(batch_size, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
    entropy = torch.empty_like(log_probs)

    for i in range(0, seq_len, chunk_size):
        chunk_hs = hidden_states[:, i:i + chunk_size, :]
        chunk_ids = next_input_ids[:, i:i + chunk_size]
        chunk_logits = chunk_hs @ lm_head_t

        chunk_selected = torch.gather(chunk_logits, -1, chunk_ids.unsqueeze(-1)).squeeze(-1)
        chunk_lse = torch.logsumexp(chunk_logits, dim=-1)
        log_probs[:, i:i + chunk_size] = chunk_selected - chunk_lse

        log_p_full = chunk_logits - chunk_lse.unsqueeze(-1)
        entropy[:, i:i + chunk_size] = (-torch.exp(log_p_full) * log_p_full).sum(dim=-1)

        del chunk_hs, chunk_ids, chunk_logits, chunk_selected, chunk_lse, log_p_full

    del hidden_states
    return log_probs, entropy


def gc_and_empty_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
