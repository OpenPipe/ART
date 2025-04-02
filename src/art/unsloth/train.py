import asyncio
from contextlib import nullcontext
import gc
import nest_asyncio
import os
import torch
from typing import TYPE_CHECKING

from ..types import TuneConfig

if TYPE_CHECKING:
    from peft.peft_model import PeftModel
    from trl import GRPOTrainer
    from .service import TuneInputs

nest_asyncio.apply()


async def train(
    trainer: "GRPOTrainer", inputs_queue: asyncio.Queue["TuneInputs"]
) -> None:

    def compute_loss(
        model: "PeftModel",
        inputs: "TuneInputs",
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        try:
            config: TuneConfig = inputs.pop("config")  # type: ignore

            if optimizer := trainer.optimizer:
                optimizer = getattr(optimizer, "optimizer", optimizer)
                if param_groups := getattr(optimizer, "param_groups"):
                    for param_group in param_groups:
                        param_group["lr"] = config.lr
                        param_group["betas"] = config.betas
                        if param_group.get("weight_decay"):
                            param_group["weight_decay"] = config.weight_decay

            # Move tensors to the correct device
            inputs = {key: tensor.to(trainer.accelerator.device) for key, tensor in inputs.items()}  # type: ignore

            # Unsloth code
            if not hasattr(trainer, "_autocast_dtype"):
                dtype = (
                    torch.float16
                    if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                    else torch.bfloat16
                )
                if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
                    dtype = torch.float16
                setattr(trainer, "_autocast_dtype", dtype)

            # Create grouped causal mask
            batch_size, seq_len = inputs["tokens"].size()
            causal_mask = (
                torch.tril(
                    torch.ones(
                        seq_len,
                        seq_len,
                        dtype=torch.bool,
                        device=trainer.accelerator.device,
                    )
                )
                .unsqueeze(0)
                .expand(batch_size, seq_len, seq_len)
            )
            group_mask = inputs["group_ids"].unsqueeze(2) == inputs[
                "group_ids"
            ].unsqueeze(1)
            parent_mask = inputs["parent_ids"].unsqueeze(2) == inputs[
                "group_ids"
            ].unsqueeze(1)
            mask = causal_mask & (group_mask | parent_mask)
            # Use the same dtype as autocast to save memory and avoid dtype conversions
            attn_bias = torch.where(
                mask,
                torch.tensor(
                    0.0,
                    dtype=getattr(trainer, "_autocast_dtype"),
                    device=trainer.accelerator.device,
                ),
                torch.tensor(
                    float("-inf"),
                    dtype=getattr(trainer, "_autocast_dtype"),
                    device=trainer.accelerator.device,
                ),
            )
            del mask

            # Calculate log probabilities
            lm_head_t = trainer.model.get_output_embeddings().weight.t()
            assistant_mask = inputs["assistant_mask"][:, 1:]
            max_tokens = 1024
            new_logprobs = calculate_log_probs(
                trainer,
                inputs["tokens"],
                attn_bias,
                lm_head_t,
                assistant_mask,
                max_tokens=max_tokens,
                reference_logprobs=False,
            )
            if config.kl_coef > 0.0:
                free_memory()
                reference_logprobs = calculate_log_probs(
                    trainer,
                    inputs["tokens"],
                    attn_bias,
                    lm_head_t,
                    assistant_mask,
                    max_tokens=max_tokens,
                    reference_logprobs=True,
                )
            else:
                reference_logprobs = None
            del attn_bias

            # Calculate policy loss
            old_logprobs = inputs["logprobs"][:, 1:][assistant_mask]
            old_logprobs = torch.where(
                torch.isnan(old_logprobs), new_logprobs, old_logprobs
            )
            advantages = inputs["advantages"][:, 1:][assistant_mask]
            diff = new_logprobs - old_logprobs
            prob_ratio = torch.exp(diff)
            policy_loss = -torch.min(
                prob_ratio * advantages,
                torch.clip(prob_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                * advantages,
            )
            # Calculate reverse KL divergence
            if reference_logprobs is not None:
                reverse_kl_divergence = (
                    torch.exp(reference_logprobs - new_logprobs)
                    - (reference_logprobs - new_logprobs)
                    - 1.0
                )
            else:
                reverse_kl_divergence = torch.zeros_like(advantages)
            mean_policy_loss = policy_loss.mean()
            mean_kl_divergence = reverse_kl_divergence.mean()
            trainer._metrics["policy_loss"].append(mean_policy_loss.item())
            if config.kl_coef > 0.0:
                trainer._metrics["kl_divergence"].append(mean_kl_divergence.item())
            return mean_policy_loss + config.kl_coef * mean_kl_divergence
        finally:
            for tensor in inputs.values():
                tensor.to("cpu")  # type: ignore
            free_memory()
            inputs_queue.task_done()

    _compute_loss = trainer.compute_loss
    trainer.compute_loss = compute_loss
    try:
        trainer.train()
    finally:
        trainer.compute_loss = _compute_loss


def calculate_log_probs(
    trainer: "GRPOTrainer",
    input_ids: torch.Tensor,
    causal_mask: torch.Tensor,
    lm_head_t: torch.Tensor,
    assistant_mask: torch.Tensor,
    max_tokens: int,
    reference_logprobs: bool,
) -> torch.Tensor:
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    with (
        torch.amp.autocast_mode.autocast(
            device_type="cuda", dtype=getattr(trainer, "_autocast_dtype")
        ),
        torch.inference_mode() if reference_logprobs else nullcontext(),
        (
            trainer.accelerator.unwrap_model(
                trainer.model, keep_fp32_wrapper=False
            ).disable_adapter()
            if reference_logprobs
            else nullcontext()
        ),
    ):
        hidden_states = trainer.model(
            input_ids=input_ids, causal_mask=causal_mask
        ).logits
        lm_head_t = lm_head_t.to(hidden_states.dtype)

        # Prepare tensors for masking (align dimensions)
        hidden_states = hidden_states[:, :-1, :]  # Shape: [B, S-1, H]
        next_tokens = input_ids[:, 1:]  # Shape: [B, S-1]

        # Apply mask first - flattens batch and sequence dimensions
        masked_hidden_states = hidden_states[assistant_mask]  # Shape: [N, H]
        masked_next_tokens = next_tokens[assistant_mask]  # Shape: [N]

        # N is the total number of assistant tokens across the batch
        N = masked_hidden_states.shape[0]
        if N == 0:
            # Handle cases with no assistant tokens if necessary
            return torch.tensor(
                [], device=hidden_states.device, dtype=hidden_states.dtype
            )  # Or appropriate dtype

        log_probs_chunks = []
        chunk_size = max_tokens  # Use max_tokens as the chunk size for the N dimension

        # Chunk the flattened tensors along the N dimension (dim=0)
        for i in range(0, N, chunk_size):
            # Get chunks of the masked hidden states and corresponding next tokens
            chunk_hs = masked_hidden_states[
                i : i + chunk_size
            ]  # Shape: [current_chunk_size, H]
            chunk_nt = masked_next_tokens[
                i : i + chunk_size
            ]  # Shape: [current_chunk_size]

            if chunk_nt.numel() == 0:
                continue  # Should not happen if N > 0, but safety check

            # Calculate logits for this chunk
            chunk_logits = torch.matmul(
                chunk_hs, lm_head_t
            )  # Shape: [current_chunk_size, V]

            # Select logits by token indices (gather operation)
            selected_logits = torch.gather(
                chunk_logits, dim=-1, index=chunk_nt.unsqueeze(-1)
            ).squeeze(
                -1
            )  # Shape: [current_chunk_size]

            # Calculate log_softmax manually: selected_logits - logsumexp
            logsumexp_values = torch.logsumexp(
                chunk_logits, dim=-1
            )  # Shape: [current_chunk_size]
            chunk_log_probs = (
                selected_logits - logsumexp_values
            )  # Shape: [current_chunk_size]

            log_probs_chunks.append(chunk_log_probs)

        # Concatenate all chunks along the N dimension
        log_probs = (
            torch.cat(log_probs_chunks, dim=0)
            if log_probs_chunks
            else torch.tensor([], device=masked_hidden_states.device)
        )  # Shape: [N]

        return log_probs


def free_memory() -> None:
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
