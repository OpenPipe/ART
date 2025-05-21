import asyncio
from contextlib import nullcontext
import nest_asyncio
import os
from peft.peft_model import PeftModel
import torch
from trl import GRPOTrainer
from typing import cast, Callable, TYPE_CHECKING

from .. import dev
from ..types import TrainConfig

if TYPE_CHECKING:
    from .service import TrainInputs

nest_asyncio.apply()


async def train(
    trainer: "GRPOTrainer",
    results_queue: asyncio.Queue[dict[str, float]],
) -> None:
    _compute_loss = trainer.compute_loss
    _log = trainer.log
    trainer.compute_loss = get_compute_loss_fn(trainer)
    trainer.log = get_log_fn(trainer, results_queue)
    try:
        trainer.train()
    finally:
        trainer.compute_loss = _compute_loss
        trainer.log = _log


def get_compute_loss_fn(trainer: "GRPOTrainer") -> Callable[..., torch.Tensor]:
    def compute_loss(
        model: "PeftModel",
        inputs: "TrainInputs",
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        config: TrainConfig = inputs.pop("config")  # type: ignore
        _config: dev.TrainConfig = inputs.pop("_config")  # type: ignore

        if optimizer := trainer.optimizer:
            optimizer = getattr(optimizer, "optimizer", optimizer)
            if param_groups := getattr(optimizer, "param_groups"):
                for param_group in param_groups:
                    param_group["lr"] = config.learning_rate
                    # param_group["betas"] = config.betas
                    # if param_group.get("weight_decay"):
                    #     param_group["weight_decay"] = config.weight_decay

        # Move tensors to the correct device
        inputs = {key: tensor.to(trainer.accelerator.device) for key, tensor in inputs.items()}  # type: ignore

        # Unsloth code
        autocast_dtype = (
            torch.float16
            if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
            else torch.bfloat16
        )
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            autocast_dtype = torch.float16

        batch_size, seq_len = inputs["tokens"].size()
        attn_bias = calculate_attn_bias(
            batch_size,
            seq_len,
            trainer.accelerator.device,
            inputs["group_ids"],
            inputs["parent_ids"],
            autocast_dtype,
        )

        # Calculate log probabilities
        lm_head_t = cast(
            torch.Tensor, trainer.model.get_output_embeddings().weight.t()  # type: ignore
        )  # Shape [H, V]
        next_input_ids = shift_tensor(inputs["tokens"], 0)
        chunk_size = _config.get("logprob_calculation_chunk_size", 1024)
        # Assert that sequence length is evenly divisible by the chunk size
        assert (
            seq_len % chunk_size == 0
        ), f"Sequence length ({seq_len}) must be evenly divisible by chunk size ({chunk_size})"
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        new_logprobs = calculate_logprobs(
            autocast_dtype,
            trainer,
            inputs["tokens"],
            attn_bias,
            next_input_ids,
            lm_head_t,
            chunk_size=chunk_size,
            reference_logprobs=False,
        )
        if config.beta > 0.0:
            ref_logprobs = calculate_logprobs(
                autocast_dtype,
                trainer,
                inputs["tokens"],
                attn_bias,
                next_input_ids,
                lm_head_t,
                chunk_size=chunk_size,
                reference_logprobs=True,
            )
        else:
            ref_logprobs = None
        del attn_bias

        # Shift inputs for loss calculation
        old_logprobs = shift_tensor(inputs["logprobs"], 0.0)
        advantages = shift_tensor(inputs["advantages"], 0.0)
        assistant_mask = shift_tensor(inputs["assistant_mask"], False).to(
            new_logprobs.dtype
        )
        weights = shift_tensor(inputs["weights"], 0.0)
        # Assume missing old logprobs were sampled under the current policy
        old_logprobs = torch.where(
            torch.isnan(old_logprobs),
            new_logprobs,
            old_logprobs,
        )
        prob_ratio = torch.exp(new_logprobs - old_logprobs)
        epsilon = _config.get("epsilon", 0.2)
        epsilon_high = _config.get("epsilon_high", epsilon)
        if epsilon_high is None:
            epsilon_high = epsilon
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
        )
        if ref_logprobs is not None:
            kl_div = (
                torch.exp(ref_logprobs - new_logprobs)
                - (ref_logprobs - new_logprobs)
                - 1.0
            )
        else:
            kl_div = torch.zeros_like(policy_loss)

        policy_loss = policy_loss * weights * assistant_mask
        kl_div = kl_div * weights * assistant_mask
        mean_policy_loss = policy_loss.sum() / (assistant_mask.sum() + 1e-6)
        mean_kl = kl_div.sum() / (assistant_mask.sum() + 1e-6)

        trainer._metrics["learning_rate"].append(config.learning_rate)
        trainer._metrics["policy_loss"].append(mean_policy_loss.item())
        if config.beta > 0.0:
            trainer._metrics["kl_div"].append(mean_kl.item())
        return mean_policy_loss + config.beta * mean_kl

    return compute_loss


def get_log_fn(
    trainer: "GRPOTrainer", results_queue: asyncio.Queue[dict[str, float]]
) -> Callable[..., None]:
    def log(logs: dict[str, float], start_time: float | None = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in trainer._metrics.items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        logs.pop("learning_rate", None)
        results_queue.put_nowait(logs)
        trainer._metrics.clear()

    return log


def calculate_attn_bias(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    causal_mask = (
        torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        )
        .unsqueeze(0)
        .expand(batch_size, seq_len, seq_len)
    )
    group_mask = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    parent_mask = parent_ids.unsqueeze(2) == group_ids.unsqueeze(1)
    mask = causal_mask & (group_mask | parent_mask)
    # Use the same dtype as autocast to save memory and avoid dtype conversions
    attn_bias = torch.where(
        mask,
        torch.tensor(
            0.0,
            dtype=autocast_dtype,
            device=device,
        ),
        torch.tensor(
            float("-inf"),
            dtype=autocast_dtype,
            device=device,
        ),
    )
    del mask
    return attn_bias


def calculate_logprobs(
    autocast_dtype: torch.dtype,
    trainer: "GRPOTrainer",
    input_ids: torch.Tensor,
    causal_mask: torch.Tensor,
    next_input_ids: torch.Tensor,
    lm_head_t: torch.Tensor,
    chunk_size: int,
    reference_logprobs: bool,
) -> torch.Tensor:  # Returns shape [B, S]
    with (
        torch.amp.autocast_mode.autocast(device_type="cuda", dtype=autocast_dtype),
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
        ).logits  # Shape [B, S, H]
    return _calculate_logprobs(lm_head_t, hidden_states, next_input_ids, chunk_size)


def _calculate_logprobs(
    lm_head_t: torch.Tensor,  # Shape [H, V]
    hidden_states: torch.Tensor,  # Shape [B, S, H]
    next_input_ids: torch.Tensor,  # Shape [B, S]
    chunk_size: int,
) -> torch.Tensor:  # Returns shape [B, S]
    batch_size, seq_len, _ = hidden_states.shape
    # Output shape is [B, S]
    log_probs = torch.empty(
        (batch_size, seq_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    # Ensure lm_head_t is in the same dtype as hidden_states
    lm_head_t = lm_head_t.to(hidden_states.dtype)

    # Chunk over sequence length S using Python range
    for i in range(0, seq_len, chunk_size):
        chunk_hs = hidden_states[:, i : i + chunk_size, :]  # [B, chunk_size, H]
        chunk_input_ids = next_input_ids[:, i : i + chunk_size]  # [B, chunk_size]
        chunk_logits = torch.matmul(chunk_hs, lm_head_t)  # [B, chunk_size, V]
        chunk_selected_logits = torch.gather(
            chunk_logits, dim=-1, index=chunk_input_ids.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [B, chunk_size]
        chunk_logsumexp = torch.logsumexp(chunk_logits, dim=-1)  # [B, chunk_size]
        log_probs[:, i : i + chunk_size] = chunk_selected_logits - chunk_logsumexp
        del (
            chunk_hs,
            chunk_input_ids,
            chunk_logits,
            chunk_selected_logits,
            chunk_logsumexp,
        )
    del hidden_states
    return log_probs


def shift_tensor(tensor: torch.Tensor, pad: int | float | bool) -> torch.Tensor:
    return torch.nn.functional.pad(tensor[:, 1:], (0, 1), value=pad)
