import asyncio
from collections import defaultdict
from contextlib import nullcontext
import gc
import os
from typing import TYPE_CHECKING, Callable, cast

import nest_asyncio
from peft.peft_model import PeftModel
import torch
from trl import GRPOTrainer

from .. import dev
from ..loss import loss_fn, shift_tensor
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
    trainer.log = get_log_fn(trainer, results_queue)  # ty:ignore[invalid-assignment]
    # Ensure we have a metrics container in the expected format
    try:
        is_dict = isinstance(getattr(trainer, "_metrics", None), dict)
        is_train_dict = is_dict and isinstance(trainer._metrics.get("train"), dict)
    except Exception:
        is_train_dict = False
    if not is_train_dict:
        trainer._metrics = {"train": defaultdict(list)}
    try:
        trainer.train()
    finally:
        trainer.compute_loss = _compute_loss
        trainer.log = _log  # ty:ignore[invalid-assignment]


def get_compute_loss_fn(trainer: "GRPOTrainer") -> Callable[..., torch.Tensor]:
    def compute_loss(
        model: "PeftModel",
        inputs: "TrainInputs",
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        config: TrainConfig = inputs.pop("config")  # type: ignore
        _config: dev.TrainConfig = inputs.pop("_config")  # type: ignore
        return_new_logprobs: bool = inputs.pop("return_new_logprobs", False)  # type: ignore

        num_trajectories_learning_rate_multiplier = (
            torch.unique(inputs["group_ids"]).numel()
            - torch.unique(inputs["parent_ids"]).numel()
        ) ** _config.get("num_trajectories_learning_rate_multiplier_power", 0.0)
        if optimizer := trainer.optimizer:
            optimizer = getattr(optimizer, "optimizer", optimizer)
            if param_groups := getattr(optimizer, "param_groups"):
                for param_group in param_groups:
                    param_group["lr"] = (
                        config.learning_rate * num_trajectories_learning_rate_multiplier
                    )
                    # param_group["betas"] = config.betas
                    # if param_group.get("weight_decay"):
                    #     param_group["weight_decay"] = config.weight_decay

        if inputs.get("pixel_values") and inputs["pixel_values"][0] is not None:
            inputs["pixel_values"] = inputs["pixel_values"][0]  # type: ignore
        else:
            del inputs["pixel_values"]  # type: ignore
        if inputs.get("image_grid_thw") and inputs["image_grid_thw"][0] is not None:
            inputs["image_grid_thw"] = inputs["image_grid_thw"][0]  # type: ignore
        else:
            del inputs["image_grid_thw"]  # type: ignore

        # Move tensors to the correct device
        inputs = {
            key: tensor.to(trainer.accelerator.device)  # type: ignore
            for key, tensor in inputs.items()
        }  # ty:ignore[invalid-assignment]

        accelerate_mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION")
        force_float32 = os.environ.get("UNSLOTH_FORCE_FLOAT32")

        if (
            accelerate_mixed_precision is None
            or accelerate_mixed_precision == "fp16"
            or force_float32 == "1"
        ):
            dtype_for_autocasting = torch.float16
        else:
            dtype_for_autocasting = torch.bfloat16

        batch_size, seq_len = inputs["tokens"].size()
        attn_bias = calculate_attn_bias(
            batch_size,
            seq_len,
            trainer.accelerator.device,
            inputs["group_ids"],
            inputs["parent_ids"],
            dtype_for_autocasting,
        )

        # Calculate log probabilities
        lm_head_t = cast(
            torch.Tensor,
            trainer.model.get_output_embeddings().weight.t(),  # type: ignore
        )  # Shape [H, V]
        next_input_ids = shift_tensor(inputs["tokens"], 0)
        chunk_size = _config.get("logprob_calculation_chunk_size", 1024)
        # Assert that sequence length is evenly divisible by the chunk size
        assert seq_len % chunk_size == 0, (
            f"Sequence length ({seq_len}) must be evenly divisible by chunk size ({chunk_size})"
        )
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        forward_kwargs = {}
        if "pixel_values" in inputs:
            forward_kwargs["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            forward_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
        new_logprobs, entropies = calculate_logprobs(
            dtype_for_autocasting,
            trainer,
            inputs["tokens"],
            attn_bias,
            forward_kwargs,
            next_input_ids,
            lm_head_t,
            chunk_size=chunk_size,
            inference_mode=return_new_logprobs,
            no_grad=return_new_logprobs,
            reference_logprobs=False,
        )
        if return_new_logprobs:
            return torch.nn.functional.pad(new_logprobs[:, :-1], (1, 0), value=0.0)
        if config.beta > 0.0:
            ref_logprobs, _ = calculate_logprobs(
                dtype_for_autocasting,
                trainer,
                inputs["tokens"],
                attn_bias,
                forward_kwargs,
                next_input_ids,
                lm_head_t,
                chunk_size=chunk_size,
                inference_mode=True,
                no_grad=False,
                reference_logprobs=True,
            )
        else:
            ref_logprobs = None
        del attn_bias

        loss = loss_fn(
            inputs,
            new_logprobs,
            ref_logprobs,
            entropies,
            _config,
        )

        trainer._metrics["train"]["learning_rate"].append(config.learning_rate)
        trainer._metrics["train"]["policy_loss"].append(loss.mean_policy_loss.item())
        if loss.mean_entropy is not None:
            trainer._metrics["train"]["entropy"].append(loss.mean_entropy.item())
        if config.beta > 0.0:
            trainer._metrics["train"]["kl_div"].append(loss.mean_kl.item())
        return loss.mean_policy_loss + config.beta * loss.mean_kl

    return compute_loss


def get_log_fn(
    trainer: "GRPOTrainer", results_queue: asyncio.Queue[dict[str, float]]
) -> Callable[..., None]:
    def log(logs: dict[str, float], start_time: float | None = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in trainer._metrics["train"].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        logs.pop("learning_rate", None)
        results_queue.put_nowait(logs)
        trainer._metrics["train"].clear()

    return log


def calculate_attn_bias(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = calculate_mask(batch_size, seq_len, device, group_ids, parent_ids)
    # Use the same dtype as autocast to save memory and avoid dtype conversions
    attn_bias = torch.where(
        mask,
        torch.tensor(
            0.0,
            dtype=dtype,
            device=device,
        ),
        torch.tensor(
            float("-inf"),
            dtype=dtype,
            device=device,
        ),
    )
    del mask
    return attn_bias


def calculate_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    group_ids: torch.Tensor,
    parent_ids: torch.Tensor,
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
    return mask


def calculate_logprobs(
    dtype_for_autocast: torch.dtype,
    trainer: "GRPOTrainer",
    input_ids: torch.Tensor,
    causal_mask: torch.Tensor,
    forward_kwargs: dict[str, torch.Tensor],
    next_input_ids: torch.Tensor,
    lm_head_t: torch.Tensor,
    chunk_size: int,
    inference_mode: bool,
    no_grad: bool,
    reference_logprobs: bool,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # Returns (log_probs, entropy) both shape [B, S]
    with (
        torch.inference_mode() if inference_mode else nullcontext(),
        torch.no_grad() if no_grad else nullcontext(),
        (
            trainer.accelerator.unwrap_model(
                trainer.model, keep_fp32_wrapper=False
            ).disable_adapter()
            if reference_logprobs
            else nullcontext()
        ),
        torch.amp.autocast_mode.autocast(device_type="cuda", dtype=dtype_for_autocast),
    ):
        hidden_states = trainer.model(  # type: ignore
            input_ids=input_ids, causal_mask=causal_mask, **forward_kwargs
        ).logits  # Shape [B, S, H]
    return _calculate_logprobs(lm_head_t, hidden_states, next_input_ids, chunk_size)


def _calculate_logprobs(
    lm_head_t: torch.Tensor,  # Shape [H, V]
    hidden_states: torch.Tensor,  # Shape [B, S, H]
    next_input_ids: torch.Tensor,  # Shape [B, S]
    chunk_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # Returns (log_probs, entropy) both shape [B, S]
    batch_size, seq_len, _ = hidden_states.shape
    # Output shape is [B, S]
    log_probs = torch.empty(
        (batch_size, seq_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    entropy = torch.empty(
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
        ).squeeze(-1)  # [B, chunk_size]
        chunk_logsumexp = torch.logsumexp(chunk_logits, dim=-1)  # [B, chunk_size]
        log_probs[:, i : i + chunk_size] = chunk_selected_logits - chunk_logsumexp

        # Compute entropy for the chunk
        log_probs_full = chunk_logits - chunk_logsumexp.unsqueeze(-1)
        chunk_entropy = (-torch.exp(log_probs_full) * log_probs_full).sum(
            dim=-1
        )  # [B, chunk_size]
        entropy[:, i : i + chunk_size] = chunk_entropy

        del (
            chunk_hs,
            chunk_input_ids,
            chunk_logits,
            chunk_selected_logits,
            chunk_logsumexp,
            log_probs_full,
            chunk_entropy,
        )
    del hidden_states
    return log_probs, entropy


def gc_and_empty_cuda_cache(n: int = 3) -> None:
    [gc.collect() >= 0 and torch.cuda.empty_cache() for _ in range(n)]
