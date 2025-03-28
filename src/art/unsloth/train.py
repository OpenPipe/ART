import asyncio
from datasets import Dataset
import nest_asyncio
import os
from peft.peft_model import PeftModel
import torch
from transformers import PreTrainedTokenizerBase
from trl import GRPOConfig, GRPOTrainer
from typing import cast, TYPE_CHECKING

from .grpo import GRPO
from ..types import TuneConfig

if TYPE_CHECKING:
    from ..model_service import TuneInputs

nest_asyncio.apply()


def get_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    args: GRPOConfig,
    inputs_queue: asyncio.Queue["TuneInputs"],
) -> GRPOTrainer:
    trainer = GRPOTrainer(
        model=model,  # type: ignore
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_list([{"prompt": ""} for _ in range(100_000)]),
        processing_class=tokenizer,
    )

    def _async_prepare_inputs(*_, **__) -> dict[str, torch.Tensor]:
        async def get_inputs() -> "TuneInputs":
            return await inputs_queue.get()

        inputs = asyncio.run(get_inputs())

        return cast(dict[str, torch.Tensor], inputs)

    trainer._prepare_inputs = _async_prepare_inputs
    return trainer


async def train(
    trainer: GRPOTrainer, inputs_queue: asyncio.Queue["TuneInputs"]
) -> None:
    # loss_fn = GRPO()
    # loss_fn._forward_chunk = torch.compile(
    #     loss_fn._forward_chunk,
    #     backend=os.environ.get("TORCH_COMPILE_BACKEND", "inductor"),
    # )

    def compute_loss(
        model: PeftModel,
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
            attn_bias = torch.where(mask, 0.0, float("-inf"))

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

            with torch.amp.autocast_mode.autocast(
                device_type="cuda", dtype=getattr(trainer, "_autocast_dtype")
            ):
                logits = model(input_ids=inputs["tokens"], causal_mask=attn_bias).logits
                logits = logits[:, :-1, :]

            mask = inputs["assistant_mask"][:, 1:]
            logits = logits[mask]

            selected_logits = (
                torch.gather(
                    logits, dim=-1, index=inputs["tokens"][:, 1:][mask].unsqueeze(-1)
                )
                .squeeze(-1)
                .to(torch.float32)
            )
            new_logprobs = selected_logits - torch.logsumexp(logits, dim=-1)
            del logits
            old_logprobs = inputs["logprobs"][:, 1:][mask]
            old_logprobs = torch.where(
                torch.isnan(old_logprobs), new_logprobs, old_logprobs
            )
            advantages = inputs["advantages"][:, 1:][mask]
            diff = new_logprobs - old_logprobs
            prob_ratio = torch.exp(diff)
            policy_loss = -torch.min(
                prob_ratio * advantages,
                torch.clip(prob_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
                * advantages,
            )
            del new_logprobs, old_logprobs, diff, prob_ratio, advantages
            return policy_loss.mean()
        finally:
            for tensor in inputs.values():
                tensor.to("cpu")  # type: ignore
            torch.cuda.empty_cache()
            inputs_queue.task_done()

    _compute_loss = trainer.compute_loss
    trainer.compute_loss = compute_loss
    try:
        trainer.train()
    finally:
        trainer.compute_loss = _compute_loss
