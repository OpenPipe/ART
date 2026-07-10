from itertools import islice
import os

import torch
import torch.distributed as dist
from trainer_rank_support import load_random_checkpoint_slots
from transformers import AutoTokenizer
import typer

from art.trainer_rank import AdamParams, ForwardInput, TrainerRank


def main(
    model: str = "Qwen/Qwen3-0.6B",
    samples: int = 16,
    steps: int = 1,
    lr: float = 5e-5,
    layers: int = 2,
    lora_rank: int = 8,
    max_seq_length: int = 256,
) -> None:
    os.environ.setdefault("ART_MEGATRON_TENSOR_MODEL_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_CONTEXT_PARALLEL_SIZE", "1")
    os.environ.setdefault("ART_MEGATRON_PIPELINE_MODEL_PARALLEL_SIZE", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("dev/trainer_rank.py requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

    try:
        from datasets import load_dataset

        from art.megatron import train as megatron_train

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        inputs: list[ForwardInput[torch.Tensor, None, None, None]] = []
        rows = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for row in islice(rows, samples):
            token_ids = tokenizer(
                str(row["text"]),  # type: ignore[index]
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_length + 1,
                return_tensors="pt",
            )["input_ids"].reshape(-1)
            inputs.append(
                ForwardInput(
                    input_tokens=token_ids[:-1],
                    target_tokens=token_ids[1:],
                )
            )

        runtime = megatron_train.build_training_runtime(
            model_identifier=model,
            provider_configure=lambda provider: setattr(provider, "num_layers", layers),
            print_env=dist.get_rank() == 0,
        )
        rank = TrainerRank(runtime)
        (slot,) = load_random_checkpoint_slots(runtime, rank, 1, lora_rank=lora_rank)
        rank.set_checkpoint(slot)

        for step in range(steps):
            loss_sum = torch.tensor(0.0, device=rank.device)
            token_count = torch.tensor(0.0, device=rank.device)
            for micro in rank.forward_micro_batches(inputs):
                loss = torch.tensor(0.0, device=rank.device)
                for output in micro.outputs:
                    assert output.target_logprobs is not None
                    loss = loss - output.target_logprobs.sum()
                    token_count += output.target_logprobs.numel()
                loss.backward()
                loss_sum += loss.detach()

            rank.dp_reduce(loss_sum)
            rank.dp_reduce(token_count)
            scale = 1.0 / max(float(token_count.item()), 1.0)
            metrics = rank.optim_step(
                params=AdamParams(learning_rate=lr),
                scale_grads=scale,
            )
            metrics["loss"] = float(loss_sum.item() * scale)
            metrics["tokens"] = float(token_count.item())
            if dist.get_rank() == 0:
                print(f"step={step} {metrics}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    typer.run(main)
