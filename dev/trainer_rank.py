from __future__ import annotations

import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
import typer

from art.trainer_rank import AdamParams, ForwardInput, TrainerRank


def main(
    model: str = "Qwen/Qwen3-0.6B",
    dataset: str = "roneneldan/TinyStories",
    split: str = "train",
    text_column: str = "text",
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
        for row in load_dataset(dataset, split=split, streaming=True):
            text = str(row.get(text_column, "")).strip()  # type: ignore[union-attr]
            if not text:
                continue
            token_ids = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq_length + 1,
                return_tensors="pt",
            )["input_ids"].reshape(-1)
            if int(token_ids.numel()) <= 1:
                continue
            inputs.append(
                ForwardInput(
                    input_tokens=token_ids[:-1],
                    target_tokens=token_ids[1:],
                )
            )
            if len(inputs) >= samples:
                break
        if not inputs:
            raise RuntimeError("dataset produced no tokenized training examples")

        runtime = megatron_train.build_training_runtime(
            model_identifier=model,
            provider_configure=lambda provider: setattr(
                provider,
                "num_layers",
                layers,
            ),
            print_env=dist.get_rank() == 0,
        )
        rank = TrainerRank(runtime)
        from art.megatron.lora import LoRA

        adapter: dict[str, torch.Tensor] = {}
        generator = torch.Generator(device=rank.device).manual_seed(0)
        for chunk in runtime.model:
            for module in chunk.modules():
                if not isinstance(module, LoRA):
                    continue
                for a_key, b_key in zip(
                    module._expected_weight_keys("lora_A"),
                    module._expected_weight_keys("lora_B"),
                    strict=True,
                ):
                    adapter[a_key] = torch.randn(
                        lora_rank,
                        module.in_features,
                        device=rank.device,
                        dtype=module.A_T.dtype,
                        generator=generator,
                    )
                    adapter[b_key] = torch.zeros(
                        module.out_features,
                        lora_rank,
                        device=rank.device,
                        dtype=module.B_T.dtype,
                    )
        loaded_sites = rank.load_checkpoint_slot("student", adapter)
        if loaded_sites == 0:
            raise RuntimeError("TrainerRank dev script requires LoRA adapter sites")
        rank.set_checkpoint("student")
        if dist.get_rank() == 0:
            print(
                "TrainerRank ready: "
                f"dp={megatron_train.ps.get_data_parallel_world_size()} "
                f"device={rank.device}",
                flush=True,
            )

        for step in range(steps):
            loss_sum = torch.tensor(0.0, device=rank.device)
            token_count = torch.tensor(0.0, device=rank.device)
            for micro in rank.forward_micro_batches(inputs):
                loss = torch.tensor(0.0, device=rank.device)
                for output in micro.outputs:
                    assert output.target_logprobs is not None
                    loss = loss - output.target_logprobs.sum()
                    token_count += output.target_logprobs.numel()
                if loss.requires_grad:
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

        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    typer.run(main)
