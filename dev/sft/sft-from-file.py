"""Simple SFT training script using train_sft_from_file helper."""

import asyncio
import random

import art
from art.megatron import MegatronBackend
from art.utils.sft import train_sft_from_file


async def main():
    art.init_megatron_runtime_config(
        topology=art.MegatronTopologyConfig(),
        packed_sequence_length=4096,
    )
    model_name = "run-" + "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)
    )
    model = art.TrainableModel(
        run_name=model_name,
        name=model_name,
        project="sft-from-file",
        base_model="Qwen/Qwen3.6-35B-A3B",
    )
    async with MegatronBackend() as backend:
        await model.register(backend)
        await train_sft_from_file(
            model=model,
            file_path="dev/sft/dataset.jsonl",
            epochs=1,
            peak_lr=2e-4,
        )

    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
