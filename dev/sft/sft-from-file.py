"""Simple SFT training script using train_sft_from_file helper."""

import asyncio
import random

import art
from art.local import LocalBackend
from art.utils.sft import train_sft_from_file


async def main():
    backend = LocalBackend()

    model_name = "run-" + "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8)
    )
    model = art.TrainableModel(
        name=model_name,
        project="sft-from-file",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
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
