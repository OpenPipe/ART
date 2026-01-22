"""Simple SFT training script using train_sft_from_file helper."""

import asyncio

import art
from art.local import LocalBackend
from art.utils.sft import train_sft_from_file


async def main():
    backend = LocalBackend()
    model = art.TrainableModel(
        name="run-5",
        project="sft-from-file",
        base_model="Qwen/Qwen2.5-7B-Instruct",
    )
    await model.register(backend)

    await train_sft_from_file(
        model=model,
        file_path="dev/sft/dataset.jsonl",
        epochs=1,
        chunk_size=10,
        peak_lr=2e-4,
    )

    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
