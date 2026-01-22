"""Distillation example: Train a small model using completions from a large model."""

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

import art
from art.local import LocalBackend

load_dotenv()

if not os.environ.get("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

TEACHER_MODEL = "qwen/qwen3-235b-a22b-2507"
STUDENT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Explain the concept of recursion in programming with a simple example."


async def main():
    # Get completion from teacher model
    teacher_client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    print(f"Getting completion from teacher model ({TEACHER_MODEL})...")
    completion = await teacher_client.chat.completions.create(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    teacher_response = completion.choices[0].message.content
    print(
        f"Teacher response ({len(teacher_response)} chars):\n{teacher_response[:500]}..."
    )

    # Create trajectory from teacher completion
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": teacher_response},
        ],
        reward=0.0,
    )

    # Train student model
    backend = LocalBackend()
    student = art.TrainableModel(
        name="distillation-demo-11",
        project="sft-distillation",
        base_model=STUDENT_BASE_MODEL,
    )
    await student.register(backend)

    print(f"Training student model ({STUDENT_BASE_MODEL})...")
    await student.train_sft(
        [trajectory, trajectory, trajectory],
        config=art.SFTConfig(learning_rate=2e-4),
        verbose=True,
    )
    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
