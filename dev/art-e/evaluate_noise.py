#!/usr/bin/env python3

import asyncio
import os
import statistics

from all_experiments import models
from art_e.data.local_email_db import generate_database
from art_e.evaluate.benchmark import benchmark_model
from dotenv import load_dotenv

from art.local import LocalBackend

load_dotenv()


async def evaluate_validation_noise(model_key: str, num_runs: int = 20):
    """
    Evaluate validation set noise by running the same model multiple times
    and measuring the variance in results.
    """
    print(f"Evaluating validation noise for model {model_key} with {num_runs} runs...")

    # Get the model from all_experiments
    if model_key not in models:
        raise ValueError(
            f"Model {model_key} not found. Available: {list(models.keys())}"
        )

    model = models[model_key].model_copy(deep=True)
    generate_database()

    # Override validation runs to 1 for this test (we want to see run-to-run variance)
    model.config.num_validation_runs = 1

    with LocalBackend() as backend:
        print(
            f"Pulling latest checkpoint from S3 bucket: `{os.environ['BACKUP_BUCKET']}`"
        )
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
            verbose=True,
            only_step="latest",
            exclude=["trajectories"],
        )
        await model.register(backend)

        scores = []

        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")

            # Run benchmark with report=False to avoid logging noise
            result_df = await benchmark_model(
                model,
                limit=model.config.val_set_size,  # Use the model's configured val set size
                step=0,
                report=False,
            )

            # Extract the answer_correct metric (number of correct answers)
            if "answer_correct" in result_df.columns:
                correct_answers = (
                    result_df["answer_correct"][0] * result_df["n_trajectories"][0]
                )
                scores.append(int(correct_answers))
                print(
                    f"  Correct answers: {int(correct_answers)}/{result_df['n_trajectories'][0]}"
                )
            else:
                print("  Warning: 'answer_correct' metric not found in results")

        if scores:
            print(f"\n--- Results for Model {model_key} ---")
            print(f"Individual runs: {scores}")
            print(f"Average: {statistics.mean(scores):.2f}")
            print(f"Standard deviation: {statistics.stdev(scores):.2f}")
            print(f"Min: {min(scores)}")
            print(f"Max: {max(scores)}")
            print(f"Range: {max(scores) - min(scores)}")

            # Calculate coefficient of variation (std dev / mean)
            cv = statistics.stdev(scores) / statistics.mean(scores) * 100
            print(f"Coefficient of variation: {cv:.1f}%")
        else:
            print("No valid scores collected!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate validation set noise")
    parser.add_argument(
        "--model", default="231", help="Model key to evaluate (default: 231)"
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="Number of runs (default: 20)"
    )

    args = parser.parse_args()

    asyncio.run(evaluate_validation_noise(args.model, args.runs))
