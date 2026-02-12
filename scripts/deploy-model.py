import argparse
import asyncio
import os

from dotenv import load_dotenv

import art
from art.utils.deployment import TogetherDeploymentConfig, deploy_model
from art.utils.get_model_step import get_model_step
from art.utils.output_dirs import get_model_dir, get_step_checkpoint_dir
from art.utils.s3 import pull_model_from_s3

load_dotenv()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a model checkpoint using ART")

    parser.add_argument("--project", required=True, help="ART project name")
    parser.add_argument("--model", required=True, help="Name of the model to deploy")
    parser.add_argument("--base-model", required=True, help="Base model to use")

    # Optional arguments
    parser.add_argument(
        "--backup-bucket",
        help="Name of the S3 bucket containing model checkpoints",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="latest",
        help="Training step to deploy (should correspond to a saved checkpoint)",
    )
    parser.add_argument(
        "--art-path", type=str, help="Path to the ART directory", default=".art"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main deployment routine
# ---------------------------------------------------------------------------


async def deploy() -> None:
    args = parse_args()

    backup_bucket = args.backup_bucket or os.environ["BACKUP_BUCKET"]

    model = art.TrainableModel(
        name=args.model,
        project=args.project,
        base_model=args.base_model,
    )

    if args.step == "latest":
        print("Pulling all checkpoints to determine the latest step…")
        # pull all checkpoints to determine the latest step
        await pull_model_from_s3(
            model_name=model.name,
            project=model.project,
            art_path=args.art_path,
            s3_bucket=backup_bucket,
        )
        step = get_model_step(model, args.art_path)
    else:
        print(f"Pulling checkpoint for step {args.step}…")
        step = int(args.step)
        # only pull the checkpoint we need
        await pull_model_from_s3(
            model_name=model.name,
            project=model.project,
            art_path=args.art_path,
            s3_bucket=backup_bucket,
            step=step,
        )

    print(
        f"Deploying {args.model} (project={args.project}, step={step}) "
        f"using checkpoints from s3://{backup_bucket}…"
    )

    # Construct the checkpoint path from the pulled model
    checkpoint_path = get_step_checkpoint_dir(
        get_model_dir(model=model, art_path=args.art_path), step
    )

    deployment_result = await deploy_model(
        model=model,
        checkpoint_path=checkpoint_path,
        step=step,
        provider="together",
        config=TogetherDeploymentConfig(
            s3_bucket=backup_bucket,
            wait_for_completion=True,
        ),
        verbose=True,
    )

    print("Deployment successful! ✨")
    print(f"Model deployed under name: {deployment_result.inference_model_name}")


if __name__ == "__main__":
    asyncio.run(deploy())
