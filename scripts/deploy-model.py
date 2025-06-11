import argparse
import asyncio
import os
from dotenv import load_dotenv

import art
from art.utils.deploy_model import deploy_model
from art.utils.get_model_step import get_model_step
from art.utils.s3 import pull_model_from_s3


load_dotenv()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a model checkpoint using ART & SkyPilot"
    )

    parser.add_argument("--model", required=True, help="Name of the model to deploy")
    parser.add_argument("--project", required=True, help="ART project name")

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
        # base model is not used for deployment, but is required for model init
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
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

    deployment_result = await deploy_model(
        deploy_to="together",
        model=model,
        step=step,
        verbose=True,
        pull_s3=False,
        wait_for_completion=True,
        art_path=args.art_path,
    )

    if deployment_result.status == "Failed":
        raise RuntimeError(f"Deployment failed: {deployment_result.failure_reason}")

    print("Deployment successful! ✨")
    print(
        f"Model deployed at Together under name: {deployment_result.model_name} (job_id={deployment_result.job_id})"
    )


if __name__ == "__main__":
    asyncio.run(deploy())
