import os

from art.local import LocalBackend
from art.model import Model as ArtModel


async def pull_model_trajectories(model: ArtModel) -> None:
    """Pull trajectory checkpoints for *model* from the configured S3 bucket.

    This is a lightweight helper that mirrors the S3-sync logic used inside
    ``art_e.train`` but without performing any training.  It can be invoked from
    notebooks or other scripts to ensure that the local ART project directory
    contains all trajectory files for subsequent evaluation / analysis.

    Parameters
    ----------
    model : art.Model
        Any ART model instance (trainable or not) that should be synchronised.

    Environment
    -----------
    BACKUP_BUCKET : str
        Name of the S3 bucket that stores model artefacts. The variable is
        loaded from the current environment (``dotenv`` is consulted so that
        values from a local *.env* file are respected).
    """

    bucket = os.getenv("BACKUP_BUCKET")
    if bucket is None:
        raise EnvironmentError(
            "Environment variable BACKUP_BUCKET is required but was not found."
        )

    # Use the LocalBackend context manager to work with the on-disk artefacts.
    with LocalBackend() as backend:
        print(
            f"Pulling trajectories for model '{model.name}' from S3 bucket '{bucket}'…",
            flush=True,
        )

        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=bucket,
            verbose=True,
            exclude=["checkpoints", "logs"],
        )

        print("Finished pulling trajectories.", flush=True)
