from __future__ import annotations

import os
import asyncio
from asyncio.subprocess import DEVNULL
from typing import Optional, Sequence

from art.errors import ForbiddenBucketCreationError
from art.utils.output_dirs import get_output_dir_from_model_properties

from ..utils import limit_concurrency

__all__: Sequence[str] = ("s3_sync",)


class S3SyncError(RuntimeError):
    """Raised when the underlying *aws s3 sync* command exits with a non‑zero status."""


def build_s3_path(
    *,
    model: str,
    project: str,
    s3_bucket: str | None = None,
    prefix: str | None = None,
) -> str:
    """Return the fully-qualified S3 URI for this model directory."""
    if s3_bucket is None:
        s3_bucket = os.environ["BACKUP_BUCKET"]

    prefix_part = f"{prefix.strip('/')}/" if prefix else ""
    return f"s3://{s3_bucket}/{prefix_part}{project}/models/{model}"


@limit_concurrency(1)
async def s3_sync(
    source: str,
    destination: str,
    *,
    profile: Optional[str] = None,
    verbose: bool = False,
    delete: bool = False,
) -> None:
    """Synchronise *source* and *destination* using the AWS CLI.

    Either *source* or *destination* (or both) can point to an S3 URI, making it
    possible to copy from local disk to S3 or from S3 to local disk.

    The function is asynchronous: while the `aws` process runs, control is
    yielded back to the event loop so other tasks can continue executing.

    Args:
        source: The *from* path. Can be a local path or an ``s3://`` URI.
        destination: The *to* path. Can be a local path or an ``s3://`` URI.
        profile: Optional AWS profile name to pass to the CLI.
        verbose: When *True*, the output of the AWS CLI is streamed to the
            calling process; otherwise it is suppressed.

    Raises:
        S3SyncError: If the *aws s3 sync* command exits with a non‑zero status.
    """

    cmd: list[str] = ["aws"]
    if profile:
        cmd += ["--profile", profile]

    cmd += ["s3", "sync"]
    if delete:
        cmd.append("--delete")
    cmd += [source, destination]

    # Suppress output unless verbose mode is requested.
    stdout = None if verbose else DEVNULL
    stderr = None if verbose else DEVNULL

    process = await asyncio.create_subprocess_exec(*cmd, stdout=stdout, stderr=stderr)
    return_code = await process.wait()

    if return_code != 0:
        raise S3SyncError(f"{' '.join(cmd)} exited with status {return_code}")


async def ensure_bucket_exists(
    s3_bucket: str | None = None, profile: str | None = None
) -> None:
    if s3_bucket is None:
        s3_bucket = os.environ["BACKUP_BUCKET"]

    # Check if bucket exists
    cmd = ["aws"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["s3api", "head-bucket", "--bucket", s3_bucket]

    result = await asyncio.create_subprocess_exec(*cmd, stdout=DEVNULL, stderr=DEVNULL)
    return_code = await result.wait()

    if return_code == 0:
        return  # Bucket exists

    # Try to create the bucket
    print(f"S3 bucket {s3_bucket} does not exist, creating it")
    cmd = ["aws"]
    if profile:
        cmd += ["--profile", profile]
    cmd += ["s3api", "create-bucket", "--bucket", s3_bucket]

    result = await asyncio.create_subprocess_exec(*cmd)
    return_code = await result.wait()

    if return_code != 0:
        raise ForbiddenBucketCreationError(
            message=f"Failed to create bucket {s3_bucket}. It may already exist and belong to another user, or your credentials may be insufficient to create an S3 bucket."
        )


async def pull_model_from_s3(
    model_name: str,
    project: str,
    s3_bucket: str | None = None,
    prefix: str | None = None,
    verbose: bool = False,
    delete: bool = False,
    art_path: str | None = None,
) -> str:
    """Pull a model from S3 to the local directory.

    Args:
        model_name: The name of the model to pull.
        project: The project name.
        s3_bucket: The S3 bucket to pull from.
        prefix: The prefix to pull from.
        verbose: When *True*, the output of the AWS CLI is streamed to the
            calling process; otherwise it is suppressed.
        delete: When *True*, delete the local model directory if it exists.
        art_path: The path to the ART directory.

    Returns:
        The local directory path.
    """
    local_model_dir = get_output_dir_from_model_properties(
        project=project,
        name=model_name,
        art_path=art_path,
    )
    os.makedirs(local_model_dir, exist_ok=True)

    s3_path = build_s3_path(
        model=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )
    await ensure_bucket_exists(s3_bucket)
    await s3_sync(s3_path, local_model_dir, verbose=verbose, delete=delete)

    return local_model_dir


async def push_model_to_s3(
    model_name: str,
    project: str,
    s3_bucket: str | None = None,
    prefix: str | None = None,
    verbose: bool = False,
    delete: bool = False,
    art_path: str | None = None,
) -> None:
    """Push a model to S3.

    Args:
        model_name: The name of the model to push.
        project: The project name.
        s3_bucket: The S3 bucket to push to.
        prefix: The prefix to push to.
        verbose: When *True*, the output of the AWS CLI is streamed to the
            calling process; otherwise it is suppressed.
        delete: When *True*, delete the local model directory if it exists.
        art_path: The path to the ART directory.
    """
    local_model_dir = get_output_dir_from_model_properties(
        project=project,
        name=model_name,
        art_path=art_path,
    )
    if not os.path.exists(local_model_dir):
        raise FileNotFoundError(
            f"Local model directory {local_model_dir} does not exist."
        )
    s3_path = build_s3_path(
        model=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )
    await s3_sync(local_model_dir, s3_path, verbose=verbose, delete=delete)
