from __future__ import annotations

import os
import asyncio
from asyncio.subprocess import DEVNULL
import tempfile
from typing import Optional, Sequence
import zipfile

from art.errors import ForbiddenBucketCreationError
from art.utils.output_dirs import (
    get_output_dir_from_model_properties,
    get_step_checkpoint_dir,
)

from ..utils import limit_concurrency

__all__: Sequence[str] = ("s3_sync",)


class S3SyncError(RuntimeError):
    """Raised when the underlying *aws s3 sync* command exits with a non‑zero status."""


def build_s3_path(
    *,
    model_name: str,
    project: str,
    step: int | None = None,
    s3_bucket: str | None = None,
    prefix: str | None = None,
) -> str:
    """Return the fully-qualified S3 URI for this model directory."""
    if s3_bucket is None:
        s3_bucket = os.environ["BACKUP_BUCKET"]

    prefix_part = f"{prefix.strip('/')}/" if prefix else ""
    path = f"s3://{s3_bucket}/{prefix_part}{project}/models/{model_name}"
    if step is not None:
        path += f"/{step:04d}"
    return path


def build_s3_zipped_step_path(
    *,
    model_name: str,
    project: str,
    step: int,
    s3_bucket: str | None = None,
    prefix: str | None = None,
) -> str:
    """Return the fully-qualified S3 URI for a zipped step in a model directory."""
    base_path = build_s3_path(
        model_name=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )
    return f"{base_path}/zipped-steps/{step:04d}.zip"


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

    cmd += ["s3"]
    # us cp for files, sync for directories
    if os.path.isfile(source):
        cmd += ["cp"]
    else:
        cmd += ["sync"]

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
    step: int | None = None,
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
        step: A specific step to pull from S3. If None, all steps will be pulled.
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
    local_dir = local_model_dir
    if step is not None:
        local_step_dir = get_step_checkpoint_dir(local_model_dir, step)
        os.makedirs(local_step_dir, exist_ok=True)
        local_dir = local_step_dir

    s3_path = build_s3_path(
        model_name=model_name,
        project=project,
        step=step,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )
    await ensure_bucket_exists(s3_bucket)
    await s3_sync(s3_path, local_dir, verbose=verbose, delete=delete)

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
        model_name=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )
    await s3_sync(local_model_dir, s3_path, verbose=verbose, delete=delete)


async def archive_and_presign_step_url(
    model_name: str,
    project: str,
    step: int,
    s3_bucket: str | None = None,
    prefix: str | None = None,
    verbose: bool = False,
    delete: bool = False,
    art_path: str | None = None,
) -> str:
    """Get a presigned URL for a step in a model."""
    model_output_dir = get_output_dir_from_model_properties(
        project=project,
        name=model_name,
        art_path=art_path,
    )
    local_step_dir = get_step_checkpoint_dir(model_output_dir, step)
    if not os.path.exists(local_step_dir):
        raise ValueError(f"Local step directory does not exist: {local_step_dir}")

    s3_step_path = build_s3_zipped_step_path(
        model_name=model_name,
        project=project,
        step=step,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )

    # Create temporary directory for the zip file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create zip archive
        archive_path = os.path.join(temp_dir, "model.zip")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(local_step_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to zip with relative path
                    arcname = os.path.relpath(file_path, local_step_dir)
                    zipf.write(file_path, arcname)

        await ensure_bucket_exists(s3_bucket)
        await s3_sync(archive_path, s3_step_path, verbose=verbose, delete=delete)

    # Remove the s3:// prefix to get the key
    s3_key = s3_step_path.removeprefix("s3://")

    # Generate presigned URL with 1 hour expiration
    cmd = ["aws", "s3", "presign", s3_key, "--expires-in", "3600"]

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Failed to generate presigned URL: {stderr.decode()}")

    presigned_url = stdout.decode().strip()
    if verbose:
        print("presigned_url", presigned_url)
    return presigned_url
