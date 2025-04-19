from __future__ import annotations

import asyncio
from asyncio.subprocess import DEVNULL
from typing import Optional, Sequence
from ..utils import limit_concurrency

__all__: Sequence[str] = ("s3_sync",)


class S3SyncError(RuntimeError):
    """Raised when the underlying *aws s3 sync* command exits with a non‑zero status."""


@limit_concurrency(1)
async def s3_sync(
    source: str,
    destination: str,
    *,
    profile: Optional[str] = None,
    verbose: bool = False,
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
    cmd += ["s3", "sync", "--delete", source, destination]

    # Suppress output unless verbose mode is requested.
    stdout = None if verbose else DEVNULL
    stderr = None if verbose else DEVNULL

    process = await asyncio.create_subprocess_exec(*cmd, stdout=stdout, stderr=stderr)
    return_code = await process.wait()

    if return_code != 0:
        raise S3SyncError(f"{' '.join(cmd)} exited with status {return_code}")
