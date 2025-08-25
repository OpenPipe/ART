"""Utilities for working with S3 checkpoints."""

import asyncio
from asyncio.subprocess import PIPE


async def get_latest_checkpoint_step_from_s3(
    model_name: str,
    project: str,
    s3_bucket: str | None = None,
    prefix: str | None = None,
) -> int | None:
    """
    Get the latest checkpoint step number from S3 without downloading files.

    Returns:
        The latest step number, or None if no checkpoints exist.
    """
    from .s3 import build_s3_path

    s3_path = build_s3_path(
        model_name=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )

    # List checkpoint directories in S3
    cmd = ["aws", "s3", "ls", f"{s3_path}/checkpoints/"]

    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        # No checkpoints found or error
        return None

    # Parse output to find checkpoint directories
    lines = stdout.decode().strip().split("\n")
    checkpoint_steps = []

    for line in lines:
        if line.strip():
            # AWS S3 ls output format: "PRE 0001/"
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "PRE":
                dirname = parts[1].rstrip("/")
                if dirname.isdigit():
                    checkpoint_steps.append(int(dirname))

    return max(checkpoint_steps) if checkpoint_steps else None


async def get_checkpoint_step_not_after_from_s3(
    model_name: str,
    project: str,
    not_after_step: int,
    s3_bucket: str | None = None,
    prefix: str | None = None,
) -> int | None:
    """
    Get the latest checkpoint step number that is not after the specified step from S3.

    Args:
        not_after_step: Find the latest checkpoint <= this step.

    Returns:
        The step number, or None if no suitable checkpoint exists.
    """
    from .s3 import build_s3_path

    s3_path = build_s3_path(
        model_name=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )

    # List checkpoint directories in S3
    cmd = ["aws", "s3", "ls", f"{s3_path}/checkpoints/"]

    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        # No checkpoints found or error
        return None

    # Parse output to find checkpoint directories
    lines = stdout.decode().strip().split("\n")
    valid_steps = []

    for line in lines:
        if line.strip():
            # AWS S3 ls output format: "PRE 0001/"
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "PRE":
                dirname = parts[1].rstrip("/")
                if dirname.isdigit():
                    step = int(dirname)
                    if step <= not_after_step:
                        valid_steps.append(step)

    return max(valid_steps) if valid_steps else None


async def migrate_s3_checkpoints_to_new_structure(
    model_name: str,
    project: str,
    s3_bucket: str | None = None,
    prefix: str | None = None,
    dry_run: bool = False,
) -> None:
    """
    Migrate existing checkpoints in S3 from the old structure to the new structure.

    Old: s3://bucket/prefix/project/models/model_name/0001/
    New: s3://bucket/prefix/project/models/model_name/checkpoints/0001/

    Args:
        model_name: The name of the model to migrate.
        project: The project name.
        s3_bucket: The S3 bucket. If None, uses BACKUP_BUCKET env var.
        prefix: Optional prefix for the S3 path.
        dry_run: If True, only print what would be done without making changes.
    """
    import os

    from .s3 import build_s3_path

    if s3_bucket is None:
        s3_bucket = os.environ.get("BACKUP_BUCKET")
        if not s3_bucket:
            raise ValueError(
                "BACKUP_BUCKET environment variable not set and no bucket provided"
            )

    s3_path = build_s3_path(
        model_name=model_name,
        project=project,
        s3_bucket=s3_bucket,
        prefix=prefix,
    )

    print(f"Checking for checkpoints to migrate in {s3_path}")

    # List all directories in the model path
    cmd = ["aws", "s3", "ls", f"{s3_path}/"]
    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        print(f"Error listing S3 path: {stderr.decode()}")
        return

    # Parse output to find checkpoint directories
    lines = stdout.decode().strip().split("\n")
    checkpoint_dirs = []

    for line in lines:
        if line.strip():
            # AWS S3 ls output format: "PRE 0001/"
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "PRE":
                dirname = parts[1].rstrip("/")
                # Check if it's a 4-digit checkpoint directory (old format)
                if dirname.isdigit() and len(dirname) == 4:
                    checkpoint_dirs.append(dirname)

    if not checkpoint_dirs:
        print("No checkpoints found in old format to migrate")
        return

    print(
        f"Found {len(checkpoint_dirs)} checkpoint(s) to migrate: {', '.join(checkpoint_dirs)}"
    )

    if dry_run:
        print("DRY RUN: Would migrate the following checkpoints:")
        for checkpoint in checkpoint_dirs:
            print(f"  {s3_path}/{checkpoint}/ -> {s3_path}/checkpoints/{checkpoint}/")
        return

    # Perform migration
    migrated_count = 0
    for checkpoint in checkpoint_dirs:
        old_path = f"{s3_path}/{checkpoint}/"
        new_path = f"{s3_path}/checkpoints/{checkpoint}/"

        print(f"Migrating checkpoint {checkpoint}...")

        # Check if already exists in new location
        check_cmd = ["aws", "s3", "ls", new_path]
        check_process = await asyncio.create_subprocess_exec(
            *check_cmd, stdout=PIPE, stderr=PIPE
        )
        check_stdout, _ = await check_process.communicate()

        if check_process.returncode == 0 and check_stdout.decode().strip():
            print(f"  Checkpoint {checkpoint} already exists in new location, skipping")
            continue

        # Copy checkpoint to new location (using sync to preserve structure)
        sync_cmd = ["aws", "s3", "sync", old_path, new_path]
        sync_process = await asyncio.create_subprocess_exec(
            *sync_cmd, stdout=PIPE, stderr=PIPE
        )
        _, sync_stderr = await sync_process.communicate()

        if sync_process.returncode != 0:
            print(f"  Error copying checkpoint {checkpoint}: {sync_stderr.decode()}")
            continue

        # Verify copy was successful by checking if files exist in new location
        verify_cmd = ["aws", "s3", "ls", new_path, "--recursive"]
        verify_process = await asyncio.create_subprocess_exec(
            *verify_cmd, stdout=PIPE, stderr=PIPE
        )
        verify_stdout, _ = await verify_process.communicate()

        if verify_process.returncode != 0 or not verify_stdout.decode().strip():
            print(
                f"  Error: Checkpoint {checkpoint} not found in new location after copy"
            )
            continue

        # Remove old checkpoint directory
        rm_cmd = ["aws", "s3", "rm", old_path, "--recursive"]
        rm_process = await asyncio.create_subprocess_exec(
            *rm_cmd, stdout=PIPE, stderr=PIPE
        )
        _, rm_stderr = await rm_process.communicate()

        if rm_process.returncode != 0:
            print(
                f"  Warning: Failed to remove old checkpoint {checkpoint}: {rm_stderr.decode()}"
            )
            print(
                "  Checkpoint was successfully copied to new location but old files remain"
            )
        else:
            print(f"  Successfully migrated checkpoint {checkpoint}")
            migrated_count += 1

    print(f"\nMigration complete. Successfully migrated {migrated_count} checkpoint(s)")
