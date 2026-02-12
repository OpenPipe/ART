#!/usr/bin/env python3
"""
Script to migrate model checkpoints in S3 from old to new structure.

Old structure: s3://bucket/prefix/project/models/model_name/0001/
New structure: s3://bucket/prefix/project/models/model_name/checkpoints/0001/

Usage:
    python scripts/migrate-s3-checkpoints.py --project myproject --model mymodel
    python scripts/migrate-s3-checkpoints.py --project myproject --model mymodel --dry-run
    python scripts/migrate-s3-checkpoints.py --project myproject --model mymodel --bucket custom-bucket --prefix custom-prefix
"""

import argparse
import asyncio
from pathlib import Path
import sys

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from art.utils.s3_checkpoint_utils import migrate_s3_checkpoints_to_new_structure


async def main():
    parser = argparse.ArgumentParser(
        description="Migrate model checkpoints in S3 from old to new structure"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project name",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--bucket",
        help="S3 bucket name (defaults to BACKUP_BUCKET env var)",
    )
    parser.add_argument(
        "--prefix",
        help="S3 prefix",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done without making changes",
    )

    args = parser.parse_args()

    await migrate_s3_checkpoints_to_new_structure(
        model_name=args.model,
        project=args.project,
        s3_bucket=args.bucket,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    asyncio.run(main())
