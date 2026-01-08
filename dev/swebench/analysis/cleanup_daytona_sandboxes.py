#!/usr/bin/env python3
"""
Script to delete all Daytona sandboxes.
Useful for cleaning up when disk quota is exceeded.
"""

import asyncio

import daytona_sdk
from dotenv import load_dotenv

load_dotenv()


async def cleanup_all_sandboxes():
    """Delete all existing Daytona sandboxes."""
    async with daytona_sdk.AsyncDaytona() as daytona:
        # List all sandboxes
        sandboxes = await daytona.list()

        if not sandboxes:
            print("No Daytona sandboxes found.")
            return

        print(f"Found {len(sandboxes)} Daytona sandbox(es) to delete...")

        # Delete all sandboxes in parallel
        delete_tasks = [sandbox.delete() for sandbox in sandboxes]
        results = await asyncio.gather(*delete_tasks, return_exceptions=True)

        # Count successes and failures
        success_count = 0
        failure_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  ❌ Failed to delete sandbox {i + 1}: {result}")
                failure_count += 1
            else:
                success_count += 1

        print("\nSummary:")
        print(f"  ✅ Successfully deleted: {success_count}")
        if failure_count > 0:
            print(f"  ❌ Failed to delete: {failure_count}")

        print("\nCleanup completed!")


if __name__ == "__main__":
    asyncio.run(cleanup_all_sandboxes())
