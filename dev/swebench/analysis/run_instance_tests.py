#!/usr/bin/env python3
"""
Script to run tests on SWE-bench instances and log results to JSONL.
Runs up to 256 sandboxes concurrently using the Daytona provider.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict

import daytona_sdk
from dotenv import load_dotenv
from instances import Instance, as_instances_iter, get_filtered_swe_smith_instances_df
from sandbox.new import new_sandbox
from sandbox.sandbox import Provider

load_dotenv()

# Configuration
RESULTS_FILE = "instance_test_results.jsonl"
MAX_CONCURRENT_SANDBOXES = 128
PROVIDER: Provider = "daytona"


async def cleanup_all_sandboxes() -> None:
    """Delete all existing Daytona sandboxes to prevent disk quota issues."""
    async with daytona_sdk.AsyncDaytona() as daytona:
        sandboxes = await daytona.list()

        if sandboxes:
            print(f"Cleaning up {len(sandboxes)} existing Daytona sandbox(es)...")
            delete_tasks = [sandbox.delete() for sandbox in sandboxes]
            results = await asyncio.gather(*delete_tasks, return_exceptions=True)

            success_count = sum(1 for r in results if not isinstance(r, Exception))
            failure_count = sum(1 for r in results if isinstance(r, Exception))

            print(f"  Deleted: {success_count} sandbox(es)")
            if failure_count > 0:
                print(f"  Failed: {failure_count} sandbox(es)")


async def load_existing_results() -> tuple[set[str], set[str]]:
    """Load existing results and identify perfect passes to skip.

    Returns:
        - Set of instance IDs that passed perfectly (to skip)
        - Set of all instance IDs that have been attempted (for stats)
    """
    perfect_passes = set()
    all_instances = set()

    if not os.path.exists(RESULTS_FILE):
        return perfect_passes, all_instances

    try:
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    instance_id = result.get("instance_id")
                    if instance_id:
                        all_instances.add(instance_id)

                        # Check if this is a perfect pass
                        if (
                            not result.get("error")
                            and result.get("f2p_initial", {}).get("failed") == 0
                            and result.get("f2p_initial", {}).get("passed") is not None
                            and result.get("f2p_post_patch", {}).get("passed") == 0
                            and result.get("f2p_post_patch", {}).get("failed")
                            is not None
                            and result.get("p2p", {}).get("failed") == 0
                            and result.get("p2p", {}).get("passed") is not None
                        ):
                            perfect_passes.add(instance_id)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading existing results: {e}")

    return perfect_passes, all_instances


async def run_instance_tests(instance: Instance) -> Dict[str, Any]:
    """Run tests for a single instance and return results."""
    instance_id = instance["instance_id"]
    print(f"Starting tests for {instance_id}")

    result = {
        "instance_id": instance_id,
        "repo": instance["repo"],
        "timestamp": datetime.utcnow().isoformat(),
        "provider": PROVIDER,
        "f2p_initial": {"failed": None, "passed": None, "error": None},
        "f2p_post_patch": {"failed": None, "passed": None, "error": None},
        "p2p": {"failed": None, "passed": None, "error": None},
    }

    # Calculate dynamic timeout based on number of tests
    base_timeout = 120  # Base time for dependency installation
    per_test_time = 0.05  # Per-test time

    # Skip instances with extreme test counts
    if len(instance["PASS_TO_PASS"]) > 8000:
        result["error"] = (
            f"Skipped: {len(instance['PASS_TO_PASS'])} PASS_TO_PASS tests (system limits)"
        )
        return result

    fail_to_pass_timeout = int(
        base_timeout + len(instance["FAIL_TO_PASS"]) * per_test_time
    )
    pass_to_pass_timeout = int(
        base_timeout + len(instance["PASS_TO_PASS"]) * per_test_time
    )

    try:
        async with new_sandbox(
            image=instance["image_name"], provider=PROVIDER
        ) as sandbox:
            # Run initial FAIL_TO_PASS tests (should pass without patch)
            try:
                failed, passed = await sandbox.run_tests(
                    instance["FAIL_TO_PASS"], fail_to_pass_timeout
                )
                result["f2p_initial"]["failed"] = failed
                result["f2p_initial"]["passed"] = passed
            except Exception as e:
                result["f2p_initial"]["error"] = str(e)

            # Apply patch
            try:
                await sandbox.apply_patch(instance["patch"], 10)

                # Run FAIL_TO_PASS tests after patch (should fail)
                try:
                    failed, passed = await sandbox.run_tests(
                        instance["FAIL_TO_PASS"], fail_to_pass_timeout
                    )
                    result["f2p_post_patch"]["failed"] = failed
                    result["f2p_post_patch"]["passed"] = passed
                except Exception as e:
                    result["f2p_post_patch"]["error"] = str(e)

                # Run PASS_TO_PASS tests (should pass)
                try:
                    failed, passed = await sandbox.run_tests(
                        instance["PASS_TO_PASS"], pass_to_pass_timeout
                    )
                    result["p2p"]["failed"] = failed
                    result["p2p"]["passed"] = passed
                except Exception as e:
                    result["p2p"]["error"] = str(e)

            except Exception as e:
                result["error"] = f"Failed to apply patch: {e}"

    except Exception as e:
        result["error"] = f"Sandbox error: {e}"

    print(f"Completed tests for {instance_id}")
    return result


async def append_result(result: Dict[str, Any]) -> None:
    """Append a result to the JSONL file."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


async def process_instance(instance: Instance, semaphore: asyncio.Semaphore) -> None:
    """Process a single instance with semaphore for concurrency control."""
    async with semaphore:
        try:
            result = await run_instance_tests(instance)
            await append_result(result)
        except Exception as e:
            print(f"Error processing {instance['instance_id']}: {e}")
            error_result = {
                "instance_id": instance["instance_id"],
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            }
            await append_result(error_result)


async def main():
    """Main function to run tests on all instances."""
    # Clean up any existing sandboxes first
    await cleanup_all_sandboxes()

    # Load existing results
    perfect_passes, all_attempted = await load_existing_results()
    print(f"Found {len(perfect_passes)} perfect passes to skip")
    print(f"Found {len(all_attempted)} total attempted instances")

    # Get all instances
    instances_df = get_filtered_swe_smith_instances_df()
    all_instances = list(as_instances_iter(instances_df))

    # Filter out only perfect passes (we'll retry failures and errors)
    instances_to_run = [
        inst for inst in all_instances if inst["instance_id"] not in perfect_passes
    ]

    print(f"Total instances: {len(all_instances)}")
    print(f"Instances to run: {len(instances_to_run)}")

    if not instances_to_run:
        print("All instances have been processed!")
        return

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SANDBOXES)

    # Create tasks for all instances
    tasks = [process_instance(instance, semaphore) for instance in instances_to_run]

    # Run all tasks concurrently
    print(
        f"Starting concurrent execution with up to {MAX_CONCURRENT_SANDBOXES} sandboxes..."
    )
    await asyncio.gather(*tasks)

    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
