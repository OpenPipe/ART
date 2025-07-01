import asyncio
import base64
import daytona_sdk
import argparse
import sys
from dotenv import load_dotenv
import re

load_dotenv()

from instances import Instance, as_instances_iter, get_filtered_swe_smith_instances_df

instances = list(
    get_filtered_swe_smith_instances_df()
    .sample(fraction=1.0, shuffle=True, seed=42)
    .pipe(as_instances_iter)
)


async def write_file_chunked(
    sandbox, content: str, target_path: str, chunk_size: int = 1000
) -> None:
    """Write content to a file in chunks to avoid command length limits.

    Args:
        sandbox: The sandbox instance
        content: The content to write (will be base64 encoded)
        target_path: Path to write the file to
        chunk_size: Size of each chunk in characters
    """
    content_b64 = base64.b64encode(content.encode()).decode()

    # Remove existing file
    await sandbox.process.exec(f"rm -f {target_path}.b64", cwd="/testbed")

    # Write in chunks
    for i in range(0, len(content_b64), chunk_size):
        chunk = content_b64[i : i + chunk_size]
        await sandbox.process.exec(
            f"echo -n '{chunk}' >> {target_path}.b64", cwd="/testbed"
        )

    # Decode the file
    await sandbox.process.exec(
        f"base64 -d {target_path}.b64 > {target_path}", cwd="/testbed"
    )


def extract_missing_modules(output: str) -> list[str]:
    """Extract missing module names from pytest output.

    Handles various formats:
    - ModuleNotFoundError: No module named 'X'
    - E   ModuleNotFoundError: No module named 'X'
    - ImportError patterns

    Args:
        output: The pytest output to parse

    Returns:
        List of unique missing module names
    """
    missing_modules = []

    # Pattern for regular ModuleNotFoundError
    missing_modules.extend(
        re.findall(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", output)
    )

    # Pattern for collection phase errors (e.g., E   ModuleNotFoundError: No module named 'jwt')
    missing_modules.extend(
        re.findall(
            r"E\s+ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", output
        )
    )

    # Also catch simpler patterns without E prefix
    missing_modules.extend(re.findall(r"No module named ['\"]([^'\"]+)['\"]", output))

    return list(set(missing_modules))  # Remove duplicates


async def install_missing_module(sandbox, module: str, uv_cmd: str) -> bool:
    """Try to install a missing module, attempting various package name transformations.

    Args:
        sandbox: The sandbox instance
        module: The module name to install
        uv_cmd: The uv command with environment setup

    Returns:
        True if installation succeeded, False otherwise
    """
    # Try the module name as-is first
    install_res = await sandbox.process.exec(
        f"{uv_cmd} pip install -q {module} 2>&1", cwd="/testbed"
    )

    if "successfully installed" in install_res.result.lower():
        return True

    # If that fails, try common transformations
    if (
        "could not find" in install_res.result.lower()
        or "no matching distribution" in install_res.result.lower()
    ):
        # Common module name transformations
        alternatives = []

        # Special case for common packages
        if module == "jwt":
            alternatives.append("PyJWT")

        # Try with underscores replaced by hyphens
        if "_" in module:
            alternatives.append(module.replace("_", "-"))

        # Try with python- prefix
        if not module.startswith("python-"):
            alternatives.append(f"python-{module}")

        # Try with py prefix
        if not module.startswith("py"):
            alternatives.append(f"py{module}")

        for alt in alternatives:
            alt_res = await sandbox.process.exec(
                f"{uv_cmd} pip install -q {alt} 2>&1", cwd="/testbed"
            )
            if "successfully installed" in alt_res.result.lower():
                return True

    return False


def analyze_test_results(output: str, instance: Instance) -> dict:
    """Analyze pytest output and return structured results.

    Args:
        output: The pytest output
        instance: The instance being tested

    Returns:
        Dictionary with test counts and analysis
    """
    failed_count = output.count(" FAILED")
    passed_count = output.count(" PASSED")
    error_count = output.count(" ERROR")

    fail_to_pass_count = len(instance["FAIL_TO_PASS"])
    pass_to_pass_count = len(instance["PASS_TO_PASS"])

    total_issues = failed_count + error_count

    # Determine if results are as expected
    is_expected = (fail_to_pass_count > 0 and total_issues > 0) or (
        fail_to_pass_count == 0 and total_issues == 0
    )

    return {
        "failed": failed_count,
        "passed": passed_count,
        "errors": error_count,
        "fail_to_pass_count": fail_to_pass_count,
        "pass_to_pass_count": pass_to_pass_count,
        "total_issues": total_issues,
        "is_expected": is_expected,
    }


async def install_base_dependencies(sandbox, uv_cmd: str) -> None:
    """Install base dependencies for testing.

    Args:
        sandbox: The sandbox instance
        uv_cmd: The uv command with environment setup
    """
    # Install pytest first
    print("  Installing pytest...")
    await sandbox.process.exec(f"{uv_cmd} pip install -q pytest", cwd="/testbed")

    # Try to sync dependencies if pyproject.toml with dependencies exists
    pyproject_check = await sandbox.process.exec(
        "test -f pyproject.toml && grep -q dependencies pyproject.toml && echo 1 || echo 0",
        cwd="/testbed",
    )
    if pyproject_check.result.strip() == "1":
        print("  Syncing dependencies with uv...")
        # For projects with pyproject.toml, try installing directly
        await sandbox.process.exec(f"{uv_cmd} pip install -q -e .", cwd="/testbed")

    # Install from requirements files
    req_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        "test-requirements.txt",
        "dev-requirements.txt",
        "tests/requirements.txt",
    ]
    for req_file in req_files:
        check = await sandbox.process.exec(
            f"test -f {req_file} && echo 1", cwd="/testbed"
        )
        if check.result.strip() == "1":
            print(f"  Installing from {req_file}")
            await sandbox.process.exec(
                f"{uv_cmd} pip install -q -r {req_file}", cwd="/testbed"
            )

    # Install the package itself if not already done
    print("  Installing package...")
    install_result = await sandbox.process.exec(
        f"{uv_cmd} pip install -q -e . 2>&1", cwd="/testbed"
    )

    # Check if installation had issues (but don't fail - some packages might not be installable)
    if (
        "error" in install_result.result.lower()
        and "no module named" in install_result.result.lower()
    ):
        # Try without -e flag
        await sandbox.process.exec(f"{uv_cmd} pip install -q . 2>&1", cwd="/testbed")


async def run_tests(daytona: daytona_sdk.AsyncDaytona, instance: Instance) -> None:
    """Run tests for a SWE-bench instance.

    The patch in the instance data introduces a bug that needs to be fixed.
    FAIL_TO_PASS tests should fail after the patch is applied.
    PASS_TO_PASS tests should continue to pass after the patch.
    """
    sandbox = await daytona.create(
        daytona_sdk.CreateSandboxFromImageParams(image=instance["image_name"])
    )
    try:
        print(f"\n=== {instance['instance_id']} ===")

        # Apply patch to introduce the bug
        await write_file_chunked(sandbox, instance["patch"], "/tmp/patch.txt")
        await sandbox.process.exec("patch -p1 < /tmp/patch.txt", cwd="/testbed")
        print("Patch applied (bug introduced)")

        # Install dependencies and package
        print("Installing dependencies...")

        # 1. Install uv if not already present
        uv_check = await sandbox.process.exec("which uv", cwd="/testbed")
        if uv_check.exit_code != 0:
            print("  Installing uv...")
            await sandbox.process.exec(
                "curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet",
                cwd="/testbed",
            )

        # 2. Set up environment for uv commands
        # UV_SYSTEM_PYTHON=true uses system python instead of creating venv
        # Add .local/bin to PATH for uv
        uv_cmd = "UV_SYSTEM_PYTHON=true PATH=$HOME/.local/bin:$PATH uv"

        # 3. Install dependencies
        await install_base_dependencies(sandbox, uv_cmd)

        # Prepare and run tests
        tests = instance["FAIL_TO_PASS"] + instance["PASS_TO_PASS"]
        await write_file_chunked(sandbox, "\n".join(tests), "/tmp/tests.txt")

        # 4. Try running tests and install missing dependencies if needed
        print("\nRunning tests...")
        max_retries = 5
        for attempt in range(max_retries):
            result = await sandbox.process.exec(
                "cat /tmp/tests.txt | xargs -d '\n' python -m pytest -v -o addopts= --tb=short",
                cwd="/testbed",
            )

            # Check for import errors
            if "ModuleNotFoundError" in result.result or "ImportError" in result.result:
                # Extract missing module names from both test execution and collection errors
                missing_modules = extract_missing_modules(result.result)

                if missing_modules and attempt < max_retries - 1:
                    print(f"  Missing modules detected: {', '.join(missing_modules)}")
                    print(f"  Installing missing dependencies...")

                    # Try to install the missing modules
                    for module in missing_modules:
                        await install_missing_module(sandbox, module, uv_cmd)

                    # Retry the tests
                    print(f"  Retrying tests (attempt {attempt + 2}/{max_retries})...")
                    continue

            # No more import errors or max retries reached
            break

        # Analyze results
        print(f"\nTest Results:")
        print(f"Exit code: {result.exit_code}")

        output = result.result
        results = analyze_test_results(output, instance)

        print(f"Failed: {results['failed']}")
        print(f"Passed: {results['passed']}")
        print(f"Errors: {results['errors']}")

        # Show expectations and summary
        print(f"\nExpectations:")
        print(f"FAIL_TO_PASS tests ({results['fail_to_pass_count']}): Should fail")
        print(f"PASS_TO_PASS tests ({results['pass_to_pass_count']}): Should pass")

        if results["is_expected"]:
            print("✓ Tests are failing as expected")
        else:
            print("⚠️  Unexpected test results")

        # Show sample failures
        if results["total_issues"] > 0:
            print("\nSample failures/errors:")
            lines = output.split("\n")
            failure_lines = [
                line for line in lines if "FAILED" in line or "ERROR" in line
            ][:5]
            for line in failure_lines:
                if line.strip():
                    print(f"  {line.strip()}")

            # If we have collection errors, show more detail
            if results["errors"] > 0 and "ERROR collecting" in output:
                error_details = [
                    line
                    for line in lines
                    if "ModuleNotFoundError" in line or "ImportError" in line
                ][:3]
                if error_details:
                    print("\nImport errors detected:")
                    for line in error_details:
                        print(f"  {line.strip()}")

        print(f"\n{'✅' if results['is_expected'] else '⚠️ '} Ready for agent")

    finally:
        await sandbox.delete()


async def test_instances(instance_indices: list[int]) -> None:
    """Test specified instances"""
    async with daytona_sdk.AsyncDaytona() as daytona:
        print(f"Testing {len(instance_indices)} instance(s)...")
        print("=" * 60)

        for idx in instance_indices:
            if 0 <= idx < len(instances):
                await run_tests(daytona, instances[idx])
            else:
                print(
                    f"\nError: Instance index {idx} is out of range (0-{len(instances)-1})"
                )
            print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench tests on specified instances"
    )
    parser.add_argument(
        "indices", nargs="*", type=int, help="Instance indices to test (default: 0 1 2)"
    )
    parser.add_argument("--all", action="store_true", help="Test all instances")
    parser.add_argument(
        "--list", action="store_true", help="List all available instances"
    )

    args = parser.parse_args()

    if args.list:
        print("Available instances:")
        for i, instance in enumerate(instances):
            print(f"{i}: {instance['instance_id']}")
        return

    if args.all:
        indices = list(range(len(instances)))
    elif args.indices:
        indices = args.indices
    else:
        indices = [0, 1, 2]  # Default to first 3

    asyncio.run(test_instances(indices))


if __name__ == "__main__":
    main()
