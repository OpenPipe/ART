import argparse
import asyncio
import base64
import re
from typing import Literal

import daytona_sdk
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

from instances import (  # noqa: E402
    Instance,
    as_instances_iter,
    get_filtered_swe_smith_instances_df,
)

instances = list(
    get_filtered_swe_smith_instances_df()
    .sample(fraction=1.0, shuffle=True, seed=42)
    .pipe(as_instances_iter)
)


Logging = Literal["as-you-go", "on-exit", "on-error", "none"]


class Logger:
    def __init__(self, logging: Logging) -> None:
        self.logging = logging
        self.messages = []

    def log(self, message: str) -> None:
        if self.logging == "as-you-go":
            print(message)
        elif self.logging == "on-exit":
            self.messages.append(message)
        elif self.logging == "on-error":
            self.messages.append(message)
        elif self.logging == "none":
            pass

    def print_queued_messages(self) -> None:
        for message in self.messages:
            print(message)


async def write_file_chunked(
    sandbox: daytona_sdk.AsyncSandbox,
    content: str,
    target_path: str,
    chunk_size: int = 1000,
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


async def install_missing_module(
    sandbox: daytona_sdk.AsyncSandbox, module: str, uv_cmd: str
) -> bool:
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
    # CRITICAL: We need EXACT matches, not just "some" failures
    # - The number of failed/error tests should equal the number of FAIL_TO_PASS tests
    # - The number of passed tests should equal the number of PASS_TO_PASS tests
    is_expected = (
        total_issues == fail_to_pass_count and passed_count == pass_to_pass_count
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


async def install_base_dependencies(
    sandbox: daytona_sdk.AsyncSandbox, uv_cmd: str, logger: Logger
) -> None:
    """Install base dependencies for testing.

    Args:
        sandbox: The sandbox instance
        uv_cmd: The uv command with environment setup
    """
    # Install pytest first
    logger.log("  Installing pytest...")
    await sandbox.process.exec(f"{uv_cmd} pip install -q pytest", cwd="/testbed")

    # Try to sync dependencies if pyproject.toml with dependencies exists
    pyproject_check = await sandbox.process.exec(
        "test -f pyproject.toml && grep -q dependencies pyproject.toml && echo 1 || echo 0",
        cwd="/testbed",
    )
    if pyproject_check.result.strip() == "1":
        logger.log("  Syncing dependencies with uv...")
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
            logger.log(f"  Installing from {req_file}")
            await sandbox.process.exec(
                f"{uv_cmd} pip install -q -r {req_file}", cwd="/testbed"
            )

    # Install the package itself if not already done
    logger.log("  Installing package...")
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


async def run_tests(
    daytona: daytona_sdk.AsyncDaytona,
    instance: Instance,
    logging: Logging = "as-you-go",
    index: int = -1,
) -> None:
    """Run tests for a SWE-bench instance.

    The patch in the instance data introduces a bug that needs to be fixed.
    FAIL_TO_PASS tests should fail after the patch is applied.
    PASS_TO_PASS tests should continue to pass after the patch.
    """
    logger = Logger(logging)
    sandbox = await daytona.create(
        daytona_sdk.CreateSandboxFromImageParams(image=instance["image_name"])
    )
    try:
        logger.log(f"\n=== [{index}] {instance['instance_id']} ===")

        # Apply patch to introduce the bug
        await write_file_chunked(sandbox, instance["patch"], "/tmp/patch.txt")
        await sandbox.process.exec("patch -p1 < /tmp/patch.txt", cwd="/testbed")
        logger.log("Patch applied (bug introduced)")

        # Install dependencies and package
        logger.log("Installing dependencies...")

        # 1. Install uv if not already present
        uv_check = await sandbox.process.exec("which uv", cwd="/testbed")
        if uv_check.exit_code != 0:
            logger.log("  Installing uv...")
            await sandbox.process.exec(
                "curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet",
                cwd="/testbed",
            )

        # 2. Set up environment for uv commands
        # UV_SYSTEM_PYTHON=true uses system python instead of creating venv
        # Add .local/bin to PATH for uv
        uv_cmd = "UV_SYSTEM_PYTHON=true PATH=$HOME/.local/bin:$PATH uv"

        # 3. Install dependencies
        await install_base_dependencies(sandbox, uv_cmd, logger)

        # Prepare and run tests
        tests = instance["FAIL_TO_PASS"] + instance["PASS_TO_PASS"]

        # Simple filtering - just skip obvious non-test files
        filtered_tests = []
        for test in tests:
            # Skip documentation files
            if test.endswith((".md", ".rst", ".txt")) and "::" in test:
                logger.log(f"  Skipping documentation file: {test}")
                continue
            filtered_tests.append(test)

        tests = filtered_tests
        await write_file_chunked(sandbox, "\n".join(tests), "/tmp/tests.txt")

        # 4. Try running tests and install missing dependencies if needed
        logger.log("\nRunning tests...")
        num_tests = len(tests)  # Store for later use in retry logic  # noqa: F841
        max_retries = 5
        for attempt in range(max_retries):
            # Create a Python script that uses pytest's Python API to avoid command line limits
            pytest_script = """
import sys
import os

# Add testbed to path
sys.path.insert(0, '/testbed')

# Read all test paths
with open('/tmp/tests.txt', 'r') as f:
    tests = [line.strip() for line in f if line.strip()]

print(f"DEBUG: Total tests to run: {len(tests)}", file=sys.stderr)
print(f"DEBUG: First few tests: {tests[:3]}", file=sys.stderr)

# Use pytest.main() which doesn't have command line length limits
import pytest

# Prepare arguments for pytest
args = ['-v', '-o', 'addopts=', '--tb=short', '--no-header'] + tests

print(f"DEBUG: Running pytest with {len(tests)} tests...", file=sys.stderr)
exit_code = pytest.main(args)
print(f"DEBUG: Pytest exit code: {exit_code}", file=sys.stderr)

sys.exit(exit_code)
"""
            await write_file_chunked(sandbox, pytest_script, "/tmp/run_pytest.py")
            result = await sandbox.process.exec(
                "python /tmp/run_pytest.py 2>&1",
                cwd="/testbed",
            )

            # Capture any debug output
            if "DEBUG:" in result.result:
                logger.log("Debug output from pytest runner:")
                for line in result.result.split("\n"):
                    if "DEBUG:" in line:
                        logger.log(f"  {line}")

            # Check for missing pytest plugins
            if "Missing required plugins:" in result.result:
                plugin_line = [
                    line
                    for line in result.result.split("\n")
                    if "Missing required plugins:" in line
                ]
                if plugin_line and attempt < max_retries - 1:
                    # Extract plugin names
                    plugins_str = (
                        plugin_line[0].split("Missing required plugins:")[1].strip()
                    )
                    plugins = [p.strip() for p in plugins_str.split(",")]
                    logger.log(
                        f"  Missing pytest plugins detected: {', '.join(plugins)}"
                    )
                    logger.log("  Installing missing plugins...")

                    # Install the plugins
                    for plugin in plugins:
                        await sandbox.process.exec(
                            f"{uv_cmd} pip install -q {plugin}", cwd="/testbed"
                        )

                    logger.log(
                        f"  Retrying tests (attempt {attempt + 2}/{max_retries})..."
                    )
                    continue

            # Check for import errors
            if "ModuleNotFoundError" in result.result or "ImportError" in result.result:
                # Extract missing module names from both test execution and collection errors
                missing_modules = extract_missing_modules(result.result)

                # Skip retries for known problematic modules that require compilation
                unfixable_modules = ["torch", "tensorflow", "pandas._libs"]
                if any(module in str(missing_modules) for module in unfixable_modules):
                    logger.log(
                        f"  Skipping retry - requires compiled dependencies: {missing_modules}"
                    )
                    break

                if missing_modules and attempt < max_retries - 1:
                    logger.log(
                        f"  Missing modules detected: {', '.join(missing_modules)}"
                    )
                    logger.log("  Installing missing dependencies...")

                    # Try to install the missing modules
                    for module in missing_modules:
                        await install_missing_module(sandbox, module, uv_cmd)

                    # Retry the tests
                    logger.log(
                        f"  Retrying tests (attempt {attempt + 2}/{max_retries})..."
                    )
                    continue

            # No more import errors or max retries reached
            break

        # Analyze results
        logger.log("\nTest Results:")
        logger.log(f"Exit code: {result.exit_code}")

        output = result.result

        # Simple conftest error handling - just check if package[extras] is suggested
        if "ImportError while loading conftest" in output and "pip install" in output:
            # Look for package[extras] pattern
            for line in output.split("\n"):
                if "pip install" in line and "[" in line and "]" in line:
                    import re

                    match = re.search(r"pip install\s+`?(\S+\[[^\]]+\])`?", line)
                    if match:
                        package_with_extras = match.group(1)
                        logger.log(
                            f"  Conftest loading error - installing {package_with_extras}"
                        )
                        install_res = await sandbox.process.exec(
                            f"{uv_cmd} pip install -q '{package_with_extras}'",
                            cwd="/testbed",
                        )
                        if install_res.exit_code == 0:
                            logger.log(
                                "  Retrying tests after installing optional dependencies..."
                            )
                            result = await sandbox.process.exec(
                                "python /tmp/run_pytest.py",
                                cwd="/testbed",
                            )
                            output = result.result
                        break

        results = analyze_test_results(output, instance)

        logger.log(f"Failed: {results['failed']}")
        logger.log(f"Passed: {results['passed']}")
        logger.log(f"Errors: {results['errors']}")

        # Show expectations and summary
        logger.log("\nExpectations:")
        logger.log(f"FAIL_TO_PASS tests ({results['fail_to_pass_count']}): Should fail")
        logger.log(f"PASS_TO_PASS tests ({results['pass_to_pass_count']}): Should pass")

        # Check if any tests actually ran (diagnostic addition)
        total_tests_ran = results["failed"] + results["passed"] + results["errors"]
        if total_tests_ran == 0 and (
            results["fail_to_pass_count"] > 0 or results["pass_to_pass_count"] > 0
        ):
            # No tests ran at all
            logger.log("⚠️  WARNING: No tests were executed!")
            logger.log("\nPossible issues:")
            logger.log("  - Test paths may be incorrect")
            logger.log("  - Tests may require special setup or configuration")
            logger.log("  - The test collection mechanism may be incompatible")

            # Additional diagnostics to understand why tests didn't run
            if output:
                # Check for common pytest collection errors
                if "collected 0 items" in output:
                    logger.log(
                        "\nPytest collected 0 items - checking test file existence..."
                    )
                    # Check if test files exist
                    sample_tests = (
                        instance["FAIL_TO_PASS"] + instance["PASS_TO_PASS"]
                    )[:3]
                    for test_path in sample_tests:
                        test_file = (
                            test_path.split("::")[0] if "::" in test_path else test_path
                        )
                        check_result = await sandbox.process.exec(
                            f"test -f {test_file} && echo 'exists' || echo 'not found'",
                            cwd="/testbed",
                        )
                        logger.log(f"  {test_file}: {check_result.result.strip()}")

                # Look for other pytest errors
                if "INTERNALERROR>" in output:
                    logger.log("\nPytest internal error detected")
                    internal_error_lines = [
                        line for line in output.split("\n") if "INTERNALERROR>" in line
                    ][:3]
                    for line in internal_error_lines:
                        logger.log(f"  {line.strip()}")

                # Check for configuration issues
                if (
                    "pytest.ini" in output
                    or "setup.cfg" in output
                    or "pyproject.toml" in output
                ):
                    logger.log("\nPossible pytest configuration issue detected")

            # Add this instance to potential blacklist candidates
            error_msg = f"No tests executed for {instance['instance_id']}\n"
            error_msg += f"Expected to run {results['fail_to_pass_count'] + results['pass_to_pass_count']} tests\n"
            error_msg += "This instance may need to be blacklisted if the issue cannot be resolved."
            assert False, error_msg
        elif results["is_expected"]:
            logger.log("✓ Tests are failing as expected")
        else:
            logger.log("⚠️  Unexpected test results")

        # Show sample failures
        if results["total_issues"] > 0:
            logger.log("\nSample failures/errors:")
            lines = output.split("\n")
            failure_lines = [
                line for line in lines if "FAILED" in line or "ERROR" in line
            ][:5]
            for line in failure_lines:
                if line.strip():
                    logger.log(f"  {line.strip()}")

            # If we have collection errors, show more detail
            if results["errors"] > 0 and "ERROR collecting" in output:
                error_details = [
                    line
                    for line in lines
                    if "ModuleNotFoundError" in line or "ImportError" in line
                ][:3]
                if error_details:
                    logger.log("\nImport errors detected:")
                    for line in error_details:
                        logger.log(f"  {line.strip()}")

        if not results["is_expected"]:
            # Simplified error message
            fail_diff = results["total_issues"] - results["fail_to_pass_count"]
            pass_diff = results["passed"] - results["pass_to_pass_count"]

            error_msg = "Test count mismatch!\n"
            error_msg += f"Expected: {results['fail_to_pass_count']} failures, {results['pass_to_pass_count']} passes\n"
            error_msg += f"Actual:   {results['total_issues']} failures/errors, {results['passed']} passes\n"

            if fail_diff != 0:
                error_msg += f"Failure mismatch: {'+' if fail_diff > 0 else ''}{fail_diff} tests\n"
            if pass_diff != 0:
                error_msg += (
                    f"Pass mismatch: {'+' if pass_diff > 0 else ''}{pass_diff} tests\n"
                )

            assert False, error_msg

        logger.log("✅ Ready for agent")

    except Exception as e:
        if logging == "on-error":
            logger.print_queued_messages()
        raise e
    finally:
        await sandbox.delete()
        if logging == "on-exit":
            logger.print_queued_messages()


async def test_instances(
    instance_indices: list[int],
    logging: Logging = "as-you-go",
    parallel: bool = False,
    allow_exceptions: bool = False,
    use_pbar: bool = False,
) -> None:
    """Test specified instances"""
    async with daytona_sdk.AsyncDaytona() as daytona:
        if sandboxes := await daytona.list():
            print(f"Deleting {len(sandboxes)} running sandboxes...")
            await asyncio.gather(*[sandbox.delete() for sandbox in sandboxes])
        print(f"Testing {len(instance_indices)} instance(s)...")
        print("=" * 60)

        if use_pbar:
            pbar = tqdm(
                total=len(instance_indices),
                desc="instances",
                postfix={"indices": f"{min(instance_indices)}-{max(instance_indices)}"},
            )
        else:
            pbar = None

        num_exceptions = 0
        try:
            coros = [
                run_tests(daytona, instances[idx], logging, idx)
                for idx in instance_indices
            ]
            for awaitable in asyncio.as_completed(coros) if parallel else coros:
                try:
                    await awaitable
                except Exception as e:
                    if allow_exceptions:
                        if logging != "none":
                            print(e if str(e) else type(e))
                        if logging == "on-error":
                            print("\n" + "=" * 60)
                        num_exceptions += 1
                    else:
                        raise e
                finally:
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix({"exceptions": num_exceptions})
                    if logging == "as-you-go" or logging == "on-exit":
                        print("\n" + "=" * 60)
        finally:
            if pbar:
                pbar.close()


def parse_indices(index_args: list[str]) -> list[int]:
    """Parse index arguments which can be individual numbers or ranges.

    Args:
        index_args: List of strings like ["0", "5-10", "15"]

    Returns:
        List of individual indices

    Examples:
        >>> parse_indices(["0", "5-10", "15"])
        [0, 5, 6, 7, 8, 9, 10, 15]
    """
    indices = []
    for arg in index_args:
        if "-" in arg:
            # Parse range
            parts = arg.split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    if start <= end:
                        indices.extend(range(start, end + 1))
                    else:
                        raise ValueError(f"Invalid range: {arg} (start > end)")
                except ValueError as e:
                    raise ValueError(f"Invalid range format: {arg}") from e
            else:
                raise ValueError(f"Invalid range format: {arg}")
        else:
            # Parse individual index
            try:
                indices.append(int(arg))
            except ValueError:
                raise ValueError(f"Invalid index: {arg}")

    # Remove duplicates and sort
    return sorted(set(indices))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SWE-bench tests on specified instances"
    )
    parser.add_argument(
        "indices",
        nargs="*",
        type=str,
        help="Instance indices to test (e.g., '0 5-10 15', default: 0 1 2)",
    )
    parser.add_argument("--all", action="store_true", help="Test all instances")
    parser.add_argument(
        "--list", action="store_true", help="List all available instances"
    )
    parser.add_argument(
        "--logging",
        choices=["as-you-go", "on-exit", "on-error", "none"],
        default="as-you-go",
        help="Logging mode (default: as-you-go)",
    )
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--allow-exceptions",
        action="store_true",
        help="Print exceptions instead of raising them",
    )
    parser.add_argument("--use-pbar", action="store_true", help="Use progress bar")

    args = parser.parse_args()

    if args.list:
        print("Available instances:")
        for i, instance in enumerate(instances):
            print(f"{i}: {instance['instance_id']}")
        return

    if args.all:
        indices = list(range(len(instances)))
    elif args.indices:
        indices = parse_indices(args.indices)
    else:
        indices = [0, 1, 2]  # Default to first 3

    asyncio.run(
        test_instances(
            indices,
            logging=args.logging,
            parallel=args.parallel,
            allow_exceptions=args.allow_exceptions,
            use_pbar=args.use_pbar,
        )
    )


if __name__ == "__main__":
    main()
