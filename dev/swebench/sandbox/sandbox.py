from abc import ABC, abstractmethod
from typing import Literal

Provider = Literal["daytona", "modal"]


class Sandbox(ABC):
    """
    Base class for all sandboxes.

    Provides a common interface for all sandboxes, as well as shared logic and functionality.
    """

    provider: Provider

    @abstractmethod
    async def exec(self, command: str, timeout: int) -> tuple[int, str]:
        raise NotImplementedError

    async def apply_patch(self, patch: str, timeout: int) -> None:
        import base64

        # Convert patch to base64 to handle special characters and large size
        patch_b64 = base64.b64encode(patch.encode()).decode()

        # Remove existing patch file if it exists
        await self.exec("rm -f /tmp/patch.txt.b64", timeout)

        # Write patch in chunks to avoid command length limits
        chunk_size = 1000
        for i in range(0, len(patch_b64), chunk_size):
            chunk = patch_b64[i : i + chunk_size]
            await self.exec(f"echo -n '{chunk}' >> /tmp/patch.txt.b64", timeout)

        # Decode the patch file
        await self.exec("base64 -d /tmp/patch.txt.b64 > /tmp/patch.txt", timeout)

        # Apply the patch in the /testbed directory
        exit_code, output = await self.exec(
            "cd /testbed && patch -p1 < /tmp/patch.txt", timeout
        )

        # Clean up
        await self.exec("rm -f /tmp/patch.txt /tmp/patch.txt.b64", timeout)

        if exit_code != 0:
            raise RuntimeError(f"Failed to apply patch: {output}")

    async def run_tests(self, tests: list[str], timeout: int) -> tuple[int, int]:
        import base64
        import re

        # First, ensure uv is installed
        exit_code, _ = await self.exec("which uv", timeout)
        if exit_code != 0:
            await self.exec(
                "curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet", timeout
            )

        # Set up uv environment
        uv_cmd = "UV_SYSTEM_PYTHON=true PATH=$HOME/.local/bin:$PATH uv"

        # Install pytest first
        await self.exec(f"{uv_cmd} pip install -q pytest", timeout)

        # Try to install dependencies
        # Check for pyproject.toml with dependencies
        exit_code, output = await self.exec(
            "test -f pyproject.toml && grep -q dependencies pyproject.toml && echo 1 || echo 0",
            timeout,
        )
        if "1" in output:
            await self.exec(f"{uv_cmd} pip install -q -e .", timeout)

        # Install from common requirements files
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "test-requirements.txt",
        ]
        for req_file in req_files:
            exit_code, output = await self.exec(
                f"test -f {req_file} && echo 1", timeout
            )
            if "1" in output:
                await self.exec(f"{uv_cmd} pip install -q -r {req_file}", timeout)

        # Install the package itself
        await self.exec(f"{uv_cmd} pip install -q -e .", timeout)

        # Filter tests to remove obvious non-test files (like in daytona.py)
        filtered_tests = []
        for test in tests:
            # Skip documentation files
            if test.endswith((".md", ".rst", ".txt")) and "::" in test:
                continue
            filtered_tests.append(test)

        # Use the same chunked writing approach as daytona.py
        test_list_content = "\n".join(filtered_tests)

        # Helper function to write content in chunks (from daytona.py)
        async def write_file_chunked(
            content: str, target_path: str, chunk_size: int = 1000
        ) -> None:
            content_b64 = base64.b64encode(content.encode()).decode()

            # Remove existing file
            await self.exec(f"rm -f {target_path}.b64", timeout)

            # Write in chunks
            for i in range(0, len(content_b64), chunk_size):
                chunk = content_b64[i : i + chunk_size]
                await self.exec(f"echo -n '{chunk}' >> {target_path}.b64", timeout)

            # Decode the file
            await self.exec(f"base64 -d {target_path}.b64 > {target_path}", timeout)

        # Write test list
        await write_file_chunked(test_list_content, "/tmp/tests.txt")

        # Create pytest runner script that properly handles both regular tests and doctests
        pytest_script = """
import sys
import os
import json

# Add testbed to path
sys.path.insert(0, '/testbed')

# Read all test paths
with open('/tmp/tests.txt', 'r') as f:
    tests = [line.strip() for line in f if line.strip()]

print(f"DEBUG: Total tests to run: {len(tests)}", file=sys.stderr)
print(f"DEBUG: First few tests: {tests[:3]}", file=sys.stderr)

# Use pytest.main() which doesn't have command line length limits
import pytest

# For doctests, we need to collect the Python files and run with --doctest-modules
# Separate doctest paths from regular test paths
doctest_files = set()
regular_tests = []

for test in tests:
    if "::" in test:
        file_path, test_name = test.split("::", 1)
        # Check if this is a doctest path (module path with dots, not starting with test_)
        if "." in test_name and not test_name.startswith("test_") and file_path.endswith(".py"):
            # For doctests, we should run them individually, not the whole file
            # This ensures we only run the specific doctests in our test list
            regular_tests.append(test)
        else:
            regular_tests.append(test)
    else:
        regular_tests.append(test)

# Run all tests together
all_args = ['-v', '-o', 'addopts=', '--tb=short', '--no-header']

# Always add --doctest-modules flag to handle any doctests
all_args.append('--doctest-modules')

# Add all tests
all_args.extend(tests)

print(f"DEBUG: Running pytest with args: {all_args[:10]}...", file=sys.stderr)
exit_code = pytest.main(all_args)
print(f"DEBUG: Pytest exit code: {exit_code}", file=sys.stderr)

sys.exit(exit_code)
"""

        # Write pytest script
        await write_file_chunked(pytest_script, "/tmp/run_pytest.py")

        # Run the tests with retry logic for missing dependencies
        max_retries = 5
        for attempt in range(max_retries):
            exit_code, output = await self.exec(
                "cd /testbed && python /tmp/run_pytest.py 2>&1", timeout
            )

            # Check for missing dependencies and try to install them
            if "ModuleNotFoundError" in output or "ImportError" in output:
                # Extract missing module names
                missing_modules = []
                missing_modules.extend(
                    re.findall(
                        r"ModuleNotFoundError: No module named [']([^']+)[']", output
                    )
                )
                missing_modules.extend(
                    re.findall(r"No module named [']([^']+)[']", output)
                )
                # Also catch from E prefix lines in pytest output
                missing_modules.extend(
                    re.findall(
                        r"E\s+ModuleNotFoundError: No module named [']([^']+)[']",
                        output,
                    )
                )

                # Remove duplicates
                missing_modules = list(set(missing_modules))

                if missing_modules and attempt < max_retries - 1:
                    # Try to install missing modules
                    for module in missing_modules:
                        await self.exec(f"{uv_cmd} pip install -q {module}", timeout)
                        # Also try common alternatives
                        if module == "jwt":
                            await self.exec(f"{uv_cmd} pip install -q PyJWT", timeout)
                        elif "_" in module:
                            await self.exec(
                                f"{uv_cmd} pip install -q {module.replace('_', '-')}",
                                timeout,
                            )

                    # Retry the test
                    continue

            # No more dependency issues or max retries reached
            break

        # Parse results - look for FAILED and PASSED (with leading space like in pytest output)
        failed_count = output.count(" FAILED")
        passed_count = output.count(" PASSED")
        error_count = output.count(" ERROR")

        # Count collection errors specifically (ERROR: not found)
        collection_errors = output.count("ERROR: not found:")

        # Clean up
        await self.exec(
            "rm -f /tmp/tests.txt /tmp/tests.txt.b64 /tmp/run_pytest.py /tmp/run_pytest.py.b64",
            timeout,
        )

        # Return failed count (including errors and collection errors) and passed count
        total_failed = failed_count + error_count + collection_errors
        return total_failed, passed_count
