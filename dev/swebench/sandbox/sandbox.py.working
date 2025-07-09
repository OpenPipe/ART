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

    async def safe_exec(self, command: str, timeout: int, error_msg: str) -> str:
        """Execute a command and raise RuntimeError on failure with custom error message."""
        exit_code, output = await self.exec(command, timeout)
        if exit_code != 0:
            raise RuntimeError(f"{error_msg}: {output}")
        return output

    async def write_file(self, content: str, target_path: str, timeout: int) -> None:
        """Write content to a file using heredoc to handle special characters."""
        await self.safe_exec(
            f"cat > {target_path} << 'EOF'\n{content}\nEOF",
            timeout,
            f"Failed to write to {target_path}",
        )

    async def apply_patch(self, patch: str, timeout: int) -> None:
        # Write patch to file
        await self.write_file(patch, "/tmp/patch.txt", timeout)

        # Apply the patch in the /testbed directory
        await self.safe_exec(
            "cd /testbed && patch -p1 < /tmp/patch.txt",
            timeout,
            "Failed to apply patch",
        )

        # Clean up
        await self.exec("rm -f /tmp/patch.txt", timeout)

    async def run_tests(self, tests: list[str], timeout: int) -> tuple[int, int]:
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

        # Write test list
        await self.write_file("\n".join(tests), "/tmp/tests.txt", timeout)

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
        await self.write_file(pytest_script, "/tmp/run_pytest.py", timeout)

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

        # Clean up
        await self.exec(
            "rm -f /tmp/tests.txt /tmp/run_pytest.py",
            timeout,
        )

        # Return failed count (including errors) and passed count
        total_failed = failed_count + error_count
        return total_failed, passed_count
