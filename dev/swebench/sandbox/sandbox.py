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
        # Write patch to file using heredoc
        exit_code, output = await self.exec(
            f"cat > /tmp/patch.txt << 'EOF'\n{patch}\nEOF", timeout
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to write patch: {output}")

        # Apply the patch in the /testbed directory
        exit_code, output = await self.exec(
            "cd /testbed && patch -p1 < /tmp/patch.txt", timeout
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to apply patch: {output}")

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

        # Write test list in chunks to avoid command length limits
        # First, clear any existing file
        await self.exec("rm -f /tmp/tests.txt", timeout)
        
        # Write tests in chunks
        chunk_size = 50  # Write 50 tests at a time
        for i in range(0, len(tests), chunk_size):
            chunk = tests[i:i + chunk_size]
            test_chunk = "\n".join(chunk)
            exit_code, output = await self.exec(
                f"cat >> /tmp/tests.txt << 'EOF'\n{test_chunk}\nEOF", timeout
            )
            if exit_code != 0:
                raise RuntimeError(f"Failed to write test chunk: {output}")

        # Create a Python script to run pytest with proper handling of special characters
        pytest_script = """
import sys
sys.path.insert(0, '/testbed')

with open('/tmp/tests.txt', 'r') as f:
    tests = [line.strip() for line in f if line.strip()]

import pytest
args = ['-v', '-o', 'addopts=', '--tb=short', '--no-header', '--doctest-modules'] + tests
exit_code = pytest.main(args)
sys.exit(exit_code)
"""
        exit_code, output = await self.exec(
            f"cat > /tmp/run_pytest.py << 'EOF'\n{pytest_script}\nEOF", timeout
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to write pytest script: {output}")

        # Run the tests with retry logic for missing dependencies
        max_retries = 5
        for attempt in range(max_retries):
            exit_code, output = await self.exec(
                "cd /testbed && python /tmp/run_pytest.py 2>&1", timeout
            )

            # Check for missing dependencies and try to install them
            if "ModuleNotFoundError" in output or "ImportError" in output:
                # Extract missing module names using a single pattern
                missing_modules = list(
                    set(re.findall(r"No module named [']([^']+)[']", output))
                )

                if missing_modules and attempt < max_retries - 1:
                    # Try to install missing modules
                    for module in missing_modules:
                        await self.exec(f"{uv_cmd} pip install -q {module}", timeout)

                    # Retry the test
                    continue

            # No more dependency issues or max retries reached
            break

        # Parse results - look for FAILED and PASSED (with leading space like in pytest output)
        failed_count = output.count(" FAILED") + output.count(" ERROR")
        passed_count = output.count(" PASSED")

        return failed_count, passed_count
