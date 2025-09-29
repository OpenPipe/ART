from abc import ABC, abstractmethod
from typing import List, Literal, Optional

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

        # Try to install the project itself if it has a setup.py or pyproject.toml
        # This will install all project dependencies
        setup_exists = await self.exec(
            "test -f /testbed/setup.py && echo exists", timeout
        )
        pyproject_exists = await self.exec(
            "test -f /testbed/pyproject.toml && echo exists", timeout
        )
        if (
            setup_exists[1].strip() == "exists"
            or pyproject_exists[1].strip() == "exists"
        ):
            await self.exec(
                f"cd /testbed && {uv_cmd} pip install -q -e . 2>/dev/null", timeout
            )

        # Write test list in chunks to avoid command length limits
        # First, clear any existing file
        await self.exec("rm -f /tmp/tests.txt", timeout)

        # Write tests in chunks
        chunk_size = 50  # Write 50 tests at a time
        for i in range(0, len(tests), chunk_size):
            chunk = tests[i : i + chunk_size]
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
args = ['-v', '-o', 'addopts=', '--tb=short', '--no-header'] + tests
exit_code = pytest.main(args)
sys.exit(exit_code)
"""
        exit_code, output = await self.exec(
            f"cat > /tmp/run_pytest.py << 'EOF'\n{pytest_script}\nEOF", timeout
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to write pytest script: {output}")

        # Run the tests with retry logic for missing dependencies
        max_retries = 20
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
                        # Handle special cases where import name differs from package name
                        package_name = module
                        if module == "OpenSSL":
                            package_name = "pyOpenSSL"
                        elif module == "yaml":
                            package_name = "pyyaml"
                        elif module == "cv2":
                            package_name = "opencv-python"

                        await self.exec(
                            f"{uv_cmd} pip install -q {package_name}", timeout
                        )

                    # Retry the test
                    continue

            # No more dependency issues or max retries reached
            break

        # Parse results - look for FAILED and PASSED (with leading space like in pytest output)
        # Note: We only count test results, not pytest framework messages
        # Test results appear in format like "tests/test_file.py::test_name FAILED"
        failed_count = output.count(" FAILED")
        passed_count = output.count(" PASSED")

        # Handle edge case: if pytest exits with code 4 (collection errors) and no tests ran,
        # we should count the requested tests as failures since they couldn't be executed
        if exit_code == 4 and failed_count == 0 and passed_count == 0:
            # Check if there were collection errors preventing tests from running
            if (
                "ERROR collecting" in output
                or "ImportError" in output
                or "ModuleNotFoundError" in output
            ):
                # Count all requested tests as failures since they couldn't run
                failed_count = len(tests)

        return failed_count, passed_count

    async def edit(
        self,
        command: str,
        path: str,
        file_text: Optional[str] = None,
        view_range: Optional[List[int]] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        insert_line: Optional[int] = None,
        timeout: int = 10,
    ) -> str:
        """
        Execute the edit_anthropic tool to view, create, and edit files in the sandbox.

        Args:
            command: The command to run. Options: "view", "create", "str_replace", "insert", "undo_edit"
            path: Absolute path to file or directory
            file_text: Required for "create" command - content of the file to be created
            view_range: Optional for "view" command - line number range to display [start, end]
            old_str: Required for "str_replace" command - string to replace
            new_str: Optional for "str_replace", required for "insert" - new string to add
            insert_line: Required for "insert" command - line number after which to insert
            timeout: Command execution timeout in seconds

        Returns:
            str: The output from the edit command

        Raises:
            RuntimeError: If the command fails or required parameters are missing
        """
        import shlex

        # First, ensure the str_replace_editor tool is available in the sandbox
        # Check if it already exists
        exit_code, _ = await self.exec(
            "test -f /tmp/str_replace_editor && echo exists", timeout
        )
        if exit_code != 0:
            # Copy the tool and its dependencies into the sandbox
            host_tool_path = "/home/brad/art/dev/swebench/tools/edit_anthropic/bin/str_replace_editor"

            # Read the tool from host
            with open(host_tool_path, "r") as f:
                tool_content = f.read()

            # Patch the tool to persist file history after modifications
            # We need to patch specific locations to avoid breaking indentation

            # Patch in create_file method
            tool_content = tool_content.replace(
                """        self.write_file(path, file_text)
        self._file_history[path].append(file_text)
        print(f"File created successfully at: {path}")""",
                """        self.write_file(path, file_text)
        self._file_history[path].append(file_text)
        self._file_history = self._file_history  # Trigger setter to save
        print(f"File created successfully at: {path}")""",
            )

            # Patch in str_replace method (for file_content)
            tool_content = tool_content.replace(
                """        # Save the content to history
        self._file_history[path].append(file_content)""",
                """        # Save the content to history
        self._file_history[path].append(file_content)
        self._file_history = self._file_history  # Trigger setter to save""",
            )

            # Patch in insert method (for file_text)
            tool_content = tool_content.replace(
                """        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)""",
                """        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)
        self._file_history = self._file_history  # Trigger setter to save""",
            )

            # Patch in undo_edit method
            tool_content = tool_content.replace(
                """        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)""",
                """        old_text = self._file_history[path].pop()
        self._file_history = self._file_history  # Trigger setter to save
        self.write_file(path, old_text)""",
            )

            # Also need to handle Path objects being used as keys
            # The setter converts to JSON which requires string keys
            tool_content = tool_content.replace(
                'REGISTRY["file_history"] = json.dumps(value)',
                """# Convert Path keys to strings for JSON serialization
        str_value = {str(k): v for k, v in value.items()}
        REGISTRY["file_history"] = json.dumps(str_value)
        self._file_history_cache = None  # Clear cache so it reloads next time""",
            )

            # Most importantly, we need to cache the file history instead of recreating it
            # Replace the getter to cache the defaultdict
            tool_content = tool_content.replace(
                """    @property
    def _file_history(self):
        return defaultdict(list, json.loads(REGISTRY.get("file_history", "{}")))""",
                """    _file_history_cache = None
    
    @property
    def _file_history(self):
        if self._file_history_cache is None:
            # Load from registry and create defaultdict that converts Path keys to strings
            data = json.loads(REGISTRY.get("file_history", "{}"))
            from collections import defaultdict
            
            class PathKeyDefaultDict(defaultdict):
                def __getitem__(self, key):
                    if hasattr(key, '__fspath__'):  # It's a Path object
                        key = str(key)
                    return super().__getitem__(key)
                
                def __setitem__(self, key, value):
                    if hasattr(key, '__fspath__'):  # It's a Path object
                        key = str(key)
                    super().__setitem__(key, value)
                    
                def __contains__(self, key):
                    if hasattr(key, '__fspath__'):  # It's a Path object
                        key = str(key)
                    return super().__contains__(key)
            
            self._file_history_cache = PathKeyDefaultDict(list, data)
        return self._file_history_cache""",
            )

            # Write the patched tool to sandbox
            exit_code, output = await self.exec(
                f"cat > /tmp/str_replace_editor << 'EOF'\n{tool_content}\nEOF", timeout
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"Failed to copy str_replace_editor to sandbox: {output}"
                )

            # Make it executable
            exit_code, output = await self.exec(
                "chmod +x /tmp/str_replace_editor", timeout
            )
            if exit_code != 0:
                raise RuntimeError(
                    f"Failed to make str_replace_editor executable: {output}"
                )

            # Create a registry.py module that the tool imports with persistent state
            # The tool uses Path objects as keys, but JSON requires string keys
            registry_content = """
import json
import os
from collections import defaultdict
from pathlib import Path

class PersistentList(list):
    \"\"\"A list that notifies when modified.\"\"\"
    def __init__(self, parent, key, *args):
        super().__init__(*args)
        self.parent = parent
        self.key = key
    
    def append(self, item):
        super().append(item)
        self.parent._on_modify()
    
    def pop(self, *args):
        result = super().pop(*args)
        self.parent._on_modify()
        return result

class PathDefaultDict(defaultdict):
    \"\"\"A defaultdict that converts Path keys to strings and persists changes.\"\"\"
    def __init__(self, registry, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = registry
        
    def __getitem__(self, key):
        if isinstance(key, Path):
            key = str(key)
        value = super().__getitem__(key)
        # Wrap lists to track modifications
        if isinstance(value, list) and not isinstance(value, PersistentList):
            value = PersistentList(self, key, value)
            super().__setitem__(key, value)
        return value
    
    def __setitem__(self, key, value):
        if isinstance(key, Path):
            key = str(key)
        super().__setitem__(key, value)
        self._on_modify()
    
    def _on_modify(self):
        # Save state when modified
        self.registry._save_file_history(self)

class Registry:
    def __init__(self):
        self.state_file = '/tmp/edit_state.json'
        self._load_state()
        self._file_history = None
    
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
            except:
                self.state = {}
        else:
            self.state = {}
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
    
    def _save_file_history(self, history_dict):
        # Convert to regular dict for JSON serialization
        self.state['file_history'] = json.dumps(dict(history_dict))
        self._save_state()
    
    def get(self, key, default=None):
        # Special handling for file_history to return a PathDefaultDict
        if key == 'file_history':
            if self._file_history is None:
                history_data = json.loads(self.state.get(key, '{}'))
                # Create PathDefaultDict from loaded data
                self._file_history = PathDefaultDict(self, list)
                self._file_history.update(history_data)
            # Return as JSON string as expected by the tool
            return json.dumps(dict(self._file_history))
        return self.state.get(key, default)
    
    def __setitem__(self, key, value):
        self.state[key] = value
        self._save_state()
    
    def __getitem__(self, key):
        return self.state[key]

registry = Registry()
"""
            exit_code, output = await self.exec(
                f"cat > /tmp/registry.py << 'EOF'\n{registry_content}\nEOF", timeout
            )
            if exit_code != 0:
                raise RuntimeError(f"Failed to create registry module: {output}")

        # Build the command
        cmd = f"cd /tmp && python /tmp/str_replace_editor {command} {shlex.quote(path)}"

        # Add optional arguments based on command type
        if command == "create":
            if file_text is None:
                raise RuntimeError(
                    "Parameter 'file_text' is required for create command"
                )
            cmd += f" --file_text {shlex.quote(file_text)}"
        elif command == "view":
            if view_range is not None:
                cmd += f" --view_range {view_range[0]} {view_range[1]}"
        elif command == "str_replace":
            if old_str is None:
                raise RuntimeError(
                    "Parameter 'old_str' is required for str_replace command"
                )
            cmd += f" --old_str {shlex.quote(old_str)}"
            if new_str is not None:
                cmd += f" --new_str {shlex.quote(new_str)}"
        elif command == "insert":
            if insert_line is None:
                raise RuntimeError(
                    "Parameter 'insert_line' is required for insert command"
                )
            if new_str is None:
                raise RuntimeError("Parameter 'new_str' is required for insert command")
            cmd += f" --insert_line {insert_line}"
            cmd += f" --new_str {shlex.quote(new_str)}"
        elif command == "undo_edit":
            # No additional parameters needed
            pass
        else:
            raise RuntimeError(f"Unrecognized command: {command}")

        # Execute the command
        exit_code, output = await self.exec(cmd, timeout)

        # Handle errors based on exit codes
        if exit_code != 0:
            error_messages = {
                1: "Parameter 'file_text' is required for create command",
                2: "Parameter 'old_str' is required for str_replace command",
                3: "Parameter 'insert_line' is required for insert command",
                4: "Parameter 'new_str' is required for insert command",
                5: "Unrecognized command",
                6: "Path is not absolute",
                7: "Path does not exist",
                8: "File already exists",
                9: "Path is directory (for non-view commands)",
                10: "Invalid view range",
                11: "View range out of bounds",
                12: "String not found in file",
                13: "Multiple matches found - string is not unique",
                14: "Invalid line number for insert",
                15: "No edits to undo",
                16: "Failed to read file",
                17: "Failed to write file",
                18: "File encoding error",
                19: "Invalid JSON state file",
                20: "Permission denied",
                21: "Unknown error",
            }
            error_msg = error_messages.get(
                exit_code, f"Command failed with exit code {exit_code}"
            )
            if output:
                error_msg = f"{error_msg}: {output}"
            raise RuntimeError(error_msg)

        # Return the output from the command
        return output
