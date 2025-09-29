import pytest
from dotenv import load_dotenv

from ..instances import as_instances_iter, get_filtered_swe_smith_instances_df
from .new import new_sandbox
from .sandbox import Provider

load_dotenv()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["daytona", "modal"])
async def test_sandbox(provider: Provider) -> None:
    async with new_sandbox(image="python:3.10", provider=provider) as sandbox:
        code, stdout = await sandbox.exec("echo 'Hello, world!'", 10)
        assert code == 0
        assert stdout == "Hello, world!\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["daytona", "modal"])
@pytest.mark.parametrize("instance_idx", range(16))
async def test_run_tests(provider: Provider, instance_idx: int) -> None:
    instance = next(
        get_filtered_swe_smith_instances_df()
        .pipe(lambda df: df.tail(-instance_idx) if instance_idx > 0 else df)
        .pipe(as_instances_iter)
    )

    # Calculate dynamic timeout based on number of tests
    # Formula: base_timeout + num_tests * per_test_time
    base_timeout = 120  # Base time for dependency installation
    per_test_time = 0.05  # Per-test time (reduced since most tests are fast)

    # Skip instances with extreme test counts that may hit system limits
    if len(instance["PASS_TO_PASS"]) > 3000:
        pytest.skip(
            f"Skipping instance with {len(instance['PASS_TO_PASS'])} PASS_TO_PASS tests (system limits)"
        )

    fail_to_pass_timeout = int(
        base_timeout + len(instance["FAIL_TO_PASS"]) * per_test_time
    )
    pass_to_pass_timeout = int(
        base_timeout + len(instance["PASS_TO_PASS"]) * per_test_time
    )

    async with new_sandbox(image=instance["image_name"], provider=provider) as sandbox:
        failed, passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        assert failed == 0
        assert passed == len(instance["FAIL_TO_PASS"])
        await sandbox.apply_patch(instance["patch"], 10)
        failed, passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        assert failed == len(instance["FAIL_TO_PASS"])
        assert passed == 0
        failed, passed = await sandbox.run_tests(
            instance["PASS_TO_PASS"], pass_to_pass_timeout
        )
        assert failed == 0
        assert passed == len(instance["PASS_TO_PASS"])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["daytona", "modal"])
async def test_edit_anthropic(provider: Provider) -> None:
    """Test the edit_anthropic tool functionality through the Sandbox.edit method."""
    # Use the first instance to get a valid image
    instance = next(get_filtered_swe_smith_instances_df().pipe(as_instances_iter))

    async with new_sandbox(image=instance["image_name"], provider=provider) as sandbox:
        # Setup test file path
        test_file = "/tmp/test_file.py"

        # Test 1: Create a new file
        file_content = """def hello():
    print("Hello, World!")

def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
"""
        output = await sandbox.edit(
            command="create", path=test_file, file_text=file_content
        )
        assert "File created successfully at: /tmp/test_file.py" in output

        # Verify file was created
        code, output = await sandbox.exec(f"cat {test_file}", 10)
        assert code == 0
        assert "Hello, World!" in output

        # Test 2: View the entire file
        output = await sandbox.edit(command="view", path=test_file)
        assert "Here's the result of running `cat -n` on /tmp/test_file.py:" in output
        assert "def hello():" in output
        assert "def add(a, b):" in output
        assert "def multiply(x, y):" in output

        # Test 3: View a specific range
        output = await sandbox.edit(command="view", path=test_file, view_range=[2, 4])
        assert '2\t    print("Hello, World!")' in output
        assert "4\tdef add(a, b):" in output
        # Should not include line 1 or lines after 4
        assert "1\tdef hello():" not in output
        assert "5\t    return a + b" not in output

        # Test 4: String replace
        output = await sandbox.edit(
            command="str_replace",
            path=test_file,
            old_str='    print("Hello, World!")',
            new_str='    print("Hello, Python!")',
        )
        assert "The file /tmp/test_file.py has been edited" in output
        assert 'print("Hello, Python!")' in output

        # Verify replacement
        code, output = await sandbox.exec(f"cat {test_file}", 10)
        assert "Hello, Python!" in output
        assert "Hello, World!" not in output

        # Test 5: Insert new lines
        output = await sandbox.edit(
            command="insert",
            path=test_file,
            insert_line=8,
            new_str="\ndef subtract(a, b):\n    return a - b",
        )
        assert "The file /tmp/test_file.py has been edited" in output
        assert "def subtract(a, b):" in output
        assert "return a - b" in output

        # Verify insertion
        code, output = await sandbox.exec(f"cat {test_file}", 10)
        assert "subtract" in output

        # Test 6: Undo last edit
        output = await sandbox.edit(command="undo_edit", path=test_file)
        assert "Last edit to /tmp/test_file.py undone successfully" in output
        # The subtract function should be gone after undo
        assert "def subtract" not in output

        # Verify undo
        code, output = await sandbox.exec(f"cat {test_file}", 10)
        assert "subtract" not in output

        # Test 7: View a directory
        output = await sandbox.edit(command="view", path="/tmp")
        assert "Here's the files and directories up to 2 levels deep in /tmp" in output
        assert "/tmp/test_file.py" in output

        # Test 8: Test error cases
        # Try to create file that already exists
        with pytest.raises(RuntimeError):
            await sandbox.edit(
                command="create", path=test_file, file_text="This should fail"
            )

        # Try to replace non-existent string
        with pytest.raises(RuntimeError):
            await sandbox.edit(
                command="str_replace",
                path=test_file,
                old_str="This does not exist",
                new_str="Replacement",
            )

        # Try to view non-existent file
        with pytest.raises(RuntimeError):
            await sandbox.edit(command="view", path="/tmp/nonexistent.txt")

        # Test 9: Complex multiline replacement
        complex_old = """def add(a, b):
    return a + b"""
        complex_new = """def add(a, b):
    # Add two numbers
    result = a + b
    return result"""

        output = await sandbox.edit(
            command="str_replace",
            path=test_file,
            old_str=complex_old,
            new_str=complex_new,
        )
        assert "The file /tmp/test_file.py has been edited" in output
        assert "# Add two numbers" in output
        assert "result = a + b" in output
        assert "return result" in output

        # Verify complex replacement
        code, output = await sandbox.exec(f"cat {test_file}", 10)
        assert "# Add two numbers" in output
        assert "result = a + b" in output

        # Test 10: Empty string replacement (deletion)
        output = await sandbox.edit(
            command="str_replace",
            path=test_file,
            old_str="    # Add two numbers\n",
            new_str="",  # Empty string to delete the line
        )
        assert "The file /tmp/test_file.py has been edited" in output

        # Verify deletion
        code, output = await sandbox.exec(f"cat {test_file}", 10)
        assert "# Add two numbers" not in output
        assert "result = a + b" in output  # Other content remains

        # Test 11: Insert at line 0 (beginning of file)
        output = await sandbox.edit(
            command="insert",
            path=test_file,
            insert_line=0,
            new_str="#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n",
        )
        assert "The file /tmp/test_file.py has been edited" in output

        # Verify insertion at beginning
        code, output = await sandbox.exec(f"head -3 {test_file}", 10)
        assert "#!/usr/bin/env python3" in output
        assert "# -*- coding: utf-8 -*-" in output

        # Test 12: View with -1 as end line (view to end of file)
        output = await sandbox.edit(
            command="view",
            path=test_file,
            view_range=[1, -1],  # From line 1 to end of file
        )
        assert "#!/usr/bin/env python3" in output
        assert "return result" in output  # Should include the last line

        # Test 13: Multiple undo operations
        # First, make a change
        output = await sandbox.edit(
            command="str_replace",
            path=test_file,
            old_str="#!/usr/bin/env python3",
            new_str="#!/usr/bin/env python",
        )
        assert "The file /tmp/test_file.py has been edited" in output

        # Undo the change
        output = await sandbox.edit(command="undo_edit", path=test_file)
        assert "Last edit to /tmp/test_file.py undone successfully" in output

        # Undo again to remove the insert at line 0
        output = await sandbox.edit(command="undo_edit", path=test_file)
        assert "Last edit to /tmp/test_file.py undone successfully" in output

        # Verify we're back to the state before the insert
        code, output = await sandbox.exec(f"head -1 {test_file}", 10)
        assert "#!/usr/bin/env python3" not in output
        assert "def hello():" in output

        # Test 14: Additional error cases
        # Try to use non-absolute path
        with pytest.raises(RuntimeError, match="Path is not absolute"):
            await sandbox.edit(command="view", path="relative/path.txt")

        # Try to str_replace on a directory
        with pytest.raises(RuntimeError, match="Path is directory"):
            await sandbox.edit(
                command="str_replace", path="/tmp", old_str="something", new_str="else"
            )

        # Try to insert at invalid line number
        with pytest.raises(RuntimeError, match="Invalid.*insert_line.*parameter"):
            await sandbox.edit(
                command="insert",
                path=test_file,
                insert_line=1000,  # Way beyond file length
                new_str="This should fail",
            )

        # Test 15: Create file with parent directory that doesn't exist
        with pytest.raises(RuntimeError, match="Unknown error"):
            await sandbox.edit(
                command="create",
                path="/nonexistent/directory/file.txt",
                file_text="This should fail",
            )

        # Test 16: Test omitting new_str in str_replace (should default to empty string)
        # First create a test file with a comment
        test_file2 = "/tmp/test_delete.py"
        output = await sandbox.edit(
            command="create",
            path=test_file2,
            file_text="# This is a comment\nprint('hello')\n",
        )
        assert "File created successfully" in output

        # Now replace the comment with nothing (omit new_str)
        output = await sandbox.edit(
            command="str_replace",
            path=test_file2,
            old_str="# This is a comment\n",
            # new_str is omitted, should default to empty string
        )
        assert "The file /tmp/test_delete.py has been edited" in output

        # Verify the comment was deleted
        code, output = await sandbox.exec(f"cat {test_file2}", 10)
        assert "# This is a comment" not in output
        assert "print('hello')" in output
