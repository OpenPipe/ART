# Implementation Instructions for Sandbox.edit() Method

## Overview
The `edit` method needs to integrate the `edit_anthropic` tool (located at `/tools/edit_anthropic/bin/str_replace_editor`) into the Sandbox class to provide file editing capabilities within sandbox environments.

## Method Signature
```python
async def edit(
    self,
    command: str,
    path: str,
    file_text: Optional[str] = None,
    view_range: Optional[List[int]] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    timeout: int = 10
) -> None:
```

## Implementation Steps

### 1. Construct the Command
Build the command string to execute `str_replace_editor` with appropriate arguments:

```python
# Base command
cmd = f"cd /testbed && str_replace_editor {command} {path}"

# Add optional arguments based on command type
if command == "create" and file_text is not None:
    # Escape the file_text properly for shell execution
    cmd += f" --file_text {shlex.quote(file_text)}"
elif command == "view" and view_range is not None:
    cmd += f" --view_range {view_range[0]} {view_range[1]}"
elif command == "str_replace":
    if old_str is not None:
        cmd += f" --old_str {shlex.quote(old_str)}"
    if new_str is not None:
        cmd += f" --new_str {shlex.quote(new_str)}"
elif command == "insert":
    if insert_line is not None:
        cmd += f" --insert_line {insert_line}"
    if new_str is not None:
        cmd += f" --new_str {shlex.quote(new_str)}"
```

### 2. Handle Tool Availability
The `str_replace_editor` tool may not be available in the sandbox by default. You'll need to:

1. Check if the tool exists in the sandbox
2. If not, copy it from the host system or install it

Options:
- **Option A**: Copy the tool files into the sandbox during execution
- **Option B**: Ensure the tool is pre-installed in the Docker images
- **Option C**: Install the tool on-demand using the sandbox's package manager

### 3. Execute the Command
Use the sandbox's `exec` method to run the command:

```python
exit_code, output = await self.exec(cmd, timeout)
```

### 4. Handle Errors
The `str_replace_editor` tool uses different exit codes for different errors:
- Exit code 1: `file_text` required for create command
- Exit code 2: `old_str` required for str_replace command
- Exit code 3: `insert_line` required for insert command
- Exit code 4: `new_str` required for insert command
- Exit code 5: Unrecognized command
- Exit code 6: Path is not absolute
- Exit code 7: Path does not exist
- Exit code 8: File already exists (for create)
- Exit code 9: Path is directory (for non-view commands)
- Exit code 10-21: Various other validation errors

Convert these to RuntimeError exceptions with appropriate messages:

```python
if exit_code != 0:
    error_messages = {
        1: "Parameter 'file_text' is required for create command",
        2: "Parameter 'old_str' is required for str_replace command",
        # ... etc
    }
    error_msg = error_messages.get(exit_code, f"Command failed with exit code {exit_code}")
    if output:
        error_msg = f"{error_msg}: {output}"
    raise RuntimeError(error_msg)
```

### 5. Handle Output
For successful commands, the tool prints output to stdout. You may want to:
- Log the output for debugging
- Parse specific outputs for certain commands
- Return the output if needed (though current signature returns None)

## Alternative Implementation Approach

Instead of executing the external tool, you could reimplement the core logic directly in Python:

1. **For `create`**: Use `exec()` to write the file
2. **For `view`**: Use `exec()` with `cat -n` or similar
3. **For `str_replace`**: Read file, perform replacement, write back
4. **For `insert`**: Read file, insert at line, write back
5. **For `undo_edit`**: Maintain history in the Sandbox instance

This approach would be more complex but wouldn't require the external tool.

## Testing Considerations

The test in `sandbox/test.py::test_edit_anthropic` covers:
1. File creation
2. Full file viewing
3. Range viewing
4. String replacement
5. Line insertion
6. Undo functionality
7. Directory viewing
8. Error cases (existing file, non-existent string, non-existent file)
9. Complex multiline replacements

Ensure your implementation handles all these cases correctly.

## Example Implementation Skeleton

```python
async def edit(self, command: str, path: str, **kwargs) -> None:
    import shlex
    
    # Build command
    cmd = f"str_replace_editor {command} {shlex.quote(path)}"
    
    # Add arguments based on command type
    if command == "create":
        if kwargs.get("file_text") is None:
            raise RuntimeError("file_text is required for create command")
        cmd += f" --file_text {shlex.quote(kwargs['file_text'])}"
    # ... handle other commands
    
    # Execute
    exit_code, output = await self.exec(cmd, kwargs.get("timeout", 10))
    
    # Handle errors
    if exit_code != 0:
        raise RuntimeError(f"Edit command failed: {output}")
```

## Notes
- The `str_replace_editor` tool maintains state in a JSON file for undo functionality
- The tool uses the `registry` module for configuration
- Consider whether state should persist across sandbox sessions
- The tool supports various text encodings (utf-8, latin-1, etc.)