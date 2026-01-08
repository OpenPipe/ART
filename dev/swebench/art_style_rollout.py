import json
import traceback

from instances import Instance
from openai.types.chat import (
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel
from sandbox import Sandbox

import art


class ARTModelConfig(BaseModel):
    """Configuration for ART-style models."""

    max_steps: int = 40
    max_output_length: int = 1000
    temperature: float = 1.0
    system_prompt: str = (
        "You are a helpful assistant that can interact with a computer to solve tasks."
    )
    instance_prompt_template: str = """
<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a python code repository in the directory {working_dir}. Consider the following PR description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long.
""".strip()


# Tool definitions
tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the sandbox and get the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Multi-function text editor for viewing and editing files",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "view",
                            "create",
                            "str_replace",
                            "insert",
                            "undo_edit",
                        ],
                        "description": "The editor command to execute",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Line range to view [start, end] (1-indexed, inclusive)",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "String to replace (for str_replace command)",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "New string to insert (for str_replace and create commands)",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Line number to insert at (for insert command)",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
]


async def art_style_rollout(
    model: art.Model[ARTModelConfig],
    instance: Instance,
    sandbox: Sandbox,
    reward_power: float = 1.0,
) -> art.Trajectory:
    """
    Execute an ART-style rollout for solving a SWE-bench instance.

    Args:
        model: The ART model to use for generation
        instance: The SWE-bench instance to solve
        sandbox: The sandbox environment for code execution
        reward_power: Power to apply to progress metric in reward calculation

    Returns:
        art.Trajectory with messages, choices, reward, and metrics
    """
    # Initialize trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": model.config.system_prompt,
            },
            {
                "role": "user",
                "content": model.config.instance_prompt_template.format(
                    problem_statement=instance["problem_statement"],
                    working_dir="/testbed",
                ),
            },
        ],
        tools=tools,
        reward=0.0,
        metrics={
            "resolved": False,
            "progress": 0.0,
            "maintenance": 0.0,
            "regression": 0.0,
            "steps_taken": 0,
        },
        metadata={
            "instance_id": instance["instance_id"],
        },
    )

    # Get OpenAI client
    client = model.openai_client()

    # Main interaction loop
    for step in range(model.config.max_steps):
        trajectory.logs.append(f"Step {step + 1}/{model.config.max_steps}")

        try:
            # Get model completion
            response = await client.chat.completions.create(
                model=model.get_inference_name(),
                messages=trajectory.messages(),
                tools=tools,
                tool_choice="auto",
                temperature=model.config.temperature,
            )
        except Exception as e:
            error_msg = f"Error getting completion: {type(e).__name__}: {str(e)}"
            trajectory.logs.append(error_msg)
            print(error_msg)
            break

        # Extract assistant message
        assistant_message = response.choices[0].message

        # Store the choice in trajectory
        trajectory.messages_and_choices.append(response.choices[0])

        # Log assistant response
        if assistant_message.content:
            trajectory.logs.append(f"Assistant: {assistant_message.content[:200]}...")

        # Check if we're done (no tool calls)
        if not assistant_message.tool_calls:
            trajectory.logs.append("No tool calls - completing rollout")
            trajectory.metrics["steps_taken"] = step + 1
            break

        # Execute tool calls
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            # Parse tool arguments
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in tool arguments: {str(e)}"
                trajectory.logs.append(f"Error: {error_msg}")
                print(f"Error: {error_msg}")

                error_response: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_msg,
                }
                trajectory.messages_and_choices.append(error_response)
                continue

            trajectory.logs.append(
                f"Executing {tool_name}: {json.dumps(tool_args, indent=2)}"
            )

            # Execute the tool
            try:
                if tool_name == "bash":
                    command = tool_args.get("command", "")
                    exit_code, output = await sandbox.exec(command, timeout=60)

                    # Truncate output if needed
                    if len(output) > model.config.max_output_length:
                        output = (
                            output[: model.config.max_output_length // 2]
                            + "\n\n[... truncated ...]\n\n"
                            + output[-model.config.max_output_length // 2 :]
                        )

                    tool_result = f"Exit code: {exit_code}\nOutput:\n{output}"

                elif tool_name == "str_replace_editor":
                    output = await sandbox.edit(
                        command=tool_args["command"],
                        path=tool_args.get("path"),
                        file_text=(
                            tool_args.get("new_str")
                            if tool_args["command"] == "create"
                            else None
                        ),
                        view_range=tool_args.get("view_range"),
                        old_str=tool_args.get("old_str"),
                        new_str=(
                            tool_args.get("new_str")
                            if tool_args["command"] != "create"
                            else None
                        ),
                        insert_line=tool_args.get("insert_line"),
                    )

                    # Truncate output if needed
                    if len(output) > model.config.max_output_length:
                        output = (
                            output[: model.config.max_output_length // 2]
                            + "\n\n[... truncated ...]\n\n"
                            + output[-model.config.max_output_length // 2 :]
                        )

                    tool_result = output

                else:
                    tool_result = f"Unknown tool: {tool_name}"

                trajectory.logs.append(f"Tool result: {tool_result[:200]}...")

                # Add tool response to messages
                tool_response: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
                trajectory.messages_and_choices.append(tool_response)

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {type(e).__name__}: {str(e)}"
                trajectory.logs.append(f"Error: {error_msg}")
                print(f"Error: {error_msg}")

                exec_error_response: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_msg,
                }
                trajectory.messages_and_choices.append(exec_error_response)

    # Run final evaluation
    trajectory.logs.append("Running final evaluation...")

    try:
        # Run FAIL_TO_PASS tests
        failed_f2p, passed_f2p = await sandbox.run_tests(
            instance["FAIL_TO_PASS"],
            timeout=120 + int(len(instance["FAIL_TO_PASS"]) * 0.05),
        )

        # Run PASS_TO_PASS tests
        failed_p2p, passed_p2p = await sandbox.run_tests(
            instance["PASS_TO_PASS"],
            timeout=120 + int(len(instance["PASS_TO_PASS"]) * 0.05),
        )

        # Calculate metrics
        total_f2p = len(instance["FAIL_TO_PASS"])
        total_p2p = len(instance["PASS_TO_PASS"])

        progress = passed_f2p / total_f2p if total_f2p > 0 else 0.0
        failure = failed_f2p / total_f2p if total_f2p > 0 else 0.0
        maintenance = passed_p2p / total_p2p if total_p2p > 0 else 1.0
        regression = failed_p2p / total_p2p if total_p2p > 0 else 0.0

        # Reconcile metrics pessimistically
        progress = min(progress, 1 - failure)
        maintenance = min(maintenance, 1 - regression)

        # Check if resolved
        resolved = (
            failed_f2p == 0
            and passed_f2p == total_f2p
            and failed_p2p == 0
            and passed_p2p == total_p2p
        )

        # Calculate reward
        trajectory.reward = (
            0.2 * maintenance + 0.3 * (progress**reward_power) + 0.5 * float(resolved)
        )

        # Update metrics
        trajectory.metrics.update(
            {
                "progress": progress,
                "maintenance": maintenance,
                "regression": regression,
                "resolved": resolved,
                "failed_f2p": failed_f2p,
                "passed_f2p": passed_f2p,
                "failed_p2p": failed_p2p,
                "passed_p2p": passed_p2p,
            }
        )

        trajectory.logs.append(
            f"Evaluation complete - Progress: {progress:.2f}, "
            f"Maintenance: {maintenance:.2f}, Resolved: {resolved}"
        )

    except Exception as e:
        error_msg = f"Error during evaluation: {type(e).__name__}: {str(e)}"
        trajectory.logs.append(error_msg)
        print(error_msg)
        traceback.print_exc()

    # DEBUG
    # Serialize trajectory to file
    import os

    trajectory_path = f"./trajectories/{instance['instance_id']}.json"
    os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
    with open(trajectory_path, "w") as f:
        f.write(trajectory.model_dump_json(indent=2))
    trajectory.logs.append(f"Trajectory saved to {trajectory_path}")

    return trajectory
