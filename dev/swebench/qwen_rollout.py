import asyncio
import json
import traceback

import openai
from dotenv import load_dotenv
from instances import as_instances_iter, get_filtered_swe_smith_instances_df
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
)
from sandbox import Sandbox, new_sandbox

load_dotenv()

instance_prompt = """
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

MAX_OUTPUT_LENGTH = 1000


async def qwen_rollout(
    client: openai.AsyncOpenAI,
    problem_statement: str,
    sandbox: Sandbox,
    max_steps: int = 30,
) -> str:
    """
    Roll out a Qwen agent on a given problem statement in an isolated sandbox.

    Args:
        problem_statement: The problem statement to solve.
        sandbox: The sandbox to use for the rollout.
        max_steps: Maximum number of steps before terminating.

    Returns:
        The contents of the assistant's final response.
    """
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can interact with a computer to solve tasks.",
        },
        {
            "role": "user",
            "content": instance_prompt.format(
                problem_statement=problem_statement, working_dir="/testbed"
            ),
        },
    ]
    for step in range(max_steps):
        print(f"\n[Step {step + 1}/{max_steps}]")
        try:
            response = await client.chat.completions.create(
                model="willcb/Qwen3-32B",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096,
            )
        except Exception as e:
            print(f"Error getting completion: {type(e).__name__}: {str(e)}")
            raise
        assistant_message = response.choices[0].message
        assistant_msg_dict: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": assistant_message.content,
        }
        if assistant_message.tool_calls:
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            for tool_call in assistant_message.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            assistant_msg_dict["tool_calls"] = tool_calls
        messages.append(assistant_msg_dict)
        if assistant_message.content:
            print(f"Assistant: {assistant_message.content[:200]}...")
        if not assistant_message.tool_calls:
            print("No tool calls - completing rollout")
            return assistant_message.content or "No response generated"
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in tool arguments: {str(e)}"
                print(f"Error: {error_msg}")
                error_response: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_msg,
                }
                messages.append(error_response)
                continue
            print(f"Executing {tool_name}: {json.dumps(tool_args, indent=2)}")
            try:
                if tool_name == "bash":
                    command = tool_args.get("command", "")
                    exit_code, output = await sandbox.exec(command, timeout=60)
                    if len(output) > MAX_OUTPUT_LENGTH:
                        output = (
                            output[: MAX_OUTPUT_LENGTH // 2]
                            + "\n\n[... truncated ...]\n\n"
                            + output[-MAX_OUTPUT_LENGTH // 2 :]
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
                    if len(output) > MAX_OUTPUT_LENGTH:
                        output = (
                            output[: MAX_OUTPUT_LENGTH // 2]
                            + "\n\n[... truncated ...]\n\n"
                            + output[-MAX_OUTPUT_LENGTH // 2 :]
                        )
                    tool_result = output
                else:
                    tool_result = f"Unknown tool: {tool_name}"
                print(f"Tool result: {tool_result[:200]}...")
                tool_response: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
                messages.append(tool_response)
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {type(e).__name__}: {str(e)}"
                print(f"Error: {error_msg}")
                exec_error_response: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": error_msg,
                }
                messages.append(exec_error_response)
    print(f"\nExhausted {max_steps} steps without completion")
    raise Exception(f"Agent did not complete task within {max_steps} steps")


async def test_qwen_rollout(instance_idx: int) -> None:
    client = openai.AsyncOpenAI(api_key="default", base_url="http://localhost:8000/v1")

    instance = next(
        get_filtered_swe_smith_instances_df()
        .pipe(lambda df: df.tail(-instance_idx) if instance_idx > 0 else df)
        .pipe(as_instances_iter)
    )

    print(f"\n{'=' * 60}")
    print(f"Testing instance: {instance.get('instance_id', 'Unknown')}")
    print(f"Repository: {instance.get('repo', 'Unknown')}")
    print(f"{'=' * 60}\n")

    # Calculate dynamic timeout based on number of tests
    # Formula: base_timeout + num_tests * per_test_time
    base_timeout = 120  # Base time for dependency installation
    per_test_time = 0.05  # Per-test time (reduced since most tests are fast)

    fail_to_pass_timeout = int(
        base_timeout + len(instance["FAIL_TO_PASS"]) * per_test_time
    )
    pass_to_pass_timeout = int(
        base_timeout + len(instance["PASS_TO_PASS"]) * per_test_time
    )

    async with new_sandbox(image=instance["image_name"], provider="daytona") as sandbox:
        print("1. Running initial FAIL_TO_PASS tests (should all pass)...")
        failed, passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        assert failed == 0
        assert passed == len(instance["FAIL_TO_PASS"])
        print(f"   ✓ All {passed} tests passed\n")

        print("2. Applying patch to break tests...")
        await sandbox.apply_patch(instance["patch"], 10)

        print("3. Running FAIL_TO_PASS tests after patch (should all fail)...")
        failed, passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        assert failed == len(instance["FAIL_TO_PASS"])
        assert passed == 0
        print(f"   ✓ All {failed} tests now failing as expected\n")

        print("4. Running PASS_TO_PASS tests (should all pass)...")
        failed, passed = await sandbox.run_tests(
            instance["PASS_TO_PASS"], pass_to_pass_timeout
        )
        assert failed == 0
        assert passed == len(instance["PASS_TO_PASS"])
        print(f"   ✓ All {passed} tests passed\n")

        print("5. Running Qwen rollout to fix the issue...")
        print(f"\nProblem statement:\n{instance['problem_statement'][:500]}...\n")

        try:
            response = await qwen_rollout(
                client, instance["problem_statement"], sandbox
            )
            print(f"\nQwen's final response: {response[:200]}...")
        except Exception as e:
            print(f"\nQwen rollout failed: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

        print("\n6. Running final tests...")
        f2p_failed, f2p_passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        p2p_failed, p2p_passed = await sandbox.run_tests(
            instance["PASS_TO_PASS"], pass_to_pass_timeout
        )

        print("\nFinal results:")
        print(
            f"FAIL_TO_PASS: {f2p_passed} passed, {f2p_failed} failed (out of {len(instance['FAIL_TO_PASS'])})"
        )
        print(
            f"PASS_TO_PASS: {p2p_passed} passed, {p2p_failed} failed (out of {len(instance['PASS_TO_PASS'])})"
        )

        if (
            f2p_failed == 0
            and f2p_passed == len(instance["FAIL_TO_PASS"])
            and p2p_failed == 0
            and p2p_passed == len(instance["PASS_TO_PASS"])
        ):
            print("\n✅ SUCCESS: All tests passing!")
        else:
            print("\n❌ FAILURE: Some tests still failing")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_qwen_rollout(instance_idx=5))
