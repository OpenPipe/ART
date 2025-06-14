import art
from langfuse.decorators import langfuse_context
from pathlib import Path
from pydantic import SecretStr
from sweagent.agent.agents import DefaultAgentConfig, TemplateConfig
from sweagent.agent.history_processors import (
    CacheControlHistoryProcessor,
    DefaultHistoryProcessor,
)
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.run.run_single import RunSingleConfig
from sweagent.tools.bundle import Bundle
from sweagent.tools.parsing import FunctionCallingParser, XMLFunctionCallingParser
from sweagent.tools.tools import ToolConfig
from swerex.deployment.config import ModalDeploymentConfig
from typing import Any, TYPE_CHECKING

from instances import Instance

if TYPE_CHECKING:
    # Avoid circular import
    from rollout import ModelConfig

SYSTEM_TEMPLATE = """
You are a helpful assistant that can interact with a computer to solve tasks.
""".strip()

XML_SYSTEM_TEMPLATE = """
You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>

You have access to the following functions:

---- BEGIN FUNCTION #1: bash ----
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----

---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`

Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
Allowed values: [`view`, `create`, `str_replace`, `insert`, `undo_edit`]
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #3 ----


If you choose to call a function ONLY reply in the following format with NO suffix:

Provide any reasoning for the function call here.
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Always provide reasoning for your function call in natural language BEFORE the function call (not after)
</IMPORTANT>
""".strip()

INSTANCE_TEMPLATE = """
<uploaded_files>
{{working_dir}}
</uploaded_files>
I've uploaded a python code repository in the directory {{working_dir}}. Consider the following PR description:

<pr_description>
{{problem_statement}}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {{working_dir}} directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long.
""".strip()

NEXT_STEP_TEMPLATE = """
OBSERVATION:
{{observation}}
""".strip()

NEXT_STEP_NO_OUTPUT_TEMPLATE = """
Your command ran successfully and did not produce any output.
""".strip()

SUBMIT_REVIEW_MESSAGE = """
Thank you for your work on this issue. Please carefully follow the steps below to help review your changes.

1. If you made any changes to your code after running the reproduction script, please run the reproduction script again.
If the reproduction script is failing, please revisit your changes and make sure they are correct.
If you have already removed your reproduction script, please ignore this step.
2. Remove your reproduction script (if you haven't done so already).
3. If you have modified any TEST files, please revert them to the state they had before you started fixing the issue.
You can do this with `git checkout -- /path/to/test/file.py`. Use below <diff> to find the files you need to revert.
4. Run the submit command again to confirm.

Here is a list of all of your changes:

<diff>
{{diff}}
</diff>
""".strip()


def get_config(
    model: "art.Model[ModelConfig]",
    instance: Instance,
    completion_kwargs: dict[str, Any] | None = None,
) -> RunSingleConfig:
    if completion_kwargs is None:
        completion_kwargs = {}
    completion_kwargs.update(model.config.completion_kwargs.copy())
    if model.trainable and model.config.xml_function_calling:
        completion_kwargs["stop"] = "</function>"
        completion_kwargs["include_stop_str_in_output"] = True
        completion_kwargs["timeout"] = 60.0 * 10
    completion_kwargs["metadata"] = {
        "trace_id": langfuse_context.get_current_trace_id(),
        "parent_observation_id": langfuse_context.get_current_observation_id(),
    }
    return RunSingleConfig(
        env=EnvironmentConfig(
            deployment=ModalDeploymentConfig(
                image=instance["image_name"],
                deployment_timeout=60.0 * 60 * 2,
            ),
            repo=PreExistingRepoConfig(
                repo_name="testbed", base_commit=instance["base_commit"]
            ),
        ),
        agent=DefaultAgentConfig(
            templates=TemplateConfig(
                system_template=(
                    XML_SYSTEM_TEMPLATE
                    if model.config.xml_function_calling
                    else SYSTEM_TEMPLATE
                )
                + model.config.system_prompt_suffix,
                instance_template=INSTANCE_TEMPLATE,
                next_step_template=NEXT_STEP_TEMPLATE,
                next_step_no_output_template=NEXT_STEP_NO_OUTPUT_TEMPLATE,
            ),
            tools=ToolConfig(
                bundles=[
                    Bundle(path=Path("tools/registry").absolute()),
                    Bundle(path=Path("tools/edit_anthropic").absolute()),
                    Bundle(path=Path("tools/review_on_submit_m").absolute()),
                ],
                parse_function=(
                    XMLFunctionCallingParser()
                    if model.config.xml_function_calling
                    else FunctionCallingParser()
                ),
                registry_variables={
                    "USE_FILEMAP": "true",
                    "SUBMIT_REVIEW_MESSAGES": [SUBMIT_REVIEW_MESSAGE],
                },
            ),
            history_processors=(
                [CacheControlHistoryProcessor()]
                if any(
                    claude_keyword in model.name.lower()
                    for claude_keyword in [
                        "claude",
                        "anthropic",
                        "haiku",
                        "sonnet",
                        "opus",
                    ]
                )
                else [DefaultHistoryProcessor()]
            ),
            model=GenericAPIModelConfig(
                name=f"{'openai/' if model.inference_base_url else ''}{model.get_inference_name()}",
                max_input_tokens=model.config.max_input_tokens,
                temperature=1.0,
                completion_kwargs=completion_kwargs,
                api_base=model.inference_base_url,
                api_key=(
                    SecretStr(model.inference_api_key)
                    if model.inference_api_key
                    else None
                ),
                per_instance_call_limit=model.config.per_instance_call_limit,
                per_instance_cost_limit=model.config.per_instance_cost_limit,
            ),
            max_requeries=1,
        ),
        problem_statement=TextProblemStatement(text=instance["problem_statement"]),
    )
