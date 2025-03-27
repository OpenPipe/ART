from dataclasses import dataclass
from itertools import takewhile
import math
import random
from transformers import PreTrainedTokenizerBase
from typing import cast, Generator, TypedDict

from ..types import Trajectory


@dataclass
class TokenizedResult:
    trajectory: Trajectory
    advantage: float
    conversation: list
    chat_template: str
    chat: str
    tokens: list[str]
    token_ids: list[int]
    input_pos: list[int]
    assistant_mask: list[int]
    logprobs: list[float]
    prompt_id: int = 0
    prompt_length: int = 0

    def without_prompt(self) -> "TokenizedResult":
        return TokenizedResult(
            trajectory=self.trajectory,
            conversation=self.conversation,
            advantage=self.advantage,
            chat_template=self.chat_template,
            chat=self.chat,
            tokens=self.tokens[self.prompt_length :],
            token_ids=self.token_ids[self.prompt_length :],
            input_pos=self.input_pos[self.prompt_length :],
            assistant_mask=self.assistant_mask[self.prompt_length :],
            logprobs=self.logprobs[self.prompt_length :],
            prompt_id=self.prompt_id,
            prompt_length=0,
        )


class TokenizedResultDict(TypedDict):
    input_ids: list[int]
    assistant_masks: list[int]


def tokenize_trajectory_groups(
    tokenizer: PreTrainedTokenizerBase,
    trajectory_groups: list[list[Trajectory]],
) -> Generator["TokenizedResult", None, None]:
    for group in trajectory_groups:
        if not group:
            continue
        results: list[TokenizedResult] = []
        # Calculate GRPO group mean and standard deviation
        reward_mean = sum(trajectory.reward for trajectory in group) / len(group)
        reward_std = math.sqrt(
            sum((trajectory.reward - reward_mean) ** 2 for trajectory in group)
            / len(group)
        )
        for trajectory in group:
            # Calculate GRPO advantage for this trajectory
            advantage = (trajectory.reward - reward_mean) / (reward_std + 1e-6)
            # Skip trajectories with no advantage
            if advantage == 0:
                continue
            results.append(
                tokenize_trajectory(
                    tokenizer,
                    trajectory,
                    advantage,
                )
            )
        # Choose a random prompt id
        prompt_id = random.randint(-(2**63), 2**63 - 1)
        # Find the longest shared prefix
        # TODO: Potentially support multiple prompts per group
        # Initial thought is to sort the results by token_ids and then
        # successively group prompts with the same prefix.
        prompt_length = len(
            list(
                takewhile(
                    lambda x: len(set(x)) == 1,
                    zip(*(r.token_ids for r in results)),
                )
            )
        )
        # Set the prompt id and length
        for result in results:
            result.prompt_id = prompt_id
            result.prompt_length = prompt_length
            # zero out assistant prompt tokens
            # TODO: If we support multiple prompts per group,
            # this may be harmful.
            result.assistant_mask[:prompt_length] = [0] * prompt_length
        random.shuffle(results)
        yield from results


def tokenize_trajectory(
    tokenizer: PreTrainedTokenizerBase,
    trajectory: Trajectory,
    advantage: float,
) -> TokenizedResult:
    """
    Tokenizes a trajectory and returns a TokenizedResult.
    """
    # Convert Choice objects to dicts for compatibility with the tokenizer.
    conversation: list = [
        (
            message_or_choice
            if isinstance(message_or_choice, dict)
            else {
                "role": "assistant",
                "content": message_or_choice.message.content or "",
                "tool_calls": [
                    tool_call.model_dump()
                    for tool_call in message_or_choice.message.tool_calls or []
                ],
            }
        )
        for message_or_choice in trajectory.messages_and_choices
    ]
    # Update the chat template to add generation tags for assistant token masking.
    # TODO: Improve the way we get chat templates, potentially just use the default
    # chat template and identify the assistant tokens a different way.
    chat_template = _updated_chat_template(
        tokenizer.get_chat_template("tool_use" if trajectory.tools else None)
    )
    # Apply the chat template to the conversation to get a string representation.
    chat = cast(
        str,
        tokenizer.apply_chat_template(
            conversation,
            tools=list(trajectory.tools) if trajectory.tools else None,  # type: ignore
            chat_template=chat_template,
            tokenize=False,
        ),
    )
    # Tokenize the conversation and get the tokenized result and assistant mask.
    tokenized_result = cast(
        TokenizedResultDict,
        tokenizer.apply_chat_template(
            conversation,
            tools=list(trajectory.tools) if trajectory.tools else None,  # type: ignore
            chat_template=chat_template,
            return_dict=True,
            return_assistant_tokens_mask=True,
        ),
    )
    # Initialize logprobs with NaNs.
    logprobs = [float("nan")] * len(tokenized_result["input_ids"])
    # Update the tokenized result and logprobs with the logprobs from the chat completion choices.
    _update_tokenized_result_and_logprobs(
        trajectory,
        tokenized_result,
        logprobs,
    )
    # Decode the tokenized result to get the string tokens.
    tokens = [tokenizer.decode(token_id) for token_id in tokenized_result["input_ids"]]
    # Return the TokenizedResult.
    return TokenizedResult(
        trajectory=trajectory,
        conversation=conversation,
        advantage=advantage,
        chat_template=chat_template,
        chat=chat,
        tokens=tokens,
        token_ids=tokenized_result["input_ids"],
        input_pos=list(range(len(tokens))),
        assistant_mask=tokenized_result["assistant_masks"],
        logprobs=logprobs,
    )


def _update_tokenized_result_and_logprobs(
    trajectory: Trajectory,
    tokenized_result: TokenizedResultDict,
    logprobs: list[float],
) -> None:
    """
    Updates the tokenized result and logprobs for a trajectory using the token logprobs
    from the chat completion choices.
    """
    start = 0
    end = 0

    def update_assistant_range() -> None:
        """
        Updates the start and end indices to the next range of generated tokens
        """
        nonlocal start, end
        try:
            start = end + tokenized_result["assistant_masks"][end:].index(1)
        except ValueError:
            start = len(tokenized_result["assistant_masks"])
        try:
            end = start + tokenized_result["assistant_masks"][start:].index(0)
        except ValueError:
            end = len(tokenized_result["assistant_masks"])

    for message_or_choice in trajectory.messages_and_choices:
        if isinstance(message_or_choice, dict):
            if message_or_choice["role"] == "assistant":
                update_assistant_range()
                # If it's an assistant message, but not a Choice object,
                # then we assume it's part of the prompt and not generated.
                tokenized_result["assistant_masks"][start:end] = [0] * (end - start)
            continue
        choice = message_or_choice
        update_assistant_range()
        if choice.message.content and not choice.logprobs:
            raise ValueError(
                "Chat completion choices with content must have corresponding token logprobs"
            )
        elif not choice.message.content and not choice.logprobs:
            # This is a strange case that I encountered, but it may not be necessary
            # anymore or perhaps we should do something different here.
            continue
        assert choice.logprobs and (
            token_logprobs := choice.logprobs.content or choice.logprobs.refusal
        ), "Chat completion choices must have logprobs"
        # We update the token ids, assistant mask, and logprobs with the logprobs from the chat
        # completion choice. We do this because the tokenization is not always the same, probably
        # due to multi-token unicode characters.
        tokenized_result["input_ids"][start:end] = [
            int(token_logprob.token.split(":")[1]) for token_logprob in token_logprobs
        ]
        tokenized_result["assistant_masks"][start:end] = [1 for _ in token_logprobs]
        logprobs[start:end] = [
            token_logprob.logprob for token_logprob in token_logprobs
        ]
        # We update the end index because the number of token logprobs may be different
        # from the original number of tokens in the tokenized result.
        end = start + len(token_logprobs)


def _updated_chat_template(chat_template: str) -> str:
    """
    Returns an updated chat template that adds generation tags for assistant token masking.
    """
    # TODO: This is a mess. We should either just use the default chat template and find
    # the assistant tokens a different way or store full chat templates somewhere for
    # each model.
    return (
        chat_template
        # Remove template logic that strips reasoning content from the chat messages
        .replace(
            "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}",
            "",
        )
        # Add generation tags for assistant token masking
        .replace(
            "{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}",
            "{{'<｜Assistant｜>'}}{% generation %}{{ content }}{% endgeneration %}{{'<｜end▁of▁sentence｜>'}}",
        )
        # Add generation tags for assistant token masking (for Hermes 2 Theta)
        .replace(
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}",
            "{{'<|im_start|>' + message['role'] + '\n'}}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}{% else %}{{ message['content'] }}{% endif %}{{'<|im_end|>' + '\n'}}",
        )
        # Add generation tags for assistant token masking (for Qwen 2.5 Instruct)
        .replace(
            """
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    """.strip(),
            """
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" and not message.tool_calls %}
        {{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{ message.content }}{% endgeneration %}{{ '<|im_end|>' + '\\n' }}
""".strip(),
        ).replace(
            """
        {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}""".strip(),
            """
        {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' }}{% generation %}{{ message.content }}{% endgeneration %}
        {%- endif %}""".strip(),
        )
        # Add generation tags for assistant token masking (for Llama 3.1 8B)
        .replace(
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}",
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + '<|eot_id|>' }}{% endgeneration %}{% else %}{{ message['content'] | trim + '<|eot_id|>' }}{% endif %}",
        )
        # Add generation tags for assistant token masking (for Llama 3.3 70B)
        .replace(
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}",
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}{%- if message['role'] == 'assistant' %}{% generation %}{{ message['content'] | trim + '<|eot_id|>' }}{% endgeneration %}{% else %}{{ message['content'] | trim + '<|eot_id|>' }}{% endif %}",
        )
        # Add generation tags for assistant token masking (for Hermes 3)
        .replace(
            hermes3_old,
            hermes3_new,
        )
        # Add generation tags for assistant token masking (for Hermes 3 w/tool use)
        .replace(
            hermes3_old_with_tool_use,
            hermes3_new_with_tool_use,
        )
    )


hermes3_old = """{{bos_token}}{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful assistant.<|im_end|>
' }}{% endif %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""


hermes3_new = """{{bos_token}}{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system
You are a helpful assistant.<|im_end|>
' }}{% endif %}{{'<|im_start|>' + message['role'] + '
'}}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}{% else %}{{ message['content'] + '<|im_end|>' }}{% endif %}{{ '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""


hermes3_old_with_tool_use = """    {%- if message.role == "user" or message.role == "system" or (message.role == "assistant" and message.tool_calls is not defined) %}
        {{- '<|im_start|>' + message.role + '
' + message.content + '<|im_end|>' + '
' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
    {%- for tool_call in message.tool_calls %}
       {{- '
<tool_call>
' }}           {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '{' }}
            {{- '"name": "' }}
            {{- tool_call.name }}
            {{- '"' }}
            {{- ', '}}
            {%- if tool_call.arguments is defined %}
                {{- '"arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments|tojson }}
                {%- endif %}
            {%- endif %}
             {{- '}' }}
            {{- '
</tool_call>' }}
    {%- endfor %}
        {{- '<|im_end|>
' }}"""

hermes3_new_with_tool_use = """    {%- if message.role == "user" or message.role == "system" %}
        {{- '<|im_start|>' + message.role + '
' + message.content + '<|im_end|>' + '
' }}
    {%- elif message.role == "assistant" and message.tool_calls is not defined %}
        {{- '<|im_start|>' + message.role + '
' }}{% generation %}{{ message.content + '<|im_end|>' }}{% endgeneration %}{{ '
' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
    {%- for tool_call in message.tool_calls %}{% generation %}
       {{- '
<tool_call>
' }}           {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '{' }}
            {{- '"name": "' }}
            {{- tool_call.name }}
            {{- '"' }}
            {{- ', '}}
            {%- if tool_call.arguments is defined %}
                {{- '"arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments|tojson }}
                {%- endif %}
            {%- endif %}
             {{- '}' }}
            {{- '
</tool_call>' }}
    {%- endgeneration %}{% endfor %}
        {{- '<|im_end|>
' }}"""
