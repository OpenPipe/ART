"""Task completion checking using LLM evaluation."""

import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import weave

import art
from art.utils import limit_concurrency

load_dotenv()


@weave.op()
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    reraise=True,
)
@limit_concurrency(n=128)
async def check_successful(trajectory: art.Trajectory) -> bool:
    """Check if the agent completed the task successfully using GPT-4 evaluation.

    Args:
        trajectory: The trajectory containing messages, tool calls, and task information

    Returns:
        bool: True if task was completed successfully, False otherwise
    """

    # Get the conversation history
    messages = trajectory.messages()

    # Prepare the conversation history for evaluation
    conversation_text = ""
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")

        if role == "system":
            continue  # Skip system messages for brevity
        elif role == "user":
            conversation_text += f"User: {content}\n"
        elif role == "assistant":
            if "tool_calls" in message:
                conversation_text += f"Assistant: {content}\n"
                for tool_call in message["tool_calls"]:
                    function_name = tool_call.get("function", {}).get(
                        "name", "unknown_function"
                    )
                    function_args = tool_call.get("function", {}).get("arguments", "{}")
                    conversation_text += (
                        f"{function_name} called with arguments: {function_args}\n"
                    )
            else:
                conversation_text += f"Assistant: {content}\n"
        elif role == "tool":
            # Truncate long tool responses for readability
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            conversation_text += f"Tool Result: {content}\n"

    # Prepare the evaluation prompt
    evaluation_prompt = f"""You are an expert evaluator determining whether an AI agent successfully completed a given task.

CONVERSATION HISTORY:
{conversation_text}

EVALUATION CRITERIA:
1. Did the agent obtain the specific information or perform the actions requested in the task?
2. If the task required multiple steps, were all steps completed?
3. Did the agent provide a meaningful response to the user's request?
4. If the agent encountered errors, did it handle them appropriately and still accomplish the core objective?

Based on the conversation above, determine if the agent successfully completed the task. Consider:
- Whether the core objective was achieved
- If required data was obtained and presented
- Whether any calculations or analysis requested were performed
- If the agent provided useful, accurate information relevant to the task

Respond with only a JSON object containing {{"success": true}} if the task was completed successfully, or {{"success": false}} if it was not completed or only partially completed."""

    try:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = await client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_completion_tokens=1000,
            response_format={"type": "json_object"},
        )

        # Parse the response
        result = json.loads(response.choices[0].message.content.strip())

        if result["success"]:
            return True
        else:
            return False

    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        return False
