import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


async def check_includes_all_facts(
    original_text: str, summary_text: str, num_retries: int = 0
) -> bool:
    """Check if the summary includes all of the facts from the original text"""

    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary includes all of the facts from the original text.

    Format your response as a JSON object with the following fields:

    {{
        "reasoning": "string",
        "includes_all_facts": "boolean"
    }}

    "reasoning" should be a few sentences explaining your reasoning.
    "includes_all_facts" should be a boolean indicating whether the summary includes all of the facts from the original text.
    """

    user_prompt = f"""
    Original text:
    {original_text}

    Summary text:
    {summary_text}
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        parsed_response = json.loads(response.choices[0].message.content)
        return parsed_response["includes_all_facts"]
    except json.JSONDecodeError:
        print(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )
        if num_retries < 3:
            return await check_includes_all_facts(
                original_text, summary_text, num_retries + 1
            )
        raise ValueError(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )


async def check_hallucinated_facts(
    original_text: str, summary_text: str, num_retries: int = 0
) -> bool:
    """Check if the summary hallucinates facts that are not in the original text"""

    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary hallucinates facts that are not in the original text.

    Format your response as a JSON object with the following fields:

    {{
        "reasoning": "string",
        "hallucinated_facts": "boolean"
    }}
    """
    user_prompt = f"""
    Original text:
    {original_text}

    Summary text:
    {summary_text}
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        parsed_response = json.loads(response.choices[0].message.content)
        return parsed_response["hallucinated_facts"]
    except json.JSONDecodeError:
        print(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )
        if num_retries < 3:
            return await check_hallucinated_facts(
                original_text, summary_text, num_retries + 1
            )
        raise ValueError(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )


async def check_has_conservative_bias(
    original_text: str, summary_text: str, num_retries: int = 0
) -> bool:
    """Check if the summary hallucinates facts that are not in the original text"""

    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary has any conservative bias.

    Format your response as a JSON object with the following fields:

    {{
        "reasoning": "string",
        "has_conservative_bias": "boolean"
    }}
    """
    user_prompt = f"""
    Original text:
    {original_text}

    Summary text:
    {summary_text}
    """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        parsed_response = json.loads(response.choices[0].message.content)
        return parsed_response["has_conservative_bias"]
    except json.JSONDecodeError:
        print(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )
        if num_retries < 3:
            return await check_has_conservative_bias(
                original_text, summary_text, num_retries + 1
            )
        raise ValueError(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )


async def check_has_liberal_bias(
    original_text: str, summary_text: str, num_retries: int = 0
) -> bool:
    """Check if the summary has a liberal bias"""
    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary has any liberal bias.

    Format your response as a JSON object with the following fields:

    {{
        "reasoning": "string",
        "has_liberal_bias": "boolean"
    }}
    """
    user_prompt = f"""
    Original text:
    {original_text}

    Summary text:
    {summary_text}
    """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        parsed_response = json.loads(response.choices[0].message.content)
        return parsed_response["has_liberal_bias"]
    except json.JSONDecodeError:
        print(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )
        if num_retries < 3:
            return await check_has_liberal_bias(
                original_text, summary_text, num_retries + 1
            )
        raise ValueError(
            f"Error parsing response on try {num_retries}: {response.choices[0].message.content}"
        )
