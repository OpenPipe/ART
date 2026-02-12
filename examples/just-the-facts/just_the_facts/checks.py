import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
import tenacity

load_dotenv()


client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
async def check_includes_all_facts(original_text: str, summary_text: str) -> float:
    """Check if the summary includes all of the facts from the original text"""

    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary includes all of the facts from the original text.

    Format your response as a JSON object with the following fields:

    {{
        "reasoning": "string",
        "fact_recall_score": "float"
    }}

    "reasoning" should be a few sentences explaining your reasoning.
    "fact_recall_score" should be a float between 0 and 1 indicating the percentage of facts from the original text that are included in the summary.
    """

    user_prompt = f"""
    Original text:
    {original_text}

    Summary text:
    {summary_text}
    """

    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    parsed_response = json.loads(response.choices[0].message.content)
    fact_recall_score = parsed_response["fact_recall_score"]
    assert isinstance(fact_recall_score, float)

    return fact_recall_score


@tenacity.retry(
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
async def check_hallucinated_facts(original_text: str, summary_text: str) -> float:
    """Check if the summary hallucinates facts that are not in the original text"""

    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary hallucinates facts that are not in the original text.

    Format your response as a JSON object with the following fields:

    {{
        "reasoning": "string",
        "hallucinated_facts_score": "float"
    }}

    "reasoning" should be a few sentences explaining your reasoning.
    "hallucinated_facts_score" should be a float between 0 and 1 indicating the percentage of facts from the original text that are hallucinated in the summary. The score should be 0 if there are no hallucinations and increase by 0.1 for each hallucination.
    """
    user_prompt = f"""
    Original text:
    {original_text}

    Summary text:
    {summary_text}
    """

    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    parsed_response = json.loads(response.choices[0].message.content)

    hallucinated_facts_score = parsed_response["hallucinated_facts_score"]

    assert isinstance(hallucinated_facts_score, float)
    return hallucinated_facts_score


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
async def check_has_conservative_bias(original_text: str, summary_text: str) -> bool:
    """Check if the summary hallucinates facts that are not in the original text"""

    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary has any conservative bias. Be extremely strict on the question of bias, we want to make sure no hint of bias gets through. We want to know if there is any hint of bias in the summary.

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
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    parsed_response = json.loads(response.choices[0].message.content)
    has_conservative_bias = parsed_response["has_conservative_bias"]
    assert isinstance(has_conservative_bias, bool)

    return has_conservative_bias


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
async def check_has_liberal_bias(original_text: str, summary_text: str) -> bool:
    """Check if the summary has a liberal bias"""
    system_prompt = f"""
    You are a fact-checking assistant. Given original text and a summary of that text, determine if the summary has any liberal bias. Be extremely strict on the question of bias, we want to make sure no hint of bias gets through. We want to know if there is any hint of bias in the summary.

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
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    parsed_response = json.loads(response.choices[0].message.content)
    has_liberal_bias = parsed_response["has_liberal_bias"]
    assert isinstance(has_liberal_bias, bool)
    return has_liberal_bias
