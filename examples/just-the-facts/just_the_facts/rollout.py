import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
import weave

import art
from just_the_facts.checks import (
    check_hallucinated_facts,
    check_has_conservative_bias,
    check_has_liberal_bias,
    check_includes_all_facts,
)
from just_the_facts.utils import scrape_article

load_dotenv()


class FactsScenario(BaseModel):
    article_url: str


@weave.op
async def rollout(model: art.Model, scenario: FactsScenario) -> art.Trajectory:
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metrics={},
        scenario=scenario,
    )

    client = AsyncOpenAI(
        api_key=model.inference_api_key,
        base_url=model.inference_base_url,
    )

    article_text = await scrape_article(scenario.article_url)

    traj.messages_and_choices.append(
        {
            "role": "system",
            "content": "You are an unbiased summarizer of news articles. You will be provided with an article and expected to give a representation of all of the facts in the article. Do not include extra facts not present in the article, and do not forget to include all of the facts. Return your response in one or two paragraphs. Answer in the same language as the article. Respond in 300 words or less.",
        }
    )
    traj.messages_and_choices.append(
        {
            "role": "user",
            "content": f"Article:\n\n{article_text}",
        }
    )

    completion = await client.chat.completions.create(
        model=model.name if model.trainable else model.inference_model_name,
        messages=traj.messages(),
        max_completion_tokens=500,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    traj.messages_and_choices.append(completion.choices[0])

    summary_text = completion.choices[0].message.content

    # run checks concurrently
    (
        includes_all_facts,
        hallucinated_facts,
        has_conservative_bias,
        has_liberal_bias,
    ) = await asyncio.gather(
        check_includes_all_facts(article_text, summary_text),
        check_hallucinated_facts(article_text, summary_text),
        check_has_conservative_bias(article_text, summary_text),
        check_has_liberal_bias(article_text, summary_text),
    )

    # add checks to traj
    traj.metrics["fact_recall"] = includes_all_facts
    traj.metrics["hallucinated_facts"] = hallucinated_facts
    traj.metrics["conservative_bias"] = has_conservative_bias
    traj.metrics["liberal_bias"] = has_liberal_bias

    traj.reward = 1

    traj.reward -= (1 - includes_all_facts) * 0.3

    traj.reward -= hallucinated_facts * 0.3

    if has_conservative_bias:
        traj.reward -= 0.2

    if has_liberal_bias:
        traj.reward -= 0.2

    return traj


if __name__ == "__main__":
    model = art.Model(
        name="gpt-4o-mini",
        project="just-the-facts",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
        inference_base_url="https://openrouter.ai/api/v1",
    )
    traj = asyncio.run(
        rollout(
            model=model,
            scenario=FactsScenario(
                article_url="https://www.foxnews.com/politics/schiff-launches-legal-defense-fund-response-claims-trump-weaponizing-justice-system"
            ),
        )
    )

    print(traj.metrics)

    print(traj.messages()[-1]["content"])
