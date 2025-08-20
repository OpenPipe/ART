import asyncio
import os

import weave
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from utils import scrape_article

import art

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
            "content": "You are an unbiased summarizer of news articles. You will be provided with an article and expected to give a completely unbiased representation of all of the facts in the article, with no bias or opinion whatsoever. Do not include extra facts not present in the article, and do not forget to include all of the facts. Return your response in one or two paragraphs.",
        }
    )
    traj.messages_and_choices.append(
        {
            "role": "user",
            "content": article_text,
        }
    )

    completion = await client.chat.completions.create(
        model=model.name,
        messages=traj.messages(),
    )

    traj.messages_and_choices.append(completion.choices[0])

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

    print(traj.messages()[-1]["content"])
