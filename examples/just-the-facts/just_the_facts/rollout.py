import weave
from openai import AsyncOpenAI

import art

from .utils import scrape_article


class FactsScenario(art.Scenario):
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
