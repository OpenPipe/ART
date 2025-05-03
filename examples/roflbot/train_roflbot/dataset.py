from pydantic import BaseModel
from datasets import load_dataset, Dataset
from datetime import datetime


class RedditJoke(BaseModel):
    id: str
    title: str
    selftext: str
    log_score: float
    created_at: datetime


def get_jokes(split: str, limit: int | None = None) -> list[RedditJoke]:
    dataset: Dataset = load_dataset("corbt/reddit_jokes", split=split)  # type: ignore
    if limit is not None:
        dataset = dataset.select(range(limit))

    return [RedditJoke(**d) for d in dataset]  # type: ignore
