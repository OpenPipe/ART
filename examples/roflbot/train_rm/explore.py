# %%

from datasets import load_dataset
from datetime import datetime

dataset = load_dataset("corbt/reddit_jokes")


def format_joke_for_rm(
    title: str,
    text: str,
    created_at: datetime = datetime(2025, 1, 1),
    poster: str = "unknown",
) -> str:
    return f"Poster: {poster}\nTimestamp: {created_at.strftime('%A %b %d, %Y at %H:%M')}\n\nTitle: {title}\n\nText: {text}"


entry = dataset["train"][0]  # type: ignore
formatted = format_joke_for_rm(
    entry["title"],
    entry["selftext"],
    entry["created_at"],
    entry["author"],
)

print(formatted)

# %%
