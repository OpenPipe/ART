import os
from panza import SQLiteCache, limit_concurrency
from datetime import datetime
import httpx

cache = SQLiteCache("rm_cache.db")


@cache.cache()
@limit_concurrency(20)
async def score_joke(
    title: str,
    text: str,
    poster: str = "unknown",
    created_at: str = datetime(2025, 1, 1).isoformat(),
) -> float:
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            os.environ["REWARD_MODEL_URL"],
            json={
                "title": title,
                "text": text,
                "poster": poster,
                "created_at": created_at,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["score"]
