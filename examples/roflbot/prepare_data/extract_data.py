# %%

# This script assumes you've downloaded one or more monthly dumps of Reddit submissions data from https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13/tech&filelist=1 into a local directory.

import zstandard as zstd
import orjson
from pathlib import Path
import io
from tqdm import tqdm
import os
import multiprocessing
from functools import partial
import polars as pl
from panza import limit_concurrency
from litellm import acompletion
from litellm.types.utils import ModelResponse
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

load_dotenv()

SKIP_INITIAL_PROCESSING = False


def extract_subreddit_posts(zst_file_path: Path, target_subreddit: str) -> list[dict]:
    """
    Extracts posts from a specific subreddit from a zstd-compressed JSON lines file,
    using a file-based cache.

    Args:
        zst_file_path: Path to the .zst file.
        target_subreddit: The name of the subreddit to extract posts from (case-insensitive).

    Returns:
        A list of dictionaries, where each dictionary represents a post.
    """
    # Generate cache file path inline
    cache_filename = f"{zst_file_path.stem}.{target_subreddit}.cache.json"
    cache_file_path = zst_file_path.parent / cache_filename

    if cache_file_path.exists():
        return orjson.loads(cache_file_path.read_bytes())

    if SKIP_INITIAL_PROCESSING:
        return []

    data = []
    cctx = zstd.ZstdDecompressor(max_window_size=2**31)

    total_posts_processed = 0
    target_posts_found = 0

    total_size = os.path.getsize(zst_file_path)
    with open(zst_file_path, "rb") as fh:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Reading {zst_file_path.name}",
            position=1,
        ) as pbar:
            with cctx.stream_reader(fh) as zreader:
                reader = io.TextIOWrapper(zreader, encoding="utf-8")
                last_pos = 0
                for line in reader:
                    # Update progress based on underlying file handle position
                    current_pos = fh.tell()
                    pbar.update(current_pos - last_pos)
                    if total_posts_processed % 50000 == 0:
                        pbar.set_postfix(
                            {
                                "Processed": f"{total_posts_processed:,}",
                                f"r/{target_subreddit}": f"{target_posts_found:,}",
                            },
                        )

                    last_pos = current_pos

                    total_posts_processed += 1

                    # Optimization: Check if subreddit name is in the raw line before parsing
                    if target_subreddit not in line:
                        continue  # Skip this line if the subreddit name isn't present

                    # Use orjson for faster parsing. 2x speedup
                    post = orjson.loads(line)

                    subreddit = post.get("subreddit", "")
                    if subreddit == target_subreddit:
                        target_posts_found += 1
                        data.append(post)

            # Ensure the progress bar reaches 100% if reading finished early
            if pbar.n < total_size:
                pbar.update(total_size - pbar.n)
            # Set final postfix
            pbar.set_postfix(
                {
                    "Processed": f"{total_posts_processed:,}",
                    f"r/{target_subreddit}": f"{target_posts_found:,}",
                },
                refresh=True,
            )

    # Cache the extracted posts so we don't need to parse the file again.
    with open(cache_file_path, "wb") as cf:
        cf.write(orjson.dumps(data))

    return data


def preview_jokes(df: pl.DataFrame):
    for r in df.rows(named=True):
        print("-" * 20)
        print(f"https://www.reddit.com/r/{target_sub}/comments/{r['id']}")
        print(r["title"] + "//")
        print(r["selftext"].replace("\r", ""))
        print(f"Score: {r['score']}, Comments: {r['num_comments']}")
        print()


@limit_concurrency(20)
async def get_formatted_joke(
    joke: dict, model: str = "gemini/gemini-2.0-flash"
) -> ModelResponse:
    response = await acompletion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You're helping build a dataset of jokes from Reddit. The joke is sometimes split across both the title and the selftext. Please combine them and format the joke as a single string that reads naturally. Return only the joke, no other text.",
            },
            {
                "role": "user",
                "content": f"Title: {joke['title']}\n\nSelftext: {joke['selftext']}",
            },
        ],
    )
    return response  # type: ignore


if __name__ == "__main__":
    data_dir = Path("/Users/kyle/Downloads/reddit/submissions")
    target_sub = "Jokes"
    num_processes = 4

    all_files = sorted(list(data_dir.glob("RS_*.zst")))  # Sort for consistent order
    print(
        f"\nFound {len(all_files)} files in {data_dir} to process for r/{target_sub} using {num_processes} processes"
    )

    all_jokes_data = []

    # Create a partial function with the target_subreddit fixed
    extract_func = partial(extract_subreddit_posts, target_subreddit=target_sub)

    # Use multiprocessing Pool to process files in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm to track progress over the files being processed
        results = list(
            tqdm(
                pool.imap_unordered(extract_func, all_files),
                total=len(all_files),
                desc="Processing files",
                unit="file",
                position=0,
            )
        )

    # Flatten the list of lists into a single list
    for result in results:
        all_jokes_data.extend(result)

    print(
        f"\nFound a total of {len(all_jokes_data):,} posts for r/{target_sub} across all files."
    )

    # %%

    schema = pl.Schema(
        {
            "id": pl.Utf8,
            "title": pl.Utf8,
            "selftext": pl.Utf8,
            "score": pl.Int64,
            "num_comments": pl.Int64,
            "thumbnail": pl.Utf8,
            "author": pl.Utf8,
            "created_utc": pl.Int64,
            "upvote_ratio": pl.Float64,
        }
    )

    df = pl.DataFrame(all_jokes_data, schema=schema)

    df = df.filter(pl.col("selftext").str.len_chars() > 20)
    df = df.filter(pl.col("thumbnail").ne("nsfw"))

    print(f"Found {len(df)} jokes after filtering")

    df = df.unique(subset=["title", "selftext"])
    print(f"Found {len(df)} jokes after deduplicating")

    df = df.with_columns(
        pl.col("score").replace(0, 1).log().alias("log_score"),
        pl.from_epoch(pl.col("created_utc"), time_unit="s").alias("created_at"),
    )

    # Add token count
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    def count_tokens(title: str, selftext: str) -> int:
        text = f"{title} {selftext}"
        return len(tokenizer.encode(text))

    df = df.with_columns(
        pl.struct(["title", "selftext"])
        .map_batches(
            lambda s: pl.Series(
                [count_tokens(r["title"], r["selftext"]) for r in s.to_list()]
            )
        )
        .alias("num_tokens")
    )

    df.head()

    # %%

    # preview_jokes(df.sample(50))

    # %%

    # Plot histogram of log_scores
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["log_score"].to_numpy(), bins=50, edgecolor="black"
    )  # Convert to numpy array for matplotlib
    plt.title("Histogram of Log Scores for Jokes")
    plt.xlabel("Log Score")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()

    # %%

    # Plot histogram of joke lengths
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["num_tokens"].to_numpy(),
        bins=50,
        edgecolor="black",
        cumulative=True,
        density=True,  # Normalize to show CDF
    )
    plt.title("Cumulative Distribution of Joke Lengths")
    plt.xlabel("Length")
    plt.ylabel("Cumulative Probability")  # Update y-axis label
    plt.grid(axis="y", alpha=0.75)
    plt.show()

    # show the 95th percentile

    df.with_columns(pl.col("num_tokens").alias("length")).quantile(0.99)

    # %%

    preview_jokes(df.sample(50))

    # %%

    df = df.select(
        [
            "id",
            "author",
            "title",
            "selftext",
            "score",
            "log_score",
            "num_comments",
            "created_at",
            "num_tokens",
        ]
    )

    df = df.sample(fraction=1, seed=42)

    # %% Add split, conversion, and push to hub logic
    hf_dataset = Dataset.from_polars(df)

    # Split the dataset
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)

    print(
        f"Dataset split into train ({len(hf_dataset['train'])}) and test ({len(hf_dataset['test'])}) sets."
    )

    # Push to Hugging Face Hub
    print("Pushing dataset to Hugging Face Hub...")
    repo_name = "corbt/reddit_jokes"
    hf_dataset.push_to_hub(repo_name)
    print(f"Dataset successfully pushed to {repo_name}")
