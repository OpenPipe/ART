from art_e.data.types_enron import SyntheticQuery
from typing import List, Optional, Literal
from datasets import load_dataset
import random

# Define the Hugging Face repository ID
HF_REPO_ID = "corbt/enron_emails_sample_questions"


def load_synthetic_queries(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[SyntheticQuery]:
    dataset = load_dataset(HF_REPO_ID, split=split)  # type: ignore

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if shuffle or seed is not None:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()

    # Convert each row (dict) in the dataset to a SyntheticQuery object
    # Apply the limit *after* conversion if specified
    queries = [
        SyntheticQuery(**row, split=split)  # type: ignore
        for row in dataset  # type: ignore
    ]

    if max_messages is not None:
        queries = [query for query in queries if len(query.message_ids) <= max_messages]

    if shuffle:
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(queries)
        else:
            random.shuffle(queries)

    if limit is not None:
        return queries[:limit]
    else:
        return queries
