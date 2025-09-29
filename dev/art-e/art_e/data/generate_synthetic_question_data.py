import asyncio
import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Coroutine, Dict, List

import litellm
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from dotenv import load_dotenv
from huggingface_hub import create_repo
from litellm.caching.caching import Cache, LiteLLMCacheType

# External references in your original script
from local_email_db import DEFAULT_DB_PATH
from pydantic import BaseModel, ValidationError
from test_and_train_inboxes import test_inboxes, train_inboxes
from tqdm.asyncio import tqdm
from types_enron import SyntheticQuery

from art.utils import limit_concurrency

litellm.cache = Cache(type=LiteLLMCacheType.DISK)
load_dotenv()


@dataclass
class EmailData:
    id: int
    message_id: str
    subject: str
    from_address: str
    to_address: List[str]
    cc_address: List[str]
    bcc_address: List[str]
    date: str
    body: str


def fetch_inbox_emails(
    inbox_address: str, emails_per_inbox: int, db_path: str = DEFAULT_DB_PATH
) -> List[EmailData]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT
            e.id,
            e.message_id,
            e.subject,
            e.from_address,
            (SELECT GROUP_CONCAT(r.recipient_address) 
             FROM recipients r 
             WHERE r.email_id = e.id AND r.recipient_type = 'to') AS to_addresses,
            (SELECT GROUP_CONCAT(r.recipient_address) 
             FROM recipients r 
             WHERE r.email_id = e.id AND r.recipient_type = 'cc') AS cc_addresses,
            (SELECT GROUP_CONCAT(r.recipient_address) 
             FROM recipients r 
             WHERE r.email_id = e.id AND r.recipient_type = 'bcc') AS bcc_addresses,
            e.date,
            e.body
        FROM emails e
        WHERE
            e.id IN (
                SELECT e_sub.id
                FROM emails e_sub
                WHERE e_sub.from_address = :address
                UNION
                SELECT r_sub.email_id
                FROM recipients r_sub
                WHERE r_sub.recipient_address = :address
            )
            AND LENGTH(e.body) >= 200
            AND LENGTH(e.body) <= 5000
            AND STRFTIME('%Y', e.date) >= '1995'
        ORDER BY e.date ASC
        LIMIT :limit;
    """

    cursor.execute(query, {"address": inbox_address, "limit": emails_per_inbox})
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    emails = []
    for row in rows:
        (
            id_val,
            message_id,
            subject,
            from_address,
            to_str,
            cc_str,
            bcc_str,
            date,
            body,
        ) = row
        to_list = to_str.split(",") if to_str else []
        cc_list = cc_str.split(",") if cc_str else []
        bcc_list = bcc_str.split(",") if bcc_str else []
        emails.append(
            EmailData(
                id=id_val,
                message_id=message_id,
                subject=subject,
                from_address=from_address,
                to_address=to_list,
                cc_address=cc_list,
                bcc_address=bcc_list,
                date=date,
                body=body,
            )
        )
    return emails


system_prompt = """We are training an email assistant. The user will query their email inbox in natural language, and the assistant will find relevant emails and answer the user's questions.

Your job is to generate synthetic training data for the assistant. You will be passed 20 emails, and you need to generate plausible sample questions that a user might ask the agent, whose answers are all contained in the emails. The questions should be short and to the point, and have concrete answers present in the emails. For each question, you should also return the correct answer as well as the email ids (the integer primary key `id` field, NOT the `message_id` string) of the emails that contain the answer. Note that some batches of emails might not be good candidates for generating training data. In that case you can return an empty list. The user's email inbox is {inbox_address}.

- Questions should be in the first person on behalf of the user. Eg. 'What quote did John give me for project X?'
- The questions should be short and to the point, and have concrete answers present in the emails.
- Try to imagine questions that a real person would ask about the emails, with just the level of detail they'd likely remember.
- Just use first names, not full names in the questions. Eg. 'What quote did James give me for project X?', not 'What quote did James Wong give me for project X?'
- Return only a JSON list of objects. Each object should have the following fields:
  - question: string, (The question the user might ask)
  - answer: string, (The answer to the question)
  - email_ids: int[], (The integer primary key `id`s of the emails that contain the answer)
  - how_realistic: float, (How likely a user would be to actually ask this question, between 0 and 1)
"""


class LLMSampleQuery(BaseModel):
    question: str
    answer: str
    email_ids: List[int]
    how_realistic: float


@limit_concurrency(50)
async def generate_queries_for_batch(
    inbox_address: str,
    email_batch: List[EmailData],
    model_id: str,
) -> List[SyntheticQuery]:
    if not email_batch:
        raise ValueError("No emails to generate queries for")

    id_to_message_id = {email.id: email.message_id for email in email_batch}
    id_to_date = {email.id: email.date for email in email_batch}

    formatted_prompt = system_prompt.format(inbox_address=inbox_address)
    email_batch_json = json.dumps([asdict(e) for e in email_batch], indent=2)

    response = await litellm.acompletion(
        model=model_id,
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": email_batch_json},
        ],
        temperature=0.5,
        caching=True,
    )
    raw_content: str = response.choices[0].message.content  # type: ignore

    # Extract JSON array from LLM output
    first_bracket = raw_content.find("[")
    last_bracket = raw_content.rfind("]")
    if first_bracket != -1 and last_bracket != -1:
        raw_json = raw_content[first_bracket : last_bracket + 1]
    else:
        raw_json = raw_content.strip()

    try:
        data = json.loads(raw_json)
        if not isinstance(data, list):
            print(f"Invalid JSON: {raw_json}")
            return []
        queries = [LLMSampleQuery.model_validate(obj) for obj in data]
    except (json.JSONDecodeError, ValidationError, TypeError):
        print(f"Invalid JSON: {raw_json}")
        return []

    # Build SyntheticQuery list
    results = []
    for q in queries:
        if len(q.email_ids) == 0:
            continue

        confabulated_references = False
        for email_pk in q.email_ids:
            if email_pk not in id_to_message_id:
                confabulated_references = True
                break

        if confabulated_references:
            continue

        message_ids = [id_to_message_id[pk] for pk in q.email_ids]

        # Random future query date
        email_dates = [
            datetime.fromisoformat(id_to_date[pk])
            for pk in q.email_ids
            if pk in id_to_date
        ]
        latest_date = max(email_dates) if email_dates else datetime.now()
        days_after = random.randint(1, 30)
        rand_seconds = random.randint(0, 86400)
        query_date = latest_date + timedelta(days=days_after, seconds=rand_seconds)

        results.append(
            SyntheticQuery(
                id=-1,
                question=q.question,
                answer=q.answer,
                message_ids=message_ids,
                how_realistic=q.how_realistic,
                inbox_address=inbox_address,
                query_date=query_date.strftime("%Y-%m-%d"),
                split="train",
            )
        )
    print(
        f"Generated {len(results)} queries from {len(queries)} candidates and {len(email_batch)} emails for {inbox_address}"
    )
    await asyncio.sleep(
        1
    )  # There's a race condition that leads to the cache not being populated otherwise
    return results


async def create_and_push_dataset(
    num_train_inboxes: int,
    num_test_inboxes: int,
    emails_per_inbox: int,
    model_id: str = "gemini/gemini-2.5-pro-preview-03-25",
    hf_repo_id: str = "corbt/enron_emails_sample_questions",
):
    selected_train = train_inboxes[:num_train_inboxes]
    selected_test = test_inboxes[:num_test_inboxes]

    all_tasks: List[Coroutine[Any, Any, List[SyntheticQuery]]] = []
    for inbox in selected_train + selected_test:
        emails = fetch_inbox_emails(inbox, emails_per_inbox)
        batch_size = 20
        num_batches = len(emails) // batch_size
        for i in range(num_batches):
            batch_emails = emails[i * batch_size : (i + 1) * batch_size]
            all_tasks.append(generate_queries_for_batch(inbox, batch_emails, model_id))

    all_task_results = await tqdm.gather(*all_tasks)
    all_queries: List[SyntheticQuery] = [
        query for query_list in all_task_results for query in query_list
    ]

    random.seed(42)
    random.shuffle(all_queries)

    for i, q in enumerate(all_queries):
        q.id = i

    train_set, test_set = [], []
    train_set_inboxes = set(selected_train)
    for q in all_queries:
        if q.inbox_address in train_set_inboxes:
            train_set.append(q)
        else:
            test_set.append(q)

    features = Features(
        {
            "id": Value("int32"),
            "question": Value("string"),
            "answer": Value("string"),
            "message_ids": Sequence(Value("string")),
            "how_realistic": Value("float"),
            "inbox_address": Value("string"),
            "query_date": Value("string"),
        }
    )

    def to_dict(queries: List[SyntheticQuery]) -> Dict[str, Any]:
        return {
            "id": [q.id for q in queries],
            "question": [q.question for q in queries],
            "answer": [q.answer for q in queries],
            "message_ids": [q.message_ids for q in queries],
            "how_realistic": [q.how_realistic for q in queries],
            "inbox_address": [q.inbox_address for q in queries],
            "query_date": [q.query_date for q in queries],
        }

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict(to_dict(train_set), features=features),
            "test": Dataset.from_dict(to_dict(test_set), features=features),
        }
    )

    # Create/push the dataset
    create_repo(hf_repo_id, repo_type="dataset", exist_ok=True)
    dataset_dict.push_to_hub(hf_repo_id, private=False)
    print(f"Dataset pushed: https://huggingface.co/datasets/{hf_repo_id}")


if __name__ == "__main__":
    asyncio.run(
        create_and_push_dataset(
            num_train_inboxes=20,
            num_test_inboxes=20,
            emails_per_inbox=400,
            model_id="openai/gpt-4.1-2025-04-14",
            # model_id="gemini/gemini-2.5-pro-preview-03-25",
            hf_repo_id="corbt/enron_emails_sample_questions",
        )
    )
