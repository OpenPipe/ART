"""
Consolidated email search agent training functionality.
This file combines all the essential components needed to train an email search agent.
"""

import art
from art.local import LocalBackend
import asyncio
from dotenv import load_dotenv
from typing import List, Optional, Literal, cast
import sqlite3
import logging
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field, validate_call, ValidationError
import json
import textwrap
import statistics
import os
from datasets import load_dataset, Features, Value, Sequence
from tqdm import tqdm
from litellm import acompletion
import litellm
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm.caching.caching import LiteLLMCacheType, Cache
from art.utils.litellm import convert_litellm_choice_to_openai
from art.utils import iterate_dataset
from tenacity import retry, stop_after_attempt
import weave
from weave.trace.autopatch import AutopatchSettings
from rich import print

# Setup
load_dotenv()
litellm.cache = Cache(type=LiteLLMCacheType.DISK)
litellm.drop_params = True
logging.getLogger("weave.trace.op").setLevel(logging.WARNING)

# ==================== DATA TYPES ====================


class TrainingConfig(BaseModel):
    trajectories_per_group: int = 6
    groups_per_step: int = 1
    learning_rate: float = 1.2e-5
    eval_steps: int = 30
    val_set_size: int = 100
    training_dataset_size: int = 4000
    num_epochs: int = 4
    use_judge_group_variant: Literal["v1"] | Literal["v2"] | None = None
    group_judge_model: str = "openai/o3"
    minimum_reward_std_dev: float = 0.0
    training_dataset_seed: int | None = None


class ProjectPolicyConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048
    log_to_openpipe: bool = False
    litellm_model_name: str | None = None
    stupid_simple_reward_fn: bool = False
    training_config: TrainingConfig | None = None


class SyntheticQuery(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]


class Email(BaseModel):
    message_id: str
    date: str
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []
    body: Optional[str] = None
    file_name: Optional[str] = None


@dataclass
class SearchResult:
    message_id: str
    snippet: str


# ==================== DATABASE SETUP ====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "..", "data", "enron_emails.db")
DEFAULT_REPO_ID = "corbt/enron-emails"
HF_REPO_ID = "corbt/enron_emails_sample_questions"

# Database creation SQL
SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS recipients;
DROP TABLE IF EXISTS emails_fts;
DROP TABLE IF EXISTS emails;

CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    subject TEXT,
    from_address TEXT,
    date TEXT,
    body TEXT,
    file_name TEXT
);

CREATE TABLE recipients (
    email_id INTEGER,
    recipient_address TEXT,
    recipient_type TEXT,
    FOREIGN KEY(email_id) REFERENCES emails(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_recipients_address ON recipients(recipient_address);
CREATE INDEX idx_recipients_type ON recipients(recipient_type);
CREATE INDEX idx_recipients_email_id ON recipients(email_id);
CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject,
    body,
    content='emails',
    content_rowid='id'
);

CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts (rowid, subject, body)
    VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
    DELETE FROM emails_fts WHERE rowid=old.id;
END;

CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
    UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
END;

INSERT INTO emails_fts (rowid, subject, body) SELECT id, subject, body FROM emails;
"""


def generate_database(overwrite: bool = False):
    """Generate the email database from Hugging Face dataset."""
    if os.path.exists(DEFAULT_DB_PATH) and not overwrite:
        logging.info(f"Database already exists at {DEFAULT_DB_PATH}")
        return

    os.makedirs(os.path.dirname(DEFAULT_DB_PATH), exist_ok=True)

    # Download dataset
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    dataset = load_dataset(DEFAULT_REPO_ID, features=expected_features, split="train")

    # Create database
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()

    # Populate database
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")
    conn.execute("BEGIN TRANSACTION;")

    processed_emails = set()

    for email_data in tqdm(dataset, desc="Inserting emails"):
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]

        # Filter long emails and high recipient counts
        if len(body) > 5000:
            continue

        to_list = [str(addr) for addr in email_data["to"] if addr]
        cc_list = [str(addr) for addr in email_data["cc"] if addr]
        bcc_list = [str(addr) for addr in email_data["bcc"] if addr]

        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)
        if total_recipients > 30:
            continue

        # Deduplicate
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            continue
        processed_emails.add(email_key)

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (message_id, subject, from_address, date_str, body, file_name),
        )

        email_pk_id = cursor.lastrowid

        # Insert recipients
        recipient_data = []
        for addr in to_list:
            recipient_data.append((email_pk_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((email_pk_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((email_pk_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
            """,
                recipient_data,
            )

    conn.execute("COMMIT;")
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()

    logging.info(f"Database successfully created at {DEFAULT_DB_PATH}")


# ==================== EMAIL SEARCH TOOLS ====================

conn = None


def get_conn():
    global conn
    if conn is None:
        conn = sqlite3.connect(
            f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return conn


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """Search the email database based on keywords and filters."""
    if not keywords:
        raise ValueError("No keywords provided for search.")

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    cursor = get_conn().cursor()
    where_clauses = []
    params = []

    # Keywords (FTS)
    fts_query = " ".join('"' + k.replace('"', '""') + '"' for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # Inbox filter
    where_clauses.append("""
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.id
        ))
    """)
    params.extend([inbox, inbox])

    # Optional filters
    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    if to_addr:
        where_clauses.append("""
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.id
            )
        """)
        params.append(to_addr)

    if sent_after:
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    if sent_before:
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC
        LIMIT ?;
    """
    params.append(max_results)

    cursor.execute(sql, params)
    results = cursor.fetchall()

    return [SearchResult(message_id=row[0], snippet=row[1]) for row in results]


def read_email(message_id: str) -> Optional[Email]:
    """Retrieve a single email by its message_id."""
    cursor = get_conn().cursor()

    # Get email details
    email_sql = """
        SELECT message_id, date, subject, from_address, body, file_name
        FROM emails
        WHERE message_id = ?;
    """
    cursor.execute(email_sql, (message_id,))
    email_row = cursor.fetchone()

    if not email_row:
        return None

    msg_id, date, subject, from_addr, body, file_name = email_row

    # Get recipients
    recipients_sql = """
        SELECT recipient_address, recipient_type
        FROM recipients
        WHERE email_id = ?;
    """
    cursor.execute(recipients_sql, (message_id,))
    recipient_rows = cursor.fetchall()

    to_addresses = []
    cc_addresses = []
    bcc_addresses = []

    for addr, type in recipient_rows:
        type_lower = type.lower()
        if type_lower == "to":
            to_addresses.append(addr)
        elif type_lower == "cc":
            cc_addresses.append(addr)
        elif type_lower == "bcc":
            bcc_addresses.append(addr)

    return Email(
        message_id=msg_id,
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )


# ==================== DATA LOADING ====================

bad_queries = [49, 101, 129, 171, 208, 266, 327]


def load_synthetic_queries(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
    exclude_known_bad_queries: bool = True,
) -> List[SyntheticQuery]:
    """Load synthetic query dataset."""
    dataset = load_dataset(HF_REPO_ID, split=split)

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if exclude_known_bad_queries:
        dataset = dataset.filter(lambda x: x["id"] not in bad_queries)

    if shuffle or seed is not None:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()

    queries = [SyntheticQuery(**row, split=split) for row in dataset]

    if max_messages is not None:
        queries = [query for query in queries if len(query.message_ids) <= max_messages]

    if limit is not None:
        return queries[:limit]
    else:
        return queries


# ==================== TRAJECTORY AND ROLLOUT ====================


@dataclass
class FinalRubric:
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_metrics(self) -> dict[str, float | int]:
        metrics = {k: int(v) for k, v in asdict(self).items()}
        metrics["failed_format_validation"] = int(
            self.bad_tool_call_name
            or self.bad_tool_call_args
            or self.cant_parse_tool_call
        )
        return metrics


def calculate_reward(
    policy_config: ProjectPolicyConfig, rubric: FinalRubric, traj: art.Trajectory
) -> float:
    """Calculate reward based on trajectory performance."""
    if policy_config.stupid_simple_reward_fn:
        return float(rubric.answer_correct)

    # Partial rewards (sum to less than 0.5)
    partial_rewards = 0
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0

    # Formatting errors
    if rubric.cant_parse_tool_call:
        return -2 + partial_rewards
    if rubric.bad_tool_call_name:
        return -1.9 + partial_rewards
    if rubric.bad_tool_call_args:
        return -1.8 + partial_rewards

    # Wrong answer
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1 + partial_rewards

    # No answer
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        return 0 + partial_rewards

    # Correct answer
    if rubric.answer_correct:
        reward = 1
        reward += 0.3 if rubric.sources_correct else 0
        reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0
        reward += 0.1 * (1 - rubric.num_turns / policy_config.max_turns)
        return reward

    raise ValueError("Rubric not handled properly")


class CorrectnessJudgeResponse(BaseModel):
    thinking: str = Field(description="Explanation of the reasoning process.")
    accept: bool = Field(description="Whether the AI answer should be accepted.")


@retry(stop=stop_after_attempt(3))
async def judge_correctness(
    answer: str, query: SyntheticQuery
) -> CorrectnessJudgeResponse:
    """Use an LLM to judge whether answer matches the expected answer."""
    system_prompt = textwrap.dedent("""
        You are given a question, the reference answer, and an AI-generated answer.
        
        Follow these steps:
        1. Identify EXACTLY what information the question is asking for.
        2. Extract ONLY the essential facts from the reference answer.
        3. Verify that every essential fact appears in the AI answer.
        4. If any essential fact is missing or contradicted, set accept to false.
        
        Return pure JSON with this schema:
        {
          "thinking": string,
          "accept": boolean
        }
    """)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Question: {query.question}\n"
                f"Reference answer: {query.answer}\n"
                f"AI answer: {answer}"
            ),
        },
    ]

    response = await acompletion(
        model="openai/gpt-4.1",
        messages=messages,
        caching=True,
        response_format=CorrectnessJudgeResponse,
    )

    first_choice = response.choices[0]
    raw_content = first_choice.message.content or "{}"

    try:
        return CorrectnessJudgeResponse.model_validate_json(raw_content)
    except Exception as e:
        return CorrectnessJudgeResponse(
            thinking=f"Parse error: {e}\nRaw: {raw_content}", accept=False
        )


class ProjectTrajectory(art.Trajectory):
    scenario: SyntheticQuery
    generated_answer: str | None = None


@retry(stop=stop_after_attempt(3))
async def rollout(
    model: art.Model[ProjectPolicyConfig],
    scenario: SyntheticQuery,
) -> ProjectTrajectory:
    """Execute a single trajectory rollout."""
    rubric = FinalRubric()
    traj = ProjectTrajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"email_inbox": scenario.inbox_address, "scenario_id": scenario.id},
        scenario=scenario,
    )

    system_prompt = textwrap.dedent(f"""\
        You are an email search agent. You are given a user query and tools to search emails. 
        Use the tools to find the answer to the user's query. You may take up to {model.config.max_turns} turns.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}
    """)

    async def search_emails_tool(keywords: list[str]) -> list[dict]:
        """Search the user's email inbox for emails matching keywords."""
        resp = search_emails(
            inbox=scenario.inbox_address,
            sent_before=scenario.query_date,
            keywords=keywords,
        )

        for r in resp:
            if r.message_id == scenario.message_ids[0]:
                rubric.ever_found_right_email = True
        return [asdict(r) for r in resp]

    async def read_email_tool(message_id: str) -> Email | dict:
        """Read the content of an email."""
        email_content = read_email(message_id)

        if message_id == scenario.message_ids[0]:
            rubric.ever_read_right_email = True
        if email_content is None:
            return {"error": "Email not found"}
        else:
            return email_content.model_dump()

    async def return_final_answer(answer: str, sources: list[str]):
        """Return the final answer with sources."""
        rubric.attempted_answer = True
        traj.generated_answer = answer

        if answer == "I don't know":
            rubric.returned_i_dont_know = True
        else:
            async with traj.track_duration("determine_if_answer_is_correct"):
                judge_response = await judge_correctness(answer, scenario)
                traj.log(f"Correctness judge response: {judge_response}")
                rubric.answer_correct = judge_response.accept
            rubric.sources_correct = scenario.message_ids[0] in sources

    tools = [search_emails_tool, read_email_tool, return_final_answer]
    traj.tools = [convert_to_openai_tool(t) for t in tools]

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]

    while not rubric.attempted_answer:
        rubric.num_turns += 1

        if rubric.num_turns > model.config.max_turns:
            rubric.ran_out_of_turns = True
            break

        litellm_model_name = model.config.litellm_model_name
        if litellm_model_name is None:
            litellm_model_name = f"hosted_vllm/{model.name}"

        async with traj.track_duration("llm_completion"):
            llm_response = await acompletion(
                model=litellm_model_name,
                base_url=model.inference_base_url,
                messages=traj.messages(),
                caching=not model.trainable,
                api_key=model.inference_api_key,
                max_completion_tokens=model.config.max_tokens,
                tools=traj.tools,
            )

        rubric.prompt_tokens += llm_response.usage.prompt_tokens
        rubric.completion_tokens += llm_response.usage.completion_tokens
        choice = llm_response.choices[0]

        # Handle only one tool call at a time
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 1:
            choice.message.tool_calls = choice.message.tool_calls[:1]
        traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))

        if choice.message.tool_calls is None:
            rubric.bad_tool_call_name = True
            break

        for tool_call in choice.message.tool_calls:
            if tool_call is None:
                rubric.bad_tool_call_args = True
                break

            try:
                tool_args = json.loads(tool_call.function.arguments)
            except Exception:
                rubric.bad_tool_call_args = True
                break

            for tool_fn in tools:
                if tool_fn.__name__ == tool_call.function.name:
                    try:
                        validated_fn = validate_call(tool_fn)
                        result = await validated_fn(**tool_args)
                        traj.messages_and_choices.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result),
                            }
                        )
                    except ValidationError as e:
                        rubric.bad_tool_call_args = True
                        traj.logs.append(
                            f"Invalid args for {tool_call.function.name}: {e}"
                        )
                        break
                    break
            else:
                rubric.bad_tool_call_name = True
                break

        if rubric.bad_tool_call_name or rubric.bad_tool_call_args:
            break

    reward = calculate_reward(model.config, rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()

    traj.finish()
    return traj


# ==================== GROUP JUDGE ====================


class Issue(BaseModel):
    label: str = Field(description="A short label for the issue.")
    explanation: str = Field(description="A human-readable explanation of the issue.")
    severity: Literal["minor", "major", "fatal"] = Field(
        description="The severity of the issue."
    )


class RolloutScore(BaseModel):
    rollout_id: str = Field(description="The id of the rollout being scored.")
    explanation: str = Field(
        description="A short explanation of why you gave this score."
    )
    score: float = Field(description="A score between 0 and 1.")
    issues: List[str] = Field(
        description="The list of labels for each issue identified."
    )


class JudgeGroupResponse(BaseModel):
    new_issues: List[Issue] = Field(
        description="Any new issues identified on the rollouts."
    )
    scores: List[RolloutScore] = Field(description="The scores for each rollout.")


DEFAULT_RUBRIC = """
- A rollout that achieves its goal should always get a significantly higher score than a rollout that does not achieve its goal.
- A rollout that achieves its goal more efficiently should get a higher score than a rollout that achieves its goal less efficiently.
- If one rollout is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
- You may give some partial credit for a rollout that makes progress towards its goal but does not complete it.
"""


class GroupJudge:
    """LLM-based judge for groups of rollouts."""

    def __init__(
        self,
        project: str,
        judge_model: str = "openai/o3",
        rubric: str = DEFAULT_RUBRIC,
        initial_issues: List[Issue] = None,
    ):
        self.project = project
        self.judge_model = judge_model
        self.rubric = rubric
        self.all_issues = initial_issues or [
            Issue(
                label="looping",
                explanation="The assistant repeats itself unnecessarily but is able to recover.",
                severity="minor",
            ),
            Issue(
                label="fatal_looping",
                explanation="The assistant began repeating itself and is unable to recover.",
                severity="fatal",
            ),
        ]

    async def judge(self, rollouts: list[ProjectTrajectory]) -> list[ProjectTrajectory]:
        """Score every trajectory in rollouts and write the score to traj.reward."""
        if not rollouts:
            return rollouts

        # Determine common prefix to save tokens
        message_lists = [traj.messages() for traj in rollouts]
        common_prefix_len = 0
        for i, msg in enumerate(message_lists[0]):
            if all(msg_list[i] == msg for msg_list in message_lists):
                common_prefix_len += 1
            else:
                break

        user_text = ""
        if common_prefix_len > 0:
            common_prefix_messages = message_lists[0][:common_prefix_len]
            user_text += (
                "<context>\n" + json.dumps(common_prefix_messages) + "\n</context>\n\n"
            )

        # Serialize rollouts without common prefix
        serialized_rollouts = []
        for idx, (traj, full_messages) in enumerate(
            zip(rollouts, message_lists), start=1
        ):
            traj.metrics["independent_reward"] = traj.reward
            trimmed_messages = full_messages[common_prefix_len:]
            serialized_rollouts.append(
                f'<rollout id="{idx}">\n'
                + json.dumps(trimmed_messages)
                + "\n</rollout>"
            )

        user_text += "Rollouts:\n\n" + "\n\n".join(serialized_rollouts)

        judge_prompt = f"""
All of the rollouts below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1.

Grading standards:
{self.rubric}

Existing issues:
{json.dumps([issue.model_dump() for issue in self.all_issues], indent=2)}
"""

        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_text},
        ]

        response = await acompletion(
            model=self.judge_model,
            messages=messages,
            response_format=JudgeGroupResponse,
            caching=True,
        )

        first_choice = response.choices[0]
        content = first_choice.message.content or "{}"
        parsed = JudgeGroupResponse.model_validate_json(content)

        # Merge new issues
        if parsed.new_issues:
            existing_labels = {fm.label for fm in self.all_issues}
            for fm in parsed.new_issues:
                if fm.label not in existing_labels:
                    self.all_issues.append(fm)
                    existing_labels.add(fm.label)

        # Apply scores
        for traj, score in zip(rollouts, parsed.scores):
            traj.metrics["group_judge_score"] = score.score
            traj.reward = (
                score.score
                if traj.metrics.get("failed_format_validation", 0) == 0
                else 0
            )
            traj.log(f"Judge group explanation: {score.explanation}")

            # Record issue metrics
            for issue in self.all_issues:
                metric_key = f"issues/{issue.severity}/{issue.label}"
                traj.metrics[metric_key] = issue.label in score.issues

        return rollouts


# ==================== TRAJECTORY REPORTING ====================


def report_trajectory(
    model: art.Model,
    trajectory: ProjectTrajectory,
    step: int = 0,
):
    """Report trajectory to Weave for logging."""
    client = weave.init(
        model.project, autopatch_settings=AutopatchSettings(disable_autopatch=True)
    )

    inputs = {
        "model": model.name,
        "scenario": trajectory.scenario,
        "step": step,
    }

    if isinstance(model, art.TrainableModel):
        inputs["base_model"] = model.base_model

    call = client.create_call("trajectory", inputs=inputs)
    client.finish_call(call, output={"tr": trajectory})


# ==================== MAIN TRAINING FUNCTION ====================


async def train(model: art.TrainableModel[ProjectPolicyConfig]):
    """Main training function for the email search agent."""
    generate_database()

    if model.config.training_config is None:
        raise ValueError("Training config is not set")

    group_judge = GroupJudge(
        project=model.project,
        judge_model=model.config.training_config.group_judge_model,
    )

    with LocalBackend() as backend:
        print(f"Pulling from S3 bucket: `{os.environ['BACKUP_BUCKET']}`")
        await backend._experimental_pull_from_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
            verbose=True,
        )
        await model.register(backend)

        print("Loading training data...")
        tc = model.config.training_config
        seed = tc.training_dataset_seed if tc is not None else None
        train_scenarios = load_synthetic_queries(
            split="train",
            limit=tc.training_dataset_size if tc is not None else None,
            seed=seed,
        )
        print("Loading validation data...")
        val_scenarios = load_synthetic_queries(
            split="test", limit=model.config.training_config.val_set_size
        )

        print(f"Training data size: {len(train_scenarios)}")
        print(f"Validation data size: {len(val_scenarios)}")

        train_iterator = iterate_dataset(
            train_scenarios,
            groups_per_step=model.config.training_config.groups_per_step,
            num_epochs=model.config.training_config.num_epochs,
            initial_step=await model.get_step(),
        )

        for batch, epoch, global_step, epoch_step in train_iterator:
            if global_step % model.config.training_config.eval_steps == 0:
                print(f"\n--- Evaluating at Iteration {global_step} ---")
                # Note: Evaluation/benchmarking code removed as requested
                await model.delete_checkpoints()
                await backend._experimental_push_to_s3(
                    model,
                    s3_bucket=os.environ["BACKUP_BUCKET"],
                )

            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            rollout(model, scenario)
                            for _ in range(
                                model.config.training_config.trajectories_per_group
                            )
                        )
                    )
                    for scenario in batch
                )
            )

            # Apply group judge if configured
            training_cfg = model.config.training_config
            if training_cfg.use_judge_group_variant is not None:
                judge_tasks = [
                    group_judge.judge(cast(list[ProjectTrajectory], g.trajectories))
                    for g in groups
                ]

                results = await asyncio.gather(*judge_tasks, return_exceptions=True)

                successful_groups = []
                for grp_idx, (g, res) in enumerate(zip(groups, results)):
                    if isinstance(res, Exception):
                        print(
                            f"WARNING:JUDGE_GROUP_FAILED group={grp_idx} step={global_step}: {res!r}"
                        )
                    else:
                        successful_groups.append(g)

                groups = successful_groups

                for g in groups:
                    for t in g.trajectories:
                        report_trajectory(model, t, global_step)

                if not groups:
                    print(
                        f"WARNING:ALL_JUDGE_GROUPS_FAILED step={global_step}; skipping training step"
                    )
                    continue

            # Filter groups by reward standard deviation
            if (
                training_cfg.minimum_reward_std_dev is not None
                and training_cfg.minimum_reward_std_dev > 0
            ):
                filtered_groups = []
                for grp_idx, g in enumerate(groups):
                    rewards = [t.reward for t in g.trajectories]
                    if len(rewards) < 2:
                        std_dev = 0.0
                    else:
                        std_dev = statistics.pstdev(rewards)
                    if std_dev < training_cfg.minimum_reward_std_dev:
                        print(
                            f"WARNING:REWARD_STD_DEV_TOO_LOW group={grp_idx} step={global_step} stddev={std_dev:.4f}; dropping group"
                        )
                        continue
                    filtered_groups.append(g)

                groups = filtered_groups

                if not groups:
                    print(
                        f"WARNING:ALL_GROUPS_DROPPED_LOW_STD_DEV step={global_step}; skipping training step"
                    )
                    continue

            await model.train(
                groups,
                config=art.TrainConfig(
                    learning_rate=model.config.training_config.learning_rate
                ),
            )

        # Final evaluation and backup
        await backend._experimental_push_to_s3(
            model,
            s3_bucket=os.environ["BACKUP_BUCKET"],
        )
        print("Training finished.")


if __name__ == "__main__":
    model = art.TrainableModel(
        name="email-agent-002",
        project="email_agent",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=ProjectPolicyConfig(
            max_turns=10,
            log_to_openpipe=True,
            training_config=TrainingConfig(
                trajectories_per_group=6,
                groups_per_step=8,
                learning_rate=1.2e-5,
                eval_steps=30,
                val_set_size=100,
                training_dataset_size=4000,
                num_epochs=1,
            ),
        ),
    )
    asyncio.run(train(model))
