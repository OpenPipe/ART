from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import re
from typing import Any, TypedDict


class EmailMessage(TypedDict):
    id: str
    sender: str
    recipients: list[str]
    sent_at: str
    subject: str
    body: str


class SearchResult(TypedDict):
    id: str
    sender: str
    sent_at: str
    subject: str
    snippet: str


@dataclass(frozen=True)
class EmailScenario:
    id: str
    inbox_address: str
    query_date: str
    question: str
    answer: str
    reference_message_ids: tuple[str, ...]
    inbox: tuple[EmailMessage, ...]


SCENARIOS: tuple[EmailScenario, ...] = (
    EmailScenario(
        id="quarterly-budget-owner",
        inbox_address="alex@acme.test",
        query_date="2026-03-15",
        question=(
            "Who owns the quarterly budget deck, and when did they say the "
            "draft would be ready?"
        ),
        answer="Maya owns the quarterly budget deck and said the draft would be ready by Friday.",
        reference_message_ids=("msg-budget-2",),
        inbox=(
            {
                "id": "msg-budget-1",
                "sender": "nora@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-03-01",
                "subject": "Budget planning kickoff",
                "body": "Let's collect assumptions for the quarterly budget review.",
            },
            {
                "id": "msg-budget-2",
                "sender": "maya@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-03-08",
                "subject": "Quarterly budget deck owner",
                "body": (
                    "I will own the quarterly budget deck. The draft will be "
                    "ready by Friday so finance can review it before Monday."
                ),
            },
            {
                "id": "msg-budget-3",
                "sender": "finance@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-03-12",
                "subject": "Reminder: travel budget",
                "body": "Please submit travel budget updates before the end of the month.",
            },
        ),
    ),
    EmailScenario(
        id="customer-escalation-time",
        inbox_address="alex@acme.test",
        query_date="2026-04-05",
        question=(
            "What time is the Northwind escalation call, and which customer "
            "issue should be discussed first?"
        ),
        answer="The Northwind escalation call is at 3 PM UTC, and the login outage should be discussed first.",
        reference_message_ids=("msg-northwind-2",),
        inbox=(
            {
                "id": "msg-northwind-1",
                "sender": "support@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-04-01",
                "subject": "Northwind weekly notes",
                "body": "Northwind asked for the usual weekly usage report.",
            },
            {
                "id": "msg-northwind-2",
                "sender": "sam@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-04-03",
                "subject": "Northwind escalation call",
                "body": (
                    "The Northwind escalation call is at 3 PM UTC. Please "
                    "discuss the login outage first, then the reporting delay."
                ),
            },
            {
                "id": "msg-northwind-3",
                "sender": "calendar@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-04-04",
                "subject": "Daily standup moved",
                "body": "Daily standup is moving to 10 AM local time next week.",
            },
        ),
    ),
    EmailScenario(
        id="contract-renewal-discount",
        inbox_address="alex@acme.test",
        query_date="2026-04-20",
        question="What renewal discount did Finch Labs approve?",
        answer="Finch Labs approved a 12% renewal discount.",
        reference_message_ids=("msg-finch-3",),
        inbox=(
            {
                "id": "msg-finch-1",
                "sender": "sales@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-04-09",
                "subject": "Finch Labs renewal",
                "body": "Finch Labs is reviewing renewal pricing this week.",
            },
            {
                "id": "msg-finch-2",
                "sender": "legal@acme.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-04-11",
                "subject": "Finch Labs contract language",
                "body": "The renewal contract language is approved from legal.",
            },
            {
                "id": "msg-finch-3",
                "sender": "riley@finch.test",
                "recipients": ["alex@acme.test"],
                "sent_at": "2026-04-18",
                "subject": "Re: Finch Labs renewal",
                "body": "We approved the 12% renewal discount. Please send the final order form.",
            },
        ),
    ),
)


def search_emails(
    scenario: EmailScenario,
    keywords: list[str],
    sent_before: str | None = None,
    limit: int = 5,
) -> list[SearchResult]:
    normalized_keywords = [keyword.lower() for keyword in keywords if keyword.strip()]
    try:
        before = date.fromisoformat(sent_before) if sent_before else None
    except ValueError:
        before = None
    results: list[SearchResult] = []

    for email in scenario.inbox:
        sent_at = date.fromisoformat(email["sent_at"])
        if before and sent_at >= before:
            continue

        haystack = " ".join(
            [email["sender"], email["subject"], email["body"]]
        ).lower()
        if normalized_keywords and not all(
            keyword in haystack for keyword in normalized_keywords
        ):
            continue

        results.append(
            {
                "id": email["id"],
                "sender": email["sender"],
                "sent_at": email["sent_at"],
                "subject": email["subject"],
                "snippet": email["body"][:160],
            }
        )

    return results[:limit]


def read_email(scenario: EmailScenario, message_id: str) -> EmailMessage | None:
    for email in scenario.inbox:
        if email["id"] == message_id:
            return email
    return None


def parse_json_command(content: str, tag: str) -> dict[str, Any] | None:
    match = re.search(rf"<{tag}>\s*(\{{.*?\}})\s*</{tag}>", content, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9% ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def score_answer(
    scenario: EmailScenario,
    answer: str,
    reference_message_ids: list[str],
) -> tuple[float, dict[str, float]]:
    normalized_answer = normalize_text(answer)
    expected_answer = normalize_text(scenario.answer)

    answer_score = 1.0 if expected_answer in normalized_answer else 0.0
    expected_refs = set(scenario.reference_message_ids)
    provided_refs = set(reference_message_ids)
    citation_score = 1.0 if expected_refs.issubset(provided_refs) else 0.0
    reward = (0.75 * answer_score) + (0.25 * citation_score)

    return reward, {
        "answer_correct": answer_score,
        "citations_correct": citation_score,
    }
