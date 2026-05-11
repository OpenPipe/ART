from dataclasses import dataclass
import re

from .types import Scenario


@dataclass(frozen=True)
class AnswerGrade:
    reward: float
    answer_match: bool
    cited_expected_message: bool


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def answer_token_recall(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & _tokens(candidate)) / len(reference_tokens)


def grade_answer(
    scenario: Scenario,
    answer: str | None,
    source_ids: list[str] | None,
) -> AnswerGrade:
    if not answer:
        return AnswerGrade(reward=0.0, answer_match=False, cited_expected_message=False)

    recall = answer_token_recall(scenario.answer, answer)
    cited_expected_message = bool(
        set(source_ids or []) & set(scenario.expected_message_ids)
    )
    answer_match = recall >= 0.6

    reward = 0.75 * recall + (0.25 if cited_expected_message else 0.0)
    return AnswerGrade(
        reward=min(reward, 1.0),
        answer_match=answer_match,
        cited_expected_message=cited_expected_message,
    )
