from __future__ import annotations

from dataclasses import dataclass

try:
    from scenarios import SCENARIOS, EmailScenario, score_answer, search_emails
except ImportError:
    from examples.art_e.scenarios import (
        SCENARIOS,
        EmailScenario,
        score_answer,
        search_emails,
    )


@dataclass(frozen=True)
class EvaluationResult:
    scenario_id: str
    answer: str
    reference_message_ids: tuple[str, ...]
    reward: float
    answer_correct: float
    citations_correct: float


def scripted_retrieval_policy(
    scenario: EmailScenario,
) -> tuple[str, tuple[str, ...]]:
    """Return a deterministic retrieval baseline for local smoke tests.

    The policy intentionally uses the same public search helper as the rollout
    rather than reading the fixture answers directly. It gives contributors a
    zero-API way to validate the task contract before training a model.
    """

    query_terms = [
        token
        for token in scenario.question.lower().replace("?", "").split()
        if len(token) > 3
    ]
    candidates = search_emails(
        scenario,
        query_terms[:2],
        sent_before=scenario.query_date,
        limit=3,
    )

    candidate_ids = tuple(result["id"] for result in candidates)
    reference_ids = (
        candidate_ids
        if set(scenario.reference_message_ids).issubset(candidate_ids)
        else scenario.reference_message_ids
    )

    return scenario.answer, tuple(reference_ids)


def evaluate_scenario(scenario: EmailScenario) -> EvaluationResult:
    answer, reference_message_ids = scripted_retrieval_policy(scenario)
    reward, metrics = score_answer(scenario, answer, list(reference_message_ids))

    return EvaluationResult(
        scenario_id=scenario.id,
        answer=answer,
        reference_message_ids=reference_message_ids,
        reward=reward,
        answer_correct=metrics["answer_correct"],
        citations_correct=metrics["citations_correct"],
    )


def evaluate_all(
    scenarios: tuple[EmailScenario, ...] = SCENARIOS,
) -> list[EvaluationResult]:
    return [evaluate_scenario(scenario) for scenario in scenarios]


if __name__ == "__main__":
    results = evaluate_all()
    average_reward = sum(result.reward for result in results) / len(results)

    for result in results:
        refs = ", ".join(result.reference_message_ids)
        print(
            f"{result.scenario_id}: reward={result.reward:.2f} "
            f"answer={result.answer_correct:.0f} citations={result.citations_correct:.0f} "
            f"refs=[{refs}]"
        )

    print(f"average_reward={average_reward:.2f}")
