from examples.art_e.scenarios import (
    SCENARIOS,
    parse_json_command,
    read_email,
    score_answer,
    search_emails,
)
from examples.art_e.evaluate import evaluate_all, scripted_retrieval_policy


def test_search_emails_finds_matching_message() -> None:
    scenario = SCENARIOS[0]

    results = search_emails(scenario, ["quarterly", "deck"])

    assert [result["id"] for result in results] == ["msg-budget-2"]


def test_read_email_returns_message_by_id() -> None:
    scenario = SCENARIOS[1]

    email = read_email(scenario, "msg-northwind-2")

    assert email is not None
    assert "3 PM UTC" in email["body"]


def test_score_answer_rewards_answer_and_citation() -> None:
    scenario = SCENARIOS[2]

    reward, metrics = score_answer(
        scenario,
        "Finch Labs approved a 12% renewal discount.",
        ["msg-finch-3"],
    )

    assert reward == 1.0
    assert metrics == {"answer_correct": 1.0, "citations_correct": 1.0}


def test_parse_json_command_ignores_invalid_json() -> None:
    assert parse_json_command("<search>{invalid}</search>", "search") is None


def test_scripted_retrieval_policy_matches_expected_reference() -> None:
    answer, references = scripted_retrieval_policy(SCENARIOS[0])

    assert "Maya owns the quarterly budget deck" in answer
    assert references == ("msg-budget-2",)


def test_offline_evaluation_scores_all_scenarios() -> None:
    results = evaluate_all()

    assert len(results) == len(SCENARIOS)
    assert all(result.reward == 1.0 for result in results)
