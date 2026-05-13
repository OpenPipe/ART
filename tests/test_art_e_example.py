from examples.art_e.scenarios import (
    SCENARIOS,
    parse_json_command,
    read_email,
    score_answer,
    search_emails,
)


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
