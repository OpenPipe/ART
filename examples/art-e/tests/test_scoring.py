from art_e.data import validation_scenarios
from art_e.scoring import grade_answer


def test_grade_rewards_correct_answer_with_source() -> None:
    scenario = validation_scenarios[0]

    grade = grade_answer(
        scenario,
        "The team offsite is April 9 at the North Pier studio.",
        ["msg_offsite"],
    )

    assert grade.reward == 1.0
    assert grade.answer_match is True
    assert grade.cited_expected_message is True


def test_grade_penalizes_missing_answer() -> None:
    scenario = validation_scenarios[0]

    grade = grade_answer(scenario, "", [])

    assert grade.reward == 0.0
    assert grade.answer_match is False
    assert grade.cited_expected_message is False
