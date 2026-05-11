from art_e.data import validation_scenarios
from art_e.email_tools import LocalInbox
from art_e.scoring import grade_answer


def main() -> None:
    scenario = validation_scenarios[0]
    inbox = LocalInbox(scenario.messages)
    matching = inbox.search(scenario.inbox_address, ["offsite"], scenario.query_date)
    answer = "The offsite is on April 9 at the North Pier studio."
    grade = grade_answer(scenario, answer, [matching[0].message_id])

    print(f"Scenario: {scenario.question}")
    print(f"Retrieved: {matching[0].subject}")
    print(f"Reward: {grade.reward:.2f}")


if __name__ == "__main__":
    main()
