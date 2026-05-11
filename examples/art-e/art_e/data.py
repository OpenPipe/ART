from .types import EmailMessage, Scenario


def _messages() -> list[EmailMessage]:
    return [
        EmailMessage(
            message_id="msg_budget_q4",
            subject="Q4 budget review",
            from_address="finance@example.com",
            to_address="user@example.com",
            date="2026-01-08",
            body=(
                "The Q4 budget review is scheduled for January 18 at 10 AM. "
                "Please bring the revised hiring forecast."
            ),
        ),
        EmailMessage(
            message_id="msg_project_shift",
            subject="Phoenix launch timeline",
            from_address="pm@example.com",
            to_address="user@example.com",
            date="2026-01-10",
            body=(
                "The Phoenix launch deadline moved to February 14. "
                "Design freeze remains January 24."
            ),
        ),
        EmailMessage(
            message_id="msg_offsite",
            subject="Team offsite logistics",
            from_address="ops@example.com",
            to_address="user@example.com",
            date="2026-01-11",
            body=(
                "The team offsite is on April 9 at the North Pier studio. "
                "Breakfast starts at 8:30 AM."
            ),
        ),
        EmailMessage(
            message_id="msg_vendor",
            subject="Vendor renewal",
            from_address="procurement@example.com",
            to_address="user@example.com",
            date="2026-01-12",
            body=(
                "The analytics vendor renewal is due February 2. "
                "Legal approved the updated data-processing addendum."
            ),
        ),
    ]


train_scenarios = [
    Scenario(
        id="budget-review",
        question="When is the Q4 budget review and what should I bring?",
        answer="The Q4 budget review is January 18 at 10 AM; bring the revised hiring forecast.",
        inbox_address="user@example.com",
        query_date="2026-01-15",
        expected_message_ids=["msg_budget_q4"],
        messages=_messages(),
    ),
    Scenario(
        id="phoenix-deadline",
        question="What is the new Phoenix launch deadline?",
        answer="The Phoenix launch deadline moved to February 14.",
        inbox_address="user@example.com",
        query_date="2026-01-15",
        expected_message_ids=["msg_project_shift"],
        messages=_messages(),
    ),
]

validation_scenarios = [
    Scenario(
        id="offsite-date",
        question="Where and when is the team offsite?",
        answer="The team offsite is April 9 at the North Pier studio.",
        inbox_address="user@example.com",
        query_date="2026-01-15",
        expected_message_ids=["msg_offsite"],
        messages=_messages(),
    ),
    Scenario(
        id="vendor-renewal",
        question="When is the analytics vendor renewal due?",
        answer="The analytics vendor renewal is due February 2.",
        inbox_address="user@example.com",
        query_date="2026-01-15",
        expected_message_ids=["msg_vendor"],
        messages=_messages(),
    ),
]
