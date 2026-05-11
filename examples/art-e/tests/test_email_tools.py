from art_e.data import validation_scenarios
from art_e.email_tools import LocalInbox


def test_search_filters_and_ranks_messages() -> None:
    scenario = validation_scenarios[0]
    inbox = LocalInbox(scenario.messages)

    results = inbox.search(
        scenario.inbox_address, ["offsite", "studio"], scenario.query_date
    )

    assert [message.message_id for message in results] == ["msg_offsite"]


def test_read_unknown_message_returns_none() -> None:
    scenario = validation_scenarios[0]
    inbox = LocalInbox(scenario.messages)

    assert inbox.read("missing") is None
