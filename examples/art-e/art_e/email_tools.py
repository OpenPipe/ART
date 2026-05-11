from datetime import date

from .types import EmailMessage


class LocalInbox:
    def __init__(self, messages: list[EmailMessage]):
        self._messages = {message.message_id: message for message in messages}

    def search(
        self,
        inbox_address: str,
        keywords: list[str],
        sent_before: str,
        limit: int = 5,
    ) -> list[EmailMessage]:
        query_terms = [term.casefold() for term in keywords if term.strip()]
        cutoff = date.fromisoformat(sent_before)
        scored: list[tuple[int, EmailMessage]] = []

        for message in self._messages.values():
            if message.to_address.casefold() != inbox_address.casefold():
                continue
            if date.fromisoformat(message.date) > cutoff:
                continue

            haystack = (
                f"{message.subject} {message.from_address} {message.body}".casefold()
            )
            score = sum(term in haystack for term in query_terms)
            if score:
                scored.append((score, message))

        scored.sort(key=lambda item: (-item[0], item[1].date, item[1].message_id))
        return [message for _, message in scored[:limit]]

    def read(self, message_id: str) -> EmailMessage | None:
        return self._messages.get(message_id)
