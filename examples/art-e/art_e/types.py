from pydantic import BaseModel, Field


class EmailMessage(BaseModel):
    message_id: str
    subject: str
    from_address: str
    to_address: str
    date: str
    body: str

    def preview(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "subject": self.subject,
            "from_address": self.from_address,
            "date": self.date,
            "snippet": self.body[:180],
        }


class FinalAnswer(BaseModel):
    answer: str = Field(description="Answer to the user's question.")
    source_ids: list[str] = Field(description="Email message IDs used as evidence.")


class Scenario(BaseModel):
    id: str
    question: str
    answer: str
    inbox_address: str
    query_date: str
    expected_message_ids: list[str]
    messages: list[EmailMessage]
