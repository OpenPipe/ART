from pydantic import BaseModel
from typing import List, Optional, Literal


class SyntheticQuery(BaseModel):
    id: int
    question: str
    answer: str
    message_ids: List[str]  # message_ids (strings) of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]


class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []  # Populated from recipients table
    cc_addresses: List[str] = []  # Populated from recipients table
    bcc_addresses: List[str] = []  # Populated from recipients table
    body: Optional[str] = None
    file_name: Optional[str] = None
