import logging
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from art_e.data.local_email_db import DEFAULT_DB_PATH
from art_e.data.types_enron import Email

conn = None


def get_conn():
    global conn
    if conn is None:
        conn = sqlite3.connect(
            f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return conn


@dataclass
class SearchResult:
    message_id: str
    snippet: str


def search_emails(
    inbox: str,
    keywords: List[str],
    from_addr: Optional[str] = None,
    to_addr: Optional[str] = None,
    sent_after: Optional[str] = None,
    sent_before: Optional[str] = None,
    max_results: int = 10,
) -> List[SearchResult]:
    """
    Searches the email database based on keywords, inbox, sender, recipient, and date range.

    Args:
        inbox: The email address of the user performing the search.
               Results include emails sent from or to (inc. cc/bcc) this address.
        keywords: A list of keywords that must all appear in the subject or body.
        from_addr: Optional email address to filter emails sent *from*.
        to_addr: Optional email address to filter emails sent *to* (inc. cc/bcc).
        sent_after: Optional date string 'YYYY-MM-DD'. Filters for emails sent on or after this date.
        sent_before: Optional date string 'YYYY-MM-DD'. Filters for emails sent before this date.
        max_results: The maximum number of results to return. Cannot exceed 10.

    Returns:
        A list of SearchResult objects, each containing 'message_id' and 'snippet'.
        Returns an empty list if no results are found or an error occurs.
    """

    # Initialize sql and params
    sql: Optional[str] = None
    params: List[str | int] = []

    cursor = get_conn().cursor()

    # --- Build Query ---
    where_clauses: List[str] = []

    # 1. Keywords (FTS)
    if not keywords:
        return []

    if max_results > 10:
        raise ValueError("max_results must be less than or equal to 10.")

    # FTS5 default is AND, so just join keywords. Escape quotes for safety.
    fts_query = " ".join(f""" "{k.replace('"', '""')}" """ for k in keywords)
    where_clauses.append("fts.emails_fts MATCH ?")
    params.append(fts_query)

    # 2. Inbox filter (must be from OR to/cc/bcc the inbox user)
    # Use the composite index idx_recipients_address_email here
    where_clauses.append(
        """
        (e.from_address = ? OR EXISTS (
            SELECT 1 FROM recipients r_inbox
            WHERE r_inbox.recipient_address = ? AND r_inbox.email_id = e.id
        ))
        """
    )
    params.extend([inbox, inbox])

    # 3. Optional From filter
    if from_addr:
        where_clauses.append("e.from_address = ?")
        params.append(from_addr)

    # 4. Optional To filter (includes to, cc, bcc)
    # Use the composite index idx_recipients_address_email here
    if to_addr:
        where_clauses.append(
            """
            EXISTS (
                SELECT 1 FROM recipients r_to
                WHERE r_to.recipient_address = ? AND r_to.email_id = e.id
            )
            """
        )
        params.append(to_addr)

    # 5. Optional Sent After filter
    if sent_after:
        # Assumes date format 'YYYY-MM-DD'
        # Compare against the start of the day
        where_clauses.append("e.date >= ?")
        params.append(f"{sent_after} 00:00:00")

    # 6. Optional Sent Before filter
    if sent_before:
        # Assumes date format 'YYYY-MM-DD'
        # Compare against the start of the day (exclusive)
        where_clauses.append("e.date < ?")
        params.append(f"{sent_before} 00:00:00")

    # --- Construct Final Query ---
    # snippet(<table>, <column_index>, <highlight_start>, <highlight_end>, <ellipsis>, <tokens>)
    # -1 means highlight across all columns (subject, body)
    sql = f"""
        SELECT
            e.message_id,
            snippet(emails_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
        FROM
            emails e JOIN emails_fts fts ON e.id = fts.rowid
        WHERE
            {" AND ".join(where_clauses)}
        ORDER BY
            e.date DESC -- Order by date for relevance
        LIMIT ?;
    """
    params.append(max_results)

    # --- Execute and Fetch ---
    logging.debug(f"Executing SQL: {sql}")
    logging.debug(f"With params: {params}")
    cursor.execute(sql, params)
    results = cursor.fetchall()

    # Format results
    formatted_results = [
        SearchResult(message_id=row[0], snippet=row[1]) for row in results
    ]
    return formatted_results


def read_email(message_id: str) -> Optional[Email]:
    """
    Retrieves a single email by its message_id from the database.

    Args:
        message_id: The unique identifier of the email to retrieve.

    Returns:
        An Email object containing the details of the found email,
        or None if the email is not found or an error occurs.
    """

    cursor = get_conn().cursor()

    # --- Query for Email Core Details ---
    email_sql = """
        SELECT message_id, date, subject, from_address, body, file_name
        FROM emails
        WHERE message_id = ?;
    """
    cursor.execute(email_sql, (message_id,))
    email_row = cursor.fetchone()

    if not email_row:
        logging.warning(f"Email with message_id '{message_id}' not found.")
        return None

    (
        msg_id,
        date,
        subject,
        from_addr,
        body,
        file_name,
    ) = email_row

    # --- Query for Recipients ---
    recipients_sql = """
        SELECT recipient_address, recipient_type
        FROM recipients
        WHERE email_id = ?;
    """
    cursor.execute(recipients_sql, (message_id,))
    recipient_rows = cursor.fetchall()

    to_addresses: List[str] = []
    cc_addresses: List[str] = []
    bcc_addresses: List[str] = []

    for addr, type in recipient_rows:
        type_lower = type.lower()
        if type_lower == "to":
            to_addresses.append(addr)
        elif type_lower == "cc":
            cc_addresses.append(addr)
        elif type_lower == "bcc":
            bcc_addresses.append(addr)

    # --- Construct Email Object ---
    email_obj = Email(
        message_id=msg_id,  # Convert to string to match Pydantic model
        date=date,
        subject=subject,
        from_address=from_addr,
        to_addresses=to_addresses,
        cc_addresses=cc_addresses,
        bcc_addresses=bcc_addresses,
        body=body,
        file_name=file_name,
    )

    return email_obj
