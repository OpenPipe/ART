import sqlite3
import os
import logging
from datasets import load_dataset, Dataset, Features, Value, Sequence
from tqdm import tqdm
from datetime import datetime

# Resolve paths relative to this file so it works regardless of the current working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Database will live in "../data/enron_emails.db" relative to project root
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "..", "..", "data", "enron_emails.db")

DEFAULT_REPO_ID = "corbt/enron-emails"


# --- Database Schema ---
SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS recipients;
DROP TABLE IF EXISTS emails_fts;
DROP TABLE IF EXISTS emails;

CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    subject TEXT,
    from_address TEXT,
    date TEXT, -- Store as ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    body TEXT,
    file_name TEXT
);

CREATE TABLE recipients (
    email_id INTEGER,
    recipient_address TEXT,
    recipient_type TEXT, -- 'to', 'cc', 'bcc'
    FOREIGN KEY(email_id) REFERENCES emails(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_recipients_address ON recipients(recipient_address);
CREATE INDEX idx_recipients_type ON recipients(recipient_type);
CREATE INDEX idx_recipients_email_id ON recipients(email_id);
CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject,
    body,
    content='emails',
    content_rowid='id'
);

CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts (rowid, subject, body)
    VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
    DELETE FROM emails_fts WHERE rowid=old.id;
END;

CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
    UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
END;

INSERT INTO emails_fts (rowid, subject, body) SELECT id, subject, body FROM emails;
"""


# --- Functions ---


def download_dataset(repo_id: str) -> Dataset:
    """Downloads the dataset from Hugging Face Hub."""
    logging.info(f"Attempting to download dataset from Hugging Face Hub: {repo_id}")
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    dataset_obj = load_dataset(repo_id, features=expected_features, split="train")
    # Basic type check remains useful
    if not isinstance(dataset_obj, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset_obj)}")
    logging.info(
        f"Successfully loaded dataset '{repo_id}' with {len(dataset_obj)} records."
    )
    return dataset_obj


def create_database(db_path: str):
    """Creates the SQLite database and tables."""
    logging.info(f"Creating SQLite database and tables at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    conn.close()
    logging.info("Database tables created successfully.")


def populate_database(db_path: str, dataset: Dataset):
    """Populates the database with data from the Hugging Face dataset."""
    logging.info(f"Populating database {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Performance Pragmas ---
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    record_count = 0
    skipped_count = 0  # Keep track of skipped emails due to filters
    duplicate_count = 0  # Keep track of skipped duplicate emails
    processed_emails = set()  # Track (subject, body, from) tuples to dedupe

    conn.execute("BEGIN TRANSACTION;")  # Single transaction for bulk insert

    for email_data in tqdm(dataset, desc="Inserting emails"):
        assert isinstance(email_data, dict)
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list_raw = email_data["to"]
        cc_list_raw = email_data["cc"]
        bcc_list_raw = email_data["bcc"]

        # --- Data Cleaning and Filtering ---
        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        to_list = [str(addr) for addr in to_list_raw if addr]
        cc_list = [str(addr) for addr in cc_list_raw if addr]
        bcc_list = [str(addr) for addr in bcc_list_raw if addr]

        # Check body length
        if len(body) > 5000:
            logging.debug(
                f"Skipping email {message_id}: Body length > 5000 characters."
            )
            skipped_count += 1
            continue

        # Check total recipients
        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)
        if total_recipients > 30:
            logging.debug(
                f"Skipping email {message_id}: Total recipients ({total_recipients}) > 30."
            )
            skipped_count += 1
            continue
        # --- End Filtering ---

        # --- Deduplication Check ---
        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            logging.debug(
                f"Skipping duplicate email (Subject: {subject[:50]}..., From: {from_address})"
            )
            duplicate_count += 1
            continue
        else:
            processed_emails.add(email_key)
        # --- End Deduplication ---

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, subject, from_address, date_str, body, file_name),
        )
        email_pk_id = cursor.lastrowid

        recipient_data = []
        for addr in to_list:
            recipient_data.append((email_pk_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((email_pk_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((email_pk_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
                """,
                recipient_data,
            )
        record_count += 1

    conn.commit()
    conn.close()
    logging.info(f"Successfully inserted {record_count} email records.")
    if skipped_count > 0:
        logging.info(
            f"Skipped {skipped_count} email records due to length or recipient limits."
        )
    if duplicate_count > 0:
        logging.info(
            f"Skipped {duplicate_count} duplicate email records (based on subject, body, from)."
        )


def create_indexes_and_triggers(db_path: str):
    """Creates indexes and triggers on the populated database."""
    logging.info(f"Creating indexes and triggers for database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()
    logging.info("Indexes and triggers created successfully.")


def generate_database(overwrite: bool = False):
    """
    Generates the SQLite database from the specified Hugging Face dataset.
    Simplified version without extensive error handling.

    Args:
        repo_id: The Hugging Face repository ID for the dataset.
        db_path: The path where the SQLite database file should be created.
        overwrite: If True, any existing database file at db_path will be removed.
    """
    logging.info(
        f"Starting database generation for repo '{DEFAULT_REPO_ID}' at '{DEFAULT_DB_PATH}'"
    )
    logging.info(f"Overwrite existing database: {overwrite}")

    db_dir = os.path.dirname(DEFAULT_DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        logging.info(f"Creating data directory: {db_dir}")
        os.makedirs(db_dir)

    if overwrite and os.path.exists(DEFAULT_DB_PATH):
        logging.warning(f"Removing existing database file: {DEFAULT_DB_PATH}")
        os.remove(DEFAULT_DB_PATH)
    elif not overwrite and os.path.exists(DEFAULT_DB_PATH):
        # If not overwriting and file exists, subsequent steps might fail or behave unexpectedly.
        # We are removing the explicit error here as requested.
        logging.warning(
            f"Database file {DEFAULT_DB_PATH} exists and overwrite is False. Assuming file is already generated."
        )
        return

    # 1. Download dataset
    dataset = download_dataset(DEFAULT_REPO_ID)

    # 2. Create database schema (Tables only)
    # Note: This will fail if overwrite=False and the file exists with incompatible schema/data.
    create_database(DEFAULT_DB_PATH)

    # 3. Populate database
    populate_database(DEFAULT_DB_PATH, dataset)

    # 4. Create Indexes and Triggers
    create_indexes_and_triggers(DEFAULT_DB_PATH)

    logging.info(f"Database generation process completed for {DEFAULT_DB_PATH}.")


if __name__ == "__main__":
    generate_database(overwrite=True)
