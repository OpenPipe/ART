# %%
import pandas as pd
from datasets import load_dataset
import sqlite3
import collections.abc
from art_e.data.local_email_db import DEFAULT_DB_PATH
from IPython.display import display, HTML
import json

# Load the dataset from Hugging Face
dataset = load_dataset("corbt/enron_emails_sample_questions")

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset["train"])  # type: ignore

# Print basic dataset info
print(f"Dataset size: {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")
print(f"Number of unique inboxes: {df['inbox_address'].nunique()}")
print(f"Average realism score: {df['how_realistic'].mean():.2f}")

# Connect to the SQLite database
conn = sqlite3.connect(DEFAULT_DB_PATH)
cursor = conn.cursor()

samples = df[df.how_realistic > 0.85].sample(frac=1).head(50)

# Display the samples
for sample in samples.itertuples():
    html_output = f"<h2>--- Question {sample.id} ---</h2>"
    html_output += f"<p><b>Question:</b> {sample.question}</p>"
    html_output += f"<p><b>Answer:</b> {sample.answer}</p>"
    html_output += f"<p><b>Inbox:</b> {sample.inbox_address}</p>"
    html_output += f"<p><b>Message IDs:</b> {sample.message_ids}</p>"
    html_output += f"<p><b>Realism score:</b> {sample.how_realistic:.2f}</p>"
    html_output += f"<p><b>Query date:</b> {sample.query_date}</p>"

    # Look up and display the referenced emails
    html_output += "<h3>Referenced Emails:</h3>"

    # Handle different types for message_ids
    ids_raw = sample.message_ids
    message_ids_list = []
    if ids_raw is not None:
        if isinstance(ids_raw, collections.abc.Iterable) and not isinstance(
            ids_raw, (str, bytes)
        ):
            # Filter out None/NaN from the iterable
            message_ids_list = [
                item for item in ids_raw if item is not None and not pd.isna(item)
            ]  # type: ignore
        else:
            # Handle scalar: add only if it's not None
            if ids_raw is not None:  # Simplified check for scalar
                message_ids_list = [ids_raw]

    if not message_ids_list:
        html_output += "<p><i>No valid message IDs provided or found.</i></p>"
    else:
        # Ensure IDs are strings for the DB query
        valid_msg_ids = [str(msg_id) for msg_id in message_ids_list]

        for msg_id in valid_msg_ids:
            # Query the database for the email with this message_id
            cursor.execute(
                """
                SELECT e.id, e.subject, e.from_address, e.date, e.body
                FROM emails e
                WHERE e.message_id = ?
                """,
                (msg_id,),
            )
            email = cursor.fetchone()

            if email:
                email_id, subject, from_address, date, body = email

                # Get recipients
                cursor.execute(
                    """
                    SELECT recipient_address, recipient_type
                    FROM recipients
                    WHERE email_id = ?
                    """,
                    (email_id,),
                )
                recipients = cursor.fetchall()

                # Format recipients by type
                to_list = [addr for addr, type in recipients if type == "to"]
                cc_list = [addr for addr, type in recipients if type == "cc"]
                bcc_list = [addr for addr, type in recipients if type == "bcc"]

                # Add email details to HTML
                html_output += f"<h4>Message ID: {msg_id}</h4>"
                html_output += f"<p><b>Subject:</b> {subject}</p>"
                html_output += f"<p><b>From:</b> {from_address}</p>"
                html_output += f"<p><b>Date:</b> {date}</p>"
                html_output += f"<p><b>To:</b> {', '.join(to_list)}</p>"
                if cc_list:
                    html_output += f"<p><b>CC:</b> {', '.join(cc_list)}</p>"
                if bcc_list:
                    html_output += f"<p><b>BCC:</b> {', '.join(bcc_list)}</p>"
                # Escape HTML in body and display truncated version
                escaped_body = (
                    body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )
                truncated_body = (
                    escaped_body[:500] + "..."
                    if len(escaped_body) > 500
                    else escaped_body
                )
                html_output += f"<p><b>Body:</b><br><pre style='white-space: pre-wrap;'>{truncated_body}</pre></p>"
            else:
                html_output += (
                    f"<p><i>Message ID: {msg_id} - Not found in database</i></p>"
                )

    display(HTML(html_output))

# Close the database connection
conn.close()

# %%

from IPython.display import display, HTML
import sqlite3
import collections.abc
from art_e.data.local_email_db import DEFAULT_DB_PATH

# good_examples = [2796, 749, 4097, 3916, 1273, 5374, 1139, 4102, 4092, 543, 628, 2770]
good_examples = [1139, 543, 3916, 2770, 2796]

# Filter the DataFrame for the good examples and select relevant columns
good_df = df[df["id"].isin(good_examples)][["id", "question", "answer"]]

# Reorder good_df based on the order in good_examples
good_df["id"] = pd.Categorical(good_df["id"], categories=good_examples, ordered=True)
good_df = good_df.sort_values("id").reset_index(drop=True)

# Connect to the SQLite database again
conn_good = sqlite3.connect(DEFAULT_DB_PATH)
cursor_good = conn_good.cursor()


# Function to get the source email body for a given question ID
def get_source_email_body(question_id, original_df, db_cursor):
    # Find the row in the original DataFrame
    row = original_df[original_df["id"] == question_id].iloc[0]
    ids_raw = row["message_ids"]

    message_ids_list = []
    if ids_raw is not None:
        # Check if ids_raw is iterable but not a string/bytes
        if isinstance(ids_raw, collections.abc.Iterable) and not isinstance(
            ids_raw, (str, bytes)
        ):
            # Filter out None/NaN using pd.isna for robustness across types
            message_ids_list = [
                item for item in ids_raw if item is not None and not pd.isna(item)
            ]
        elif not pd.isna(ids_raw):  # Handle scalar, ensuring it's not NaN/None
            message_ids_list = [ids_raw]

    if not message_ids_list:
        return None  # Or some placeholder string like "No source email found"

    # Take the first valid message ID found
    first_msg_id = str(message_ids_list[0])

    # Query the database for the email body
    db_cursor.execute("SELECT body FROM emails WHERE message_id = ?", (first_msg_id,))
    result = db_cursor.fetchone()

    if result:
        return result[0]  # Return the body
    else:
        return f"Email body not found for message_id: {first_msg_id}"  # Placeholder if not found


# Apply the function to create the new column
# Assumes 'df' from the first cell is available in the environment
good_df["source_email"] = good_df["id"].apply(
    lambda q_id: get_source_email_body(q_id, df, cursor_good)
)

# Close the database connection
conn_good.close()

# Convert the filtered DataFrame to an HTML table
# index=False prevents writing the DataFrame index as a column
# escape=False allows potential HTML within question/answer to render (use with caution)
html_table = good_df.to_html(index=False, escape=True)

# Display the HTML table
display(HTML(html_table))

# Convert the filtered DataFrame to a list of dictionaries
good_df_list = good_df.to_dict(orient="records")



good_df_json = json.dumps(good_df_list, indent=2)

# Print the JSON string
print(good_df_json)

for row in good_df.itertuples():
    pass
# %%
