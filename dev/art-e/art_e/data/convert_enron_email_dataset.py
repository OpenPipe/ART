import os
import pandas as pd
import mailparser
import argparse
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
from datasets import Dataset


def download_enron_dataset():
    """Download the Enron dataset from Kaggle if it doesn't already exist"""
    # Check if data file already exists
    if os.path.exists("./data/emails.csv"):
        print("Enron dataset already exists in ./data, skipping download")
        return

    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()  # Requires kaggle.json credentials in ~/.kaggle/

    print("Downloading Enron email dataset...")
    api.dataset_download_files(
        "wcukierski/enron-email-dataset", path="./data", unzip=True
    )
    print("Dataset downloaded and extracted to ./data")


def parse_emails(csv_path, max_emails=100):
    """Parse the emails and extract structured data"""
    print("Loading emails from CSV...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} emails")
    if max_emails and max_emails < len(df):
        print(f"Limiting to {max_emails} emails")
        df = df.sample(max_emails, random_state=42)
    structured_emails = []
    print("Parsing emails...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            mail = mailparser.parse_from_string(row["message"])
            if len(mail.from_) != 1:
                print(f"Warning: Expected 1 from, got {len(mail.from_)}")
            if mail.from_[0][0] != "":
                print(f"Warning: Expected From, got '{mail.from_[0][0]}'")

            structured_email = {
                "message_id": mail.message_id,
                "subject": mail.subject,
                "from": mail.from_[0][1],
                "to": [t[1] for t in mail.to],
                "cc": [t[1] for t in mail.cc],
                "bcc": [t[1] for t in mail.bcc],
                "date": mail.date,
                "body": mail.body,
                "file_name": row["file"],
            }
            structured_emails.append(structured_email)
        except Exception as e:
            print(f"Error parsing email {i} (file: {row['file']}): {str(e)}")
            continue
    print(f"Successfully parsed {len(structured_emails)} out of {len(df)} emails")
    return structured_emails


def upload_to_huggingface(structured_emails, repo_id="corbt/enron-emails"):
    """Upload the structured emails to a HuggingFace dataset"""
    print(f"Preparing dataset for upload to {repo_id}...")

    # Convert to Hugging Face dataset directly
    dataset = Dataset.from_list(structured_emails)

    # Push to Hugging Face
    print(f"Uploading dataset to Hugging Face ({repo_id})...")
    dataset.push_to_hub(repo_id, private=False)

    print(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process Enron email dataset")
    parser.add_argument(
        "--max-emails",
        type=int,
        default=100,
        help="Maximum number of emails to process (default: 100)",
    )
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    # Download the dataset
    download_enron_dataset()

    # Parse emails
    structured_emails = parse_emails(
        csv_path="./data/emails.csv",
        max_emails=args.max_emails,
    )

    # Upload to HuggingFace
    upload_to_huggingface(structured_emails)

    print(f"Processing complete. {len(structured_emails)} emails parsed and uploaded.")


if __name__ == "__main__":
    main()
