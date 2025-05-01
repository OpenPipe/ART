import argparse
import os
import s3fs
import tempfile
from huggingface_hub import HfApi, upload_folder
from dotenv import load_dotenv
from run_training import models
from training_helpers import RunConfig, get_s3_model_path

load_dotenv()


def push_to_hub(model_key: str, hf_username: str):
    """Downloads a model from S3 and pushes it to Hugging Face Hub."""
    if model_key not in models:
        raise ValueError(f"Model key '{model_key}' not found in configured models.")

    config: RunConfig = models[model_key]
    print(f"Preparing to push model: {config.run_name}")

    s3_path = get_s3_model_path(config.run_name)
    repo_id = f"{hf_username}/roflbot_rm_{config.run_name}"
    print(f"Target S3 path: {s3_path}")
    print(f"Target Hugging Face repo ID: {repo_id}")

    # Ensure HF token is available (reads from env HUGGING_FACE_HUB_TOKEN or login)
    api = HfApi()
    api.whoami()  # Checks login status, will raise if not logged in
    print(f"Authenticated to Hugging Face Hub as: {api.whoami()['name']}")

    # Create the repository on the Hub if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Ensured repository '{repo_id}' exists on the Hub.")

    s3 = s3fs.S3FileSystem()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading model from {s3_path} to temporary directory {tmpdir}...")
        try:
            s3.get(s3_path, tmpdir, recursive=True)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading from S3: {e}")
            # Check if the S3 path is empty or doesn't exist
            if not s3.exists(s3_path):
                print(f"Error: S3 path {s3_path} does not exist or is empty.")
            else:
                print(f"Contents of {s3_path}: {s3.ls(s3_path)}")
            raise

        # --- Delete existing files on Hugging Face Hub ---
        print(f"Checking for existing files in Hugging Face repo: {repo_id}")
        try:
            existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            # Filter out .gitattributes if present, as it's often managed by HF
            files_to_delete = [f for f in existing_files if f != ".gitattributes"]

            if files_to_delete:
                print(
                    f"Deleting {len(files_to_delete)} existing files from {repo_id}..."
                )
                api.delete_files(
                    repo_id=repo_id,
                    # Use delete_patterns, passing the list of exact filenames
                    delete_patterns=files_to_delete,
                    repo_type="model",
                    commit_message="Clear existing model files before upload",
                )
                print("Existing files deleted.")
            else:
                print("No existing files to delete (or only .gitattributes found).")
        except Exception as e:
            print(f"Warning: Could not delete existing files from repo {repo_id}: {e}")
            # Decide if this should be a fatal error or just a warning
            # For now, we'll proceed with the upload attempt
        # --- End Deletion ---

        print(
            f"Uploading contents of {tmpdir} to Hugging Face Hub repository {repo_id}..."
        )
        try:
            # Define the path to the actual model contents within the temp directory
            local_model_path = os.path.join(
                tmpdir, config.run_name
            )  # Expecting s3 download to create run_name subdir

            # Check if the expected subdirectory exists
            if not os.path.isdir(local_model_path):
                # Fallback if the subdirectory wasn't created as expected
                print(
                    f"Warning: Expected subdirectory '{config.run_name}' not found in {tmpdir}. Uploading root of temp directory."
                )
                local_model_path = tmpdir
            else:
                print(f"Uploading from: {local_model_path}")

            upload_folder(
                folder_path=local_model_path,  # Upload contents from the subdirectory
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload model {config.run_name}",
                ignore_patterns=[
                    "dataset_predictions.parquet",
                    "training_args.bin",
                ],  # Exclude specified files
            )
            print("Upload complete.")
        except Exception as e:
            print(f"Error uploading to Hugging Face Hub: {e}")
            raise

    print(f"Successfully pushed model {config.run_name} to {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a model from S3 and push to Hugging Face Hub."
    )
    parser.add_argument(
        "--model", required=True, type=str, help="Model key (e.g., 002) to push."
    )
    # Optional: Allow overriding HF username via CLI
    parser.add_argument(
        "--hf_user", type=str, default="corbt", help="Hugging Face username."
    )

    args = parser.parse_args()

    # Ensure REMOTE_BUCKET is set before proceeding
    if not os.getenv("REMOTE_BUCKET"):
        raise ValueError(
            "Environment variable REMOTE_BUCKET is not set. Please ensure it's in your .env file or environment."
        )

    print("Starting push process...")
    push_to_hub(args.model, args.hf_user)
    print("Push process finished.")
