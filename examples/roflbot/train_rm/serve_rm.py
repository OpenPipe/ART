# To deploy:
# uv run modal deploy train_rm/serve_rm.py

"""
## To test (replace with your actual Modal deployment URL):

curl -X POST \
  https://openpipe-dev--roflbot-rm-011-serve-roflbot-rm.modal.run/score \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Why did the scarecrow win an award?",
    "text": "Because he was outstanding in his field!",
    "poster": "comedian_user",
    "created_at": "2024-08-15T10:30:00Z"
  }'
"""

import modal
import logging
import time
import os
from datetime import datetime
import json  # Added import
from peft import PeftModel  # Added import

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import s3fs
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from training_helpers import get_s3_model_path, format_joke_for_rm

# --- Configuration ---
LOCAL_MODEL_PATH = "/model_cache"  # Using a dedicated cache directory
ROFLBOT_RUN_NAME = "011"  # Specify the model run you want to serve
MODEL_MAX_LENGTH = 2048  # Max token length the model can score
SERVICE_VERSION = f"roflbot-rm-{ROFLBOT_RUN_NAME}-0.1"


class RoflbotScoreRequest(BaseModel):
    title: str
    text: str
    poster: str = "unknown"
    created_at: str = datetime(2025, 1, 1).isoformat()


class ScoreResponse(BaseModel):
    score: float
    version: str


def download_model_weights():
    """
    Downloads the PEFT adapter from S3, loads the base model,
    merges the adapter, and saves the final merged model to LOCAL_MODEL_PATH.
    """
    s3_adapter_path = get_s3_model_path(ROFLBOT_RUN_NAME)
    local_adapter_path = "/adapter_cache/"
    os.makedirs(local_adapter_path)
    os.makedirs(LOCAL_MODEL_PATH)

    print(
        f"Downloading adapter '{ROFLBOT_RUN_NAME}' from {s3_adapter_path} to {local_adapter_path}..."
    )
    fs = s3fs.S3FileSystem()
    fs.get(s3_adapter_path + "/", local_adapter_path, recursive=True)

    print("Loading base model and merging adapter...")

    # 1. Read base model name from adapter config
    adapter_config_path = os.path.join(local_adapter_path, "adapter_config.json")

    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    base_model_name_or_path = adapter_config.get("base_model_name_or_path")

    print(f"Loading base model '{base_model_name_or_path}'...")
    # Use bf16 for potential speedup during merge/load, adjust if needed
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name_or_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    print("Base model loaded.")

    # 3. Load PEFT model (adapter + base)
    print(f"Loading adapter from {local_adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, local_adapter_path)
    print("Adapter loaded onto base model.")

    # 4. Merge adapter into the base model
    print("Merging adapter weights...")
    merged_model = peft_model.merge_and_unload()  # type: ignore
    print("Adapter merged.")

    # 5. Load tokenizer (prefer adapter's tokenizer if exists)
    print(f"Loading tokenizer from {local_adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(local_adapter_path)

    print(f"Saving merged model and tokenizer to {LOCAL_MODEL_PATH}...")
    merged_model.save_pretrained(LOCAL_MODEL_PATH)
    tokenizer.save_pretrained(LOCAL_MODEL_PATH)

    print(f"Model and tokenizer saved to {LOCAL_MODEL_PATH}.")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("uv")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(
        "uv pip install --system "
        "datasets==3.5.1 "
        "liger-kernel==0.5.8 "
        "panza==0.1.0 "
        "peft==0.15.2 "
        "polars==1.28.1 "
        "s3fs==2025.3.0 "
        "transformers==4.51.3 "
        "scikit-learn==1.6.1 "
        "hf-transfer==0.1.9 "
        "fastapi==0.115.12",
    )
    .run_function(
        download_model_weights,
        secrets=[modal.Secret.from_dotenv()],
    )
)

app = modal.App(f"roflbot-rm-{ROFLBOT_RUN_NAME}", image=image)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.function(
    gpu="A10G",
    scaledown_window=300,
    secrets=[modal.Secret.from_dotenv()],
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def serve_roflbot_rm():
    from liger_kernel.transformers import _apply_liger_kernel_to_instance

    web_app = FastAPI()

    logger.info("Initializing Roflbot RewardModel Service...")
    start_time = time.time()

    logger.info(f"Loading tokenizer from {LOCAL_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Ensure padding side matches training

    logger.info(f"Loading model from {LOCAL_MODEL_PATH}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH,
        num_labels=1,  # Regression task
        device_map="auto",  # Automatically handle device placement
        torch_dtype=torch.bfloat16,  # Use bfloat16 for performance
    )
    logger.info("Model loaded.")

    try:
        _apply_liger_kernel_to_instance(model)
        logger.info("Applied Liger kernel optimization.")
    except Exception as e:
        logger.warning(f"Could not apply Liger kernel: {e}")

    model.eval()
    logger.info("Model set to evaluation mode.")

    end_time = time.time()
    logger.info(
        f"Service initialization complete. Took {end_time - start_time:.2f} seconds."
    )

    @web_app.post("/score")
    async def get_score(request: RoflbotScoreRequest) -> ScoreResponse:
        """Scores a joke based on its title, text, poster, and creation time."""
        request_start_time = time.time()
        try:
            # Parse timestamp string into datetime object
            try:
                # Handle potential Z timezone suffix
                created_at_dt = datetime.fromisoformat(
                    request.created_at.replace("Z", "+00:00")
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid 'created_at' format. Please use ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SSZ).",
                )

            # Format the input text for the reward model
            formatted_text = format_joke_for_rm(
                title=request.title,
                text=request.text,
                poster=request.poster,
                created_at=created_at_dt,
            )

            # Tokenize the formatted text
            inputs = tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MODEL_MAX_LENGTH,
            )
            token_count = inputs["input_ids"].shape[-1]

            # Move tensors to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Extract the score (assuming model output logits are [batch_size, 1])
            score = outputs.logits[0][0].item()

            duration = time.time() - request_start_time
            logger.info(
                f"Request completed: input tokens: {token_count}, score: {score:.4f}, duration: {duration:.2f}s"
            )
            return ScoreResponse(score=score, version=SERVICE_VERSION)

        except HTTPException as http_exc:
            # Re-raise HTTP exceptions directly
            raise http_exc
        except Exception as e:
            duration = time.time() - request_start_time
            logger.error(f"Request failed: duration: {duration:.2f}s", exc_info=True)
            # Return a generic server error
            raise HTTPException(
                status_code=500, detail=f"Internal Server Error: {str(e)}"
            )

    return web_app
