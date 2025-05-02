# Art•E(mail)

## Project Overview

You can find a full write-up of this project in [this post on the OpenPipe blog](https://openpipe.ai/blog/art-e-mail-agent).

This project trains models using reinforcement learning (RL) to search through email datasets and answer user queries. The goal is to create agents that could potentially integrate with email providers (like a Gmail plugin), allowing users to ask questions like "what time does my wife's flight arrive on Friday" or "what are the next steps I committed to for project X" and receive accurate answers based on email content.

## Features

- Email database creation and management with SQLite
- Advanced search capabilities (keywords, date ranges, sender/recipient filters)
- Email question-answering with reinforcement learning
- Comprehensive model evaluation framework
- Performance benchmarking against multiple models

## Dataset

This project uses the Enron Email Dataset, downloaded from Kaggle. The dataset is processed and stored in a SQLite database with full-text search capabilities.

To generate training data, the system:

1. Processes email inboxes of Enron employees
2. Generates synthetic questions about email content
3. Creates training/validation splits for model training

## Requirements

- Python 3.10+
- Key dependencies:
  - art (for reinforcement learning)
  - pandas, polars (data processing)
  - mailparser, kaggle (dataset handling)
  - sqlite3 (database)
  - litellm (inference)
  - datasets, tqdm (data management)
- API keys:
  - Kaggle (for dataset download)
  - Model providers for evaluation (OpenAI, etc.)
- S3 bucket to store logs and model checkpoints (set via `BACKUP_BUCKET` env var)

## Installation

```bash
# Clone repository
git clone https://github.com/OpenPipe/ART
cd ART/examples/art-e

# Install package
uv sync

# Create .env file with required variables
# BACKUP_BUCKET=your-s3-bucket-name
# OPENPIPE_API_KEY=your-key  # Optional, for logging
```

## Usage

### Creating the Synthetic Dataset

The following commands were used to create the initial dataset. However, you can skip this if you just want to reproduce the results, since the processed dataset is freely hosted at https://huggingface.co/datasets/corbt/enron_emails_sample_questions.

```bash
# Download and process the Enron dataset (default: 100 emails)
python -m email_deep_research.data.convert_enron_email_dataset

# Process more emails
python -m email_deep_research.data.convert_enron_email_dataset --max-emails 10000

# Generate SQLite database
python -c "from email_deep_research.data.local_email_db import generate_database; generate_database(overwrite=True)"
```

### Training Models

I used `skypilot` with the [Runpod](https://www.runpod.io/) backend to train these models. Once you've authenticated Runpod for use with skypilot, the following command should start a training job that replicates our reported results:

```bash
uv run run_training_job.py 008 --fast
```

You can see the other model variants I tried training in `train.py`.

### Evaluating Models

```python
# Benchmark a model
from email_deep_research.evaluate.benchmark import benchmark_model
from email_deep_research.project_types import ProjectPolicyConfig
import asyncio
import art

# Create model
model = art.Model(
    name="gpt-4o",  # Can also use your trained models
    project="email_agent",
    config=ProjectPolicyConfig(
        litellm_model_name="openai/gpt-4o",
        use_tools=True,
    ),
)

# Run benchmark
results = asyncio.run(benchmark_model(model))
print(results)
```

## Project Structure

- **data/**: Dataset processing and management

  - `convert_enron_email_dataset.py`: Downloads/processes Enron dataset
  - `local_email_db.py`: SQLite database creation and management
  - `generate_synthetic_question_data.py`: Creates question-answer pairs
  - `query_iterators.py`: Loads datasets for training/evaluation
  - `types_enron.py`: Data models for emails and queries

- **email_search_tools.py**: Tools for searching and retrieving emails

- **evaluate/**: Model evaluation

  - `benchmark.py`: Performance benchmarking
  - `evaluate.py`: Analysis and visualization tools

- **train.py**: Main training script

- **rollout.py**: Defines model-environment interaction

- **project_types.py**: Configuration classes

## How It Works

1. **Data Preparation**: Process Enron emails into a searchable database
2. **Question Generation**: Create synthetic questions about emails
3. **Training**: Models learn via reinforcement learning to:
   - Search for relevant emails using keywords
   - Read email content to extract information
   - Formulate correct answers with proper citations
4. **Rewards**: System provides rewards based on answer correctness, sourcing, and efficiency
5. **Evaluation**: Compare models on metrics like accuracy, turn count, and source citation

## Performance Metrics

The evaluation framework tracks:

- Answer correctness (semantic match to ground truth)
- Source citation accuracy (identifying the correct email)
- Efficiency (number of turns to find answer)
- Tool use effectiveness
- Search strategy quality

Models are benchmarked against commercial LLMs like GPT-4.1 and Gemini 2.5 Pro to measure relative performance.
