# ART-Style Training for SWE-bench

This implementation provides an ART (Agent Reinforcement Trainer) style training script for SWE-bench, inspired by `qwen_rollout.py` but following idiomatic ART patterns.

## Files

- `art_style_rollout.py` - Core rollout function that executes agent interactions in ART style
- `train_art_style.py` - Main training script with both inference and training modes
- `test_art_style.py` - Test script to verify the implementation

## Key Features

### ART-Style Rollout (`art_style_rollout.py`)

The rollout function follows ART idioms:
- Returns `art.Trajectory` objects with messages, rewards, and metrics
- Uses retry decorators for robustness
- Tracks detailed metrics including progress, maintenance, and resolution
- Implements proper tool handling for bash commands and file editing
- Calculates rewards based on test pass/fail rates

### Training Script (`train_art_style.py`)

Supports two modes:

1. **Inference Mode** - For testing with existing models:
   ```bash
   python train_art_style.py --mode inference --num-instances 10
   ```

2. **Training Mode** - For training new models with gradients:
   ```bash
   python train_art_style.py --mode train --epochs 1 --batch-size 4
   ```

### Configuration

The `ARTModelConfig` class allows customization of:
- `max_steps`: Maximum interaction steps (default: 30)
- `temperature`: Model temperature (default: 0.0)
- `max_tokens`: Maximum tokens per response (default: 4096)
- `system_prompt`: System prompt for the model
- `instance_prompt_template`: Template for problem descriptions

### Command-Line Options

Key training script options:
- `--mode`: Choose between `inference` (no gradients) or `train` (with gradients)
- `--model`: Model name or path
- `--num-instances`: Number of instances to use (inference mode)
- `--batch-size`: Batch size for training
- `--rollouts-per-instance`: Number of rollouts per instance
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate for training
- `--reward-power`: Power to apply to progress metric
- `--no-quality-filter`: Disable quality filtering (not recommended)
- `--require-non-zero-tests`: Require instances to have tests (default: True)

## Usage Examples

### Quick Test
```bash
# Test a single instance
python test_art_style.py

# Test a trajectory group
python test_art_style.py group
```

### Inference with Local Model
```bash
python train_art_style.py \
    --mode inference \
    --model "willcb/Qwen3-32B" \
    --api-base "http://localhost:8000/v1" \
    --num-instances 5 \
    --rollouts-per-instance 2
```

### Disable Quality Filtering (Not Recommended)
```bash
# Use all instances without quality filtering
python train_art_style.py \
    --mode inference \
    --model "willcb/Qwen3-32B" \
    --num-instances 10 \
    --no-quality-filter
```

### Training a New Model
```bash
python train_art_style.py \
    --mode train \
    --model "Qwen/Qwen3-32B" \
    --batch-size 4 \
    --rollouts-per-instance 4 \
    --epochs 1 \
    --learning-rate 5e-5
```

## Reward Calculation

The reward function follows the same formula as the original implementation:
- 20% weight on test maintenance (keeping passing tests passing)
- 30% weight on progress (fixing failing tests)
- 50% weight on full resolution (all tests passing)

The `reward_power` parameter can be used to adjust the progress component.

## Quality Filtering

The implementation includes an instance quality filter that identifies instances with reliable test behavior. **Quality filtering is enabled by default.**

- **Filters instances where**:
  1. All FAIL_TO_PASS tests initially pass
  2. All FAIL_TO_PASS tests fail after applying the patch (bug introduction)
  3. All PASS_TO_PASS tests remain passing

- **Usage**: Quality filtering is automatic. To disable it (not recommended), use `--no-quality-filter`
- **Statistics**: Approximately 54% of instances (4,577 out of 8,480) meet quality criteria

## Differences from Original Implementation

1. **Simplified Architecture**: No dependency on SWE-Agent framework
2. **Direct Tool Implementation**: Tools are implemented directly without complex abstractions
3. **ART Integration**: Native support for ART training loops and trajectory management
4. **Cleaner Error Handling**: Uses ART retry decorators and proper exception handling
5. **Quality Filtering**: Built-in filtering based on test reliability

## Requirements

- Python 3.8+
- ART framework (`art`)
- OpenAI Python client
- PyTorch (for training mode)
- Access to SWE-bench instances and sandboxes

## Environment Variables

- `OPENAI_API_KEY`: API key for OpenAI (can be "default" for local inference)
- `OPENAI_BASE_URL`: Base URL for API (e.g., "http://localhost:8000/v1")
- Standard SWE-bench environment variables