Unsloth does not yet support the vLLM V1 engine or multi-device training. A realistic solution is to decouple vLLM for inference and the Unsloth model for training so that we can update them independently.

## Implementation Status

### Completed:
1. Created `DecoupledUnslothService` in `/src/art/unsloth/decoupled_service.py` that follows the `TorchtuneService` pattern:
   - Uses vLLM V1 engine for inference
   - Runs Unsloth training in a separate process (`/src/art/unsloth/train_process.py`)
   - Implements sleep/wake mechanism for offloading vLLM during training
   - Saves LoRA checkpoints and reloads them after training

2. Updated `/src/art/local/backend.py` to support the new service:
   - Added config check for `use_decoupled_unsloth` flag
   - Service selection logic: TorchtuneService > DecoupledUnslothService > UnslothService

3. Created test scripts:
   - `/dev/yes-no-maybe.py` - Basic test script converted from notebook
   - `/dev/yes-no-maybe-decoupled.py` - Test script using DecoupledUnslothService

### Usage:
To use the DecoupledUnslothService, set `use_decoupled_unsloth: true` in the model's internal config:

```python
model = art.TrainableModel(
    name="my-model",
    project="my-project",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    _internal_config={
        "use_decoupled_unsloth": True
    }
)
```

### Next Steps:
- Run the yes-no-maybe.py test for 6 minutes to verify 2%+ reward improvement
- Test the DecoupledUnslothService with yes-no-maybe-decoupled.py
- Further optimize the training process communication and weight loading
