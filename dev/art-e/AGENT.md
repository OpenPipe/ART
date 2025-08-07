## Adding New Models to the Email Agent

When adding a new model to the email agent project, follow these steps:

1. **Add the model definition to `all_experiments.py`**:

   ```python
   # Model XXX: Description of what makes this model unique
   models["XXX"] = models["BASE_MODEL_ID"].model_copy(deep=True)
   models["XXX"].name = "email-agent-XXX"
   # Add any custom configuration here
   ```

2. **If you need a new configuration option**:

   a. Add it to `ProjectPolicyConfig` in `project_types.py`:

   ```python
   class ProjectPolicyConfig(BaseModel):
       # ... existing fields ...
       my_new_option: bool = True  # Add description and default value
   ```

   b. If it affects training, update `train.py` to pass it to the training function:

   ```python
   await model.train(
       groups,
       config=art.TrainConfig(learning_rate=model.config.learning_rate),
       _config=art.dev.TrainConfig(
           # ... existing parameters ...
           my_new_option=model.config.my_new_option,
       ),
   )
   ```

   c. If it affects rollouts, update `rollout.py` to use the new option.

3. **Common model variations**:
   - Base model change: `models["XXX"].base_model = "new-base-model"`
   - Learning rate: `models["XXX"].config.learning_rate = 1e-5`
   - Training epochs: `models["XXX"].config.num_epochs = 3`
   - Judge model: `models["XXX"].config.group_judge_model = "model-name"`
   - Fork from checkpoint:
     ```python
     models["XXX"].config.fork_from_model = "email-agent-YYY"
     models["XXX"].config.fork_not_after_step = 90
     ```
