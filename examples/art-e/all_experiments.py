import art
from art_e.project_types import ProjectPolicyConfig


models: dict[str, art.TrainableModel[ProjectPolicyConfig]] = {
    "002": art.TrainableModel(
        name="email-agent-002",
        project="email_agent",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=ProjectPolicyConfig(
            max_turns=10,
            log_to_openpipe=True,
            trajectories_per_group=6,
            groups_per_step=8,
            learning_rate=1.2e-5,
            eval_steps=30,
            val_set_size=100,
            training_dataset_size=4000,
            num_epochs=1,
        ),
    )
}


models["004"] = models["002"].model_copy(deep=True)
models["004"].name = "email-agent-004"
models["004"].config.max_turns = 30

models["005"] = models["002"].model_copy(deep=True)
models["005"].name = "email-agent-005"

models["006"] = models["005"].model_copy(deep=True)
models["006"].name = "email-agent-006"

models["007"] = models["005"].model_copy(deep=True)
models["007"].name = "email-agent-007"

models["008"] = models["005"].model_copy(deep=True)
models["008"].name = "email-agent-008"
models["008"].config.trajectories_per_group = 4
models["008"].config.groups_per_step = 12
models["008"].config.num_epochs = 3

models["011"] = models["008"].model_copy(deep=True)
models["011"].name = "email-agent-011"
models["011"].config.num_epochs = 4

models["012"] = models["008"].model_copy(deep=True)
models["012"].name = "email-agent-012"

models["013"] = models["002"].model_copy(deep=True)
models["013"].name = "email-agent-013"
models["013"].config.num_epochs = 4
models["013"].config.trajectories_per_group = 4
models["013"].config.groups_per_step = 24

models["014"] = models["008"].model_copy(deep=True)
models["014"].name = "email-agent-014"
models["014"].config.stupid_simple_reward_fn = True

models["201"] = models["008"].model_copy(deep=True)
models["201"].name = "email-agent-201"

# Model 202: like 008 but with judge-group rescoring during training
models["202"] = models["008"].model_copy(deep=True)
models["202"].name = "email-agent-202"
# Enable the new flag
models["202"].config.use_judge_group_variant = "v1"

# Model 204: like 202 but with judge-group rescoring variant v2
models["204"] = models["202"].model_copy(deep=True)
models["204"].name = "email-agent-204"
# Enable the v2 flag
models["204"].config.use_judge_group_variant = "v2"

# Model 205: like 204 but using Gemini 2.5 Flash as the judge-group model
models["205"] = models["204"].model_copy(deep=True)
models["205"].name = "email-agent-205"
# Set the judge group model
models["205"].config.group_judge_model = "gemini/gemini-2.5-flash"

# Model 206: like 204 but using Qwen3 32B as the judge-group model
models["206"] = models["204"].model_copy(deep=True)
models["206"].name = "email-agent-206"
# Set the judge group model
models["206"].config.group_judge_model = "openrouter/qwen/qwen3-32b"

# Model 207: like 205 but only uses 12 training examples total
models["207"] = models["205"].model_copy(deep=True)
models["207"].name = "email-agent-207"
models["207"].config.training_dataset_size = 12
models["207"].config.num_epochs = 500

models["208"] = models["206"].model_copy(deep=True)
models["208"].name = "email-agent-208"

# Model 209: like 205 but with min reward std dev filtering enabled
models["209"] = models["205"].model_copy(deep=True)
models["209"].name = "email-agent-209"
models["209"].config.minimum_reward_std_dev = 0.05

# Model 210-* sweep: based on 206 but varying training_dataset_size across powers of 4
# Generates models 210-1, 210-4, 210-16, 210-64, 210-256, 210-1024, 210-4096
for _size in [1, 4, 16, 64, 256, 1024, 4096]:
    key = f"210-{_size}"
    models[key] = models["206"].model_copy(deep=True)
    models[key].name = f"ea-{key}"
    # Set the dataset size
    models[key].config.training_dataset_size = _size
    # For very small datasets, match groups_per_step to the dataset size (<=16)
    if _size <= 16:
        models[key].config.groups_per_step = _size
    # Compute num_epochs so that total training steps ~= 600.
    # Approximation: total_steps ≈ num_epochs * (training_dataset_size / groups_per_step)
    # => num_epochs ≈ 600 * groups_per_step / training_dataset_size
    models[key].config.num_epochs = max(
        1, round(600 * models[key].config.groups_per_step / _size)
    )

# Model 210-16-s* variants: explore robustness to different data mixes by varying the random seed.
for _seed in [1, 2, 3]:
    key = f"210-16-s{_seed}"
    models[key] = models["210-16"].model_copy(deep=True)
    models[key].name = f"ea-{key}"
    # Set the seed
    models[key].config.training_dataset_seed = _seed

models["211"] = models["206"].model_copy(deep=True)
models["211"].name = "email-agent-211"

models["212"] = models["206"].model_copy(deep=True)
models["212"].name = "email-agent-212-30"

models["213"] = models["206"].model_copy(deep=True)
models["213"].name = "email-agent-213"
models["213"].config.group_judge_model = "openai/o3"

models["215"] = models["008"].model_copy(deep=True)
models["215"].name = "email-agent-215"

models["216"] = models["008"].model_copy(deep=True)
models["216"].name = "email-agent-216-3"

# Model 217: like 206 but with Qwen/Qwen3-14B base model and nothink option enabled
models["217"] = models["206"].model_copy(deep=True)
models["217"].name = "email-agent-217-3"
models["217"].base_model = "Qwen/Qwen3-14B"
models["217"].config.include_qwen3_nothink = True

models["218"] = models["206"].model_copy(deep=True)
models["218"].name = "email-agent-218-5"
models["218"].base_model = "Qwen/Qwen3-32B"
models["218"].config.group_judge_model = "base_model"
models["218"].config.include_qwen3_nothink = True

# Model 219: like 008 but with custom internal config (low max_grad_norm) and high learning rate
models["219"] = models["008"].model_copy(deep=True)
models["219"].name = "email-agent-219"
models["219"].config.learning_rate = 1e-2
models["219"]._internal_config = art.dev.InternalModelConfig(
    trainer_args=art.dev.TrainerArgs(
        max_grad_norm=1e-7,
    )
)

models["220"] = models["217"].model_copy(deep=True)
models["220"].name = "email-agent-220"
models["220"].base_model = "willcb/Qwen3-14B"

models["221"] = models["008"].model_copy(deep=True)
models["221"].name = "email-agent-221"
models["221"].config.include_qwen3_nothink = True
models["221"].base_model = "willcb/Qwen3-32B"
models["221"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(
        num_scheduler_steps=1,
    )
)

models["222"] = models["206"].model_copy(deep=True)
models["222"].name = "email-agent-222"
models["222"].base_model = "willcb/Qwen3-32B"
models["222"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(
        num_scheduler_steps=1,
    )
)
models["222"].config.group_judge_model = "base_model"
models["222"].config.include_qwen3_nothink = True

models["223"] = models["206"].model_copy(deep=True)
models["223"].name = "email-agent-223"
models["223"].base_model = "willcb/Qwen3-32B"
models["223"].config.include_qwen3_nothink = True
models["223"]._internal_config = art.dev.InternalModelConfig(
    engine_args=art.dev.EngineArgs(
        num_scheduler_steps=1,
    )
)

models["224"] = models["223"].model_copy(deep=True)
models["224"].name = "email-agent-224"
models["224"].config.learning_rate = 1e-6
models["224"].config.num_epochs = 6

models["225"] = models["224"].model_copy(deep=True)
models["225"].name = "email-agent-225"
