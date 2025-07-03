# Define model configurations for tau-bench RL experiments
models = {
    "001": {
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        "env": "retail",
        "model": "tau-bench-rl-001-final",
        "model_provider": "hosted_vllm",
        "user_model": "gpt-4o",
        "user_model_provider": "openai",
        "agent_strategy": "tool-calling-rl",
        "temperature": 1.0,
        "task_split": "test",
        "start_index": 0,
        "end_index": -1,
        "trajectories_per_group": 6,
        "groups_per_step": 10,
        "learning_rate": 1.2e-5,
        "eval_steps": 10,
        "val_set_size": 85,
        "training_dataset_size": 30,
        "num_epochs": 50,
        "reward_type": "real",
        "max_num_steps": 30,
        "train_mode": "sync_rl",
        "skip_eval": False,
        "add_shadow_trajectory": False,
        "messages_only": False,
    }
}

# # Retail environment variants
models["002"] = models["001"].copy()
models["002"]["model"] = "tau-bench-rl-002-final"
models["002"]["reward_type"] = "general_rm"

models["003"] = models["001"].copy()
models["003"]["model"] = "tau-bench-rl-003-final"
models["003"]["reward_type"] = "general_rm"
models["003"]["learning_rate"] = 5e-6

models["004"] = models["001"].copy()
models["004"]["model"] = "tau-bench-rl-004-final"
models["004"]["reward_type"] = "general_rm"
models["004"]["learning_rate"] = 5e-6
models["004"]["max_num_steps"] = 14

models["005"] = models["003"].copy()
models["005"]["model"] = "tau-bench-rl-005-final-4"
models["005"]["train_mode"] = "async_rl"
models["005"]["val_set_size"] = 30

models["006"] = models["001"].copy()
models["006"]["model"] = "tau-bench-rl-006-final-4"
models["006"]["train_mode"] = "async_rl"
models["006"]["learning_rate"] = 5e-6
models["006"]["val_set_size"] = 30

models["007"] = models["005"].copy()
models["007"]["model"] = "tau-bench-rl-007"
models["007"]["base_model"] = "Qwen/Qwen2.5-32B-Instruct"

models["008"] = models["001"].copy()
models["008"]["model"] = "tau-bench-rl-debug-1"
models["008"]["base_model"] = "Qwen/Qwen2.5-7B-Instruct"
models["008"]["learning_rate"] = 5e-6
models["008"]["trajectories_per_group"] = 3
models["008"]["groups_per_step"] = 3
models["008"]["val_set_size"] = 4
models["008"]["training_dataset_size"] = 6
models["008"]["train_mode"] = "async_rl"
models["008"]["reward_type"] = "general_rm"
models["008"]["skip_eval"] = True

models["009"] = models["003"].copy()
models["009"]["model"] = "tau-bench-rl-009-2"
models["009"]["train_mode"] = "async_rl"
models["009"]["val_set_size"] = 30
models["009"]["trajectories_per_group"] = 10

models["010"] = models["001"].copy()
models["010"]["model"] = "tau-bench-rl-010-2"
models["010"]["train_mode"] = "async_rl"
models["010"]["learning_rate"] = 5e-6
models["010"]["val_set_size"] = 30
models["010"]["trajectories_per_group"] = 10

models["011"] = models["005"].copy()
models["011"]["model"] = "tau-bench-rl-011"
models["011"]["base_model"] = "Qwen/Qwen2.5-32B-Instruct"
models["011"]["trajectories_per_group"] = 10

models["012"] = models["001"].copy()
models["012"]["model"] = "tau-bench-rl-012"
models["012"]["learning_rate"] = 5e-6
models["012"]["val_set_size"] = 30
models["012"]["trajectories_per_group"] = 10
models["012"]["reward_type"] = "general_rm"

# same as 012 but this one has slightly more graded rewards since max token trajs get -1 reward
models["013"] = models["001"].copy()
models["013"]["model"] = "tau-bench-rl-013"
models["013"]["learning_rate"] = 5e-6
models["013"]["val_set_size"] = 30
models["013"]["trajectories_per_group"] = 10
models["013"]["reward_type"] = "general_rm"

# same as 013 but this one is running on latest ART
models["014"] = models["001"].copy()
models["014"]["model"] = "tau-bench-rl-014"
models["014"]["learning_rate"] = 5e-6
models["014"]["val_set_size"] = 30
models["014"]["trajectories_per_group"] = 10
models["014"]["reward_type"] = "general_rm"

# same as 013 but this one is running on latest ART WITH tools info in the trajectory
models["015"] = models["001"].copy()
models["015"]["model"] = "tau-bench-rl-015"
models["015"]["learning_rate"] = 5e-6
models["015"]["val_set_size"] = 30
models["015"]["trajectories_per_group"] = 10
models["015"]["reward_type"] = "general_rm"

models["016"] = models["001"].copy()
models["016"]["model"] = "tau-bench-rl-016"
models["016"]["val_set_size"] = 30
models["016"]["groups_per_step"] = 8
models["016"]["reward_type"] = "general_rm"

# same as 016 but with the forced stop logging and reward implication
models["017"] = models["001"].copy()
models["017"]["model"] = "tau-bench-rl-017-2"
models["017"]["skip_eval"] = True
models["017"]["training_dataset_size"] = 20
models["017"]["reward_type"] = "general_rm"

# same as 015 but with the forced stop logging and reward implication, and also real rewards
models["018"] = models["001"].copy()
models["018"]["model"] = "tau-bench-rl-018-2"
models["018"]["learning_rate"] = 5e-6
models["018"]["skip_eval"] = True
models["018"]["training_dataset_size"] = 20
models["018"]["trajectories_per_group"] = 10
models["018"]["reward_type"] = "real"

# tried training with messages only
models["019"] = models["001"].copy()
models["019"]["model"] = "tau-bench-rl-019-2"
models["019"]["skip_eval"] = True
models["019"]["training_dataset_size"] = 10
models["019"]["trajectories_per_group"] = 8
models["019"]["groups_per_step"] = 5
models["019"]["num_epochs"] = 150
models["019"]["reward_type"] = "general_rm"
models["019"]["learning_rate"] = 8e-6
models["019"]["messages_only"] = True

# same as 019 but with shadow trajectories
models["020"] = models["019"].copy()
models["020"]["model"] = "tau-bench-rl-020-4"
models["020"]["add_shadow_trajectory"] = True

# same as 017 but with 32B model
models["021"] = models["017"].copy()
models["021"]["model"] = "tau-bench-rl-021"
models["021"]["base_model"] = "Qwen/Qwen2.5-32B-Instruct"

# same as 017 but with Qwen3 14B model instead
models["022"] = models["017"].copy()
models["022"]["model"] = "tau-bench-rl-022-5"
models["022"]["base_model"] = "unsloth/Qwen3-14B"
models["022"]["reward_type"] = "real"

# same as 017 but with OpenPipe Qwen3 14B model instead
models["023"] = models["017"].copy()
models["023"]["model"] = "tau-bench-rl-023-2"
models["023"]["base_model"] = "OpenPipe/Qwen3-14B-custom-template"
models["023"]["reward_type"] = "real"

# same as 017 but with real rewards
models["024"] = models["017"].copy()
models["024"]["model"] = "tau-bench-rl-024-2"
models["024"]["reward_type"] = "real"
# --------------------------------------------------------------------
