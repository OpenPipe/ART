from pydantic import BaseModel

import art


class McpPolicyConfig(BaseModel):
    max_turns: int = 5
    max_tokens: int = 2048

    base_model: str = "Qwen/Qwen2.5-14B-Instruct"

    # MCP server configuration
    mcp_server_name: str = "mcp_alphavantage"  # Default to alphavantage server

    # Training configuration fields
    trajectories_per_group: int = 7
    groups_per_step: int = 4
    learning_rate: float = 1e-6
    eval_steps: int = 1
    val_set_size: int = 8
    training_dataset_size: int = 16
    num_epochs: int = 80
    # Model name to use for RULER rescoring (LLM-as-a-judge)
    ruler_judge_model: str = "openrouter/openai/o4-mini"
    minimum_reward_std_dev: float = 0.0
    # Random seed to control which subset of the training data is sampled
    training_dataset_seed: int | None = None

    # Fork configuration
    fork_from_model: str | None = None
    fork_from_project: str | None = None
    fork_not_after_step: int | None = None

    # Training configuration
    scale_rewards: bool = True


models: dict[str, art.TrainableModel[McpPolicyConfig]] = {
    "mcp-7b-001": art.TrainableModel(
        name="mcp-7b-001",
        project="mcp-agent-training",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        config=McpPolicyConfig(
            num_epochs=20,
        ),
    )
}


models["mcp-14b-001"] = models["mcp-7b-001"].model_copy(deep=True)
models["mcp-14b-001"].name = "mcp-14b-001"
models["mcp-14b-001"].base_model = "Qwen/Qwen2.5-14B-Instruct"
models["mcp-14b-001"].config.num_epochs = 160

# Model using alphavantage server with explicit specification
models["mcp-14b-alpha-001"] = models["mcp-7b-001"].model_copy(deep=True)
models["mcp-14b-alpha-001"].project = "mcp_alphavantage"
models["mcp-14b-alpha-001"].name = "mcp-14b-alpha-001"
models["mcp-14b-alpha-001"].config.mcp_server_name = "mcp_alphavantage"
models["mcp-14b-alpha-001"].config.num_epochs = 300


models["mcp-14b-alpha-002"] = models["mcp-14b-alpha-001"].model_copy(deep=True)
models["mcp-14b-alpha-002"].name = "mcp-14b-alpha-002"


models["mcp-14b-alpha-003"] = models["mcp-14b-alpha-001"].model_copy(deep=True)
models["mcp-14b-alpha-003"].name = "mcp-14b-alpha-003"


models["mcp-14b-alpha-004"] = models["mcp-14b-alpha-001"].model_copy(deep=True)
models["mcp-14b-alpha-004"].name = "mcp-14b-alpha-004"
models["mcp-14b-alpha-004"].config.learning_rate = 1e-6

# Model using balldontlie server
models["mcp-14b-ball-001"] = models["mcp-7b-001"].model_copy(deep=True)
models["mcp-14b-ball-001"].project = "mcp_balldontlie"
models["mcp-14b-ball-001"].name = "mcp-14b-ball-001"
models["mcp-14b-ball-001"].config.mcp_server_name = "mcp_balldontlie"
models["mcp-14b-ball-001"].config.num_epochs = 300
