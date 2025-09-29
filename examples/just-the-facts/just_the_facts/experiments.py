from pydantic import BaseModel

import art


class JustTheFactsConfig(BaseModel):
    learning_rate: float = 1e-6
    num_epochs: int = 20

    eval_steps: int = 1
    groups_per_step: int = 3
    trajectories_per_group: int = 4
    scale_rewards: bool = False


models: dict[str, art.TrainableModel[JustTheFactsConfig]] = {
    "facts-14b-001": art.TrainableModel(
        name="facts-14b-001",
        project="just-the-facts",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=JustTheFactsConfig(
            num_epochs=20,
        ),
    )
}

models["facts-14b-002"] = models["facts-14b-001"].model_copy(deep=True)
models["facts-14b-002"].name = "facts-14b-002"
models["facts-14b-002"].base_model = "Qwen/Qwen2.5-14B-Instruct"

models["facts-14b-003"] = models["facts-14b-001"].model_copy(deep=True)
models["facts-14b-003"].name = "facts-14b-003"
models["facts-14b-003"].base_model = "Qwen/Qwen2.5-14B-Instruct"
models["facts-14b-003"].config.scale_rewards = True
models["facts-14b-003"].config.trajectories_per_group = 12


models["facts-7b-001"] = models["facts-14b-001"].model_copy(deep=True)
models["facts-7b-001"].name = "facts-7b-001"
models["facts-7b-001"].base_model = "Qwen/Qwen2.5-7B-Instruct"

models["facts-qwen3-4b-001"] = models["facts-14b-001"].model_copy(deep=True)
models["facts-qwen3-4b-001"].name = "facts-qwen3-4b-001"
models["facts-qwen3-4b-001"].base_model = "Qwen/Qwen3-4B"

models["facts-qwen3-4b-002"] = models["facts-14b-001"].model_copy(deep=True)
models["facts-qwen3-4b-002"].name = "facts-qwen3-4b-002"
models["facts-qwen3-4b-002"].base_model = "Qwen/Qwen3-4B"

models["facts-qwen3-4b-003"] = models["facts-14b-001"].model_copy(deep=True)
models["facts-qwen3-4b-003"].name = "facts-qwen3-4b-003"
models["facts-qwen3-4b-003"].base_model = "Qwen/Qwen3-4B"
models["facts-qwen3-4b-003"].config.groups_per_step = 8
