from pydantic import BaseModel

import art


class JustTheFactsConfig(BaseModel):
    num_epochs: int = 20
    eval_steps: int = 5
    trajectories_per_group: int = 4
    learning_rate: float = 1e-6


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


models["facts-7b-001"] = models["facts-14b-001"].model_copy(deep=True)
models["facts-7b-001"].name = "facts-7b-001"
models["facts-7b-001"].base_model = "Qwen/Qwen2.5-7B-Instruct"
