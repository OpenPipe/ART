from pydantic import BaseModel

import art


class JustTheFactsConfig(BaseModel):
    num_epochs: int = 20


models: dict[str, art.TrainableModel[JustTheFactsConfig]] = {
    "just-the-facts-14b-001": art.TrainableModel(
        name="just-the-facts-14b-001",
        project="just-the-facts",
        base_model="Qwen/Qwen2.5-14B-Instruct",
        config=JustTheFactsConfig(
            num_epochs=20,
        ),
    )
}


models["just-the-facts-7b-001"] = models["just-the-facts-14b-001"].model_copy(deep=True)
models["just-the-facts-7b-001"].name = "just-the-facts-7b-001"
models["just-the-facts-7b-001"].base_model = "Qwen/Qwen2.5-7B-Instruct"
