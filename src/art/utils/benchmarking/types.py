

class BenchmarkedModelKey:
    model: str
    split: str
    iteration: int | None = None

    def __init__(self, model: str, split: str, iteration: int | None = None):
        self.model = model
        self.split = split
        self.iteration = iteration


    def __str__(self):
        return f"{self.model} {self.split} {self.iteration}"

class BenchmarkedModel:
    model_key: BenchmarkedModelKey
    metrics: dict[str, float] = {}

    def __init__(self, model_key: BenchmarkedModelKey, metrics: dict[str, float] | None = None):
        self.model_key = model_key
        self.metrics = metrics if metrics is not None else {}

    def __str__(self):
        return f"{self.model_key} {self.metrics}"