

class BenchmarkedModelKey:
    model: str
    split: str
    iteration: int | None = None

    def __init__(self, model: str, split: str, iteration: int | None = None):
        self.model = model
        self.split = split
        self.iteration = iteration

class BenchmarkedModel:
    model_key: BenchmarkedModelKey
    metrics: dict[str, float] = {}

    def __init__(self, model_key: BenchmarkedModelKey):
        self.model_key = model_key
