

class BenchmarkedModelKey:
    model: str
    split: str
    iteration_indices: list[int] | None = None

    def __init__(self, model: str, split: str, iteration_indices: list[int] | None = None):
        self.model = model
        self.split = split
        self.iteration_indices = iteration_indices


    def __str__(self):
        iterations_str = ""
        if self.iteration_indices is not None:
            if len(self.iteration_indices) == 1:
                iterations_str = f"{self.iteration_indices[0]}"
            else:
                iterations_str = f"{self.iteration_indices[0]}-{self.iteration_indices[-1]}"
        return f"{self.model} {self.split} {iterations_str} "

class BenchmarkedModelIteration:
    index: int
    metrics: dict[str, float] = {}

    def __init__(self, index: int, metrics: dict[str, float] | None = None):
        self.index = index
        self.metrics = metrics if metrics is not None else {}

    def __str__(self):
        return f"{self.index} {self.metrics}"

class BenchmarkedModel:
    model_key: BenchmarkedModelKey
    iterations: list[BenchmarkedModelIteration] = []

    def __init__(self, model_key: BenchmarkedModelKey, iterations: list[BenchmarkedModelIteration] | None = None):
        self.model_key = model_key
        self.iterations = iterations if iterations is not None else []

    def __str__(self):
        iterations_str = '\n'.join([str(iteration) for iteration in self.iterations])
        return f"{self.model_key}\n{iterations_str}"