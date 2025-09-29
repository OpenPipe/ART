from datetime import datetime


class BenchmarkedModelKey:
    model: str
    split: str
    step_indices: list[int] | None = None

    def __init__(self, model: str, split: str, step_indices: list[int] | None = None):
        self.model = model
        self.split = split
        self.step_indices = step_indices

    def __str__(self):
        steps_str = ""
        if self.step_indices is not None:
            if len(self.step_indices) == 1:
                steps_str = f"{self.step_indices[0]}"
            else:
                steps_str = f"{self.step_indices[0]}-{self.step_indices[-1]}"
        return f"{self.model} {self.split} {steps_str}"


class BenchmarkedModelStep:
    index: int
    recorded_at: datetime | None = None
    metrics: dict[str, float] = {}

    def __init__(self, index: int, metrics: dict[str, float] | None = None):
        self.index = index
        self.metrics = metrics if metrics is not None else {}

    def __str__(self):
        return f"{self.index} {self.metrics}"


class BenchmarkedModel:
    model_key: BenchmarkedModelKey
    steps: list[BenchmarkedModelStep] = []

    def __init__(
        self,
        model_key: BenchmarkedModelKey,
        steps: list[BenchmarkedModelStep] | None = None,
    ):
        self.model_key = model_key
        self.steps = steps if steps is not None else []

    def __str__(self):
        steps_str = "\n".join([str(step) for step in self.steps])
        return f"{self.model_key}\n{steps_str}"
