class LinearTemperatureAnnealer:
    """Linearly anneals between two temperature values."""

    def __init__(self, *, start: float, end: float, steps: int) -> None:
        self.start = start
        self.end = end
        self.steps = max(1, steps)
        self._step = 0

    def step(self) -> float:
        """Advance the schedule and return the new temperature."""
        fraction = min(self._step / (self.steps - 1), 1.0)
        temperature = self.start + (self.end - self.start) * fraction
        self._step += 1
        return temperature

