class BenchmarkModelKey:
    name: str
    display_name: str
    split: str

    def __init__(
        self, name: str, display_name: str | None = None, split: str | None = None
    ):
        self.name = name
        self.display_name = display_name or name
        self.split = split or "val"

    def __str__(self):
        return self.display_name
