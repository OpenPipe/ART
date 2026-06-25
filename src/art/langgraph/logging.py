from typing import Any


class FileLogger:
    def __init__(self, filepath):
        self.text_path = filepath
        self._logs: list[tuple[str, Any]] = []

    def log(self, name, entry):
        # Log as readable text
        with open(self.text_path, "a") as f:
            f.write(f"{name}: {entry}\n")

        self._logs.append((name, entry))

    def load_logs(self):
        """Load all structured logs captured by this logger."""
        return list(self._logs)
