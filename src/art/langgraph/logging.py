import json
import os


class FileLogger:
    def __init__(self, filepath):
        self.text_path = filepath
        self.jsonl_path = filepath + ".jsonl"

    def log(self, name, entry):
        # Log as readable text
        with open(self.text_path, "a") as f:
            f.write(f"{name}: {entry}\n")

        # Append to JSON Lines log
        with open(self.jsonl_path, "a") as jf:
            jf.write(json.dumps([name, entry]) + "\n")

    def load_logs(self):
        """Load all logs from the JSON Lines file."""
        if not os.path.exists(self.jsonl_path):
            return []
        logs = []
        with open(self.jsonl_path, "r") as jf:
            for line in jf:
                logs.append(tuple(json.loads(line)))
        return logs
