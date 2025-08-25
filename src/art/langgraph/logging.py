import os
import pickle


class FileLogger:
    def __init__(self, filepath):
        self.text_path = filepath
        self.pickle_path = filepath + ".pkl"

    def log(self, name, entry):
        # Log as readable text
        with open(self.text_path, "a") as f:
            f.write(f"{name}: {entry}\n")

        # Append to pickle log
        with open(self.pickle_path, "ab") as pf:
            pickle.dump((name, entry), pf)

    def load_logs(self):
        """Load all logs from the pickle file."""
        if not os.path.exists(self.pickle_path):
            return []
        logs = []
        with open(self.pickle_path, "rb") as pf:
            try:
                while True:
                    logs.append(pickle.load(pf))
            except EOFError:
                pass
        return logs
