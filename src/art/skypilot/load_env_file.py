from pathlib import Path
from typing import Dict


def load_env_file(env_path: str) -> Dict[str, str]:
    """Load a simple dotenv style file (KEY=VALUE per line)."""
    envs: Dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        print(f"Warning: env file {env_path} does not exist – continuing without it.")
        return envs

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue  # skip malformed lines
            key, val = line.split("=", 1)
            envs[key.strip()] = val.strip()
    return envs
